import json
from pathlib import Path
from typing import Any

import pytest

from lingxu.domain.enums import JobStatus
from lingxu.resources import deepseek_full_transcript_minutes_template
from lingxu.worker.full_transcript_preview import FullTranscriptPreviewResult
from lingxu.worker.service import WorkerService
from lingxu.worker.structured_minutes import STRUCTURED_MINUTES_REVISION
from tests.integration.worker_fixtures import FakePublisher, _pipeline_fixture


class StructuredProvider:
    output_format = "json_object"
    max_output_tokens = 131_072

    def __init__(self, *, bad_evidence: bool = False) -> None:
        self.calls = 0
        self.bad_evidence = bad_evidence

    async def generate(self, messages: list[dict[str, str]]) -> FullTranscriptPreviewResult:
        self.calls += 1
        packet = json.loads(messages[1]["content"])
        assert "full_transcript_json" not in messages[1]["content"]
        assert packet["output_schema"]["additionalProperties"] is False
        human = packet["speech_windows"][0]["rows"][0]
        evidence = [999] if self.bad_evidence else [human["id"]]
        claim = {"text": "提出测试建议, 尚未形成明确决策。", "evidence_ids": evidence}
        content = json.dumps(
            {
                "version": 1,
                "summary": [claim],
                "topics": [{"title": "测试", "discussion": [claim]}],
                "decisions": [],
                "agreements": [],
                "actions": [],
                "questions": [],
                "proposals": [claim],
                "risks": [],
                "disagreements": [],
                "transcript_edits": [],
            },
            ensure_ascii=False,
        )
        return FullTranscriptPreviewResult(content, "stop", "complete", len(content.encode()), {})


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("bad_evidence", [False, True])
async def test_structured_worker_delivery_cache_and_safe_failure_detail(
    tmp_path: Path, bad_evidence: bool
) -> None:
    provider = StructuredProvider(bad_evidence=bad_evidence)
    publisher = FakePublisher(
        expected_content="已生成基础纪要" if bad_evidence else "已验证议题结果已生成"
    )
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
        full_transcript_minutes_provider=provider,
        full_transcript_minutes_template=deepseek_full_transcript_minutes_template(),
    )
    service = WorkerService(jobs=jobs, worker_id="structured-worker", pipeline=pipeline)
    assert await service.run_once()
    job = await jobs.get(job_id)
    assert job is not None and job.status is JobStatus.COMPLETED
    assert provider.calls == 1 and publisher.calls == 1
    directory = tmp_path / "sessions" / session_id / "meeting"
    manifest = json.loads((directory / "full_transcript_minutes.json").read_text(encoding="utf-8"))
    attempt = json.loads(
        (directory / "full_transcript_minutes_attempt.json").read_text(encoding="utf-8")
    )
    assert manifest["output_format"] == "json_object"
    assert manifest["structured_minutes_revision"] == STRUCTURED_MINUTES_REVISION
    public = (directory / "minutes.md").read_text(encoding="utf-8")
    assert "source_ordinals" not in public and "evidence_ordinals" not in public
    assert "## 投票结果" in public
    assert "**[主持人甲]**:" in public
    assert await pipeline._minutes_are_current(session_id)
    reused = await pipeline._stage_generate_minutes(session_id)
    assert reused["artifact_reused"]
    assert (await pipeline._stage_publish_discord(session_id))["deduplicated"]
    assert not await service.run_once()
    assert provider.calls == 1 and publisher.calls == 1
    if bad_evidence:
        assert attempt["status"] == "failed"
        assert attempt["provider_response_content_bytes"] > 0
        assert len(attempt["provider_response_sha256"]) == 64
        assert "提出测试建议" not in json.dumps(attempt, ensure_ascii=False)
        assert attempt["validation_details"] == {
            "reason": "unknown_reference",
            "section": "summary",
            "item_index": 0,
            "reference_count": 1,
        }
        assert manifest["validation_details"] == attempt["validation_details"]
        assert manifest["generation_method"] == "deterministic_basic"
    else:
        assert attempt["status"] == "complete"
        assert manifest["generation_method"] == "deepseek_structured"
        assert "提出测试建议" in public
        manifest["summary_bindings"][0]["evidence_ids"] = [999]
        (directory / "full_transcript_minutes.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        assert not await pipeline._full_transcript_minutes_are_current(session_id)
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_changed_request_cannot_repeat_a_started_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = StructuredProvider()
    publisher = FakePublisher(expected_content="已生成基础纪要")
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
        full_transcript_minutes_provider=provider,
        full_transcript_minutes_template=deepseek_full_transcript_minutes_template(),
    )
    original = pipeline._generate_model_full_transcript_minutes
    attempt_path = (
        tmp_path / "sessions" / session_id / "meeting" / "full_transcript_minutes_attempt.json"
    )

    async def simulate_existing_attempt(session_id: str, document: Any) -> Any:
        attempt_path.write_text(
            json.dumps(
                {
                    "status": "started",
                    "request_sha256": "0" * 64,
                    "attempt_generation": 0,
                }
            ),
            encoding="utf-8",
        )
        return await original(session_id, document)

    monkeypatch.setattr(
        pipeline, "_generate_model_full_transcript_minutes", simulate_existing_attempt
    )
    assert await WorkerService(jobs=jobs, worker_id="recovery-worker", pipeline=pipeline).run_once()
    job = await jobs.get(job_id)
    assert job is not None and job.status is JobStatus.COMPLETED
    assert provider.calls == 0 and publisher.calls == 1
    assert json.loads(attempt_path.read_text(encoding="utf-8"))["status"] == "started"
    await database.close()
