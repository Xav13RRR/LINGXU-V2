"""Topic artifact and stage durability integration contracts."""

import hashlib
import json
from pathlib import Path

import pytest

from lingxu.domain.enums import JobStatus
from lingxu.worker.service import WorkerService
from lingxu.worker.topics import (
    APPROVED_TOPIC_MODEL,
    APPROVED_TOPIC_MODEL_DIGEST,
    APPROVED_TOPIC_PROVIDER,
    TopicBoundaryProviderPolicy,
)
from tests.integration.worker_fixtures import (
    CrashAfterTopicWriteRepository,
    FakePublisher,
    FakeSummaryProvider,
    _pipeline_fixture,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_segments_topics_only_after_bounded_model_confirmation(
    tmp_path: Path,
) -> None:
    provider = FakeSummaryProvider()
    database, jobs, pipeline, _, transcripts, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="确定性证据验证"),
        provider,
    )
    boundary = await transcripts.upsert(
        session_id=session_id,
        source_key="legacy:topic-transition",
        user_id=100,
        raw_text="接下来讨论部署风险。",
        normalised_text="接下来讨论部署风险。",
        started_ms=3_000,
        ended_ms=3_500,
    )
    await transcripts.upsert(
        session_id=session_id,
        source_key="legacy:topic-backchannel",
        user_id=100,
        raw_text="嗯",
        normalised_text="嗯",
        started_ms=3_600,
        ended_ms=3_800,
    )

    assert await WorkerService(
        jobs=jobs,
        worker_id="topic-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    topic_payload = json.loads(
        (tmp_path / "sessions" / session_id / "meeting" / "topic_segments.json").read_text(
            encoding="utf-8"
        )
    )
    assert topic_payload["schema_version"] == 2
    assert topic_payload["candidates"][0]["entry_id"] == boundary.id
    assert len(topic_payload["segments"]) == 2
    assert topic_payload["segments"][1]["entry_ids"] == [boundary.id]
    assert "嗯" not in provider.topic_context_text
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_topic_policy_failure_has_no_call_and_deterministic_detail(
    tmp_path: Path,
) -> None:
    provider = FakeSummaryProvider(
        TopicBoundaryProviderPolicy(
            provider=APPROVED_TOPIC_PROVIDER,
            base_url="https://example.com/v1",
            model=APPROVED_TOPIC_MODEL,
            expected_model_digest=APPROVED_TOPIC_MODEL_DIGEST,
            allow_remote=True,
        )
    )
    database, jobs, pipeline, _, transcripts, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="安全回退"),
        provider,
    )
    boundary = await transcripts.upsert(
        session_id=session_id,
        source_key="legacy:remote-policy-transition",
        user_id=100,
        raw_text="接下来讨论部署风险。",
        normalised_text="接下来讨论部署风险。",
        started_ms=3_000,
        ended_ms=3_500,
    )

    assert await WorkerService(jobs=jobs, worker_id="policy-worker", pipeline=pipeline).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    payload = json.loads(
        (tmp_path / "sessions" / session_id / "meeting" / "topic_segments.json").read_text(
            encoding="utf-8"
        )
    )
    assert provider.topic_calls == 0
    assert provider.outcome_calls == 0
    assert payload["confirmations"] == []
    assert payload["failures"] == [{"entry_id": boundary.id, "code": "endpoint_not_approved"}]
    assert len(payload["segments"]) == 1
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_topic_artifact_reused_after_post_write_crash_without_second_model_call(
    tmp_path: Path,
) -> None:
    provider = FakeSummaryProvider()
    database, _, pipeline, artifacts, transcripts, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="确定性证据验证"),
        provider,
    )
    await transcripts.upsert(
        session_id=session_id,
        source_key="legacy:crash-topic-transition",
        user_id=100,
        raw_text="接下来讨论部署风险。",
        normalised_text="接下来讨论部署风险。",
        started_ms=3_000,
        ended_ms=3_500,
    )
    crash_jobs = CrashAfterTopicWriteRepository(database.sessions)

    assert await WorkerService(
        jobs=crash_jobs,
        worker_id="topic-crash-worker",
        pipeline=pipeline,
    ).run_once()
    failed = await crash_jobs.get(job_id)
    assert failed is not None and failed.status is JobStatus.PENDING
    topic_path = tmp_path / "sessions" / session_id / "meeting" / "topic_segments.json"
    first_bytes = topic_path.read_bytes()
    first_payload = json.loads(first_bytes)
    assert provider.topic_calls == 1
    assert len(first_payload["provenance"]["canonical_input_hash"]) == 64
    assert len(first_payload["provenance"]["candidate_set_hash"]) == 64
    assert len(first_payload["provenance"]["confirmation_hash"]) == 64
    assert len(first_payload["provenance"]["topic_artifact_hash"]) == 64
    assert first_payload["provenance"]["approved_model_digest"] == APPROVED_TOPIC_MODEL_DIGEST

    provider.topic_unavailable = True
    assert await WorkerService(
        jobs=crash_jobs,
        worker_id="topic-retry-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await crash_jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert provider.topic_calls == 1
    assert topic_path.read_bytes() == first_bytes
    record = next(
        item
        for item in await artifacts.list(session_id)
        if item.artifact_type == "TOPIC_SEGMENTS_JSON"
    )
    assert record.sha256 == hashlib.sha256(first_bytes).hexdigest()
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_topic_artifact_input_mismatch_fails_closed_without_budget_reset(
    tmp_path: Path,
) -> None:
    provider = FakeSummaryProvider()
    database, _, pipeline, _, transcripts, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="确定性证据验证"),
        provider,
    )
    await transcripts.upsert(
        session_id=session_id,
        source_key="legacy:mismatch-topic-transition",
        user_id=100,
        raw_text="接下来讨论部署风险。",
        normalised_text="接下来讨论部署风险。",
        started_ms=3_000,
        ended_ms=3_500,
    )
    crash_jobs = CrashAfterTopicWriteRepository(database.sessions)

    assert await WorkerService(
        jobs=crash_jobs,
        worker_id="topic-mismatch-crash-worker",
        pipeline=pipeline,
    ).run_once()
    topic_path = tmp_path / "sessions" / session_id / "meeting" / "topic_segments.json"
    first_bytes = topic_path.read_bytes()
    canonical_path = tmp_path / "sessions" / session_id / "meeting" / "canonical_transcript.json"
    canonical_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical_payload["entries"][0]["text"] += "修订"
    canonical_path.write_text(
        json.dumps(canonical_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert await WorkerService(
        jobs=crash_jobs,
        worker_id="topic-mismatch-retry-worker",
        pipeline=pipeline,
    ).run_once()
    failed = await crash_jobs.get(job_id)
    assert failed is not None and failed.status is JobStatus.FAILED
    assert failed.attempt_count == 2
    assert failed.last_error_type == "OutcomePartitionError[job_revision_mismatch]@SEGMENT_TOPICS"
    assert provider.topic_calls == 1
    assert topic_path.read_bytes() == first_bytes
    await database.close()
