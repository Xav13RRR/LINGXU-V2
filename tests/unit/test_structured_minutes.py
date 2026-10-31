import json
from typing import Any

import pytest

from lingxu.domain.models import TranscriptEntryRecord
from lingxu.resources import deepseek_full_transcript_minutes_template
from lingxu.worker.full_transcript_preview import (
    FullTranscriptPreviewError,
    build_preview_messages,
    build_preview_request,
    normalize_generated_evidence,
    prepare_authoritative_meeting_facts,
    prepare_full_transcript_records,
    reconcile_generated_transcript,
    render_public_minutes,
    request_sha256,
    validate_generated_minutes,
    validate_public_minutes,
)
from lingxu.worker.minutes_diagnostics import MinutesValidationDetails
from lingxu.worker.structured_minutes import (
    build_structured_messages,
    compose_structured_minutes,
    validate_summary_bindings,
)


def _prepared(count: int = 3) -> Any:
    return prepare_full_transcript_records(
        [
            TranscriptEntryRecord(
                id=f"internal-db-row-{index}",
                session_id="internal-session",
                source_key="BOT_TTS:0" if index == 0 else f"internal-speaker-key-{index}",
                user_id=None if index == 0 else 10000 + index % 2,
                raw_text="系统上下文" if index == 0 else f"发言{index}: 我建议先测试, 暂未确认。",
                normalised_text=None,
                started_ms=index * 2500,
                ended_ms=index * 2500 + 1000,
            )
            for index in range(count)
        ],
        speaker_names={10000: "乙", 10001: "甲"},
    )


def _payload() -> dict[str, Any]:
    return {
        "version": 1,
        "summary": [{"text": "参会者提出先测试的建议。", "evidence_ids": [1, 2]}],
        "topics": [
            {"title": "测试", "discussion": [{"text": "建议先测试。", "evidence_ids": [2, 1, 1]}]}
        ],
        "decisions": [],
        "agreements": [],
        "actions": [],
        "questions": [],
        "proposals": [{"text": "先测试。", "evidence_ids": [1]}],
        "risks": [],
        "disagreements": [],
        "transcript_edits": [],
    }


@pytest.mark.parametrize("count", [395, 1179])
def test_compact_input_preserves_every_source_and_excludes_internal_identifiers(count: int) -> None:
    prepared = _prepared(count)
    facts = prepare_authoritative_meeting_facts(participant_labels=("[甲]", "[乙]"))
    messages = build_structured_messages(prepared=prepared, authoritative_facts=facts)
    packet = json.loads(messages[1]["content"])
    rows = packet["system_context"] + [
        row for window in packet["speech_windows"] for row in window["rows"]
    ]
    assert sorted(row["id"] for row in rows) == list(range(count))
    assert {row["id"]: row["text"] for row in rows} == {
        row.ordinal: row.text for row in prepared.source_rows
    }
    assert all(row["speaker"] == "s0" for row in packet["system_context"])
    assert len(packet["speech_windows"]) > 1
    for forbidden in (
        "internal-db-row",
        "internal-speaker-key",
        "internal-session",
        '"user_id"',
        '"markdown_row"',
    ):
        assert forbidden not in messages[1]["content"]
    legacy = build_preview_messages(
        prepared=prepared,
        authoritative_facts=facts,
        template=deepseek_full_transcript_minutes_template().read_text(encoding="utf-8"),
    )
    compact_bytes = len(json.dumps(messages, ensure_ascii=False).encode())
    legacy_bytes = len(json.dumps(legacy, ensure_ascii=False).encode())
    assert compact_bytes < legacy_bytes * 0.7


def test_structured_composition_keeps_source_coverage_and_local_evidence_formatting() -> None:
    prepared = _prepared()
    facts = prepare_authoritative_meeting_facts(participant_labels=("[甲]", "[乙]"))
    draft = compose_structured_minutes(
        json.dumps(_payload()), prepared=prepared, authoritative_facts=facts
    )
    content, coverage = reconcile_generated_transcript(draft.content, prepared=prepared)
    content, _ = normalize_generated_evidence(
        content,
        expected_speaker_labels=prepared.expected_speaker_labels,
        expected_source_speaker_labels=prepared.source_speaker_labels,
        authoritative_facts=facts,
    )
    validate_generated_minutes(
        content,
        expected_speaker_labels=prepared.expected_speaker_labels,
        expected_source_speaker_labels=prepared.source_speaker_labels,
        authoritative_facts=facts,
    )
    public, audit = render_public_minutes(content)
    validate_public_minutes(
        public,
        evidence_audit=audit,
        expected_speaker_labels=prepared.expected_speaker_labels,
        expected_source_speaker_labels=prepared.source_speaker_labels,
        authoritative_facts=facts,
    )
    assert "source_ordinals" not in public and "evidence_ordinals" not in public
    assert coverage.fallback_source_count == 3
    assert "建议先测试。" in public
    assert validate_summary_bindings(public, draft.summary_bindings, prepared)
    assert not validate_summary_bindings(
        public.replace("参会者提出", "伪造"), draft.summary_bindings, prepared
    )


@pytest.mark.parametrize(
    "ids,reason",
    [
        ([], "missing_reference"),
        ([999], "unknown_reference"),
        ([0], "system_only_reference"),
        ([True], "schema_invalid"),
    ],
)
def test_precise_reference_failures_never_disclose_model_text(ids: list[Any], reason: str) -> None:
    payload = _payload()
    payload["summary"][0] = {"text": "PRIVATE_MODEL_TEXT_DO_NOT_LOG", "evidence_ids": ids}
    with pytest.raises(FullTranscriptPreviewError) as raised:
        compose_structured_minutes(
            json.dumps(payload),
            prepared=_prepared(),
            authoritative_facts=prepare_authoritative_meeting_facts(),
        )
    error = raised.value
    assert error.validation_details is not None
    assert error.validation_details.reason == reason
    assert error.validation_details.section == "summary"
    assert "PRIVATE_MODEL" not in str(error) + json.dumps(error.validation_details.as_dict())


def test_consensus_requires_two_humans_even_when_only_one_has_source_rows() -> None:
    payload = _payload()
    payload.update(summary=[], topics=[], proposals=[])
    payload["agreements"] = [{"text": "同意。", "evidence_ids": [1]}]
    with pytest.raises(FullTranscriptPreviewError) as raised:
        compose_structured_minutes(
            json.dumps(payload),
            prepared=_prepared(2),
            authoritative_facts=prepare_authoritative_meeting_facts(
                participant_labels=("[甲]", "[乙]")
            ),
        )
    assert raised.value.validation_details is not None
    assert raised.value.validation_details.reason == "insufficient_consensus_speakers"


@pytest.mark.parametrize(
    "raw",
    [
        '{"version":1,"version":1}',
        "not json",
        '{"version":true}',
        '{"version":1,"PRIVATE_FIELD_SECRET":"PRIVATE_VALUE_SECRET"}',
    ],
)
def test_invalid_json_and_schema_errors_are_content_free(raw: str) -> None:
    with pytest.raises(FullTranscriptPreviewError) as raised:
        compose_structured_minutes(
            raw, prepared=_prepared(), authoritative_facts=prepare_authoritative_meeting_facts()
        )
    assert raised.value.validation_details is not None
    assert "PRIVATE" not in str(raised.value) + json.dumps(
        raised.value.validation_details.as_dict()
    )


@pytest.mark.parametrize(
    "source_ids,reason",
    [
        ([1, 2], "mixed_speaker_edit"),
        ([1, 999], "unknown_reference"),
        ([1, 1], "duplicate_coverage"),
        ([2, 1], "nonconsecutive_edit"),
    ],
)
def test_invalid_transcript_edits_do_not_erase_original_rows(
    source_ids: list[int], reason: str
) -> None:
    payload = _payload()
    payload["transcript_edits"] = [{"source_ids": source_ids, "text": "拟修订文本"}]
    with pytest.raises(FullTranscriptPreviewError) as raised:
        compose_structured_minutes(
            json.dumps(payload),
            prepared=_prepared(),
            authoritative_facts=prepare_authoritative_meeting_facts(),
        )
    assert raised.value.validation_details is not None
    assert raised.value.validation_details.reason == reason


def test_request_format_is_explicit_and_bound_to_request_hash() -> None:
    messages = build_structured_messages(
        prepared=_prepared(), authoritative_facts=prepare_authoritative_meeting_facts()
    )
    request = build_preview_request(messages, output_format="json_object")
    assert request["response_format"] == {"type": "json_object"}
    assert request_sha256(messages, output_format="json_object") != request_sha256(messages)


def test_validation_detail_taxonomy_rejects_arbitrary_values() -> None:
    for kwargs in (
        {"reason": "PRIVATE"},
        {"reason": "schema_invalid", "section": "PRIVATE"},
        {"reason": "schema_invalid", "item_index": True},
    ):
        with pytest.raises(ValueError):
            MinutesValidationDetails(**kwargs)


@pytest.mark.parametrize("content", ["", "## 执行摘要\n", "\n## 议程清单", "damaged"])
def test_damaged_summary_headings_fail_closed(content: str) -> None:
    assert not validate_summary_bindings(content, [], _prepared())


def test_valid_edits_and_actions_are_composed_without_markdown_injection() -> None:
    payload = _payload()
    payload["actions"] = [
        {"text": "测试 | 安全", "evidence_ids": [1], "owner": "s1", "due_date": None}
    ]
    payload["transcript_edits"] = [{"source_ids": [1], "text": "修订后的测试发言。"}]
    payload["summary"][0]["text"] = "<script>text</script> [link](target)\n## injected"
    prepared = _prepared()
    facts = prepare_authoritative_meeting_facts(participant_labels=("[甲]", "[乙]"))
    draft = compose_structured_minutes(
        json.dumps(payload), prepared=prepared, authoritative_facts=facts
    )
    content, coverage = reconcile_generated_transcript(draft.content, prepared=prepared)
    content, _ = normalize_generated_evidence(
        content,
        expected_speaker_labels=prepared.expected_speaker_labels,
        expected_source_speaker_labels=prepared.source_speaker_labels,
        authoritative_facts=facts,
    )
    validate_generated_minutes(
        content,
        expected_speaker_labels=prepared.expected_speaker_labels,
        expected_source_speaker_labels=prepared.source_speaker_labels,
        authoritative_facts=facts,
    )
    assert coverage.fallback_source_count == 2
    assert "修订后的测试发言。" in content
    assert "测试 &#124; 安全" in content
    assert "<script>" not in content and "[link]" not in content
    assert "\n## injected" not in content
