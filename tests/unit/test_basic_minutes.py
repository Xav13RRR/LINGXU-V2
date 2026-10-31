import json
from pathlib import Path

import pytest

from lingxu.worker.basic_minutes import BASIC_MINUTES_NOTICE, render_basic_minutes
from lingxu.worker.full_transcript_preview import (
    prepare_authoritative_meeting_facts,
    prepare_full_transcript,
    prepare_full_transcript_records,
    validate_public_minutes,
)


@pytest.mark.parametrize("human_rows", [0, 1, 2, 80])
def test_basic_minutes_preserve_complete_coverage_and_bound_unconfirmed_excerpts(
    tmp_path: Path,
    human_rows: int,
) -> None:
    values = [
        {
            "id": "system",
            "user_id": None,
            "source_key": "BOT_TTS:0",
            "started_ms": 0,
            "ended_ms": 1,
            "raw_text": "开始会议",
        },
        *[
            {
                "id": f"row-{index}",
                "user_id": 10 + index % 2,
                "source_key": f"asr:{index}",
                "started_ms": index * 1000 + 10,
                "ended_ms": index * 1000 + 900,
                "raw_text": f"发言{index}: 只是建议, 尚未确认。" + "条件仍需讨论。" * 60,
            }
            for index in range(human_rows)
        ],
    ]
    source = tmp_path / "source.json"
    source.write_text(json.dumps(values), encoding="utf-8")
    prepared = prepare_full_transcript(source, speaker_names={10: "甲", 11: "乙"})
    facts = prepare_authoritative_meeting_facts(participant_labels=("[甲]", "[乙]"))
    content, manifest = render_basic_minutes(prepared, facts)
    assert BASIC_MINUTES_NOTICE in content
    assert "source_ordinals" not in content and "evidence_ordinals" not in content
    assert content.count("原始发言摘录: ") == min(human_rows, 24)
    audit = manifest["evidence_audit"]
    assert isinstance(audit, dict)
    assert audit["transcript_binding_count"] == human_rows + 1
    assert audit["narrative_binding_count"] == min(human_rows, 24)
    for value in values:
        assert value["raw_text"] in content
    if human_rows == 0:
        assert "没有可用于纪要的已授权真人发言记录" in content
    assert render_basic_minutes(prepared, facts) == (content, manifest)
    validate_public_minutes(
        content,
        evidence_audit=audit,
        expected_speaker_labels=prepared.expected_speaker_labels,
        expected_source_speaker_labels=prepared.source_speaker_labels,
        authoritative_facts=facts,
    )


def test_basic_minutes_allow_an_empty_consented_transcript_without_inventing_speech() -> None:
    prepared = prepare_full_transcript_records([], speaker_names={}, allow_empty=True)
    facts = prepare_authoritative_meeting_facts()
    content, manifest = render_basic_minutes(prepared, facts)
    assert "没有可用于纪要的已授权真人发言记录" in content
    assert "无已完成投票" in content
    assert manifest["status"] == "fallback"
