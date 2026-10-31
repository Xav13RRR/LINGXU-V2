"""Unit tests for deterministic, audit-preserving transcript canonicalisation."""

from typing import Any

import pytest

from lingxu.domain.models import TranscriptEntryRecord
from lingxu.transcript.canonical import (
    _material_signature,
    _materially_compatible,
    _source_entry,
    build_canonical_transcript,
)


def entry(
    entry_id: str,
    *,
    user: int | None = 1,
    start: int = 1_000,
    end: int = 2_000,
    text: str = "发布方案",
    raw_text: str | None = None,
    normalised_text: str | None = None,
    source_key: str | None = None,
    revisions: tuple[dict[str, Any], ...] = (),
) -> TranscriptEntryRecord:
    """Build a privacy-eligible synthetic source entry without Meeting content."""

    return TranscriptEntryRecord(
        id=entry_id,
        session_id="session-1",
        source_key=source_key or f"utterance:{entry_id}",
        user_id=user,
        raw_text=raw_text if raw_text is not None else text,
        normalised_text=normalised_text if normalised_text is not None else text,
        started_ms=start,
        ended_ms=end,
        revision_json=revisions,
    )


def test_exact_same_speaker_duplicate_keeps_one_entry_and_all_source_ids() -> None:
    result = build_canonical_transcript(
        (
            entry("offline", text="发布方案已经确认"),
            entry("realtime", start=1_010, end=2_010, text="发布方案已经确认"),
        )
    )

    assert len(result.entries) == 1
    assert result.entries[0].source_entry_ids == ("offline", "realtime")
    assert result.metrics.duplicate_entry_count == 1


def test_same_start_different_length_keeps_more_complete_entry() -> None:
    result = build_canonical_transcript(
        (
            entry("short", end=1_500, text="发布方案已经"),
            entry("long", start=1_050, end=2_000, text="发布方案已经确认"),
        )
    )

    assert len(result.entries) == 1
    assert result.entries[0].entry_id == "long"
    assert result.entries[0].text == "发布方案已经确认"


def test_contained_same_speaker_overlap_keeps_complete_entry_and_sources() -> None:
    result = build_canonical_transcript(
        (
            entry("short", user=1, start=1_000, end=2_000, text="发布方案已经"),
            entry("long", user=1, start=1_010, end=2_100, text="发布方案已经确认"),
        )
    )

    assert len(result.entries) == 1
    assert result.entries[0].text == "发布方案已经确认"
    assert result.entries[0].source_entry_ids == ("long", "short")


def test_short_contained_boundary_fragment_is_not_a_duplicate() -> None:
    result = build_canonical_transcript(
        (
            entry(
                "fragment",
                start=1_000,
                end=1_400,
                text="明天整理最终方案",
                source_key="utterance:boundary",
            ),
            entry(
                "complete",
                start=1_100,
                end=3_000,
                text="明天整理最终方案已经确认",
                source_key="segment:complete",
            ),
        )
    )

    assert tuple(item.source_entry_ids for item in result.entries) == (
        ("fragment",),
        ("complete",),
    )


def test_high_similarity_segment_and_utterance_align_despite_low_time_iou() -> None:
    result = build_canonical_transcript(
        (
            entry(
                "segment",
                start=1_000,
                end=4_000,
                text="明天整理最终方案",
                source_key="segment:offline",
            ),
            entry(
                "utterance",
                start=2_100,
                end=3_900,
                text="明日整理最终方案",
                source_key="utterance:realtime",
            ),
        )
    )

    assert tuple(item.source_entry_ids for item in result.entries) == (("segment", "utterance"),)


def test_exclusive_segment_and_utterance_align_equal_length_asr_variants() -> None:
    result = build_canonical_transcript(
        (
            entry(
                "segment",
                start=1_000,
                end=4_000,
                text="离线最终转写",
                source_key="segment:offline",
            ),
            entry(
                "utterance",
                start=2_000,
                end=3_000,
                text="实时听写偏差",
                source_key="utterance:realtime",
            ),
        )
    )

    assert tuple(item.source_entry_ids for item in result.entries) == (("segment", "utterance"),)


def test_divergent_segment_spanning_multiple_utterances_is_not_aligned() -> None:
    result = build_canonical_transcript(
        (
            entry(
                "segment",
                start=1_000,
                end=5_000,
                text="离线最终转写",
                source_key="segment:offline",
            ),
            entry(
                "utterance-a",
                start=1_500,
                end=2_500,
                text="实时听写偏差",
                source_key="utterance:first",
            ),
            entry(
                "utterance-b",
                start=3_000,
                end=4_000,
                text="另一句实时文本",
                source_key="utterance:second",
            ),
        )
    )

    assert tuple(item.source_entry_ids for item in result.entries) == (
        ("segment",),
        ("utterance-a",),
        ("utterance-b",),
    )


@pytest.mark.parametrize(
    ("segment_text", "utterance_text"),
    (
        ("确认两项计划", "确认三项计划"),
        ("不发布方案吗", "发布方案"),
    ),
)
def test_cross_source_alignment_preserves_material_conflict_guards(
    segment_text: str,
    utterance_text: str,
) -> None:
    result = build_canonical_transcript(
        (
            entry(
                "segment",
                start=1_000,
                end=4_000,
                text=segment_text,
                source_key="segment:offline",
            ),
            entry(
                "utterance",
                start=2_000,
                end=3_000,
                text=utterance_text,
                source_key="utterance:realtime",
            ),
        )
    )

    assert tuple(item.source_entry_ids for item in result.entries) == (
        ("segment",),
        ("utterance",),
    )


@pytest.mark.parametrize(
    ("first_text", "second_text"),
    (
        ("同意", "不同意"),
        ("发布方案吗", "发布方案"),
        ("发布本周会议纪要需要马上完成并记录", "安排本周会议纪要需要马上完成并记录"),
        ("2", "12"),
        ("停止录音", "录音"),
    ),
)
def test_overlapping_material_signals_with_different_intent_are_never_merged(
    first_text: str,
    second_text: str,
) -> None:
    result = build_canonical_transcript(
        (
            entry("first", text=first_text),
            entry("second", start=1_050, end=2_050, text=second_text),
        )
    )

    assert tuple(item.source_entry_ids for item in result.entries) == (("first",), ("second",))


@pytest.mark.parametrize(
    ("assertion", "negated_variant"),
    (
        ("可以", "不可以"),
        ("发布方案", "不发布方案"),
        ("可以发布方案", "不能发布方案"),
        ("同意发布方案", "未同意发布方案"),
        ("有录音", "无录音"),
        ("发布方案", "否发布方案"),
        ("发布方案", "非发布方案"),
        ("发布方案", "别发布方案"),
        ("发布方案", "勿发布方案"),
        ("发布方案", "莫发布方案"),
        ("发布方案", "禁止发布方案"),
        ("发布方案", "不得发布方案"),
        ("发布方案", "不许发布方案"),
        ("发布方案", "不要发布方案"),
        ("发布方案", "拒绝发布方案"),
        ("发布方案", "取消发布方案"),
        ("发布方案", "停止发布方案"),
        ("发布方案", "反对发布方案"),
        ("发布", "勿发布"),
        ("发布", "不得发布"),
        ("发布", "反对发布"),
    ),
)
def test_semantic_negation_never_merges_with_a_positive_contained_variant(
    assertion: str,
    negated_variant: str,
) -> None:
    result = build_canonical_transcript(
        (
            entry("assertion", text=assertion),
            entry("negated", start=1_050, end=2_050, text=negated_variant),
        )
    )

    assert tuple(item.source_entry_ids for item in result.entries) == (
        ("assertion",),
        ("negated",),
    )


def test_equally_negative_prohibitions_are_not_opposite_by_polarity() -> None:
    first = _source_entry(entry("first", text="别发布方案"))
    second = _source_entry(entry("second", text="禁止发布方案"))

    assert _material_signature(first.text).polarity == "negated"
    assert _material_signature(second.text).polarity == "negated"
    assert _materially_compatible(first, second)


@pytest.mark.parametrize(
    "question",
    (
        "是否发布方案",
        "能否发布方案",
        "可否发布方案",
        "要不要发布方案",
        "有没有发布方案",
        "谁发布方案",
        "什么发布方案",
        "何时发布方案",
        "哪里发布方案",
        "为什么发布方案",
        "怎么发布方案",
    ),
)
def test_interrogative_prefixes_never_merge_with_the_contained_assertion(question: str) -> None:
    result = build_canonical_transcript(
        (
            entry("assertion", text="发布方案"),
            entry("question", start=1_050, end=2_050, text=question),
        )
    )

    assert tuple(item.source_entry_ids for item in result.entries) == (
        ("assertion",),
        ("question",),
    )


def test_high_overlap_similar_entries_merge() -> None:
    result = build_canonical_transcript(
        (
            entry("first", start=1_000, end=2_000, text="明天发布最终方案"),
            entry("second", start=1_050, end=2_050, text="明天发布最终方案。"),
        )
    )

    assert len(result.entries) == 1
    assert result.entries[0].source_entry_ids == ("first", "second")


def test_duplicate_chain_does_not_merge_incompatible_endpoints() -> None:
    result = build_canonical_transcript(
        (
            entry("a", text="abcdefghij"),
            entry("b", start=1_050, end=2_050, text="abcdefghiX"),
            entry("c", start=1_150, end=2_150, text="abcdefghXY"),
        )
    )

    assert tuple(item.source_entry_ids for item in result.entries) == (("a", "b"), ("c",))


def test_duplicate_chain_cannot_extend_past_the_bounded_window() -> None:
    result = build_canonical_transcript(
        (
            entry("a", start=0, end=1_000, text="abcdefghij"),
            entry("b", start=100, end=31_000, text="abcdefghiX"),
            entry("c", start=30_050, end=31_050, text="abcdefghiX"),
        )
    )

    assert tuple(item.source_entry_ids for item in result.entries) == (("a", "b"), ("c",))


def test_same_speaker_repeated_speech_at_separate_times_is_preserved() -> None:
    result = build_canonical_transcript(
        (
            entry("first", start=1_000, end=2_000, text="发布方案"),
            entry("second", start=10_000, end=11_000, text="发布方案"),
        )
    )

    assert tuple(item.entry_id for item in result.entries) == ("first", "second")


def test_cross_speaker_overlap_is_never_merged() -> None:
    result = build_canonical_transcript(
        (
            entry("speaker-one", user=1, text="发布方案已经确认"),
            entry("speaker-two", user=2, text="发布方案已经确认"),
        )
    )

    assert tuple(item.source_entry_ids for item in result.entries) == (
        ("speaker-one",),
        ("speaker-two",),
    )


def test_filler_is_retained_and_classified_as_backchannel() -> None:
    result = build_canonical_transcript((entry("filler", text="嗯嗯"),))

    assert len(result.entries) == 1
    assert result.entries[0].text == "嗯嗯"
    assert result.entries[0].entry_type == "backchannel"
    assert result.metrics.backchannel_entry_count == 1


def test_short_agreement_is_not_classified_as_a_backchannel() -> None:
    result = build_canonical_transcript((entry("agreement", text="同意"),))

    assert result.entries[0].entry_type == "statement"
    assert result.metrics.backchannel_entry_count == 0


def test_short_number_is_not_classified_as_a_backchannel() -> None:
    result = build_canonical_transcript((entry("number", text="2"),))

    assert result.entries[0].text == "2"
    assert result.entries[0].entry_type == "statement"


def test_recognised_asr_tags_move_to_metadata() -> None:
    result = build_canonical_transcript(
        (entry("tagged", text="<|zh|><|NEUTRAL|><|Speech|><|withitn|>发布方案"),)
    )

    canonical = result.entries[0]
    assert canonical.text == "发布方案"
    assert canonical.asr_tags == ("NEUTRAL", "Speech", "withitn", "zh")
    assert canonical.language == "zh"
    assert canonical.emotion == "NEUTRAL"


def test_raw_provider_tags_are_retained_when_normalised_text_is_clean() -> None:
    result = build_canonical_transcript(
        (
            entry(
                "provider-shaped",
                raw_text="<|zh|><|HAPPY|><|Speech|><|withitn|>发布方案",
                normalised_text="发布方案",
            ),
        )
    )

    canonical = result.entries[0]
    assert canonical.text == "发布方案"
    assert canonical.asr_tags == ("HAPPY", "Speech", "withitn", "zh")
    assert canonical.language == "zh"
    assert canonical.emotion == "HAPPY"


def test_newest_authoritative_revision_outranks_a_longer_duplicate() -> None:
    result = build_canonical_transcript(
        (
            entry(
                "offline",
                text="发布方案已经确认",
                source_key="segment:offline-1",
            ),
            entry(
                "corrected",
                start=1_050,
                end=2_050,
                text="发布方案已经",
                source_key="utterance:revised-1",
                revisions=(
                    {
                        "raw_text": "发布方案已经确认",
                        "normalised_text": "发布方案已经确认",
                        "revision": 2,
                    },
                ),
            ),
        )
    )

    assert len(result.entries) == 1
    assert result.entries[0].entry_id == "corrected"
    assert result.entries[0].text == "发布方案已经"
    assert result.entries[0].source_entry_ids == ("corrected", "offline")
    assert result.entries[0].raw_texts == ("发布方案已经", "发布方案已经确认")


def test_offline_segment_source_precedence_outranks_a_longer_realtime_duplicate() -> None:
    result = build_canonical_transcript(
        (
            entry(
                "realtime",
                text="发布方案已经确认",
                source_key="utterance:realtime-1",
            ),
            entry(
                "offline",
                start=1_050,
                end=2_050,
                text="发布方案已经",
                source_key="segment:offline-1",
            ),
        )
    )

    assert len(result.entries) == 1
    assert result.entries[0].entry_id == "offline"
    assert result.entries[0].text == "发布方案已经"


def test_same_speaker_entries_share_a_group_only_without_meaningful_interruption() -> None:
    uninterrupted = build_canonical_transcript(
        (
            entry("first", start=1_000, end=2_000),
            entry("second", start=2_400, end=3_000),
        )
    )
    interrupted = build_canonical_transcript(
        (
            entry("first", start=1_000, end=2_000),
            entry("other", user=2, start=2_100, end=2_500, text="需要确认"),
            entry("second", start=2_400, end=3_000),
        )
    )

    assert (
        uninterrupted.entries[0].utterance_group_id == uninterrupted.entries[1].utterance_group_id
    )
    assert interrupted.entries[0].utterance_group_id != interrupted.entries[2].utterance_group_id


def test_serialized_hash_is_deterministic_for_source_order() -> None:
    entries = (
        entry("second", start=2_000, end=3_000, text="同意"),
        entry("first", start=1_000, end=2_000, text="发布方案"),
    )

    forward = build_canonical_transcript(entries)
    reverse = build_canonical_transcript(tuple(reversed(entries)))

    assert forward.serialized_hash == reverse.serialized_hash
    assert forward.to_dict() == reverse.to_dict()
