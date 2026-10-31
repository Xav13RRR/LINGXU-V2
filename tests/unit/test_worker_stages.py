import pytest

from lingxu.worker.checkpoint_recovery import (
    recover_attempt_generation,
    recover_completed_stages,
    recover_renderer_input_sha256,
)
from lingxu.worker.pipeline import PostMeetingPipeline
from lingxu.worker.stages import (
    POST_MEETING_STAGE_DEFINITIONS,
    POST_MEETING_STAGES,
    truncate_completed_before,
)


def test_post_meeting_stage_contract_is_exact_and_handler_complete() -> None:
    assert POST_MEETING_STAGES == (
        "VERIFY_SESSION",
        "BUILD_CONSENT_TIMELINE",
        "APPLY_PRIVACY_CUTOFFS",
        "VERIFY_AUDIO_SEGMENTS",
        "FINALISE_REALTIME_UTTERANCES",
        "TRANSCRIBE_MISSING_SEGMENTS",
        "VALIDATE_LANGUAGES",
        "CANONICALISE_TRANSCRIPT",
        "NORMALISE_ENTITIES",
        "SEGMENT_TOPICS",
        "EXTRACT_TOPIC_OUTCOMES",
        "LINK_EVIDENCE",
        "VALIDATE_OUTCOMES",
        "MERGE_TRANSCRIPT",
        "MERGE_MEETING_EVENTS",
        "CALCULATE_LANGUAGE_DISTRIBUTION",
        "GENERATE_EVIDENCE",
        "GENERATE_MINUTES",
        "VALIDATE_MINUTES_CONSISTENCY",
        "RENDER_ARTIFACTS",
        "PUBLISH_DISCORD",
        "SCHEDULE_CLEANUP",
    )
    assert len(set(POST_MEETING_STAGES)) == len(POST_MEETING_STAGES)
    assert tuple(definition.name for definition in POST_MEETING_STAGE_DEFINITIONS) == (
        POST_MEETING_STAGES
    )
    assert all(
        hasattr(PostMeetingPipeline, definition.handler_name)
        for definition in POST_MEETING_STAGE_DEFINITIONS
    )


def test_checkpoint_truncation_keeps_only_the_ordered_prefix() -> None:
    completed = list(POST_MEETING_STAGES[:12])

    assert truncate_completed_before(completed, "SEGMENT_TOPICS") == list(POST_MEETING_STAGES[:9])
    assert truncate_completed_before(completed, "PUBLISH_DISCORD") == completed


def test_checkpoint_recovery_normalises_schema_evolution_and_inserted_stages() -> None:
    current_with_gap = {
        "schema_version": 4,
        "completed_stages": [*POST_MEETING_STAGES[:7], "NORMALISE_ENTITIES"],
    }
    schema_three = {
        "schema_version": 3,
        "completed_stages": list(POST_MEETING_STAGES),
    }
    schema_two = {
        "schema_version": 2,
        "completed_stages": ["VERIFY_SESSION", "PUBLISH_DISCORD"],
    }

    assert recover_completed_stages(current_with_gap) == list(POST_MEETING_STAGES[:7])
    assert recover_completed_stages(schema_three) == list(
        POST_MEETING_STAGES[: POST_MEETING_STAGES.index("VALIDATE_MINUTES_CONSISTENCY")]
    )
    assert recover_completed_stages(schema_two) == ["VERIFY_SESSION"]
    assert recover_completed_stages({"schema_version": 1}) == []


def test_checkpoint_recovery_accepts_only_bounded_attempt_and_sha256_values() -> None:
    digest = "a" * 64

    assert recover_renderer_input_sha256({"renderer_input_sha256": digest}) == digest
    assert recover_renderer_input_sha256({"renderer_input_sha256": "invalid"}) is None
    assert recover_attempt_generation({}) == 0
    assert (
        recover_attempt_generation({"full_transcript_minutes_attempt_generation": 1_000}) == 1_000
    )

    for value in (True, -1, 1_001, "1"):
        with pytest.raises(ValueError, match="attempt generation"):
            recover_attempt_generation({"full_transcript_minutes_attempt_generation": value})
