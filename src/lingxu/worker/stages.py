"""Closed post-Meeting stage registry and checkpoint-order helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PostMeetingStageDefinition:
    """Bind one durable public stage name to its pipeline handler name."""

    name: str

    @property
    def handler_name(self) -> str:
        return f"_stage_{self.name.lower()}"


POST_MEETING_STAGES = (
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
POST_MEETING_STAGE_DEFINITIONS = tuple(
    PostMeetingStageDefinition(name) for name in POST_MEETING_STAGES
)
_STAGE_INDEX = {name: index for index, name in enumerate(POST_MEETING_STAGES)}

if len(_STAGE_INDEX) != len(POST_MEETING_STAGES):
    raise RuntimeError("post-Meeting stage names must be unique")


def stage_index(stage: str) -> int:
    """Return the stable position of one closed stage name."""

    try:
        return _STAGE_INDEX[stage]
    except KeyError as error:
        raise ValueError("post-Meeting stage is not recognized") from error


def truncate_completed_before(completed: Sequence[str], stage: str) -> list[str]:
    """Keep only completed stages that precede the selected restart stage."""

    selected_index = stage_index(stage)
    return [item for item in completed if stage_index(item) < selected_index]
