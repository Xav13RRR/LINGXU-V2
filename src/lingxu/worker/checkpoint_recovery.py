"""Pure validation and schema recovery for Worker pipeline checkpoints."""

from __future__ import annotations

import re
from collections.abc import Mapping

from lingxu.worker.stages import (
    POST_MEETING_STAGES,
    stage_index,
    truncate_completed_before,
)

CHECKPOINT_SCHEMA_VERSION = 4
_INSERTED_STAGES = (
    "CANONICALISE_TRANSCRIPT",
    "NORMALISE_ENTITIES",
    "SEGMENT_TOPICS",
    "EXTRACT_TOPIC_OUTCOMES",
    "LINK_EVIDENCE",
    "VALIDATE_OUTCOMES",
    "VALIDATE_MINUTES_CONSISTENCY",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def recover_renderer_input_sha256(checkpoint: Mapping[str, object]) -> str | None:
    """Return only a checkpointed renderer digest with the exact SHA-256 shape."""

    value = checkpoint.get("renderer_input_sha256")
    return value if isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) else None


def recover_attempt_generation(checkpoint: Mapping[str, object]) -> int:
    """Validate the bounded retry generation stored by the queue repository."""

    value = checkpoint.get("full_transcript_minutes_attempt_generation", 0)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 1_000:
        raise ValueError("full transcript minutes attempt generation is invalid")
    return value


def _known_completed(checkpoint: Mapping[str, object]) -> list[str]:
    raw = checkpoint.get("completed_stages", [])
    if not isinstance(raw, list):
        return []
    return [str(stage) for stage in raw if str(stage) in POST_MEETING_STAGES]


def recover_completed_stages(checkpoint: Mapping[str, object]) -> list[str]:
    """Project legacy checkpoint schemas onto the current ordered stage prefix."""

    schema_version = checkpoint.get("schema_version")
    if schema_version in {3, CHECKPOINT_SCHEMA_VERSION}:
        completed = _known_completed(checkpoint)
        completed_set = set(completed)
        for inserted_stage in _INSERTED_STAGES:
            if inserted_stage in completed_set:
                continue
            inserted_index = stage_index(inserted_stage)
            if any(stage_index(stage) > inserted_index for stage in completed_set):
                completed = [stage for stage in completed if stage_index(stage) < inserted_index]
            break
        if schema_version == 3:
            return truncate_completed_before(completed, "VALIDATE_MINUTES_CONSISTENCY")
        return completed
    if schema_version == 2:
        old_completed = set(_known_completed(checkpoint))
        return ["VERIFY_SESSION"] if "VERIFY_SESSION" in old_completed else []
    return []
