from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lingxu.transcript.models import (
    CanonicalTranscript,
    CanonicalTranscriptEntry,
    CanonicalTranscriptMetrics,
)
from lingxu.worker.outcome_artifacts import budget_controller
from lingxu.worker.outcome_partitions import (
    MAX_TOPIC_BOUNDARY_CALLS_PER_JOB,
    OutcomePartitionError,
)
from lingxu.worker.topic_artifacts import (
    read_topic_segments,
    segment_topics,
    stored_topic_policy_provenance,
    topic_policy_provenance,
)
from lingxu.worker.topics import (
    APPROVED_TOPIC_MODEL_DIGEST,
    APPROVED_TOPIC_PROVIDER,
)


def _payload_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_with_transition_candidates(candidate_count: int) -> CanonicalTranscript:
    entries = tuple(
        CanonicalTranscriptEntry(
            entry_id=f"entry-{index:03d}",
            source_entry_ids=(f"source-{index:03d}",),
            speaker_user_id=100,
            speaker_name="participant",
            started_ms=index * 1_000,
            ended_ms=index * 1_000 + 800,
            raw_texts=("初始说明" if index == 0 else "接下来讨论后续事项",),
            text="初始说明" if index == 0 else "接下来讨论后续事项",
            entry_type="statement",
            language="zh",
            confidence=0.95,
            utterance_group_id=f"group-{index:03d}",
            normalisation_changes=(),
            asr_tags=(),
            emotion=None,
        )
        for index in range(candidate_count + 1)
    )
    return CanonicalTranscript(
        entries=entries,
        metrics=CanonicalTranscriptMetrics(
            source_entry_count=len(entries),
            canonical_entry_count=len(entries),
            duplicate_entry_count=0,
            backchannel_entry_count=0,
            asr_tag_count=0,
        ),
    )


def test_disabled_topic_provider_has_content_free_hashed_policy() -> None:
    policy = topic_policy_provenance(None)

    assert policy == {
        "status": "provider_disabled",
        "provider": None,
        "model": None,
        "approved_model_digest": APPROVED_TOPIC_MODEL_DIGEST,
        "endpoint_policy": "official_remote_https_only",
        "remote_allow_authorizes_topic_calls": True,
        "policy_hash": policy["policy_hash"],
    }
    assert policy["policy_hash"] == _payload_hash(
        {key: value for key, value in policy.items() if key != "policy_hash"}
    )


def test_stored_topic_policy_accepts_only_bound_approved_identity(tmp_path: Path) -> None:
    policy = topic_policy_provenance(None)
    approved = {
        **policy,
        "status": "approved",
        "provider": APPROVED_TOPIC_PROVIDER,
        "model": "deepseek-flash",
    }
    approved["policy_hash"] = _payload_hash(
        {key: value for key, value in approved.items() if key != "policy_hash"}
    )
    path = tmp_path / "topic_segments.json"
    path.write_text(json.dumps({"provenance": approved}), encoding="utf-8")

    assert stored_topic_policy_provenance(path) == approved

    tampered = {**approved, "provider": "ollama"}
    tampered["policy_hash"] = _payload_hash(
        {key: value for key, value in tampered.items() if key != "policy_hash"}
    )
    path.write_text(json.dumps({"provenance": tampered}), encoding="utf-8")
    with pytest.raises(ValueError, match="topic policy is invalid"):
        stored_topic_policy_provenance(path)


def test_read_topic_segments_validates_shape_and_unique_identity(tmp_path: Path) -> None:
    path = tmp_path / "topic_segments.json"
    segment = {
        "id": "topic-001",
        "title": "部署边界",
        "entry_ids": ["entry-001", "entry-002"],
        "started_ms": 100,
        "ended_ms": 900,
        "method": "deterministic",
    }
    path.write_text(json.dumps({"segments": [segment]}), encoding="utf-8")

    assert read_topic_segments(tmp_path)[0].entry_ids == ("entry-001", "entry-002")

    path.write_text(json.dumps({"segments": [segment, segment]}), encoding="utf-8")
    with pytest.raises(ValueError, match="identifiers are not unique"):
        read_topic_segments(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate_count", (65, 120))
async def test_segment_topics_accepts_owner_approved_long_meeting_candidate_counts(
    tmp_path: Path,
    candidate_count: int,
) -> None:
    result = await segment_topics(
        canonical=_canonical_with_transition_candidates(candidate_count),
        meeting_dir=tmp_path,
        provider=None,
        budget_controller_factory=budget_controller,
        pipeline_revision="a" * 64,
    )

    assert MAX_TOPIC_BOUNDARY_CALLS_PER_JOB == 120
    assert result["candidate_count"] == candidate_count
    payload = json.loads((tmp_path / "topic_segments.json").read_text(encoding="utf-8"))
    assert len(payload["candidates"]) == candidate_count
    assert len(payload["failures"]) == candidate_count


@pytest.mark.asyncio
async def test_segment_topics_rejects_candidate_121_before_budget_or_provider_state(
    tmp_path: Path,
) -> None:
    with pytest.raises(OutcomePartitionError, match="topic_boundary_call_cap"):
        await segment_topics(
            canonical=_canonical_with_transition_candidates(121),
            meeting_dir=tmp_path,
            provider=None,
            budget_controller_factory=budget_controller,
            pipeline_revision="a" * 64,
        )

    assert not (tmp_path / "outcome_budget_journal.json").exists()
    assert not (tmp_path / "topic_segments.json").exists()
