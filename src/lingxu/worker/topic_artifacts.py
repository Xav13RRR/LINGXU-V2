"""Durable topic artifact policy, reuse validation, and stage orchestration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, cast

from lingxu.storage.session_files import atomic_write_text
from lingxu.transcript.models import CanonicalTranscript
from lingxu.worker.outcome_partitions import (
    MAX_TOPIC_BOUNDARY_CALLS_PER_JOB,
    LlmBudgetController,
    OutcomePartitionError,
    boundary_task_hash,
    digest_task_hash,
    llm_job_hash,
)
from lingxu.worker.summary import PostMeetingInferenceProvider
from lingxu.worker.topics import (
    APPROVED_TOPIC_MODEL,
    APPROVED_TOPIC_MODEL_DIGEST,
    APPROVED_TOPIC_PROVIDER,
    TOPIC_CANDIDATE_CONFIG_VERSION,
    TopicBoundaryConfirmation,
    TopicBoundaryFailure,
    TopicSegment,
    apply_confirmations,
    confirm_topic_candidates,
    topic_boundary_policy_failure,
    topic_candidates,
)

__all__ = [
    "read_topic_segments",
    "reusable_topic_payload",
    "segment_topics",
    "stored_topic_policy_provenance",
    "topic_policy_provenance",
]

TOPIC_ARTIFACT_SCHEMA_VERSION = 2


class BudgetControllerFactory(Protocol):
    """Construct the existing durable unified-LLM budget controller."""

    def __call__(
        self,
        path: Path,
        *,
        job_hash: str,
        expected_task_hashes: tuple[str, ...],
    ) -> LlmBudgetController: ...


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def topic_policy_provenance(
    provider: PostMeetingInferenceProvider | None,
) -> dict[str, object]:
    """Bind the current artifact to the exact owner-approved provider identity."""

    if provider is None:
        status = "provider_disabled"
    else:
        status = topic_boundary_policy_failure(provider) or "approved"
        if (
            status == "approved"
            and provider.topic_boundary_policy.provider != APPROVED_TOPIC_PROVIDER
        ):
            status = "provider_not_approved"
    policy = {
        "status": status,
        "provider": APPROVED_TOPIC_PROVIDER if status == "approved" else None,
        "model": APPROVED_TOPIC_MODEL if status == "approved" else None,
        "approved_model_digest": APPROVED_TOPIC_MODEL_DIGEST,
        "endpoint_policy": "official_remote_https_only",
        "remote_allow_authorizes_topic_calls": True,
    }
    return {**policy, "policy_hash": _payload_hash(policy)}


def _decode_topic_confirmations(value: object) -> tuple[TopicBoundaryConfirmation, ...]:
    if not isinstance(value, list):
        raise ValueError("topic confirmations are malformed")
    confirmations: list[TopicBoundaryConfirmation] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "entry_id",
            "confirmed",
            "confidence",
        }:
            raise ValueError("topic confirmation is malformed")
        confidence = item["confidence"]
        if (
            not isinstance(item["entry_id"], str)
            or not isinstance(item["confirmed"], bool)
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
        ):
            raise ValueError("topic confirmation has invalid types")
        confirmations.append(
            TopicBoundaryConfirmation(
                entry_id=item["entry_id"],
                confirmed=item["confirmed"],
                confidence=float(confidence),
            )
        )
    return tuple(confirmations)


def _topic_payload_hash_input(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("topic provenance is malformed")
    return {
        "schema_version": payload.get("schema_version"),
        "provenance": {
            key: value for key, value in provenance.items() if key != "topic_artifact_hash"
        },
        "candidates": payload.get("candidates"),
        "confirmations": payload.get("confirmations"),
        "failures": payload.get("failures"),
        "segments": payload.get("segments"),
    }


def reusable_topic_payload(
    *,
    path: Path,
    canonical: CanonicalTranscript,
    candidate_payload: list[dict[str, Any]],
    policy_provenance: dict[str, object],
    candidate_config_version: int = TOPIC_CANDIDATE_CONFIG_VERSION,
) -> dict[str, Any] | None:
    """Return an exact current durable artifact, or reject it without mutation."""

    if not path.is_file():
        return None
    try:
        serialized = path.read_text(encoding="utf-8")
        payload = json.loads(serialized)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != TOPIC_ARTIFACT_SCHEMA_VERSION
        ):
            return None
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            return None
        if provenance.get("canonical_input_hash") != canonical.serialized_hash:
            return None
        if provenance.get("candidate_config_version") != candidate_config_version:
            return None
        if provenance.get("candidate_set_hash") != _payload_hash(candidate_payload):
            return None
        if any(provenance.get(key) != value for key, value in policy_provenance.items()):
            return None
        if payload.get("candidates") != candidate_payload:
            return None
        confirmations = _decode_topic_confirmations(payload.get("confirmations"))
        candidate_ids = {candidate["entry_id"] for candidate in candidate_payload}
        confirmation_ids = [confirmation.entry_id for confirmation in confirmations]
        if any(entry_id not in candidate_ids for entry_id in confirmation_ids) or len(
            set(confirmation_ids)
        ) != len(confirmation_ids):
            return None
        if provenance.get("confirmation_hash") != _payload_hash(payload["confirmations"]):
            return None
        expected_segments = [
            {
                **asdict(segment),
                "entry_ids": list(segment.entry_ids),
            }
            for segment in apply_confirmations(
                canonical.entries,
                topic_candidates(canonical.entries),
                confirmations,
            )
        ]
        if payload.get("segments") != expected_segments:
            return None
        if provenance.get("topic_artifact_hash") != _payload_hash(
            _topic_payload_hash_input(payload)
        ):
            return None
        failures = payload.get("failures")
        if not isinstance(failures, list) or any(
            not isinstance(failure, dict)
            or set(failure) != {"entry_id", "code"}
            or not isinstance(failure["entry_id"], str)
            or failure["entry_id"] not in candidate_ids
            or not isinstance(failure["code"], str)
            for failure in failures
        ):
            return None
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return payload


def stored_topic_policy_provenance(path: Path) -> dict[str, object]:
    """Validate persisted provider policy without consulting a current provider."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload.get("provenance") if isinstance(payload, dict) else None
        if not isinstance(provenance, dict):
            raise ValueError("topic provenance is malformed")
        keys = (
            "status",
            "provider",
            "model",
            "approved_model_digest",
            "endpoint_policy",
            "remote_allow_authorizes_topic_calls",
        )
        policy = {key: provenance.get(key) for key in keys}
        status = policy["status"]
        if status == "approved":
            if (
                policy["provider"] != APPROVED_TOPIC_PROVIDER
                or policy["model"] != APPROVED_TOPIC_MODEL
            ):
                raise ValueError("topic policy is invalid")
        elif status in {
            "provider_disabled",
            "policy_unproven",
            "provider_not_approved",
            "model_not_approved",
            "digest_not_approved",
            "endpoint_not_private",
            "endpoint_not_approved",
        }:
            if policy["provider"] is not None or policy["model"] is not None:
                raise ValueError("topic policy is invalid")
        else:
            raise ValueError("topic policy is invalid")
        if (
            policy["approved_model_digest"] != APPROVED_TOPIC_MODEL_DIGEST
            or policy["endpoint_policy"] != "official_remote_https_only"
            or policy["remote_allow_authorizes_topic_calls"] is not True
        ):
            raise ValueError("topic policy is invalid")
        policy_hash = provenance.get("policy_hash")
        if policy_hash != _payload_hash(policy):
            raise ValueError("topic policy hash is invalid")
        return {**policy, "policy_hash": policy_hash}
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("topic policy is malformed") from exc


async def segment_topics(
    *,
    canonical: CanonicalTranscript,
    meeting_dir: Path,
    provider: PostMeetingInferenceProvider | None,
    budget_controller_factory: BudgetControllerFactory,
    pipeline_revision: str,
) -> dict[str, Any]:
    """Create or reuse the durable topic artifact under the unified LLM budget."""

    candidates = topic_candidates(canonical.entries)
    if len(candidates) > MAX_TOPIC_BOUNDARY_CALLS_PER_JOB:
        raise OutcomePartitionError("topic_boundary_call_cap")
    job_hash = llm_job_hash(
        canonical_hash=canonical.serialized_hash,
        pipeline_revision=pipeline_revision,
    )
    budget_path = meeting_dir / "outcome_budget_journal.json"
    candidate_payload = [
        {**asdict(candidate), "reason": list(candidate.reason)} for candidate in candidates
    ]
    policy_provenance = topic_policy_provenance(provider)
    boundary_tasks = tuple(
        boundary_task_hash(job_hash, candidate.entry_id) for candidate in candidates
    )
    binder = getattr(provider, "bind_llm_budget", None) if provider is not None else None
    provider_ready = policy_provenance["status"] == "approved" and callable(binder)
    expected_tasks = (
        (digest_task_hash(job_hash), *boundary_tasks) if provider_ready and candidates else ()
    )
    budget_controller = budget_controller_factory(
        budget_path,
        job_hash=job_hash,
        expected_task_hashes=expected_tasks,
    )
    if provider_ready:
        cast(Callable[[LlmBudgetController], None], binder)(budget_controller)
    topic_path = meeting_dir / "topic_segments.json"
    reusable = reusable_topic_payload(
        path=topic_path,
        canonical=canonical,
        candidate_payload=candidate_payload,
        policy_provenance=policy_provenance,
    )
    if reusable is not None:
        for task_hash in boundary_tasks if provider_ready else ():
            budget_controller.reserve_task(task_hash)
            budget_controller.reconcile_result(task_hash)
        return {
            "candidate_count": len(reusable["candidates"]),
            "confirmed_boundary_count": max(0, len(reusable["segments"]) - 1),
            "topic_count": len(reusable["segments"]),
            "provider_failures": len(reusable["failures"]),
            "artifact_reused": True,
            "topic_artifact_sha256": _sha256(topic_path),
        }
    confirmations: tuple[TopicBoundaryConfirmation, ...] = ()
    failures: tuple[TopicBoundaryFailure, ...] = ()
    if provider_ready and candidates:
        assert provider is not None
        confirmations, failures = await confirm_topic_candidates(
            canonical.entries,
            candidates,
            provider,
        )
    elif candidates:
        failure_code = str(policy_provenance["status"])
        failures = tuple(
            TopicBoundaryFailure(candidate.entry_id, failure_code) for candidate in candidates
        )
    segments = apply_confirmations(canonical.entries, candidates, confirmations)
    confirmation_payload = [asdict(confirmation) for confirmation in confirmations]
    payload: dict[str, Any] = {
        "schema_version": TOPIC_ARTIFACT_SCHEMA_VERSION,
        "provenance": {
            **policy_provenance,
            "canonical_input_hash": canonical.serialized_hash,
            "candidate_config_version": TOPIC_CANDIDATE_CONFIG_VERSION,
            "candidate_set_hash": _payload_hash(candidate_payload),
            "confirmation_hash": _payload_hash(confirmation_payload),
        },
        "candidates": candidate_payload,
        "confirmations": confirmation_payload,
        "failures": [asdict(failure) for failure in failures],
        "segments": [
            {**asdict(segment), "entry_ids": list(segment.entry_ids)} for segment in segments
        ],
    }
    payload["provenance"]["topic_artifact_hash"] = _payload_hash(_topic_payload_hash_input(payload))
    atomic_write_text(topic_path, _json_text(payload))
    for task_hash in boundary_tasks if provider_ready else ():
        budget_controller.reserve_task(task_hash)
        budget_controller.reconcile_result(task_hash)
    return {
        "candidate_count": len(candidates),
        "confirmed_boundary_count": max(0, len(segments) - 1),
        "topic_count": len(segments),
        "provider_failures": len(failures),
        "failure_codes": sorted({failure.code for failure in failures}),
        "artifact_reused": False,
        "topic_artifact_sha256": _sha256(topic_path),
    }


def read_topic_segments(meeting_dir: Path) -> tuple[TopicSegment, ...]:
    """Strictly decode the durable topic artifact used by downstream stages."""

    payload = json.loads((meeting_dir / "topic_segments.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ValueError("topic segment artifact is malformed")
    segments: list[TopicSegment] = []
    for item in payload["segments"]:
        if not isinstance(item, dict) or set(item) != {
            "id",
            "title",
            "entry_ids",
            "started_ms",
            "ended_ms",
            "method",
        }:
            raise ValueError("topic segment is malformed")
        entry_ids = item["entry_ids"]
        if (
            not isinstance(item["id"], str)
            or not isinstance(item["title"], str)
            or not isinstance(entry_ids, list)
            or not all(isinstance(entry_id, str) for entry_id in entry_ids)
            or isinstance(item["started_ms"], bool)
            or not isinstance(item["started_ms"], int)
            or isinstance(item["ended_ms"], bool)
            or not isinstance(item["ended_ms"], int)
            or not isinstance(item["method"], str)
        ):
            raise ValueError("topic segment has invalid types")
        segments.append(
            TopicSegment(
                id=item["id"],
                title=item["title"],
                entry_ids=tuple(entry_ids),
                started_ms=item["started_ms"],
                ended_ms=item["ended_ms"],
                method=item["method"],
            )
        )
    if len({segment.id for segment in segments}) != len(segments):
        raise ValueError("topic segment identifiers are not unique")
    return tuple(segments)
