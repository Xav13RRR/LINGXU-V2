"""Durable outcome artifact codecs, policy provenance, and budget journals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from lingxu.storage.session_files import atomic_write_text
from lingxu.worker.outcome_partitions import (
    LlmBudgetController,
    OutcomePartitionError,
    OutcomePartitionManifest,
    OutcomePartitionResult,
    UnifiedLlmBudgetJournal,
    expected_partition_result_keys,
    migrate_fix33_outcome_partition_result,
    validate_partition_result_keys,
)
from lingxu.worker.outcomes import OutcomeRequestKind, TopicOutcomeSet, VoteResult
from lingxu.worker.summary import PostMeetingInferenceProvider
from lingxu.worker.topics import (
    APPROVED_TOPIC_MODEL,
    APPROVED_TOPIC_MODEL_DIGEST,
    APPROVED_TOPIC_PROVIDER,
    topic_boundary_policy_failure,
)

__all__ = [
    "budget_controller",
    "decode_outcome_sets",
    "outcome_hash_input",
    "outcome_policy_provenance",
    "read_budget_journal",
    "read_partition_results",
    "reusable_outcome_payload",
    "sanitized_vote_result",
    "stored_outcome_policy_provenance",
    "validated_outcome_hash_input",
    "write_budget_journal",
    "write_partition_results",
]


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _payload_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def decode_outcome_sets(value: object) -> tuple[TopicOutcomeSet, ...]:
    """Strictly decode topic outcome sets from a persisted JSON collection."""

    if not isinstance(value, list):
        raise ValueError("topic outcome sets are malformed")
    return tuple(
        TopicOutcomeSet.model_validate_json(
            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        )
        for item in value
    )


def outcome_hash_input(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("outcome provenance is malformed")
    return {
        "schema_version": payload.get("schema_version"),
        "provenance": {
            key: value for key, value in provenance.items() if not key.endswith("_artifact_hash")
        },
        "topics": payload.get("topics"),
        "validation_metrics": payload.get("validation_metrics"),
    }


def validated_outcome_hash_input(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("validated outcome provenance is malformed")
    return {
        "schema_version": payload.get("schema_version"),
        "provenance": {
            key: value for key, value in provenance.items() if key != "validated_artifact_hash"
        },
        "topics": payload.get("topics"),
        "validation_metrics": payload.get("validation_metrics"),
        "binding_resolutions": payload.get("binding_resolutions"),
        "individual_rejections": payload.get("individual_rejections"),
    }


def outcome_policy_provenance(
    provider: PostMeetingInferenceProvider | None,
) -> dict[str, object]:
    """Bind outcome artifacts to the exact approved provider identity."""

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
        "remote_allow_authorizes_outcome_calls": True,
    }
    return {**policy, "policy_hash": _payload_hash(policy)}


def stored_outcome_policy_provenance(
    provenance: dict[str, object],
) -> dict[str, object]:
    """Validate persisted outcome policy without consulting a current provider."""

    keys = (
        "status",
        "provider",
        "model",
        "approved_model_digest",
        "endpoint_policy",
        "remote_allow_authorizes_outcome_calls",
    )
    policy = {key: provenance.get(key) for key in keys}
    status = policy["status"]
    provider_interface = provenance.get("provider_interface")
    if status == "approved":
        if (
            policy["provider"] != APPROVED_TOPIC_PROVIDER
            or policy["model"] != APPROVED_TOPIC_MODEL
            or provider_interface not in {"single_kind", "unsupported"}
        ):
            raise ValueError("outcome policy is invalid")
    elif status in {
        "provider_disabled",
        "policy_unproven",
        "provider_not_approved",
        "model_not_approved",
        "digest_not_approved",
        "endpoint_not_private",
        "endpoint_not_approved",
    }:
        if (
            policy["provider"] is not None
            or policy["model"] is not None
            or provider_interface != "unsupported"
        ):
            raise ValueError("outcome policy is invalid")
    else:
        raise ValueError("outcome policy is invalid")
    if (
        policy["approved_model_digest"] != APPROVED_TOPIC_MODEL_DIGEST
        or policy["endpoint_policy"] != "official_remote_https_only"
        or policy["remote_allow_authorizes_outcome_calls"] is not True
    ):
        raise ValueError("outcome policy is invalid")
    policy_hash = provenance.get("policy_hash")
    if policy_hash != _payload_hash(policy):
        raise ValueError("outcome policy hash is invalid")
    return {**policy, "policy_hash": policy_hash}


def sanitized_vote_result(payload: dict[str, Any]) -> VoteResult:
    """Reduce supported vote result shapes to the closed outcome vocabulary."""

    result = payload.get("result")
    if result is True:
        return "accepted"
    if result is False:
        return "rejected"
    if isinstance(result, str):
        normalized = result.strip().casefold().replace(" ", "_")
        aliases = {"pass": "passed", "tie": "tied", "voted_down": "rejected"}
        normalized = aliases.get(normalized, normalized)
        allowed = {
            "accepted",
            "approved",
            "passed",
            "yes",
            "adopted",
            "carried",
            "rejected",
            "denied",
            "failed",
            "no",
            "tied",
            "pending",
        }
        return cast(VoteResult, normalized) if normalized in allowed else "unknown"
    for key, normalized in (
        ("accepted", "accepted"),
        ("passed", "passed"),
        ("approved", "approved"),
    ):
        if payload.get(key) is True:
            return cast(VoteResult, normalized)
    return "unknown"


def reusable_outcome_payload(
    *,
    path: Path,
    expected_provenance: dict[str, object],
    topic_ids: tuple[str, ...],
) -> dict[str, Any] | None:
    """Return an exact current outcome artifact, or reject without mutation."""

    if not path.is_file():
        return None
    try:
        serialized = path.read_text(encoding="utf-8")
        payload = json.loads(serialized)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return None
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict) or any(
            provenance.get(key) != value for key, value in expected_provenance.items()
        ):
            return None
        outcome_sets = decode_outcome_sets(payload.get("topics"))
        if tuple(outcome.topic_id for outcome in outcome_sets) != topic_ids:
            return None
        if provenance.get("outcome_artifact_hash") != _payload_hash(outcome_hash_input(payload)):
            return None
        if serialized != _json_text(payload):
            return None
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload


def read_partition_results(
    path: Path,
    manifest: OutcomePartitionManifest,
    *,
    pipeline_revision: str,
    journal: UnifiedLlmBudgetJournal | None = None,
) -> dict[tuple[str, OutcomeRequestKind], OutcomePartitionResult]:
    """Read, validate, and when authorized migrate terminal partition results."""

    if not path.is_file():
        return {}
    legacy_payload = False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "manifest_hash",
            "results",
        }:
            raise OutcomePartitionError("partition_result_tamper")
        schema_version = payload["schema_version"]
        if schema_version not in {1, 2}:
            raise OutcomePartitionError("partition_result_tamper")
        if payload["manifest_hash"] != manifest.manifest_hash:
            return {}
        raw_results = payload["results"]
        if not isinstance(raw_results, list):
            raise OutcomePartitionError("partition_result_tamper")
        partitions = {partition.partition_id: partition for partition in manifest.partitions}
        results: dict[tuple[str, OutcomeRequestKind], OutcomePartitionResult] = {}
        migrated = schema_version == 1
        legacy_payload = migrated
        if migrated and journal is None:
            raise OutcomePartitionError("outcome_disposition_migration_required")
        if migrated:
            assert journal is not None
            journal.validate_final()
        for raw in raw_results:
            if not isinstance(raw, dict):
                raise OutcomePartitionError("partition_result_tamper")
            raw_partition_id = raw.get("partition_id")
            partition = (
                partitions.get(raw_partition_id) if isinstance(raw_partition_id, str) else None
            )
            if partition is None:
                raise OutcomePartitionError("partition_result_tamper")
            if migrated:
                assert journal is not None
                result = migrate_fix33_outcome_partition_result(
                    raw,
                    partition=partition,
                    journal=journal,
                    pipeline_revision=pipeline_revision,
                )
            else:
                result = OutcomePartitionResult.model_validate_json(
                    json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
                )
            partition = partitions.get(result.partition_id)
            if partition is None:
                raise OutcomePartitionError("partition_result_tamper")
            result.validate_for(partition, result.request_kind, pipeline_revision)
            key = (result.partition_id, result.request_kind)
            if key in results:
                raise OutcomePartitionError("partition_result_tamper")
            results[key] = result
        if migrated:
            validate_partition_result_keys(
                tuple(results),
                expected_partition_result_keys(manifest),
            )
            write_partition_results(path, manifest, results)
        return results
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if legacy_payload:
            raise OutcomePartitionError("outcome_disposition_migration_required") from exc
        if isinstance(exc, OutcomePartitionError):
            raise
        raise OutcomePartitionError("partition_result_tamper") from exc


def write_partition_results(
    path: Path,
    manifest: OutcomePartitionManifest,
    results: dict[tuple[str, OutcomeRequestKind], OutcomePartitionResult],
) -> None:
    atomic_write_text(
        path,
        _json_text(
            {
                "schema_version": 2,
                "manifest_hash": manifest.manifest_hash,
                "results": [
                    result.model_dump(mode="json")
                    for _, result in sorted(
                        results.items(), key=lambda item: (item[0][0], item[0][1].value)
                    )
                ],
            }
        ),
    )


def read_budget_journal(
    path: Path,
    *,
    job_hash: str,
    expected_task_hashes: tuple[str, ...],
    prior_task_hashes: tuple[str, ...] | None = None,
) -> UnifiedLlmBudgetJournal:
    """Read and validate the unified LLM budget journal or create an empty authority."""

    if not path.is_file():
        return UnifiedLlmBudgetJournal.new(
            job_hash=job_hash,
            expected_task_hashes=expected_task_hashes,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"journal"}:
            raise OutcomePartitionError("budget_journal_tamper")
        journal = UnifiedLlmBudgetJournal.model_validate_json(
            json.dumps(payload["journal"], ensure_ascii=False, separators=(",", ":"))
        )
        journal.validate_job(job_hash)
        if journal.expected_task_hashes == expected_task_hashes:
            return journal
        if prior_task_hashes is not None:
            return journal.transition_expected(
                prior_task_hashes=prior_task_hashes,
                task_hashes=expected_task_hashes,
            )
        journal.validate_expected(expected_task_hashes)
        return journal
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, OutcomePartitionError):
            raise
        raise OutcomePartitionError("budget_journal_tamper") from exc


def write_budget_journal(path: Path, journal: UnifiedLlmBudgetJournal) -> None:
    atomic_write_text(
        path,
        _json_text({"journal": journal.model_dump(mode="json")}),
    )


def budget_controller(
    path: Path,
    *,
    job_hash: str,
    expected_task_hashes: tuple[str, ...],
    prior_task_hashes: tuple[str, ...] | None = None,
) -> LlmBudgetController:
    """Open and durably persist a controller before any provider boundary."""

    journal = read_budget_journal(
        path,
        job_hash=job_hash,
        expected_task_hashes=expected_task_hashes,
        prior_task_hashes=prior_task_hashes,
    )
    write_budget_journal(path, journal)
    return LlmBudgetController(
        journal,
        lambda updated: write_budget_journal(path, updated),
    )
