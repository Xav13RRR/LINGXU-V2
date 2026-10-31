from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lingxu.worker.outcome_artifacts import (
    budget_controller,
    decode_outcome_sets,
    outcome_policy_provenance,
    read_budget_journal,
    sanitized_vote_result,
)
from lingxu.worker.outcome_partitions import OutcomePartitionError
from lingxu.worker.topics import APPROVED_TOPIC_MODEL_DIGEST


def _payload_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_disabled_outcome_provider_has_exact_hashed_policy() -> None:
    policy = outcome_policy_provenance(None)

    assert policy["status"] == "provider_disabled"
    assert policy["provider"] is None
    assert policy["model"] is None
    assert policy["approved_model_digest"] == APPROVED_TOPIC_MODEL_DIGEST
    assert policy["remote_allow_authorizes_outcome_calls"] is True
    assert policy["policy_hash"] == _payload_hash(
        {key: value for key, value in policy.items() if key != "policy_hash"}
    )


def test_budget_controller_persists_and_reloads_exact_journal(tmp_path: Path) -> None:
    path = tmp_path / "outcome_budget_journal.json"
    job_hash = "a" * 64
    expected_tasks = ("b" * 64,)

    controller = budget_controller(
        path,
        job_hash=job_hash,
        expected_task_hashes=expected_tasks,
    )
    controller.reserve_task(expected_tasks[0])
    controller.reconcile_result(expected_tasks[0])
    reloaded = read_budget_journal(
        path,
        job_hash=job_hash,
        expected_task_hashes=expected_tasks,
    )

    assert reloaded == controller.journal
    assert reloaded.result_task_hashes == expected_tasks

    path.write_text('{"journal":{}}', encoding="utf-8")
    with pytest.raises(OutcomePartitionError, match="budget_journal_tamper"):
        read_budget_journal(
            path,
            job_hash=job_hash,
            expected_task_hashes=expected_tasks,
        )


def test_outcome_decoding_and_vote_sanitization_remain_closed() -> None:
    decoded = decode_outcome_sets(
        [
            {
                "topic_id": "topic-001",
                "candidates": [],
                "extraction_method": "deterministic_fallback",
                "failure_types": ["provider_disabled"],
            }
        ]
    )

    assert decoded[0].topic_id == "topic-001"
    assert sanitized_vote_result({"result": "voted down"}) == "rejected"
    assert sanitized_vote_result({"result": "unexpected"}) == "unknown"
    assert sanitized_vote_result({"approved": True}) == "approved"
    with pytest.raises(ValueError, match="topic outcome sets are malformed"):
        decode_outcome_sets({"topic_id": "topic-001"})
