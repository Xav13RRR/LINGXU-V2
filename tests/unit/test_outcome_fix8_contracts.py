import hashlib
import json

import pytest

from lingxu.transcript.models import CanonicalTranscriptEntry
from lingxu.worker.outcome_partitions import (
    MAX_TOTAL_LLM_HTTP_REQUESTS_PER_JOB,
    MAX_TOTAL_LLM_INFERENCE_CALLS_PER_JOB,
    LlmActionKind,
    OutcomePartitionError,
    OutcomePartitionResult,
    UnifiedLlmBudgetJournal,
    build_outcome_partition_manifest,
    plan_outcome_partitions,
    serialize_outcome_request,
    validate_outcome_partition_manifest,
)
from lingxu.worker.outcomes import (
    ModelOutcomeCandidate,
    OutcomeEventFact,
    OutcomeRequestKind,
    OutcomeType,
    TopicOutcomeSet,
    materialise_outcome_candidate,
)
from lingxu.worker.topics import TopicSegment


def _entry(index: int, text: str = "bounded evidence") -> CanonicalTranscriptEntry:
    return CanonicalTranscriptEntry(
        entry_id=f"entry-{index:03d}",
        source_entry_ids=(f"source-{index:03d}",),
        speaker_user_id=7,
        speaker_name="participant",
        started_ms=index * 1_000,
        ended_ms=index * 1_000 + 800,
        raw_texts=(text,),
        text=text,
        entry_type="statement",
        language="en",
        confidence=0.95,
        utterance_group_id=f"group-{index:03d}",
        normalisation_changes=(),
        asr_tags=(),
        emotion=None,
    )


def _topic(entries: tuple[CanonicalTranscriptEntry, ...]) -> TopicSegment:
    return TopicSegment(
        id="topic-001",
        title="Topic",
        entry_ids=tuple(entry.entry_id for entry in entries),
        started_ms=0,
        ended_ms=max(entry.ended_ms for entry in entries),
        method="deterministic_fallback",
    )


def _manifest(
    entries: tuple[CanonicalTranscriptEntry, ...],
    events: tuple[OutcomeEventFact, ...] = (),
):
    topic = _topic(entries)
    return build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((topic, entries, events),),
        pipeline_revision="e" * 64,
    )


def test_event_source_closure_includes_every_fragment_and_request_bytes_are_bound() -> None:
    entry = _entry(1, "x" * 3_900)
    event = OutcomeEventFact(
        sequence_number=1,
        topic_id="topic-001",
        subject="vote",
        result="passed",
        source_entry_ids=(entry.entry_id,),
    )
    manifest = _manifest((entry,), (event,))
    event_partition = next(part for part in manifest.partitions if part.core_events)
    visible = (*event_partition.core_entries, *event_partition.overlap_entries)
    expected_fragments = {
        (item.entry_id, item.fragment_index)
        for part in manifest.partitions
        for item in part.core_entries
        if item.entry_id == entry.entry_id
    }
    assert {(item.entry_id, item.fragment_index) for item in visible} == expected_fragments
    for kind in OutcomeRequestKind:
        payload = serialize_outcome_request(event_partition, kind)
        assert payload == event_partition.request_bytes(kind)
        assert manifest.request_hashes[event_partition.partition_id][kind.value] == (
            __import__("hashlib").sha256(payload).hexdigest()
        )


def test_manifest_validation_rebuilds_authoritative_inventory_and_rejects_tamper() -> None:
    entries = (_entry(1), _entry(2))
    topic = _topic(entries)
    manifest = _manifest(entries)
    validate_outcome_partition_manifest(
        manifest,
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((topic, entries, ()),),
        pipeline_revision="e" * 64,
    )
    damaged = manifest.model_copy(update={"coverage_digest": "0" * 64})
    with pytest.raises(OutcomePartitionError, match="manifest_tamper"):
        validate_outcome_partition_manifest(
            damaged,
            canonical_hash="a" * 64,
            topic_artifact_hash="b" * 64,
            event_facts_hash="c" * 64,
            participant_hash="d" * 64,
            topics=((topic, entries, ()),),
            pipeline_revision="e" * 64,
        )


def test_unified_journal_counts_actual_actions_and_never_resets_on_revision_change() -> None:
    journal = UnifiedLlmBudgetJournal.new(
        job_hash="a" * 64,
        expected_task_hashes=("b" * 64,),
        now=100.0,
    )
    reserved, timeout = journal.reserve_task("b" * 64).reserve_action(
        "c" * 64,
        task_hash="b" * 64,
        kind=LlmActionKind.OUTCOME_INITIAL,
        configured_timeout_seconds=180.0,
        now=101.0,
    )
    assert timeout == 180.0
    assert reserved.actual_http_requests == 1
    assert reserved.actual_inference_calls == 1
    with pytest.raises(OutcomePartitionError, match="duplicate_action"):
        reserved.reserve_action(
            "c" * 64,
            task_hash="b" * 64,
            kind=LlmActionKind.OUTCOME_INITIAL,
            configured_timeout_seconds=180.0,
            now=102.0,
        )
    with pytest.raises(OutcomePartitionError, match="job_revision_mismatch"):
        reserved.validate_job("d" * 64)


def test_unified_journal_zero_remaining_wall_makes_no_reservation() -> None:
    journal = UnifiedLlmBudgetJournal.new(
        job_hash="a" * 64,
        expected_task_hashes=("b" * 64,),
        now=0.0,
    )
    with pytest.raises(OutcomePartitionError, match="wall_budget"):
        journal.reserve_action(
            "c" * 64,
            task_hash="b" * 64,
            kind=LlmActionKind.OUTCOME_INITIAL,
            configured_timeout_seconds=180.0,
            now=5_400.0,
        )
    assert journal.actual_http_requests == 0
    assert journal.actual_inference_calls == 0


def test_result_cartesian_invariants_and_crash_reconciliation() -> None:
    journal = UnifiedLlmBudgetJournal.new(
        job_hash="a" * 64,
        expected_task_hashes=("b" * 64,),
        now=100.0,
    ).reserve_task("b" * 64)
    reconciled = journal.reconcile_result("b" * 64)
    assert reconciled.completed_task_hashes == ("b" * 64,)
    assert reconciled.result_task_hashes == ("b" * 64,)
    assert reconciled.actual_http_requests == 0
    reconciled.validate_final()
    with pytest.raises(OutcomePartitionError, match="unknown_task"):
        reconciled.reserve_task("f" * 64)


def test_partition_result_has_independent_twenty_candidate_capacity() -> None:
    entries = (_entry(1),)
    partition = _manifest(entries).partitions[0]
    candidates = tuple(
        materialise_outcome_candidate(
            topic_id=partition.topic_id,
            kind=OutcomeRequestKind.ACTION,
            partition_id=partition.partition_id,
            batch_ordinal=index,
            candidate=ModelOutcomeCandidate(
                type=OutcomeType.ACTION_ITEM,
                text=f"action-{index}",
                owner_ref=None,
                owner_name=None,
                deadline=None,
                source_entry_refs=("e0",),
                source_event_refs=(),
                confidence=0.9,
            ),
            owner_user_id=None,
            source_entry_ids=(entries[0].entry_id,),
            source_event_ids=(),
        )
        for index in range(20)
    )
    result = OutcomePartitionResult.terminal(
        partition=partition,
        kind=OutcomeRequestKind.ACTION,
        result=TopicOutcomeSet(
            topic_id=partition.topic_id,
            candidates=candidates,
            extraction_method="local_llm",
            failure_types=(),
        ),
        pipeline_revision="e" * 64,
    )
    assert len(result.candidates) == 20
    assert result.wire_input_row_count == 20
    assert len(result.wire_row_resolutions) == 20
    assert len(result.as_topic_outcome_set().candidates) == 20
    assert not hasattr(result, "result")


def test_fragment_refs_map_back_to_canonical_entry_and_noncore_is_rejected() -> None:
    entries = (_entry(1, "x" * 9_000),)
    partitions = plan_outcome_partitions(_topic(entries), entries, ())
    first, second = partitions[0], partitions[-1]
    payload = json.loads(serialize_outcome_request(second, OutcomeRequestKind.ACTION))
    core_ref = payload["c"][0][0]
    assert second.resolve_wire_entry_refs((core_ref,), require_core=True) == (entries[0].entry_id,)
    assert json.loads(serialize_outcome_request(first, OutcomeRequestKind.ACTION))["c"]
    with pytest.raises(OutcomePartitionError, match="noncore_evidence"):
        second.resolve_wire_entry_refs(("e999",), require_core=True)


def test_unified_ledger_enforces_exact_http_inference_and_wall_caps() -> None:
    task_hash = "b" * 64
    journal = UnifiedLlmBudgetJournal.new(
        job_hash="a" * 64,
        expected_task_hashes=(task_hash,),
        now=100.0,
    ).reserve_task(task_hash)
    for index in range(MAX_TOTAL_LLM_INFERENCE_CALLS_PER_JOB):
        action_hash = hashlib.sha256(f"inference-{index}".encode()).hexdigest()
        journal, _ = journal.reserve_action(
            action_hash,
            task_hash=task_hash,
            kind=LlmActionKind.OUTCOME_INITIAL,
            configured_timeout_seconds=180.0,
            now=101.0,
        )
    assert journal.actual_inference_calls == MAX_TOTAL_LLM_INFERENCE_CALLS_PER_JOB
    with pytest.raises(OutcomePartitionError, match="inference_budget"):
        journal.reserve_action(
            hashlib.sha256(b"inference-overflow").hexdigest(),
            task_hash=task_hash,
            kind=LlmActionKind.OUTCOME_INITIAL,
            configured_timeout_seconds=180.0,
            now=101.0,
        )
    for index in range(MAX_TOTAL_LLM_HTTP_REQUESTS_PER_JOB - MAX_TOTAL_LLM_INFERENCE_CALLS_PER_JOB):
        action_hash = hashlib.sha256(f"digest-{index}".encode()).hexdigest()
        journal, _ = journal.reserve_action(
            action_hash,
            task_hash=task_hash,
            kind=LlmActionKind.DIGEST,
            configured_timeout_seconds=180.0,
            now=101.0,
        )
    assert journal.actual_http_requests == MAX_TOTAL_LLM_HTTP_REQUESTS_PER_JOB
    with pytest.raises(OutcomePartitionError, match="http_budget"):
        journal.reserve_action(
            hashlib.sha256(b"http-overflow").hexdigest(),
            task_hash=task_hash,
            kind=LlmActionKind.DIGEST,
            configured_timeout_seconds=180.0,
            now=101.0,
        )

    fresh = UnifiedLlmBudgetJournal.new(
        job_hash="a" * 64,
        expected_task_hashes=(task_hash,),
        now=100.0,
    ).reserve_task(task_hash)
    _, timeout = fresh.reserve_action(
        "c" * 64,
        task_hash=task_hash,
        kind=LlmActionKind.DIGEST,
        configured_timeout_seconds=180.0,
        now=5_400.0,
    )
    assert timeout == 100.0


def test_unified_ledger_final_state_rejects_each_incomplete_crash_window() -> None:
    task_hash = "b" * 64
    reserved = UnifiedLlmBudgetJournal.new(
        job_hash="a" * 64,
        expected_task_hashes=(task_hash,),
        now=100.0,
    ).reserve_task(task_hash)
    with pytest.raises(OutcomePartitionError, match="budget_final_incomplete"):
        reserved.validate_final()
    completed_without_result = reserved.complete_task(task_hash)
    with pytest.raises(OutcomePartitionError, match="budget_final_incomplete"):
        completed_without_result.validate_final()
    reconciled = reserved.reconcile_result(task_hash)
    reconciled.validate_final()
