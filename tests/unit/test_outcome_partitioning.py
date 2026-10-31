import hashlib
import json
from dataclasses import replace

import pytest

import lingxu.worker.outcome_partitions as outcome_partitions_module
from lingxu.transcript.models import CanonicalTranscriptEntry
from lingxu.worker.outcome_partitions import (
    MAX_DEEPSEEK_REPAIR_OUTPUT_TOKENS,
    MAX_LLM_WALL_SECONDS_PER_JOB,
    MAX_OUTCOME_OUTPUT_TOKENS,
    MAX_OUTCOME_PARTITIONS_PER_JOB,
    MAX_OUTCOME_PARTITIONS_PER_TOPIC,
    MAX_TOPIC_BOUNDARY_CALLS_PER_JOB,
    MAX_TOTAL_LLM_HTTP_REQUESTS_PER_JOB,
    MAX_TOTAL_LLM_INFERENCE_CALLS_PER_JOB,
    LlmActionKind,
    OutcomePartitionError,
    OutcomePartitionManifest,
    OutcomePartitionResult,
    OutcomeWireRowResolution,
    UnifiedLlmBudgetJournal,
    build_outcome_partition_manifest,
    llm_action_hash,
    migrate_fix33_outcome_partition_result,
    outcome_preflight_capacity_failure,
    outcome_task_hash,
    plan_outcome_partitions,
    summarize_outcome_dispositions,
    validate_outcome_partition_manifest,
)
from lingxu.worker.outcomes import (
    OutcomeCapacitySubtype,
    OutcomeEventFact,
    OutcomeFailureAttempt,
    OutcomeFailureCategory,
    OutcomeFailureRequestKind,
    OutcomeFormatSubtype,
    OutcomeProviderFailure,
    OutcomeRequestKind,
    OutcomeSizeBucket,
    OutcomeTaskDisposition,
    OutcomeType,
    TopicOutcomeSet,
    VerificationStatus,
)
from lingxu.worker.topics import TopicSegment


def _entry(index: int, text: str = "short evidence") -> CanonicalTranscriptEntry:
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
        title="Topic 1",
        entry_ids=tuple(entry.entry_id for entry in entries),
        started_ms=0,
        ended_ms=max(entry.ended_ms for entry in entries),
        method="deterministic_fallback",
    )


def _event(index: int, entry_id: str) -> OutcomeEventFact:
    return OutcomeEventFact(
        sequence_number=index + 1,
        topic_id="topic-001",
        subject=f"vote-{index:03d}",
        result="passed",
        source_entry_ids=(entry_id,),
    )


def _event_for_sources(*entry_ids: str) -> OutcomeEventFact:
    return OutcomeEventFact(
        sequence_number=1,
        topic_id="topic-001",
        subject="accepted",
        result="passed",
        source_entry_ids=entry_ids,
    )


def test_fix34_terminal_disposition_matrix_accepts_valid_empty_repair_success() -> None:
    entries = (_entry(1),)
    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((_topic(entries), entries, ()),),
        pipeline_revision="e" * 64,
    )
    partition = manifest.partitions[0]
    initial_failure = OutcomeProviderFailure(
        category=OutcomeFailureCategory.JSON,
        request_kind=OutcomeFailureRequestKind.QUESTION,
        attempt=OutcomeFailureAttempt.INITIAL,
        count=1,
        size_bucket=OutcomeSizeBucket.LTE_1K,
        capacity_subtype=None,
        format_subtype=OutcomeFormatSubtype.JSON_SYNTAX_OTHER,
    )

    repaired = OutcomePartitionResult.terminal(
        partition=partition,
        kind=OutcomeRequestKind.QUESTION,
        result=TopicOutcomeSet(
            topic_id=partition.topic_id,
            candidates=(),
            extraction_method="local_llm_partial",
            failure_types=(),
            provider_failures=(initial_failure,),
            task_disposition=OutcomeTaskDisposition.STRICT_REPAIR_SUCCESS,
        ),
        pipeline_revision="e" * 64,
    )

    assert repaired.task_disposition is OutcomeTaskDisposition.STRICT_REPAIR_SUCCESS
    assert repaired.candidates == ()
    task_hash = outcome_task_hash("f" * 64, partition, OutcomeRequestKind.QUESTION)
    journal = UnifiedLlmBudgetJournal.new(
        job_hash="f" * 64,
        expected_task_hashes=(task_hash,),
        now=0,
    ).reserve_task(task_hash)
    for action_kind in (LlmActionKind.OUTCOME_INITIAL, LlmActionKind.OUTCOME_REPAIR):
        action_hash = llm_action_hash(task_hash, action_kind)
        journal, _ = journal.reserve_action(
            action_hash,
            task_hash=task_hash,
            kind=action_kind,
            configured_timeout_seconds=1,
            now=1,
        )
        journal = journal.complete_action(action_hash)
    journal = journal.reconcile_result(task_hash)
    metrics = summarize_outcome_dispositions(
        {(partition.partition_id, OutcomeRequestKind.QUESTION): repaired},
        journal,
        manifest,
    )
    assert metrics.recovered_initial_incidents == 1
    assert metrics.recovered_task_count == 1
    assert metrics.terminal_provider_failure_incidents == 0
    assert metrics.terminal_failure_task_count == 0

    legacy = repaired.model_dump(mode="json")
    for field in ("schema_version", "disposition_revision", "task_disposition"):
        legacy.pop(field)
    legacy["result_hash"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in legacy.items() if key != "result_hash"},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    migrated = migrate_fix33_outcome_partition_result(
        legacy,
        partition=partition,
        journal=journal,
        pipeline_revision="e" * 64,
    )
    assert migrated.task_disposition is OutcomeTaskDisposition.STRICT_REPAIR_SUCCESS
    assert migrated.candidates == ()
    assert migrated.provider_failures == repaired.provider_failures
    assert migrated.failure_types == repaired.failure_types

    missing_actions = journal.model_copy(
        update={
            "actions": (),
            "actual_http_requests": 0,
            "actual_inference_calls": 0,
        }
    )
    with pytest.raises(OutcomePartitionError, match="outcome_disposition_migration_required"):
        migrate_fix33_outcome_partition_result(
            legacy,
            partition=partition,
            journal=missing_actions,
            pipeline_revision="e" * 64,
        )
    missing_repair = journal.model_copy(
        update={
            "actions": journal.actions[:1],
            "actual_http_requests": 1,
            "actual_inference_calls": 1,
        }
    )
    with pytest.raises(OutcomePartitionError, match="outcome_disposition_invalid"):
        summarize_outcome_dispositions(
            {(partition.partition_id, OutcomeRequestKind.QUESTION): repaired},
            missing_repair,
            manifest,
        )
    forged_action_hash = journal.model_copy(
        update={
            "actions": (
                journal.actions[0].model_copy(update={"action_hash": "0" * 64}),
                journal.actions[1],
            )
        }
    )
    with pytest.raises(OutcomePartitionError, match="outcome_disposition_invalid"):
        summarize_outcome_dispositions(
            {(partition.partition_id, OutcomeRequestKind.QUESTION): repaired},
            forged_action_hash,
            manifest,
        )
    payload = repaired.model_dump()
    with pytest.raises(ValueError, match="outcome task disposition"):
        OutcomePartitionResult.model_validate(
            {**payload, "task_disposition": OutcomeTaskDisposition.STRICT_INITIAL_SUCCESS}
        )


def test_fix39_second_repair_success_requires_two_exact_repair_action_ordinals() -> None:
    entries = (_entry(1),)
    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((_topic(entries), entries, ()),),
        pipeline_revision="e" * 64,
    )
    partition = manifest.partitions[0]
    failures = tuple(
        OutcomeProviderFailure(
            category=OutcomeFailureCategory.OUTPUT_CAPACITY,
            request_kind=OutcomeFailureRequestKind.QUESTION,
            attempt=attempt,
            count=1,
            size_bucket=OutcomeSizeBucket.LTE_1K,
            capacity_subtype=OutcomeCapacitySubtype.COMPLETE_FALSE,
        )
        for attempt in (OutcomeFailureAttempt.INITIAL, OutcomeFailureAttempt.REPAIR)
    )
    result = OutcomePartitionResult.terminal(
        partition=partition,
        kind=OutcomeRequestKind.QUESTION,
        result=TopicOutcomeSet(
            topic_id=partition.topic_id,
            candidates=(),
            extraction_method="local_llm_partial",
            failure_types=(),
            provider_failures=failures,
            task_disposition=OutcomeTaskDisposition.STRICT_REPAIR_SUCCESS,
        ),
        pipeline_revision="e" * 64,
    )
    task_hash = outcome_task_hash("f" * 64, partition, OutcomeRequestKind.QUESTION)
    journal = UnifiedLlmBudgetJournal.new(
        job_hash="f" * 64,
        expected_task_hashes=(task_hash,),
        now=0,
    ).reserve_task(task_hash)
    action_kinds = (
        LlmActionKind.OUTCOME_INITIAL,
        LlmActionKind.OUTCOME_REPAIR,
        LlmActionKind.OUTCOME_REPAIR,
    )
    for index, kind in enumerate(action_kinds):
        ordinal = 0 if index < 2 else 1
        action_hash = llm_action_hash(task_hash, kind, ordinal)
        journal, _ = journal.reserve_action(
            action_hash,
            task_hash=task_hash,
            kind=kind,
            configured_timeout_seconds=1,
            now=1,
        )
        journal = journal.complete_action(action_hash)
    journal = journal.reconcile_result(task_hash)

    metrics = summarize_outcome_dispositions(
        {(partition.partition_id, OutcomeRequestKind.QUESTION): result},
        journal,
        manifest,
    )
    assert metrics.recovered_initial_incidents == 2
    assert metrics.recovered_task_count == 1
    assert metrics.terminal_provider_failure_incidents == 0

    forged = journal.model_copy(
        update={
            "actions": (
                *journal.actions[:2],
                journal.actions[2].model_copy(
                    update={"action_hash": llm_action_hash(task_hash, LlmActionKind.OUTCOME_REPAIR)}
                ),
            )
        }
    )
    with pytest.raises(OutcomePartitionError, match="outcome_disposition_invalid"):
        summarize_outcome_dispositions(
            {(partition.partition_id, OutcomeRequestKind.QUESTION): result},
            forged,
            manifest,
        )


def _fix34_disposition_case(
    *,
    failures: tuple[OutcomeProviderFailure, ...],
    action_failures: tuple[OutcomeFailureCategory | None, ...],
) -> tuple[OutcomePartitionManifest, OutcomePartitionResult, UnifiedLlmBudgetJournal]:
    entries = (_entry(1),)
    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((_topic(entries), entries, ()),),
        pipeline_revision="e" * 64,
    )
    partition = manifest.partitions[0]
    result = OutcomePartitionResult.terminal(
        partition=partition,
        kind=OutcomeRequestKind.QUESTION,
        result=TopicOutcomeSet(
            topic_id=partition.topic_id,
            candidates=(),
            extraction_method="local_llm_partial",
            failure_types=(),
            provider_failures=failures,
            task_disposition=OutcomeTaskDisposition.TERMINAL_FAILURE,
        ),
        pipeline_revision="e" * 64,
    )
    task_hash = outcome_task_hash("f" * 64, partition, OutcomeRequestKind.QUESTION)
    journal = UnifiedLlmBudgetJournal.new(
        job_hash="f" * 64,
        expected_task_hashes=(task_hash,),
        now=0,
    ).reserve_task(task_hash)
    repair_ordinal = 0
    for index, failure_category in enumerate(action_failures):
        action_kind = LlmActionKind.OUTCOME_INITIAL if index == 0 else LlmActionKind.OUTCOME_REPAIR
        action_hash = llm_action_hash(task_hash, action_kind, repair_ordinal)
        if action_kind is LlmActionKind.OUTCOME_REPAIR:
            repair_ordinal += 1
        journal, _ = journal.reserve_action(
            action_hash,
            task_hash=task_hash,
            kind=action_kind,
            configured_timeout_seconds=1,
            now=1,
        )
        journal = journal.complete_action(action_hash, failure_category=failure_category)
    journal = journal.reconcile_result(task_hash)
    return manifest, result, journal


def _fix34_failure(
    category: OutcomeFailureCategory,
    attempt: OutcomeFailureAttempt,
    *,
    format_subtype: OutcomeFormatSubtype | None = None,
) -> OutcomeProviderFailure:
    return OutcomeProviderFailure(
        category=category,
        request_kind=OutcomeFailureRequestKind.QUESTION,
        attempt=attempt,
        count=1,
        size_bucket=OutcomeSizeBucket.LTE_1K,
        capacity_subtype=None,
        format_subtype=format_subtype,
    )


@pytest.mark.parametrize(
    ("failures", "action_failures"),
    (
        (
            (
                _fix34_failure(
                    OutcomeFailureCategory.JSON,
                    OutcomeFailureAttempt.INITIAL,
                    format_subtype=OutcomeFormatSubtype.JSON_DECODE,
                ),
                _fix34_failure(
                    OutcomeFailureCategory.SCHEMA,
                    OutcomeFailureAttempt.INITIAL,
                    format_subtype=OutcomeFormatSubtype.SCHEMA_ROOT,
                ),
            ),
            (None, None),
        ),
        (
            (
                _fix34_failure(
                    OutcomeFailureCategory.SCHEMA,
                    OutcomeFailureAttempt.REPAIR,
                    format_subtype=OutcomeFormatSubtype.SCHEMA_ROOT,
                ),
            ),
            (None, None),
        ),
        (
            (
                _fix34_failure(
                    OutcomeFailureCategory.TRANSPORT_ERROR,
                    OutcomeFailureAttempt.INITIAL,
                ),
            ),
            (None,),
        ),
    ),
    ids=("multiple-initial-facts", "repair-only-fact", "unrecorded-transport-failure"),
)
def test_fix34_disposition_matrix_rejects_facts_not_proven_by_journal(
    failures: tuple[OutcomeProviderFailure, ...],
    action_failures: tuple[OutcomeFailureCategory | None, ...],
) -> None:
    manifest, result, journal = _fix34_disposition_case(
        failures=failures,
        action_failures=action_failures,
    )

    with pytest.raises(OutcomePartitionError, match="outcome_disposition_invalid"):
        summarize_outcome_dispositions(
            {(manifest.partitions[0].partition_id, OutcomeRequestKind.QUESTION): result},
            journal,
            manifest,
        )


@pytest.mark.parametrize(
    ("failures", "action_failures"),
    (
        (
            (
                _fix34_failure(
                    OutcomeFailureCategory.JSON,
                    OutcomeFailureAttempt.INITIAL,
                    format_subtype=OutcomeFormatSubtype.JSON_DECODE,
                ),
                _fix34_failure(
                    OutcomeFailureCategory.INPUT_LIMIT,
                    OutcomeFailureAttempt.PREFLIGHT,
                ),
            ),
            (None,),
        ),
        (
            (
                _fix34_failure(
                    OutcomeFailureCategory.TRANSPORT_ERROR,
                    OutcomeFailureAttempt.INITIAL,
                ),
            ),
            (OutcomeFailureCategory.TRANSPORT_ERROR,),
        ),
        (
            (
                _fix34_failure(
                    OutcomeFailureCategory.SCHEMA,
                    OutcomeFailureAttempt.INITIAL,
                    format_subtype=OutcomeFormatSubtype.DUPLICATE_BINDING_WIRE_DIVERGENT,
                ),
                _fix34_failure(
                    OutcomeFailureCategory.SCHEMA,
                    OutcomeFailureAttempt.REPAIR,
                    format_subtype=OutcomeFormatSubtype.DUPLICATE_BINDING_WIRE_DIVERGENT,
                ),
                _fix34_failure(
                    OutcomeFailureCategory.SCHEMA,
                    OutcomeFailureAttempt.REPAIR,
                    format_subtype=OutcomeFormatSubtype.DUPLICATE_BINDING_WIRE_DIVERGENT,
                ),
            ),
            (None, None, None),
        ),
        (
            (
                _fix34_failure(
                    OutcomeFailureCategory.JSON,
                    OutcomeFailureAttempt.INITIAL,
                    format_subtype=OutcomeFormatSubtype.JSON_DECODE,
                ),
                _fix34_failure(
                    OutcomeFailureCategory.TRANSPORT_TIMEOUT,
                    OutcomeFailureAttempt.REPAIR,
                ),
            ),
            (None, OutcomeFailureCategory.TRANSPORT_TIMEOUT),
        ),
        (
            (
                _fix34_failure(
                    OutcomeFailureCategory.JSON,
                    OutcomeFailureAttempt.INITIAL,
                    format_subtype=OutcomeFormatSubtype.JSON_DECODE,
                ),
                _fix34_failure(
                    OutcomeFailureCategory.SCHEMA,
                    OutcomeFailureAttempt.REPAIR,
                    format_subtype=OutcomeFormatSubtype.SCHEMA_ROOT,
                ),
                _fix34_failure(
                    OutcomeFailureCategory.SCHEMA,
                    OutcomeFailureAttempt.REPAIR,
                    format_subtype=OutcomeFormatSubtype.SCHEMA_ROOT,
                ),
            ),
            (None, None, None),
        ),
    ),
    ids=(
        "repair-input-preflight",
        "initial-transport-boundary",
        "duplicate-binding-exhausted",
        "repair-transport-boundary",
        "second-repair-parsed-schema",
    ),
)
def test_fix34_disposition_matrix_accepts_exact_terminal_authority(
    failures: tuple[OutcomeProviderFailure, ...],
    action_failures: tuple[OutcomeFailureCategory | None, ...],
) -> None:
    manifest, result, journal = _fix34_disposition_case(
        failures=failures,
        action_failures=action_failures,
    )

    metrics = summarize_outcome_dispositions(
        {(manifest.partitions[0].partition_id, OutcomeRequestKind.QUESTION): result},
        journal,
        manifest,
    )

    assert metrics.recovered_initial_incidents == 0
    assert metrics.terminal_provider_failure_incidents == len(failures)
    assert metrics.terminal_failure_task_count == 1


def test_partition_plan_preserves_more_than_eighty_entries_in_order_exactly_once() -> None:
    entries = tuple(_entry(index) for index in range(100))
    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((_topic(entries), entries, ()),),
    )

    core = [fragment for partition in manifest.partitions for fragment in partition.core_entries]
    assert [fragment.entry_id for fragment in core] == [entry.entry_id for entry in entries]
    assert all(fragment.start_offset == 0 for fragment in core)
    assert all(
        fragment.end_offset == len(entry.text)
        for fragment, entry in zip(core, entries, strict=True)
    )
    assert len(manifest.partitions) <= MAX_OUTCOME_PARTITIONS_PER_TOPIC
    for partition in manifest.partitions:
        for kind in OutcomeRequestKind:
            assert len(partition.context_json(kind)) <= 6_000


def test_partition_plan_preserves_sixteen_events_and_source_closure() -> None:
    entries = tuple(_entry(index) for index in range(20))
    events = tuple(_event(index, entries[index].entry_id) for index in range(16))
    partitions = plan_outcome_partitions(_topic(entries), entries, events)

    assert [event.sequence_number for part in partitions for event in part.core_events] == list(
        range(1, 17)
    )
    for partition in partitions:
        visible_ids = {
            fragment.entry_id for fragment in (*partition.core_entries, *partition.overlap_entries)
        }
        assert all(set(event.source_entry_ids) <= visible_ids for event in partition.core_events)


def test_giant_entry_fragments_have_lossless_codepoint_coverage() -> None:
    text = ("alpha beta gamma。" * 900) + "END"
    entry = _entry(0, text)
    partitions = plan_outcome_partitions(_topic((entry,)), (entry,), ())

    fragments = [fragment for partition in partitions for fragment in partition.core_entries]
    assert "".join(fragment.text for fragment in fragments) == text
    assert [fragment.start_offset for fragment in fragments] == [
        0,
        *[fragment.end_offset for fragment in fragments[:-1]],
    ]
    assert all(
        len(partition.context_json(kind)) <= 6_000
        for partition in partitions
        for kind in OutcomeRequestKind
    )


def test_adaptive_smaller_context_target_is_lossless_and_stays_within_hard_caps() -> None:
    """RED: reviewed dry planning may lower its target without changing the 6,000 hard cap."""

    target_context_chars = 4_000
    per_topic_counts: list[int] = []
    for topic_index in range(10):
        text = f"topic-{topic_index}:" + ("会" * 9_000)
        entry = replace(_entry(topic_index, text), entry_id=f"adaptive-{topic_index:02d}")
        topic = replace(
            _topic((entry,)),
            id=f"adaptive-topic-{topic_index:02d}",
            entry_ids=(entry.entry_id,),
        )
        planned = plan_outcome_partitions(
            topic,
            (entry,),
            (),
            target_context_chars=target_context_chars,
        )
        per_topic_counts.append(len(planned))
        assert (
            "".join(fragment.text for partition in planned for fragment in partition.core_entries)
            == text
        )
        assert all(
            len(partition.context_json(kind)) <= target_context_chars <= 6_000
            for partition in planned
            for kind in OutcomeRequestKind
        )

    assert max(per_topic_counts) <= MAX_OUTCOME_PARTITIONS_PER_TOPIC
    assert sum(per_topic_counts) <= MAX_OUTCOME_PARTITIONS_PER_JOB


def test_fix38_twelve_partition_ceiling_keeps_dense_topic_at_smallest_target() -> None:
    entry = _entry(0, "x" * 18_000)
    topic = _topic((entry,))

    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((topic, (entry,), ()),),
    )

    assert MAX_OUTCOME_PARTITIONS_PER_TOPIC == 12
    assert manifest.context_target_chars == 2_000
    assert 9 <= len(manifest.partitions) <= MAX_OUTCOME_PARTITIONS_PER_TOPIC
    assert manifest.target_attempts[0].result.value == "selected"
    assert (
        "".join(
            fragment.text
            for partition in manifest.partitions
            for fragment in partition.core_entries
        )
        == entry.text
    )


def test_fix25_manifest_selects_smallest_whole_meeting_target_and_binds_attempts() -> None:
    entries = (_entry(0, "small complete inventory"),)
    topics = ((_topic(entries), entries, ()),)

    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=topics,
    )

    assert manifest.context_target_chars == 2_000
    assert manifest.target_ladder == (2_000, 2_500, 3_000, 4_000)
    assert (
        manifest.target_ladder_revision == outcome_partitions_module.OUTCOME_TARGET_LADDER_REVISION
    )
    assert [attempt.model_dump(mode="json") for attempt in manifest.target_attempts] == [
        {
            "target_context_chars": 2_000,
            "result": "selected",
            "reason_counts": {},
            "search_states_used": 6,
        }
    ]
    assert manifest.planning_search_states_used == 6
    validate_outcome_partition_manifest(
        manifest,
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=topics,
        pipeline_revision=manifest.pipeline_revision,
    )


def test_fix38_ladder_selects_3000_after_two_closed_fragment_capacity_rejections() -> None:
    entry = _entry(0, "x" * 27_000)
    topic = _topic((entry,))

    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((topic, (entry,), ()),),
    )

    assert manifest.context_target_chars == 3_000
    assert len(manifest.partitions) == 11
    assert [attempt.target_context_chars for attempt in manifest.target_attempts] == [
        2_000,
        2_500,
        3_000,
    ]
    assert [attempt.result for attempt in manifest.target_attempts] == [
        "rejected",
        "rejected",
        "selected",
    ]
    assert [attempt.reason_counts for attempt in manifest.target_attempts] == [
        {"fragment_capacity": 1},
        {"fragment_capacity": 1},
        {},
    ]


def test_fix25_whole_meeting_ladder_selects_first_plan_within_job_cap() -> None:
    topic_rows = []
    for topic_index in range(8):
        entry = replace(
            _entry(topic_index, f"topic-{topic_index}:" + ("x" * 9_000)),
            entry_id=f"ladder-entry-{topic_index:02d}",
        )
        topic = replace(
            _topic((entry,)),
            id=f"ladder-topic-{topic_index:02d}",
            entry_ids=(entry.entry_id,),
        )
        topic_rows.append((topic, (entry,), ()))

    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=tuple(topic_rows),
    )

    assert manifest.context_target_chars == 2_500
    assert len(manifest.partitions) == MAX_OUTCOME_PARTITIONS_PER_JOB
    assert (
        max(
            sum(partition.topic_id == topic.id for partition in manifest.partitions)
            for topic, _, _ in topic_rows
        )
        == 5
    )
    assert [attempt.model_dump(mode="json") for attempt in manifest.target_attempts] == [
        {
            "target_context_chars": 2_000,
            "result": "rejected",
            "reason_counts": {"job_partition_cap": 1},
            "search_states_used": manifest.target_attempts[0].search_states_used,
        },
        {
            "target_context_chars": 2_500,
            "result": "selected",
            "reason_counts": {},
            "search_states_used": manifest.target_attempts[1].search_states_used,
        },
    ]
    assert manifest.planning_search_states_used == sum(
        attempt.search_states_used for attempt in manifest.target_attempts
    )
    assert manifest.planning_search_states_used <= manifest.planning_search_state_limit


def test_fix25_shared_ladder_search_exhaustion_fails_without_trying_larger_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(outcome_partitions_module, "MAX_EVENT_SEARCH_STATES", 0)
    entry = _entry(0, "votes")
    event = _event_for_sources(entry.entry_id)

    with pytest.raises(OutcomePartitionError, match="planning_search_capacity"):
        build_outcome_partition_manifest(
            canonical_hash="a" * 64,
            topic_artifact_hash="b" * 64,
            event_facts_hash="c" * 64,
            participant_hash="d" * 64,
            topics=((_topic((entry,)), (entry,), (event,)),),
        )


def test_fix25_ladder_metadata_tamper_and_legacy_fixed_target_manifest_fail_closed() -> None:
    entries = (_entry(0),)
    topics = ((_topic(entries), entries, ()),)
    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=topics,
    )

    tampered = manifest.model_copy(update={"context_target_chars": 4_000})
    with pytest.raises(OutcomePartitionError, match="manifest_tamper"):
        validate_outcome_partition_manifest(tampered)

    changed_attempt = manifest.target_attempts[0].model_copy(
        update={"search_states_used": manifest.target_attempts[0].search_states_used + 1}
    )
    self_consistent_tamper = manifest.model_copy(
        update={
            "target_attempts": (changed_attempt,),
            "planning_search_states_used": manifest.planning_search_states_used + 1,
        }
    )
    self_consistent_tamper = self_consistent_tamper.model_copy(
        update={
            "manifest_hash": outcome_partitions_module._hash(self_consistent_tamper.hash_input())
        }
    )
    with pytest.raises(OutcomePartitionError, match="manifest_tamper"):
        validate_outcome_partition_manifest(
            self_consistent_tamper,
            canonical_hash="a" * 64,
            topic_artifact_hash="b" * 64,
            event_facts_hash="c" * 64,
            participant_hash="d" * 64,
            topics=topics,
            pipeline_revision=manifest.pipeline_revision,
        )

    legacy = manifest.model_dump(mode="json")
    legacy["schema_version"] = 10
    for field in ("target_ladder", "target_ladder_revision", "target_attempts"):
        legacy.pop(field)
    with pytest.raises(ValueError):
        OutcomePartitionManifest.model_validate_json(json.dumps(legacy))


def test_event_source_closure_is_the_only_explicit_target_exception() -> None:
    entry = _entry(0, "会" * 5_000)
    event = _event(0, entry.entry_id)
    partitions = plan_outcome_partitions(_topic((entry,)), (entry,), (event,))
    exceptions = [partition for partition in partitions if partition.event_closure_exception]
    assert exceptions
    assert all(partition.core_events for partition in exceptions)
    for partition in partitions:
        maximum = max(len(partition.context_json(kind)) for kind in OutcomeRequestKind)
        if partition.event_closure_exception:
            assert 4_000 < maximum <= 6_000
        else:
            assert maximum <= 4_000

    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((_topic((entry,)), (entry,), (event,)),),
    )
    assert manifest.context_target_chars == 2_000
    assert manifest.context_hard_cap_chars == 6_000
    manifest_exceptions = [
        partition for partition in manifest.partitions if partition.event_closure_exception
    ]
    assert manifest.event_closure_exception_count == len(manifest_exceptions)
    assert manifest.event_closure_exception_reason_counts == {"event_source_closure": 1}
    for partition in manifest.partitions:
        maximum = max(len(partition.context_json(kind)) for kind in OutcomeRequestKind)
        if partition.event_closure_exception:
            assert manifest.context_target_chars < maximum <= 6_000
        else:
            assert maximum <= manifest.context_target_chars
    assert manifest.max_request_chars == max(
        len(partition.context_json(kind))
        for partition in manifest.partitions
        for kind in OutcomeRequestKind
    )


def test_event_uses_later_existing_normal_fit_before_earlier_hard_fit() -> None:
    entries = (_entry(0, "a" * 3_200), _entry(1, "b" * 500))
    event = _event_for_sources(entries[1].entry_id)

    partitions = plan_outcome_partitions(_topic(entries), entries, (event,))

    selected = next(partition for partition in partitions if partition.core_events)
    assert selected.ordinal == 1
    assert selected.event_closure_exception is False
    assert selected.event_closure_exception_reason is None
    assert max(len(selected.context_json(kind)) for kind in OutcomeRequestKind) <= 4_000


def test_event_creates_normal_event_only_partition_before_any_hard_fit() -> None:
    entries = (
        _entry(0, "a" * 3_000),
        _entry(1, "b" * 300),
        _entry(2, "c" * 3_000),
        _entry(3, "d" * 300),
    )
    event = _event_for_sources(entries[1].entry_id, entries[3].entry_id)

    partitions = plan_outcome_partitions(_topic(entries), entries, (event,))

    selected = next(partition for partition in partitions if partition.core_events)
    assert len(partitions) == 3
    assert selected.ordinal == 2
    assert selected.core_entries == ()
    assert selected.event_closure_exception is False
    assert selected.event_closure_exception_reason is None
    assert max(len(selected.context_json(kind)) for kind in OutcomeRequestKind) <= 4_000


def test_truly_unavoidable_event_closure_records_hard_exception_reason() -> None:
    entry = _entry(0, "会" * 5_000)
    event = _event_for_sources(entry.entry_id)

    partitions = plan_outcome_partitions(_topic((entry,)), (entry,), (event,))

    selected = next(partition for partition in partitions if partition.core_events)
    maximum = max(len(selected.context_json(kind)) for kind in OutcomeRequestKind)
    assert 4_000 < maximum <= 6_000
    assert selected.event_closure_exception is True
    assert selected.event_closure_exception_reason == "event_source_closure"


def test_event_closure_at_pt12_uses_smallest_existing_hard_candidate() -> None:
    entries = tuple(
        entry
        for group in range(12)
        for entry in (
            _entry(group * 2, chr(97 + group) * 3_000),
            _entry(group * 2 + 1, chr(65 + group) * 300),
        )
    )
    event = _event_for_sources(entries[1].entry_id, entries[-1].entry_id)
    base = plan_outcome_partitions(_topic(entries), entries, ())
    closure = tuple(
        fragment
        for partition in base
        for fragment in partition.core_entries
        if fragment.entry_id in event.source_entry_ids
    )
    ranked: list[tuple[int, int, int]] = []
    for partition in base:
        existing_refs = {
            fragment.fragment_ref
            for fragment in (*partition.core_entries, *partition.overlap_entries)
        }
        overlap = tuple(
            fragment for fragment in closure if fragment.fragment_ref not in existing_refs
        )
        trial = partition.model_copy(
            update={
                "overlap_entries": (*partition.overlap_entries, *overlap),
                "core_events": (event,),
            }
        )
        ranked.append(
            (
                max(len(trial.context_json(kind)) for kind in OutcomeRequestKind),
                len(overlap),
                partition.ordinal,
            )
        )
    assert all(4_000 < size <= 6_000 for size, _, _ in ranked)
    expected_ordinal = min(ranked)[2]

    partitions = plan_outcome_partitions(_topic(entries), entries, (event,))

    selected = next(partition for partition in partitions if partition.core_events)
    assert len(partitions) == MAX_OUTCOME_PARTITIONS_PER_TOPIC
    assert selected.ordinal == expected_ordinal
    assert selected.event_closure_exception_reason == "event_source_closure"


def test_three_small_events_split_across_two_partitions_without_preflight_failure() -> None:
    entry = _entry(0, "votes")
    events = tuple(
        OutcomeEventFact(
            sequence_number=index,
            topic_id="topic-001",
            subject=f"accepted-{index}",
            result="passed",
            source_entry_ids=(entry.entry_id,),
        )
        for index in (1, 3, 8)
    )

    partitions = plan_outcome_partitions(_topic((entry,)), (entry,), events)

    event_partitions = [partition for partition in partitions if partition.core_events]
    assert [len(partition.core_events) for partition in event_partitions] == [2, 1]
    assert [
        event.sequence_number for partition in event_partitions for event in partition.core_events
    ] == [1, 3, 8]
    assert all(
        not outcome_preflight_capacity_failure(
            partition,
            OutcomeRequestKind.DECISION_AGREEMENT,
        )
        for partition in event_partitions
    )


def test_late_normal_temptation_preserves_global_slots_for_twenty_four_events() -> None:
    entries = tuple(
        entry
        for group in range(12)
        for entry in (
            _entry(group * 2, chr(97 + group) * 3_000),
            _entry(group * 2 + 1, chr(65 + group) * 300),
        )
    )
    events = tuple(
        OutcomeEventFact(
            sequence_number=index,
            topic_id="topic-001",
            subject=f"accepted-{index}",
            result="passed",
            source_entry_ids=(entries[-1].entry_id,),
        )
        for index in range(1, 25)
    )

    partitions = plan_outcome_partitions(_topic(entries), entries, events)

    assert len(partitions) == MAX_OUTCOME_PARTITIONS_PER_TOPIC
    assert [len(partition.core_events) for partition in partitions] == [2] * 12
    assert [
        event.sequence_number for partition in partitions for event in partition.core_events
    ] == list(range(1, 25))


def test_bounded_search_backtracks_when_late_normal_blocks_later_large_closure() -> None:
    entries = tuple(
        entry
        for group in range(12)
        for entry in (
            _entry(group * 2, chr(97 + group) * 3_000),
            _entry(group * 2 + 1, chr(65 + group) * 300),
        )
    )
    events = (
        OutcomeEventFact(
            sequence_number=1,
            topic_id="topic-001",
            subject="first",
            result="passed",
            source_entry_ids=(entries[-1].entry_id,),
        ),
        OutcomeEventFact(
            sequence_number=2,
            topic_id="topic-001",
            subject="second",
            result="passed",
            source_entry_ids=(entries[0].entry_id,),
        ),
    )

    partitions = plan_outcome_partitions(_topic(entries), entries, events)

    selected = [partition for partition in partitions if partition.core_events]
    assert len(selected) == 1
    assert selected[0].ordinal == 0
    assert [event.sequence_number for event in selected[0].core_events] == [1, 2]
    assert selected[0].event_closure_exception_reason == "event_source_closure"


def test_two_phase_solver_avoids_hard_when_complete_normal_plan_exists() -> None:
    entries = tuple(
        entry
        for group in range(12)
        for entry in (
            _entry(group * 2, chr(97 + group) * 2_400),
            _entry(group * 2 + 1, chr(65 + group) * 300),
        )
    )
    events = (
        OutcomeEventFact(
            sequence_number=1,
            topic_id="topic-001",
            subject="first",
            result="passed",
            source_entry_ids=(entries[-1].entry_id,),
        ),
        OutcomeEventFact(
            sequence_number=2,
            topic_id="topic-001",
            subject="second",
            result="passed",
            source_entry_ids=(entries[0].entry_id,),
        ),
    )

    partitions = plan_outcome_partitions(_topic(entries), entries, events)

    assert sum(partition.event_closure_exception for partition in partitions) == 0
    selected = next(partition for partition in partitions if partition.core_events)
    assert selected.ordinal == 0
    assert [event.sequence_number for event in selected.core_events] == [1, 2]


def test_two_phase_solver_minimises_unavoidable_hard_exception_count() -> None:
    entry = _entry(0, "会" * 5_000)
    events = (
        _event_for_sources(entry.entry_id),
        OutcomeEventFact(
            sequence_number=2,
            topic_id="topic-001",
            subject="confirmed",
            result="passed",
            source_entry_ids=(entry.entry_id,),
        ),
    )

    partitions = plan_outcome_partitions(_topic((entry,)), (entry,), events)

    exceptions = [partition for partition in partitions if partition.event_closure_exception]
    assert len(exceptions) == 1
    assert [event.sequence_number for event in exceptions[0].core_events] == [1, 2]


def test_two_phase_solver_breaks_equal_size_ties_by_ordinal_then_hash() -> None:
    entries = tuple(
        entry
        for group in range(8)
        for entry in (
            _entry(group * 2, "x" * 2_800),
            _entry(group * 2 + 1, "y" * 300),
        )
    )
    event = _event_for_sources(entries[1].entry_id, entries[3].entry_id)

    first = plan_outcome_partitions(_topic(entries), entries, (event,))
    second = plan_outcome_partitions(_topic(entries), entries, (event,))

    assert tuple(partition.partition_hash for partition in first) == tuple(
        partition.partition_hash for partition in second
    )
    assert next(partition.ordinal for partition in first if partition.core_events) == 0


def test_phase_one_search_exhaustion_fails_before_accepting_any_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(outcome_partitions_module, "MAX_EVENT_SEARCH_STATES", 0)
    entry = _entry(0, "votes")

    with pytest.raises(OutcomePartitionError, match="planning_search_capacity"):
        plan_outcome_partitions(
            _topic((entry,)),
            (entry,),
            (_event_for_sources(entry.entry_id),),
        )


def test_phase_two_search_exhaustion_never_accepts_unproven_hard_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(outcome_partitions_module, "MAX_EVENT_SEARCH_STATES", 1)
    entry = _entry(0, "会" * 5_000)

    with pytest.raises(OutcomePartitionError, match="planning_search_capacity"):
        plan_outcome_partitions(
            _topic((entry,)),
            (entry,),
            (_event_for_sources(entry.entry_id),),
        )


@pytest.mark.parametrize("event_count", (1, 3, 8, 16))
def test_up_to_sixteen_small_events_fit_without_output_preflight(event_count: int) -> None:
    entry = _entry(0, "votes")
    events = tuple(
        OutcomeEventFact(
            sequence_number=index,
            topic_id="topic-001",
            subject=f"accepted-{index}",
            result="passed",
            source_entry_ids=(entry.entry_id,),
        )
        for index in range(1, event_count + 1)
    )

    partitions = plan_outcome_partitions(_topic((entry,)), (entry,), events)

    assert len(partitions) <= MAX_OUTCOME_PARTITIONS_PER_TOPIC
    assert all(len(partition.core_events) <= 2 for partition in partitions)
    assert sum(len(partition.core_events) for partition in partitions) == event_count
    assert all(
        not outcome_preflight_capacity_failure(
            partition,
            OutcomeRequestKind.DECISION_AGREEMENT,
        )
        for partition in partitions
    )


def test_twenty_five_events_fail_planning_with_explicit_output_capacity() -> None:
    entry = _entry(0, "votes")
    events = tuple(
        OutcomeEventFact(
            sequence_number=index,
            topic_id="topic-001",
            subject=f"accepted-{index}",
            result="passed",
            source_entry_ids=(entry.entry_id,),
        )
        for index in range(1, 26)
    )

    with pytest.raises(OutcomePartitionError, match="event_output_capacity"):
        plan_outcome_partitions(_topic((entry,)), (entry,), events)


def test_partition_caps_fail_closed_without_truncation() -> None:
    text = "x" * 100_000
    entry = _entry(0, text)
    with pytest.raises(OutcomePartitionError, match="fragment_capacity"):
        plan_outcome_partitions(_topic((entry,)), (entry,), ())

    topics = tuple(
        (
            _topic((replace(_entry(index), entry_id=f"t{index}"),)),
            (replace(_entry(index), entry_id=f"t{index}"),),
            (),
        )
        for index in range(MAX_OUTCOME_PARTITIONS_PER_JOB + 1)
    )
    with pytest.raises(OutcomePartitionError, match="job_partition_cap"):
        build_outcome_partition_manifest(
            canonical_hash="a" * 64,
            topic_artifact_hash="b" * 64,
            event_facts_hash="c" * 64,
            participant_hash="d" * 64,
            topics=topics,
        )


def test_exact_partition_cap_edges_accept_p12_p40_and_reject_p13_p41() -> None:
    p12_entries = tuple(_entry(index, chr(97 + index) * 3_000) for index in range(12))
    assert len(plan_outcome_partitions(_topic(p12_entries), p12_entries, ())) == 12

    p13_entries = tuple(_entry(index, chr(97 + index) * 3_000) for index in range(13))
    with pytest.raises(OutcomePartitionError, match="topic_partition_cap"):
        plan_outcome_partitions(_topic(p13_entries), p13_entries, ())

    def topic_row(index: int):
        entry = replace(_entry(index), entry_id=f"cap-{index:02d}")
        topic = replace(_topic((entry,)), id=f"topic-cap-{index:02d}")
        return (topic, (entry,), ())

    p40 = tuple(topic_row(index) for index in range(40))
    assert (
        len(
            build_outcome_partition_manifest(
                canonical_hash="a" * 64,
                topic_artifact_hash="b" * 64,
                event_facts_hash="c" * 64,
                participant_hash="d" * 64,
                topics=p40,
            ).partitions
        )
        == 40
    )
    with pytest.raises(OutcomePartitionError, match="job_partition_cap"):
        build_outcome_partition_manifest(
            canonical_hash="a" * 64,
            topic_artifact_hash="b" * 64,
            event_facts_hash="c" * 64,
            participant_hash="d" * 64,
            topics=(*p40, topic_row(40)),
        )


def test_manifest_self_hash_and_revision_binding_reject_tamper() -> None:
    entries = (_entry(1),)
    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((_topic(entries), entries, ()),),
    )
    validate_outcome_partition_manifest(manifest)
    bad_partition = manifest.partitions[0].model_copy(update={"partition_hash": "0" * 64})
    tampered = manifest.model_copy(update={"partitions": (bad_partition,)})
    with pytest.raises(OutcomePartitionError, match="manifest_tamper"):
        validate_outcome_partition_manifest(tampered)

    schema_tamper = manifest.model_copy(
        update={
            "response_schema_hashes": {
                manifest.partitions[0].partition_id: {
                    **manifest.response_schema_hashes[manifest.partitions[0].partition_id],
                    OutcomeRequestKind.QUESTION.value: "0" * 64,
                }
            }
        }
    )
    schema_tamper = schema_tamper.model_copy(
        update={"manifest_hash": outcome_partitions_module._hash(schema_tamper.hash_input())}
    )
    with pytest.raises(OutcomePartitionError, match="manifest_tamper"):
        validate_outcome_partition_manifest(schema_tamper)

    prior_config = manifest.model_copy(update={"extraction_config_version": 23})
    prior_config = prior_config.model_copy(
        update={"manifest_hash": outcome_partitions_module._hash(prior_config.hash_input())}
    )
    with pytest.raises(OutcomePartitionError, match="manifest_tamper"):
        validate_outcome_partition_manifest(prior_config)


def test_owner_approved_fix7_limits_are_exact() -> None:
    assert MAX_TOPIC_BOUNDARY_CALLS_PER_JOB == 120
    assert MAX_OUTCOME_PARTITIONS_PER_TOPIC == 12
    assert MAX_OUTCOME_PARTITIONS_PER_JOB == 40
    assert MAX_TOTAL_LLM_INFERENCE_CALLS_PER_JOB == 600
    assert MAX_TOTAL_LLM_HTTP_REQUESTS_PER_JOB == 601
    assert MAX_LLM_WALL_SECONDS_PER_JOB == 5_400
    assert MAX_OUTCOME_OUTPUT_TOKENS == 768
    assert MAX_DEEPSEEK_REPAIR_OUTPUT_TOKENS == 32_768


@pytest.mark.parametrize(
    "unsafe_reason",
    ("UPPERCASE", "contains spaces", "message: private detail", ""),
)
def test_partition_error_replaces_unapproved_reason_shapes(unsafe_reason: str) -> None:
    error = OutcomePartitionError(unsafe_reason)

    assert error.reason == "outcome_partition_error"
    assert str(error) == "outcome_partition_error"


def test_partition_manifest_serialization_contains_no_source_text_telemetry() -> None:
    sentinel = "PRIVATE-CONTENT-SENTINEL"
    entries = (_entry(1, sentinel),)
    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((_topic(entries), entries, ()),),
    )
    telemetry = json.dumps(manifest.telemetry(), sort_keys=True)
    assert sentinel not in telemetry
    assert set(manifest.telemetry()) == {
        "partition_count",
        "size_buckets",
        "target_ladder_revision",
        "target_attempts",
        "planning_search_state_limit",
        "planning_search_states_used",
        "context_target_chars",
        "max_request_chars",
        "event_closure_exception_count",
        "event_closure_exception_reason_counts",
        "manifest_hash",
    }


def test_terminal_result_cache_binds_partition_kind_model_and_revision() -> None:
    entries = (_entry(1),)
    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((_topic(entries), entries, ()),),
        pipeline_revision="e" * 64,
    )
    partition = manifest.partitions[0]
    result = OutcomePartitionResult.terminal(
        partition=partition,
        kind=OutcomeRequestKind.ACTION,
        result=TopicOutcomeSet(
            topic_id=partition.topic_id,
            candidates=(),
            extraction_method="local_llm",
            failure_types=(),
        ),
        pipeline_revision="e" * 64,
    )
    result.validate_for(partition, OutcomeRequestKind.ACTION, "e" * 64)
    with pytest.raises(OutcomePartitionError, match="partition_result_tamper"):
        result.model_copy(update={"result_hash": "0" * 64}).validate_for(
            partition, OutcomeRequestKind.ACTION, "e" * 64
        )
    with pytest.raises(OutcomePartitionError, match="partition_result_tamper"):
        result.validate_for(partition, OutcomeRequestKind.QUESTION, "e" * 64)
    with pytest.raises(OutcomePartitionError, match="partition_result_tamper"):
        result.model_copy(update={"response_schema_hash": "0" * 64}).validate_for(
            partition, OutcomeRequestKind.ACTION, "e" * 64
        )


def test_fix33_terminal_result_hash_binds_complete_wire_row_resolution_coverage() -> None:
    entries = (_entry(1),)
    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((_topic(entries), entries, ()),),
        pipeline_revision="e" * 64,
    )
    partition = manifest.partitions[0]
    candidate = outcome_partitions_module.OutcomeCandidate(
        id="outcome-test",
        type=OutcomeType.OPEN_QUESTION,
        text="supported question",
        owner_user_id=None,
        owner_name=None,
        deadline=None,
        source_entry_ids=(entries[0].entry_id,),
        source_event_ids=(),
        confidence=0.0,
        requires_confirmation=False,
        verification_status=VerificationStatus.UNVERIFIED,
    )
    result = OutcomePartitionResult.terminal(
        partition=partition,
        kind=OutcomeRequestKind.QUESTION,
        result=TopicOutcomeSet(
            topic_id=partition.topic_id,
            candidates=(candidate,),
            extraction_method="local_llm",
            failure_types=(),
            wire_input_row_count=2,
            wire_row_resolutions=(
                OutcomeWireRowResolution(
                    output_candidate_id=candidate.id,
                    retained_ordinal=0,
                    contributor_ordinals=(0, 1),
                ),
            ),
        ),
        pipeline_revision="e" * 64,
    )

    assert result.wire_input_row_count == 2
    assert result.exact_duplicate_group_count == 1
    assert result.dropped_wire_row_count == 1
    result.validate_for(partition, OutcomeRequestKind.QUESTION, "e" * 64)

    payload = json.loads(result.model_dump_json())
    payload["wire_row_resolutions"][0]["contributor_ordinals"] = [0]
    with pytest.raises(ValueError):
        OutcomePartitionResult.model_validate_json(json.dumps(payload))


def test_fix33_manifest_binds_observed_topic_partition_count_and_schema_hash() -> None:
    entries = (_entry(1),)
    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((_topic(entries), entries, ()),),
    )
    partition = manifest.partitions[0]
    assert partition.topic_partition_count == 1
    original_schema_hash = outcome_partitions_module.outcome_response_schema_hash(
        partition, OutcomeRequestKind.DECISION_AGREEMENT
    )
    rebound = partition.model_copy(update={"topic_partition_count": 8, "partition_hash": "0" * 64})
    rebound = rebound.model_copy(
        update={"partition_hash": outcome_partitions_module._hash(rebound.hash_input())}
    )
    assert (
        outcome_partitions_module.outcome_response_schema_hash(
            rebound, OutcomeRequestKind.DECISION_AGREEMENT
        )
        != original_schema_hash
    )
    tampered = manifest.model_copy(update={"partitions": (rebound,)})
    tampered = tampered.model_copy(
        update={"manifest_hash": outcome_partitions_module._hash(tampered.hash_input())}
    )
    with pytest.raises(OutcomePartitionError, match="manifest_tamper"):
        validate_outcome_partition_manifest(tampered)


def test_terminal_result_persists_closed_capacity_subtype_and_rejects_tamper() -> None:
    entries = (_entry(1),)
    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((_topic(entries), entries, ()),),
        pipeline_revision="e" * 64,
    )
    partition = manifest.partitions[0]
    failure = OutcomeProviderFailure(
        category=OutcomeFailureCategory.OUTPUT_CAPACITY,
        request_kind=OutcomeFailureRequestKind.ACTION,
        attempt=OutcomeFailureAttempt.INITIAL,
        count=1,
        size_bucket=OutcomeSizeBucket.LTE_1K,
        capacity_subtype=OutcomeCapacitySubtype.ROW_SATURATION,
    )
    result = OutcomePartitionResult.terminal(
        partition=partition,
        kind=OutcomeRequestKind.ACTION,
        result=TopicOutcomeSet(
            topic_id=partition.topic_id,
            candidates=(),
            extraction_method="local_llm_partial",
            failure_types=(),
            provider_failures=(failure,),
        ),
        pipeline_revision="e" * 64,
    )

    payload = json.loads(result.model_dump_json())
    assert payload["provider_failures"][0]["capacity_subtype"] == "row_saturation"
    restored = OutcomePartitionResult.model_validate_json(result.model_dump_json())
    restored.validate_for(partition, OutcomeRequestKind.ACTION, "e" * 64)

    for invalid_subtype in (None, "unknown"):
        invalid = json.loads(result.model_dump_json())
        invalid["provider_failures"][0]["capacity_subtype"] = invalid_subtype
        with pytest.raises(ValueError):
            OutcomePartitionResult.model_validate_json(json.dumps(invalid))

    missing = json.loads(result.model_dump_json())
    del missing["provider_failures"][0]["capacity_subtype"]
    with pytest.raises(ValueError):
        OutcomePartitionResult.model_validate_json(json.dumps(missing))

    impossible = json.loads(result.model_dump_json())
    impossible["provider_failures"][0]["capacity_subtype"] = "event_subject"
    with pytest.raises(ValueError):
        OutcomePartitionResult.model_validate_json(json.dumps(impossible))

    valid_but_tampered = json.loads(result.model_dump_json())
    valid_but_tampered["provider_failures"][0]["capacity_subtype"] = "complete_false"
    tampered = OutcomePartitionResult.model_validate_json(json.dumps(valid_but_tampered))
    with pytest.raises(OutcomePartitionError, match="partition_result_tamper"):
        tampered.validate_for(partition, OutcomeRequestKind.ACTION, "e" * 64)


def test_fix27_terminal_result_persists_closed_schema_subtype_and_rejects_tamper() -> None:
    entries = (_entry(1),)
    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((_topic(entries), entries, ()),),
        pipeline_revision="e" * 64,
    )
    partition = manifest.partitions[0]
    failure = OutcomeProviderFailure(
        category=OutcomeFailureCategory.SCHEMA,
        request_kind=OutcomeFailureRequestKind.QUESTION,
        attempt=OutcomeFailureAttempt.REPAIR,
        count=1,
        size_bucket=OutcomeSizeBucket.LTE_1K,
        capacity_subtype=None,
        format_subtype=OutcomeFormatSubtype.SCHEMA_ENTRY_REFS,
    )
    result = OutcomePartitionResult.terminal(
        partition=partition,
        kind=OutcomeRequestKind.QUESTION,
        result=TopicOutcomeSet(
            topic_id=partition.topic_id,
            candidates=(),
            extraction_method="local_llm_partial",
            failure_types=(),
            provider_failures=(failure,),
        ),
        pipeline_revision="e" * 64,
    )

    restored = OutcomePartitionResult.model_validate_json(result.model_dump_json())
    assert restored.provider_failures[0].format_subtype is (OutcomeFormatSubtype.SCHEMA_ENTRY_REFS)
    restored.validate_for(partition, OutcomeRequestKind.QUESTION, "e" * 64)
    payload = json.loads(result.model_dump_json())
    payload["provider_failures"][0]["format_subtype"] = "schema_event_refs"
    with pytest.raises(ValueError):
        OutcomePartitionResult.model_validate_json(json.dumps(payload))
    payload["provider_failures"][0]["format_subtype"] = "schema_text"
    tampered = OutcomePartitionResult.model_validate_json(json.dumps(payload))
    with pytest.raises(OutcomePartitionError, match="partition_result_tamper"):
        tampered.validate_for(partition, OutcomeRequestKind.QUESTION, "e" * 64)


@pytest.mark.parametrize(
    "duplicate_subtype",
    (
        OutcomeFormatSubtype.DUPLICATE_BINDING_WIRE_DIVERGENT,
        OutcomeFormatSubtype.DUPLICATE_BINDING_CANONICAL_ALIAS_COLLISION,
    ),
)
def test_fix32_terminal_result_persists_duplicate_subtypes_and_rejects_tamper(
    duplicate_subtype: OutcomeFormatSubtype,
) -> None:
    entries = (_entry(1),)
    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((_topic(entries), entries, ()),),
        pipeline_revision="e" * 64,
    )
    partition = manifest.partitions[0]
    failure = OutcomeProviderFailure(
        category=OutcomeFailureCategory.SCHEMA,
        request_kind=OutcomeFailureRequestKind.QUESTION,
        attempt=OutcomeFailureAttempt.INITIAL,
        count=1,
        size_bucket=OutcomeSizeBucket.LTE_1K,
        capacity_subtype=None,
        format_subtype=duplicate_subtype,
    )
    result = OutcomePartitionResult.terminal(
        partition=partition,
        kind=OutcomeRequestKind.QUESTION,
        result=TopicOutcomeSet(
            topic_id=partition.topic_id,
            candidates=(),
            extraction_method="local_llm_partial",
            failure_types=(),
            provider_failures=(failure,),
        ),
        pipeline_revision="e" * 64,
    )

    restored = OutcomePartitionResult.model_validate_json(result.model_dump_json())
    assert restored.provider_failures[0].format_subtype is duplicate_subtype
    restored.validate_for(partition, OutcomeRequestKind.QUESTION, "e" * 64)
    payload = json.loads(result.model_dump_json())
    payload["provider_failures"][0]["format_subtype"] = "schema_text"
    tampered = OutcomePartitionResult.model_validate_json(json.dumps(payload))
    with pytest.raises(OutcomePartitionError, match="partition_result_tamper"):
        tampered.validate_for(partition, OutcomeRequestKind.QUESTION, "e" * 64)


@pytest.mark.parametrize(
    ("kind", "category", "subtype"),
    (
        (
            OutcomeRequestKind.QUESTION,
            OutcomeFailureCategory.SCHEMA,
            OutcomeFormatSubtype.SCHEMA_ENTRY_AUTHORITY,
        ),
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            OutcomeFailureCategory.SCHEMA,
            OutcomeFormatSubtype.SCHEMA_EVENT_AUTHORITY,
        ),
        (
            OutcomeRequestKind.ACTION,
            OutcomeFailureCategory.SCHEMA,
            OutcomeFormatSubtype.SCHEMA_OWNER_AUTHORITY,
        ),
        (
            OutcomeRequestKind.ACTION,
            OutcomeFailureCategory.JSON,
            OutcomeFormatSubtype.JSON_EMPTY,
        ),
        (
            OutcomeRequestKind.QUESTION,
            OutcomeFailureCategory.JSON,
            OutcomeFormatSubtype.JSON_EXTRA_DATA,
        ),
        (
            OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT,
            OutcomeFailureCategory.JSON,
            OutcomeFormatSubtype.JSON_UNTERMINATED_STRING,
        ),
        (
            OutcomeRequestKind.DECISION_AGREEMENT,
            OutcomeFailureCategory.JSON,
            OutcomeFormatSubtype.JSON_SYNTAX_OTHER,
        ),
    ),
)
def test_fix30_terminal_result_persists_closed_format_subtype_and_rejects_tamper(
    kind: OutcomeRequestKind,
    category: OutcomeFailureCategory,
    subtype: OutcomeFormatSubtype,
) -> None:
    entries = (_entry(1),)
    manifest = build_outcome_partition_manifest(
        canonical_hash="a" * 64,
        topic_artifact_hash="b" * 64,
        event_facts_hash="c" * 64,
        participant_hash="d" * 64,
        topics=((_topic(entries), entries, ()),),
        pipeline_revision="e" * 64,
    )
    partition = manifest.partitions[0]
    failure = OutcomeProviderFailure(
        category=category,
        request_kind=OutcomeFailureRequestKind(kind.value),
        attempt=OutcomeFailureAttempt.INITIAL,
        count=1,
        size_bucket=OutcomeSizeBucket.LTE_1K,
        capacity_subtype=None,
        format_subtype=subtype,
    )
    result = OutcomePartitionResult.terminal(
        partition=partition,
        kind=kind,
        result=TopicOutcomeSet(
            topic_id=partition.topic_id,
            candidates=(),
            extraction_method="local_llm_partial",
            failure_types=(),
            provider_failures=(failure,),
        ),
        pipeline_revision="e" * 64,
    )

    restored = OutcomePartitionResult.model_validate_json(result.model_dump_json())
    assert restored.provider_failures[0].format_subtype is subtype
    restored.validate_for(partition, kind, "e" * 64)
    payload = json.loads(result.model_dump_json())
    payload["provider_failures"][0]["format_subtype"] = (
        OutcomeFormatSubtype.SCHEMA_ROOT
        if category is OutcomeFailureCategory.SCHEMA
        else OutcomeFormatSubtype.JSON_DECODE
    )
    tampered = OutcomePartitionResult.model_validate_json(json.dumps(payload))
    with pytest.raises(OutcomePartitionError, match="partition_result_tamper"):
        tampered.validate_for(partition, kind, "e" * 64)
