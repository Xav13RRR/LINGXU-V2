from itertools import permutations
from time import perf_counter

import pytest

import lingxu.worker.outcome_merge as outcome_merge_module
from lingxu.worker.outcome_merge import (
    MAX_MERGED_CANDIDATE_SOURCES,
    MergedCandidateSource,
    MergedOutcomeProvenance,
    MergedOutcomes,
    ValidatedOutcomeBindingDisposition,
    ValidatedOutcomeCandidateRejection,
    ValidatedOutcomeOriginScope,
    merge_outcomes,
    merge_topic_outcomes,
    resolve_validated_topic_bindings,
    validate_validated_binding_artifact,
)
from lingxu.worker.outcomes import (
    MAX_CANDIDATES_PER_TOPIC,
    MAX_TOPIC_OUTCOME_SETS,
    OutcomeCandidate,
    OutcomeType,
    TopicOutcomeSet,
    VerificationStatus,
)


def _candidate(
    candidate_id: str,
    outcome_type: OutcomeType,
    text: str,
    entry_id: str,
    *,
    owner_user_id: int | None = None,
    owner_name: str | None = None,
    deadline: str | None = None,
    status: VerificationStatus = VerificationStatus.EVIDENCE_LINKED,
) -> OutcomeCandidate:
    return OutcomeCandidate(
        id=candidate_id,
        type=outcome_type,
        text=text,
        owner_user_id=owner_user_id,
        owner_name=owner_name,
        deadline=deadline,
        source_entry_ids=(entry_id,),
        source_event_ids=(),
        confidence=0.9,
        requires_confirmation=False,
        verification_status=status,
    )


def test_merge_combines_only_equivalent_outcomes_and_preserves_conflict() -> None:
    first = _candidate(
        "a-1",
        OutcomeType.ACTION_ITEM,
        "Prepare the release package.",
        "e1",
        owner_user_id=7,
        owner_name="participant",
    )
    duplicate = _candidate(
        "a-2",
        OutcomeType.ACTION_ITEM,
        "prepare the release package",
        "e2",
        owner_user_id=7,
        owner_name="participant",
    )
    conflict = _candidate(
        "d-1",
        OutcomeType.DISAGREEMENT,
        "Do not prepare the release package",
        "e3",
    )

    merged = merge_outcomes((first, duplicate, conflict))

    assert len(merged.actions) == 1
    assert merged.actions[0].provenance.source_entry_ids == ("e1", "e2")
    assert len(merged.disagreements) == 1
    assert merged.disagreements[0].text == conflict.text


def test_fix27_exact_binding_duplicates_coalesce_with_complete_private_provenance() -> None:
    first = _candidate("partition-b", OutcomeType.RISK, "Capacity risk", "entry-a")
    second = first.model_copy(update={"id": "partition-a"})
    source = TopicOutcomeSet(
        topic_id="topic-01",
        candidates=(first, second),
        extraction_method="local_llm",
        failure_types=(),
    )

    resolved = resolve_validated_topic_bindings(
        source,
        candidate_origins={first.id: "partition-b", second.id: "partition-a"},
    )

    assert resolved.outcome_set.candidates == (second,)
    assert resolved.coalesced_candidate_count == 1
    assert resolved.ambiguous_binding_count == 0
    assert resolved.ambiguous_candidate_count == 0
    assert len(resolved.resolutions) == 1
    resolution = resolved.resolutions[0]
    assert resolution.disposition is ValidatedOutcomeBindingDisposition.COALESCED
    assert resolution.output_candidate_id == second.id
    assert resolution.contributor_candidate_ids == (second.id, first.id)
    assert resolution.origin_scope is ValidatedOutcomeOriginScope.CROSS_PARTITION
    assert resolution.origin_partition_count == 2
    assert resolution.difference_fields == ()
    assert len(resolution.binding_hash) == 64


def test_fix27_binding_resolution_is_input_order_and_restart_deterministic() -> None:
    values = tuple(
        _candidate(
            f"partition-{index}",
            OutcomeType.ACTION_ITEM,
            "Prepare release",
            "entry-a",
        )
        for index in range(3)
    )
    serialised = set()
    for order in permutations(values):
        result = resolve_validated_topic_bindings(
            TopicOutcomeSet(
                topic_id="topic-01",
                candidates=order,
                extraction_method="local_llm",
                failure_types=(),
            ),
            candidate_origins={candidate.id: "partition-a" for candidate in values},
        )
        serialised.add(result.model_dump_json())

    assert len(serialised) == 1


def test_fix27_same_binding_with_any_authority_difference_fails_closed() -> None:
    original = _candidate("candidate-a", OutcomeType.ACTION_ITEM, "Prepare release", "entry-a")
    for changed in (
        original.model_copy(update={"id": "candidate-b", "text": "Delay release"}),
        original.model_copy(update={"id": "candidate-b", "confidence": 0.8}),
        original.model_copy(update={"id": "candidate-b", "deadline": "Friday"}),
    ):
        resolved = resolve_validated_topic_bindings(
            TopicOutcomeSet(
                topic_id="topic-01",
                candidates=(original, changed),
                extraction_method="local_llm",
                failure_types=(),
            ),
            candidate_origins={original.id: "partition-a", changed.id: "partition-a"},
        )

        assert resolved.outcome_set.candidates == ()
        assert resolved.coalesced_candidate_count == 0
        assert resolved.ambiguous_binding_count == 1
        assert resolved.ambiguous_candidate_count == 2
        assert resolved.resolutions[0].origin_scope is ValidatedOutcomeOriginScope.SAME_PARTITION
        assert resolved.resolutions[0].origin_partition_count == 1
        assert resolved.resolutions[0].disposition is (
            ValidatedOutcomeBindingDisposition.AMBIGUOUS_REJECTED
        )
        assert resolved.resolutions[0].output_candidate_id is None


@pytest.mark.parametrize(
    ("update", "expected_field"),
    (
        ({"text": "Delay release"}, "text"),
        ({"owner_user_id": 8}, "owner_user_id"),
        ({"owner_name": "other participant"}, "owner_name"),
        ({"deadline": "Friday"}, "deadline"),
        ({"confidence": 0.8}, "confidence"),
        ({"requires_confirmation": True}, "requires_confirmation"),
        ({"verification_status": VerificationStatus.NOT_CHECKED}, "verification_status"),
    ),
)
def test_fix28_binding_origin_diagnostics_expose_only_closed_difference_fields(
    update: dict[str, object],
    expected_field: str,
) -> None:
    original = _candidate(
        "candidate-a",
        OutcomeType.ACTION_ITEM,
        "Prepare release",
        "entry-a",
        owner_user_id=7,
        owner_name="participant",
    )
    changed = original.model_copy(update={"id": "candidate-b", **update})

    resolved = resolve_validated_topic_bindings(
        TopicOutcomeSet(
            topic_id="topic-01",
            candidates=(original, changed),
            extraction_method="local_llm",
            failure_types=(),
        ),
        candidate_origins={original.id: "partition-a", changed.id: "partition-a"},
    )

    diagnostic = resolved.resolutions[0]
    assert diagnostic.origin_scope is ValidatedOutcomeOriginScope.SAME_PARTITION
    assert diagnostic.origin_partition_count == 1
    assert diagnostic.difference_fields == (expected_field,)
    assert set(diagnostic.model_dump()) == {
        "topic_id",
        "binding_hash",
        "disposition",
        "output_candidate_id",
        "contributor_candidate_ids",
        "origin_scope",
        "origin_partition_count",
        "difference_fields",
    }


def test_fix28_binding_origin_diagnostics_sort_multiple_fields_and_require_complete_mapping() -> (
    None
):
    original = _candidate("candidate-a", OutcomeType.RISK, "Capacity risk", "entry-a")
    changed = original.model_copy(
        update={
            "id": "candidate-b",
            "text": "Schedule risk",
            "deadline": "Friday",
            "confidence": 0.8,
        }
    )
    source = TopicOutcomeSet(
        topic_id="topic-01",
        candidates=(original, changed),
        extraction_method="local_llm",
        failure_types=(),
    )

    resolved = resolve_validated_topic_bindings(
        source,
        candidate_origins={original.id: "partition-a", changed.id: "partition-b"},
    )
    assert resolved.resolutions[0].origin_scope is ValidatedOutcomeOriginScope.CROSS_PARTITION
    assert resolved.resolutions[0].origin_partition_count == 2
    assert resolved.resolutions[0].difference_fields == ("confidence", "deadline", "text")
    with pytest.raises(ValueError, match="candidate origin authority"):
        resolve_validated_topic_bindings(
            source,
            candidate_origins={original.id: "partition-a"},
        )


def test_fix27_ordered_evidence_refs_are_distinct_binding_authority() -> None:
    first = _candidate("candidate-a", OutcomeType.RISK, "Capacity risk", "entry-a").model_copy(
        update={"source_entry_ids": ("entry-a", "entry-b")}
    )
    second = first.model_copy(
        update={"id": "candidate-b", "source_entry_ids": ("entry-b", "entry-a")}
    )

    resolved = resolve_validated_topic_bindings(
        TopicOutcomeSet(
            topic_id="topic-01",
            candidates=(first, second),
            extraction_method="local_llm",
            failure_types=(),
        ),
        candidate_origins={first.id: "partition-a", second.id: "partition-b"},
    )

    assert resolved.outcome_set.candidates == (first, second)
    assert len(resolved.resolutions) == 2
    assert all(
        item.disposition is ValidatedOutcomeBindingDisposition.RETAINED
        for item in resolved.resolutions
    )


def test_fix27_validated_artifact_rejects_missing_extra_or_overlapping_provenance() -> None:
    rejected = _candidate("candidate-rejected", OutcomeType.RISK, "Other risk", "entry-b")
    first = _candidate("candidate-b", OutcomeType.RISK, "Capacity risk", "entry-a")
    second = first.model_copy(update={"id": "candidate-a"})
    linked = TopicOutcomeSet(
        topic_id="topic-01",
        candidates=(first, second, rejected),
        extraction_method="local_llm",
        failure_types=(),
    )
    resolved = resolve_validated_topic_bindings(
        linked.model_copy(update={"candidates": (first, second)}),
        candidate_origins={first.id: "partition-a", second.id: "partition-b"},
    )
    rejections = (
        ValidatedOutcomeCandidateRejection(
            topic_id=linked.topic_id,
            candidate_id=rejected.id,
        ),
    )
    metrics = (
        {
            "topic_id": linked.topic_id,
            "candidate_count": 3,
            "valid_count": 1,
            "rejected_count": 1,
            "individual_rejected_count": 1,
            "ambiguous_binding_count": 0,
            "ambiguous_rejected_count": 0,
            "coalesced_count": 1,
            "error_types": {},
        },
    )

    validate_validated_binding_artifact(
        linked_sets=(linked,),
        validated_sets=(resolved.outcome_set,),
        resolutions=resolved.resolutions,
        individual_rejections=rejections,
        metrics=metrics,
        candidate_origins={
            first.id: "partition-a",
            second.id: "partition-b",
            rejected.id: "partition-c",
        },
    )
    for altered_resolutions, altered_rejections in (
        ((), rejections),
        (resolved.resolutions + resolved.resolutions, rejections),
        (
            resolved.resolutions,
            (
                *rejections,
                ValidatedOutcomeCandidateRejection(
                    topic_id=linked.topic_id,
                    candidate_id=first.id,
                ),
            ),
        ),
    ):
        with pytest.raises(ValueError):
            validate_validated_binding_artifact(
                linked_sets=(linked,),
                validated_sets=(resolved.outcome_set,),
                resolutions=altered_resolutions,
                individual_rejections=altered_rejections,
                metrics=metrics,
                candidate_origins={
                    first.id: "partition-a",
                    second.id: "partition-b",
                    rejected.id: "partition-c",
                },
            )


def test_fix27_validated_artifact_binds_cross_topic_provenance_order() -> None:
    linked_sets = tuple(
        TopicOutcomeSet(
            topic_id=f"topic-{index}",
            candidates=(
                _candidate(
                    f"candidate-{index}",
                    OutcomeType.RISK,
                    "Capacity risk",
                    f"entry-{index}",
                ),
            ),
            extraction_method="local_llm",
            failure_types=(),
        )
        for index in (1, 2)
    )
    origins = {
        candidate.id: f"partition-{index}"
        for index, item in enumerate(linked_sets, start=1)
        for candidate in item.candidates
    }
    results = tuple(
        resolve_validated_topic_bindings(
            item,
            candidate_origins={
                candidate.id: origins[candidate.id] for candidate in item.candidates
            },
        )
        for item in linked_sets
    )
    metrics = tuple(
        {
            "topic_id": item.topic_id,
            "candidate_count": 1,
            "valid_count": 1,
            "rejected_count": 0,
            "individual_rejected_count": 0,
            "ambiguous_binding_count": 0,
            "ambiguous_rejected_count": 0,
            "coalesced_count": 0,
            "error_types": {},
        }
        for item in linked_sets
    )
    resolutions = tuple(result.resolutions[0] for result in results)

    validate_validated_binding_artifact(
        linked_sets=linked_sets,
        validated_sets=tuple(result.outcome_set for result in results),
        resolutions=resolutions,
        individual_rejections=(),
        metrics=metrics,
        candidate_origins=origins,
    )
    with pytest.raises(ValueError, match="provenance order"):
        validate_validated_binding_artifact(
            linked_sets=linked_sets,
            validated_sets=tuple(result.outcome_set for result in results),
            resolutions=tuple(reversed(resolutions)),
            individual_rejections=(),
            metrics=metrics,
            candidate_origins=origins,
        )


def test_merge_never_combines_conflicting_owners_or_separate_deadlines() -> None:
    candidates = (
        _candidate(
            "a-1",
            OutcomeType.ACTION_ITEM,
            "Prepare release",
            "e1",
            owner_user_id=7,
            owner_name="participant",
            deadline="Friday",
        ),
        _candidate(
            "a-2",
            OutcomeType.ACTION_ITEM,
            "Prepare release",
            "e2",
            owner_user_id=8,
            owner_name="reviewer",
            deadline="Friday",
        ),
        _candidate(
            "a-3",
            OutcomeType.ACTION_ITEM,
            "Prepare release",
            "e3",
            owner_user_id=7,
            owner_name="participant",
            deadline="Monday",
        ),
    )

    assert len(merge_outcomes(candidates).actions) == 3


def test_merge_keeps_opposite_decisions_independent() -> None:
    accepted = _candidate("d-1", OutcomeType.DECISION, "Enable automatic deployment", "e1")
    rejected = _candidate("d-2", OutcomeType.DECISION, "Do not enable automatic deployment", "e2")

    assert len(merge_outcomes((accepted, rejected)).decisions) == 2


def test_merge_routes_unverified_speaker_claims_without_promoting_them() -> None:
    claim = _candidate(
        "r-1",
        OutcomeType.RISK,
        "A vendor says capacity is limited",
        "e1",
        status=VerificationStatus.NOT_CHECKED,
    )

    merged = merge_outcomes((claim,))

    assert merged.risks == ()
    assert len(merged.speaker_claims) == 1
    assert merged.speaker_claims[0].text == claim.text


def test_merge_is_deterministic_across_input_order() -> None:
    candidates = (
        _candidate("q-2", OutcomeType.OPEN_QUESTION, "When do we release?", "e2"),
        _candidate("q-1", OutcomeType.OPEN_QUESTION, "when do we release", "e1"),
        _candidate("r-1", OutcomeType.RISK, "Capacity risk", "e3"),
    )

    serialised = {merge_outcomes(order).model_dump_json() for order in permutations(candidates)}

    assert len(serialised) == 1


@pytest.mark.parametrize(
    ("positive", "reversal"),
    (
        (
            "The team will continue the production deployment configuration rollout tomorrow",
            "The team will stop the production deployment configuration rollout tomorrow",
        ),
        (
            "The team will continue the production deployment configuration rollout tomorrow",
            "The team will terminate the production deployment configuration rollout tomorrow",
        ),
        (
            "The team will approve the production deployment configuration rollout tomorrow",
            "The team will revoke the production deployment configuration rollout tomorrow",
        ),
        ("团队决定继续执行生产环境部署配置更新计划", "团队决定停止执行生产环境部署配置更新计划"),
        ("团队决定继续执行生产环境部署配置更新计划", "团队决定废止执行生产环境部署配置更新计划"),
        ("团队决定继续执行生产环境部署配置更新计划", "团队决定终止执行生产环境部署配置更新计划"),
        ("团队决定继续执行生产环境部署配置更新计划", "团队决定暂停执行生产环境部署配置更新计划"),
        ("团队决定继续执行生产环境部署配置更新计划", "团队决定stop执行生产环境部署配置更新计划"),
    ),
)
def test_merge_never_combines_mixed_script_reversals(positive: str, reversal: str) -> None:
    outcomes = (
        _candidate("d-1", OutcomeType.DECISION, positive, "e1"),
        _candidate("d-2", OutcomeType.DECISION, reversal, "e2"),
    )

    assert len(merge_outcomes(outcomes).decisions) == 2


def test_merge_fails_closed_on_material_action_delta() -> None:
    outcomes = (
        _candidate(
            "a-1",
            OutcomeType.ACTION_ITEM,
            "The owner will deploy the approved API configuration to production tomorrow",
            "e1",
        ),
        _candidate(
            "a-2",
            OutcomeType.ACTION_ITEM,
            "The owner will delete the approved API configuration from production tomorrow",
            "e2",
        ),
    )

    assert len(merge_outcomes(outcomes).actions) == 2


@pytest.mark.parametrize(
    ("first_text", "second_text"),
    (
        (
            "The owner will archive the approved production database snapshot tomorrow",
            "The owner will restore the approved production database snapshot tomorrow",
        ),
        (
            "The owner will encrypt the approved production database snapshot tomorrow",
            "The owner will decrypt the approved production database snapshot tomorrow",
        ),
    ),
)
def test_merge_fails_closed_on_unclassified_material_verb_delta(
    first_text: str,
    second_text: str,
) -> None:
    outcomes = (
        _candidate("a-1", OutcomeType.ACTION_ITEM, first_text, "e1"),
        _candidate("a-2", OutcomeType.ACTION_ITEM, second_text, "e2"),
    )

    assert len(merge_outcomes(outcomes).actions) == 2


def test_merge_preserves_more_than_candidate_evidence_limit_in_dedicated_provenance() -> None:
    topic_sets = tuple(
        TopicOutcomeSet(
            topic_id=f"topic-{index:02d}",
            candidates=(
                _candidate(
                    f"action-{index:02d}",
                    OutcomeType.ACTION_ITEM,
                    "Prepare the release package",
                    f"e{index:02d}",
                    owner_user_id=7,
                    owner_name="participant",
                ),
            ),
            extraction_method="local_llm",
            failure_types=(),
        )
        for index in range(13)
    )

    merged = merge_topic_outcomes(topic_sets)

    assert len(merged.actions) == 1
    assert len(merged.actions[0].provenance.source_entry_ids) == 13
    assert len(merged.actions[0].provenance.candidate_ids) == 13
    assert len(merged.actions[0].provenance.topic_ids) == 13
    assert MergedOutcomes.model_validate_json(merged.model_dump_json()) == merged


def test_merge_preserves_a_501_candidate_cluster_across_valid_topic_batches() -> None:
    candidate_total = 501
    per_topic = 80
    topic_sets = tuple(
        TopicOutcomeSet(
            topic_id=f"topic-{topic_index:02d}",
            candidates=tuple(
                _candidate(
                    f"action-{candidate_index:03d}",
                    OutcomeType.ACTION_ITEM,
                    "Prepare the release package",
                    f"e{candidate_index:03d}",
                )
                for candidate_index in range(
                    topic_index * per_topic,
                    min((topic_index + 1) * per_topic, candidate_total),
                )
            ),
            extraction_method="local_llm",
            failure_types=(),
        )
        for topic_index in range((candidate_total + per_topic - 1) // per_topic)
    )

    merged = merge_topic_outcomes(topic_sets)

    assert len(merged.actions) == 1
    assert len(merged.actions[0].provenance.candidate_sources) == candidate_total
    assert len(merged.actions[0].provenance.source_entry_ids) == candidate_total


def test_merged_provenance_accepts_declared_maximum_upstream_contributors() -> None:
    assert MAX_MERGED_CANDIDATE_SOURCES == (MAX_TOPIC_OUTCOME_SETS * MAX_CANDIDATES_PER_TOPIC)
    sources = tuple(
        MergedCandidateSource(
            topic_id=f"topic-{index // MAX_CANDIDATES_PER_TOPIC:03d}",
            candidate_id=f"candidate-{index:05d}",
            source_entry_ids=("shared-evidence",),
            source_event_ids=(),
        )
        for index in range(MAX_MERGED_CANDIDATE_SOURCES)
    )

    provenance = MergedOutcomeProvenance(candidate_sources=sources)

    assert len(provenance.candidate_sources) == MAX_MERGED_CANDIDATE_SOURCES
    assert provenance.source_entry_ids == ("shared-evidence",)


def test_exact_equivalence_grouping_is_linear_not_pairwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_total = 960
    topic_sets = tuple(
        TopicOutcomeSet(
            topic_id=f"topic-{topic_index:03d}",
            candidates=tuple(
                _candidate(
                    f"linear-{topic_index:03d}-{candidate_index:02d}",
                    OutcomeType.ACTION_ITEM,
                    "Prepare the release package",
                    f"linear-e-{topic_index:03d}-{candidate_index:02d}",
                )
                for candidate_index in range(MAX_CANDIDATES_PER_TOPIC)
            ),
            extraction_method="local_llm",
            failure_types=(),
        )
        for topic_index in range(candidate_total // MAX_CANDIDATES_PER_TOPIC)
    )
    comparisons = 0
    original = outcome_merge_module._equivalent

    def count_equivalence(left: object, right: object) -> bool:
        nonlocal comparisons
        comparisons += 1
        return original(left, right)  # type: ignore[arg-type]

    monkeypatch.setattr(outcome_merge_module, "_equivalent", count_equivalence)

    merged = merge_topic_outcomes(topic_sets)

    assert len(merged.actions) == 1
    assert comparisons <= candidate_total * 2


def test_full_declared_capacity_merge_and_json_round_trip_stays_bounded() -> None:
    topic_sets = tuple(
        TopicOutcomeSet(
            topic_id=f"capacity-topic-{topic_index:03d}",
            candidates=tuple(
                _candidate(
                    f"capacity-{topic_index:03d}-{candidate_index:02d}",
                    OutcomeType.ACTION_ITEM,
                    "Prepare the release package",
                    f"capacity-e-{topic_index:03d}-{candidate_index:02d}",
                )
                for candidate_index in range(MAX_CANDIDATES_PER_TOPIC)
            ),
            extraction_method="local_llm",
            failure_types=(),
        )
        for topic_index in range(MAX_TOPIC_OUTCOME_SETS)
    )

    started = perf_counter()
    merged = merge_topic_outcomes(topic_sets)
    serialised = merged.model_dump_json()
    recovered = MergedOutcomes.model_validate_json(serialised)
    elapsed = perf_counter() - started

    assert recovered == merged
    assert len(recovered.actions[0].provenance.candidate_sources) == (
        MAX_TOPIC_OUTCOME_SETS * MAX_CANDIDATES_PER_TOPIC
    )
    assert elapsed < 30.0
