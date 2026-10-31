"""Deterministic cross-topic merge for already validated outcome candidates."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lingxu.worker.outcomes import (
    MAX_CANDIDATES_PER_TOPIC,
    MAX_EVIDENCE_IDS_PER_CANDIDATE,
    MAX_TOPIC_OUTCOME_SETS,
    OutcomeCandidate,
    OutcomeType,
    TopicOutcomeSet,
    VerificationStatus,
)
from lingxu.worker.semantic_policy import contains_reversal, material_actions

_LATIN_TERM = re.compile(r"[a-z0-9]+")
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")
MAX_MERGED_CANDIDATE_SOURCES = MAX_TOPIC_OUTCOME_SETS * MAX_CANDIDATES_PER_TOPIC

_TYPE_ORDER = {
    outcome_type: index
    for index, outcome_type in enumerate(
        (
            OutcomeType.DECISION,
            OutcomeType.AGREEMENT,
            OutcomeType.ACTION_ITEM,
            OutcomeType.OPEN_QUESTION,
            OutcomeType.PROPOSAL,
            OutcomeType.RISK,
            OutcomeType.DISAGREEMENT,
        )
    )
}
_STATUS_SAFETY_ORDER = {
    VerificationStatus.EVIDENCE_LINKED: 0,
    VerificationStatus.UNCONFIRMED: 1,
    VerificationStatus.NOT_CHECKED: 2,
    VerificationStatus.UNVERIFIED: 3,
}


class MergedCandidateSource(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )

    topic_id: str | None = Field(default=None, min_length=1, max_length=128)
    candidate_id: str = Field(min_length=1, max_length=128)
    source_entry_ids: tuple[str, ...] = Field(max_length=MAX_EVIDENCE_IDS_PER_CANDIDATE)
    source_event_ids: tuple[int, ...] = Field(max_length=MAX_EVIDENCE_IDS_PER_CANDIDATE)

    @model_validator(mode="after")
    def evidence_identifiers_are_unique(self) -> MergedCandidateSource:
        if len(set(self.source_entry_ids)) != len(self.source_entry_ids):
            raise ValueError("candidate source entry identifiers must be unique")
        if len(set(self.source_event_ids)) != len(self.source_event_ids):
            raise ValueError("candidate source event identifiers must be unique")
        return self


class MergedOutcomeProvenance(BaseModel):
    """All contributing validated identities without OutcomeCandidate's per-call limit."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )

    candidate_sources: tuple[MergedCandidateSource, ...] = Field(
        min_length=1, max_length=MAX_MERGED_CANDIDATE_SOURCES
    )

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> MergedOutcomeProvenance:
        identities = tuple(
            (source.topic_id, source.candidate_id) for source in self.candidate_sources
        )
        if len(set(identities)) != len(identities):
            raise ValueError("merged candidate sources must be unique")
        return self

    @property
    def source_entry_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    entry_id
                    for source in self.candidate_sources
                    for entry_id in source.source_entry_ids
                }
            )
        )

    @property
    def source_event_ids(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    event_id
                    for source in self.candidate_sources
                    for event_id in source.source_event_ids
                }
            )
        )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(source.candidate_id for source in self.candidate_sources)

    @property
    def topic_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    source.topic_id
                    for source in self.candidate_sources
                    if source.topic_id is not None
                }
            )
        )


class MergedMinutesOutcome(BaseModel):
    """One cross-topic minutes outcome with independently bounded aggregate provenance."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )

    id: str = Field(min_length=1, max_length=128)
    type: OutcomeType
    text: str = Field(min_length=1, max_length=500)
    owner_user_id: int | None
    owner_name: str | None = Field(max_length=128)
    deadline: str | None = Field(max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)
    requires_confirmation: bool
    verification_status: VerificationStatus
    provenance: MergedOutcomeProvenance

    @model_validator(mode="after")
    def values_are_coherent(self) -> MergedMinutesOutcome:
        if (self.owner_user_id is None) != (self.owner_name is None):
            raise ValueError("owner id and name must both be set or both be null")
        if not self.provenance.source_entry_ids and not self.provenance.source_event_ids:
            raise ValueError("merged outcome must preserve evidence")
        return self


class MergedOutcomes(BaseModel):
    """Final deterministic projections, split by user-facing semantic section."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )

    decisions: tuple[MergedMinutesOutcome, ...]
    agreements: tuple[MergedMinutesOutcome, ...]
    actions: tuple[MergedMinutesOutcome, ...]
    questions: tuple[MergedMinutesOutcome, ...]
    proposals: tuple[MergedMinutesOutcome, ...]
    risks: tuple[MergedMinutesOutcome, ...]
    disagreements: tuple[MergedMinutesOutcome, ...]
    deferred_items: tuple[MergedMinutesOutcome, ...]
    speaker_claims: tuple[MergedMinutesOutcome, ...]


class ValidatedOutcomeBindingDisposition(StrEnum):
    """Closed production disposition for one annotation-authority binding."""

    RETAINED = "retained"
    COALESCED = "coalesced"
    AMBIGUOUS_REJECTED = "ambiguous_rejected"


class ValidatedOutcomeOriginScope(StrEnum):
    """Closed, content-free partition scope for one resolved binding."""

    SAME_PARTITION = "same_partition"
    CROSS_PARTITION = "cross_partition"


type ValidatedOutcomeDifferenceField = Literal[
    "text",
    "owner_user_id",
    "owner_name",
    "deadline",
    "confidence",
    "requires_confirmation",
    "verification_status",
]
_VALIDATED_DIFFERENCE_FIELDS: tuple[ValidatedOutcomeDifferenceField, ...] = (
    "confidence",
    "deadline",
    "owner_name",
    "owner_user_id",
    "requires_confirmation",
    "text",
    "verification_status",
)


class ValidatedOutcomeBindingResolution(BaseModel):
    """Private contributor provenance for one exact ordered evidence binding."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )

    topic_id: str = Field(min_length=1, max_length=128)
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: ValidatedOutcomeBindingDisposition
    output_candidate_id: str | None = Field(default=None, min_length=1, max_length=128)
    contributor_candidate_ids: tuple[str, ...] = Field(
        min_length=1, max_length=MAX_CANDIDATES_PER_TOPIC
    )
    origin_scope: ValidatedOutcomeOriginScope
    origin_partition_count: int = Field(ge=1, le=MAX_CANDIDATES_PER_TOPIC)
    difference_fields: tuple[ValidatedOutcomeDifferenceField, ...] = Field(max_length=7)

    @model_validator(mode="after")
    def disposition_is_coherent(self) -> ValidatedOutcomeBindingResolution:
        contributors = self.contributor_candidate_ids
        if tuple(sorted(contributors)) != contributors or len(set(contributors)) != len(
            contributors
        ):
            raise ValueError("binding contributors must be canonical and unique")
        if self.disposition is ValidatedOutcomeBindingDisposition.RETAINED:
            if len(contributors) != 1 or self.output_candidate_id != contributors[0]:
                raise ValueError("retained binding provenance is incoherent")
        elif self.disposition is ValidatedOutcomeBindingDisposition.COALESCED:
            if len(contributors) < 2 or self.output_candidate_id != contributors[0]:
                raise ValueError("coalesced binding provenance is incoherent")
        elif self.output_candidate_id is not None or len(contributors) < 2:
            raise ValueError("ambiguous binding provenance is incoherent")
        if (
            (self.origin_partition_count == 1)
            != (self.origin_scope is ValidatedOutcomeOriginScope.SAME_PARTITION)
            or self.origin_partition_count > len(contributors)
            or tuple(sorted(self.difference_fields)) != self.difference_fields
            or len(set(self.difference_fields)) != len(self.difference_fields)
        ):
            raise ValueError("binding origin diagnostics are incoherent")
        if self.disposition is ValidatedOutcomeBindingDisposition.AMBIGUOUS_REJECTED:
            if not self.difference_fields:
                raise ValueError("ambiguous binding differences are missing")
        elif self.difference_fields:
            raise ValueError("non-ambiguous binding differences are invalid")
        return self


class ValidatedOutcomeBindingResult(BaseModel):
    """One topic after exact duplicate coalescing and ambiguous collision rejection."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )

    outcome_set: TopicOutcomeSet
    resolutions: tuple[ValidatedOutcomeBindingResolution, ...] = Field(
        max_length=MAX_CANDIDATES_PER_TOPIC
    )
    coalesced_candidate_count: int = Field(ge=0, le=MAX_CANDIDATES_PER_TOPIC)
    ambiguous_binding_count: int = Field(ge=0, le=MAX_CANDIDATES_PER_TOPIC)
    ambiguous_candidate_count: int = Field(ge=0, le=MAX_CANDIDATES_PER_TOPIC)


class ValidatedOutcomeCandidateRejection(BaseModel):
    """Private identity of one candidate rejected by individual evidence validation."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )

    topic_id: str = Field(min_length=1, max_length=128)
    candidate_id: str = Field(min_length=1, max_length=128)


@dataclass(frozen=True, slots=True)
class _MergeInput:
    candidate: OutcomeCandidate
    topic_id: str | None


def _normalise_text(value: str) -> str:
    normalised = unicodedata.normalize("NFKC", value).casefold()
    return " ".join((*_LATIN_TERM.findall(normalised), *_CJK_RUN.findall(normalised)))


def _validated_binding_key(
    topic_id: str,
    candidate: OutcomeCandidate,
) -> tuple[str, str, tuple[str, ...], tuple[int, ...]]:
    """Match Golden's exact ordered binding authority without importing evaluator code."""

    return (
        topic_id,
        candidate.type.value,
        candidate.source_entry_ids,
        candidate.source_event_ids,
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def resolve_validated_topic_bindings(
    outcome_set: TopicOutcomeSet,
    *,
    candidate_origins: dict[str, str],
) -> ValidatedOutcomeBindingResult:
    """Coalesce only byte-exact duplicates; reject ambiguous annotation bindings closed."""

    candidate_ids = tuple(candidate.id for candidate in outcome_set.candidates)
    if (
        len(set(candidate_ids)) != len(candidate_ids)
        or set(candidate_origins) != set(candidate_ids)
        or any(not isinstance(value, str) or not value for value in candidate_origins.values())
    ):
        raise ValueError("candidate origin authority is invalid")
    grouped: dict[tuple[str, str, tuple[str, ...], tuple[int, ...]], list[OutcomeCandidate]] = {}
    for candidate in outcome_set.candidates:
        grouped.setdefault(_validated_binding_key(outcome_set.topic_id, candidate), []).append(
            candidate
        )

    retained: list[OutcomeCandidate] = []
    resolutions: list[ValidatedOutcomeBindingResolution] = []
    coalesced_count = 0
    ambiguous_binding_count = 0
    ambiguous_count = 0
    for binding in sorted(grouped):
        candidates = tuple(sorted(grouped[binding], key=lambda item: item.id))
        contributor_ids = tuple(candidate.id for candidate in candidates)
        canonical_values = []
        for candidate in candidates:
            values = candidate.model_dump(mode="json")
            values.pop("id")
            canonical_values.append(_canonical_bytes(values))
        partition_count = len({candidate_origins[candidate.id] for candidate in candidates})
        origin_scope = (
            ValidatedOutcomeOriginScope.SAME_PARTITION
            if partition_count == 1
            else ValidatedOutcomeOriginScope.CROSS_PARTITION
        )
        candidate_payloads = tuple(candidate.model_dump(mode="json") for candidate in candidates)
        difference_fields = tuple(
            field
            for field in _VALIDATED_DIFFERENCE_FIELDS
            if len({_canonical_bytes(payload[field]) for payload in candidate_payloads}) > 1
        )
        binding_hash = hashlib.sha256(_canonical_bytes(binding)).hexdigest()
        if len(set(canonical_values)) != 1:
            ambiguous_binding_count += 1
            ambiguous_count += len(candidates)
            resolutions.append(
                ValidatedOutcomeBindingResolution(
                    topic_id=outcome_set.topic_id,
                    binding_hash=binding_hash,
                    disposition=ValidatedOutcomeBindingDisposition.AMBIGUOUS_REJECTED,
                    output_candidate_id=None,
                    contributor_candidate_ids=contributor_ids,
                    origin_scope=origin_scope,
                    origin_partition_count=partition_count,
                    difference_fields=difference_fields,
                )
            )
            continue
        representative = candidates[0]
        retained.append(representative)
        disposition = ValidatedOutcomeBindingDisposition.RETAINED
        if len(candidates) > 1:
            disposition = ValidatedOutcomeBindingDisposition.COALESCED
            coalesced_count += len(candidates) - 1
        resolutions.append(
            ValidatedOutcomeBindingResolution(
                topic_id=outcome_set.topic_id,
                binding_hash=binding_hash,
                disposition=disposition,
                output_candidate_id=representative.id,
                contributor_candidate_ids=contributor_ids,
                origin_scope=origin_scope,
                origin_partition_count=partition_count,
                difference_fields=(),
            )
        )
    retained.sort(key=lambda candidate: _sort_key(_MergeInput(candidate, outcome_set.topic_id)))
    return ValidatedOutcomeBindingResult(
        outcome_set=outcome_set.model_copy(update={"candidates": tuple(retained)}),
        resolutions=tuple(resolutions),
        coalesced_candidate_count=coalesced_count,
        ambiguous_binding_count=ambiguous_binding_count,
        ambiguous_candidate_count=ambiguous_count,
    )


def validate_validated_binding_artifact(
    *,
    linked_sets: tuple[TopicOutcomeSet, ...],
    validated_sets: tuple[TopicOutcomeSet, ...],
    resolutions: tuple[ValidatedOutcomeBindingResolution, ...],
    individual_rejections: tuple[ValidatedOutcomeCandidateRejection, ...],
    metrics: tuple[dict[str, object], ...],
    candidate_origins: dict[str, str],
) -> None:
    """Prove exact contributor coverage and deterministic resolution from the linked artifact."""

    if tuple(item.topic_id for item in linked_sets) != tuple(
        item.topic_id for item in validated_sets
    ):
        raise ValueError("validated binding topic order is invalid")
    resolutions_by_topic: dict[str, list[ValidatedOutcomeBindingResolution]] = {}
    for resolution in resolutions:
        resolutions_by_topic.setdefault(resolution.topic_id, []).append(resolution)
    rejections_by_topic: dict[str, list[str]] = {}
    for rejection in individual_rejections:
        rejections_by_topic.setdefault(rejection.topic_id, []).append(rejection.candidate_id)
    if set(resolutions_by_topic) - {item.topic_id for item in linked_sets} or set(
        rejections_by_topic
    ) - {item.topic_id for item in linked_sets}:
        raise ValueError("validated binding topic authority is invalid")
    if len(metrics) != len(linked_sets):
        raise ValueError("validated binding metrics are invalid")
    linked_ids = tuple(candidate.id for item in linked_sets for candidate in item.candidates)
    if len(set(linked_ids)) != len(linked_ids) or set(candidate_origins) != set(linked_ids):
        raise ValueError("candidate origin authority is invalid")

    expected_all_resolutions: list[ValidatedOutcomeBindingResolution] = []
    expected_all_rejections: list[ValidatedOutcomeCandidateRejection] = []
    for linked, validated, metric in zip(linked_sets, validated_sets, metrics, strict=True):
        linked_by_id = {candidate.id: candidate for candidate in linked.candidates}
        if len(linked_by_id) != len(linked.candidates):
            raise ValueError("linked candidate identity is invalid")
        rejected_ids = tuple(sorted(rejections_by_topic.get(linked.topic_id, ())))
        if len(rejected_ids) != len(set(rejected_ids)) or any(
            candidate_id not in linked_by_id for candidate_id in rejected_ids
        ):
            raise ValueError("individual rejection authority is invalid")
        individually_valid = tuple(
            candidate
            for candidate in linked.candidates
            if candidate.id not in frozenset(rejected_ids)
        )
        expected = resolve_validated_topic_bindings(
            linked.model_copy(update={"candidates": individually_valid}),
            candidate_origins={
                candidate.id: candidate_origins[candidate.id] for candidate in individually_valid
            },
        )
        expected_all_resolutions.extend(expected.resolutions)
        expected_all_rejections.extend(
            ValidatedOutcomeCandidateRejection(
                topic_id=linked.topic_id,
                candidate_id=candidate_id,
            )
            for candidate_id in rejected_ids
        )
        observed_resolutions = tuple(resolutions_by_topic.get(linked.topic_id, ()))
        if observed_resolutions != expected.resolutions or validated != expected.outcome_set:
            raise ValueError("validated binding resolution is invalid")
        contributor_ids = tuple(
            candidate_id
            for resolution in observed_resolutions
            for candidate_id in resolution.contributor_candidate_ids
        )
        if (
            len(contributor_ids) != len(set(contributor_ids))
            or set(contributor_ids) & set(rejected_ids)
            or set(contributor_ids) | set(rejected_ids) != set(linked_by_id)
        ):
            raise ValueError("validated binding contributor coverage is invalid")
        expected_metric = {
            "topic_id": linked.topic_id,
            "candidate_count": len(linked.candidates),
            "valid_count": len(expected.outcome_set.candidates),
            "rejected_count": len(rejected_ids) + expected.ambiguous_candidate_count,
            "individual_rejected_count": len(rejected_ids),
            "ambiguous_binding_count": expected.ambiguous_binding_count,
            "ambiguous_rejected_count": expected.ambiguous_candidate_count,
            "coalesced_count": expected.coalesced_candidate_count,
        }
        if set(metric) != {*expected_metric, "error_types"} or any(
            metric.get(key) != value for key, value in expected_metric.items()
        ):
            raise ValueError("validated binding metric authority is invalid")
        error_types = metric.get("error_types")
        if (
            not isinstance(error_types, dict)
            or error_types.get("ambiguous_outcome_binding", 0) != expected.ambiguous_binding_count
        ):
            raise ValueError("validated binding error metrics are invalid")
    if resolutions != tuple(expected_all_resolutions) or individual_rejections != tuple(
        expected_all_rejections
    ):
        raise ValueError("validated binding provenance order is invalid")


def _sort_key(value: _MergeInput) -> tuple[object, ...]:
    candidate = value.candidate
    return (
        _TYPE_ORDER[candidate.type],
        candidate.owner_user_id if candidate.owner_user_id is not None else -1,
        candidate.deadline or "",
        _normalise_text(candidate.text),
        value.topic_id or "",
        candidate.id,
    )


def _equivalent(left: _MergeInput, right: _MergeInput) -> bool:
    return _equivalence_key(left) == _equivalence_key(right)


def _equivalence_key(value: _MergeInput) -> tuple[object, ...]:
    candidate = value.candidate
    normalised_text = _normalise_text(candidate.text)
    return (
        candidate.type,
        candidate.owner_user_id,
        candidate.owner_name,
        candidate.deadline,
        candidate.verification_status is VerificationStatus.NOT_CHECKED,
        contains_reversal(candidate.text),
        tuple(sorted(material_actions(candidate.text))),
        normalised_text,
    )


def _combine(values: tuple[_MergeInput, ...]) -> MergedMinutesOutcome:
    ordered = tuple(sorted(values, key=_sort_key))
    representative = ordered[0].candidate
    status = max(
        (value.candidate.verification_status for value in ordered),
        key=_STATUS_SAFETY_ORDER.__getitem__,
    )
    return MergedMinutesOutcome(
        id=representative.id,
        type=representative.type,
        text=representative.text,
        owner_user_id=representative.owner_user_id,
        owner_name=representative.owner_name,
        deadline=representative.deadline,
        confidence=min(value.candidate.confidence for value in ordered),
        requires_confirmation=any(value.candidate.requires_confirmation for value in ordered),
        verification_status=status,
        provenance=MergedOutcomeProvenance(
            candidate_sources=tuple(
                MergedCandidateSource(
                    topic_id=value.topic_id,
                    candidate_id=value.candidate.id,
                    source_entry_ids=value.candidate.source_entry_ids,
                    source_event_ids=value.candidate.source_event_ids,
                )
                for value in ordered
            ),
        ),
    )


def _deterministic_clusters(
    values: tuple[_MergeInput, ...],
) -> tuple[tuple[_MergeInput, ...], ...]:
    """Group exact, hashable equivalence keys after one deterministic sort."""

    clusters: dict[tuple[object, ...], list[_MergeInput]] = {}
    for value in sorted(values, key=_sort_key):
        clusters.setdefault(_equivalence_key(value), []).append(value)
    return tuple(tuple(cluster) for cluster in clusters.values())


def _merge(values: tuple[_MergeInput, ...]) -> MergedOutcomes:
    claims: list[_MergeInput] = []
    mergeable: list[_MergeInput] = []
    for value in values:
        target = (
            claims
            if value.candidate.verification_status is VerificationStatus.NOT_CHECKED
            else mergeable
        )
        target.append(value)
    claims.sort(key=_sort_key)
    merged = tuple(_combine(cluster) for cluster in _deterministic_clusters(tuple(mergeable)))

    def of_type(outcome_type: OutcomeType) -> tuple[MergedMinutesOutcome, ...]:
        return tuple(candidate for candidate in merged if candidate.type is outcome_type)

    return MergedOutcomes(
        decisions=of_type(OutcomeType.DECISION),
        agreements=of_type(OutcomeType.AGREEMENT),
        actions=of_type(OutcomeType.ACTION_ITEM),
        questions=of_type(OutcomeType.OPEN_QUESTION),
        proposals=of_type(OutcomeType.PROPOSAL),
        risks=of_type(OutcomeType.RISK),
        disagreements=of_type(OutcomeType.DISAGREEMENT),
        deferred_items=(),
        speaker_claims=tuple(_combine((claim,)) for claim in claims),
    )


def merge_outcomes(candidates: tuple[OutcomeCandidate, ...]) -> MergedOutcomes:
    """Merge candidates without topic metadata for unit/compatibility callers."""

    return _merge(tuple(_MergeInput(candidate, None) for candidate in candidates))


def merge_topic_outcomes(outcome_sets: tuple[TopicOutcomeSet, ...]) -> MergedOutcomes:
    """Merge validated per-topic sets while preserving every topic/candidate relationship."""

    return _merge(
        tuple(
            _MergeInput(candidate, outcome_set.topic_id)
            for outcome_set in outcome_sets
            for candidate in outcome_set.candidates
        )
    )
