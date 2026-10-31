"""Deterministic evidence linking and validation for untrusted outcomes."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from lingxu.transcript.models import CanonicalTranscriptEntry
from lingxu.worker.outcomes import (
    MAX_CANDIDATES_PER_TOPIC,
    MAX_EVIDENCE_IDS_PER_CANDIDATE,
    OutcomeCandidate,
    OutcomeEventFact,
    OutcomeType,
    VerificationStatus,
)
from lingxu.worker.semantic_policy import (
    SpeechActKind,
    classify_speech_acts,
)
from lingxu.worker.topics import TopicSegment

_FIRST_PERSON_OWNER = re.compile(
    r"(?:\bi\s+(?:will|can|shall|accept|own|am\s+going\s+to|take)\b|\bi(?:'|`)ll\b|"
    r"我(?:会|来|负责|接受|承担)|交给我)",
    re.IGNORECASE,
)
_ASSIGNMENT = re.compile(
    r"(?:\bwill\b|\bassigned?\b|\bowner\b|\bresponsible\b|负责|交给|由.+(?:处理|完成|负责))",
    re.IGNORECASE,
)
_ACCEPTED_RESULTS = frozenset({"accepted", "approved", "passed", "yes", "adopted", "carried"})
MAX_CACHED_ENTRY_REDUCTIONS_PER_TOPIC = MAX_CANDIDATES_PER_TOPIC * MAX_EVIDENCE_IDS_PER_CANDIDATE
MAX_CACHED_CANDIDATE_REDUCTIONS_PER_TOPIC = MAX_CANDIDATES_PER_TOPIC


@dataclass(frozen=True, slots=True)
class EvidenceContext:
    topic: TopicSegment
    canonical_entries: tuple[CanonicalTranscriptEntry, ...]
    authoritative_events: tuple[OutcomeEventFact, ...]
    allowed_participant_ids: frozenset[int]
    participant_names: dict[int, str]


@dataclass(frozen=True, slots=True)
class OutcomeValidation:
    valid: bool
    error_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompactSpeechActReduction:
    positive_acceptance: bool
    disqualifying: bool
    nondecision: bool
    attributive_mention: bool
    reported_scope: bool
    external_claim_attribution: bool
    fail_closed_reason: str | None


def _compact(text: str) -> CompactSpeechActReduction:
    analysis = classify_speech_acts(text)
    kinds = frozenset(signal.kind for signal in analysis.signals)
    reason: str | None = None
    if analysis.exceeded_input_limit:
        reason = "semantic_input_too_long"
    elif analysis.signal_overflow:
        reason = "semantic_signal_capacity"
    else:
        ambiguous_reasons = {
            signal.reason for signal in analysis.signals if signal.kind is SpeechActKind.AMBIGUOUS
        }
        if "quote_depth" in ambiguous_reasons:
            reason = "semantic_quote_depth"
        elif "quote_mismatch" in ambiguous_reasons:
            reason = "semantic_quote_mismatch"
        elif "unknown_attribution" in ambiguous_reasons:
            reason = "semantic_unknown_attribution"
        elif ambiguous_reasons:
            reason = "semantic_ambiguous"
    return CompactSpeechActReduction(
        positive_acceptance=SpeechActKind.POSITIVE_ACCEPTANCE in kinds,
        disqualifying=analysis.has_disqualifying_signal,
        nondecision=SpeechActKind.NONDECISION in kinds,
        attributive_mention=SpeechActKind.ATTRIBUTIVE_MENTION in kinds,
        reported_scope=analysis.reported_scope,
        external_claim_attribution=analysis.external_claim_attribution,
        fail_closed_reason=reason,
    )


@dataclass(slots=True)
class SpeechActReductionCache:
    """Bounded content-transient cache shared by all candidates in one topic."""

    context: EvidenceContext | None = None
    entry_by_id: dict[str, CompactSpeechActReduction] = field(default_factory=dict)
    candidate_by_text: dict[str, CompactSpeechActReduction] = field(default_factory=dict)
    entry_hits: int = 0
    entry_misses: int = 0
    candidate_hits: int = 0
    candidate_misses: int = 0
    classified_codepoints: int = 0
    canonical_entries_by_id: Mapping[str, CanonicalTranscriptEntry] = field(init=False)
    events_by_sequence: Mapping[int, OutcomeEventFact] = field(init=False)
    topic_entry_ids: frozenset[str] = field(init=False)
    duplicate_structural_ids: bool = field(init=False)
    structural_map_builds: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        context = self.context
        if context is None:
            entries: tuple[CanonicalTranscriptEntry, ...] = ()
            events: tuple[OutcomeEventFact, ...] = ()
            topic_ids: tuple[str, ...] = ()
        else:
            entries = context.canonical_entries
            events = context.authoritative_events
            topic_ids = context.topic.entry_ids
        entry_map = {entry.entry_id: entry for entry in entries}
        event_map = {event.sequence_number: event for event in events}
        self.canonical_entries_by_id = MappingProxyType(entry_map)
        self.events_by_sequence = MappingProxyType(event_map)
        self.topic_entry_ids = frozenset(topic_ids)
        self.duplicate_structural_ids = len(entries) != len(entry_map) or len(events) != len(
            event_map
        )
        self.structural_map_builds = 1

    @staticmethod
    def _capacity_failure() -> CompactSpeechActReduction:
        return CompactSpeechActReduction(
            False, True, False, False, False, False, "reduction_cache_capacity"
        )

    def reduce_entry(self, entry: CanonicalTranscriptEntry) -> CompactSpeechActReduction:
        existing = self.entry_by_id.get(entry.entry_id)
        if existing is not None:
            self.entry_hits += 1
            return existing
        if len(self.entry_by_id) >= MAX_CACHED_ENTRY_REDUCTIONS_PER_TOPIC:
            return self._capacity_failure()
        reduction = _compact(entry.text)
        self.entry_by_id[entry.entry_id] = reduction
        self.entry_misses += 1
        self.classified_codepoints += len(entry.text)
        return reduction

    def reduce_candidate_text(self, text: str) -> CompactSpeechActReduction:
        existing = self.candidate_by_text.get(text)
        if existing is not None:
            self.candidate_hits += 1
            return existing
        if len(self.candidate_by_text) >= MAX_CACHED_CANDIDATE_REDUCTIONS_PER_TOPIC:
            return self._capacity_failure()
        reduction = _compact(text)
        self.candidate_by_text[text] = reduction
        self.candidate_misses += 1
        self.classified_codepoints += len(text)
        return reduction


class DecisionSupport(StrEnum):
    AUTHORITATIVE_EVENT = "authoritative_event"
    TRANSCRIPT_SPEECH_ACT = "transcript_speech_act"
    SPEAKER_CLAIM = "speaker_claim"
    UNSUPPORTED = "unsupported"


def _cited_entries(
    candidate: OutcomeCandidate,
    cache: SpeechActReductionCache,
) -> tuple[CanonicalTranscriptEntry, ...]:
    entries = cache.canonical_entries_by_id
    return tuple(
        entries[entry_id]
        for entry_id in candidate.source_entry_ids
        if entry_id in entries and entry_id in cache.topic_entry_ids
    )


def _cited_events(
    candidate: OutcomeCandidate,
    cache: SpeechActReductionCache,
) -> tuple[OutcomeEventFact, ...]:
    events = cache.events_by_sequence
    return tuple(events[event_id] for event_id in candidate.source_event_ids if event_id in events)


def _normalise_subject(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _subject_matches(candidate: OutcomeCandidate, event: OutcomeEventFact) -> bool:
    candidate_subject = _normalise_subject(candidate.text)
    event_subject = _normalise_subject(event.subject)
    return bool(candidate_subject and event_subject and event_subject == candidate_subject)


def _event_is_topic_bound(event: OutcomeEventFact, context: EvidenceContext) -> bool:
    return (
        event.topic_id == context.topic.id
        and bool(event.source_entry_ids)
        and frozenset(event.source_entry_ids) <= frozenset(context.topic.entry_ids)
    )


def _all_cited_events_are_exactly_accepted(
    candidate: OutcomeCandidate,
    context: EvidenceContext,
    cache: SpeechActReductionCache,
) -> bool:
    if candidate.type not in {OutcomeType.DECISION, OutcomeType.AGREEMENT}:
        return False
    cited = _cited_events(candidate, cache)
    return (
        bool(cited)
        and len(cited) == len(candidate.source_event_ids)
        and all(
            _event_is_topic_bound(event, context)
            and event.result in _ACCEPTED_RESULTS
            and _subject_matches(candidate, event)
            for event in cited
        )
    )


def _owner_evidence_kind(candidate: OutcomeCandidate, cache: SpeechActReductionCache) -> str | None:
    owner_user_id = candidate.owner_user_id
    if owner_user_id is None:
        return "none"
    entries = _cited_entries(candidate, cache)
    if any(
        entry.speaker_user_id == owner_user_id and _FIRST_PERSON_OWNER.search(entry.text)
        for entry in entries
    ):
        return "direct"
    owner_name = (candidate.owner_name or "").strip().casefold()
    if owner_name and any(
        owner_name in entry.text.casefold() and _ASSIGNMENT.search(entry.text) for entry in entries
    ):
        return "indirect"
    return None


def _deadline_supported(candidate: OutcomeCandidate, cache: SpeechActReductionCache) -> bool:
    if candidate.deadline is None:
        return True
    deadline = candidate.deadline.strip().casefold()
    return bool(deadline) and any(
        deadline in entry.text.casefold() for entry in _cited_entries(candidate, cache)
    )


def _semantic_reductions(
    candidate: OutcomeCandidate,
    context: EvidenceContext,
    cache: SpeechActReductionCache,
) -> tuple[CompactSpeechActReduction, tuple[CompactSpeechActReduction, ...]]:
    return (
        cache.reduce_candidate_text(candidate.text),
        tuple(cache.reduce_entry(entry) for entry in _cited_entries(candidate, cache)),
    )


def _external_claim(
    candidate: OutcomeCandidate,
    context: EvidenceContext,
    cache: SpeechActReductionCache,
) -> bool:
    _, evidence = _semantic_reductions(candidate, context, cache)
    return any(
        item.external_claim_attribution and item.fail_closed_reason is None for item in evidence
    )


def _decision_support(
    candidate: OutcomeCandidate,
    context: EvidenceContext,
    cache: SpeechActReductionCache,
) -> DecisionSupport:
    if candidate.type not in {OutcomeType.DECISION, OutcomeType.AGREEMENT}:
        return DecisionSupport.TRANSCRIPT_SPEECH_ACT
    if candidate.source_event_ids:
        return (
            DecisionSupport.AUTHORITATIVE_EVENT
            if _all_cited_events_are_exactly_accepted(candidate, context, cache)
            else DecisionSupport.UNSUPPORTED
        )
    if candidate.verification_status is VerificationStatus.NOT_CHECKED:
        return DecisionSupport.SPEAKER_CLAIM
    candidate_reduction, evidence = _semantic_reductions(candidate, context, cache)
    if candidate_reduction.disqualifying or any(item.disqualifying for item in evidence):
        return DecisionSupport.UNSUPPORTED
    if any(item.positive_acceptance for item in evidence):
        return DecisionSupport.TRANSCRIPT_SPEECH_ACT
    return DecisionSupport.UNSUPPORTED


def link_outcome_evidence(
    candidate: OutcomeCandidate,
    context: EvidenceContext,
    cache: SpeechActReductionCache | None = None,
) -> OutcomeCandidate:
    """Recompute status while resolving accepted event authority before text."""

    reduction_cache = cache or SpeechActReductionCache(context)
    if (
        candidate.type in {OutcomeType.DECISION, OutcomeType.AGREEMENT}
        and candidate.source_event_ids
    ):
        authoritative = _all_cited_events_are_exactly_accepted(candidate, context, reduction_cache)
        status = (
            VerificationStatus.EVIDENCE_LINKED if authoritative else VerificationStatus.UNVERIFIED
        )
        requires_confirmation = not authoritative
    elif _external_claim(candidate, context, reduction_cache):
        status = VerificationStatus.NOT_CHECKED
        requires_confirmation = True
    elif _owner_evidence_kind(candidate, reduction_cache) == "indirect":
        status = VerificationStatus.UNCONFIRMED
        requires_confirmation = True
    elif candidate.source_entry_ids or candidate.source_event_ids:
        status = VerificationStatus.EVIDENCE_LINKED
        requires_confirmation = False
    else:
        status = VerificationStatus.UNVERIFIED
        requires_confirmation = True
    return OutcomeCandidate.model_validate(
        {
            **candidate.model_dump(mode="python"),
            "requires_confirmation": requires_confirmation,
            "verification_status": status,
        }
    )


def validate_outcome(
    candidate: OutcomeCandidate,
    context: EvidenceContext,
    cache: SpeechActReductionCache | None = None,
) -> OutcomeValidation:
    """Validate exact structure, then the disjoint event or semantic authority path."""

    errors: list[str] = []
    reduction_cache = cache or SpeechActReductionCache(context)
    if reduction_cache.duplicate_structural_ids:
        errors.append("duplicate_source_entry")
        return OutcomeValidation(False, tuple(errors))
    entries = reduction_cache.canonical_entries_by_id
    events = reduction_cache.events_by_sequence
    topic_ids = reduction_cache.topic_entry_ids
    if not candidate.source_entry_ids and not candidate.source_event_ids:
        errors.append("missing_evidence")
    if set(candidate.source_entry_ids) - entries.keys():
        errors.append("unknown_source_entry")
    if any(
        entry_id in entries and entry_id not in topic_ids for entry_id in candidate.source_entry_ids
    ):
        errors.append("source_outside_topic")
    if set(candidate.source_event_ids) - events.keys():
        errors.append("unknown_source_event")
    if any(
        event_id in events and not _event_is_topic_bound(events[event_id], context)
        for event_id in candidate.source_event_ids
    ):
        errors.append("source_event_outside_topic")
    if not math.isfinite(candidate.confidence):
        errors.append("invalid_confidence")

    event_path = candidate.type in {OutcomeType.DECISION, OutcomeType.AGREEMENT} and bool(
        candidate.source_event_ids
    )
    semantic_external = False
    scan_failure: str | None = None
    if event_path:
        support = _decision_support(candidate, context, reduction_cache)
    else:
        candidate_reduction, evidence_reductions = _semantic_reductions(
            candidate, context, reduction_cache
        )
        for reduction in (candidate_reduction, *evidence_reductions):
            if reduction.fail_closed_reason is not None:
                scan_failure = reduction.fail_closed_reason
                break
        if scan_failure is not None:
            errors.append(scan_failure)
        semantic_external = any(
            item.external_claim_attribution and item.fail_closed_reason is None
            for item in evidence_reductions
        )
        support = _decision_support(candidate, context, reduction_cache)

    if candidate.type not in {OutcomeType.DECISION, OutcomeType.AGREEMENT}:
        if candidate.source_event_ids:
            errors.append("event_evidence_not_allowed")
        if not candidate.source_entry_ids:
            errors.append("transcript_evidence_required")

    if candidate.owner_user_id is not None:
        if candidate.owner_user_id not in context.allowed_participant_ids:
            errors.append("owner_not_allowed")
        expected_name = context.participant_names.get(candidate.owner_user_id)
        if (
            expected_name is not None
            and (candidate.owner_name or "").casefold() != expected_name.casefold()
        ):
            errors.append("owner_name_mismatch")
        if _owner_evidence_kind(candidate, reduction_cache) is None:
            errors.append("owner_evidence_missing")
    if not _deadline_supported(candidate, reduction_cache):
        errors.append("unsupported_deadline")

    if candidate.type in {OutcomeType.DECISION, OutcomeType.AGREEMENT}:
        if support is DecisionSupport.UNSUPPORTED:
            errors.append("decision_not_accepted")
        elif support is DecisionSupport.SPEAKER_CLAIM:
            if not (
                semantic_external
                and candidate.source_entry_ids
                and not candidate.source_event_ids
                and candidate.verification_status is VerificationStatus.NOT_CHECKED
                and candidate.requires_confirmation
                and scan_failure is None
            ):
                errors.append("speaker_claim_not_marked")

    claim_marked = candidate.verification_status is VerificationStatus.NOT_CHECKED
    if claim_marked:
        if candidate.source_event_ids:
            errors.append("speaker_claim_event_evidence_not_allowed")
        if not semantic_external or not candidate.source_entry_ids or scan_failure is not None:
            errors.append("speaker_claim_not_marked")
    elif semantic_external:
        errors.append("speaker_claim_not_marked")

    if candidate.verification_status is VerificationStatus.UNVERIFIED:
        errors.append("evidence_not_linked")
    if candidate.requires_confirmation and candidate.verification_status not in {
        VerificationStatus.UNCONFIRMED,
        VerificationStatus.NOT_CHECKED,
    }:
        errors.append("confirmation_not_marked")
    if not candidate.requires_confirmation and candidate.verification_status in {
        VerificationStatus.UNCONFIRMED,
        VerificationStatus.NOT_CHECKED,
    }:
        errors.append("confirmation_flag_missing")
    return OutcomeValidation(not errors, tuple(dict.fromkeys(errors)))
