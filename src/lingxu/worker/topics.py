"""Deterministic-first topic boundary candidates and confirmation-only splits."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal, Protocol

from lingxu.domain.errors import SummaryProviderError
from lingxu.transcript.models import CanonicalTranscriptEntry

_EXCLUDED_ENTRY_TYPES = frozenset(
    {"backchannel", "noise", "excluded", "privacy_control", "privacy-control"}
)
_TRANSITION_PHRASES = (
    "接下来",
    "下一项",
    "下一个议题",
    "换个话题",
    "转到",
    "现在讨论",
    "moving on",
    "next topic",
    "next item",
    "turning to",
)
_LONG_SILENCE_MS = 15_000
_CONFIDENCE_THRESHOLD = 0.8
_WINDOW_LIMIT = 10
_ASCII_WORD = re.compile(r"[a-z][a-z0-9_-]{3,}")
_CJK_RUN = re.compile(r"[\u3400-\u9fff]{2,}")

APPROVED_TOPIC_PROVIDER = "deepseek"
APPROVED_TOPIC_BASE_URL = "https://api.deepseek.com"
APPROVED_TOPIC_MODEL: Literal["deepseek-flash"] = "deepseek-flash"
APPROVED_TOPIC_MODEL_DIGEST = "4a87243abc84b3147f3e9c2fc3416e539ae1ebd634706612996fc62018d32a0f"
TOPIC_CANDIDATE_CONFIG_VERSION = 5


@dataclass(frozen=True, slots=True)
class TopicBoundaryCandidate:
    entry_id: str
    reason: tuple[str, ...]
    deterministic_score: float


@dataclass(frozen=True, slots=True)
class TopicBoundaryConfirmation:
    entry_id: str
    confirmed: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class TopicBoundaryFailure:
    entry_id: str
    code: str


@dataclass(frozen=True, slots=True)
class TopicBoundaryProviderPolicy:
    provider: str
    base_url: str
    model: str
    expected_model_digest: str
    allow_remote: bool


@dataclass(frozen=True, slots=True)
class TopicBoundaryWindow:
    before: tuple[CanonicalTranscriptEntry, ...]
    after: tuple[CanonicalTranscriptEntry, ...]


@dataclass(frozen=True, slots=True)
class TopicSegment:
    id: str
    title: str
    entry_ids: tuple[str, ...]
    started_ms: int
    ended_ms: int
    method: str


class TopicBoundaryProvider(Protocol):
    @property
    def topic_boundary_policy(self) -> TopicBoundaryProviderPolicy: ...

    async def confirm_topic_boundary(
        self,
        *,
        candidate: TopicBoundaryCandidate,
        before: tuple[CanonicalTranscriptEntry, ...],
        after: tuple[CanonicalTranscriptEntry, ...],
    ) -> TopicBoundaryConfirmation: ...


def topic_boundary_policy_failure(provider: TopicBoundaryProvider) -> str | None:
    """Return a content-free failure code unless the exact owner-approved path is proven."""

    try:
        policy: object = provider.topic_boundary_policy
    except (AttributeError, TypeError, ValueError):
        return "policy_unproven"
    if not isinstance(policy, TopicBoundaryProviderPolicy):
        return "provider_not_approved"
    if policy.provider == APPROVED_TOPIC_PROVIDER:
        if policy.model != APPROVED_TOPIC_MODEL:
            return "model_not_approved"
        if policy.expected_model_digest.lower() != APPROVED_TOPIC_MODEL_DIGEST:
            return "digest_not_approved"
        if policy.base_url.rstrip("/") != APPROVED_TOPIC_BASE_URL or not policy.allow_remote:
            return "endpoint_not_approved"
        return None
    return "provider_not_approved"


def is_summary_eligible(entry: CanonicalTranscriptEntry) -> bool:
    """Keep control/noise classifications outside every topic/model boundary."""

    return entry.entry_type not in _EXCLUDED_ENTRY_TYPES and bool(entry.text.strip())


def _eligible_entries(
    entries: tuple[CanonicalTranscriptEntry, ...],
) -> tuple[CanonicalTranscriptEntry, ...]:
    return tuple(
        sorted(
            (entry for entry in entries if is_summary_eligible(entry)),
            key=lambda entry: (entry.started_ms, entry.ended_ms, entry.entry_id),
        )
    )


def _keyword_signature(text: str) -> frozenset[str]:
    """Extract bounded lexical signals without a model or unbounded corpus state."""

    normalised = text.casefold()
    tokens = set(_ASCII_WORD.findall(normalised))
    for match in _CJK_RUN.finditer(normalised):
        run = match.group(0)
        tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return frozenset(tokens)


def _has_keyword_shift(
    entries: tuple[CanonicalTranscriptEntry, ...],
    index: int,
) -> bool:
    """Require repetition on both sides inside a fixed two-entry local window."""

    if index < 2 or index + 1 >= len(entries):
        return False
    before = _keyword_signature(entries[index - 2].text) & _keyword_signature(
        entries[index - 1].text
    )
    after = _keyword_signature(entries[index].text) & _keyword_signature(entries[index + 1].text)
    return bool(before and after and before.isdisjoint(after))


def topic_candidates(
    entries: tuple[CanonicalTranscriptEntry, ...],
) -> tuple[TopicBoundaryCandidate, ...]:
    """Produce conservative local boundary candidates; never split by itself."""

    eligible = _eligible_entries(entries)
    candidates: list[TopicBoundaryCandidate] = []
    running_prior_end = eligible[0].ended_ms if eligible else 0
    for index, entry in enumerate(eligible[1:], start=1):
        reasons: list[str] = []
        lowered = entry.text.casefold()
        if any(phrase in lowered for phrase in _TRANSITION_PHRASES):
            reasons.append("transition_phrase")
        if entry.started_ms - running_prior_end >= _LONG_SILENCE_MS:
            reasons.append("long_silence")
        if _has_keyword_shift(eligible, index):
            reasons.append("keyword_shift")
        running_prior_end = max(running_prior_end, entry.ended_ms)
        if not reasons:
            continue
        weights = {
            "transition_phrase": 0.55,
            "long_silence": 0.45,
            "keyword_shift": 0.4,
        }
        score = min(0.95, sum(weights[reason] for reason in reasons))
        candidates.append(
            TopicBoundaryCandidate(
                entry_id=entry.entry_id,
                reason=tuple(reasons),
                deterministic_score=round(score, 2),
            )
        )
    return tuple(candidates)


def boundary_window(
    entries: tuple[CanonicalTranscriptEntry, ...],
    candidate: TopicBoundaryCandidate,
) -> TopicBoundaryWindow:
    """Return at most ten eligible entries on either side of one candidate."""

    eligible = _eligible_entries(entries)
    try:
        index = next(
            index for index, entry in enumerate(eligible) if entry.entry_id == candidate.entry_id
        )
    except StopIteration as exc:
        raise ValueError("topic candidate does not identify an eligible entry") from exc
    if index == 0:
        raise ValueError("the first eligible entry cannot be a topic boundary")
    return TopicBoundaryWindow(
        before=eligible[max(0, index - _WINDOW_LIMIT) : index],
        after=eligible[index : index + _WINDOW_LIMIT],
    )


async def confirm_topic_candidates(
    entries: tuple[CanonicalTranscriptEntry, ...],
    candidates: tuple[TopicBoundaryCandidate, ...],
    provider: TopicBoundaryProvider,
) -> tuple[tuple[TopicBoundaryConfirmation, ...], tuple[TopicBoundaryFailure, ...]]:
    """Ask only confirm/reject questions and fail closed per candidate."""

    confirmations: list[TopicBoundaryConfirmation] = []
    failures: list[TopicBoundaryFailure] = []
    policy_failure = topic_boundary_policy_failure(provider)
    if policy_failure is not None:
        return (), tuple(
            TopicBoundaryFailure(candidate.entry_id, policy_failure) for candidate in candidates
        )
    for candidate in candidates:
        window = boundary_window(entries, candidate)
        try:
            confirmation = await provider.confirm_topic_boundary(
                candidate=candidate,
                before=window.before,
                after=window.after,
            )
        except SummaryProviderError:
            failures.append(TopicBoundaryFailure(candidate.entry_id, "provider_failure"))
            continue
        if confirmation.entry_id != candidate.entry_id:
            failures.append(TopicBoundaryFailure(candidate.entry_id, "candidate_mismatch"))
            continue
        confirmations.append(confirmation)
    return tuple(confirmations), tuple(failures)


def apply_confirmations(
    entries: tuple[CanonicalTranscriptEntry, ...],
    candidates: tuple[TopicBoundaryCandidate, ...],
    confirmations: tuple[TopicBoundaryConfirmation, ...],
) -> tuple[TopicSegment, ...]:
    """Split only at exact, affirmative, finite, high-confidence confirmations."""

    eligible = _eligible_entries(entries)
    if not eligible:
        return ()
    candidate_ids = {candidate.entry_id for candidate in candidates}
    confirmed_ids = {
        confirmation.entry_id
        for confirmation in confirmations
        if confirmation.entry_id in candidate_ids
        and confirmation.confirmed
        and math.isfinite(confirmation.confidence)
        and confirmation.confidence >= _CONFIDENCE_THRESHOLD
    }
    groups: list[list[CanonicalTranscriptEntry]] = [[]]
    for entry in eligible:
        if groups[-1] and entry.entry_id in confirmed_ids:
            groups.append([])
        groups[-1].append(entry)
    method = "model_confirmed" if len(groups) > 1 else "deterministic_fallback"
    return tuple(
        TopicSegment(
            id=f"topic-{index:03d}",
            title=f"Topic {index}",
            entry_ids=tuple(entry.entry_id for entry in group),
            started_ms=min(entry.started_ms for entry in group),
            ended_ms=max(entry.ended_ms for entry in group),
            method=method,
        )
        for index, group in enumerate(groups, start=1)
    )
