from __future__ import annotations

import pytest

from lingxu.domain.errors import SummaryProviderError
from lingxu.transcript.models import CanonicalTranscriptEntry
from lingxu.worker.topics import (
    APPROVED_TOPIC_BASE_URL,
    APPROVED_TOPIC_MODEL,
    APPROVED_TOPIC_MODEL_DIGEST,
    APPROVED_TOPIC_PROVIDER,
    TopicBoundaryCandidate,
    TopicBoundaryConfirmation,
    TopicBoundaryFailure,
    TopicBoundaryProviderPolicy,
    apply_confirmations,
    boundary_window,
    confirm_topic_candidates,
    topic_candidates,
)


def _entry(
    index: int,
    text: str,
    *,
    started_ms: int | None = None,
    ended_ms: int | None = None,
    entry_type: str = "statement",
) -> CanonicalTranscriptEntry:
    start = index * 1_000 if started_ms is None else started_ms
    end = start + 800 if ended_ms is None else ended_ms
    return CanonicalTranscriptEntry(
        entry_id=f"entry-{index:02d}",
        source_entry_ids=(f"source-{index:02d}",),
        speaker_user_id=100 + index % 2,
        speaker_name=f"speaker-{index % 2}",
        started_ms=start,
        ended_ms=end,
        raw_texts=(text,),
        text=text,
        entry_type=entry_type,
        language="zh",
        confidence=0.95,
        utterance_group_id=f"group-{index:02d}",
        normalisation_changes=(),
        asr_tags=(),
        emotion=None,
    )


def entries_with_transition() -> tuple[CanonicalTranscriptEntry, ...]:
    return (
        _entry(0, "我们先确认发布范围。"),
        _entry(1, "接下来讨论部署风险。"),
        _entry(2, "部署需要保留回滚方案。"),
    )


def test_transition_phrase_creates_local_candidate_but_low_confidence_does_not_split() -> None:
    entries = entries_with_transition()

    candidates = topic_candidates(entries)

    assert candidates[0].reason == ("transition_phrase",)
    topics = apply_confirmations(
        entries,
        candidates,
        confirmations=(
            TopicBoundaryConfirmation(
                entry_id=candidates[0].entry_id,
                confirmed=True,
                confidence=0.79,
            ),
        ),
    )
    assert len(topics) == 1
    assert topics[0].entry_ids == tuple(entry.entry_id for entry in entries)


def test_long_silence_creates_candidate_and_valid_confirmation_splits() -> None:
    entries = (
        _entry(0, "发布范围已经确认。", started_ms=0, ended_ms=1_000),
        _entry(1, "部署窗口需要评估。", started_ms=21_000, ended_ms=22_000),
    )

    candidates = topic_candidates(entries)

    assert candidates == (
        TopicBoundaryCandidate(
            entry_id="entry-01",
            reason=("long_silence",),
            deterministic_score=0.45,
        ),
    )
    topics = apply_confirmations(
        entries,
        candidates,
        confirmations=(TopicBoundaryConfirmation("entry-01", confirmed=True, confidence=0.8),),
    )
    assert [topic.entry_ids for topic in topics] == [("entry-00",), ("entry-01",)]
    assert all(topic.method == "model_confirmed" for topic in topics)


def test_overlapping_speaker_prevents_false_long_silence_candidate() -> None:
    entries = (
        _entry(0, "first", started_ms=0, ended_ms=30_000),
        _entry(1, "brief overlap", started_ms=1_000, ended_ms=2_000),
        _entry(2, "still overlapping", started_ms=18_000, ended_ms=19_000),
    )

    assert topic_candidates(entries) == ()


def test_segment_bounds_cover_every_overlapping_entry() -> None:
    entries = (
        _entry(0, "long contribution", started_ms=0, ended_ms=10_000),
        _entry(1, "short overlap", started_ms=1_000, ended_ms=2_000),
        _entry(2, "later short overlap", started_ms=3_000, ended_ms=4_000),
    )

    topics = apply_confirmations(entries, (), confirmations=())

    assert len(topics) == 1
    assert topics[0].started_ms == 0
    assert topics[0].ended_ms == 10_000


def test_continuous_discussion_has_no_boundary_candidate() -> None:
    entries = (
        _entry(0, "发布计划需要审核。"),
        _entry(1, "审核完成后安排发布。"),
        _entry(2, "发布后观察错误率。"),
    )

    assert topic_candidates(entries) == ()
    assert len(apply_confirmations(entries, (), confirmations=())) == 1


def test_repeated_bounded_keywords_can_create_a_shift_candidate() -> None:
    entries = (
        _entry(0, "deployment plan and deployment owner"),
        _entry(1, "deployment risk needs review"),
        _entry(2, "budget limits and budget owner"),
        _entry(3, "budget approval needs review"),
    )

    candidates = topic_candidates(entries)

    assert candidates == (
        TopicBoundaryCandidate(
            entry_id="entry-02",
            reason=("keyword_shift",),
            deterministic_score=0.4,
        ),
    )


def test_model_window_is_bounded_and_excludes_non_summary_entries() -> None:
    entries = tuple(_entry(index, f"statement {index}") for index in range(25))
    entries = (
        *entries[:11],
        _entry(25, "嗯", started_ms=11_100, entry_type="backchannel"),
        *entries[11:],
    )
    candidate = TopicBoundaryCandidate("entry-12", ("long_silence",), 0.45)

    window = boundary_window(entries, candidate)

    assert len(window.before) == 10
    assert len(window.after) == 10
    assert window.after[0].entry_id == candidate.entry_id
    assert all(entry.entry_type == "statement" for entry in (*window.before, *window.after))


class _RecordingProvider:
    def __init__(
        self,
        *,
        fail: bool = False,
        policy: TopicBoundaryProviderPolicy | None = None,
    ) -> None:
        self.fail = fail
        self.calls: list[tuple[int, int]] = []
        self._policy = policy or TopicBoundaryProviderPolicy(
            provider=APPROVED_TOPIC_PROVIDER,
            base_url=APPROVED_TOPIC_BASE_URL,
            model=APPROVED_TOPIC_MODEL,
            expected_model_digest=APPROVED_TOPIC_MODEL_DIGEST,
            allow_remote=True,
        )

    @property
    def topic_boundary_policy(self) -> TopicBoundaryProviderPolicy:
        return self._policy

    async def confirm_topic_boundary(
        self,
        *,
        candidate: TopicBoundaryCandidate,
        before: tuple[CanonicalTranscriptEntry, ...],
        after: tuple[CanonicalTranscriptEntry, ...],
    ) -> TopicBoundaryConfirmation:
        self.calls.append((len(before), len(after)))
        if self.fail:
            raise SummaryProviderError("invalid model JSON")
        return TopicBoundaryConfirmation(candidate.entry_id, confirmed=True, confidence=0.9)


@pytest.mark.asyncio
async def test_provider_failure_has_deterministic_no_split_fallback() -> None:
    entries = entries_with_transition()
    candidates = topic_candidates(entries)
    provider = _RecordingProvider(fail=True)

    confirmations, failures = await confirm_topic_candidates(entries, candidates, provider)
    topics = apply_confirmations(entries, candidates, confirmations)

    assert provider.calls == [(1, 2)]
    assert confirmations == ()
    assert failures == (TopicBoundaryFailure(candidates[0].entry_id, "provider_failure"),)
    assert len(topics) == 1
    assert topics[0].method == "deterministic_fallback"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "failure_code"),
    (
        (
            TopicBoundaryProviderPolicy(
                provider=APPROVED_TOPIC_PROVIDER,
                base_url=APPROVED_TOPIC_BASE_URL,
                model="arbitrary-model",
                expected_model_digest=APPROVED_TOPIC_MODEL_DIGEST,
                allow_remote=True,
            ),
            "model_not_approved",
        ),
        (
            TopicBoundaryProviderPolicy(
                provider=APPROVED_TOPIC_PROVIDER,
                base_url=APPROVED_TOPIC_BASE_URL,
                model=APPROVED_TOPIC_MODEL,
                expected_model_digest="b" * 64,
                allow_remote=True,
            ),
            "digest_not_approved",
        ),
        (
            TopicBoundaryProviderPolicy(
                provider=APPROVED_TOPIC_PROVIDER,
                base_url="https://example.com/v1",
                model=APPROVED_TOPIC_MODEL,
                expected_model_digest=APPROVED_TOPIC_MODEL_DIGEST,
                allow_remote=True,
            ),
            "endpoint_not_approved",
        ),
    ),
)
async def test_unapproved_provider_policy_fails_before_context_or_provider_call(
    policy: TopicBoundaryProviderPolicy,
    failure_code: str,
) -> None:
    entries = entries_with_transition()
    candidates = topic_candidates(entries)
    provider = _RecordingProvider(policy=policy)

    confirmations, failures = await confirm_topic_candidates(entries, candidates, provider)

    assert confirmations == ()
    assert failures == (TopicBoundaryFailure(candidates[0].entry_id, failure_code),)
    assert provider.calls == []


def test_invalid_or_rejected_confirmation_never_splits() -> None:
    entries = entries_with_transition()
    candidates = topic_candidates(entries)

    topics = apply_confirmations(
        entries,
        candidates,
        confirmations=(
            TopicBoundaryConfirmation("unknown", confirmed=True, confidence=1.0),
            TopicBoundaryConfirmation(
                candidates[0].entry_id,
                confirmed=False,
                confidence=1.0,
            ),
        ),
    )

    assert len(topics) == 1
