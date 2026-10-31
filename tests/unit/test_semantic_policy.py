from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from lingxu.worker.semantic_policy import (
    MAX_SPEECH_ACT_SIGNALS,
    MAX_SPEECH_ACT_TEXT_CODEPOINTS,
    SpeechActKind,
    SpeechActLanguage,
    classify_speech_acts,
    contains_rejection,
)


@pytest.mark.parametrize(
    ("text", "kind"),
    (
        ("approved plan", SpeechActKind.ATTRIBUTIVE_MENTION),
        ("accepted format", SpeechActKind.ATTRIBUTIVE_MENTION),
        ("approved proposal", SpeechActKind.ATTRIBUTIVE_MENTION),
        ("accepted deployment", SpeechActKind.ATTRIBUTIVE_MENTION),
        ("approved decision", SpeechActKind.ATTRIBUTIVE_MENTION),
        ("approved yesterday", SpeechActKind.AMBIGUOUS),
        ("Alice approved", SpeechActKind.AMBIGUOUS),
        ("We discussed an approved plan", SpeechActKind.ATTRIBUTIVE_MENTION),
        ("We reviewed an accepted format", SpeechActKind.ATTRIBUTIVE_MENTION),
        ("A consensus proposal was discussed", SpeechActKind.ATTRIBUTIVE_MENTION),
        ("The group reached consensus to deploy", SpeechActKind.POSITIVE_ACCEPTANCE),
        ("Consensus was reached on deployment", SpeechActKind.POSITIVE_ACCEPTANCE),
        ("Agreed to deploy", SpeechActKind.POSITIVE_ACCEPTANCE),
        ("已批准的方案", SpeechActKind.ATTRIBUTIVE_MENTION),
        ("讨论已批准的方案", SpeechActKind.ATTRIBUTIVE_MENTION),
        ("审阅通过的提案", SpeechActKind.ATTRIBUTIVE_MENTION),
        ("讨论共识方案", SpeechActKind.ATTRIBUTIVE_MENTION),
        ("团队达成共识并部署", SpeechActKind.POSITIVE_ACCEPTANCE),
        ("已同意部署", SpeechActKind.POSITIVE_ACCEPTANCE),
    ),
)
def test_classifier_distinguishes_decisions_from_descriptions(
    text: str, kind: SpeechActKind
) -> None:
    analysis = classify_speech_acts(text)

    assert kind in {signal.kind for signal in analysis.signals}
    assert analysis.stats.visited_tokens == analysis.stats.emitted_tokens


@pytest.mark.parametrize(
    ("text", "required", "forbidden"),
    (
        ("We did not agree to deploy", SpeechActKind.NEGATED_ACCEPTANCE, None),
        ("We decided not to deploy", SpeechActKind.POSITIVE_ACCEPTANCE, None),
        ("The proposal was rejected", SpeechActKind.AFFIRMATIVE_REJECTION, None),
        ("We did not reject the proposal", SpeechActKind.NONDECISION, None),
        (
            "We did not reject the proposal; we agreed to deploy",
            SpeechActKind.POSITIVE_ACCEPTANCE,
            SpeechActKind.AFFIRMATIVE_REJECTION,
        ),
        ("我们不同意部署", SpeechActKind.NEGATED_ACCEPTANCE, None),
        ("我们决定不部署", SpeechActKind.POSITIVE_ACCEPTANCE, None),
        ("方案被拒绝", SpeechActKind.AFFIRMATIVE_REJECTION, None),
        ("我们不拒绝方案", SpeechActKind.NONDECISION, None),
    ),
)
def test_classifier_preserves_directional_speech_act(
    text: str,
    required: SpeechActKind,
    forbidden: SpeechActKind | None,
) -> None:
    kinds = {signal.kind for signal in classify_speech_acts(text).signals}

    assert required in kinds
    if forbidden is not None:
        assert forbidden not in kinds


@pytest.mark.parametrize(
    "text",
    (
        "We agreed not",
        "We agreed not to",
        "我们同意不",
        "According to",
        "According to the supplier",
        "据供应商",
        "据供应商称",
        "The supplier said",
        "供应商表示",
        "withdraw",
        "voted",
    ),
)
def test_unresolved_directional_state_is_ambiguous(text: str) -> None:
    analysis = classify_speech_acts(text)

    assert analysis.has_disqualifying_signal
    assert any(signal.kind is SpeechActKind.AMBIGUOUS for signal in analysis.signals)


@pytest.mark.parametrize(
    "text",
    (
        "We agreed not to deploy",
        "我们同意不部署",
    ),
)
def test_complete_negative_action_remains_positive(text: str) -> None:
    analysis = classify_speech_acts(text)

    assert analysis.has_positive_acceptance
    assert not analysis.has_disqualifying_signal


@pytest.mark.parametrize(
    ("text", "reported", "external"),
    (
        ('"We agreed"', True, False),
        ("I said we agreed earlier", True, False),
        ("The supplier said they approved", True, True),
        ('"We approved," the supplier said', True, True),
        ("我报告一个风险", True, False),
        ("据供应商称方案已批准", True, True),
    ),
)
def test_reported_scope_is_independent_from_external_attribution(
    text: str, reported: bool, external: bool
) -> None:
    analysis = classify_speech_acts(text)

    assert analysis.reported_scope is reported
    assert analysis.external_claim_attribution is external


@pytest.mark.parametrize(
    ("text", "external", "expected_kind"),
    (
        ("We agreed, the supplier said", True, SpeechActKind.ATTRIBUTIVE_MENTION),
        ("我们同意\uff0c供应商表示", True, SpeechActKind.ATTRIBUTIVE_MENTION),
        ("We agreed, I said", False, SpeechActKind.ATTRIBUTIVE_MENTION),
        ("We agreed, Alice said", False, SpeechActKind.AMBIGUOUS),
    ),
)
def test_late_reporting_tail_rewrites_pending_direct_signal_without_rescan(
    text: str,
    external: bool,
    expected_kind: SpeechActKind,
) -> None:
    analysis = classify_speech_acts(text)

    assert analysis.reported_scope
    assert analysis.external_claim_attribution is external
    assert not analysis.has_positive_acceptance
    assert analysis.signals[-1].kind is expected_kind


@pytest.mark.parametrize(
    "text",
    (
        '"We agreed',
        "“我们同意'",
        '「『“\u2018"we agreed"\u2019”』」',
    ),
)
def test_quote_failures_are_ambiguous(text: str) -> None:
    analysis = classify_speech_acts(text)

    assert analysis.has_disqualifying_signal
    assert any(signal.kind is SpeechActKind.AMBIGUOUS for signal in analysis.signals)


@dataclass
class _Observer:
    cursor_ranges: list[tuple[int, int]] = field(default_factory=list)
    emitted: list[int] = field(default_factory=list)
    visited: list[int] = field(default_factory=list)
    lookaheads: list[int] = field(default_factory=list)

    def cursor_advanced(self, old: int, new: int) -> None:
        self.cursor_ranges.append((old, new))

    def token_emitted(self, token_index: int, start: int, end: int) -> None:
        del start, end
        self.emitted.append(token_index)

    def token_visited(self, token_index: int) -> None:
        self.visited.append(token_index)

    def trie_lookahead(self, start: int, comparisons: int) -> None:
        del start
        self.lookaheads.append(comparisons)


@pytest.mark.parametrize("repeat", (1, 10, 100, 1_000))
def test_scanner_structurally_visits_each_token_once(repeat: int) -> None:
    text = ("agree,同意;" * repeat).rstrip(";")
    observer = _Observer()

    analysis = classify_speech_acts(text, _observer=observer)

    assert analysis.stats.visited_tokens == analysis.stats.emitted_tokens
    assert observer.emitted == observer.visited == list(range(len(observer.emitted)))
    assert sum(end - start for start, end in observer.cursor_ranges) == len(text)
    assert all(
        left[1] == right[0]
        for left, right in zip(observer.cursor_ranges, observer.cursor_ranges[1:], strict=False)
    )
    assert max(observer.lookaheads, default=0) <= 8


def test_scanner_growth_is_linear_by_structural_counters() -> None:
    base = "We approved deployment; 团队同意部署。"
    once = classify_speech_acts(base)
    twice = classify_speech_acts(base + base)

    assert twice.stats.visited_tokens <= once.stats.visited_tokens * 2 + 2
    assert twice.stats.bounded_lookahead_steps <= once.stats.bounded_lookahead_steps * 2 + 8


def test_input_limit_fails_without_scanning_prefix() -> None:
    at_limit = classify_speech_acts("x" * MAX_SPEECH_ACT_TEXT_CODEPOINTS)
    above_limit = classify_speech_acts("x" * (MAX_SPEECH_ACT_TEXT_CODEPOINTS + 1))

    assert at_limit.stats.emitted_tokens > 0
    assert above_limit.exceeded_input_limit
    assert above_limit.stats.emitted_tokens == 0
    assert len(above_limit.signals) == 1
    assert above_limit.signals[0].reason == "input_too_long"


def test_signal_overflow_scans_to_end_and_returns_one_compact_ambiguity() -> None:
    text = "; ".join("We agreed" for _ in range(MAX_SPEECH_ACT_SIGNALS + 10))
    observer = _Observer()

    analysis = classify_speech_acts(text, _observer=observer)

    assert analysis.signal_overflow
    assert len(analysis.signals) == 1
    assert analysis.signals[0].kind is SpeechActKind.AMBIGUOUS
    assert analysis.signals[0].reason == "signal_capacity"
    assert sum(end - start for start, end in observer.cursor_ranges) == len(text)


def test_compatibility_wrapper_owns_no_separate_positive_logic() -> None:
    assert not contains_rejection("We discussed an approved plan")
    assert contains_rejection("We did not agree")
    assert classify_speech_acts("plain text").signals == ()
    assert classify_speech_acts("plain text").signals == ()
    assert SpeechActLanguage.ENGLISH.value == "english"
