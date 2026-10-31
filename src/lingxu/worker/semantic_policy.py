"""Bounded streaming speech-act and object-level outcome semantics."""

from __future__ import annotations

import hashlib
import re
from collections import deque
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

MAX_SPEECH_ACT_TEXT_CODEPOINTS = 16_384
MAX_SPEECH_ACT_SIGNALS = 64
MAX_QUOTE_DEPTH = 4
SPEECH_ACT_POLICY_REVISION = hashlib.sha256(b"lingxu-speech-act-policy-task10r-v2").hexdigest()


class SpeechActKind(StrEnum):
    POSITIVE_ACCEPTANCE = "positive_acceptance"
    NEGATED_ACCEPTANCE = "negated_acceptance"
    AFFIRMATIVE_REJECTION = "affirmative_rejection"
    NONDECISION = "nondecision"
    ATTRIBUTIVE_MENTION = "attributive_mention"
    AMBIGUOUS = "ambiguous"


class SpeechActLanguage(StrEnum):
    ENGLISH = "english"
    CHINESE = "chinese"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SpeechActSignal:
    kind: SpeechActKind
    language: SpeechActLanguage
    clause_index: int
    start: int
    end: int
    reason: str


@dataclass(frozen=True, slots=True)
class SpeechActScanStats:
    input_codepoints: int
    emitted_tokens: int
    visited_tokens: int
    clauses: int
    bounded_lookahead_steps: int


@dataclass(frozen=True, slots=True)
class SpeechActAnalysis:
    signals: tuple[SpeechActSignal, ...]
    proposal_reference: bool
    reported_scope: bool
    external_claim_attribution: bool
    exceeded_input_limit: bool
    signal_overflow: bool
    stats: SpeechActScanStats

    @property
    def has_positive_acceptance(self) -> bool:
        return any(signal.kind is SpeechActKind.POSITIVE_ACCEPTANCE for signal in self.signals)

    @property
    def has_disqualifying_signal(self) -> bool:
        return self.exceeded_input_limit or any(
            signal.kind
            in {
                SpeechActKind.NEGATED_ACCEPTANCE,
                SpeechActKind.AFFIRMATIVE_REJECTION,
                SpeechActKind.AMBIGUOUS,
            }
            for signal in self.signals
        )


class _SpeechActScanObserver(Protocol):
    def cursor_advanced(self, old: int, new: int) -> None: ...

    def token_emitted(self, token_index: int, start: int, end: int) -> None: ...

    def token_visited(self, token_index: int) -> None: ...

    def trie_lookahead(self, start: int, comparisons: int) -> None: ...


class _TokenKind(StrEnum):
    WORD = "word"
    CJK = "cjk"
    QUOTE_OPEN = "quote_open"
    QUOTE_CLOSE = "quote_close"
    BOUNDARY = "boundary"
    PUNCT = "punct"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class _Token:
    kind: _TokenKind
    key: str
    start: int
    end: int
    language: SpeechActLanguage


_EN_DIRECT = frozenset({"i", "we"})
_EN_COLLECTIVE = frozenset(
    {
        "team",
        "group",
        "committee",
        "members",
        "participants",
        "attendees",
        "meeting",
        "host",
        "chair",
    }
)
_EN_THIRD_PARTY = frozenset(
    {
        "he",
        "she",
        "they",
        "supplier",
        "vendor",
        "manager",
        "lawyer",
        "customer",
        "client",
        "colleague",
        "supervisor",
        "friend",
        "consultant",
        "reporter",
        "source",
    }
)
_EN_OBJECT = frozenset(
    {
        "proposal",
        "motion",
        "request",
        "plan",
        "option",
        "vote",
        "release",
        "deployment",
        "change",
        "decision",
    }
)
_EN_ATTRIBUTIVE_NOUNS = _EN_OBJECT | frozenset({"format", "process", "method", "version"})
_EN_DETERMINERS = frozenset(
    {"a", "an", "the", "this", "that", "these", "those", "our", "my", "their"}
)
_EN_ACCEPT_ACTIVE = frozenset(
    {
        "agree",
        "agrees",
        "agreed",
        "decide",
        "decides",
        "decided",
        "approve",
        "approves",
        "approved",
        "accept",
        "accepts",
        "accepted",
        "confirm",
        "confirms",
        "confirmed",
    }
)
_EN_ACCEPT_RESULT = frozenset({"accepted", "approved", "adopted", "passed", "carried"})
_EN_REJECTION = frozenset(
    {
        "reject",
        "rejects",
        "rejected",
        "rejecting",
        "rejection",
        "rejections",
        "decline",
        "declines",
        "declined",
        "declining",
        "fail",
        "fails",
        "failed",
        "failing",
        "failure",
        "failures",
        "deny",
        "denies",
        "denied",
        "denying",
        "denial",
        "denials",
        "prohibit",
        "prohibits",
        "prohibited",
        "prohibiting",
        "prohibition",
        "prohibitions",
        "defeat",
        "defeats",
        "defeated",
        "defeating",
        "disapprove",
        "disapproves",
        "disapproved",
        "disapproving",
        "disapproval",
        "disapprovals",
        "veto",
        "vetoes",
        "vetoed",
        "vetoing",
        "overrule",
        "overrules",
        "overruled",
        "overruling",
        "lose",
        "loses",
        "lost",
        "losing",
        "loss",
        "losses",
        "refuse",
        "refuses",
        "refused",
        "refusing",
        "refusal",
        "refusals",
        "pending",
        "withdrawal",
        "withdrawals",
    }
)
_EN_PROPOSAL_REVERSAL = frozenset(
    {
        "withdraw",
        "withdraws",
        "withdrawing",
        "withdrawn",
        "withdrew",
        "cancel",
        "cancels",
        "canceled",
        "cancelled",
        "canceling",
        "cancelling",
        "cancellation",
        "cancellations",
    }
)
_EN_NEGATORS = frozenset({"no", "none", "not", "never", "without", "cannot"})
_EN_FOCUS = frozenset({"only", "merely", "just"})
_EN_BRIDGES = frozenset(
    {
        "am",
        "are",
        "be",
        "been",
        "being",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "is",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "was",
        "were",
        "will",
        "would",
        "already",
        "also",
        "clearly",
        "directly",
        "explicitly",
        "formally",
        "personally",
        "previously",
        "really",
        "still",
        "to",
        "that",
        "then",
        "yet",
        "unanimously",
        "after",
        "review",
        "careful",
        "technical",
        "operational",
        "all",
        "of",
        "us",
    }
)
_EN_REPORTING = frozenset(
    {
        "say",
        "says",
        "saying",
        "said",
        "tell",
        "tells",
        "telling",
        "told",
        "report",
        "reports",
        "reported",
        "reporting",
        "claim",
        "claims",
        "claimed",
        "claiming",
    }
)
_EN_IMPERSONAL = frozenset({"reportedly"})
_EN_ADVERSATIVES = frozenset({"but", "however", "yet", "nevertheless"})

_ZH_DIRECT = frozenset({"我", "我们", "本人"})
_ZH_COLLECTIVE = frozenset(
    {"团队", "小组", "委员会", "与会者", "参会者", "大家", "会议", "主持人", "主席"}
)
_ZH_THIRD_PARTY = frozenset(
    {
        "他",
        "她",
        "他们",
        "她们",
        "供应商",
        "厂商",
        "经理",
        "主管",
        "律师",
        "客户",
        "同事",
        "朋友",
        "顾问",
        "项目顾问",
        "我司",
        "报道方",
        "消息源",
    }
)
_ZH_OBJECT = frozenset(
    {"方案", "提案", "动议", "请求", "计划", "选项", "投票", "发布", "部署", "变更", "决定"}
)
_ZH_ACCEPT = frozenset({"同意", "决定", "确认", "批准", "接受", "通过"})
_ZH_REJECTION = frozenset({"禁止", "拒绝", "反对", "驳回", "否决", "否定"})
_ZH_REJECTION_RESULTS = frozenset(
    {"不予", "未获", "未通过", "不通过", "不得", "不能", "不许", "不要", "落选", "败北", "无共识"}
)
_ZH_PROPOSAL_REVERSAL = frozenset({"撤回", "取消"})
_ZH_NEGATORS = frozenset({"没有", "没", "不", "未", "无", "非", "否", "别", "勿", "莫"})
_ZH_FOCUS = frozenset({"不仅", "不但", "不只是"})
_ZH_BRIDGES = frozenset(
    {
        "正在",
        "已经",
        "曾经",
        "将要",
        "可以",
        "应该",
        "应当",
        "必须",
        "在",
        "已",
        "曾",
        "将",
        "会",
        "能",
        "明确",
        "清楚",
        "正式",
        "公开",
        "亲自",
        "如实",
        "直接",
        "主动",
        "同时",
        "同样",
        "再次",
        "仍然",
        "仍",
        "也",
        "还",
        "再",
        "向",
        "对",
        "给",
        "跟",
        "的",
        "被",
        "获",
        "获得",
        "一致",
        "后",
        "讨论",
        "审阅",
        "地",
    }
)
_ZH_REPORTING = frozenset(
    {"说", "说道", "表示", "声称", "宣称", "告诉", "告知", "报告", "报道称", "报告称", "转述"}
)
_ZH_IMPERSONAL = frozenset({"据说", "听说", "据称", "有报道称"})
_ZH_ADVERSATIVES = frozenset({"但", "但是", "然而", "不过", "却"})
_ZH_ATTRIBUTIVE_NOUNS = frozenset({"方案", "格式", "流程", "方法", "版本"})

_ZH_LEXEMES = frozenset(
    _ZH_DIRECT
    | _ZH_COLLECTIVE
    | _ZH_THIRD_PARTY
    | _ZH_OBJECT
    | _ZH_ACCEPT
    | _ZH_REJECTION
    | _ZH_REJECTION_RESULTS
    | _ZH_PROPOSAL_REVERSAL
    | _ZH_NEGATORS
    | _ZH_FOCUS
    | _ZH_BRIDGES
    | _ZH_REPORTING
    | _ZH_IMPERSONAL
    | _ZH_ADVERSATIVES
    | _ZH_ATTRIBUTIVE_NOUNS
    | {
        "达成",
        "形成",
        "共识",
        "据",
        "称",
        "我的",
        "我们的",
        "我同事",
        "我们同事",
        "我经理",
        "我们经理",
        "我主管",
        "我们主管",
        "我律师",
        "我们律师",
        "我朋友",
        "我们朋友",
        "我供应商",
        "我们供应商",
        "我项目顾问",
        "我们项目顾问",
        "并",
        "且",
        "一个",
        "该",
        "此",
    }
)
_MAX_ZH_LEXEME = max(map(len, _ZH_LEXEMES))

_STRONG = frozenset(".;:!?。\uff1b\uff1a\uff01\uff1f\r\n")
_NEUTRAL = frozenset(",\uff0c、()\uff08\uff09-\u2013—")
_PAIRED_QUOTES = {"“": "”", "\u2018": "\u2019", "「": "」", "『": "』"}
_QUOTE_CLOSERS = frozenset(_PAIRED_QUOTES.values())


def _is_cjk(character: str) -> bool:
    value = ord(character)
    return 0x3400 <= value <= 0x9FFF


def _language(saw_en: bool, saw_zh: bool) -> SpeechActLanguage:
    if saw_en and saw_zh:
        return SpeechActLanguage.MIXED
    if saw_en:
        return SpeechActLanguage.ENGLISH
    if saw_zh:
        return SpeechActLanguage.CHINESE
    return SpeechActLanguage.UNKNOWN


def _subject_class(key: str) -> str:
    if key in _EN_DIRECT or key in _ZH_DIRECT:
        return "direct"
    if key in _EN_COLLECTIVE or key in _ZH_COLLECTIVE:
        return "collective"
    if (
        key in _EN_THIRD_PARTY
        or key in _ZH_THIRD_PARTY
        or any(key.endswith(role) for role in _ZH_THIRD_PARTY if role != "我司")
    ):
        return "third_party"
    if key in _EN_OBJECT or key in _ZH_OBJECT:
        return "object"
    return "unknown"


def _is_bridge(key: str) -> bool:
    return key in _EN_BRIDGES or key in _ZH_BRIDGES or key in _EN_DETERMINERS


def _is_contraction_negator(key: str) -> bool:
    return len(key) > 3 and key.endswith("n't") and key[:-3].isalpha()


@dataclass(slots=True)
class _ClauseState:
    index: int
    semantic_count: int = 0
    subject: str = "absent"
    subject_token: str = ""
    predicate_clean: bool = True
    pending_negator: str | None = None
    negator_clean: bool = True
    pending_focus_not: bool = False
    proposal_seen: bool = False
    pending_proposal_reversal: tuple[_Token, bool] | None = None
    positive_seen: bool = False
    reported_scope: bool = False
    external: bool = False
    saw_en: bool = False
    saw_zh: bool = False
    according_state: int = 0
    pending_consensus: _Token | None = None
    pending_voted: bool = False
    pending_acceptance: _Token | None = None
    pending_acceptance_bridge: bool = False
    pending_reporting: _Token | None = None
    last_keys: deque[str] | None = None

    def __post_init__(self) -> None:
        if self.last_keys is None:
            self.last_keys = deque(maxlen=5)


class _Scanner:
    def __init__(self, text: str, observer: _SpeechActScanObserver | None) -> None:
        self.text = text
        self.observer = observer
        self.cursor = 0
        self.emitted = 0
        self.visited = 0
        self.lookahead = 0
        self.clause = _ClauseState(0)
        self.clauses = 0
        self.signals: list[SpeechActSignal] = []
        self.signal_overflow = False
        self.quote_stack: list[str] = []
        self.straight_quote: str | None = None
        self.quote_error: str | None = None
        self.proposal_reference = False
        self.reported_scope = False
        self.external = False

    def _advance(self, new: int) -> None:
        old = self.cursor
        self.cursor = new
        if self.observer is not None:
            self.observer.cursor_advanced(old, new)

    def _emit(self, token: _Token) -> None:
        index = self.emitted
        self.emitted += 1
        if self.observer is not None:
            self.observer.token_emitted(index, token.start, token.end)
        self.visited += 1
        if self.observer is not None:
            self.observer.token_visited(index)
        self._visit(token)

    def _add_signal(self, token: _Token, kind: SpeechActKind, reason: str) -> None:
        if self.signal_overflow:
            return
        if len(self.signals) >= MAX_SPEECH_ACT_SIGNALS:
            self.signal_overflow = True
            self.signals.clear()
            return
        self.signals.append(
            SpeechActSignal(kind, token.language, self.clause.index, token.start, token.end, reason)
        )

    def _mark_ambiguous(self, token: _Token, reason: str) -> None:
        self._add_signal(token, SpeechActKind.AMBIGUOUS, reason)

    def _resolve_pending_acceptance(self, token: _Token | None) -> bool:
        """Resolve one-token shorthand without treating an adjective as a decision."""

        state = self.clause
        pending = state.pending_acceptance
        if pending is None:
            return False
        if token is None:
            if state.pending_acceptance_bridge:
                self._mark_ambiguous(pending, "incomplete_acceptance")
            else:
                self._add_signal(pending, SpeechActKind.POSITIVE_ACCEPTANCE, "meeting_shorthand")
                state.positive_seen = True
            state.pending_acceptance = None
            state.pending_acceptance_bridge = False
            return False
        key = token.key
        if key in _EN_ATTRIBUTIVE_NOUNS or key in _ZH_ATTRIBUTIVE_NOUNS:
            self._add_signal(pending, SpeechActKind.ATTRIBUTIVE_MENTION, "descriptive_acceptance")
            state.pending_acceptance = None
            state.pending_acceptance_bridge = False
            return False
        if key in (_EN_DETERMINERS - {"that"}) or key == "的":
            state.pending_acceptance_bridge = True
            return True
        if key in {"to", "that"} or token.language is SpeechActLanguage.CHINESE:
            self._add_signal(pending, SpeechActKind.POSITIVE_ACCEPTANCE, "meeting_shorthand")
            state.positive_seen = True
        else:
            self._mark_ambiguous(pending, "incomplete_acceptance")
        state.pending_acceptance = None
        state.pending_acceptance_bridge = False
        return False

    def _close_clause(self) -> None:
        state = self.clause
        self._resolve_pending_acceptance(None)
        if state.pending_consensus is not None:
            self._add_signal(
                state.pending_consensus,
                SpeechActKind.ATTRIBUTIVE_MENTION,
                "consensus_mention",
            )
        if state.pending_proposal_reversal is not None and state.proposal_seen:
            token, negated = state.pending_proposal_reversal
            self._add_signal(
                token,
                SpeechActKind.NONDECISION if negated else SpeechActKind.AFFIRMATIVE_REJECTION,
                "proposal_reversal",
            )
            state.pending_proposal_reversal = None
        unresolved = (
            state.pending_negator is not None
            or state.pending_focus_not
            or state.according_state != 0
            or state.pending_reporting is not None
            or state.pending_proposal_reversal is not None
            or state.pending_voted
        )
        if unresolved:
            token = (
                state.pending_reporting
                or (state.pending_proposal_reversal or (None, False))[0]
                or _Token(
                    _TokenKind.OTHER,
                    "",
                    max(len(self.text) - 1, 0),
                    len(self.text),
                    SpeechActLanguage.UNKNOWN,
                )
            )
            self._mark_ambiguous(token, "unresolved_directional_state")
        clause_language = _language(state.saw_en, state.saw_zh)
        for index, signal in enumerate(self.signals):
            if signal.clause_index == state.index and signal.language is not clause_language:
                self.signals[index] = replace(signal, language=clause_language)
        self.reported_scope |= state.reported_scope
        self.external |= state.external
        self.clauses += 1
        self.clause = _ClauseState(state.index + 1)

    def _quote_token(self, token: _Token) -> None:
        if token.kind is _TokenKind.QUOTE_OPEN:
            if len(self.quote_stack) >= MAX_QUOTE_DEPTH:
                if self.quote_error is None:
                    self.quote_error = "quote_depth"
                return
            self.quote_stack.append(token.key)
            self.clause.reported_scope = True
        elif not self.quote_stack or self.quote_stack[-1] != token.key:
            if self.quote_error is None:
                self.quote_error = "quote_mismatch"
        else:
            self.quote_stack.pop()
            if not self.quote_stack:
                self.clause.subject = "absent"
                self.clause.subject_token = ""
                self.clause.predicate_clean = True

    def _set_subject(self, key: str) -> None:
        state = self.clause
        if (
            state.subject != "absent"
            or key in _EN_DETERMINERS
            or key in {"我的", "我们的"}
            or _is_bridge(key)
        ):
            return
        state.subject = _subject_class(key)
        state.subject_token = key

    def _reporting(self, token: _Token) -> bool:
        state = self.clause
        key = token.key
        if (key in _EN_REPORTING or key in _ZH_REPORTING) and state.positive_seen:
            history = tuple(state.last_keys or ())
            tail_subject = _subject_class(history[-1]) if history else "absent"
            state.reported_scope = True
            if tail_subject == "third_party":
                state.external = True
                replacement = SpeechActKind.ATTRIBUTIVE_MENTION
                reason = "late_external_attribution"
            elif tail_subject in {"direct", "collective"}:
                replacement = SpeechActKind.ATTRIBUTIVE_MENTION
                reason = "late_direct_report"
            else:
                replacement = SpeechActKind.AMBIGUOUS
                reason = "unknown_attribution"
            for index, signal in enumerate(self.signals):
                if (
                    signal.clause_index == state.index
                    and signal.kind is SpeechActKind.POSITIVE_ACCEPTANCE
                ):
                    self.signals[index] = replace(signal, kind=replacement, reason=reason)
            state.positive_seen = False
            return True
        if key in _EN_IMPERSONAL or key in _ZH_IMPERSONAL:
            state.reported_scope = True
            state.external = True
            state.pending_reporting = token
            return True
        if key in _EN_REPORTING or key in _ZH_REPORTING:
            state.reported_scope = True
            state.pending_reporting = token
            if state.subject in {"third_party", "absent"}:
                state.external = True
            elif state.subject_token == "it" and any(
                item in {"was", "is", "been"} for item in tuple(state.last_keys or ())
            ):
                state.external = True
            elif state.subject == "unknown":
                self._mark_ambiguous(token, "unknown_attribution")
            return True
        if (
            key in {"confirm", "confirms", "confirmed", "confirming", "确认"}
            and state.subject == "third_party"
        ):
            state.reported_scope = True
            state.external = True
            return True
        if key == "according":
            state.according_state = 1
            state.reported_scope = True
            state.pending_reporting = token
            return True
        if state.according_state == 1:
            if key == "to":
                state.according_state = 2
                return True
            self._mark_ambiguous(token, "unknown_attribution")
            state.according_state = 0
        elif state.according_state == 2:
            if key in _EN_DETERMINERS:
                return True
            subject = _subject_class(key)
            if subject == "third_party":
                state.external = True
            elif subject != "direct":
                self._mark_ambiguous(token, "unknown_attribution")
            state.according_state = 0
            return True
        if key == "据":
            state.according_state = 3
            state.reported_scope = True
            state.pending_reporting = token
            return True
        if state.according_state == 3:
            subject = _subject_class(key)
            if subject == "third_party":
                state.external = True
                state.according_state = 4
            elif subject == "direct":
                state.according_state = 4
            else:
                self._mark_ambiguous(token, "unknown_attribution")
                state.according_state = 0
            return True
        if state.according_state == 4 and key in {"称", "表示", "报告"}:
            state.according_state = 0
            return True
        return False

    def _acceptance(self, token: _Token) -> bool:
        state = self.clause
        key = token.key
        history = tuple(state.last_keys or ())
        is_consensus = key in {"共识", "consensus"}
        if key in {"reached", "reach"} and state.pending_consensus is not None:
            self._add_signal(token, SpeechActKind.POSITIVE_ACCEPTANCE, "consensus_result")
            state.pending_consensus = None
            state.positive_seen = True
            return True
        if key not in _EN_ACCEPT_ACTIVE and key not in _ZH_ACCEPT and not is_consensus:
            return False
        if state.pending_negator is not None:
            kind = (
                SpeechActKind.NEGATED_ACCEPTANCE if state.negator_clean else SpeechActKind.AMBIGUOUS
            )
            self._add_signal(
                token,
                kind,
                "negated_acceptance" if state.negator_clean else "ambiguous_negation_scope",
            )
            state.pending_negator = None
            return True
        consensus_actual = False
        if is_consensus:
            consensus_actual = bool(history and history[-1] in {"reached", "reach", "达成", "形成"})
            if not consensus_actual:
                if key == "consensus" and state.semantic_count <= 2:
                    state.pending_consensus = token
                else:
                    self._add_signal(token, SpeechActKind.ATTRIBUTIVE_MENTION, "consensus_mention")
                return True
        reported = bool(self.quote_stack) or state.reported_scope
        direct_subject = state.subject in {"direct", "collective"}
        passive_result = (
            state.subject == "object"
            and any(
                item in {"was", "were", "is", "been", "has", "已", "已经", "获", "获得", "被"}
                for item in history
            )
            and (key in _EN_ACCEPT_RESULT or key in {"通过", "批准", "接受", "确认"})
        )
        implicit = state.subject == "absent" and (
            state.semantic_count == 1
            or (
                bool(history) and all(item in {"already", "已", "已经", "一致"} for item in history)
            )
        )
        if reported:
            self._add_signal(token, SpeechActKind.ATTRIBUTIVE_MENTION, "reported_scope")
        elif direct_subject and (state.predicate_clean or consensus_actual):
            self._add_signal(token, SpeechActKind.POSITIVE_ACCEPTANCE, "direct_predicate")
            state.positive_seen = True
        elif passive_result:
            self._add_signal(token, SpeechActKind.POSITIVE_ACCEPTANCE, "passive_result")
            state.positive_seen = True
        elif implicit:
            state.pending_acceptance = token
            state.pending_acceptance_bridge = False
        elif state.subject == "third_party":
            self._add_signal(token, SpeechActKind.ATTRIBUTIVE_MENTION, "third_party_subject")
        elif state.subject == "unknown":
            self._add_signal(
                token,
                SpeechActKind.AMBIGUOUS,
                "unknown_subject",
            )
        else:
            self._add_signal(token, SpeechActKind.ATTRIBUTIVE_MENTION, "descriptive_acceptance")
        state.pending_negator = None
        return True

    def _rejection(self, token: _Token) -> bool:
        state = self.clause
        key = token.key
        affirmative_result = key in _ZH_REJECTION_RESULTS or key == "voted_down"
        if state.pending_voted and key == "down":
            key = "voted_down"
            affirmative_result = True
            state.pending_voted = False
        elif key == "voted":
            state.pending_voted = True
            return True
        rejection = key in _EN_REJECTION or key in _ZH_REJECTION or affirmative_result
        proposal_reversal = key in _EN_PROPOSAL_REVERSAL or key in _ZH_PROPOSAL_REVERSAL
        if not rejection and not proposal_reversal:
            return False
        negated = (
            state.pending_negator is not None and state.negator_clean and not affirmative_result
        )
        if proposal_reversal and not state.proposal_seen:
            state.pending_proposal_reversal = (token, negated)
            state.pending_negator = None
            return True
        if proposal_reversal:
            self.proposal_reference = True
        if bool(self.quote_stack) or state.reported_scope:
            kind, reason = SpeechActKind.ATTRIBUTIVE_MENTION, "reported_scope"
        elif state.pending_negator is not None and not state.negator_clean:
            kind, reason = SpeechActKind.AMBIGUOUS, "ambiguous_negation_scope"
        elif negated:
            kind, reason = SpeechActKind.NONDECISION, "negated_rejection"
        else:
            kind = SpeechActKind.AFFIRMATIVE_REJECTION
            reason = "proposal_reversal" if proposal_reversal else "affirmative_rejection"
        self._add_signal(token, kind, reason)
        state.pending_negator = None
        return True

    def _visit(self, token: _Token) -> None:
        if token.kind in {_TokenKind.QUOTE_OPEN, _TokenKind.QUOTE_CLOSE}:
            self._quote_token(token)
            return
        if token.kind is _TokenKind.BOUNDARY:
            self._close_clause()
            return
        if token.kind is _TokenKind.PUNCT:
            return
        state = self.clause
        key = token.key
        state.semantic_count += 1
        state.saw_en |= token.language is SpeechActLanguage.ENGLISH
        state.saw_zh |= token.language is SpeechActLanguage.CHINESE
        if (key in _EN_ADVERSATIVES or key in _ZH_ADVERSATIVES) and not (
            key == "yet" and state.pending_negator is not None
        ):
            self._close_clause()
            return
        if self._resolve_pending_acceptance(token):
            assert state.last_keys is not None
            state.last_keys.append(key)
            return
        if (
            state.pending_reporting is not None
            and state.according_state == 0
            and not (
                key in _EN_REPORTING
                or key in _ZH_REPORTING
                or key in _EN_DETERMINERS
                or _is_bridge(key)
            )
        ):
            state.pending_reporting = None
        if state.pending_focus_not:
            if key in _EN_FOCUS:
                state.pending_focus_not = False
                state.pending_negator = None
                assert state.last_keys is not None
                state.last_keys.append(key)
                return
            state.pending_focus_not = False
        if key == "not":
            state.pending_focus_not = True
            state.pending_negator = key
            state.negator_clean = True
            assert state.last_keys is not None
            state.last_keys.append(key)
            return
        if key in _ZH_FOCUS:
            state.pending_negator = None
            assert state.last_keys is not None
            state.last_keys.append(key)
            return
        if key == "也" and state.subject in {"direct", "collective"}:
            state.predicate_clean = True
        if key in _EN_NEGATORS or key in _ZH_NEGATORS or _is_contraction_negator(key):
            state.pending_negator = key
            state.negator_clean = True
            assert state.last_keys is not None
            state.last_keys.append(key)
            return
        handled = self._reporting(token) or self._acceptance(token) or self._rejection(token)
        if not handled:
            self._set_subject(key)
        if key in _EN_OBJECT or key in _ZH_OBJECT:
            state.proposal_seen |= key in {
                "proposal",
                "motion",
                "request",
                "vote",
                "方案",
                "提案",
                "动议",
                "请求",
                "投票",
            }
            self.proposal_reference |= state.proposal_seen
            if state.pending_proposal_reversal is not None and state.proposal_seen:
                reversal, negated = state.pending_proposal_reversal
                self._add_signal(
                    reversal,
                    SpeechActKind.NONDECISION if negated else SpeechActKind.AFFIRMATIVE_REJECTION,
                    "proposal_reversal",
                )
                state.pending_proposal_reversal = None
        if state.pending_negator is not None and not handled and not _is_bridge(key):
            state.pending_negator = None
            state.pending_focus_not = False
        if state.subject in {"direct", "collective", "object", "third_party"} and not handled:
            if key != state.subject_token and not _is_bridge(key):
                state.predicate_clean = False
        if state.last_keys is not None:
            state.last_keys.append(key)

    def scan(self) -> SpeechActAnalysis:
        text, length = self.text, len(self.text)
        while self.cursor < length:
            start, character = self.cursor, text[self.cursor]
            if character.isspace():
                end = start + 1
                while end < length and text[end].isspace() and text[end] not in "\r\n":
                    end += 1
                self._advance(end)
            elif character in _STRONG:
                self._advance(start + 1)
                self._emit(
                    _Token(
                        _TokenKind.BOUNDARY, character, start, start + 1, SpeechActLanguage.UNKNOWN
                    )
                )
            elif character in _NEUTRAL:
                self._advance(start + 1)
                self._emit(
                    _Token(_TokenKind.PUNCT, character, start, start + 1, SpeechActLanguage.UNKNOWN)
                )
            elif character in _PAIRED_QUOTES:
                self._advance(start + 1)
                self._emit(
                    _Token(
                        _TokenKind.QUOTE_OPEN,
                        _PAIRED_QUOTES[character],
                        start,
                        start + 1,
                        SpeechActLanguage.UNKNOWN,
                    )
                )
            elif character in _QUOTE_CLOSERS:
                self._advance(start + 1)
                self._emit(
                    _Token(
                        _TokenKind.QUOTE_CLOSE,
                        character,
                        start,
                        start + 1,
                        SpeechActLanguage.UNKNOWN,
                    )
                )
            elif character in {'"', "'"}:
                opening = self.straight_quote != character
                self.straight_quote = character if opening else None
                self._advance(start + 1)
                self._emit(
                    _Token(
                        _TokenKind.QUOTE_OPEN if opening else _TokenKind.QUOTE_CLOSE,
                        character,
                        start,
                        start + 1,
                        SpeechActLanguage.UNKNOWN,
                    )
                )
            elif character.isascii() and character.isalpha():
                end = start + 1
                while end < length:
                    current = text[end]
                    if current.isascii() and current.isalpha():
                        end += 1
                    elif (
                        current in {"'", "\u2019"}
                        and end + 1 < length
                        and text[end + 1].isascii()
                        and text[end + 1].isalpha()
                    ):
                        end += 1
                    else:
                        break
                key = text[start:end].casefold().replace("\u2019", "'")
                self._advance(end)
                self._emit(_Token(_TokenKind.WORD, key, start, end, SpeechActLanguage.ENGLISH))
            elif _is_cjk(character):
                comparisons, key = 0, character
                maximum = min(length - start, _MAX_ZH_LEXEME)
                for width in range(maximum, 0, -1):
                    comparisons += 1
                    candidate = text[start : start + width]
                    if candidate in _ZH_LEXEMES:
                        key = candidate
                        break
                self.lookahead += comparisons
                if self.observer is not None:
                    self.observer.trie_lookahead(start, comparisons)
                end = start + len(key)
                self._advance(end)
                self._emit(_Token(_TokenKind.CJK, key, start, end, SpeechActLanguage.CHINESE))
            else:
                end = start + 1
                while end < length:
                    following = text[end]
                    if (
                        following.isspace()
                        or following in _STRONG
                        or following in _NEUTRAL
                        or following in _PAIRED_QUOTES
                        or following in _QUOTE_CLOSERS
                        or following in {'"', "'"}
                        or (following.isascii() and following.isalpha())
                        or _is_cjk(following)
                    ):
                        break
                    end += 1
                self._advance(end)
                self._emit(_Token(_TokenKind.OTHER, "", start, end, SpeechActLanguage.UNKNOWN))
        self._close_clause()
        if (self.quote_stack or self.straight_quote is not None) and self.quote_error is None:
            self.quote_error = "quote_mismatch"
        if self.quote_error is not None:
            self._mark_ambiguous(
                _Token(_TokenKind.OTHER, "", max(length - 1, 0), length, SpeechActLanguage.UNKNOWN),
                self.quote_error,
            )
        if self.signal_overflow:
            self.signals = [
                SpeechActSignal(
                    SpeechActKind.AMBIGUOUS,
                    SpeechActLanguage.UNKNOWN,
                    max(self.clauses - 1, 0),
                    0,
                    0,
                    "signal_capacity",
                )
            ]
        return SpeechActAnalysis(
            tuple(self.signals),
            self.proposal_reference,
            self.reported_scope,
            self.external,
            False,
            self.signal_overflow,
            SpeechActScanStats(length, self.emitted, self.visited, self.clauses, self.lookahead),
        )


def classify_speech_acts(
    text: str, *, _observer: _SpeechActScanObserver | None = None
) -> SpeechActAnalysis:
    """Classify bounded EN/ZH speech acts with one monotonic character cursor."""

    length = len(text)
    if length > MAX_SPEECH_ACT_TEXT_CODEPOINTS:
        return SpeechActAnalysis(
            (
                SpeechActSignal(
                    SpeechActKind.AMBIGUOUS, SpeechActLanguage.UNKNOWN, 0, 0, 0, "input_too_long"
                ),
            ),
            False,
            False,
            False,
            True,
            False,
            SpeechActScanStats(length, 0, 0, 0, 0),
        )
    return _Scanner(text, _observer).scan()


def contains_rejection(text: str) -> bool:
    """Compatibility wrapper over the single speech-act owner."""

    return classify_speech_acts(text).has_disqualifying_signal


# Object-level merge semantics remain deliberately separate from speech-act authority.
_ENGLISH_REVERSAL_FAMILIES = (
    r"reject(?:s|ed|ing|ion(?:s)?)?",
    r"declin(?:e|es|ed|ing)",
    r"fail(?:s|ed|ing|ure(?:s)?)?",
    r"den(?:y|ies|ied|ying|ial(?:s)?)",
    r"prohibit(?:s|ed|ing|ion(?:s)?)?",
    r"defeat(?:s|ed|ing)?",
    r"disapprov(?:e|es|ed|ing|al(?:s)?)",
    r"veto(?:s|es|ed|ing)?",
    r"overrul(?:e|es|ed|ing)",
    r"los(?:e|es|ing|t|s(?:es)?)",
    r"withdraw(?:s|ing|n|al(?:s)?)?",
    r"withdrew",
    r"refus(?:e|es|ed|ing|al(?:s)?)",
    r"cancel(?:s|ed|ing|l(?:ed|ing)|lation(?:s)?)?",
    r"stop(?:s|ped|ping)?",
    r"terminat(?:e|es|ed|ing|ion(?:s)?)",
    r"revok(?:e|es|ed|ing|ation(?:s)?)",
    r"suspend(?:s|ed|ing)?",
    r"paus(?:e|es|ed|ing)",
    r"abolish(?:es|ed|ing|ment(?:s)?)?",
)
_ENGLISH_REVERSAL = re.compile(
    rf"(?:(?<![A-Za-z])(?:no|not|never|none|without|cannot|pending|voted\s+down|{'|'.join(_ENGLISH_REVERSAL_FAMILIES)})(?![A-Za-z])|(?<![A-Za-z])[A-Za-z]+n['\u2019]t(?![A-Za-z]))",
    re.IGNORECASE,
)
_CHINESE_REVERSAL = re.compile(
    r"(?:不予|未获|未通过|不通过|禁止|不得|不能|不许|不要|拒绝|取消|停止|废止|终止|暂停|中止|停用|作废|反对|驳回|否决|否定|撤回|撤销|落选|败北|不|未|无|非|否|别|勿|莫)"
)
_ACTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("deploy", re.compile(r"(?<![A-Za-z])deploy(?:s|ed|ing|ment)?(?![A-Za-z])|部署", re.I)),
    (
        "delete",
        re.compile(r"(?<![A-Za-z])(?:delete|remove)(?:s|d|ing)?(?![A-Za-z])|删除|移除", re.I),
    ),
    ("create", re.compile(r"(?<![A-Za-z])(?:create|add)(?:s|d|ing)?(?![A-Za-z])|创建|新增", re.I)),
    (
        "enable",
        re.compile(r"(?<![A-Za-z])(?:enable|start)(?:s|d|ing)?(?![A-Za-z])|启用|启动|开始", re.I),
    ),
    (
        "continue",
        re.compile(r"(?<![A-Za-z])(?:continue|resume)(?:s|d|ing)?(?![A-Za-z])|继续|恢复", re.I),
    ),
    ("prepare", re.compile(r"(?<![A-Za-z])prepar(?:e|es|ed|ing)(?![A-Za-z])|准备", re.I)),
    ("approve", re.compile(r"(?<![A-Za-z])approv(?:e|es|ed|ing|al)(?![A-Za-z])|批准|通过", re.I)),
    ("publish", re.compile(r"(?<![A-Za-z])publish(?:es|ed|ing)?(?![A-Za-z])|发布", re.I)),
)


def contains_reversal(text: str) -> bool:
    """Return whether object text contains an explicit negative/reversal marker."""

    return bool(_ENGLISH_REVERSAL.search(text) or _CHINESE_REVERSAL.search(text))


def material_actions(text: str) -> frozenset[str]:
    """Return conservative canonical object-action markers."""

    return frozenset(name for name, pattern in _ACTION_PATTERNS if pattern.search(text))
