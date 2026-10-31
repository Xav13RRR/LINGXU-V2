"""Closed schema-version-2 structures for evidence-linked Meeting minutes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lingxu.worker.outcome_merge import MergedMinutesOutcome
from lingxu.worker.outcomes import (
    MAX_TOPIC_OUTCOME_SETS,
    OutcomeType,
    TopicOutcomeSet,
    VerificationStatus,
)

SummaryMethod = Literal[
    "deterministic",
    "deterministic_fallback",
    "deepseek_partial",
    "deepseek",
    # Read compatibility for minutes created before provider-accurate provenance.
    "local_llm_partial",
    "local_llm",
]


class _ClosedModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )


class MinutesProvenance(_ClosedModel):
    outcome_pipeline_revision: str = Field(min_length=1)
    speech_act_policy_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    minutes_renderer_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    renderer_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validated_outcomes_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_transcript_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    minutes_markdown_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    minutes_payload_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class MinutesParticipants(_ClosedModel):
    host: str = Field(min_length=1, max_length=256)
    attendees: tuple[str, ...] = Field(max_length=500)


class MinutesAgenda(_ClosedModel):
    agenda_id: str = Field(min_length=1, max_length=128)
    position: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=500)
    language: str = Field(min_length=1, max_length=32)
    status: str = Field(min_length=1, max_length=64)
    duration_minutes: int | None = Field(default=None, ge=1)
    requires_vote: bool
    vote_content: str | None = Field(default=None, max_length=500)
    vote_duration_seconds: int | None = Field(default=None, ge=0)
    topic_ids: tuple[str, ...] = Field(max_length=500)
    summary: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def vote_configuration_is_coherent(self) -> MinutesAgenda:
        if not self.requires_vote and (
            self.vote_content is not None or self.vote_duration_seconds is not None
        ):
            raise ValueError("non-voting agenda cannot carry vote configuration")
        return self


class MinutesTopic(_ClosedModel):
    topic_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=500)
    started_ms: int = Field(ge=0)
    ended_ms: int = Field(ge=0)
    method: str = Field(min_length=1, max_length=64)
    source_entry_ids: tuple[str, ...] = Field(max_length=2_000)
    summary: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def interval_is_ordered(self) -> MinutesTopic:
        if self.ended_ms < self.started_ms:
            raise ValueError("topic interval must be ordered")
        return self


class MinutesVote(_ClosedModel):
    source_event_id: int = Field(ge=1)
    vote_id: str | None = Field(default=None, max_length=128)
    agenda_item_id: str | None = Field(default=None, max_length=128)
    vote_content: str | None = Field(default=None, max_length=500)
    counts: dict[str, int]
    eligible_count: int | None = Field(default=None, ge=0)
    responded_count: int = Field(ge=0)
    result: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def counts_are_authoritatively_bounded(self) -> MinutesVote:
        if any(not key or count < 0 for key, count in self.counts.items()):
            raise ValueError("vote counts contain an invalid option or count")
        total_count = sum(self.counts.values())
        if (
            self.eligible_count is not None and total_count > self.eligible_count
        ) or self.responded_count != total_count - self.counts.get("abstain", 0):
            raise ValueError("vote counts exceed eligible or responded participants")
        return self


def normalise_vote_payload(source_event_id: int, payload: dict[str, Any]) -> MinutesVote:
    """Strictly project the supported legacy vote DTO into the closed minutes schema."""

    counts_value = payload.get("counts", {})
    if not isinstance(counts_value, dict) or any(
        not isinstance(key, str)
        or not key
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for key, value in counts_value.items()
    ):
        raise ValueError("authoritative vote counts are malformed")
    counts = dict(counts_value)

    eligible_value = payload.get("eligible_count")
    if eligible_value is None:
        eligible_count = None
    elif (
        not isinstance(eligible_value, bool)
        and isinstance(eligible_value, int)
        and eligible_value == 0
        and sum(counts.values()) > 0
        and payload.get("agenda_item_id") is None
    ):
        # The pre-agenda generic-vote DTO used zero as an unknown-recipient sentinel.
        eligible_count = None
    elif (
        isinstance(eligible_value, bool)
        or not isinstance(eligible_value, int)
        or eligible_value < 0
    ):
        raise ValueError("vote eligible_count is malformed")
    else:
        eligible_count = eligible_value

    default_responded = sum(counts.values()) - counts.get("abstain", 0)
    responded_value = payload.get("responded_count", default_responded)
    if (
        isinstance(responded_value, bool)
        or not isinstance(responded_value, int)
        or responded_value < 0
    ):
        raise ValueError("vote responded_count is malformed")

    def optional_text(field: str, maximum: int) -> str | None:
        value = payload.get(field)
        if value is None:
            return None
        if not isinstance(value, str) or not value or len(value) > maximum:
            raise ValueError(f"vote {field} is malformed")
        return value

    result_value = payload.get("result")
    if isinstance(result_value, bool):
        result = "yes" if result_value else "no"
    elif result_value is None:
        result = None
    elif isinstance(result_value, str) and result_value and len(result_value) <= 64:
        result = result_value
    else:
        raise ValueError("vote result is malformed")

    return MinutesVote(
        source_event_id=source_event_id,
        vote_id=optional_text("vote_id", 128),
        agenda_item_id=optional_text("agenda_item_id", 128),
        vote_content=optional_text("vote_content", 500),
        counts=counts,
        eligible_count=eligible_count,
        responded_count=responded_value,
        result=result,
    )


class MinutesSpeakingInterval(_ClosedModel):
    """One exact active-floor interval derived from the Session event stream."""

    started_at: datetime
    ended_at: datetime
    duration_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def interval_is_exact(self) -> MinutesSpeakingInterval:
        if self.started_at.tzinfo is None or self.ended_at.tzinfo is None:
            raise ValueError("speaking interval timestamps must be timezone-aware")
        if self.ended_at < self.started_at:
            raise ValueError("speaking interval is reversed")
        expected = max(0, int((self.ended_at - self.started_at).total_seconds() * 1_000))
        if self.duration_ms != expected:
            raise ValueError("speaking interval duration is inconsistent")
        return self


class MinutesSpeakingPauseInterval(MinutesSpeakingInterval):
    """A timer pause excluded from actual floor occupancy."""

    reason: str = Field(min_length=1, max_length=128)


class MinutesSpeakingPlanTurn(_ClosedModel):
    """One privacy-filtered immutable turn from an agenda policy snapshot."""

    user_id: int | None
    name: str = Field(min_length=1, max_length=256)
    planned_seconds: int = Field(ge=1, le=14_400)

    @model_validator(mode="after")
    def identity_is_valid(self) -> MinutesSpeakingPlanTurn:
        if self.user_id is not None and self.user_id <= 0:
            raise ValueError("speaking-plan turn identity is invalid")
        return self


class MinutesSpeakingPlanParticipant(_ClosedModel):
    """One privacy-filtered presenter from an agenda policy snapshot."""

    user_id: int | None
    name: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def identity_is_valid(self) -> MinutesSpeakingPlanParticipant:
        if self.user_id is not None and self.user_id <= 0:
            raise ValueError("speaking-plan participant identity is invalid")
        return self


class MinutesSpeakingPlan(_ClosedModel):
    """Closed agenda-level speaking policy as planned, independent of execution."""

    policy_schema_version: Literal[1] = 1
    agenda_item_id: str = Field(min_length=1, max_length=128)
    mode: Literal["FREE", "ORDERED", "PRESENTATION"]
    order_strategy: Literal["NONE", "EXPLICIT", "DISCORD_ALPHABETICAL"]
    host_participation: Literal["AUTO", "MODERATOR_ONLY"]
    turns: tuple[MinutesSpeakingPlanTurn, ...] = Field(default=(), max_length=50)
    default_turn_seconds: int | None = Field(default=None, ge=1, le=14_400)
    presenters: tuple[MinutesSpeakingPlanParticipant, ...] = Field(default=(), max_length=50)
    presentation_seconds: int | None = Field(default=None, ge=1, le=86_400)
    followup_mode: Literal["NONE", "SUPPLEMENT_TURNS", "Q_AND_A"]
    mute_enforcement_requested: bool
    prep_seconds: int = Field(ge=0, le=300)

    @model_validator(mode="after")
    def policy_shape_is_coherent(self) -> MinutesSpeakingPlan:
        visible_turn_ids = tuple(value.user_id for value in self.turns if value.user_id is not None)
        visible_presenter_ids = tuple(
            value.user_id for value in self.presenters if value.user_id is not None
        )
        if len(set(visible_turn_ids)) != len(visible_turn_ids) or len(
            set(visible_presenter_ids)
        ) != len(visible_presenter_ids):
            raise ValueError("speaking-plan identities are duplicated")
        if self.mode == "FREE":
            if (
                self.order_strategy != "NONE"
                or self.turns
                or self.default_turn_seconds is not None
                or self.presenters
                or self.presentation_seconds is not None
                or self.followup_mode != "NONE"
                or self.mute_enforcement_requested
            ):
                raise ValueError("free speaking plan contains controlled-floor settings")
        elif self.mode == "ORDERED":
            if self.presenters or self.presentation_seconds is not None:
                raise ValueError("ordered speaking plan contains presentation settings")
            if self.order_strategy == "EXPLICIT":
                if not self.turns or self.default_turn_seconds is not None:
                    raise ValueError("explicit speaking plan is incomplete")
            elif self.order_strategy == "DISCORD_ALPHABETICAL":
                if self.turns or self.default_turn_seconds is None:
                    raise ValueError("alphabetical speaking plan is incomplete")
            else:
                raise ValueError("ordered speaking plan has no order strategy")
        elif (
            self.order_strategy != "NONE"
            or self.turns
            or self.default_turn_seconds is not None
            or not self.presenters
        ):
            raise ValueError("presentation speaking plan is incomplete")
        return self


MuteEnforcementFinalStatus = Literal[
    "NOT_REQUESTED",
    "REQUESTED_NOT_STARTED",
    "ACTIVE",
    "PAUSED",
    "ENDED",
    "DOWNGRADED",
    "EMERGENCY_RELEASED",
]


class MinutesMuteEnforcementSummary(_ClosedModel):
    """Content-safe operational outcome for one agenda's requested mute enforcement."""

    agenda_item_id: str = Field(min_length=1, max_length=128)
    requested: bool
    started: bool
    pause_count: int = Field(ge=0)
    resume_count: int = Field(ge=0)
    permission_loss_count: int = Field(ge=0)
    reapplication_count: int = Field(ge=0)
    host_intervention_count: int = Field(ge=0)
    final_status: MuteEnforcementFinalStatus

    @model_validator(mode="after")
    def lifecycle_is_coherent(self) -> MinutesMuteEnforcementSummary:
        operational_counts = (
            self.pause_count,
            self.resume_count,
            self.permission_loss_count,
            self.reapplication_count,
            self.host_intervention_count,
        )
        if not self.requested and (
            self.started or any(operational_counts) or self.final_status != "NOT_REQUESTED"
        ):
            raise ValueError("unrequested mute enforcement contains operational facts")
        if (
            self.requested
            and not self.started
            and (any(operational_counts) or self.final_status != "REQUESTED_NOT_STARTED")
        ):
            raise ValueError("unstarted mute enforcement contains operational facts")
        if self.started and self.final_status in {"NOT_REQUESTED", "REQUESTED_NOT_STARTED"}:
            raise ValueError("started mute enforcement has an invalid final status")
        return self


SpeakingFinishReason = Literal[
    "TIME_EXPIRED",
    "HOST_ENDED_EARLY",
    "HOST_SKIPPED",
    "PARTICIPANT_ABANDONED",
    "ABSENT",
]


class MinutesSpeakingTurn(_ClosedModel):
    """Authoritative completed main-floor facts for one controlled turn."""

    agenda_item_id: str = Field(min_length=1, max_length=128)
    turn_index: int = Field(ge=0)
    mode: Literal["ORDERED", "PRESENTATION"]
    speaker_user_ids: tuple[int | None, ...] = Field(min_length=1, max_length=50)
    speaker_names: tuple[str, ...] = Field(min_length=1, max_length=50)
    absent_user_ids: tuple[int | None, ...] = Field(default=(), max_length=50)
    absent_names: tuple[str, ...] = Field(default=(), max_length=50)
    planned_seconds: int = Field(ge=1, le=28_800)
    extension_seconds: int = Field(ge=0, le=28_800)
    started_at: datetime | None
    finished_at: datetime
    active_intervals: tuple[MinutesSpeakingInterval, ...] = Field(max_length=1_000)
    pause_intervals: tuple[MinutesSpeakingPauseInterval, ...] = Field(max_length=1_000)
    actual_floor_ms: int = Field(ge=0)
    finish_reason: SpeakingFinishReason
    finished_early: bool
    skipped: bool
    absent: bool
    overtime_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def authoritative_turn_is_coherent(self) -> MinutesSpeakingTurn:
        if self.finished_at.tzinfo is None or (
            self.started_at is not None and self.started_at.tzinfo is None
        ):
            raise ValueError("speaking turn timestamps must be timezone-aware")
        visible_user_ids = tuple(value for value in self.speaker_user_ids if value is not None)
        visible_absent_ids = tuple(value for value in self.absent_user_ids if value is not None)
        if len(self.speaker_user_ids) != len(self.speaker_names) or len(
            set(visible_user_ids)
        ) != len(visible_user_ids):
            raise ValueError("speaking turn identities are inconsistent")
        if len(self.absent_user_ids) != len(self.absent_names) or len(
            set(visible_absent_ids)
        ) != len(visible_absent_ids):
            raise ValueError("speaking turn absence identities are inconsistent")
        if set(visible_user_ids) & set(visible_absent_ids) and not self.absent:
            raise ValueError("speaking and absent identities overlap")
        if any(value <= 0 for value in (*visible_user_ids, *visible_absent_ids)):
            raise ValueError("speaking turn identity is invalid")
        if any(
            not value or len(value) > 256 for value in (*self.speaker_names, *self.absent_names)
        ):
            raise ValueError("speaking turn name is invalid")
        if self.started_at is None:
            if self.active_intervals or self.pause_intervals or self.actual_floor_ms:
                raise ValueError("unstarted speaking turn cannot contain active time")
        elif self.finished_at < self.started_at:
            raise ValueError("speaking turn finish precedes start")
        elif any(
            interval.started_at < self.started_at or interval.ended_at > self.finished_at
            for interval in (*self.active_intervals, *self.pause_intervals)
        ):
            raise ValueError("speaking interval exceeds its turn")
        if self.actual_floor_ms != sum(value.duration_ms for value in self.active_intervals):
            raise ValueError("actual speaking-floor duration is inconsistent")
        expected_overtime = max(
            0,
            self.actual_floor_ms - (self.planned_seconds + self.extension_seconds) * 1_000,
        )
        if self.overtime_ms != expected_overtime:
            raise ValueError("speaking overtime is inconsistent")
        if self.finished_early != (
            self.finish_reason in {"HOST_ENDED_EARLY", "PARTICIPANT_ABANDONED"}
        ):
            raise ValueError("speaking early-finish fact is inconsistent")
        if self.skipped != (self.finish_reason == "HOST_SKIPPED"):
            raise ValueError("speaking skip fact is inconsistent")
        if self.absent != (self.finish_reason == "ABSENT"):
            raise ValueError("speaking absence fact is inconsistent")
        if self.absent and not self.absent_user_ids:
            raise ValueError("absent speaking turn has no absent identity")
        return self


FollowupFinishReason = Literal["SPEAKER_FINISHED", "SPEAKER_REMOVED", "FOLLOWUP_FINISHED"]


class MinutesFollowupParticipation(_ClosedModel):
    """One participant who was actually selected onto the follow-up floor."""

    user_id: int | None
    name: str = Field(min_length=1, max_length=256)
    started_at: datetime
    finished_at: datetime
    finish_reason: FollowupFinishReason

    @model_validator(mode="after")
    def interval_is_ordered(self) -> MinutesFollowupParticipation:
        if self.user_id is not None and self.user_id <= 0:
            raise ValueError("follow-up participant identity is invalid")
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("follow-up timestamps must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("follow-up participation interval is reversed")
        return self


class MinutesSpeakingFollowup(_ClosedModel):
    """Authoritative optional supplement/Q&A phase and its selected participants."""

    agenda_item_id: str = Field(min_length=1, max_length=128)
    mode: Literal["SUPPLEMENT_TURNS", "Q_AND_A"]
    started_at: datetime
    finished_at: datetime
    standing_user_ids: tuple[int | None, ...] = Field(max_length=50)
    standing_names: tuple[str, ...] = Field(max_length=50)
    participations: tuple[MinutesFollowupParticipation, ...] = Field(max_length=1_000)

    @model_validator(mode="after")
    def followup_is_coherent(self) -> MinutesSpeakingFollowup:
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("follow-up timestamps must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("follow-up finish precedes start")
        visible_user_ids = tuple(value for value in self.standing_user_ids if value is not None)
        if len(self.standing_user_ids) != len(self.standing_names) or len(
            set(visible_user_ids)
        ) != len(visible_user_ids):
            raise ValueError("follow-up standing identities are inconsistent")
        if any(value <= 0 for value in visible_user_ids):
            raise ValueError("follow-up standing identity is invalid")
        if self.mode == "SUPPLEMENT_TURNS" and self.standing_user_ids:
            raise ValueError("supplement turns cannot have standing presenters")
        if any(
            value.started_at < self.started_at or value.finished_at > self.finished_at
            for value in self.participations
        ):
            raise ValueError("follow-up participation exceeds its phase")
        return self


class DeterministicCounts(_ClosedModel):
    agenda_total: int = Field(ge=0)
    agenda_completed: int = Field(ge=0)
    participant_transcript_entries: int = Field(ge=0)
    bot_tts_entries: int = Field(ge=0)
    closed_votes: int = Field(ge=0)
    topic_total: int = Field(ge=0)
    validated_outcomes: int = Field(ge=0)
    agenda_speaking_plans: int = Field(default=0, ge=0)
    mute_enforcement_summaries: int = Field(default=0, ge=0)
    controlled_speaking_turns: int = Field(default=0, ge=0)
    controlled_speaking_followups: int = Field(default=0, ge=0)


class EvidenceCoverage(_ClosedModel):
    evidence_required: int = Field(ge=0)
    evidence_linked: int = Field(ge=0)
    transcript_sources: int = Field(ge=0)
    event_sources: int = Field(ge=0)


class PrivacyExclusionCounts(_ClosedModel):
    transcript_entries: int = Field(ge=0)
    segment_rows: int = Field(ge=0)
    segment_files: int = Field(ge=0)
    measurement_available: bool = True


class CanonicalMetrics(_ClosedModel):
    source_entry_count: int = Field(ge=0)
    canonical_entry_count: int = Field(ge=0)
    duplicate_entry_count: int = Field(ge=0)
    backchannel_entry_count: int = Field(ge=0)
    asr_tag_count: int = Field(ge=0)


_OUTCOME_FIELDS = {
    "decisions": OutcomeType.DECISION,
    "agreements": OutcomeType.AGREEMENT,
    "actions": OutcomeType.ACTION_ITEM,
    "questions": OutcomeType.OPEN_QUESTION,
    "proposals": OutcomeType.PROPOSAL,
    "risks": OutcomeType.RISK,
    "disagreements": OutcomeType.DISAGREEMENT,
}


class MinutesDocumentV2(_ClosedModel):
    schema_version: Literal[2] = 2
    provenance: MinutesProvenance
    session_id: str = Field(min_length=1, max_length=128)
    generated_at: datetime
    meeting_title: str | None = Field(default=None, max_length=500)
    participants: MinutesParticipants
    agendas: tuple[MinutesAgenda, ...] = Field(max_length=500)
    topics: tuple[MinutesTopic, ...] = Field(max_length=500)
    validated_outcomes: tuple[TopicOutcomeSet, ...] = Field(max_length=MAX_TOPIC_OUTCOME_SETS)
    decisions: tuple[MergedMinutesOutcome, ...] = Field(max_length=500)
    agreements: tuple[MergedMinutesOutcome, ...] = Field(max_length=500)
    actions: tuple[MergedMinutesOutcome, ...] = Field(max_length=500)
    questions: tuple[MergedMinutesOutcome, ...] = Field(max_length=500)
    proposals: tuple[MergedMinutesOutcome, ...] = Field(max_length=500)
    risks: tuple[MergedMinutesOutcome, ...] = Field(max_length=500)
    disagreements: tuple[MergedMinutesOutcome, ...] = Field(max_length=500)
    deferred_items: tuple[MergedMinutesOutcome, ...] = Field(max_length=500)
    speaker_claims: tuple[MergedMinutesOutcome, ...] = Field(max_length=500)
    votes: tuple[MinutesVote, ...] = Field(max_length=500)
    speaking_plans: tuple[MinutesSpeakingPlan, ...] = Field(default=(), max_length=500)
    mute_enforcement_summaries: tuple[MinutesMuteEnforcementSummary, ...] = Field(
        default=(), max_length=500
    )
    speaking_turns: tuple[MinutesSpeakingTurn, ...] = Field(default=(), max_length=5_000)
    speaking_followups: tuple[MinutesSpeakingFollowup, ...] = Field(default=(), max_length=500)
    deterministic_counts: DeterministicCounts
    evidence_coverage: EvidenceCoverage
    privacy_exclusions: PrivacyExclusionCounts
    canonical_metrics: CanonicalMetrics
    summary_method: SummaryMethod
    summary_model: str | None = Field(default=None, max_length=128)
    summary_provider: str | None = Field(default=None, max_length=64)
    summary_model_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    summary_policy_identity: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    fallback_reasons: tuple[str, ...] = Field(max_length=32)
    executive_summary: str | None = Field(default=None, max_length=2_000)
    full_summary: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def outcome_sections_are_closed(self) -> MinutesDocumentV2:
        for field, expected_type in _OUTCOME_FIELDS.items():
            candidates = getattr(self, field)
            if any(candidate.type is not expected_type for candidate in candidates):
                raise ValueError(f"{field} contains an outcome of the wrong type")
            if any(
                candidate.verification_status
                in {VerificationStatus.NOT_CHECKED, VerificationStatus.UNVERIFIED}
                for candidate in candidates
            ):
                raise ValueError(f"{field} cannot contain an unverified outcome")
        if any(
            candidate.verification_status is not VerificationStatus.NOT_CHECKED
            for candidate in self.speaker_claims
        ):
            raise ValueError("speaker_claims must remain explicitly unverified")
        evidence_required = (*self.decisions, *self.actions, *self.questions, *self.risks)
        if any(
            not candidate.provenance.source_entry_ids and not candidate.provenance.source_event_ids
            for candidate in evidence_required
        ):
            raise ValueError("published outcome is missing evidence")
        if self.summary_method in {
            "deepseek",
            "deepseek_partial",
            "local_llm",
            "local_llm_partial",
        } and not all(
            (
                self.summary_model,
                self.summary_provider,
                self.summary_model_digest,
                self.summary_policy_identity,
            )
        ):
            raise ValueError("model-backed summary metadata is incomplete")
        return self
