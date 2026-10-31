"""Rule-first orchestration for LLM candidates and deterministic confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import structlog

from lingxu.domain.enums import MeetingState
from lingxu.domain.errors import StructuredIntentProviderError
from lingxu.meeting.controller import MeetingController, MeetingStatus
from lingxu.meeting.intents import IntentCandidate, RuleIntent, RuleIntentResolver
from lingxu.meeting.structured_intents import StructuredIntentProvider
from lingxu.storage.intent_repository import IntentCandidateRepository


class IntentResolutionStatus(StrEnum):
    NO_MATCH = "NO_MATCH"
    EXECUTED = "EXECUTED"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    REJECTED = "REJECTED"
    PROVIDER_FAILED = "PROVIDER_FAILED"


@dataclass(frozen=True, slots=True)
class IntentResolution:
    status: IntentResolutionStatus
    candidate: IntentCandidate | None = None
    candidate_id: str | None = None


def allowed_intents_for(
    status: MeetingStatus,
    *,
    actor_user_id: int,
) -> frozenset[RuleIntent]:
    """Return only intents the actor may request in the current public state."""

    allowed = {RuleIntent.REQUEST_SPEAK}
    if actor_user_id in {
        status.session.creator_user_id,
        status.speaking.current_user_id,
    }:
        allowed.add(RuleIntent.SPEAKER_FINISH)
    if status.vote is not None:
        allowed.update(
            {
                RuleIntent.VOTE_YES,
                RuleIntent.VOTE_NO,
                RuleIntent.VOTE_ABSTAIN,
            }
        )
    if actor_user_id == status.session.creator_user_id:
        allowed.add(RuleIntent.MEETING_STOP)
        if status.session.state is MeetingState.PAUSED:
            allowed.add(RuleIntent.MEETING_RESUME)
        else:
            allowed.update(
                {
                    RuleIntent.MEETING_PAUSE,
                    RuleIntent.MEETING_NEXT,
                    RuleIntent.MEETING_BACK,
                    RuleIntent.MEETING_SKIP,
                }
            )
    return frozenset(allowed)


class MeetingIntentService:
    """Keep LLM output advisory and route accepted intents through controller guards."""

    def __init__(
        self,
        *,
        controller: MeetingController,
        candidates: IntentCandidateRepository,
        rules: RuleIntentResolver,
        provider: StructuredIntentProvider | None,
        rule_intents_enabled: bool = True,
        llm_intents_enabled: bool = False,
        confidence_threshold: float = 0.80,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self._controller = controller
        self._candidates = candidates
        self._rules = rules
        self._provider = provider
        self._rules_enabled = rule_intents_enabled
        self._llm_enabled = llm_intents_enabled
        self._threshold = confidence_threshold
        self._log = structlog.get_logger(__name__)

    async def resolve_and_execute(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        text: str,
        utterance_id: str | None = None,
        source: str = "TEXT_COMMAND",
    ) -> IntentResolution:
        status = await self._controller.status(guild_id)
        if self._rules_enabled:
            rule_candidate = self._rules.resolve(
                text,
                actor_user_id=actor_user_id,
                host_user_id=status.session.creator_user_id,
                source=f"{source}_RULE",
            )
            if rule_candidate is not None:
                await self._execute(guild_id=guild_id, candidate=rule_candidate)
                return IntentResolution(
                    status=IntentResolutionStatus.EXECUTED,
                    candidate=rule_candidate,
                )

        if not self._llm_enabled or self._provider is None:
            return IntentResolution(status=IntentResolutionStatus.NO_MATCH)

        allowed = allowed_intents_for(status, actor_user_id=actor_user_id)
        try:
            proposed = await self._provider.resolve(
                text,
                status.session.state,
                allowed,
            )
            if proposed is None:
                return IntentResolution(status=IntentResolutionStatus.NO_MATCH)
            if proposed.intent not in allowed:
                raise StructuredIntentProviderError(
                    "provider candidate is outside the application allowlist"
                )
        except StructuredIntentProviderError as exc:
            self._log.warning(
                "structured_intent_provider_failed",
                guild_id=guild_id,
                error_type=type(exc).__name__,
            )
            return IntentResolution(status=IntentResolutionStatus.PROVIDER_FAILED)

        candidate = IntentCandidate(
            intent=proposed.intent,
            actor_user_id=actor_user_id,
            source=f"{source}_LLM",
            confidence=proposed.confidence,
        )
        record = await self._candidates.add(
            session_id=status.session.id,
            utterance_id=utterance_id,
            intent_type=candidate.intent.value,
            actor_user_id=actor_user_id,
            source=candidate.source,
            confidence=candidate.confidence,
        )
        if candidate.confidence < self._threshold:
            return IntentResolution(
                status=IntentResolutionStatus.CONFIRMATION_REQUIRED,
                candidate=candidate,
                candidate_id=record.id,
            )

        await self._candidates.decide(
            candidate_id=record.id,
            session_id=status.session.id,
            actor_user_id=actor_user_id,
            accepted=True,
        )
        try:
            await self._execute(guild_id=guild_id, candidate=candidate)
        except Exception:
            await self._candidates.mark_execution_failed(record.id)
            raise
        return IntentResolution(
            status=IntentResolutionStatus.EXECUTED,
            candidate=candidate,
            candidate_id=record.id,
        )

    async def confirm(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        candidate_id: str,
    ) -> IntentResolution:
        status = await self._controller.status(guild_id)
        record = await self._candidates.get(candidate_id)
        if record is None:
            from lingxu.domain.errors import IntentConfirmationError

            raise IntentConfirmationError("intent candidate was not found")
        intent = RuleIntent(record.intent_type)
        if intent not in allowed_intents_for(status, actor_user_id=actor_user_id):
            from lingxu.domain.errors import IntentConfirmationError

            raise IntentConfirmationError("intent is no longer allowed in the current state")
        await self._candidates.decide(
            candidate_id=candidate_id,
            session_id=status.session.id,
            actor_user_id=actor_user_id,
            accepted=True,
        )
        candidate = IntentCandidate(
            intent=intent,
            actor_user_id=actor_user_id,
            source=f"{record.source}_CONFIRMED",
            confidence=record.confidence or 0.0,
        )
        try:
            await self._execute(guild_id=guild_id, candidate=candidate)
        except Exception:
            await self._candidates.mark_execution_failed(candidate_id)
            raise
        return IntentResolution(
            status=IntentResolutionStatus.EXECUTED,
            candidate=candidate,
            candidate_id=candidate_id,
        )

    async def reject(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        candidate_id: str,
    ) -> IntentResolution:
        session = await self._controller.active(guild_id)
        record = await self._candidates.decide(
            candidate_id=candidate_id,
            session_id=session.id,
            actor_user_id=actor_user_id,
            accepted=False,
        )
        return IntentResolution(
            status=IntentResolutionStatus.REJECTED,
            candidate=IntentCandidate(
                intent=RuleIntent(record.intent_type),
                actor_user_id=actor_user_id,
                source=record.source,
                confidence=record.confidence or 0.0,
            ),
            candidate_id=candidate_id,
        )

    async def _execute(self, *, guild_id: int, candidate: IntentCandidate) -> None:
        arguments = {
            "guild_id": guild_id,
            "actor_user_id": candidate.actor_user_id,
        }
        if candidate.intent is RuleIntent.REQUEST_SPEAK:
            await self._controller.speak(
                guild_id=guild_id,
                user_id=candidate.actor_user_id,
                source=candidate.source,
            )
        elif candidate.intent is RuleIntent.SPEAKER_FINISH:
            await self._controller.speaker_finish(source=candidate.source, **arguments)
        elif candidate.intent is RuleIntent.VOTE_YES:
            await self._controller.vote_cast(
                guild_id=guild_id,
                user_id=candidate.actor_user_id,
                option_id="yes",
                source=candidate.source,
            )
        elif candidate.intent is RuleIntent.VOTE_NO:
            await self._controller.vote_cast(
                guild_id=guild_id,
                user_id=candidate.actor_user_id,
                option_id="no",
                source=candidate.source,
            )
        elif candidate.intent is RuleIntent.VOTE_ABSTAIN:
            await self._controller.vote_cast(
                guild_id=guild_id,
                user_id=candidate.actor_user_id,
                option_id="abstain",
                source=candidate.source,
            )
        elif candidate.intent is RuleIntent.MEETING_PAUSE:
            await self._controller.pause(source=candidate.source, **arguments)
        elif candidate.intent is RuleIntent.MEETING_RESUME:
            await self._controller.resume(source=candidate.source, **arguments)
        elif candidate.intent is RuleIntent.MEETING_NEXT:
            await self._controller.next(source=candidate.source, **arguments)
        elif candidate.intent is RuleIntent.MEETING_BACK:
            await self._controller.back(source=candidate.source, **arguments)
        elif candidate.intent is RuleIntent.MEETING_SKIP:
            await self._controller.skip(source=candidate.source, **arguments)
        elif candidate.intent is RuleIntent.MEETING_STOP:
            await self._controller.stop(source=candidate.source, **arguments)
