"""Anonymous vote service that never writes the Discord user id to vote entries."""

from __future__ import annotations

from datetime import datetime

from lingxu.domain.enums import MeetingEventType
from lingxu.domain.errors import VoteNotOpenError
from lingxu.domain.models import VoteRecord
from lingxu.meeting.components import VoteOutcome, VotingService
from lingxu.scenario.privacy import AnonymousVoteTokenService
from lingxu.storage.component_repositories import (
    ParticipantRepository,
    TimerRepository,
    VoteRepository,
)
from lingxu.storage.repositories import MeetingEventStore


class AnonymousVotingService:
    OPTIONS = VotingService.OPTIONS

    def __init__(
        self,
        *,
        votes: VoteRepository,
        timers: TimerRepository,
        participants: ParticipantRepository,
        events: MeetingEventStore,
        tokens: AnonymousVoteTokenService,
    ) -> None:
        self._votes = votes
        self._timers = timers
        self._participants = participants
        self._events = events
        self._tokens = tokens

    async def open(
        self,
        *,
        session_id: str,
        title: str,
        duration_seconds: float,
        actor_user_id: int,
        now: datetime | None = None,
    ) -> VoteRecord:
        async with self._votes.uow.begin() as transaction:
            existing = await self._votes.active_for_session(
                session_id,
                transaction=transaction,
            )
            if existing is not None:
                return existing
            vote = await self._votes.open(
                session_id=session_id,
                title=title,
                options=self.OPTIONS,
                duration_seconds=duration_seconds,
                anonymous=True,
                now=now,
                transaction=transaction,
            )
            timer = await self._timers.start(
                session_id=session_id,
                timer_type=f"VOTE:{vote.id}",
                duration_seconds=duration_seconds,
                now=now,
                transaction=transaction,
            )
            await transaction.append_event(
                session_id=session_id,
                event_type=MeetingEventType.VOTE_OPENED.value,
                actor_user_id=actor_user_id,
                source="SCENARIO",
                payload={"vote_id": vote.id, "anonymous": True},
            )
            await transaction.append_event(
                session_id=session_id,
                event_type=MeetingEventType.TIMER_STARTED.value,
                source="SYSTEM",
                payload={
                    "timer_id": timer.id,
                    "timer_type": timer.timer_type,
                    "deadline_at": timer.deadline_at.isoformat(),
                    "duration_seconds": duration_seconds,
                },
            )
            return vote

    async def cast(
        self,
        *,
        session_id: str,
        vote_id: str,
        user_id: int,
        option_id: str,
        now: datetime | None = None,
    ) -> str:
        async with self._votes.uow.begin() as transaction:
            vote = await self._votes.get(vote_id, transaction=transaction)
            if vote.session_id != session_id:
                raise VoteNotOpenError("vote is not open for this Meeting Session")
            await self._participants.require_present(
                session_id=vote.session_id,
                user_id=user_id,
                transaction=transaction,
            )
            voter = self._tokens.issue(
                session_id=vote.session_id,
                vote_id=vote_id,
                user_id=user_id,
            )
            selected = await self._votes.cast_anonymous(
                vote_id=vote_id,
                pseudonymous_user_id=voter.pseudonymous_user_id,
                voter_token_hash=voter.token_hash,
                option_id=option_id,
                now=now,
                transaction=transaction,
            )
            await transaction.append_event(
                session_id=vote.session_id,
                event_type=MeetingEventType.VOTE_CAST.value,
                actor_user_id=None,
                source="SCENARIO",
                payload={"vote_id": vote_id, "anonymous": True},
            )
            return selected

    async def close(self, vote_id: str) -> VoteOutcome:
        async with self._votes.uow.begin() as transaction:
            result = await self._votes.close(vote_id, transaction=transaction)
            outcome = VoteOutcome(
                result.vote,
                result.counts,
                result.eligible_count,
                result.changed,
            )
            if result.changed:
                await transaction.append_event(
                    session_id=result.vote.session_id,
                    event_type=MeetingEventType.VOTE_CLOSED.value,
                    source="SCENARIO",
                    payload={
                        "vote_id": vote_id,
                        "anonymous": True,
                        "counts": result.counts,
                        "total": outcome.total,
                    },
                )
            return outcome
