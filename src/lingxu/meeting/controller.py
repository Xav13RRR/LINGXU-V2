"""Application-level orchestration for Meeting Runtime and Phase 7 components."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from lingxu.domain.enums import (
    AgendaSpeakingMode,
    AgendaStatus,
    ConsentKind,
    JobStatus,
    MeetingEventType,
    MeetingState,
    SpeakingFlowStage,
    VoteStatus,
)
from lingxu.domain.errors import MeetingPermissionError, SessionNotFoundError
from lingxu.domain.models import (
    AgendaItemRecord,
    MeetingEventRecord,
    MeetingSessionRecord,
    TimerRecord,
    VoteRecipientRecord,
    VoteRecord,
)
from lingxu.domain.speaking import AgendaSpeakingPolicy, can_control_session
from lingxu.meeting.components import (
    AgendaService,
    SpeakingQueueService,
    SpeakingQueueState,
    TimerNotice,
    TimerService,
    VoteOutcome,
    VotingService,
)
from lingxu.meeting.mute_enforcement import MuteEnforcementService, MuteEnforcementState
from lingxu.meeting.runtime import MeetingRuntime, RuntimeResult
from lingxu.meeting.speaking_flow import (
    ControlledSpeakingFlowService,
    SpeakingFlowNotice,
    SpeakingFlowState,
    SpeakingFlowUpdate,
)
from lingxu.storage.component_repositories import (
    ConsentChangeRepository,
    GuildConsentRepository,
    ParticipantRepository,
)
from lingxu.storage.repositories import (
    MeetingEventStore,
    MeetingSessionRepository,
    ProcessingJobRepository,
)
from lingxu.storage.services import MeetingSessionService


@dataclass(frozen=True, slots=True)
class MeetingStatus:
    session: MeetingSessionRecord
    agenda: tuple[AgendaItemRecord, ...]
    speaking: SpeakingQueueState
    vote: VoteRecord | None
    floor: SpeakingFlowState | None = None
    timers: tuple[TimerRecord, ...] = ()
    enforcement: MuteEnforcementState | None = None


RuntimeActionHandler = Callable[[MeetingSessionRecord, str], Awaitable[None]]
VoteOutcomeHandler = Callable[[MeetingSessionRecord, VoteOutcome], Awaitable[None]]
SpeakingFlowHandler = Callable[[MeetingSessionRecord, SpeakingFlowNotice], Awaitable[None]]
EnforcementHandler = Callable[[MeetingSessionRecord], Awaitable[bool]]

_AUTOMATIC_RECOVERY_STATES = frozenset(
    {
        MeetingState.OPENING,
        MeetingState.AGENDA_OVERVIEW,
        MeetingState.AGENDA_ITEM_INTRO,
        MeetingState.DISCUSSION_READY,
        MeetingState.SPEAKER_ACTIVE,
        MeetingState.DISCUSSION_COMPLETE,
        MeetingState.VOTE_OPEN,
        MeetingState.VOTE_RESULT,
        MeetingState.NEXT_AGENDA_ITEM,
        MeetingState.CLOSING,
        MeetingState.COMPLETED,
        MeetingState.POST_PROCESSING,
    }
)


class ConsentRuntime(Protocol):
    async def timeline_now_ms(self, *, guild_id: int, user_id: int) -> int: ...

    async def apply_withdrawal(
        self,
        *,
        guild_id: int,
        user_id: int,
        effective_at_ms: int,
    ) -> None: ...

    async def sync_user(self, *, guild_id: int, user_id: int) -> None: ...

    async def suspend_user(self, *, guild_id: int, user_id: int) -> None: ...


class AgendaImportGuard(Protocol):
    async def has_processing(self, session_id: str) -> bool: ...


class MeetingController:
    """A thin authorization and side-effect layer used by Discord adapters."""

    def __init__(
        self,
        *,
        sessions: MeetingSessionRepository,
        events: MeetingEventStore,
        session_service: MeetingSessionService,
        runtime: MeetingRuntime,
        participants: ParticipantRepository,
        guild_consents: GuildConsentRepository,
        consent_changes: ConsentChangeRepository,
        agenda: AgendaService,
        speaking: SpeakingQueueService,
        timers: TimerService,
        voting: VotingService,
        speaking_flow: ControlledSpeakingFlowService | None = None,
        default_vote_seconds: int = 60,
        jobs: ProcessingJobRepository | None = None,
        agenda_import_guard: AgendaImportGuard | None = None,
        mute_enforcement: MuteEnforcementService | None = None,
    ) -> None:
        self._sessions = sessions
        self._events = events
        self._session_service = session_service
        self._runtime = runtime
        self._participants = participants
        self._guild_consents = guild_consents
        self._consent_changes = consent_changes
        self._agenda = agenda
        self._speaking = speaking
        self._timers = timers
        self._voting = voting
        self._speaking_flow = speaking_flow
        self._default_vote_seconds = default_vote_seconds
        self._jobs = jobs
        self._agenda_import_guard = agenda_import_guard
        self._mute_enforcement = mute_enforcement
        self._consent_locks: dict[int, asyncio.Lock] = {}
        self._flow_locks: dict[str, asyncio.Lock] = {}
        self._runtime_action_handler: RuntimeActionHandler | None = None
        self._vote_outcome_handler: VoteOutcomeHandler | None = None
        self._speaking_flow_handler: SpeakingFlowHandler | None = None
        self._enforcement_handler: EnforcementHandler | None = None
        self._consent_runtime: ConsentRuntime | None = None

    def set_runtime_action_handler(self, handler: RuntimeActionHandler) -> None:
        self._runtime_action_handler = handler

    def set_vote_outcome_handler(self, handler: VoteOutcomeHandler) -> None:
        self._vote_outcome_handler = handler

    def set_speaking_flow_handler(self, handler: SpeakingFlowHandler) -> None:
        self._speaking_flow_handler = handler

    def set_enforcement_handler(self, handler: EnforcementHandler) -> None:
        self._enforcement_handler = handler

    async def _sync_enforcement(
        self,
        session: MeetingSessionRecord,
        *,
        require_complete: bool = False,
    ) -> bool:
        if self._enforcement_handler is None:
            return True
        complete = await self._enforcement_handler(session)
        if require_complete and not complete:
            raise ValueError(
                "强制静音操作已记录, 但 Discord 权限不足或租约状态不确定; "
                "请恢复 Mute Members 权限后重试, 或人工检查成员静音。"
            )
        return complete

    def set_consent_runtime(self, runtime: ConsentRuntime) -> None:
        self._consent_runtime = runtime

    def _consent_lock(self, guild_id: int) -> asyncio.Lock:
        return self._consent_locks.setdefault(guild_id, asyncio.Lock())

    def _flow_lock(self, session_id: str) -> asyncio.Lock:
        return self._flow_locks.setdefault(session_id, asyncio.Lock())

    async def _recover_current_actions_before_navigation(
        self,
        session: MeetingSessionRecord,
    ) -> None:
        """Finish the current transition's actions before a host changes workflow state."""

        await self._recover_pending_current_actions_locked(session)

    async def _recover_pending_current_actions_locked(
        self,
        session: MeetingSessionRecord,
    ) -> RuntimeResult | None:
        """Replay unacknowledged entry/resume batches in the current workflow epoch."""

        batches = await self._runtime.recover_action_batches(session.id)
        if not batches:
            return None
        completed = await self._events.list_types(
            session.id,
            (MeetingEventType.RUNTIME_ACTION_COMPLETED.value,),
        )
        completed_by_transition: dict[str, set[str]] = {}
        for event in completed:
            transition_id = event.payload.get("transition_event_id")
            action = event.payload.get("action")
            if not isinstance(transition_id, str) or not isinstance(action, str):
                continue
            completed_by_transition.setdefault(transition_id, set()).add(action)
        latest: RuntimeResult | None = None
        for batch in batches:
            if batch.event is None or not batch.actions:
                continue
            completed_actions = completed_by_transition.get(batch.event.id, set())
            if all(action in completed_actions for action in batch.actions):
                continue
            latest = await self._execute_actions_locked(
                batch,
                continue_automatic_flow=False,
            )
        return latest

    async def _advance_if_consented_locked(self, guild_id: int) -> RuntimeResult | None:
        session = await self._sessions.active_for_guild(guild_id)
        if session is None or session.state is not MeetingState.CONSENT_CHECK:
            return None
        if not await self._participants.all_present_fully_consented(session.id):
            return None
        return await self._execute_actions_locked(
            await self._runtime.consents_satisfied(session.id)
        )

    async def _advance_if_consented(self, guild_id: int) -> RuntimeResult | None:
        async with self._consent_lock(guild_id):
            return await self._advance_if_consented_locked(guild_id)

    async def active(self, guild_id: int) -> MeetingSessionRecord:
        session = await self._sessions.active_for_guild(guild_id)
        if session is None:
            raise SessionNotFoundError("this guild has no active Meeting")
        return session

    async def _active_public_vote(
        self,
        session: MeetingSessionRecord,
    ) -> VoteRecord | None:
        vote = await self._voting.active_for_session(session.id)
        return vote if vote is not None and vote.agenda_item_id is None else None

    async def _require_no_public_vote_for_navigation(
        self,
        session: MeetingSessionRecord,
    ) -> None:
        if await self._active_public_vote(session) is not None:
            raise ValueError("公开投票进行中不能切换会议流程; 请先关闭该投票。")

    async def host_session(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
    ) -> MeetingSessionRecord:
        session = await self.active(guild_id)
        self._require_host(session, actor_user_id)
        return session

    async def set_publication_channel(
        self,
        *,
        guild_id: int,
        channel_id: int,
    ) -> MeetingSessionRecord:
        session = await self.active(guild_id)
        return await self._sessions.set_publication_channel(
            session_id=session.id,
            channel_id=channel_id,
        )

    @staticmethod
    def _require_host(session: MeetingSessionRecord, actor_user_id: int) -> None:
        if not can_control_session(
            actor_user_id=actor_user_id,
            host_user_id=session.creator_user_id,
        ):
            raise MeetingPermissionError("only the Meeting host may perform this operation")

    async def create(
        self,
        *,
        guild_id: int,
        host_user_id: int,
        publication_channel_id: int | None = None,
        setup_channel_id: int | None = None,
        host_display_name: str | None = None,
    ) -> MeetingSessionRecord:
        session = await self._session_service.create(
            guild_id=guild_id,
            creator_user_id=host_user_id,
        )
        if publication_channel_id is not None:
            session = await self._sessions.set_publication_channel(
                session_id=session.id,
                channel_id=publication_channel_id,
            )
        if setup_channel_id is not None:
            session = await self._sessions.set_setup_channel(
                session_id=session.id,
                channel_id=setup_channel_id,
            )
        grants = await self._guild_consents.grants_for(
            guild_id=guild_id,
            user_id=host_user_id,
        )
        await self._participants.join(
            session_id=session.id,
            user_id=host_user_id,
            inherited_consents=frozenset(),
            display_name=host_display_name,
            is_host=True,
        )
        result = await self._runtime.initialise(
            session.id,
            actor_user_id=host_user_id,
            source="SLASH_COMMAND",
        )
        if grants == frozenset(ConsentKind):
            await self._grant_all(
                session_id=session.id,
                guild_id=guild_id,
                user_id=host_user_id,
                effective_at_ms=0,
                trigger="inherited_guild_grant",
            )
        elif grants:
            await self._participants.set_consents(
                session_id=session.id,
                user_id=host_user_id,
                grants=grants,
            )
        return await self._sessions.get(result.session.id)

    async def status(self, guild_id: int) -> MeetingStatus:
        session = await self.active(guild_id)
        return await self.status_for_session(session.id)

    async def status_for_session(self, session_id: str) -> MeetingStatus:
        """Project status for active or terminal Sessions used by durable panels."""

        session = await self._sessions.get(session_id)
        agenda = await self._agenda.list(session.id)
        active_item = next(
            (item for item in agenda if item.status is AgendaStatus.ACTIVE),
            None,
        )
        return MeetingStatus(
            session=session,
            agenda=agenda,
            speaking=await self._speaking.state(session.id),
            vote=await self._voting.active_for_session(session.id),
            floor=(
                await self._speaking_flow.state(session.id, active_item.id)
                if self._speaking_flow is not None and active_item is not None
                else None
            ),
            timers=await self._timers.active(session.id),
            enforcement=(
                await self._mute_enforcement.state(session.id, active_item.id)
                if self._mute_enforcement is not None and active_item is not None
                else None
            ),
        )

    async def _dispatch_speaking_update_locked(
        self,
        session: MeetingSessionRecord,
        update: SpeakingFlowUpdate,
    ) -> RuntimeResult | None:
        if self._enforcement_handler is not None:
            await self._sync_enforcement(session)
        if self._speaking_flow_handler is not None:
            for notice in update.notices:
                await self._speaking_flow_handler(session, notice)
        if (
            update.state is not None
            and update.state.stage is SpeakingFlowStage.COMPLETED
            and session.state is MeetingState.DISCUSSION_READY
        ):
            following = await self._runtime.auto_transition(
                session.id,
                MeetingState.DISCUSSION_COMPLETE,
            )
            return await self._execute_actions_locked(following)
        return None

    async def sync_participants(
        self,
        *,
        guild_id: int,
        user_ids: set[int],
        display_names: dict[int, str] | None = None,
    ) -> None:
        """Make persisted presence match the selected Meeting voice channel."""

        session = await self.active(guild_id)
        present = set(await self._participants.present_user_ids(session.id))
        for user_id in sorted(user_ids - present):
            await self.participant_join(
                guild_id=guild_id,
                user_id=user_id,
                display_name=(display_names or {}).get(user_id),
            )
        for user_id in sorted(present - user_ids):
            await self.participant_leave(guild_id=guild_id, user_id=user_id)

    async def _execute_actions_locked(
        self,
        result: RuntimeResult,
        *,
        vote_outcome: VoteOutcome | None = None,
        continue_automatic_flow: bool = True,
    ) -> RuntimeResult:
        """Execute actions only while the caller owns this Guild's consent lock.

        Recursive automatic progression stays in this helper to avoid reacquiring the
        non-reentrant asyncio lock. Callers must acquire the Guild lock before selecting the
        Runtime transition, not only before consuming its actions.
        """

        selected_outcome = vote_outcome
        transition_event_id = result.event.id if result.event is not None else None
        completed_actions = (
            {
                str(event.payload["action"])
                for event in await self._events.list_types(
                    result.session.id,
                    (MeetingEventType.RUNTIME_ACTION_COMPLETED.value,),
                )
                if event.payload.get("transition_event_id") == transition_event_id
                and isinstance(event.payload.get("action"), str)
            }
            if transition_event_id is not None
            else set()
        )
        if result.session.state is MeetingState.VOTE_OPEN and "open_vote" in completed_actions:
            epoch_vote = await self._current_epoch_agenda_vote(result.session)
            if epoch_vote is None:
                completed_actions.discard("open_vote")
                completed_actions.discard("start_vote_timer")
        if result.session.state is MeetingState.AGENDA_ITEM_INTRO:
            await self._agenda.activate_next(session_id=result.session.id)
        for action in result.actions:
            if action in completed_actions:
                continue
            if action == "freeze_timers":
                await self._timers.pause(result.session.id)
            elif action == "resume_timers":
                await self._timers.resume(result.session.id)
            elif action == "cancel_timers":
                active_vote = await self._voting.active_for_session(result.session.id)
                if active_vote is not None:
                    await self._voting.close(
                        active_vote.id,
                        actor_user_id=None,
                        source="MEETING_STOPPED",
                    )
                await self._timers.cancel(result.session.id)
            elif action == "advance_agenda":
                await self._agenda.complete_active(session_id=result.session.id)
            elif action == "open_vote":
                active_vote = await self._voting.active_for_session(result.session.id)
                agenda = await self._agenda.list(result.session.id)
                active_item = next(
                    (item for item in agenda if item.status is AgendaStatus.ACTIVE),
                    None,
                )
                if active_item is None:
                    raise RuntimeError("VOTE_OPEN has no active agenda item")
                epoch_vote = await self._agenda_vote_in_current_open_epoch(
                    result.session.id,
                    agenda_item_id=active_item.id,
                )
                if active_vote is not None and (
                    epoch_vote is None
                    or active_vote.id != epoch_vote.id
                    or active_vote.agenda_item_id != active_item.id
                ):
                    if active_vote.agenda_item_id is None:
                        raise RuntimeError(
                            "VOTE_OPEN is blocked by a vote outside the current agenda epoch"
                        )
                    await self._voting.close(
                        active_vote.id,
                        actor_user_id=None,
                        source="STALE_EPOCH_OPEN_RECOVERY",
                    )
                    await self._timers.cancel_type(
                        session_id=result.session.id,
                        timer_type=f"VOTE:{active_vote.id}",
                    )
                    active_vote = None
                if epoch_vote is None:
                    title = active_item.vote_content or active_item.title
                    await self._voting.open(
                        session_id=result.session.id,
                        title=title,
                        duration_seconds=(
                            active_item.vote_duration_seconds or self._default_vote_seconds
                        ),
                        actor_user_id=result.session.creator_user_id,
                        source="SYSTEM",
                        agenda_item_id=active_item.id,
                        eligible_user_ids=await self._participants.present_user_ids(
                            result.session.id
                        ),
                    )
            elif action == "close_vote":
                if selected_outcome is None:
                    selected_outcome = await self._bound_vote_result_outcome(
                        result.session.id,
                        transition_event=result.event,
                    )
                    await self._timers.cancel_type(
                        session_id=result.session.id,
                        timer_type=f"VOTE:{selected_outcome.vote.id}",
                    )
            elif action == "trigger_post_meeting_job" and self._jobs is not None:
                await self._jobs.create(
                    session_id=result.session.id,
                    job_type="POST_MEETING",
                    idempotency_key="post-meeting-v1",
                )
            if action == "speak_vote_result":
                if selected_outcome is None:
                    selected_outcome = await self._bound_vote_result_outcome(
                        result.session.id,
                        transition_event=result.event,
                    )
                if self._vote_outcome_handler is not None:
                    await self._vote_outcome_handler(result.session, selected_outcome)
            elif self._runtime_action_handler is not None:
                await self._runtime_action_handler(result.session, action)
            if transition_event_id is not None:
                await self._events.append(
                    session_id=result.session.id,
                    event_type=MeetingEventType.RUNTIME_ACTION_COMPLETED.value,
                    payload={
                        "transition_event_id": transition_event_id,
                        "action": action,
                    },
                )
                completed_actions.add(action)
        if self._enforcement_handler is not None:
            await self._sync_enforcement(result.session)
        if not continue_automatic_flow:
            return result
        return await self._continue_automatic_actions_locked(result)

    async def _bound_vote_result_outcome(
        self,
        session_id: str,
        *,
        transition_event: MeetingEventRecord | None = None,
    ) -> VoteOutcome:
        """Bind VOTE_RESULT recovery to the vote opened in its preceding state epoch."""

        selected_transition = transition_event
        if (
            selected_transition is None
            or selected_transition.state_after is not MeetingState.VOTE_RESULT
            or selected_transition.state_before is selected_transition.state_after
            or selected_transition.state_before in {MeetingState.PAUSED, MeetingState.INTERRUPTED}
        ):
            selected_transition = next(
                (
                    event
                    for event in reversed(await self._events.list(session_id))
                    if event.state_before is not event.state_after
                    and event.state_after is MeetingState.VOTE_RESULT
                    and event.state_before not in {MeetingState.PAUSED, MeetingState.INTERRUPTED}
                ),
                None,
            )
        if selected_transition is None:
            raise RuntimeError("VOTE_RESULT has no durable state transition")
        active_item = next(
            (
                item
                for item in await self._agenda.list(session_id)
                if item.status is AgendaStatus.ACTIVE
            ),
            None,
        )
        if active_item is None:
            raise RuntimeError("VOTE_RESULT has no active agenda item")
        vote_open_transition = await self._latest_workflow_epoch_transition(
            session_id,
            MeetingState.VOTE_OPEN,
            before_sequence=selected_transition.sequence_number,
        )
        if vote_open_transition is None:
            raise RuntimeError("VOTE_RESULT has no preceding VOTE_OPEN transition")
        vote = await self._agenda_vote_opened_in_epoch(
            session_id,
            agenda_item_id=active_item.id,
            epoch_sequence=vote_open_transition.sequence_number,
            before_sequence=selected_transition.sequence_number,
        )
        if vote is None:
            raise RuntimeError("VOTE_RESULT has no matching durable agenda vote")
        return await self._voting.close(
            vote.id,
            actor_user_id=None,
            source="VOTE_RESULT_BINDING",
        )

    async def _agenda_vote_opened_in_epoch(
        self,
        session_id: str,
        *,
        agenda_item_id: str,
        epoch_sequence: int,
        before_sequence: int | None = None,
    ) -> VoteRecord | None:
        """Project the agenda vote opened inside one exact VOTE_OPEN epoch."""

        opened_events = await self._events.list_types(
            session_id,
            (MeetingEventType.VOTE_OPENED.value,),
        )
        for event in reversed(opened_events):
            if event.sequence_number <= epoch_sequence:
                continue
            if before_sequence is not None and event.sequence_number >= before_sequence:
                continue
            if event.payload.get("agenda_item_id") != agenda_item_id:
                continue
            vote_id = event.payload.get("vote_id")
            if not isinstance(vote_id, str) or not vote_id:
                continue
            vote = await self._voting.get(vote_id)
            if vote.session_id != session_id or vote.agenda_item_id != agenda_item_id:
                continue
            return vote
        return None

    async def _latest_workflow_epoch_transition(
        self,
        session_id: str,
        target: MeetingState,
        *,
        before_sequence: int | None = None,
    ) -> MeetingEventRecord | None:
        """Return the latest workflow entry into a state, excluding pause restoration."""

        return next(
            (
                event
                for event in reversed(await self._events.list(session_id))
                if event.state_before is not event.state_after
                and event.state_after is target
                and event.state_before not in {MeetingState.PAUSED, MeetingState.INTERRUPTED}
                and (before_sequence is None or event.sequence_number < before_sequence)
            ),
            None,
        )

    async def _agenda_vote_in_current_open_epoch(
        self,
        session_id: str,
        *,
        agenda_item_id: str,
    ) -> VoteRecord | None:
        """Find the agenda vote opened after the current VOTE_OPEN transition."""

        transition = await self._latest_workflow_epoch_transition(
            session_id,
            MeetingState.VOTE_OPEN,
        )
        if transition is None:
            return None
        return await self._agenda_vote_opened_in_epoch(
            session_id,
            agenda_item_id=agenda_item_id,
            epoch_sequence=transition.sequence_number,
        )

    async def _current_epoch_agenda_vote(
        self,
        session: MeetingSessionRecord,
    ) -> VoteRecord | None:
        active_item = next(
            (
                item
                for item in await self._agenda.list(session.id)
                if item.status is AgendaStatus.ACTIVE
            ),
            None,
        )
        if active_item is None:
            return None
        return await self._agenda_vote_in_current_open_epoch(
            session.id,
            agenda_item_id=active_item.id,
        )

    async def _continue_automatic_actions_locked(self, result: RuntimeResult) -> RuntimeResult:
        if self._runtime_action_handler is None or result.event is None:
            return result
        return await self._continue_automatic_flow(result)

    async def _continue_automatic_flow(self, result: RuntimeResult) -> RuntimeResult:
        """Continue deterministic states while the Guild consent lock remains owned."""

        session = result.session
        state = session.state
        if state is MeetingState.OPENING:
            opening_transition = await self._latest_workflow_epoch_transition(
                session.id,
                MeetingState.OPENING,
            )
            if opening_transition is None:
                raise RuntimeError("OPENING has no durable workflow entry transition")
            await self._timers.start_after_handoff_barrier(
                session_id=session.id,
                timer_type=f"FLOW:OPENING:{opening_transition.sequence_number}",
                duration_seconds=10,
            )
            return result
        if state is MeetingState.AGENDA_OVERVIEW:
            if await self._active_public_vote(session) is not None:
                return result
            agenda = await self._agenda.list(session.id)
            if not agenda:
                following = await self._runtime.auto_transition(session.id, MeetingState.CLOSING)
            else:
                following = await self._runtime.advance(session.id)
            return await self._execute_actions_locked(following)
        if state is MeetingState.AGENDA_ITEM_INTRO:
            return await self._execute_actions_locked(await self._runtime.advance(session.id))
        if state is MeetingState.DISCUSSION_READY:
            agenda = await self._agenda.list(session.id)
            active_item = next(
                (item for item in agenda if item.status is AgendaStatus.ACTIVE),
                None,
            )
            if (
                active_item is not None
                and active_item.speaking_policy.mode is not AgendaSpeakingMode.FREE
                and self._speaking_flow is not None
            ):
                update = await self._speaking_flow.activate(session=session, item=active_item)
                speaking_result = await self._dispatch_speaking_update_locked(session, update)
                return speaking_result or result
            if active_item is not None and active_item.duration_minutes is not None:
                await self._timers.start_after_handoff_barrier(
                    session_id=session.id,
                    timer_type=f"FLOW:AGENDA:{active_item.id}",
                    duration_seconds=active_item.duration_minutes * 60,
                )
            return result
        if state is MeetingState.DISCUSSION_COMPLETE:
            if await self._active_public_vote(session) is not None:
                return result
            agenda = await self._agenda.list(session.id)
            active_item = next(
                (item for item in agenda if item.status is AgendaStatus.ACTIVE),
                None,
            )
            if active_item is not None and active_item.requires_vote:
                following = await self._runtime.advance(session.id)
            else:
                following = await self._runtime.auto_transition(
                    session.id,
                    MeetingState.NEXT_AGENDA_ITEM,
                )
            return await self._execute_actions_locked(following)
        if state is MeetingState.VOTE_OPEN:
            active_vote = await self._voting.active_for_session(session.id)
            epoch_vote = await self._current_epoch_agenda_vote(session)
            if active_vote is not None and (epoch_vote is None or active_vote.id != epoch_vote.id):
                if active_vote.agenda_item_id is not None and await self._voting.all_eligible_voted(
                    active_vote.id
                ):
                    await self._voting.close(
                        active_vote.id,
                        actor_user_id=None,
                        source="STALE_EPOCH_RECOVERY",
                    )
                    await self._timers.cancel_type(
                        session_id=session.id,
                        timer_type=f"VOTE:{active_vote.id}",
                    )
                return result
            if epoch_vote is not None and await self._voting.all_eligible_voted(epoch_vote.id):
                outcome = await self._voting.close(
                    epoch_vote.id,
                    actor_user_id=None,
                    source="SYSTEM",
                )
                await self._timers.cancel_type(
                    session_id=session.id,
                    timer_type=f"VOTE:{epoch_vote.id}",
                )
                return await self._execute_actions_locked(
                    await self._runtime.advance(session.id),
                    vote_outcome=outcome,
                )
            return result
        if state is MeetingState.VOTE_RESULT:
            outcome = await self._bound_vote_result_outcome(
                session.id,
                transition_event=result.event,
            )
            await self._timers.start_after_handoff_barrier(
                session_id=session.id,
                timer_type=f"FLOW:VOTE_RESULT:{outcome.vote.id}",
                duration_seconds=5,
            )
            return result
        if state is MeetingState.NEXT_AGENDA_ITEM:
            if await self._active_public_vote(session) is not None:
                return result
            agenda = await self._agenda.list(session.id)
            pending = any(item.status is AgendaStatus.PENDING for item in agenda)
            if pending:
                following = await self._runtime.auto_transition(
                    session.id,
                    MeetingState.AGENDA_ITEM_INTRO,
                )
            else:
                following = await self._runtime.advance(session.id)
            return await self._execute_actions_locked(following)
        if state in {MeetingState.CLOSING, MeetingState.COMPLETED}:
            return await self._execute_actions_locked(await self._runtime.advance(session.id))
        return result

    async def _handle_timer_notice_locked(
        self,
        notice: TimerNotice,
        session: MeetingSessionRecord,
    ) -> RuntimeResult | None:
        if notice.timer_type.startswith("FLOW:SPEAKING_") and self._speaking_flow is not None:
            if session.state is not MeetingState.DISCUSSION_READY:
                return None
            update = await self._speaking_flow.handle_timer_notice(notice)
            return await self._dispatch_speaking_update_locked(session, update)
        if notice.event_type is not MeetingEventType.TIMER_EXPIRED:
            return None
        if notice.timer_type.startswith("FLOW:OPENING:"):
            if session.state is not MeetingState.OPENING:
                return None
            return await self._execute_actions_locked(await self._runtime.advance(session.id))
        if notice.timer_type.startswith("FLOW:AGENDA:"):
            if session.state is not MeetingState.DISCUSSION_READY:
                return None
            active_id = notice.timer_type.removeprefix("FLOW:AGENDA:")
            agenda = await self._agenda.list(session.id)
            if not any(
                item.id == active_id and item.status is AgendaStatus.ACTIVE for item in agenda
            ):
                return None
            following = await self._runtime.auto_transition(
                session.id,
                MeetingState.DISCUSSION_COMPLETE,
            )
            return await self._execute_actions_locked(following)
        if notice.timer_type.startswith("VOTE:"):
            vote_id = notice.timer_type.removeprefix("VOTE:")
            vote = await self._voting.get(vote_id)
            if vote.session_id != session.id:
                return None
            outcome = notice.vote_outcome
            if outcome is None:
                outcome = await self._voting.close(
                    vote_id,
                    actor_user_id=None,
                    source="TIMER_RECOVERY",
                )
            if vote.agenda_item_id is None:
                if self._vote_outcome_handler is not None:
                    await self._vote_outcome_handler(session, outcome)
                return None
            if session.state is not MeetingState.VOTE_OPEN:
                return None
            active_item = next(
                (
                    item
                    for item in await self._agenda.list(session.id)
                    if item.status is AgendaStatus.ACTIVE
                ),
                None,
            )
            if active_item is None or active_item.id != vote.agenda_item_id:
                return None
            epoch_vote = await self._agenda_vote_in_current_open_epoch(
                session.id,
                agenda_item_id=active_item.id,
            )
            if epoch_vote is None or epoch_vote.id != vote.id:
                return None
            return await self._execute_actions_locked(
                await self._runtime.advance(session.id),
                vote_outcome=outcome,
            )
        if notice.timer_type.startswith("FLOW:VOTE_RESULT:"):
            if session.state is not MeetingState.VOTE_RESULT:
                return None
            vote_id = notice.timer_type.removeprefix("FLOW:VOTE_RESULT:")
            outcome = await self._bound_vote_result_outcome(session.id)
            if outcome.vote.id != vote_id:
                return None
            return await self._execute_actions_locked(await self._runtime.advance(session.id))
        return None

    async def _consume_timer_notice_locked(
        self,
        notice: TimerNotice,
        session: MeetingSessionRecord,
    ) -> RuntimeResult | None:
        if not await self._timers.is_latest_timer_identity(notice):
            return None
        if await self._timers.handoff_is_completed(notice):
            return None
        if notice.event_type is MeetingEventType.TIMER_EXPIRED and session.state in {
            MeetingState.PAUSED,
            MeetingState.INTERRUPTED,
        }:
            return None
        result = await self._handle_timer_notice_locked(notice, session)
        await self._timers.complete_handoff(notice)
        return result

    async def _drain_timer_handoffs_locked(
        self,
        session_id: str,
    ) -> RuntimeResult | None:
        """Consume the current durable expiry set while flow and Guild locks are held."""

        latest: RuntimeResult | None = None
        for notice in await self._timers.durable_expiry_notices(session_id):
            current = await self._sessions.get(session_id)
            reconciled = await self._consume_timer_notice_locked(notice, current)
            if reconciled is not None:
                latest = reconciled
        return latest

    async def handle_timer_notice(self, notice: TimerNotice) -> RuntimeResult | None:
        """Turn an expired persisted flow timer into the next Runtime transition."""

        async with self._flow_lock(notice.session_id):
            session = await self._sessions.get(notice.session_id)
            async with self._consent_lock(session.guild_id):
                session = await self._sessions.get(notice.session_id)
                return await self._consume_timer_notice_locked(notice, session)

    async def _reconcile_timer_handoffs_for_session(
        self,
        session_id: str,
    ) -> RuntimeResult | None:
        session = await self._sessions.get(session_id)
        if session.state in {MeetingState.STOPPED, MeetingState.PUBLISHED}:
            terminal_recovered: RuntimeResult | None = None
            async with self._flow_lock(session_id):
                async with self._consent_lock(session.guild_id):
                    current = await self._sessions.get(session_id)
                    if current.state in {MeetingState.STOPPED, MeetingState.PUBLISHED}:
                        terminal_recovered = await self._recover_pending_current_actions_locked(
                            current
                        )
                        for notice in await self._timers.uncompleted_expiry_notices(session_id):
                            await self._timers.complete_handoff(notice)
            return terminal_recovered
        if session.state is MeetingState.PAUSED:
            paused_recovered: RuntimeResult | None = None
            async with self._flow_lock(session_id):
                async with self._consent_lock(session.guild_id):
                    current = await self._sessions.get(session_id)
                    if current.state is MeetingState.PAUSED:
                        paused_recovered = await self._recover_pending_current_actions_locked(
                            current
                        )
                        await self._timers.pause(session_id)
            return paused_recovered

        latest: RuntimeResult | None = None
        for notice in await self._timers.durable_expiry_notices(session_id):
            reconciled = await self.handle_timer_notice(notice)
            if reconciled is not None:
                latest = reconciled
        if latest is not None:
            return latest

        async with self._flow_lock(session_id):
            session = await self._sessions.get(session_id)
            async with self._consent_lock(session.guild_id):
                session = await self._sessions.get(session_id)
                recovered_actions: RuntimeResult | None = None
                if session.state in _AUTOMATIC_RECOVERY_STATES:
                    pending = await self._recover_pending_current_actions_locked(session)
                    if pending is not None:
                        recovered_actions = await self._continue_automatic_actions_locked(pending)
                        current = await self._sessions.get(session_id)
                        if current.state is not session.state:
                            return recovered_actions
                        session = current
                if (
                    session.state is MeetingState.DISCUSSION_READY
                    and self._speaking_flow is not None
                ):
                    controlled_item = next(
                        (
                            item
                            for item in await self._agenda.list(session.id)
                            if item.status is AgendaStatus.ACTIVE
                            and item.speaking_policy.mode is not AgendaSpeakingMode.FREE
                        ),
                        None,
                    )
                    if controlled_item is not None:
                        speaking_update = await self._speaking_flow.reconcile_current_timer(
                            session.id,
                            controlled_item.id,
                        )
                        speaking_result = await self._dispatch_speaking_update_locked(
                            session,
                            speaking_update,
                        )
                        if speaking_result is not None:
                            return speaking_result
                if session.state is MeetingState.VOTE_OPEN:
                    active_item = next(
                        (
                            item
                            for item in await self._agenda.list(session.id)
                            if item.status is AgendaStatus.ACTIVE
                        ),
                        None,
                    )
                    epoch_vote = (
                        await self._agenda_vote_in_current_open_epoch(
                            session.id,
                            agenda_item_id=active_item.id,
                        )
                        if active_item is not None
                        else None
                    )
                    if epoch_vote is not None and epoch_vote.status is VoteStatus.CLOSED:
                        outcome = await self._voting.close(
                            epoch_vote.id,
                            actor_user_id=None,
                            source="VOTE_CLOSE_RECOVERY",
                        )
                        await self._timers.cancel_type(
                            session_id=session.id,
                            timer_type=f"VOTE:{outcome.vote.id}",
                        )
                        return await self._execute_actions_locked(
                            await self._runtime.advance(session.id),
                            vote_outcome=outcome,
                        )
                if not await self._timer_actions_incomplete(session):
                    return recovered_actions
                return await self._execute_actions_locked(
                    await self._runtime.recover_actions(session_id)
                )

    async def _timer_actions_incomplete(self, session: MeetingSessionRecord) -> bool:
        if session.state in {
            MeetingState.AGENDA_OVERVIEW,
            MeetingState.AGENDA_ITEM_INTRO,
            MeetingState.DISCUSSION_COMPLETE,
            MeetingState.NEXT_AGENDA_ITEM,
            MeetingState.CLOSING,
            MeetingState.COMPLETED,
        }:
            return True
        active_timers = await self._timers.active(session.id)
        timer_types = {timer.timer_type for timer in active_timers}
        if session.state is MeetingState.OPENING:
            return not any(timer_type.startswith("FLOW:OPENING:") for timer_type in timer_types)
        if session.state is MeetingState.VOTE_OPEN:
            active_item = next(
                (
                    item
                    for item in await self._agenda.list(session.id)
                    if item.status is AgendaStatus.ACTIVE
                ),
                None,
            )
            if active_item is None:
                return False
            epoch_vote = await self._agenda_vote_in_current_open_epoch(
                session.id,
                agenda_item_id=active_item.id,
            )
            if epoch_vote is None or epoch_vote.status is VoteStatus.CLOSED:
                return True
            return f"VOTE:{epoch_vote.id}" not in timer_types
        if session.state is MeetingState.VOTE_RESULT:
            outcome = await self._bound_vote_result_outcome(session.id)
            return f"FLOW:VOTE_RESULT:{outcome.vote.id}" not in timer_types
        if session.state is not MeetingState.DISCUSSION_READY:
            return False
        active_item = next(
            (
                item
                for item in await self._agenda.list(session.id)
                if item.status is AgendaStatus.ACTIVE
            ),
            None,
        )
        return (
            active_item is not None
            and active_item.duration_minutes is not None
            and active_item.speaking_policy.mode is AgendaSpeakingMode.FREE
            and f"FLOW:AGENDA:{active_item.id}" not in timer_types
        )

    async def reconcile_timer_handoffs(
        self,
        *,
        session_ids: set[str] | None = None,
    ) -> tuple[RuntimeResult, ...]:
        """Retry committed timer handoffs and repair a paused-state crash window."""

        results: list[RuntimeResult] = []
        candidates = {
            session.id: session
            for session in (
                *await self._sessions.list_recoverable(),
                *await self._sessions.list_terminal_reconcilable(),
            )
        }
        for session in candidates.values():
            if (
                session_ids is not None
                and session.id not in session_ids
                and session.state not in {MeetingState.STOPPED, MeetingState.PUBLISHED}
            ):
                continue
            reconciled = await self._reconcile_timer_handoffs_for_session(session.id)
            if reconciled is not None:
                results.append(reconciled)
        return tuple(results)

    async def monitor_post_processing_jobs(self) -> tuple[RuntimeResult, ...]:
        """Reconcile independent Worker completion with persisted Meeting state."""

        if self._jobs is None:
            return ()
        results: list[RuntimeResult] = []
        for candidate in await self._sessions.list_post_processing_reconcilable():
            async with self._flow_lock(candidate.id):
                async with self._consent_lock(candidate.guild_id):
                    session = await self._sessions.get(candidate.id)
                    job = await self._jobs.latest_for_session(
                        session.id,
                        job_type="POST_MEETING",
                    )
                    if job is None:
                        continue
                    if (
                        session.state is MeetingState.FAILED_POST_PROCESSING
                        and job.status is JobStatus.COMPLETED
                    ):
                        selected = await self._runtime.post_processing_retry_completed(session.id)
                    elif (
                        session.state is MeetingState.POST_PROCESSING
                        and job.status is JobStatus.COMPLETED
                    ):
                        selected = await self._runtime.post_processing_completed(session.id)
                    elif (
                        session.state is MeetingState.POST_PROCESSING
                        and job.status is JobStatus.FAILED
                    ):
                        selected = await self._runtime.post_processing_failed(session.id)
                    else:
                        continue
                    results.append(await self._execute_actions_locked(selected))
        return tuple(results)

    async def recover_automatic_flows(
        self,
        *,
        session_ids: set[str] | None = None,
    ) -> tuple[RuntimeResult, ...]:
        """Resume idempotent process actions after a Bot restart."""

        reconciled = {
            result.session.id: result
            for result in await self.reconcile_timer_handoffs(session_ids=session_ids)
        }
        results: list[RuntimeResult] = list(reconciled.values())
        for candidate in await self._sessions.list_recoverable():
            if session_ids is not None and candidate.id not in session_ids:
                continue
            handoff_result = reconciled.get(candidate.id)
            if handoff_result is not None:
                continue
            async with self._flow_lock(candidate.id):
                async with self._consent_lock(candidate.guild_id):
                    session = await self._sessions.get(candidate.id)
                    if session.state not in _AUTOMATIC_RECOVERY_STATES:
                        continue
                    recovered = await self._recover_pending_current_actions_locked(session)
                    latest = recovered or await self._runtime.recover_actions(session.id)
                    results.append(await self._execute_actions_locked(latest))
        return tuple(results)

    async def ensure_start_ready(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
    ) -> MeetingSessionRecord:
        """Reject start before voice connection while document setup is still running."""

        session = await self.active(guild_id)
        self._require_host(session, actor_user_id)
        if self._agenda_import_guard is not None and await self._agenda_import_guard.has_processing(
            session.id
        ):
            raise ValueError("议程文档仍在处理中, 请等待机器人发送“准备完毕”。")
        return session

    async def start(self, *, guild_id: int, actor_user_id: int) -> RuntimeResult:
        async with self._consent_lock(guild_id):
            session = await self.ensure_start_ready(
                guild_id=guild_id,
                actor_user_id=actor_user_id,
            )
            result = await self._execute_actions_locked(
                await self._runtime.start(
                    session.id,
                    actor_user_id=actor_user_id,
                    source="SLASH_COMMAND",
                )
            )
            advanced = await self._advance_if_consented_locked(guild_id)
            return advanced or result

    async def next(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        source: str = "SLASH_COMMAND",
    ) -> RuntimeResult:
        async with self._consent_lock(guild_id):
            session = await self.active(guild_id)
            self._require_host(session, actor_user_id)
            await self._require_no_public_vote_for_navigation(session)
            await self._recover_current_actions_before_navigation(session)
            vote_outcome: VoteOutcome | None = None
            if session.state is MeetingState.VOTE_OPEN:
                active_item = next(
                    (
                        item
                        for item in await self._agenda.list(session.id)
                        if item.status is AgendaStatus.ACTIVE
                    ),
                    None,
                )
                if active_item is None:
                    raise RuntimeError("VOTE_OPEN has no active agenda item")
                epoch_vote = await self._agenda_vote_in_current_open_epoch(
                    session.id,
                    agenda_item_id=active_item.id,
                )
                if epoch_vote is None:
                    raise RuntimeError("VOTE_OPEN has no vote in the current agenda epoch")
                vote_outcome = await self._voting.close(
                    epoch_vote.id,
                    actor_user_id=actor_user_id,
                    source=source,
                )
            await self._abort_active_controlled_flow(
                session,
                actor_user_id=actor_user_id,
                source=source,
                completion_reason="AGENDA_ADVANCED",
            )
            await self._timers.cancel(session.id)
            if (
                self._runtime_action_handler is not None
                and session.state is MeetingState.DISCUSSION_READY
            ):
                return await self._execute_actions_locked(
                    await self._runtime.auto_transition(
                        session.id,
                        MeetingState.DISCUSSION_COMPLETE,
                    )
                )
            return await self._execute_actions_locked(
                await self._runtime.next(
                    session.id,
                    actor_user_id=actor_user_id,
                    source=source,
                ),
                vote_outcome=vote_outcome,
            )

    async def back(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        source: str = "SLASH_COMMAND",
    ) -> RuntimeResult:
        async with self._consent_lock(guild_id):
            session = await self.active(guild_id)
            self._require_host(session, actor_user_id)
            if session.state is MeetingState.VOTE_OPEN:
                raise ValueError("投票进行中不能返回; 请先关闭投票, 再从投票结果返回重投。")
            if session.state is MeetingState.NEXT_AGENDA_ITEM:
                raise ValueError("议程已进入换项阶段, 不能返回投票结果。")
            await self._require_no_public_vote_for_navigation(session)
            await self._recover_current_actions_before_navigation(session)
            if await self._active_controlled_agenda(session) is not None:
                raise ValueError("受控发言议程中不能返回, 请先结束当前发言或进入下一项。")
            return await self._execute_actions_locked(
                await self._runtime.back(
                    session.id,
                    actor_user_id=actor_user_id,
                    source=source,
                )
            )

    async def skip(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        source: str = "SLASH_COMMAND",
    ) -> RuntimeResult:
        async with self._consent_lock(guild_id):
            session = await self.active(guild_id)
            self._require_host(session, actor_user_id)
            await self._require_no_public_vote_for_navigation(session)
            await self._recover_current_actions_before_navigation(session)
            if await self._active_controlled_agenda(session) is not None:
                raise ValueError("受控发言议程中不能跳过固定状态, 请使用“跳过发言”或“下一项”。")
            return await self._execute_actions_locked(
                await self._runtime.skip(
                    session.id,
                    actor_user_id=actor_user_id,
                    source=source,
                )
            )

    async def pause(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        source: str = "SLASH_COMMAND",
    ) -> RuntimeResult:
        async with self._consent_lock(guild_id):
            session = await self.active(guild_id)
            result = await self._timers.freeze_for_transition(
                session.id,
                lambda: self._runtime.pause(
                    session.id,
                    actor_user_id=actor_user_id,
                    source=source,
                ),
            )
            return await self._execute_actions_locked(result)

    async def resume(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        source: str = "SLASH_COMMAND",
    ) -> RuntimeResult:
        session = await self.active(guild_id)
        async with self._flow_lock(session.id):
            async with self._consent_lock(guild_id):
                session = await self.active(guild_id)
                resumed = await self._execute_actions_locked(
                    await self._runtime.resume(
                        session.id,
                        actor_user_id=actor_user_id,
                        source=source,
                    ),
                    continue_automatic_flow=False,
                )
                recovered = await self._recover_pending_current_actions_locked(resumed.session)
                latest = await self._drain_timer_handoffs_locked(session.id)
                if latest is not None:
                    return latest
                continued = await self._continue_automatic_actions_locked(recovered or resumed)
                latest = await self._drain_timer_handoffs_locked(session.id)
                return latest or continued

    async def stop(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        source: str = "SLASH_COMMAND",
    ) -> RuntimeResult:
        async with self._consent_lock(guild_id):
            session = await self.active(guild_id)
            self._require_host(session, actor_user_id)
            await self._abort_active_controlled_flow(
                session,
                actor_user_id=actor_user_id,
                source=source,
                completion_reason="MEETING_STOPPED",
            )
            return await self._execute_actions_locked(
                await self._runtime.stop(
                    session.id,
                    actor_user_id=actor_user_id,
                    source=source,
                )
            )

    async def _active_controlled_agenda(
        self,
        session: MeetingSessionRecord,
    ) -> AgendaItemRecord | None:
        if session.state not in {MeetingState.DISCUSSION_READY, MeetingState.PAUSED}:
            return None
        return next(
            (
                item
                for item in await self._agenda.list(session.id)
                if item.status is AgendaStatus.ACTIVE
                and item.speaking_policy.mode is not AgendaSpeakingMode.FREE
            ),
            None,
        )

    async def _abort_active_controlled_flow(
        self,
        session: MeetingSessionRecord,
        *,
        actor_user_id: int,
        source: str,
        completion_reason: str,
    ) -> None:
        item = await self._active_controlled_agenda(session)
        if item is None or self._speaking_flow is None:
            return
        await self._speaking_flow.abort(
            session_id=session.id,
            agenda_item_id=item.id,
            actor_user_id=actor_user_id,
            source=source,
            completion_reason=completion_reason,
        )

    async def agenda_add(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        title: str,
        duration_minutes: int | None = None,
        requires_vote: bool = False,
        vote_content: str | None = None,
        vote_duration_seconds: int | None = None,
        language: str = "auto",
        speaking_policy: AgendaSpeakingPolicy | None = None,
    ) -> AgendaItemRecord:
        session = await self.active(guild_id)
        self._require_host(session, actor_user_id)
        selected_policy = speaking_policy or AgendaSpeakingPolicy.free()
        if selected_policy.mode is AgendaSpeakingMode.PRESENTATION and duration_minutes is None:
            raise ValueError("演讲议程必须设置流程时长。")
        return await self._agenda.add(
            session_id=session.id,
            title=title,
            duration_minutes=duration_minutes,
            requires_vote=requires_vote,
            vote_content=vote_content,
            vote_duration_seconds=vote_duration_seconds,
            language=language,
            speaking_policy=selected_policy,
            actor_user_id=actor_user_id,
            source="SLASH_COMMAND",
        )

    async def agenda_remove(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        position: int,
    ) -> bool:
        session = await self.active(guild_id)
        self._require_host(session, actor_user_id)
        return await self._agenda.remove(
            session_id=session.id,
            position=position,
            actor_user_id=actor_user_id,
            source="SLASH_COMMAND",
        )

    async def agenda_reorder(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        from_position: int,
        to_position: int,
    ) -> tuple[AgendaItemRecord, ...]:
        session = await self.active(guild_id)
        self._require_host(session, actor_user_id)
        return await self._agenda.reorder(
            session_id=session.id,
            from_position=from_position,
            to_position=to_position,
            actor_user_id=actor_user_id,
            source="SLASH_COMMAND",
        )

    async def speak(self, *, guild_id: int, user_id: int, source: str = "SLASH_COMMAND") -> bool:
        session = await self.active(guild_id)
        return await self._speaking.request(
            session_id=session.id,
            user_id=user_id,
            source=source,
        )

    async def speaker_next(self, *, guild_id: int, actor_user_id: int) -> int | None:
        session = await self.active(guild_id)
        self._require_host(session, actor_user_id)
        return await self._speaking.next(
            session_id=session.id,
            actor_user_id=actor_user_id,
            source="SLASH_COMMAND",
        )

    async def speaker_finish(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        source: str = "SLASH_COMMAND",
    ) -> int | None:
        session = await self.active(guild_id)
        state = await self._speaking.state(session.id)
        if actor_user_id not in {session.creator_user_id, state.current_user_id}:
            raise MeetingPermissionError("only the host or current speaker may finish the turn")
        return await self._speaking.finish(
            session_id=session.id,
            actor_user_id=actor_user_id,
            source=source,
        )

    async def speaker_remove(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        user_id: int,
    ) -> bool:
        session = await self.active(guild_id)
        self._require_host(session, actor_user_id)
        return await self._speaking.remove(
            session_id=session.id,
            user_id=user_id,
            actor_user_id=actor_user_id,
            source="SLASH_COMMAND",
        )

    async def controlled_speaking_extend(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        additional_seconds: int = 30,
    ) -> bool:
        async with self._consent_lock(guild_id):
            session, active_item = await self._controlled_agenda(guild_id, actor_user_id)
            update = await self._speaking_flow_required().extend_current(
                session_id=session.id,
                agenda_item_id=active_item.id,
                actor_user_id=actor_user_id,
                additional_seconds=additional_seconds,
                source="SLASH_COMMAND",
            )
            await self._dispatch_speaking_update_locked(session, update)
        return update.changed

    async def controlled_speaking_finish(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        skip: bool = False,
    ) -> bool:
        async with self._consent_lock(guild_id):
            session, active_item = await self._controlled_agenda(guild_id, actor_user_id)
            update = await self._speaking_flow_required().finish_current(
                session_id=session.id,
                agenda_item_id=active_item.id,
                actor_user_id=actor_user_id,
                reason="HOST_SKIPPED" if skip else "HOST_ENDED_EARLY",
                source="SLASH_COMMAND",
            )
            await self._dispatch_speaking_update_locked(session, update)
        return update.changed

    async def controlled_speaking_abandon(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
    ) -> bool:
        """Allow only the active ordered speaker to abandon their own turn."""

        async with self._consent_lock(guild_id):
            session = await self.active(guild_id)
            if session.state is not MeetingState.DISCUSSION_READY:
                raise ValueError("当前不在受控发言议程中。")
            active_item = next(
                (
                    item
                    for item in await self._agenda.list(session.id)
                    if item.status is AgendaStatus.ACTIVE
                ),
                None,
            )
            if active_item is None:
                raise ValueError("当前没有活动议程。")
            flow = await self._speaking_flow_required().state(session.id, active_item.id)
            if (
                flow is None
                or flow.mode is not AgendaSpeakingMode.ORDERED
                or flow.stage is not SpeakingFlowStage.ACTIVE
                or flow.current_user_ids != (actor_user_id,)
            ):
                raise MeetingPermissionError("only the active ordered speaker may abandon the turn")
            update = await self._speaking_flow_required().finish_current(
                session_id=session.id,
                agenda_item_id=active_item.id,
                actor_user_id=actor_user_id,
                reason="PARTICIPANT_ABANDONED",
                source="SLASH_COMMAND",
            )
            await self._dispatch_speaking_update_locked(session, update)
        return update.changed

    async def controlled_followup_finish(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
    ) -> bool:
        async with self._consent_lock(guild_id):
            session, active_item = await self._controlled_agenda(guild_id, actor_user_id)
            update = await self._speaking_flow_required().finish_followup(
                session_id=session.id,
                agenda_item_id=active_item.id,
                actor_user_id=actor_user_id,
                source="SLASH_COMMAND",
            )
            await self._dispatch_speaking_update_locked(session, update)
        return update.changed

    def _mute_enforcement_required(self) -> MuteEnforcementService:
        if self._mute_enforcement is None:
            raise RuntimeError("mute enforcement is unavailable")
        return self._mute_enforcement

    async def mute_enforcement_pause(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        source: str = "SLASH_COMMAND",
    ) -> bool:
        async with self._consent_lock(guild_id):
            session, item = await self._strict_agenda(guild_id, actor_user_id)
            flow = await self._speaking_flow_required().state(session.id, item.id)
            changed = await self._mute_enforcement_required().pause(
                session_id=session.id,
                agenda_item_id=item.id,
                reason="HOST_PAUSED",
                timer_type=flow.timer_type if flow is not None else None,
                actor_user_id=actor_user_id,
                source=source,
            )
            if changed and self._enforcement_handler is not None:
                await self._sync_enforcement(session, require_complete=True)
            return changed

    async def mute_enforcement_resume(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        source: str = "SLASH_COMMAND",
    ) -> bool:
        async with self._consent_lock(guild_id):
            session, item = await self._strict_agenda(guild_id, actor_user_id)
            changed = await self._mute_enforcement_required().resume(
                session_id=session.id,
                agenda_item_id=item.id,
                actor_user_id=actor_user_id,
                source=source,
                resume_timer=session.state is not MeetingState.PAUSED,
            )
            if changed and self._enforcement_handler is not None:
                await self._sync_enforcement(session, require_complete=True)
            return changed

    async def mute_enforcement_downgrade(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        emergency: bool = False,
        source: str = "SLASH_COMMAND",
    ) -> bool:
        async with self._consent_lock(guild_id):
            session, item = await self._strict_agenda(guild_id, actor_user_id)
            changed = await self._mute_enforcement_required().downgrade(
                session_id=session.id,
                agenda_item_id=item.id,
                actor_user_id=actor_user_id,
                source=source,
                emergency=emergency,
                resume_timer=session.state is not MeetingState.PAUSED,
            )
            if changed and self._enforcement_handler is not None:
                await self._sync_enforcement(session, require_complete=True)
            return changed

    async def mute_floor_exception(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        user_id: int,
        granted: bool,
        source: str = "SLASH_COMMAND",
    ) -> bool:
        async with self._consent_lock(guild_id):
            session, item = await self._strict_agenda(guild_id, actor_user_id)
            if user_id not in await self._participants.present_user_ids(session.id):
                raise ValueError("目标成员当前不在会议语音频道。")
            changed = await self._mute_enforcement_required().set_exception(
                session_id=session.id,
                agenda_item_id=item.id,
                user_id=user_id,
                granted=granted,
                actor_user_id=actor_user_id,
                source=source,
            )
            if changed and self._enforcement_handler is not None:
                await self._sync_enforcement(session, require_complete=True)
            return changed

    async def host_intervention(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        active: bool,
        source: str = "SLASH_COMMAND",
    ) -> bool:
        async with self._consent_lock(guild_id):
            session, item = await self._strict_agenda(guild_id, actor_user_id)
            enforcement = self._mute_enforcement_required()
            if active:
                if actor_user_id not in await self._participants.present_user_ids(session.id):
                    raise ValueError("主持人需先加入会议语音频道。")
                flow = await self._speaking_flow_required().state(session.id, item.id)
                changed = await enforcement.start_intervention(
                    session_id=session.id,
                    agenda_item_id=item.id,
                    timer_type=flow.timer_type if flow is not None else None,
                    actor_user_id=actor_user_id,
                    source=source,
                )
            else:
                changed = await enforcement.end_intervention(
                    session_id=session.id,
                    agenda_item_id=item.id,
                    actor_user_id=actor_user_id,
                    source=source,
                    resume_timer=session.state is not MeetingState.PAUSED,
                )
            if changed and self._enforcement_handler is not None:
                await self._sync_enforcement(session, require_complete=True)
            return changed

    def _speaking_flow_required(self) -> ControlledSpeakingFlowService:
        if self._speaking_flow is None:
            raise RuntimeError("controlled speaking flow is unavailable")
        return self._speaking_flow

    async def _controlled_agenda(
        self,
        guild_id: int,
        actor_user_id: int,
    ) -> tuple[MeetingSessionRecord, AgendaItemRecord]:
        session = await self.active(guild_id)
        self._require_host(session, actor_user_id)
        if session.state is not MeetingState.DISCUSSION_READY:
            raise ValueError("当前不在受控发言议程中。")
        active_item = next(
            (
                item
                for item in await self._agenda.list(session.id)
                if item.status is AgendaStatus.ACTIVE
            ),
            None,
        )
        if active_item is None or active_item.speaking_policy.mode is AgendaSpeakingMode.FREE:
            raise ValueError("当前议程不是按序发言或演讲。")
        return session, active_item

    async def _strict_agenda(
        self,
        guild_id: int,
        actor_user_id: int,
    ) -> tuple[MeetingSessionRecord, AgendaItemRecord]:
        session, item = await self._controlled_agenda(guild_id, actor_user_id)
        if not item.speaking_policy.mute_enforcement_requested:
            raise ValueError("当前议程未请求强制静音。")
        await self._mute_enforcement_required().ensure_started(
            session_id=session.id,
            item=item,
        )
        return session, item

    async def vote_open(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
        title: str,
        duration_seconds: int | None = None,
    ) -> VoteRecord:
        async with self._consent_lock(guild_id):
            session = await self.active(guild_id)
            self._require_host(session, actor_user_id)
            if session.state in {
                MeetingState.VOTE_OPEN,
                MeetingState.VOTE_RESULT,
                MeetingState.CLOSING,
                MeetingState.COMPLETED,
                MeetingState.POST_PROCESSING,
                MeetingState.PAUSED,
                MeetingState.INTERRUPTED,
            }:
                raise ValueError("当前会议状态不能另开公开投票。")
            return await self._voting.open(
                session_id=session.id,
                title=title,
                duration_seconds=duration_seconds or self._default_vote_seconds,
                actor_user_id=actor_user_id,
                source="SLASH_COMMAND",
            )

    async def vote_cast(
        self,
        *,
        guild_id: int,
        user_id: int,
        option_id: str,
        source: str = "SLASH_COMMAND",
    ) -> str:
        async with self._consent_lock(guild_id):
            session = await self.active(guild_id)
            vote = await self._voting.active_for_session(session.id)
            if vote is None:
                raise ValueError("there is no open vote")
            if session.state in {MeetingState.PAUSED, MeetingState.INTERRUPTED}:
                raise ValueError("ballots are not accepted while the Meeting is paused")
            if vote.agenda_item_id is not None:
                if session.state is not MeetingState.VOTE_OPEN:
                    raise ValueError("agenda ballots are accepted only while voting is open")
                epoch_vote = await self._current_epoch_agenda_vote(session)
                if epoch_vote is None or epoch_vote.id != vote.id:
                    raise ValueError("active agenda vote does not belong to the current vote epoch")
            selected = await self._voting.cast(
                vote_id=vote.id,
                session_id=session.id,
                user_id=user_id,
                option_id=option_id,
                source=source,
            )
        await self._finish_agenda_vote_if_complete(session=session, vote=vote)
        return selected

    async def vote_cast_private(
        self,
        *,
        message_id: int,
        user_id: int,
        option_id: str,
    ) -> str:
        vote = await self._voting.vote_for_dm_message(
            message_id=message_id,
            user_id=user_id,
        )
        session = await self._sessions.get(vote.session_id)
        async with self._consent_lock(session.guild_id):
            session = await self._sessions.get(vote.session_id)
            vote = await self._voting.vote_for_dm_message(
                message_id=message_id,
                user_id=user_id,
            )
            if session.state in {MeetingState.PAUSED, MeetingState.INTERRUPTED}:
                raise ValueError("ballots are not accepted while the Meeting is paused")
            if vote.agenda_item_id is not None:
                if session.state is not MeetingState.VOTE_OPEN:
                    raise ValueError("agenda ballots are accepted only while voting is open")
                epoch_vote = await self._current_epoch_agenda_vote(session)
                if epoch_vote is None or epoch_vote.id != vote.id:
                    raise ValueError("active agenda vote does not belong to the current vote epoch")
            selected = await self._voting.cast(
                vote_id=vote.id,
                session_id=session.id,
                user_id=user_id,
                option_id=option_id,
                source="DM_BUTTON",
            )
        await self._finish_agenda_vote_if_complete(session=session, vote=vote)
        return selected

    async def _finish_agenda_vote_if_complete(
        self,
        *,
        session: MeetingSessionRecord,
        vote: VoteRecord,
    ) -> RuntimeResult | None:
        if vote.agenda_item_id is None or not await self._voting.all_eligible_voted(vote.id):
            return None
        async with self._flow_lock(session.id):
            async with self._consent_lock(session.guild_id):
                current = await self._sessions.get(session.id)
                active = await self._voting.active_for_session(session.id)
                if (
                    current.state is not MeetingState.VOTE_OPEN
                    or active is None
                    or active.id != vote.id
                    or not await self._voting.all_eligible_voted(vote.id)
                ):
                    return None
                epoch_vote = await self._current_epoch_agenda_vote(current)
                if epoch_vote is None or epoch_vote.id != vote.id:
                    await self._voting.close(
                        vote.id,
                        actor_user_id=None,
                        source="STALE_EPOCH_ALL_BALLOTS",
                    )
                    await self._timers.cancel_type(
                        session_id=session.id,
                        timer_type=f"VOTE:{vote.id}",
                    )
                    return None
                outcome = await self._voting.close(
                    vote.id,
                    actor_user_id=None,
                    source="ALL_BALLOTS_CAST",
                )
                await self._timers.cancel_type(
                    session_id=session.id,
                    timer_type=f"VOTE:{vote.id}",
                )
                return await self._execute_actions_locked(
                    await self._runtime.advance(session.id),
                    vote_outcome=outcome,
                )

    async def vote_recipients(self, vote_id: str) -> tuple[VoteRecipientRecord, ...]:
        return await self._voting.recipients(vote_id)

    async def mark_vote_delivery(
        self,
        *,
        vote_id: str,
        user_id: int,
        delivery_status: str,
        dm_message_id: int | None = None,
    ) -> VoteRecipientRecord:
        return await self._voting.mark_delivery(
            vote_id=vote_id,
            user_id=user_id,
            delivery_status=delivery_status,
            dm_message_id=dm_message_id,
        )

    async def vote_close(self, *, guild_id: int, actor_user_id: int) -> VoteOutcome:
        session = await self.active(guild_id)
        async with self._flow_lock(session.id):
            async with self._consent_lock(guild_id):
                session = await self.active(guild_id)
                self._require_host(session, actor_user_id)
                vote = await self._voting.active_for_session(session.id)
                if vote is None:
                    raise ValueError("there is no open vote")
                outcome = await self._voting.close(
                    vote.id,
                    actor_user_id=actor_user_id,
                    source="SLASH_COMMAND",
                )
                await self._timers.cancel_type(
                    session_id=session.id,
                    timer_type=f"VOTE:{vote.id}",
                )
                if vote.agenda_item_id is not None:
                    current = await self._sessions.get(session.id)
                    if current.state is MeetingState.VOTE_OPEN:
                        epoch_vote = await self._current_epoch_agenda_vote(current)
                        if epoch_vote is None or epoch_vote.id != vote.id:
                            return outcome
                        await self._execute_actions_locked(
                            await self._runtime.advance(session.id),
                            vote_outcome=outcome,
                        )
                return outcome

    async def participant_join(
        self,
        *,
        guild_id: int,
        user_id: int,
        display_name: str | None = None,
    ) -> None:
        session = await self._sessions.active_for_guild(guild_id)
        if session is not None:
            grants = await self._guild_consents.grants_for(
                guild_id=guild_id,
                user_id=user_id,
            )
            await self._participants.join(
                session_id=session.id,
                user_id=user_id,
                inherited_consents=frozenset(),
                display_name=display_name,
                is_host=user_id == session.creator_user_id,
            )
            if grants == frozenset(ConsentKind):
                await self._grant_all(
                    session_id=session.id,
                    guild_id=guild_id,
                    user_id=user_id,
                    effective_at_ms=0,
                    trigger="inherited_guild_grant",
                )
            elif grants:
                await self._participants.set_consents(
                    session_id=session.id,
                    user_id=user_id,
                    grants=grants,
                )
            if self._consent_runtime is not None:
                await self._consent_runtime.sync_user(
                    guild_id=guild_id,
                    user_id=user_id,
                )
            await self._advance_if_consented(guild_id)

    async def participant_leave(self, *, guild_id: int, user_id: int) -> None:
        session = await self._sessions.active_for_guild(guild_id)
        if session is not None:
            await self._participants.leave(session_id=session.id, user_id=user_id)
            if self._consent_runtime is not None:
                await self._consent_runtime.suspend_user(
                    guild_id=guild_id,
                    user_id=user_id,
                )
            await self._speaking.remove_offline(session.id)
            await self._advance_if_consented(guild_id)

    async def _grant_all(
        self,
        *,
        session_id: str,
        guild_id: int,
        user_id: int,
        effective_at_ms: int,
        trigger: str,
    ) -> MeetingEventRecord:
        await self._participants.require_present(session_id=session_id, user_id=user_id)
        return await self._consent_changes.grant_all(
            session_id=session_id,
            guild_id=guild_id,
            user_id=user_id,
            effective_at_ms=effective_at_ms,
            trigger=trigger,
        )

    async def _withdraw_all(
        self,
        *,
        session_id: str,
        guild_id: int,
        user_id: int,
        effective_at_ms: int,
        trigger: str,
    ) -> MeetingEventRecord | None:
        return await self._consent_changes.withdraw_all(
            session_id=session_id,
            guild_id=guild_id,
            user_id=user_id,
            effective_at_ms=effective_at_ms,
            trigger=trigger,
        )

    async def consent_all(self, *, guild_id: int, user_id: int) -> MeetingState:
        async with self._consent_lock(guild_id):
            session = await self.active(guild_id)
            effective_at_ms = (
                await self._consent_runtime.timeline_now_ms(
                    guild_id=guild_id,
                    user_id=user_id,
                )
                if self._consent_runtime is not None
                else 0
            )
            await self._grant_all(
                session_id=session.id,
                guild_id=guild_id,
                user_id=user_id,
                effective_at_ms=effective_at_ms,
                trigger="slash_command",
            )
            if self._consent_runtime is not None:
                await self._consent_runtime.sync_user(
                    guild_id=guild_id,
                    user_id=user_id,
                )
        advanced = await self._advance_if_consented(guild_id)
        if advanced is not None:
            return advanced.session.state
        return (await self._sessions.get(session.id)).state

    async def withdraw_all(self, *, guild_id: int, user_id: int) -> None:
        async with self._consent_lock(guild_id):
            session = await self._sessions.active_for_guild(guild_id)
            if session is None:
                await self._guild_consents.withdraw_all(guild_id=guild_id, user_id=user_id)
                if self._consent_runtime is not None:
                    await self._consent_runtime.suspend_user(
                        guild_id=guild_id,
                        user_id=user_id,
                    )
            else:
                effective_at_ms = (
                    await self._consent_runtime.timeline_now_ms(
                        guild_id=guild_id,
                        user_id=user_id,
                    )
                    if self._consent_runtime is not None
                    else 0
                )
                withdrawal = await self._withdraw_all(
                    session_id=session.id,
                    guild_id=guild_id,
                    user_id=user_id,
                    effective_at_ms=effective_at_ms,
                    trigger="slash_command",
                )
                if self._consent_runtime is not None:
                    await self._consent_runtime.apply_withdrawal(
                        guild_id=guild_id,
                        user_id=user_id,
                        effective_at_ms=(
                            int(withdrawal.payload["effective_at_ms"])
                            if withdrawal is not None
                            else effective_at_ms
                        ),
                    )

    async def retry(self, *, guild_id: int, actor_user_id: int) -> bool:
        session = await self.retry_session(
            guild_id=guild_id,
            actor_user_id=actor_user_id,
        )
        if self._jobs is None:
            return False
        return (
            await self._jobs.retry_failed(
                session_id=session.id,
                job_type="POST_MEETING",
                authorize_content_generation_retry=True,
            )
            is not None
        )

    async def retry_session(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
    ) -> MeetingSessionRecord:
        """Validate retry ownership without changing the Worker queue."""

        session = await self._sessions.latest_for_guild(guild_id)
        if session is None:
            raise SessionNotFoundError("this guild has no Meeting")
        self._require_host(session, actor_user_id)
        return session

    async def delete(
        self,
        *,
        guild_id: int,
        actor_user_id: int,
    ) -> MeetingSessionRecord:
        async with self._consent_lock(guild_id):
            session = await self._sessions.latest_for_guild(guild_id)
            if session is None:
                raise SessionNotFoundError("this guild has no Meeting")
            self._require_host(session, actor_user_id)
            if self._runtime_action_handler is not None:
                await self._runtime_action_handler(session, "finalise_capture")
            await self._session_service.delete(
                session_id=session.id,
                actor_user_id=actor_user_id,
            )
            return session
