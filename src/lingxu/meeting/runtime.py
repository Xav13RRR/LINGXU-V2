"""Deterministic, event-sourced fixed Meeting Runtime."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from lingxu.domain.enums import MeetingEventType, MeetingState
from lingxu.domain.errors import (
    InvalidStateTransitionError,
    MeetingPermissionError,
    MeetingRecoveryError,
)
from lingxu.domain.meeting_state import back_state, forward_state
from lingxu.domain.models import MeetingEventRecord, MeetingSessionRecord
from lingxu.storage.repositories import (
    MeetingEventStore,
    MeetingSessionRepository,
    MeetingSnapshotStore,
)

_CONTROL_SOURCES = frozenset({"SLASH_COMMAND", "BUTTON", "VOICE_RULE"})

_STATE_ACTIONS: dict[MeetingState, tuple[str, ...]] = {
    MeetingState.AWAITING_PARTICIPANTS: ("publish_host_panel", "request_consent"),
    MeetingState.CONSENT_CHECK: ("validate_consents",),
    MeetingState.OPENING: ("speak_welcome",),
    MeetingState.AGENDA_OVERVIEW: ("speak_agenda",),
    MeetingState.AGENDA_ITEM_INTRO: ("speak_current_agenda_item",),
    MeetingState.DISCUSSION_READY: ("select_next_speaker",),
    MeetingState.SPEAKER_ACTIVE: ("start_speaker_timer",),
    MeetingState.DISCUSSION_COMPLETE: ("cancel_speaker_timer",),
    MeetingState.VOTE_OPEN: ("open_vote", "start_vote_timer"),
    MeetingState.VOTE_RESULT: ("close_vote", "speak_vote_result"),
    MeetingState.NEXT_AGENDA_ITEM: ("advance_agenda",),
    MeetingState.CLOSING: ("speak_closing", "finalise_capture"),
    MeetingState.COMPLETED: ("trigger_post_meeting_job",),
    MeetingState.POST_PROCESSING: ("monitor_post_meeting_job",),
    MeetingState.PUBLISHED: ("publish_completion",),
    MeetingState.PAUSED: ("freeze_timers",),
    MeetingState.STOPPED: (
        "cancel_timers",
        "speak_closing",
        "finalise_capture",
        "trigger_post_meeting_job",
    ),
}


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    """Persisted state and deterministic side-effect intents for a controller."""

    session: MeetingSessionRecord
    event: MeetingEventRecord | None
    actions: tuple[str, ...]


class MeetingRuntime:
    """The sole authority allowed to select Meeting state transitions."""

    def __init__(
        self,
        *,
        sessions: MeetingSessionRepository,
        events: MeetingEventStore,
        snapshots: MeetingSnapshotStore,
        snapshot_interval: int = 50,
    ) -> None:
        if snapshot_interval < 1:
            raise ValueError("snapshot_interval must be positive")
        self._sessions = sessions
        self._events = events
        self._snapshots = snapshots
        self._snapshot_interval = snapshot_interval
        self._log = structlog.get_logger(__name__)

    @staticmethod
    def _guard_host(
        session: MeetingSessionRecord,
        *,
        actor_user_id: int | None,
        source: str,
    ) -> None:
        if source in _CONTROL_SOURCES and actor_user_id != session.creator_user_id:
            raise MeetingPermissionError("only the Meeting host may control the runtime")

    async def _result_without_event(self, session: MeetingSessionRecord) -> RuntimeResult:
        return RuntimeResult(session=session, event=None, actions=())

    async def _apply(
        self,
        *,
        session: MeetingSessionRecord,
        target: MeetingState,
        event_type: MeetingEventType,
        actor_user_id: int | None,
        source: str,
        allow_back: bool = False,
        allow_auto: bool = False,
        allow_post_processing_recovery: bool = False,
        actions: tuple[str, ...] | None = None,
    ) -> RuntimeResult:
        selected_actions = _STATE_ACTIONS.get(target, ()) if actions is None else actions
        transition = await self._sessions.transition_with_event(
            session_id=session.id,
            expected_state=session.state,
            expected_resume_state=session.resume_state,
            expected_sequence=session.next_sequence,
            target=target,
            actor_user_id=actor_user_id,
            source=source,
            event_type=event_type.value,
            payload={
                "runtime_version": 2,
                "runtime_actions": list(selected_actions),
            },
            allow_back=allow_back,
            allow_auto=allow_auto,
            allow_post_processing_recovery=allow_post_processing_recovery,
        )
        record = transition.session
        if not transition.changed:
            return await self._result_without_event(record)
        event = transition.event
        if event is None:
            raise RuntimeError("changed Meeting transition did not return its event")
        if record.next_sequence % self._snapshot_interval == 0:
            try:
                await self._snapshots.save(
                    session_id=record.id,
                    state=record.state,
                    sequence_number=record.next_sequence,
                    payload={
                        "runtime_version": 2,
                        "resume_state": record.resume_state.value if record.resume_state else None,
                    },
                )
            except Exception as exc:
                self._log.error(
                    "meeting_snapshot_save_failed",
                    session_id=record.id,
                    sequence_number=record.next_sequence,
                    error_type=type(exc).__name__,
                )
        return RuntimeResult(session=record, event=event, actions=selected_actions)

    async def initialise(
        self,
        session_id: str,
        *,
        actor_user_id: int | None,
        source: str = "SYSTEM",
    ) -> RuntimeResult:
        session = await self._sessions.get(session_id)
        self._guard_host(session, actor_user_id=actor_user_id, source=source)
        if session.state is not MeetingState.CREATED:
            return await self._result_without_event(session)
        return await self._apply(
            session=session,
            target=MeetingState.AWAITING_PARTICIPANTS,
            event_type=MeetingEventType.MEETING_INITIALISED,
            actor_user_id=actor_user_id,
            source=source,
        )

    async def start(
        self,
        session_id: str,
        *,
        actor_user_id: int | None,
        source: str,
    ) -> RuntimeResult:
        """Idempotently start once; internal progression uses `advance`."""

        session = await self._sessions.get(session_id)
        self._guard_host(session, actor_user_id=actor_user_id, source=source)
        if session.state is not MeetingState.AWAITING_PARTICIPANTS:
            return await self._result_without_event(session)
        return await self._apply(
            session=session,
            target=MeetingState.CONSENT_CHECK,
            event_type=MeetingEventType.MEETING_STARTED,
            actor_user_id=actor_user_id,
            source=source,
        )

    async def advance(
        self,
        session_id: str,
        *,
        actor_user_id: int | None = None,
        source: str = "SYSTEM",
    ) -> RuntimeResult:
        session = await self._sessions.get(session_id)
        self._guard_host(session, actor_user_id=actor_user_id, source=source)
        target = forward_state(session.state)
        if target is None:
            raise InvalidStateTransitionError(f"no fixed next state from {session.state}")
        return await self._apply(
            session=session,
            target=target,
            event_type=MeetingEventType.STATE_ADVANCED,
            actor_user_id=actor_user_id,
            source=source,
        )

    async def consents_satisfied(self, session_id: str) -> RuntimeResult:
        """Idempotently leave CONSENT_CHECK after its external guard succeeds."""

        session = await self._sessions.get(session_id)
        if session.state is not MeetingState.CONSENT_CHECK:
            return await self._result_without_event(session)
        return await self._apply(
            session=session,
            target=MeetingState.OPENING,
            event_type=MeetingEventType.STATE_ADVANCED,
            actor_user_id=None,
            source="SYSTEM",
        )

    async def auto_transition(
        self,
        session_id: str,
        target: MeetingState,
    ) -> RuntimeResult:
        """Apply one explicit system-only automatic-flow edge."""

        session = await self._sessions.get(session_id)
        return await self._apply(
            session=session,
            target=target,
            event_type=MeetingEventType.STATE_ADVANCED,
            actor_user_id=None,
            source="SYSTEM",
            allow_auto=True,
        )

    async def next(
        self,
        session_id: str,
        *,
        actor_user_id: int,
        source: str,
    ) -> RuntimeResult:
        return await self.advance(
            session_id,
            actor_user_id=actor_user_id,
            source=source,
        )

    async def skip(
        self,
        session_id: str,
        *,
        actor_user_id: int,
        source: str,
    ) -> RuntimeResult:
        session = await self._sessions.get(session_id)
        self._guard_host(session, actor_user_id=actor_user_id, source=source)
        target = forward_state(session.state)
        if target is None:
            raise InvalidStateTransitionError(f"no fixed skip state from {session.state}")
        return await self._apply(
            session=session,
            target=target,
            event_type=MeetingEventType.STATE_SKIPPED,
            actor_user_id=actor_user_id,
            source=source,
        )

    async def back(
        self,
        session_id: str,
        *,
        actor_user_id: int,
        source: str,
    ) -> RuntimeResult:
        session = await self._sessions.get(session_id)
        self._guard_host(session, actor_user_id=actor_user_id, source=source)
        target = back_state(session.state)
        if target is None:
            raise InvalidStateTransitionError(f"no fixed back state from {session.state}")
        return await self._apply(
            session=session,
            target=target,
            event_type=MeetingEventType.MEETING_BACK,
            actor_user_id=actor_user_id,
            source=source,
            allow_back=True,
        )

    async def pause(
        self,
        session_id: str,
        *,
        actor_user_id: int,
        source: str,
    ) -> RuntimeResult:
        session = await self._sessions.get(session_id)
        self._guard_host(session, actor_user_id=actor_user_id, source=source)
        if session.state is MeetingState.PAUSED:
            return await self._result_without_event(session)
        return await self._apply(
            session=session,
            target=MeetingState.PAUSED,
            event_type=MeetingEventType.MEETING_PAUSED,
            actor_user_id=actor_user_id,
            source=source,
        )

    async def resume(
        self,
        session_id: str,
        *,
        actor_user_id: int,
        source: str,
    ) -> RuntimeResult:
        session = await self._sessions.get(session_id)
        self._guard_host(session, actor_user_id=actor_user_id, source=source)
        if session.state not in {MeetingState.PAUSED, MeetingState.INTERRUPTED}:
            raise InvalidStateTransitionError("only a paused/interrupted Meeting may resume")
        if session.resume_state is None:
            raise MeetingRecoveryError("paused Meeting has no persisted resume state")
        return await self._apply(
            session=session,
            target=session.resume_state,
            event_type=MeetingEventType.MEETING_RESUMED,
            actor_user_id=actor_user_id,
            source=source,
            actions=("resume_timers",),
        )

    async def stop(
        self,
        session_id: str,
        *,
        actor_user_id: int,
        source: str,
    ) -> RuntimeResult:
        session = await self._sessions.get(session_id)
        self._guard_host(session, actor_user_id=actor_user_id, source=source)
        if session.state is MeetingState.STOPPED:
            return await self._result_without_event(session)
        return await self._apply(
            session=session,
            target=MeetingState.STOPPED,
            event_type=MeetingEventType.MEETING_STOPPED,
            actor_user_id=actor_user_id,
            source=source,
        )

    async def post_processing_completed(self, session_id: str) -> RuntimeResult:
        """Publish a completed Worker result through Runtime's state authority."""

        session = await self._sessions.get(session_id)
        if session.state is not MeetingState.POST_PROCESSING:
            return await self._result_without_event(session)
        return await self._apply(
            session=session,
            target=MeetingState.PUBLISHED,
            event_type=MeetingEventType.STATE_ADVANCED,
            actor_user_id=None,
            source="WORKER_MONITOR",
        )

    async def post_processing_failed(self, session_id: str) -> RuntimeResult:
        """Persist a terminal post-processing failure selected by the Bot monitor."""

        session = await self._sessions.get(session_id)
        if session.state is not MeetingState.POST_PROCESSING:
            return await self._result_without_event(session)
        return await self._apply(
            session=session,
            target=MeetingState.FAILED_POST_PROCESSING,
            event_type=MeetingEventType.STATE_ADVANCED,
            actor_user_id=None,
            source="WORKER_MONITOR",
        )

    async def post_processing_retry_completed(self, session_id: str) -> RuntimeResult:
        """Converge a host-authorized retry that completed after the prior terminal failure."""

        session = await self._sessions.get(session_id)
        if session.state is not MeetingState.FAILED_POST_PROCESSING:
            return await self._result_without_event(session)
        return await self._apply(
            session=session,
            target=MeetingState.PUBLISHED,
            event_type=MeetingEventType.STATE_ADVANCED,
            actor_user_id=None,
            source="WORKER_RETRY_MONITOR",
            allow_post_processing_recovery=True,
        )

    async def recover(self, session_id: str) -> RuntimeResult:
        """Cross-check append-only events, latest snapshot, and materialised row."""

        recovered, _ = await self._recover_with_events(session_id)
        return recovered

    async def _recover_with_events(
        self,
        session_id: str,
    ) -> tuple[RuntimeResult, tuple[MeetingEventRecord, ...]]:
        """Validate once and retain the exact stream used by action recovery."""

        session = await self._sessions.get(session_id)
        events = await self._events.list(session_id)
        latest_snapshot = await self._snapshots.latest(session_id)
        current = MeetingState.CREATED
        snapshot_state: MeetingState | None = None
        for event in events:
            if event.state_before is not current:
                raise MeetingRecoveryError(
                    f"event {event.sequence_number} does not continue the state chain"
                )
            current = event.state_after
            if latest_snapshot and event.sequence_number == latest_snapshot.sequence_number:
                snapshot_state = current
        if events and events[-1].sequence_number != session.next_sequence:
            raise MeetingRecoveryError("materialised sequence differs from event stream")
        if not events and session.next_sequence != 0:
            raise MeetingRecoveryError("materialised sequence exists without events")
        if latest_snapshot and snapshot_state is not latest_snapshot.state:
            raise MeetingRecoveryError("snapshot state differs from event stream")
        if current is not session.state:
            raise MeetingRecoveryError("materialised state differs from event stream")
        return (
            RuntimeResult(session=session, event=events[-1] if events else None, actions=()),
            events,
        )

    async def recover_actions(self, session_id: str) -> RuntimeResult:
        """Recover the latest action batch in the current workflow-state epoch."""

        batches = await self.recover_action_batches(session_id)
        if batches:
            return batches[-1]
        session = await self._sessions.get(session_id)
        return RuntimeResult(session=session, event=None, actions=())

    @staticmethod
    def _actions_for_recovery_event(event: MeetingEventRecord) -> tuple[str, ...]:
        persisted_actions = event.payload.get("runtime_actions")
        if event.payload.get("runtime_version") == 2:
            if isinstance(persisted_actions, list) and all(
                isinstance(action, str) for action in persisted_actions
            ):
                selected_actions = tuple(persisted_actions)
                if event.state_after in {MeetingState.STOPPED, MeetingState.PUBLISHED}:
                    expected_actions = _STATE_ACTIONS[event.state_after]
                    if selected_actions != expected_actions:
                        return ()
                return selected_actions
            # Version 2 actions are an explicit durable batch. A malformed or
            # incomplete batch must fail closed instead of deriving side effects.
            return ()
        if event.state_after in {MeetingState.STOPPED, MeetingState.PUBLISHED}:
            # Historical terminal transitions predate durable action batches and
            # acknowledgements. Their side effects may already have happened, so
            # replaying inferred actions would duplicate prompts/jobs on upgrade.
            return ()
        if event.event_type == MeetingEventType.MEETING_RESUMED.value:
            # Historical runtime_version=1 recovery: resume is transition-specific and
            # must never fall back to the destination state's normal entry actions.
            return ("resume_timers",)
        return _STATE_ACTIONS.get(event.state_after, ())

    async def recover_action_batches(self, session_id: str) -> tuple[RuntimeResult, ...]:
        """Recover entry and resume intents for the current workflow-state epoch."""

        recovered, events = await self._recover_with_events(session_id)
        state = recovered.session.state
        if state in {MeetingState.PAUSED, MeetingState.INTERRUPTED}:
            entry = next(
                (
                    event
                    for event in reversed(events)
                    if event.state_before is not event.state_after and event.state_after is state
                ),
                None,
            )
        else:
            entry = next(
                (
                    event
                    for event in reversed(events)
                    if event.state_before is not event.state_after
                    and event.state_after is state
                    and event.event_type != MeetingEventType.MEETING_RESUMED.value
                ),
                None,
            )
        if entry is None:
            return ()
        selected = tuple(
            event
            for event in events
            if event.id == entry.id
            or (
                event.sequence_number > entry.sequence_number
                and event.event_type == MeetingEventType.MEETING_RESUMED.value
                and event.state_after is state
            )
        )
        return tuple(
            RuntimeResult(
                session=recovered.session,
                event=event,
                actions=self._actions_for_recovery_event(event),
            )
            for event in selected
        )
