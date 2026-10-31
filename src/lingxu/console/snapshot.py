"""Authorization-scoped, privacy-minimal Host console projections."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select

from lingxu.meeting.controller import MeetingController
from lingxu.scenario.repository import ScenarioRepository, ScenarioVersionRecord
from lingxu.storage.database import Database
from lingxu.storage.repositories import MeetingSessionRepository
from lingxu.storage.tables import (
    MeetingEventRow,
    MeetingParticipantRow,
    MeetingRealtimeUtteranceRow,
    MeetingTimerRow,
    MeetingVoteEntryRow,
    ScenarioDefinitionRow,
    ScenarioVersionRow,
)

_ACTIVE_TIMER_STATUSES = ("RUNNING", "PAUSED")
_EVENT_LIMIT = 24
_UTTERANCE_LIMIT = 8


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _iso(value: datetime | None) -> str | None:
    return _aware(value).isoformat() if value is not None else None


class ConsoleSnapshotStore:
    """Build one Guild-bound snapshot without exposing vote-entry identities."""

    def __init__(
        self,
        *,
        database: Database,
        controller: MeetingController,
        sessions: MeetingSessionRepository,
        guild_id: int,
    ) -> None:
        self._database = database
        self._controller = controller
        self._sessions = sessions
        self.guild_id = guild_id
        self._scenarios = ScenarioRepository(database.sessions)

    async def active_scenario(self, key: str = "fixed-meeting") -> ScenarioVersionRecord | None:
        return await self._scenarios.active(key)

    async def _scenario_summaries(self) -> list[dict[str, Any]]:
        async with self._database.sessions() as db:
            rows = list(
                await db.execute(
                    select(ScenarioDefinitionRow, ScenarioVersionRow)
                    .join(
                        ScenarioVersionRow,
                        ScenarioVersionRow.id == ScenarioDefinitionRow.active_version_id,
                    )
                    .order_by(ScenarioDefinitionRow.name)
                )
            )
        return [
            {
                "key": version.schema_json.get("key", definition.name),
                "name": version.schema_json.get("name", definition.name),
                "version": version.version,
                "status": version.status,
                "state_count": len(version.schema_json.get("states", [])),
                "source_sha256": version.source_sha256,
            }
            for definition, version in rows
        ]

    async def snapshot(self) -> dict[str, Any]:
        scenarios = await self._scenario_summaries()
        session = await self._sessions.active_for_guild(self.guild_id)
        generated_at = datetime.now(UTC)
        if session is None:
            return {
                "active": False,
                "guild_id": str(self.guild_id),
                "generated_at": generated_at.isoformat(),
                "state": "IDLE",
                "scenario": scenarios,
                "participants": [],
                "agenda": [],
                "speaker": {"current_user_id": None, "queued_user_ids": []},
                "timers": [],
                "vote": None,
                "final_utterances": [],
                "recent_events": [],
            }

        status = await self._controller.status(self.guild_id)
        async with self._database.sessions() as db:
            participant_rows = list(
                await db.scalars(
                    select(MeetingParticipantRow)
                    .where(MeetingParticipantRow.session_id == session.id)
                    .order_by(MeetingParticipantRow.joined_at, MeetingParticipantRow.user_id)
                )
            )
            timer_rows = list(
                await db.scalars(
                    select(MeetingTimerRow)
                    .where(
                        MeetingTimerRow.session_id == session.id,
                        MeetingTimerRow.status.in_(_ACTIVE_TIMER_STATUSES),
                    )
                    .order_by(MeetingTimerRow.deadline_at, MeetingTimerRow.id)
                )
            )
            event_rows = list(
                await db.scalars(
                    select(MeetingEventRow)
                    .where(MeetingEventRow.session_id == session.id)
                    .order_by(MeetingEventRow.sequence_number.desc())
                    .limit(_EVENT_LIMIT)
                )
            )
            utterance_rows = list(
                await db.scalars(
                    select(MeetingRealtimeUtteranceRow)
                    .join(
                        MeetingParticipantRow,
                        (MeetingParticipantRow.session_id == MeetingRealtimeUtteranceRow.session_id)
                        & (MeetingParticipantRow.user_id == MeetingRealtimeUtteranceRow.user_id),
                    )
                    .where(
                        MeetingRealtimeUtteranceRow.session_id == session.id,
                        MeetingParticipantRow.consent_realtime_asr.is_(True),
                    )
                    .order_by(
                        MeetingRealtimeUtteranceRow.created_at.desc(),
                        MeetingRealtimeUtteranceRow.id.desc(),
                    )
                    .limit(_UTTERANCE_LIMIT)
                )
            )
            vote_counts: dict[str, int] = {}
            if status.vote is not None:
                counts = await db.execute(
                    select(
                        MeetingVoteEntryRow.option_id,
                        func.count(MeetingVoteEntryRow.id),
                    )
                    .where(MeetingVoteEntryRow.vote_id == status.vote.id)
                    .group_by(MeetingVoteEntryRow.option_id)
                )
                vote_counts = {str(option_id): int(count) for option_id, count in counts}

        timers: list[dict[str, Any]] = []
        for row in timer_rows:
            if row.status == "PAUSED":
                remaining_ms = max(0, int(row.remaining_ms or 0))
            else:
                remaining_ms = max(
                    0,
                    int((_aware(row.deadline_at) - generated_at).total_seconds() * 1_000),
                )
            timers.append(
                {
                    "id": row.id,
                    "type": row.timer_type,
                    "status": row.status,
                    "deadline_at": _iso(row.deadline_at),
                    "remaining_ms": remaining_ms,
                }
            )

        vote: dict[str, Any] | None = None
        if status.vote is not None:
            vote = {
                "id": status.vote.id,
                "title": status.vote.title,
                "status": status.vote.status.value,
                "anonymous": status.vote.anonymous,
                "deadline_at": _iso(status.vote.deadline_at),
                "options": [
                    {
                        "id": str(option.get("id", "")),
                        "label": str(option.get("label", option.get("id", ""))),
                        "count": vote_counts.get(str(option.get("id", "")), 0),
                    }
                    for option in status.vote.options
                ],
            }

        active_agenda = next(
            (item for item in status.agenda if item.status.value == "ACTIVE"),
            None,
        )
        return {
            "active": True,
            "guild_id": str(self.guild_id),
            "generated_at": generated_at.isoformat(),
            "session_id": session.id,
            "host_user_id": str(session.creator_user_id),
            "state": session.state.value,
            "sequence": session.next_sequence,
            "current_agenda": (
                {
                    "position": active_agenda.position,
                    "title": active_agenda.title,
                }
                if active_agenda is not None
                else None
            ),
            "agenda": [
                {
                    "id": item.id,
                    "position": item.position,
                    "title": item.title,
                    "status": item.status.value,
                    "duration_minutes": item.duration_minutes,
                    "requires_vote": item.requires_vote,
                    "vote_content": item.vote_content,
                    "vote_duration_seconds": item.vote_duration_seconds,
                }
                for item in status.agenda
            ],
            "participants": [
                {
                    "user_id": str(row.user_id),
                    "present": row.left_at is None,
                    "consent": {
                        "realtime_asr": row.consent_realtime_asr,
                        "recording": row.consent_recording,
                        "minutes": row.consent_minutes,
                    },
                }
                for row in participant_rows
            ],
            "speaker": {
                "current_user_id": (
                    str(status.speaking.current_user_id)
                    if status.speaking.current_user_id is not None
                    else None
                ),
                "queued_user_ids": [str(value) for value in status.speaking.queued_user_ids],
            },
            "timers": timers,
            "vote": vote,
            "final_utterances": [
                {
                    "id": row.id,
                    "user_id": str(row.user_id),
                    "text": row.normalised_text or row.raw_text,
                    "started_ms": row.started_ms,
                    "ended_ms": row.ended_ms,
                    "created_at": _iso(row.created_at),
                }
                for row in utterance_rows
            ],
            "recent_events": [
                {
                    "sequence": row.sequence_number,
                    "type": row.event_type,
                    "source": row.source,
                    "state_before": row.state_before,
                    "state_after": row.state_after,
                    "occurred_at": _iso(row.occurred_at),
                }
                for row in reversed(event_rows)
            ],
            "scenario": scenarios,
        }
