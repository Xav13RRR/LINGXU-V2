"""Discord server-mute adapter backed by persistent ownership leases."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import discord
import structlog

from lingxu.domain.enums import (
    AgendaStatus,
    FloorPhase,
    MeetingEventType,
    MeetingState,
    MuteLeaseOwnership,
    SpeakingFlowStage,
)
from lingxu.domain.models import AgendaItemRecord, MeetingMuteLeaseRecord, MeetingSessionRecord
from lingxu.domain.speaking import FloorContext, floor_user_ids
from lingxu.meeting.controller import MeetingStatus
from lingxu.meeting.factory import MeetingApplication


@dataclass(frozen=True, slots=True)
class MuteSyncResult:
    muted: int = 0
    unmuted: int = 0
    reapplied: int = 0
    released: int = 0
    paused: bool = False


class MeetingMuteCoordinator:
    """Apply a computed floor without ever unmuting an unowned moderator mute."""

    def __init__(self, *, bot: discord.Client, application: MeetingApplication) -> None:
        self._bot = bot
        self._application = application
        self._locks: dict[int, asyncio.Lock] = {}
        self._log = structlog.get_logger(__name__)

    def _lock(self, guild_id: int) -> asyncio.Lock:
        return self._locks.setdefault(guild_id, asyncio.Lock())

    def _voice_channel(self, session: MeetingSessionRecord) -> discord.VoiceChannel | None:
        if session.publication_channel_id is None:
            return None
        channel = self._bot.get_channel(session.publication_channel_id)
        return channel if isinstance(channel, discord.VoiceChannel) else None

    @staticmethod
    def _active_item(status: MeetingStatus) -> AgendaItemRecord | None:
        return next(
            (item for item in status.agenda if item.status is AgendaStatus.ACTIVE),
            None,
        )

    @staticmethod
    def _floor(status: MeetingStatus, present_user_ids: tuple[int, ...]) -> tuple[int, ...]:
        item = MeetingMuteCoordinator._active_item(status)
        state = status.enforcement
        flow = status.floor
        if item is None or state is None or flow is None:
            return ()
        if state.host_intervention_active:
            return floor_user_ids(
                FloorContext(
                    policy=item.speaking_policy,
                    present_user_ids=present_user_ids,
                    host_user_id=status.session.creator_user_id,
                    exception_user_ids=state.exception_user_ids,
                    host_intervention_active=True,
                )
            )
        if flow.stage in {
            SpeakingFlowStage.STARTING,
            SpeakingFlowStage.PREPARING,
            SpeakingFlowStage.COMPLETED,
        }:
            return tuple(
                user_id for user_id in state.exception_user_ids if user_id in present_user_ids
            )
        return floor_user_ids(
            FloorContext(
                policy=item.speaking_policy,
                present_user_ids=present_user_ids,
                host_user_id=status.session.creator_user_id,
                current_speaker_user_id=(
                    status.speaking.current_user_id
                    if flow.stage is SpeakingFlowStage.FOLLOWUP
                    else flow.current_user_ids[0]
                    if len(flow.current_user_ids) == 1
                    else None
                ),
                current_questioner_user_id=status.speaking.current_user_id,
                exception_user_ids=state.exception_user_ids,
                phase=(
                    FloorPhase.FOLLOWUP
                    if flow.stage is SpeakingFlowStage.FOLLOWUP
                    else FloorPhase.MAIN
                ),
            )
        )

    async def sync_session(self, session: MeetingSessionRecord) -> MuteSyncResult:
        """Reconcile a live or terminal Session against Discord and durable leases."""

        async with self._lock(session.guild_id):
            status = await self._application.controller.status_for_session(session.id)
            session = status.session
            item = self._active_item(status)
            channel = self._voice_channel(session)
            if (
                session.state not in {MeetingState.DISCUSSION_READY, MeetingState.PAUSED}
                or item is None
                or not item.speaking_policy.mute_enforcement_requested
            ):
                if item is not None:
                    await self._application.mute_enforcement.end(
                        session_id=session.id,
                        agenda_item_id=item.id,
                    )
                return await self._release_all(session, channel, reason="NO_STRICT_AGENDA")

            state = await self._application.mute_enforcement.ensure_started(
                session_id=session.id,
                item=item,
            )
            status = await self._application.controller.status_for_session(session.id)
            state = status.enforcement or state
            if state is None:
                return await self._release_all(session, channel, reason="NOT_REQUESTED")
            if channel is None:
                await self._pause_for_uncertainty(status, "VOICE_CHANNEL_UNAVAILABLE")
                return MuteSyncResult(paused=True)
            if not channel.permissions_for(channel.guild.me).mute_members:
                await self._pause_for_permission(status, "CHECK_PERMISSION")
                return MuteSyncResult(paused=True)
            await self._application.mute_enforcement.maintain_owned_timer_pauses(
                session_id=session.id,
                state=state,
            )
            if not state.enabled or session.state is MeetingState.PAUSED:
                return await self._release_all(session, channel, reason="ENFORCEMENT_INACTIVE")
            if state.paused:
                if state.pause_reason == "HOST_PAUSED":
                    return await self._release_all(
                        session,
                        channel,
                        reason="HOST_PAUSED_ENFORCEMENT",
                    )
                return MuteSyncResult(paused=True)

            members = tuple(member for member in channel.members if not member.bot)
            present = tuple(member.id for member in members)
            stale = await self._release_stale_agenda_leases(
                session=session,
                agenda_item_id=item.id,
                members={member.id: member for member in members},
            )
            if stale.paused:
                return stale
            allowed = frozenset(self._floor(status, present))
            muted = reapplied = 0
            unmuted = stale.unmuted
            released = stale.released
            for member in members:
                if member.id in allowed:
                    result = await self._release_member(
                        session=session,
                        member=member,
                        reason="FLOOR_GRANTED",
                    )
                    unmuted += result.unmuted
                    released += result.released
                    if result.paused:
                        return MuteSyncResult(
                            muted=muted,
                            unmuted=unmuted,
                            reapplied=reapplied,
                            released=released,
                            paused=True,
                        )
                else:
                    result = await self._enforce_member(
                        session=session,
                        agenda_item_id=item.id,
                        member=member,
                    )
                    muted += result.muted
                    reapplied += result.reapplied
                    released += result.released
                    if result.paused:
                        return MuteSyncResult(
                            muted=muted,
                            unmuted=unmuted,
                            reapplied=reapplied,
                            released=released,
                            paused=True,
                        )
            for lease in await self._application.mute_leases.list_active(session_id=session.id):
                if lease.user_id not in present:
                    released += await self._release_absent(lease)
            return MuteSyncResult(
                muted=muted,
                unmuted=unmuted,
                reapplied=reapplied,
                released=released,
            )

    async def _release_stale_agenda_leases(
        self,
        *,
        session: MeetingSessionRecord,
        agenda_item_id: str,
        members: dict[int, discord.Member],
    ) -> MuteSyncResult:
        result = MuteSyncResult()
        for lease in await self._application.mute_leases.list_active(session_id=session.id):
            if lease.agenda_item_id == agenda_item_id:
                continue
            member = members.get(lease.user_id)
            selected = (
                await self._release_member(
                    session=session,
                    member=member,
                    reason="AGENDA_CHANGED",
                )
                if member is not None
                else MuteSyncResult(released=await self._release_absent(lease))
            )
            result = MuteSyncResult(
                unmuted=result.unmuted + selected.unmuted,
                released=result.released + selected.released,
                paused=result.paused or selected.paused,
            )
            if result.paused:
                break
        return result

    async def release_session(
        self,
        session: MeetingSessionRecord,
        *,
        reason: str,
    ) -> MuteSyncResult:
        """Best-effort cleanup invoked before the formal voice client disconnects."""

        async with self._lock(session.guild_id):
            return await self._release_all(
                session,
                self._voice_channel(session),
                reason=reason,
            )

    async def _enforce_member(
        self,
        *,
        session: MeetingSessionRecord,
        agenda_item_id: str,
        member: discord.Member,
    ) -> MuteSyncResult:
        lease = await self._application.mute_leases.active_for_member(
            session_id=session.id,
            user_id=member.id,
        )
        server_muted = bool(member.voice and member.voice.mute)
        if lease is not None:
            if lease.ownership is MuteLeaseOwnership.OWNED and not server_muted:
                try:
                    await member.edit(mute=True, reason=f"Lingxu mute lease {lease.id}")
                except discord.Forbidden:
                    await self._pause_for_permission_by_ids(
                        session=session,
                        agenda_item_id=agenda_item_id,
                        timer_type=await self._flow_timer_type(session.id, agenda_item_id),
                        operation="REAPPLY_MUTE",
                    )
                    return MuteSyncResult(paused=True)
                except discord.HTTPException as exc:
                    if exc.status == 400:
                        return MuteSyncResult(released=await self._release_absent(lease))
                    await self._mark_uncertain(lease, "REAPPLY_HTTP_UNCERTAIN")
                    await self._pause_for_uncertainty_by_ids(
                        session=session,
                        agenda_item_id=agenda_item_id,
                        timer_type=await self._flow_timer_type(session.id, agenda_item_id),
                    )
                    return MuteSyncResult(paused=True)
                await self._application.mute_enforcement.record_lease(
                    lease=lease,
                    event_type=MeetingEventType.MUTE_REAPPLIED,
                    reason="SERVER_UNMUTE_REAPPLIED",
                )
                return MuteSyncResult(reapplied=1)
            if lease.ownership is MuteLeaseOwnership.OBSERVED and not server_muted:
                try:
                    await member.edit(mute=True, reason=f"Lingxu mute lease {lease.id}")
                except discord.Forbidden:
                    await self._pause_for_permission_by_ids(
                        session=session,
                        agenda_item_id=agenda_item_id,
                        timer_type=await self._flow_timer_type(session.id, agenda_item_id),
                        operation="REAPPLY_OBSERVED_MUTE",
                    )
                    return MuteSyncResult(paused=True)
                except discord.HTTPException as exc:
                    if exc.status == 400:
                        return MuteSyncResult(released=await self._release_absent(lease))
                    await self._pause_for_uncertainty_by_ids(
                        session=session,
                        agenda_item_id=agenda_item_id,
                        timer_type=await self._flow_timer_type(session.id, agenda_item_id),
                    )
                    return MuteSyncResult(paused=True)
                await self._application.mute_enforcement.record_lease(
                    lease=lease,
                    event_type=MeetingEventType.MUTE_REAPPLIED,
                    reason="OBSERVED_SERVER_UNMUTE_REAPPLIED",
                )
                return MuteSyncResult(reapplied=1)
            return MuteSyncResult()

        ownership = MuteLeaseOwnership.OBSERVED if server_muted else MuteLeaseOwnership.APPLYING
        lease, _ = await self._application.mute_leases.acquire(
            session_id=session.id,
            agenda_item_id=agenda_item_id,
            user_id=member.id,
            original_server_mute=server_muted,
            ownership=ownership,
            reason="PREEXISTING_SERVER_MUTE" if server_muted else "FLOOR_DENIED",
        )
        if server_muted:
            return MuteSyncResult()
        try:
            await member.edit(mute=True, reason=f"Lingxu mute lease {lease.id}")
        except discord.Forbidden:
            released = await self._release_without_edit(lease, "MUTE_FORBIDDEN")
            await self._pause_for_permission_by_ids(
                session=session,
                agenda_item_id=agenda_item_id,
                timer_type=await self._flow_timer_type(session.id, agenda_item_id),
                operation="APPLY_MUTE",
            )
            return MuteSyncResult(released=released, paused=True)
        except discord.HTTPException as exc:
            if exc.status == 400:
                return MuteSyncResult(
                    released=await self._release_without_edit(lease, "MEMBER_LEFT")
                )
            await self._mark_uncertain(lease, "APPLY_HTTP_UNCERTAIN")
            await self._pause_for_uncertainty_by_ids(
                session=session,
                agenda_item_id=agenda_item_id,
                timer_type=await self._flow_timer_type(session.id, agenda_item_id),
            )
            return MuteSyncResult(paused=True)
        await self._application.mute_leases.set_ownership(
            lease_id=lease.id,
            ownership=MuteLeaseOwnership.OWNED,
            reason="DISCORD_CONFIRMED",
        )
        return MuteSyncResult(muted=1)

    async def _release_member(
        self,
        *,
        session: MeetingSessionRecord,
        member: discord.Member,
        reason: str,
    ) -> MuteSyncResult:
        lease = await self._application.mute_leases.active_for_member(
            session_id=session.id,
            user_id=member.id,
        )
        if lease is None:
            return MuteSyncResult()
        server_muted = bool(member.voice and member.voice.mute)
        if lease.ownership is MuteLeaseOwnership.OBSERVED:
            return MuteSyncResult(released=await self._release_without_edit(lease, reason))
        if lease.ownership in {
            MuteLeaseOwnership.APPLYING,
            MuteLeaseOwnership.UNCERTAIN,
        }:
            if not server_muted:
                return MuteSyncResult(released=await self._release_without_edit(lease, reason))
            await self._mark_uncertain(lease, "OWNERSHIP_UNCERTAIN")
            await self._pause_for_uncertainty_by_ids(
                session=session,
                agenda_item_id=lease.agenda_item_id,
                timer_type=await self._flow_timer_type(session.id, lease.agenda_item_id),
            )
            return MuteSyncResult(paused=True)
        if not server_muted:
            return MuteSyncResult(released=await self._release_without_edit(lease, reason))
        try:
            await member.edit(mute=False, reason=f"Release Lingxu mute lease {lease.id}")
        except discord.Forbidden:
            await self._pause_for_permission_by_ids(
                session=session,
                agenda_item_id=lease.agenda_item_id,
                timer_type=await self._flow_timer_type(session.id, lease.agenda_item_id),
                operation="RELEASE_MUTE",
            )
            return MuteSyncResult(paused=True)
        except discord.HTTPException as exc:
            if exc.status == 400:
                return MuteSyncResult(released=await self._release_absent(lease))
            return MuteSyncResult(paused=True)
        return MuteSyncResult(
            unmuted=1,
            released=await self._release_without_edit(lease, reason),
        )

    async def _release_absent(self, lease: MeetingMuteLeaseRecord) -> int:
        return await self._release_without_edit(lease, "MEMBER_ABSENT")

    async def _release_without_edit(self, lease: MeetingMuteLeaseRecord, reason: str) -> int:
        released = await self._application.mute_leases.release(lease.id, reason=reason)
        if released is None:
            return 0
        return 1

    async def _mark_uncertain(self, lease: MeetingMuteLeaseRecord, reason: str) -> None:
        if lease.ownership is not MuteLeaseOwnership.UNCERTAIN:
            lease = await self._application.mute_leases.set_ownership(
                lease_id=lease.id,
                ownership=MuteLeaseOwnership.UNCERTAIN,
                reason=reason,
            )

    async def _release_all(
        self,
        session: MeetingSessionRecord,
        channel: discord.VoiceChannel | None,
        *,
        reason: str,
    ) -> MuteSyncResult:
        if channel is None:
            released = 0
            retained = False
            for lease in await self._application.mute_leases.list_active(session_id=session.id):
                if lease.ownership is MuteLeaseOwnership.OBSERVED:
                    released += await self._release_without_edit(lease, reason)
                else:
                    retained = True
            return MuteSyncResult(released=released, paused=retained)
        members = {member.id: member for member in channel.members}
        result = MuteSyncResult()
        for lease in await self._application.mute_leases.list_active(session_id=session.id):
            member = members.get(lease.user_id)
            selected = (
                await self._release_member(
                    session=session,
                    member=member,
                    reason=reason,
                )
                if member is not None
                else MuteSyncResult(released=await self._release_absent(lease))
            )
            result = MuteSyncResult(
                muted=result.muted,
                unmuted=result.unmuted + selected.unmuted,
                reapplied=result.reapplied,
                released=result.released + selected.released,
                paused=result.paused or selected.paused,
            )
        return result

    async def reconcile_all(self) -> None:
        """Recover every retained operational lease before the Bot reports ready."""

        session_ids = {
            lease.session_id for lease in await self._application.mute_leases.list_active()
        }
        failures = 0
        for session_id in sorted(session_ids):
            try:
                session = await self._application.sessions.get(session_id)
                await self.sync_session(session)
            except Exception as exc:
                failures += 1
                self._log.error(
                    "meeting_mute_reconciliation_failed",
                    session_id=session_id,
                    error_type=type(exc).__name__,
                )
        if failures:
            raise RuntimeError("one or more Meeting mute leases failed startup reconciliation")

    async def _flow_timer_type(self, session_id: str, agenda_item_id: str) -> str | None:
        flow = await self._application.controller.status_for_session(session_id)
        if flow.floor is None or flow.floor.agenda_item_id != agenda_item_id:
            return None
        return flow.floor.timer_type

    async def _pause_for_permission(self, status: MeetingStatus, operation: str) -> None:
        item = self._active_item(status)
        if item is None:
            return
        await self._pause_for_permission_by_ids(
            session=status.session,
            agenda_item_id=item.id,
            timer_type=status.floor.timer_type if status.floor is not None else None,
            operation=operation,
        )

    async def _pause_for_permission_by_ids(
        self,
        *,
        session: MeetingSessionRecord,
        agenda_item_id: str,
        timer_type: str | None,
        operation: str,
    ) -> None:
        state = await self._application.mute_enforcement.state(session.id, agenda_item_id)
        if state is None or state.pause_reason != "PERMISSION_LOST":
            await self._application.mute_enforcement.record_permission_lost(
                session_id=session.id,
                agenda_item_id=agenda_item_id,
                operation=operation,
            )
        await self._application.mute_enforcement.pause(
            session_id=session.id,
            agenda_item_id=agenda_item_id,
            reason="PERMISSION_LOST",
            timer_type=timer_type,
            actor_user_id=None,
            source="DISCORD",
        )

    async def _pause_for_uncertainty(self, status: MeetingStatus, reason: str) -> None:
        item = self._active_item(status)
        if item is None:
            return
        await self._pause_for_uncertainty_by_ids(
            session=status.session,
            agenda_item_id=item.id,
            timer_type=status.floor.timer_type if status.floor is not None else None,
            reason=reason,
        )

    async def _pause_for_uncertainty_by_ids(
        self,
        *,
        session: MeetingSessionRecord,
        agenda_item_id: str,
        timer_type: str | None,
        reason: str = "SIDE_EFFECT_UNCERTAIN",
    ) -> None:
        await self._application.mute_enforcement.pause(
            session_id=session.id,
            agenda_item_id=agenda_item_id,
            reason=reason,
            timer_type=timer_type,
            actor_user_id=None,
            source="SYSTEM",
        )
