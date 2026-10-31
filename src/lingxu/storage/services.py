"""Coordinated database and Session filesystem lifecycle operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lingxu.domain.enums import MeetingState
from lingxu.domain.models import MeetingSessionRecord
from lingxu.storage.repositories import (
    MeetingSessionRepository,
    ProcessingJobRepository,
)
from lingxu.storage.session_files import SessionFiles, SessionManifest
from lingxu.utils import new_ulid


class MeetingSessionService:
    """Keep Session database rows and directory manifests coordinated."""

    def __init__(
        self,
        sessions: MeetingSessionRepository,
        files: SessionFiles,
    ) -> None:
        self._sessions = sessions
        self._files = files

    async def create(
        self,
        *,
        guild_id: int,
        creator_user_id: int,
        session_id: str | None = None,
    ) -> MeetingSessionRecord:
        selected_id = session_id or new_ulid()
        manifest = SessionManifest.create(
            session_id=selected_id,
            guild_id=guild_id,
            creator_user_id=creator_user_id,
        )
        session_dir = self._files.create(manifest)
        try:
            return await self._sessions.create(
                guild_id=guild_id,
                creator_user_id=creator_user_id,
                manifest_path=session_dir / "manifest.json",
                session_id=selected_id,
            )
        except BaseException:
            self._files.delete(selected_id)
            raise

    async def transition(
        self,
        *,
        session_id: str,
        target: MeetingState,
        actor_user_id: int | None,
        source: str,
    ) -> MeetingSessionRecord:
        record = await self._sessions.transition(
            session_id=session_id,
            target=target,
            actor_user_id=actor_user_id,
            source=source,
        )
        self._files.update_state(session_id, record.state)
        return record

    async def delete(self, *, session_id: str, actor_user_id: int | None) -> None:
        """Tombstone, remove files, then cascade-delete the database graph."""

        await self._sessions.mark_deleted(session_id, actor_user_id=actor_user_id)
        self._files.delete(session_id)
        await self._sessions.hard_delete(session_id)


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """Content-free startup recovery summary."""

    parts_removed: int
    expired_jobs_requeued: int
    sessions_recovered: tuple[str, ...]
    tombstones_cleaned: tuple[str, ...]


class StartupRecovery:
    """Reconcile `.part`, manifests, Session rows, tombstones, and job leases."""

    def __init__(
        self,
        *,
        sessions: MeetingSessionRepository,
        jobs: ProcessingJobRepository,
        files: SessionFiles,
    ) -> None:
        self._sessions = sessions
        self._jobs = jobs
        self._files = files

    async def run(self) -> RecoveryReport:
        parts_removed = self._files.cleanup_parts()
        expired_jobs = await self._jobs.recover_expired()
        recovered: list[str] = []
        for record in await self._sessions.list_recoverable():
            try:
                manifest = self._files.read_manifest(record.id)
            except FileNotFoundError:
                manifest = SessionManifest.create(
                    session_id=record.id,
                    guild_id=record.guild_id,
                    creator_user_id=record.creator_user_id,
                    state=record.state,
                )
                self._files.directory(record.id).mkdir(parents=True, exist_ok=True)
                for relative in ("audio", "realtime", "meeting", "asr"):
                    (self._files.directory(record.id) / relative).mkdir(exist_ok=True)
                self._files.write_manifest(manifest)
            if manifest.state != record.state.value:
                self._files.update_state(record.id, record.state)
            recovered.append(record.id)
        cleaned: list[str] = []
        for record in await self._sessions.list_deleted():
            self._files.delete(record.id)
            await self._sessions.hard_delete(record.id)
            cleaned.append(record.id)
        return RecoveryReport(
            parts_removed=parts_removed,
            expired_jobs_requeued=expired_jobs,
            sessions_recovered=tuple(recovered),
            tombstones_cleaned=tuple(cleaned),
        )


def session_manifest_path(root: Path, session_id: str) -> Path:
    """Build an explicit Session manifest path for callers that need it."""

    return root / session_id / "manifest.json"
