"""Async coordination for encrypted recording metadata and personal deletion."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lingxu.domain.enums import ConsentKind
from lingxu.domain.errors import RecordingPipelineError
from lingxu.domain.models import AudioSegmentRecord
from lingxu.storage.repositories import AudioSegmentRepository
from lingxu.storage.session_files import SessionFiles
from lingxu.voice.segment_recorder import FormalRecordingPipeline

RecordingEligibility = Callable[[int, int, int], bool]
_HASH_CHUNK_BYTES = 1024 * 1024


def _user_audio_path(user_id: int, value: object) -> Path:
    selected = Path(str(value))
    expected_parent = Path("audio") / str(user_id)
    if (
        selected.is_absolute()
        or selected.parent != expected_parent
        or not selected.name.endswith(".flac.enc")
    ):
        raise ValueError("Segment path is outside its exact user audio directory")
    return selected


def _manifest_matches_segment(value: dict[str, Any], segment: AudioSegmentRecord) -> bool:
    return (
        str(value.get("id")) == segment.id
        and int(value.get("segment_number", -1)) == segment.segment_number
        and Path(str(value.get("path"))) == segment.encrypted_path
        and str(value.get("sha256")) == segment.sha256
        and int(value.get("started_ms", -1)) == segment.started_ms
        and int(value.get("ended_ms", -1)) == segment.ended_ms
        and int(value.get("bytes_encrypted", -1)) == segment.bytes_encrypted
    )


def _encrypted_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


async def reconcile_recording_artifacts(
    *,
    session_id: str,
    segments: AudioSegmentRepository,
    files: SessionFiles,
    allows_interval: RecordingEligibility,
    user_ids: frozenset[int] | None = None,
) -> None:
    """Retain only exact DB/manifest/file triples inside raw-audio consent intervals."""

    session_dir = files.directory(session_id)
    audio_root = session_dir / "audio"
    database_values = await segments.list_session(session_id)
    selected_users = set(user_ids or ())
    if user_ids is None:
        selected_users.update(segment.user_id for segment in database_values)
        for candidate in audio_root.iterdir():
            if not candidate.is_dir():
                continue
            try:
                selected_users.add(int(candidate.name))
            except ValueError as exc:
                raise ValueError("audio directory has an invalid user ID") from exc

    invalid_segment_ids: list[str] = []
    for user_id in sorted(selected_users):
        user_dir = audio_root / str(user_id)
        manifest_path = user_dir / "segments.jsonl"
        manifest_by_path: dict[Path, dict[str, Any]] = {}
        if manifest_path.is_file():
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Segment metadata must be an object")
                path = _user_audio_path(user_id, value.get("path"))
                if path in manifest_by_path:
                    raise ValueError("Segment metadata contains a duplicate path")
                manifest_by_path[path] = value

        user_segments = [segment for segment in database_values if segment.user_id == user_id]
        database_by_path: dict[Path, AudioSegmentRecord] = {}
        for segment in user_segments:
            path = _user_audio_path(user_id, segment.encrypted_path)
            if path in database_by_path:
                raise ValueError("Segment database contains a duplicate path")
            database_by_path[path] = segment

        retained_paths: set[Path] = set()
        retained_entries: list[dict[str, Any]] = []
        for path, segment in database_by_path.items():
            value = manifest_by_path.get(path)
            encrypted_path = files.resolve(session_id, path)
            metadata_matches = (
                value is not None
                and _manifest_matches_segment(value, segment)
                and encrypted_path.is_file()
                and encrypted_path.stat().st_size == segment.bytes_encrypted
                and allows_interval(user_id, segment.started_ms, segment.ended_ms)
            )
            digest_matches = metadata_matches and (
                await asyncio.to_thread(_encrypted_file_sha256, encrypted_path) == segment.sha256
            )
            if digest_matches:
                retained_paths.add(path)
                assert value is not None
                retained_entries.append(value)
                continue
            encrypted_path.unlink(missing_ok=True)
            invalid_segment_ids.append(segment.id)

        for path in manifest_by_path.keys() - retained_paths:
            files.resolve(session_id, path).unlink(missing_ok=True)
        if user_dir.is_dir():
            for encrypted_path in user_dir.glob("*.flac.enc"):
                relative_path = encrypted_path.relative_to(session_dir)
                if relative_path not in retained_paths:
                    encrypted_path.unlink()
        if manifest_path.is_file():
            payload = b"".join(
                (
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
                for value in sorted(
                    retained_entries,
                    key=lambda item: int(item["segment_number"]),
                )
            )
            files.atomic_write(
                session_id,
                Path("audio") / str(user_id) / "segments.jsonl",
                payload,
            )

    await segments.delete_ids(
        session_id=session_id,
        segment_ids=invalid_segment_ids,
    )


class RecordingCoordinator:
    """Bridge the recorder thread with async database persistence."""

    def __init__(
        self,
        *,
        session_id: str,
        pipeline: FormalRecordingPipeline,
        segments: AudioSegmentRepository,
        files: SessionFiles,
    ) -> None:
        self._session_id = session_id
        self.pipeline = pipeline
        self._segments = segments
        self._files = files

    def start(self) -> None:
        self.pipeline.start()

    async def _persist(
        self,
        completed: tuple[AudioSegmentRecord, ...],
    ) -> tuple[AudioSegmentRecord, ...]:
        persisted: list[AudioSegmentRecord] = []
        for segment in completed:
            persisted.append(await self._segments.add(segment))
        return tuple(persisted)

    async def sync_completed(self) -> tuple[AudioSegmentRecord, ...]:
        """Persist every Segment finalised since the previous sync."""

        return await self._persist(self.pipeline.recorder.drain_completed())

    async def withdraw_recording(self, user_id: int) -> tuple[AudioSegmentRecord, ...]:
        """Linearise withdrawal, wait for finalisation, then persist metadata."""

        completed = self.pipeline.withdraw(user_id, ConsentKind.RECORDING)
        if completed is not None:
            acknowledged = await asyncio.to_thread(completed.wait, 30.0)
            if not acknowledged:
                raise TimeoutError("recording withdrawal was not finalised")
        if self.pipeline.recorder.failure is not None:
            raise RecordingPipelineError(
                "segment recorder failed during withdrawal"
            ) from self.pipeline.recorder.failure
        return await self.sync_completed()

    async def withdraw_recording_at(
        self,
        user_id: int,
        effective_monotonic_ns: int,
    ) -> tuple[AudioSegmentRecord, ...]:
        """Apply a durable cutoff, await ordered close, and persist completed metadata."""

        completed = self.pipeline.withdraw_at(
            user_id=user_id,
            effective_monotonic_ns=effective_monotonic_ns,
        )
        if completed is not None:
            acknowledged = await asyncio.to_thread(completed.wait, 30.0)
            if not acknowledged:
                raise TimeoutError("recording withdrawal was not finalised")
        if self.pipeline.recorder.failure is not None:
            raise RecordingPipelineError(
                "segment recorder failed during withdrawal"
            ) from self.pipeline.recorder.failure
        persisted = await self.sync_completed()
        cutoff_ms = self.pipeline.recorder.timeline_ms(effective_monotonic_ns)
        await reconcile_recording_artifacts(
            session_id=self._session_id,
            segments=self._segments,
            files=self._files,
            allows_interval=lambda selected_user_id, started_ms, ended_ms: (
                selected_user_id == user_id and started_ms < cutoff_ms and ended_ms <= cutoff_ms
            ),
            user_ids=frozenset((user_id,)),
        )
        return tuple(
            segment
            for segment in persisted
            if segment.user_id != user_id
            or (segment.started_ms < cutoff_ms and segment.ended_ms <= cutoff_ms)
        )

    async def stop(self) -> tuple[AudioSegmentRecord, ...]:
        """Gracefully stop the recorder and persist all final Segment rows."""

        completed = await asyncio.to_thread(self.pipeline.stop)
        return await self._persist(completed)

    async def delete_user_data(self, user_id: int) -> None:
        """Stop capture, delete encrypted files, and synchronise database deletion."""

        completed = self.pipeline.withdraw_all(user_id)
        if completed is not None:
            acknowledged = await asyncio.to_thread(completed.wait, 30.0)
            if not acknowledged:
                raise TimeoutError("personal-data withdrawal was not finalised")
        await self.sync_completed()
        self.pipeline.recorder.delete_user_plaintext(user_id)
        self._files.delete_user_audio(self._session_id, user_id)
        await self._segments.delete_user(session_id=self._session_id, user_id=user_id)
