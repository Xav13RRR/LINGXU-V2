"""Atomic Session directory and manifest management."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lingxu.domain.enums import MeetingState

_SESSION_DIRECTORIES = (
    "audio",
    "realtime",
    "meeting",
    "asr",
)


def atomic_write_bytes(path: Path, payload: bytes) -> str:
    """Write bytes through `.part`, fsync, hash, and atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part")
    try:
        with part.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(payload).hexdigest()
        os.replace(part, path)
        return digest
    except BaseException:
        try:
            part.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, payload: str) -> str:
    """Atomically write UTF-8 text."""

    return atomic_write_bytes(path, payload.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class SessionManifest:
    """Minimal restart-safe Session manifest."""

    schema_version: int
    session_id: str
    guild_id: int
    creator_user_id: int
    state: str
    created_at: str
    updated_at: str

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        guild_id: int,
        creator_user_id: int,
        state: MeetingState = MeetingState.CREATED,
    ) -> SessionManifest:
        now = datetime.now(UTC).isoformat()
        return cls(
            schema_version=1,
            session_id=session_id,
            guild_id=guild_id,
            creator_user_id=creator_user_id,
            state=state.value,
            created_at=now,
            updated_at=now,
        )

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> SessionManifest:
        return cls(
            schema_version=int(payload["schema_version"]),
            session_id=str(payload["session_id"]),
            guild_id=int(payload["guild_id"]),
            creator_user_id=int(payload["creator_user_id"]),
            state=str(payload["state"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
        )

    def with_state(self, state: MeetingState) -> SessionManifest:
        return SessionManifest(
            schema_version=self.schema_version,
            session_id=self.session_id,
            guild_id=self.guild_id,
            creator_user_id=self.creator_user_id,
            state=state.value,
            created_at=self.created_at,
            updated_at=datetime.now(UTC).isoformat(),
        )


class SessionFiles:
    """Own all filesystem operations under the independent Session root."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, session_id: str) -> Path:
        """Resolve one Session directory without permitting path traversal."""

        candidate = (self.root / session_id).resolve()
        if candidate.parent != self.root:
            raise ValueError("session_id escapes the configured Session root")
        return candidate

    def resolve(self, session_id: str, relative_path: Path) -> Path:
        """Resolve one persisted relative artifact path inside its Session."""

        if relative_path.is_absolute():
            raise ValueError("Session artifact path must be relative")
        session_dir = self.directory(session_id)
        target = (session_dir / relative_path).resolve()
        if not target.is_relative_to(session_dir):
            raise ValueError("Session artifact path escapes its Session directory")
        return target

    def create(self, manifest: SessionManifest) -> Path:
        """Create the plan-defined Session skeleton and atomic manifest."""

        session_dir = self.directory(manifest.session_id)
        session_dir.mkdir(parents=False, exist_ok=False)
        for relative in _SESSION_DIRECTORIES:
            (session_dir / relative).mkdir()
        self.write_manifest(manifest)
        return session_dir

    def write_manifest(self, manifest: SessionManifest) -> str:
        payload = json.dumps(
            asdict(manifest),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        return atomic_write_text(
            self.directory(manifest.session_id) / "manifest.json",
            f"{payload}\n",
        )

    def read_manifest(self, session_id: str) -> SessionManifest:
        path = self.directory(session_id) / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Session manifest must be a JSON object")
        manifest = SessionManifest.parse(payload)
        if manifest.session_id != session_id:
            raise ValueError("Session manifest ID does not match its directory")
        return manifest

    def update_state(self, session_id: str, state: MeetingState) -> SessionManifest:
        updated = self.read_manifest(session_id).with_state(state)
        self.write_manifest(updated)
        return updated

    def atomic_write(self, session_id: str, relative_path: Path, payload: bytes) -> str:
        """Atomically write one file while preventing path escape."""

        target = self.resolve(session_id, relative_path)
        return atomic_write_bytes(target, payload)

    def append_jsonl(
        self,
        session_id: str,
        relative_path: Path,
        value: dict[str, Any],
    ) -> str:
        """Atomically append one compact JSON line by replacing the complete file."""

        target = self.resolve(session_id, relative_path)
        existing = target.read_bytes() if target.exists() else b""
        line = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return atomic_write_bytes(target, existing + line.encode("utf-8") + b"\n")

    def cleanup_parts(self) -> int:
        """Remove incomplete `.part` files only from the Session root."""

        removed = 0
        for path in self.root.rglob("*.part"):
            resolved = path.resolve()
            if resolved.is_relative_to(self.root) and resolved.is_file():
                resolved.unlink()
                removed += 1
        return removed

    def list_manifests(self) -> tuple[SessionManifest, ...]:
        manifests: list[SessionManifest] = []
        for candidate in sorted(self.root.iterdir()):
            if not candidate.is_dir():
                continue
            manifest_path = candidate / "manifest.json"
            if manifest_path.is_file():
                manifests.append(self.read_manifest(candidate.name))
        return tuple(manifests)

    def delete(self, session_id: str) -> None:
        """Delete one exact Session directory; missing directories are idempotent."""

        session_dir = self.directory(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)

    def delete_user_audio(self, session_id: str, user_id: int) -> None:
        """Delete one authoritative Discord user's exact audio subtree."""

        if user_id < 0:
            raise ValueError("Discord user_id cannot be negative")
        session_dir = self.directory(session_id)
        audio_root = (session_dir / "audio").resolve()
        user_dir = (audio_root / str(user_id)).resolve()
        if user_dir.parent != audio_root:
            raise ValueError("user audio path escapes the Session audio root")
        if user_dir.exists():
            shutil.rmtree(user_dir)
