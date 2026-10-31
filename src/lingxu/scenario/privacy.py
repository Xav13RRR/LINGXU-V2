"""Session-scoped anonymous voting and encrypted role-private content."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path

from lingxu.domain.errors import MeetingPermissionError
from lingxu.scenario.permissions import ScenarioRole
from lingxu.storage.encryption import FileCipher
from lingxu.storage.session_files import SessionFiles
from lingxu.utils import new_ulid


@dataclass(frozen=True, slots=True)
class AnonymousVoter:
    token_hash: str
    pseudonymous_user_id: int


class AnonymousVoteTokenService:
    """Create vote-specific unlinkable tokens without persisting Discord user IDs."""

    def __init__(self, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("anonymous vote secret must be at least 32 bytes")
        self._secret = secret

    def issue(self, *, session_id: str, vote_id: str, user_id: int) -> AnonymousVoter:
        payload = f"lingxu-anon-v1:{session_id}:{vote_id}:{user_id}".encode()
        digest = hmac.digest(self._secret, payload, "sha256")
        pseudonym = -(int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) - 1
        return AnonymousVoter(
            token_hash=hashlib.sha256(digest).hexdigest(),
            pseudonymous_user_id=pseudonym,
        )


@dataclass(frozen=True, slots=True)
class PrivateContentRef:
    id: str
    session_id: str
    audience: frozenset[ScenarioRole]
    encrypted_path: Path
    sha256: str


class PrivateContentVault:
    """Store private Scenario content encrypted and enforce audience on reads."""

    def __init__(self, *, files: SessionFiles, cipher: FileCipher) -> None:
        self._files = files
        self._cipher = cipher
        self._refs: dict[str, PrivateContentRef] = {}

    def put(
        self,
        *,
        session_id: str,
        audience: frozenset[ScenarioRole],
        content: str,
    ) -> PrivateContentRef:
        if not audience:
            raise ValueError("private content requires at least one audience role")
        content_id = new_ulid()
        payload = json.dumps(
            {"schema_version": 1, "content": content},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        relative = Path("meeting") / "private" / f"{content_id}.enc"
        digest = self._files.atomic_write(
            session_id,
            relative,
            self._cipher.encrypt(payload),
        )
        reference = PrivateContentRef(
            id=content_id,
            session_id=session_id,
            audience=audience,
            encrypted_path=self._files.directory(session_id) / relative,
            sha256=digest,
        )
        self._refs[content_id] = reference
        return reference

    def read(self, reference: PrivateContentRef, *, role: ScenarioRole) -> str:
        if role not in reference.audience and role is not ScenarioRole.HOST:
            raise MeetingPermissionError("Scenario role cannot view this private content")
        envelope = reference.encrypted_path.read_bytes()
        payload = json.loads(self._cipher.decrypt(envelope))
        return str(payload["content"])
