"""Short-lived, same-origin console sessions backed by an operator secret."""

from __future__ import annotations

import hmac
import secrets
import time
from dataclasses import dataclass

from aiohttp import web

COOKIE_NAME = "lingxu_console_session"
_SESSION_TTL_SECONDS = 8 * 60 * 60
_MAX_SESSIONS = 64


@dataclass(frozen=True, slots=True)
class ConsoleSession:
    id: str
    csrf_token: str
    expires_at: float


class ConsoleSessionStore:
    """Exchange the configured secret for an HttpOnly local browser session."""

    def __init__(
        self,
        expected_token: str,
        *,
        clock: object = time.monotonic,
    ) -> None:
        if len(expected_token) < 32:
            raise ValueError("Host console token must contain at least 32 characters")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._expected_token = expected_token
        self._clock = clock
        self._sessions: dict[str, ConsoleSession] = {}

    def _now(self) -> float:
        value = self._clock()
        if not isinstance(value, (float, int)):
            raise TypeError("clock must return a numeric timestamp")
        return float(value)

    def _purge(self) -> None:
        now = self._now()
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if session.expires_at <= now
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)
        while len(self._sessions) >= _MAX_SESSIONS:
            oldest = min(self._sessions.values(), key=lambda session: session.expires_at)
            self._sessions.pop(oldest.id, None)

    def login(self, candidate: str) -> ConsoleSession | None:
        """Return a fresh session without ever retaining the submitted candidate."""

        if not hmac.compare_digest(candidate, self._expected_token):
            return None
        self._purge()
        session = ConsoleSession(
            id=secrets.token_urlsafe(32),
            csrf_token=secrets.token_urlsafe(32),
            expires_at=self._now() + _SESSION_TTL_SECONDS,
        )
        self._sessions[session.id] = session
        return session

    def authenticate(self, request: web.Request) -> ConsoleSession | None:
        self._purge()
        session_id = request.cookies.get(COOKIE_NAME)
        if not session_id:
            return None
        return self._sessions.get(session_id)

    def logout(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    @staticmethod
    def require_same_origin(request: web.Request) -> None:
        origin = request.headers.get("Origin")
        if origin is not None and origin != f"{request.scheme}://{request.host}":
            raise web.HTTPForbidden(text="cross-origin request rejected")

    @staticmethod
    def require_csrf(request: web.Request, session: ConsoleSession) -> None:
        candidate = request.headers.get("X-Lingxu-CSRF", "")
        if not hmac.compare_digest(candidate, session.csrf_token):
            raise web.HTTPForbidden(text="invalid CSRF token")
