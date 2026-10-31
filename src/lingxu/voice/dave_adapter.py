"""Safe inspection of discord.py's DAVE connection state."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DaveStatus:
    """DAVE state without privacy codes, key material, or packet content."""

    supported_by_receive_fork: bool
    enabled: bool
    ready: bool
    protocol_version: int | None
    privacy_code_available: bool


def receive_fork_supports_dave() -> bool:
    """Detect the pinned fork's inbound DAVE parser API."""

    try:
        voice_recv = importlib.import_module("discord.ext.voice_recv")
    except ImportError:
        return False
    return hasattr(voice_recv, "parse_dave_payload")


def inspect_dave_status(voice_client: Any) -> DaveStatus:
    """Read stable booleans from discord.py internals without returning sensitive values."""

    connection = getattr(voice_client, "_connection", None)
    protocol = getattr(connection, "dave_protocol_version", None)
    session = getattr(connection, "dave_session", None)
    privacy_code = getattr(voice_client, "voice_privacy_code", None)
    enabled = isinstance(protocol, int) and protocol > 0
    return DaveStatus(
        supported_by_receive_fork=receive_fork_supports_dave(),
        enabled=enabled,
        ready=bool(enabled and session is not None and getattr(session, "ready", False)),
        protocol_version=protocol if isinstance(protocol, int) else None,
        privacy_code_available=bool(privacy_code),
    )


def read_privacy_code_for_change_detection(voice_client: Any) -> str | None:
    """Return an in-memory verification code solely for equality comparison.

    Callers must never log or persist this value.
    """

    value = getattr(voice_client, "voice_privacy_code", None)
    return value if isinstance(value, str) and value else None
