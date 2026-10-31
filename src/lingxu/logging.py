"""Structured logging with conservative secret and meeting-content redaction."""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from functools import partial
from typing import Any

import structlog

_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_PARTS = (
    "token",
    "api_key",
    "encryption_key",
    "secret_key",
    "dave_key",
    "mls_key",
    "privacy_code",
    "raw_audio",
    "pcm",
    "transcript",
    "minutes",
)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.casefold()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _REDACTED if _is_sensitive_key(str(key)) else _redact_value(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    return value


def redact_sensitive_fields(
    _: Any,
    __: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Redact sensitive keys recursively and reduce binary values to their lengths."""

    redacted = _redact_value(event_dict)
    if not isinstance(redacted, MutableMapping):  # pragma: no cover - guaranteed by structlog
        return event_dict
    return redacted


def _redact_known_value(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _redact_known_value(nested, secrets) for key, nested in value.items()}
    if isinstance(value, list):
        return [_redact_known_value(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_known_value(item, secrets) for item in value)
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            redacted = redacted.replace(secret, _REDACTED)
        return redacted
    return value


def redact_known_secret_values(
    _: Any,
    __: str,
    event_dict: MutableMapping[str, Any],
    *,
    secrets: tuple[str, ...],
) -> MutableMapping[str, Any]:
    """Remove configured secret values even when nested in a nominally safe string."""

    selected = tuple(
        sorted({secret for secret in secrets if len(secret) >= 4}, key=len, reverse=True)
    )
    redacted = _redact_known_value(event_dict, selected)
    if not isinstance(redacted, MutableMapping):  # pragma: no cover - guaranteed by structlog
        return event_dict
    return redacted


def configure_logging(level: str = "INFO", *, secrets: tuple[str, ...] = ()) -> None:
    """Configure standard-library and structlog output as JSON lines."""

    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_sensitive_fields,
            partial(redact_known_secret_values, secrets=secrets),
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
