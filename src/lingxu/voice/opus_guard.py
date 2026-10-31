"""Receive error classification used by the reconnect supervisor."""

from __future__ import annotations

from typing import Any


def is_opus_decode_error(error: BaseException) -> bool:
    """Identify an Opus decode failure without persisting the exception message."""

    return "opus" in type(error).__name__.casefold()


def safe_error_label(error: BaseException | None) -> str:
    """Return a bounded, content-free label for metrics and logs."""

    if error is None:
        return "receiver_stopped"
    return type(error).__name__[:100]


_RECEIVER_COUNTER_ALLOWLIST = frozenset(
    {
        "dave_inner_decode_skipped",
        "dave_inner_decrypt_err",
        "dave_inner_decrypt_ok",
        "dave_inner_decrypt_session_not_ready",
        "dave_inner_unresolved_packets",
        "dave_marker_packets",
        "dave_needs_inner_decrypt_packets",
        "dave_parse_fail",
        "dave_parse_ok",
        "dave_strip_success",
        "dave_strip_unhandled",
        "jitter_synthetic_packets",
        "opus_decode_err",
        "opus_decode_ok",
        "opus_fec_err",
        "opus_fec_ok",
        "opus_plc_err",
        "opus_plc_ok",
        "rtp_media_audio_packets",
        "rtp_media_packets_total",
        "rtp_packets_total",
        "rtp_unknown_ssrc_dropped",
        "unknown_ssrc_expired",
        "unknown_ssrc_flushed",
        "unknown_ssrc_overflow",
        "unknown_ssrc_queued",
    }
)


def safe_receive_diagnostics(voice_client: Any) -> dict[str, int]:
    """Extract only allowlisted integer counters from the experimental receive fork."""

    getter = getattr(voice_client, "get_recv_diagnostics", None)
    if not callable(getter):
        return {}
    try:
        diagnostics = getter()
    except Exception:
        return {}
    if not isinstance(diagnostics, dict):
        return {}
    return {
        key: value
        for key, value in diagnostics.items()
        if key in _RECEIVER_COUNTER_ALLOWLIST
        and isinstance(value, int)
        and not isinstance(value, bool)
    }
