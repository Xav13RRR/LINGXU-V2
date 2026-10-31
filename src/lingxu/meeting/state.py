"""Compatibility imports for domain-owned Meeting state transitions."""

from lingxu.domain.meeting_state import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    back_state,
    forward_state,
    validate_transition,
)

__all__ = [
    "ACTIVE_STATES",
    "TERMINAL_STATES",
    "back_state",
    "forward_state",
    "validate_transition",
]
