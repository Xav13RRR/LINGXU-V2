"""Thread-safe, purpose-specific Meeting consent snapshots."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from lingxu.domain.enums import ConsentKind

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class MeetingConsentSnapshot:
    """Immutable consent view safe for receive-path reads."""

    realtime_asr_user_ids: frozenset[int]
    recording_user_ids: frozenset[int]
    minutes_user_ids: frozenset[int]

    def allows(self, user_id: int, kind: ConsentKind) -> bool:
        """Return whether the user granted one exact processing purpose."""

        if kind is ConsentKind.REALTIME_ASR:
            return user_id in self.realtime_asr_user_ids
        if kind is ConsentKind.RECORDING:
            return user_id in self.recording_user_ids
        return user_id in self.minutes_user_ids


class MeetingConsentRegistry:
    """Linearise grants, withdrawals, and short receive-path actions."""

    def __init__(self) -> None:
        self._grants = {kind: set[int]() for kind in ConsentKind}
        self._not_after_monotonic_ns: dict[tuple[int, ConsentKind], int] = {}
        self._lock = threading.RLock()

    def grant(self, user_id: int, kind: ConsentKind) -> bool:
        """Grant one purpose, returning whether the state changed."""

        with self._lock:
            grants = self._grants[kind]
            before = len(grants)
            grants.add(user_id)
            self._not_after_monotonic_ns.pop((user_id, kind), None)
            return len(grants) != before

    def withdraw(self, user_id: int, kind: ConsentKind) -> bool:
        """Withdraw one purpose immediately."""

        with self._lock:
            grants = self._grants[kind]
            if user_id not in grants:
                return False
            grants.remove(user_id)
            return True

    def withdraw_all(self, user_id: int) -> frozenset[ConsentKind]:
        """Withdraw every granted purpose for one user."""

        changed: set[ConsentKind] = set()
        with self._lock:
            for kind, grants in self._grants.items():
                if user_id in grants:
                    grants.remove(user_id)
                    changed.add(kind)
        return frozenset(changed)

    def withdraw_all_at_monotonic_ns(
        self,
        user_id: int,
        effective_monotonic_ns: int,
    ) -> frozenset[ConsentKind]:
        """Withdraw every purpose at one half-open receive-path cutoff."""

        if effective_monotonic_ns < 0:
            raise ValueError("effective_monotonic_ns must be non-negative")
        changed: set[ConsentKind] = set()
        with self._lock:
            for kind, grants in self._grants.items():
                if user_id in grants:
                    grants.remove(user_id)
                    changed.add(kind)
                key = (user_id, kind)
                previous = self._not_after_monotonic_ns.get(key)
                self._not_after_monotonic_ns[key] = (
                    effective_monotonic_ns
                    if previous is None
                    else min(previous, effective_monotonic_ns)
                )
        return frozenset(changed)

    def allows_at_monotonic_ns(
        self,
        user_id: int,
        kind: ConsentKind,
        received_monotonic_ns: int,
    ) -> bool:
        """Return whether one frame lies inside the current half-open live grant."""

        if received_monotonic_ns < 0:
            return False
        with self._lock:
            cutoff = self._not_after_monotonic_ns.get((user_id, kind))
            return user_id in self._grants[kind] and (
                cutoff is None or received_monotonic_ns < cutoff
            )

    def perform_if_consented(
        self,
        user_id: int,
        kind: ConsentKind,
        received_monotonic_ns: int,
        action: Callable[[], _T],
    ) -> tuple[bool, _T | None]:
        """Linearise a short action with grant and withdrawal."""

        with self._lock:
            cutoff = self._not_after_monotonic_ns.get((user_id, kind))
            if (
                received_monotonic_ns < 0
                or user_id not in self._grants[kind]
                or (cutoff is not None and received_monotonic_ns >= cutoff)
            ):
                return False, None
            return True, action()

    def snapshot(self) -> MeetingConsentSnapshot:
        """Return an immutable point-in-time view of all three purposes."""

        with self._lock:
            return MeetingConsentSnapshot(
                realtime_asr_user_ids=frozenset(self._grants[ConsentKind.REALTIME_ASR]),
                recording_user_ids=frozenset(self._grants[ConsentKind.RECORDING]),
                minutes_user_ids=frozenset(self._grants[ConsentKind.MINUTES]),
            )

    def clear(self) -> None:
        """Withdraw every purpose when a Meeting capture ends."""

        with self._lock:
            for grants in self._grants.values():
                grants.clear()
            self._not_after_monotonic_ns.clear()
