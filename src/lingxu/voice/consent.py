"""Thread-safe explicit probe consent."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")


class ConsentRegistry:
    """Maintain an in-memory set of users who explicitly opted into this probe."""

    def __init__(self) -> None:
        self._user_ids: set[int] = set()
        self._lock = threading.RLock()

    def grant(self, user_id: int) -> bool:
        """Grant consent, returning whether the set changed."""

        with self._lock:
            before = len(self._user_ids)
            self._user_ids.add(user_id)
            return len(self._user_ids) != before

    def withdraw(self, user_id: int) -> bool:
        """Withdraw consent, returning whether the set changed."""

        with self._lock:
            if user_id not in self._user_ids:
                return False
            self._user_ids.remove(user_id)
            return True

    def perform_if_consented(
        self,
        user_id: int,
        action: Callable[[], _T],
    ) -> tuple[bool, _T | None]:
        """Linearise one short receive action with grant and withdrawal."""

        with self._lock:
            if user_id not in self._user_ids:
                return False, None
            return True, action()

    def snapshot(self) -> frozenset[int]:
        """Return an immutable point-in-time snapshot."""

        with self._lock:
            return frozenset(self._user_ids)

    def clear(self) -> None:
        """Remove all consent grants when a probe ends."""

        with self._lock:
            self._user_ids.clear()
