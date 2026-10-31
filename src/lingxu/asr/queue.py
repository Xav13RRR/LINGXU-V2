"""Bounded fair realtime frame queue with strict per-user isolation."""

from __future__ import annotations

import threading
from collections import deque

from lingxu.voice.frames import SessionAudioFrame


class PerUserRealtimeQueue:
    """Never let one noisy user consume another user's realtime capacity."""

    def __init__(self, max_frames_per_user: int) -> None:
        if max_frames_per_user < 1:
            raise ValueError("max_frames_per_user must be positive")
        self._max = max_frames_per_user
        self._frames: dict[int, deque[SessionAudioFrame]] = {}
        self._ready_users: deque[int] = deque()
        self._ready_set: set[int] = set()
        self._closed = False
        self._condition = threading.Condition()

    def put(self, frame: SessionAudioFrame) -> bool:
        """Insert without blocking, dropping only this user's oldest frame."""

        with self._condition:
            if self._closed:
                return True
            user_frames = self._frames.setdefault(frame.user_id, deque())
            dropped = len(user_frames) >= self._max
            if dropped:
                user_frames.popleft()
            user_frames.append(frame)
            if frame.user_id not in self._ready_set:
                self._ready_users.append(frame.user_id)
                self._ready_set.add(frame.user_id)
            self._condition.notify()
            return dropped

    def get(self) -> SessionAudioFrame | None:
        """Block until the next round-robin frame or graceful close."""

        batch = self.get_many(1)
        return None if batch is None else batch[0]

    def get_many(self, max_items: int) -> tuple[SessionAudioFrame, ...] | None:
        """Block for one frame, then fairly drain a bounded batch under one lock."""

        if max_items < 1:
            raise ValueError("max_items must be positive")
        with self._condition:
            while not self._ready_users and not self._closed:
                self._condition.wait()
            if not self._ready_users:
                return None
            selected: list[SessionAudioFrame] = []
            while self._ready_users and len(selected) < max_items:
                user_id = self._ready_users.popleft()
                self._ready_set.remove(user_id)
                user_frames = self._frames[user_id]
                selected.append(user_frames.popleft())
                if user_frames:
                    self._ready_users.append(user_id)
                    self._ready_set.add(user_id)
                else:
                    del self._frames[user_id]
            return tuple(selected)

    def clear_user(self, user_id: int) -> int:
        """Discard a withdrawn user's queued PCM without touching other users."""

        with self._condition:
            removed = len(self._frames.pop(user_id, ()))
            if user_id in self._ready_set:
                self._ready_set.remove(user_id)
                self._ready_users = deque(
                    candidate for candidate in self._ready_users if candidate != user_id
                )
            return removed

    def clear_all(self) -> int:
        """Discard every queued frame while keeping the queue open."""

        with self._condition:
            removed = sum(len(frames) for frames in self._frames.values())
            self._frames.clear()
            self._ready_users.clear()
            self._ready_set.clear()
            return removed

    def close(self) -> None:
        """Allow accepted frames to drain and then release the consumer."""

        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def size_for_user(self, user_id: int) -> int:
        with self._condition:
            return len(self._frames.get(user_id, ()))
