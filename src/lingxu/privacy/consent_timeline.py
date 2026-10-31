"""Deterministic consent intervals projected from Meeting events."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from lingxu.domain.enums import ConsentKind
from lingxu.domain.models import MeetingEventRecord


@dataclass(frozen=True, slots=True)
class ConsentInterval:
    """One half-open consent interval for one user and processing purpose."""

    user_id: int
    kind: ConsentKind
    started_ms: int
    ended_ms: int | None


@dataclass(frozen=True, slots=True)
class ConsentTimeline:
    """Immutable consent eligibility projected from ordered Meeting events."""

    intervals: tuple[ConsentInterval, ...]

    @classmethod
    def from_events(cls, events: Iterable[MeetingEventRecord]) -> ConsentTimeline:
        """Project open and closed intervals without mutating the event stream."""

        opened: dict[tuple[int, ConsentKind], int] = {}
        last_effective_at_ms: dict[tuple[int, ConsentKind], int] = {}
        last_event_types: dict[tuple[int, ConsentKind], str] = {}
        completed: list[ConsentInterval] = []
        for event in sorted(events, key=lambda value: value.sequence_number):
            if event.event_type not in {"CONSENT_GRANTED", "CONSENT_WITHDRAWN"}:
                continue
            user_id = int(event.payload["subject_user_id"])
            effective_at_ms = int(event.payload["effective_at_ms"])
            if effective_at_ms < 0:
                raise ValueError("consent cutoff must be non-negative")
            for value in event.payload["consent_kinds"]:
                kind = ConsentKind(value)
                key = (user_id, kind)
                previous_cutoff = last_effective_at_ms.get(key)
                if previous_cutoff is not None and effective_at_ms < previous_cutoff:
                    raise ValueError("consent events are not monotonic")
                if event.event_type == "CONSENT_GRANTED":
                    if key in opened:
                        if previous_cutoff == effective_at_ms:
                            continue
                        raise ValueError("repeated grant at a different cutoff")
                    opened[key] = effective_at_ms
                    last_effective_at_ms[key] = effective_at_ms
                    last_event_types[key] = event.event_type
                    continue
                started_ms = opened.get(key)
                if started_ms is None:
                    if last_event_types.get(key) == event.event_type:
                        if previous_cutoff == effective_at_ms:
                            continue
                        raise ValueError("repeated withdrawal at a different cutoff")
                    raise ValueError("unmatched withdrawal")
                opened.pop(key)
                completed.append(ConsentInterval(user_id, kind, started_ms, effective_at_ms))
                last_effective_at_ms[key] = effective_at_ms
                last_event_types[key] = event.event_type
        completed.extend(
            ConsentInterval(user_id, kind, started_ms, None)
            for (user_id, kind), started_ms in opened.items()
        )
        return cls(
            tuple(
                sorted(
                    completed,
                    key=lambda item: (
                        item.user_id,
                        item.kind.value,
                        item.started_ms,
                        item.ended_ms if item.ended_ms is not None else 2**63 - 1,
                    ),
                )
            )
        )

    def allows_interval(
        self,
        user_id: int,
        kind: ConsentKind,
        started_ms: int,
        ended_ms: int,
    ) -> bool:
        """Return whether one complete interval lies inside an eligible period."""

        if started_ms < 0 or ended_ms < started_ms:
            return False
        return any(
            item.user_id == user_id
            and item.kind is kind
            and item.started_ms <= started_ms
            and (item.ended_ms is None or ended_ms < item.ended_ms)
            for item in self.intervals
        )
