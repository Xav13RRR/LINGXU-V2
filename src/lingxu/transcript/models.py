"""Closed value objects for the deterministic canonical transcript layer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any


@dataclass(frozen=True, slots=True)
class NormalisationChange:
    """One auditable text substitution made after canonicalisation."""

    source: str
    original_text: str
    replacement_text: str
    started_offset: int
    ended_offset: int
    confidence: float
    source_entry_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CanonicalTranscriptEntry:
    """One source-traceable canonical transcript entry."""

    entry_id: str
    source_entry_ids: tuple[str, ...]
    speaker_user_id: int | None
    speaker_name: str | None
    started_ms: int
    ended_ms: int
    raw_texts: tuple[str, ...]
    text: str
    entry_type: str
    language: str | None
    confidence: float | None
    utterance_group_id: str
    normalisation_changes: tuple[NormalisationChange, ...]
    asr_tags: tuple[str, ...]
    emotion: str | None


@dataclass(frozen=True, slots=True)
class CanonicalTranscriptMetrics:
    """Content-free aggregate metrics for a canonicalisation run."""

    source_entry_count: int
    canonical_entry_count: int
    duplicate_entry_count: int
    backchannel_entry_count: int
    asr_tag_count: int


@dataclass(frozen=True, slots=True)
class CanonicalTranscript:
    """A deterministic canonical transcript and its content-free metrics."""

    entries: tuple[CanonicalTranscriptEntry, ...]
    metrics: CanonicalTranscriptMetrics
    schema_version: int = 2

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CanonicalTranscript:
        """Decode current schema 2 and legacy schema 1 with explicit audit defaults."""

        schema_version = payload.get("schema_version")
        if schema_version not in {1, 2}:
            raise ValueError("canonical transcript has unsupported schema")
        entries_value = payload.get("entries")
        metrics_value = payload.get("metrics")
        if not isinstance(entries_value, list) or not isinstance(metrics_value, dict):
            raise ValueError("canonical transcript is malformed")
        entries: list[CanonicalTranscriptEntry] = []
        for value in entries_value:
            if not isinstance(value, dict):
                raise ValueError("canonical transcript entry is malformed")
            source_entry_ids = _string_tuple(value.get("source_entry_ids"), "source IDs")
            changes_value = value.get("normalisation_changes", [])
            if not isinstance(changes_value, list):
                raise ValueError("canonical transcript audit is malformed")
            changes = tuple(
                _normalisation_change_from_dict(
                    change,
                    schema_version=schema_version,
                    legacy_source_entry_ids=source_entry_ids,
                )
                for change in changes_value
            )
            entries.append(
                CanonicalTranscriptEntry(
                    entry_id=str(value["entry_id"]),
                    source_entry_ids=source_entry_ids,
                    speaker_user_id=value["speaker_user_id"],
                    speaker_name=value["speaker_name"],
                    started_ms=int(value["started_ms"]),
                    ended_ms=int(value["ended_ms"]),
                    raw_texts=_string_tuple(value.get("raw_texts"), "raw texts"),
                    text=str(value["text"]),
                    entry_type=str(value["entry_type"]),
                    language=value["language"],
                    confidence=value["confidence"],
                    utterance_group_id=str(value["utterance_group_id"]),
                    normalisation_changes=changes,
                    asr_tags=_string_tuple(value.get("asr_tags"), "ASR tags"),
                    emotion=value["emotion"],
                )
            )
        return cls(
            entries=tuple(entries),
            metrics=CanonicalTranscriptMetrics(
                source_entry_count=int(metrics_value["source_entry_count"]),
                canonical_entry_count=int(metrics_value["canonical_entry_count"]),
                duplicate_entry_count=int(metrics_value["duplicate_entry_count"]),
                backchannel_entry_count=int(metrics_value["backchannel_entry_count"]),
                asr_tag_count=int(metrics_value["asr_tag_count"]),
            ),
            schema_version=schema_version,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping with stable field and source ordering."""

        return {
            "schema_version": self.schema_version,
            "entries": [asdict(entry) for entry in self.entries],
            "metrics": asdict(self.metrics),
        }

    @property
    def serialized_hash(self) -> str:
        """Return the SHA-256 digest of canonical JSON serialization."""

        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"canonical transcript {label} are malformed")
    return tuple(str(item) for item in value)


def _normalisation_change_from_dict(
    value: object,
    *,
    schema_version: int,
    legacy_source_entry_ids: tuple[str, ...],
) -> NormalisationChange:
    if not isinstance(value, dict):
        raise ValueError("canonical transcript audit is malformed")
    if schema_version == 1:
        confidence = float(value.get("confidence", 1.0))
        source_entry_ids = (
            _string_tuple(value["source_entry_ids"], "audit source IDs")
            if "source_entry_ids" in value
            else legacy_source_entry_ids
        )
    else:
        confidence = float(value["confidence"])
        source_entry_ids = _string_tuple(value["source_entry_ids"], "audit source IDs")
    if not 0.0 <= confidence <= 1.0 or not source_entry_ids:
        raise ValueError("canonical transcript audit provenance is malformed")
    return NormalisationChange(
        source=str(value["source"]),
        original_text=str(value["original_text"]),
        replacement_text=str(value["replacement_text"]),
        started_offset=int(value["started_offset"]),
        ended_offset=int(value["ended_offset"]),
        confidence=confidence,
        source_entry_ids=source_entry_ids,
    )
