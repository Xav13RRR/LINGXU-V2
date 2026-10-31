"""Closed, content-free locations for final-minutes contract failures."""

from dataclasses import dataclass

VALIDATION_SECTIONS = frozenset(
    {
        "document",
        "summary",
        "topics",
        "decisions",
        "agreements",
        "actions",
        "questions",
        "proposals",
        "risks",
        "disagreements",
        "transcript_edits",
    }
)
VALIDATION_REASONS = frozenset(
    {
        "invalid_json",
        "schema_invalid",
        "missing_reference",
        "unknown_reference",
        "system_only_reference",
        "insufficient_consensus_speakers",
        "unknown_owner",
        "mixed_speaker_edit",
        "nonconsecutive_edit",
        "duplicate_coverage",
        "conflicting_claim",
        "unsafe_text",
        "transcript_erasure",
    }
)


@dataclass(frozen=True, slots=True)
class MinutesValidationDetails:
    reason: str
    section: str = "document"
    item_index: int | None = None
    reference_count: int | None = None

    def __post_init__(self) -> None:
        if self.reason not in VALIDATION_REASONS or self.section not in VALIDATION_SECTIONS:
            raise ValueError("invalid minutes validation category")
        for value in (self.item_index, self.reference_count):
            if value is not None and (type(value) is not int or not 0 <= value <= 100_000):
                raise ValueError("invalid minutes validation count")

    def as_dict(self) -> dict[str, str | int]:
        return {
            key: value
            for key, value in (
                ("reason", self.reason),
                ("section", self.section),
                ("item_index", self.item_index),
                ("reference_count", self.reference_count),
            )
            if value is not None
        }
