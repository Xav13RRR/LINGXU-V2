"""Deterministic, lossless partition planning for private outcome extraction."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lingxu.transcript.models import CanonicalTranscriptEntry
from lingxu.worker.outcomes import (
    MAX_MODEL_EVENT_REFS,
    MAX_MODEL_OUTCOME_ROWS,
    MAX_MODEL_OUTCOME_TEXT_CHARS,
    MAX_OUTCOME_RESPONSE_REPAIRS,
    OUTCOME_EXTRACTION_CONFIG_VERSION,
    OUTCOME_PROMPT_REVISION,
    OutcomeCandidate,
    OutcomeCapacitySubtype,
    OutcomeEventFact,
    OutcomeExtractionMethod,
    OutcomeFailureAttempt,
    OutcomeFailureCategory,
    OutcomeFormatSubtype,
    OutcomeProviderFailure,
    OutcomeRequestKind,
    OutcomeResponseSchemaError,
    OutcomeTaskDisposition,
    OutcomeWireRowResolution,
    ParsedOutcomeBatch,
    outcome_failure_request_kind,
    outcome_json_schema,
    parse_outcome_batch_with_rows,
)
from lingxu.worker.topics import APPROVED_TOPIC_MODEL, APPROVED_TOPIC_MODEL_DIGEST, TopicSegment

MAX_TOPIC_BOUNDARY_CALLS_PER_JOB = 120
MAX_OUTCOME_PARTITIONS_PER_TOPIC = 12
MAX_OUTCOME_PARTITIONS_PER_JOB = 40
MAX_EVENT_SEARCH_STATES = 100_000
MAX_TOTAL_LLM_INFERENCE_CALLS_PER_JOB = MAX_TOPIC_BOUNDARY_CALLS_PER_JOB + (
    MAX_OUTCOME_PARTITIONS_PER_JOB * len(OutcomeRequestKind) * (1 + MAX_OUTCOME_RESPONSE_REPAIRS)
)
MAX_TOTAL_LLM_HTTP_REQUESTS_PER_JOB = MAX_TOTAL_LLM_INFERENCE_CALLS_PER_JOB + 1
MAX_LLM_WALL_SECONDS_PER_JOB = 5_400
MAX_OUTCOME_CONTEXT_CHARS = 6_000
OUTCOME_TARGET_CONTEXT_CHARS = 4_000
OutcomeTargetContextChars = Literal[2000, 2500, 3000, 4000]
OUTCOME_TARGET_LADDER: tuple[OutcomeTargetContextChars, ...] = (
    2_000,
    2_500,
    3_000,
    4_000,
)
OUTCOME_TARGET_LADDER_REVISION = hashlib.sha256(
    json.dumps(
        {
            "planner": "whole-meeting-smallest-valid-v2-pt12",
            "targets": OUTCOME_TARGET_LADDER,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()
MAX_OUTCOME_OUTPUT_TOKENS = 768
MAX_DEEPSEEK_REPAIR_OUTPUT_TOKENS = 32_768
OUTCOME_PARTITION_SCHEMA_VERSION = 15
OUTCOME_PARTITION_CONFIG_VERSION = 18
OUTCOME_DISPOSITION_REVISION = hashlib.sha256(
    b"lingxu-outcome-task-disposition-v3-two-response-repairs-including-output-limit"
).hexdigest()
_RETRYABLE_OUTCOME_RESPONSE_CATEGORIES = frozenset(
    {
        OutcomeFailureCategory.JSON,
        OutcomeFailureCategory.SCHEMA,
        OutcomeFailureCategory.OUTPUT_CAPACITY,
        OutcomeFailureCategory.OUTPUT_LIMIT,
    }
)
OUTCOME_SCHEMA_REVISION = hashlib.sha256(
    json.dumps(
        {
            "extraction_config_version": OUTCOME_EXTRACTION_CONFIG_VERSION,
            "schemas": {kind.value: outcome_json_schema(kind) for kind in OutcomeRequestKind},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


_OUTCOME_PARTITION_REASON = re.compile(r"[a-z][a-z0-9_]{0,79}")


class OutcomePartitionError(ValueError):
    """Fail closed with one validated content-free reason token."""

    def __init__(self, reason: str) -> None:
        selected = (
            reason
            if isinstance(reason, str) and _OUTCOME_PARTITION_REASON.fullmatch(reason)
            else "outcome_partition_error"
        )
        self.reason = selected
        super().__init__(selected)


class _PlanningSearchBudget:
    """One explicit planning-state budget shared by the whole target ladder."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def consume(self) -> None:
        if self.used >= self.limit:
            raise OutcomePartitionError("planning_search_capacity")
        self.used += 1


class OutcomeTargetPlanResult(StrEnum):
    SELECTED = "selected"
    REJECTED = "rejected"


class OutcomeTargetPlanFailureReason(StrEnum):
    TOPIC_PARTITION_CAP = "topic_partition_cap"
    JOB_PARTITION_CAP = "job_partition_cap"
    FRAGMENT_CAPACITY = "fragment_capacity"
    EVENT_OUTPUT_CAPACITY = "event_output_capacity"


class OutcomeTargetPlanAttempt(BaseModel):
    """Content-free result of one complete whole-meeting ladder attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    target_context_chars: OutcomeTargetContextChars
    result: OutcomeTargetPlanResult
    reason_counts: dict[OutcomeTargetPlanFailureReason, int]
    search_states_used: int = Field(ge=0, le=MAX_EVENT_SEARCH_STATES)

    @model_validator(mode="after")
    def result_matches_reasons(self) -> OutcomeTargetPlanAttempt:
        if any(count <= 0 for count in self.reason_counts.values()):
            raise ValueError("target plan reason count is invalid")
        if (self.result is OutcomeTargetPlanResult.SELECTED) != (not self.reason_counts):
            raise ValueError("target plan result is inconsistent")
        return self


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def llm_job_hash(*, canonical_hash: str, pipeline_revision: str) -> str:
    return _hash(
        {
            "canonical_hash": canonical_hash,
            "model": "qwen3.5:9b",
            "model_digest": APPROVED_TOPIC_MODEL_DIGEST,
            "pipeline_revision": pipeline_revision,
        }
    )


def digest_task_hash(job_hash: str) -> str:
    return _hash({"job_hash": job_hash, "task": "model_digest"})


def boundary_task_hash(job_hash: str, candidate_entry_id: str) -> str:
    return _hash(
        {
            "job_hash": job_hash,
            "task": "boundary_confirmation",
            "candidate_entry_id": candidate_entry_id,
        }
    )


def outcome_task_hash(
    job_hash: str,
    partition: OutcomePartition,
    kind: OutcomeRequestKind,
) -> str:
    return _hash(
        {
            "job_hash": job_hash,
            "task": "outcome_partition_kind",
            "partition_hash": partition.partition_hash,
            "request_kind": kind.value,
            "response_schema_hash": outcome_response_schema_hash(partition, kind),
        }
    )


def llm_action_hash(task_hash: str, kind: LlmActionKind, ordinal: int = 0) -> str:
    return _hash({"task_hash": task_hash, "action_kind": kind.value, "ordinal": ordinal})


class OutcomeEntryFragment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    entry_id: str = Field(min_length=1, max_length=128)
    entry_ordinal: int = Field(ge=0)
    fragment_index: int = Field(ge=0)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    started_ms: int = Field(ge=0)
    ended_ms: int = Field(ge=0)
    speaker_user_id: int | None
    speaker_name: str | None = Field(max_length=128)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def offsets_match_text(self) -> OutcomeEntryFragment:
        if self.end_offset - self.start_offset != len(self.text):
            raise ValueError("fragment offsets do not match its length")
        if self.ended_ms < self.started_ms:
            raise ValueError("fragment timestamps are invalid")
        return self

    @property
    def fragment_ref(self) -> str:
        return f"{self.entry_id}#fragment-{self.fragment_index:04d}"


class OutcomePartition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    partition_id: str = Field(min_length=1, max_length=128)
    topic_id: str = Field(min_length=1, max_length=128)
    topic_title: str = Field(min_length=1, max_length=500)
    ordinal: int = Field(ge=0)
    topic_partition_count: int = Field(default=8, ge=1, le=MAX_OUTCOME_PARTITIONS_PER_TOPIC)
    allowed_participant_ids: tuple[int, ...]
    core_entries: tuple[OutcomeEntryFragment, ...]
    overlap_entries: tuple[OutcomeEntryFragment, ...]
    core_events: tuple[OutcomeEventFact, ...]
    event_closure_exception: bool = False
    event_closure_exception_reason: Literal["event_source_closure"] | None = None
    partition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def closure_exception_is_explicit(self) -> OutcomePartition:
        if self.event_closure_exception != (self.event_closure_exception_reason is not None):
            raise ValueError("event closure exception reason is inconsistent")
        return self

    def request_bytes(self, kind: OutcomeRequestKind) -> bytes:
        return serialize_outcome_request(self, kind)

    def context_json(self, kind: OutcomeRequestKind) -> str:
        return self.request_bytes(kind).decode("utf-8")

    def resolve_fragment_refs(
        self,
        refs: tuple[str, ...],
        *,
        require_core: bool,
    ) -> tuple[str, ...]:
        visible = {
            fragment.fragment_ref: fragment
            for fragment in (*self.core_entries, *self.overlap_entries)
        }
        core_refs = {fragment.fragment_ref for fragment in self.core_entries}
        resolved: list[str] = []
        for ref in refs:
            fragment = visible.get(ref)
            if fragment is None or (require_core and ref not in core_refs):
                raise OutcomePartitionError("noncore_evidence")
            if fragment.entry_id not in resolved:
                resolved.append(fragment.entry_id)
        return tuple(resolved)

    def resolve_wire_entry_refs(
        self,
        refs: tuple[str, ...],
        *,
        require_core: bool,
    ) -> tuple[str, ...]:
        """Resolve partition-local aliases without exposing canonical IDs to the model."""

        entry_aliases, _, _, _ = _wire_alias_maps(self)
        if len(refs) != len(set(refs)):
            raise OutcomePartitionError("noncore_evidence")
        canonical_refs: list[str] = []
        for ref in refs:
            fragment = entry_aliases.get(ref)
            if fragment is None:
                raise OutcomePartitionError("noncore_evidence")
            canonical_refs.append(fragment.fragment_ref)
        return self.resolve_fragment_refs(tuple(canonical_refs), require_core=require_core)

    def resolve_wire_event_refs(self, refs: tuple[str, ...]) -> tuple[int, ...]:
        """Resolve unique event aliases to their canonical sequence numbers."""

        _, event_aliases, _, _ = _wire_alias_maps(self)
        if len(refs) != len(set(refs)):
            raise OutcomePartitionError("noncore_evidence")
        try:
            return tuple(event_aliases[ref].sequence_number for ref in refs)
        except KeyError as exc:
            raise OutcomePartitionError("noncore_evidence") from exc

    def resolve_wire_owner_ref(self, ref: str | None) -> int | None:
        """Resolve one participant alias; canonical user IDs stay outside the wire context."""

        if ref is None:
            return None
        _, _, owner_aliases, _ = _wire_alias_maps(self)
        try:
            return owner_aliases[ref]
        except KeyError as exc:
            raise OutcomePartitionError("noncore_evidence") from exc

    def resolve_wire_owner_claim(
        self,
        ref: str | None,
        entry_refs: tuple[str, ...],
    ) -> tuple[int | None, str | None]:
        """Resolve owner identity/name only from partition-local authoritative claims."""

        user_id = self.resolve_wire_owner_ref(ref)
        if user_id is None:
            return None, None
        entry_aliases, _, _, _ = _wire_alias_maps(self)
        referenced = tuple(entry_aliases[item] for item in entry_refs if item in entry_aliases)
        visible = tuple(
            sorted(
                (*self.core_entries, *self.overlap_entries),
                key=lambda item: (item.entry_ordinal, item.fragment_index),
            )
        )
        name = next(
            (
                item.speaker_name
                for item in (*referenced, *visible)
                if item.speaker_user_id == user_id and item.speaker_name is not None
            ),
            None,
        )
        if name is None:
            raise OutcomePartitionError("noncore_evidence")
        return user_id, name

    def resolve_wire_speaker_ref(self, ref: str) -> tuple[int | None, str | None]:
        """Resolve one lossless entry speaker claim, separate from owner authority."""

        _, _, _, speaker_aliases = _wire_alias_maps(self)
        try:
            return speaker_aliases[ref]
        except KeyError as exc:
            raise OutcomePartitionError("noncore_evidence") from exc

    def hash_input(self) -> dict[str, object]:
        return {
            "partition_id": self.partition_id,
            "topic_id": self.topic_id,
            "topic_title": self.topic_title,
            "ordinal": self.ordinal,
            "topic_partition_count": self.topic_partition_count,
            "allowed_participant_ids": list(self.allowed_participant_ids),
            "core_entries": [item.model_dump(mode="json") for item in self.core_entries],
            "overlap_entries": [item.model_dump(mode="json") for item in self.overlap_entries],
            "core_events": [item.model_dump(mode="json") for item in self.core_events],
            "event_closure_exception": self.event_closure_exception,
            "event_closure_exception_reason": self.event_closure_exception_reason,
        }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _short_alias(prefix: str, ordinal: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    value = ordinal
    digits = "0"
    if value:
        encoded = []
        while value:
            value, remainder = divmod(value, len(alphabet))
            encoded.append(alphabet[remainder])
        digits = "".join(reversed(encoded))
    return f"{prefix}{digits}"


def _wire_alias_maps(
    partition: OutcomePartition,
) -> tuple[
    dict[str, OutcomeEntryFragment],
    dict[str, OutcomeEventFact],
    dict[str, int],
    dict[str, tuple[int | None, str | None]],
]:
    visible = tuple(
        sorted(
            (*partition.core_entries, *partition.overlap_entries),
            key=lambda fragment: (fragment.entry_ordinal, fragment.fragment_index),
        )
    )
    refs = tuple(fragment.fragment_ref for fragment in visible)
    if len(refs) != len(set(refs)):
        raise OutcomePartitionError("noncore_evidence")
    events = tuple(sorted(partition.core_events, key=lambda event: event.sequence_number))
    sequences = tuple(event.sequence_number for event in events)
    if len(sequences) != len(set(sequences)):
        raise OutcomePartitionError("event_order")
    visible_owner_ids = {
        fragment.speaker_user_id
        for fragment in visible
        if fragment.speaker_user_id is not None and fragment.speaker_name is not None
    }
    participants = tuple(
        user_id
        for user_id in sorted(set(partition.allowed_participant_ids))
        if user_id in visible_owner_ids
    )
    speaker_claims: dict[tuple[int | None, str | None], str] = {}
    for fragment in visible:
        claim = (fragment.speaker_user_id, fragment.speaker_name)
        speaker_claims.setdefault(claim, _short_alias("p", len(speaker_claims)))
    core_refs = {fragment.fragment_ref for fragment in partition.core_entries}
    core_index = 0
    overlap_index = 0
    entry_alias_by_ref: dict[str, str] = {}
    for fragment in visible:
        if fragment.fragment_ref in core_refs:
            entry_alias_by_ref[fragment.fragment_ref] = _short_alias("e", core_index)
            core_index += 1
        else:
            entry_alias_by_ref[fragment.fragment_ref] = _short_alias("o", overlap_index)
            overlap_index += 1
    return (
        {entry_alias_by_ref[fragment.fragment_ref]: fragment for fragment in visible},
        {_short_alias("v", index): event for index, event in enumerate(events)},
        {_short_alias("s", index): user_id for index, user_id in enumerate(participants)},
        {alias: claim for claim, alias in speaker_claims.items()},
    )


def outcome_wire_aliases(
    partition: OutcomePartition,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return exact partition-local model authorities without canonical identifiers."""

    entry_aliases, event_aliases, owner_aliases, _ = _wire_alias_maps(partition)
    core_refs = {fragment.fragment_ref for fragment in partition.core_entries}
    core_entry_aliases = tuple(
        alias for alias, fragment in entry_aliases.items() if fragment.fragment_ref in core_refs
    )
    return core_entry_aliases, tuple(event_aliases), tuple(owner_aliases)


def outcome_response_schema(
    partition: OutcomePartition,
    kind: OutcomeRequestKind,
) -> dict[str, Any]:
    """Build the strict response schema from exact partition-local aliases."""

    entry_aliases, event_aliases, owner_aliases = outcome_wire_aliases(partition)
    return outcome_json_schema(
        kind,
        entry_aliases=entry_aliases,
        event_aliases=event_aliases,
        owner_aliases=owner_aliases,
        topic_partition_count=partition.topic_partition_count,
    )


def outcome_response_schema_hash(
    partition: OutcomePartition,
    kind: OutcomeRequestKind,
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "extraction_config_version": OUTCOME_EXTRACTION_CONFIG_VERSION,
                "schema": outcome_response_schema(partition, kind),
            }
        )
    ).hexdigest()


def parse_partition_outcome_batch(
    content: str,
    kind: OutcomeRequestKind,
    partition: OutcomePartition,
) -> tuple[Any, ...]:
    """Parse only aliases authorized by the exact partition-local response schema."""

    return parse_partition_outcome_batch_with_rows(content, kind, partition).candidates


def parse_partition_outcome_batch_with_rows(
    content: str,
    kind: OutcomeRequestKind,
    partition: OutcomePartition,
) -> ParsedOutcomeBatch:
    """Parse exact aliases and retain private original-row grouping for terminal authority."""

    entry_aliases, event_aliases, owner_aliases = outcome_wire_aliases(partition)
    parsed = parse_outcome_batch_with_rows(
        content,
        kind,
        allowed_entry_aliases=entry_aliases,
        allowed_event_aliases=event_aliases,
        allowed_owner_aliases=owner_aliases,
        topic_partition_count=partition.topic_partition_count,
    )
    canonical_bindings = tuple(
        (
            candidate.type,
            partition.resolve_wire_entry_refs(
                candidate.source_entry_refs,
                require_core=True,
            ),
            partition.resolve_wire_event_refs(candidate.source_event_refs),
        )
        for candidate in parsed.candidates
    )
    if len(canonical_bindings) != len(set(canonical_bindings)):
        raise OutcomeResponseSchemaError(
            OutcomeFormatSubtype.DUPLICATE_BINDING_CANONICAL_ALIAS_COLLISION
        )
    return parsed


def serialize_outcome_request(
    partition: OutcomePartition,
    kind: OutcomeRequestKind,
) -> bytes:
    """Serialize the one authoritative provider request context."""

    entry_aliases, event_aliases, owner_aliases, speaker_aliases = _wire_alias_maps(partition)
    core_refs = {fragment.fragment_ref for fragment in partition.core_entries}
    fragment_to_alias = {fragment.fragment_ref: alias for alias, fragment in entry_aliases.items()}
    owner_to_alias = {user_id: alias for alias, user_id in owner_aliases.items()}
    claim_to_alias = {claim: alias for alias, claim in speaker_aliases.items()}
    speaker_identity_aliases: dict[int, str] = {}
    for user_id, _ in speaker_aliases.values():
        if user_id is not None:
            speaker_identity_aliases.setdefault(
                user_id,
                _short_alias("u", len(speaker_identity_aliases)),
            )
    core_rows: list[list[object]] = []
    overlap_rows: list[list[object]] = []
    time_base_ms = min((fragment.started_ms for fragment in entry_aliases.values()), default=0)
    for alias, fragment in entry_aliases.items():
        speaker_ref = claim_to_alias[(fragment.speaker_user_id, fragment.speaker_name)]
        row = [
            alias,
            fragment.started_ms - time_base_ms,
            fragment.ended_ms - fragment.started_ms,
            speaker_ref,
            fragment.text,
        ]
        (core_rows if fragment.fragment_ref in core_refs else overlap_rows).append(row)
    event_rows: list[list[object]] = []
    previous_sequence = min((event.sequence_number for event in event_aliases.values()), default=0)
    event_base = previous_sequence
    for alias, event in event_aliases.items():
        source_entry_ids = frozenset(event.source_entry_ids)
        event_rows.append(
            [
                alias,
                event.sequence_number - previous_sequence,
                event.subject,
                event.result,
                [
                    fragment_to_alias[fragment.fragment_ref]
                    for fragment in entry_aliases.values()
                    if fragment.entry_id in source_entry_ids
                ],
            ]
        )
        previous_sequence = event.sequence_number
    value = {
        "c": core_rows,
        "ec": ["ref", "start_offset_ms", "duration_ms", "speaker_ref", "text"],
        "k": kind.value,
        "n": partition.ordinal,
        "o": overlap_rows,
        "p": [
            [
                alias,
                speaker_identity_aliases.get(user_id) if user_id is not None else None,
                owner_to_alias.get(user_id) if user_id is not None else None,
                display_name,
            ]
            for alias, (user_id, display_name) in speaker_aliases.items()
        ],
        "pc": ["ref", "speaker_ref", "owner_ref", "display_name"],
        "s": [[alias] for alias in owner_aliases],
        "t": partition.topic_title,
        "tb": time_base_ms,
        "v": event_base,
        "vc": ["ref", "sequence_delta", "subject", "result", "entry_refs"],
        "x": event_rows,
    }
    return _canonical_json_bytes(value)


def outcome_preflight_capacity_failure(
    partition: OutcomePartition,
    kind: OutcomeRequestKind,
) -> OutcomeCapacitySubtype | None:
    """Prove that authoritative event facts cannot fit the reviewed output wire."""

    if kind is not OutcomeRequestKind.DECISION_AGREEMENT:
        return None
    if len(partition.core_events) >= MAX_MODEL_OUTCOME_ROWS:
        return OutcomeCapacitySubtype.EVENT_ROWS
    if any(len(event.subject) > MAX_MODEL_OUTCOME_TEXT_CHARS for event in partition.core_events):
        return OutcomeCapacitySubtype.EVENT_SUBJECT
    if any(len(event.source_entry_ids) > MAX_MODEL_EVENT_REFS for event in partition.core_events):
        return OutcomeCapacitySubtype.AUTHORITATIVE_ENTRY_REFS
    return None


class OutcomeInventoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    topic_id: str = Field(min_length=1, max_length=128)
    entry_id: str = Field(min_length=1, max_length=128)
    ordinal: int = Field(ge=0)
    started_ms: int = Field(ge=0)
    ended_ms: int = Field(ge=0)
    text_length: int = Field(ge=0)
    text_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fragment_count: int = Field(ge=0)


class OutcomeInventoryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    topic_id: str = Field(min_length=1, max_length=128)
    sequence_number: int = Field(ge=1)
    ordinal: int = Field(ge=0)
    event_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class OutcomePartitionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[12] = 12
    canonical_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    topic_artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_facts_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    participant_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: Literal["qwen3.5:9b", "deepseek-v4-flash", "deepseek-flash"] = "qwen3.5:9b"
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    extraction_config_version: int = Field(ge=1)
    partition_config_version: int = Field(ge=1)
    pipeline_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    entry_inventory: tuple[OutcomeInventoryEntry, ...]
    event_inventory: tuple[OutcomeInventoryEvent, ...]
    coverage_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_ladder: tuple[OutcomeTargetContextChars, ...]
    target_ladder_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_attempts: tuple[OutcomeTargetPlanAttempt, ...] = Field(min_length=1, max_length=4)
    planning_search_state_limit: Literal[100000] = 100_000
    planning_search_states_used: int = Field(ge=0, le=MAX_EVENT_SEARCH_STATES)
    context_target_chars: OutcomeTargetContextChars
    context_hard_cap_chars: Literal[6000] = 6000
    max_request_chars: int = Field(ge=0, le=MAX_OUTCOME_CONTEXT_CHARS)
    event_closure_exception_count: int = Field(ge=0, le=MAX_OUTCOME_PARTITIONS_PER_JOB)
    event_closure_exception_reason_counts: dict[str, int]
    request_hashes: dict[str, dict[str, str]]
    response_schema_hashes: dict[str, dict[str, str]]
    partitions: tuple[OutcomePartition, ...] = Field(max_length=MAX_OUTCOME_PARTITIONS_PER_JOB)
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def target_selection_is_closed(self) -> OutcomePartitionManifest:
        attempted_targets = tuple(item.target_context_chars for item in self.target_attempts)
        selected = tuple(
            item for item in self.target_attempts if item.result is OutcomeTargetPlanResult.SELECTED
        )
        if (
            self.target_ladder != OUTCOME_TARGET_LADDER
            or self.target_ladder_revision != OUTCOME_TARGET_LADDER_REVISION
            or attempted_targets != OUTCOME_TARGET_LADDER[: len(attempted_targets)]
            or len(selected) != 1
            or self.target_attempts[-1] != selected[0]
            or self.context_target_chars != selected[0].target_context_chars
            or self.planning_search_states_used
            != sum(item.search_states_used for item in self.target_attempts)
        ):
            raise ValueError("target ladder selection is invalid")
        return self

    def hash_input(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.model_dump(mode="json").items()
            if key != "manifest_hash"
        }

    def telemetry(self) -> dict[str, object]:
        buckets = Counter(
            "lte_1k" if size <= 1_000 else "lte_4k" if size <= 4_000 else "lte_6k"
            for partition in self.partitions
            for size in (max(len(partition.context_json(kind)) for kind in OutcomeRequestKind),)
        )
        return {
            "partition_count": len(self.partitions),
            "size_buckets": dict(sorted(buckets.items())),
            "target_ladder_revision": self.target_ladder_revision,
            "target_attempts": [
                attempt.model_dump(mode="json") for attempt in self.target_attempts
            ],
            "planning_search_state_limit": self.planning_search_state_limit,
            "planning_search_states_used": self.planning_search_states_used,
            "context_target_chars": self.context_target_chars,
            "max_request_chars": self.max_request_chars,
            "event_closure_exception_count": self.event_closure_exception_count,
            "event_closure_exception_reason_counts": dict(
                sorted(self.event_closure_exception_reason_counts.items())
            ),
            "manifest_hash": self.manifest_hash,
        }


class LlmActionKind(StrEnum):
    DIGEST = "digest"
    BOUNDARY_CHAT = "boundary_chat"
    OUTCOME_INITIAL = "outcome_initial"
    OUTCOME_REPAIR = "outcome_repair"

    @property
    def inference(self) -> bool:
        return self is not LlmActionKind.DIGEST


class LlmActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    action_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: LlmActionKind
    completed: bool
    failure_category: OutcomeFailureCategory | None = None

    @model_validator(mode="after")
    def terminal_failure_is_closed(self) -> LlmActionRecord:
        if not self.completed and self.failure_category is not None:
            raise ValueError("incomplete action cannot carry a terminal failure")
        return self


class UnifiedLlmBudgetJournal(BaseModel):
    """One durable ledger for every model-related HTTP action in a Worker Job."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[3] = 3
    job_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_epoch_seconds: float = Field(ge=0)
    expected_task_hashes: tuple[str, ...]
    reserved_task_hashes: tuple[str, ...]
    completed_task_hashes: tuple[str, ...]
    result_task_hashes: tuple[str, ...]
    actions: tuple[LlmActionRecord, ...]
    actual_http_requests: int = Field(ge=0, le=MAX_TOTAL_LLM_HTTP_REQUESTS_PER_JOB)
    actual_inference_calls: int = Field(ge=0, le=MAX_TOTAL_LLM_INFERENCE_CALLS_PER_JOB)

    @model_validator(mode="after")
    def coherent(self) -> UnifiedLlmBudgetJournal:
        hash_groups = (
            self.expected_task_hashes,
            self.reserved_task_hashes,
            self.completed_task_hashes,
            self.result_task_hashes,
        )
        if any(len(group) != len(set(group)) for group in hash_groups):
            raise ValueError("budget task hashes must be unique")
        if any(
            len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
            for group in hash_groups
            for value in group
        ):
            raise ValueError("budget task hash is invalid")
        expected = set(self.expected_task_hashes)
        reserved = set(self.reserved_task_hashes)
        completed = set(self.completed_task_hashes)
        results = set(self.result_task_hashes)
        if not results <= completed <= reserved <= expected:
            raise ValueError("budget task state is incoherent")
        action_hashes = [action.action_hash for action in self.actions]
        if len(action_hashes) != len(set(action_hashes)):
            raise ValueError("budget action hashes must be unique")
        if any(action.task_hash not in reserved for action in self.actions):
            raise ValueError("budget action task was not reserved")
        if self.actual_http_requests != len(self.actions):
            raise ValueError("actual HTTP count is incoherent")
        if self.actual_inference_calls != sum(action.kind.inference for action in self.actions):
            raise ValueError("actual inference count is incoherent")
        return self

    @classmethod
    def new(
        cls,
        *,
        job_hash: str,
        expected_task_hashes: tuple[str, ...],
        now: float | None = None,
    ) -> UnifiedLlmBudgetJournal:
        return cls(
            job_hash=job_hash,
            started_epoch_seconds=time.time() if now is None else now,
            expected_task_hashes=expected_task_hashes,
            reserved_task_hashes=(),
            completed_task_hashes=(),
            result_task_hashes=(),
            actions=(),
            actual_http_requests=0,
            actual_inference_calls=0,
        )

    def validate_job(self, job_hash: str) -> None:
        if self.job_hash != job_hash:
            raise OutcomePartitionError("job_revision_mismatch")

    def validate_expected(self, task_hashes: tuple[str, ...]) -> None:
        if self.expected_task_hashes != task_hashes:
            raise OutcomePartitionError("budget_expected_mismatch")

    def transition_expected(
        self,
        *,
        prior_task_hashes: tuple[str, ...],
        task_hashes: tuple[str, ...],
    ) -> UnifiedLlmBudgetJournal:
        self.validate_expected(prior_task_hashes)
        self.validate_final()
        if task_hashes[: len(prior_task_hashes)] != prior_task_hashes or len(task_hashes) != len(
            set(task_hashes)
        ):
            raise OutcomePartitionError("budget_expected_mismatch")
        return self.model_copy(update={"expected_task_hashes": task_hashes})

    def reserve_task(self, task_hash: str) -> UnifiedLlmBudgetJournal:
        if task_hash not in self.expected_task_hashes:
            raise OutcomePartitionError("unknown_task")
        if task_hash in self.reserved_task_hashes:
            return self
        return self.model_copy(
            update={"reserved_task_hashes": (*self.reserved_task_hashes, task_hash)}
        )

    def reserve_action(
        self,
        action_hash: str,
        *,
        task_hash: str,
        kind: LlmActionKind,
        configured_timeout_seconds: float,
        now: float | None = None,
    ) -> tuple[UnifiedLlmBudgetJournal, float]:
        observed_now = time.time() if now is None else now
        remaining = MAX_LLM_WALL_SECONDS_PER_JOB - (observed_now - self.started_epoch_seconds)
        if remaining <= 0:
            raise OutcomePartitionError("wall_budget")
        if task_hash not in self.reserved_task_hashes:
            raise OutcomePartitionError("unreserved_action")
        if any(action.action_hash == action_hash for action in self.actions):
            raise OutcomePartitionError("duplicate_action")
        if self.actual_http_requests >= MAX_TOTAL_LLM_HTTP_REQUESTS_PER_JOB:
            raise OutcomePartitionError("http_budget")
        if kind.inference and self.actual_inference_calls >= MAX_TOTAL_LLM_INFERENCE_CALLS_PER_JOB:
            raise OutcomePartitionError("inference_budget")
        timeout = min(configured_timeout_seconds, remaining)
        updated = self.model_copy(
            update={
                "actions": (
                    *self.actions,
                    LlmActionRecord(
                        action_hash=action_hash,
                        task_hash=task_hash,
                        kind=kind,
                        completed=False,
                        failure_category=None,
                    ),
                ),
                "actual_http_requests": self.actual_http_requests + 1,
                "actual_inference_calls": self.actual_inference_calls + int(kind.inference),
            }
        )
        return updated, timeout

    def require_capacity(self, *, http_requests: int, inference_calls: int) -> None:
        if self.actual_http_requests + http_requests > MAX_TOTAL_LLM_HTTP_REQUESTS_PER_JOB:
            raise OutcomePartitionError("http_budget")
        if self.actual_inference_calls + inference_calls > MAX_TOTAL_LLM_INFERENCE_CALLS_PER_JOB:
            raise OutcomePartitionError("inference_budget")

    def complete_action(
        self,
        action_hash: str,
        *,
        failure_category: OutcomeFailureCategory | None = None,
    ) -> UnifiedLlmBudgetJournal:
        if not any(action.action_hash == action_hash for action in self.actions):
            raise OutcomePartitionError("unknown_action")
        return self.model_copy(
            update={
                "actions": tuple(
                    action.model_copy(
                        update={"completed": True, "failure_category": failure_category}
                    )
                    if action.action_hash == action_hash
                    else action
                    for action in self.actions
                )
            }
        )

    def complete_task(self, task_hash: str) -> UnifiedLlmBudgetJournal:
        if task_hash not in self.reserved_task_hashes:
            raise OutcomePartitionError("unreserved_result")
        if task_hash in self.completed_task_hashes:
            return self
        return self.model_copy(
            update={"completed_task_hashes": (*self.completed_task_hashes, task_hash)}
        )

    def reconcile_result(self, task_hash: str) -> UnifiedLlmBudgetJournal:
        if task_hash not in self.reserved_task_hashes:
            raise OutcomePartitionError("unreserved_result")
        completed = self.complete_task(task_hash)
        if task_hash in completed.result_task_hashes:
            return completed
        return completed.model_copy(
            update={"result_task_hashes": (*completed.result_task_hashes, task_hash)}
        )

    def validate_final(self) -> None:
        if not (
            self.expected_task_hashes
            == self.reserved_task_hashes
            == self.completed_task_hashes
            == self.result_task_hashes
        ):
            raise OutcomePartitionError("budget_final_incomplete")


class LlmBudgetController:
    """Persist each ledger transition before or after its corresponding network boundary."""

    def __init__(
        self,
        journal: UnifiedLlmBudgetJournal,
        persist: Callable[[UnifiedLlmBudgetJournal], None],
    ) -> None:
        self.journal = journal
        self._persist = persist

    @property
    def job_hash(self) -> str:
        return self.journal.job_hash

    def reserve_task(self, task_hash: str) -> None:
        self.journal = self.journal.reserve_task(task_hash)
        self._persist(self.journal)

    def reserve_action(
        self,
        action_hash: str,
        *,
        task_hash: str,
        kind: LlmActionKind,
        configured_timeout_seconds: float,
    ) -> float:
        self.journal, timeout = self.journal.reserve_action(
            action_hash,
            task_hash=task_hash,
            kind=kind,
            configured_timeout_seconds=configured_timeout_seconds,
        )
        self._persist(self.journal)
        return timeout

    def reserve_next_action(
        self,
        *,
        task_hash: str,
        kind: LlmActionKind,
        configured_timeout_seconds: float,
    ) -> tuple[str, float]:
        ordinal = sum(
            action.task_hash == task_hash and action.kind is kind for action in self.journal.actions
        )
        action_hash = llm_action_hash(task_hash, kind, ordinal)
        timeout = self.reserve_action(
            action_hash,
            task_hash=task_hash,
            kind=kind,
            configured_timeout_seconds=configured_timeout_seconds,
        )
        return action_hash, timeout

    def complete_action(
        self,
        action_hash: str,
        *,
        failure_category: OutcomeFailureCategory | None = None,
    ) -> None:
        self.journal = self.journal.complete_action(
            action_hash,
            failure_category=failure_category,
        )
        self._persist(self.journal)

    def reconcile_result(self, task_hash: str) -> None:
        self.journal = self.journal.reconcile_result(task_hash)
        self._persist(self.journal)


class OutcomePartitionResult(BaseModel):
    """Terminal parsed result; raw model responses are never persisted."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[2] = 2
    disposition_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    topic_id: str = Field(min_length=1, max_length=128)
    partition_id: str = Field(min_length=1, max_length=128)
    partition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_kind: OutcomeRequestKind
    model: Literal["qwen3.5:9b", "deepseek-v4-flash", "deepseek-flash"] = "qwen3.5:9b"
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    extraction_config_version: int = Field(ge=1)
    pipeline_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: tuple[OutcomeCandidate, ...] = Field(max_length=20)
    # The current per-kind native Schema exposes headroom up to five raw rows (PRD, Pt<=7), while
    # saturation permits at most four successful rows. The terminal artifact independently retains
    # its reviewed twenty-candidate capacity for deterministic/frozen adapters.
    wire_input_row_count: int = Field(ge=0, le=20)
    wire_row_resolutions: tuple[OutcomeWireRowResolution, ...] = Field(max_length=20)
    extraction_method: OutcomeExtractionMethod
    failure_types: tuple[str, ...] = Field(max_length=8)
    provider_failures: tuple[OutcomeProviderFailure, ...] = Field(max_length=22)
    task_disposition: OutcomeTaskDisposition
    result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def wire_row_resolution_is_complete(self) -> OutcomePartitionResult:
        output_ids = tuple(item.output_candidate_id for item in self.wire_row_resolutions)
        candidate_ids = tuple(item.id for item in self.candidates)
        contributor_ordinals = tuple(
            ordinal for item in self.wire_row_resolutions for ordinal in item.contributor_ordinals
        )
        if (
            output_ids != candidate_ids
            or tuple(item.retained_ordinal for item in self.wire_row_resolutions)
            != tuple(sorted(item.retained_ordinal for item in self.wire_row_resolutions))
            or len(contributor_ordinals) != len(set(contributor_ordinals))
            or tuple(sorted(contributor_ordinals)) != tuple(range(self.wire_input_row_count))
        ):
            raise ValueError("wire row resolution coverage is invalid")
        return self

    @model_validator(mode="after")
    def task_disposition_is_closed(self) -> OutcomePartitionResult:
        if self.disposition_revision != OUTCOME_DISPOSITION_REVISION:
            raise ValueError("outcome task disposition revision is invalid")
        if self.task_disposition is OutcomeTaskDisposition.STRICT_INITIAL_SUCCESS:
            valid = not self.failure_types and not self.provider_failures
        elif self.task_disposition is OutcomeTaskDisposition.STRICT_REPAIR_SUCCESS:
            expected_attempts = (
                OutcomeFailureAttempt.INITIAL,
                *(OutcomeFailureAttempt.REPAIR for _ in range(len(self.provider_failures) - 1)),
            )
            valid = (
                not self.failure_types
                and 1 <= len(self.provider_failures) <= MAX_OUTCOME_RESPONSE_REPAIRS
                and tuple(failure.attempt for failure in self.provider_failures)
                == expected_attempts
                and all(
                    failure.category in _RETRYABLE_OUTCOME_RESPONSE_CATEGORIES
                    for failure in self.provider_failures
                )
            )
        else:
            valid = (
                not self.candidates
                and self.wire_input_row_count == 0
                and not self.wire_row_resolutions
                and bool(self.failure_types or self.provider_failures)
            )
        if not valid:
            raise ValueError("outcome task disposition is invalid")
        return self

    @property
    def exact_duplicate_group_count(self) -> int:
        return sum(len(item.contributor_ordinals) > 1 for item in self.wire_row_resolutions)

    @property
    def dropped_wire_row_count(self) -> int:
        return self.wire_input_row_count - len(self.wire_row_resolutions)

    def hash_input(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.model_dump(mode="json").items()
            if key != "result_hash"
        }

    @classmethod
    def terminal(
        cls,
        *,
        partition: OutcomePartition,
        kind: OutcomeRequestKind,
        result: Any,
        pipeline_revision: str,
    ) -> OutcomePartitionResult:
        provided_count = result.wire_input_row_count
        provided_resolutions = tuple(result.wire_row_resolutions)
        if provided_count is None:
            provided_count = len(result.candidates)
            provided_resolutions = tuple(
                OutcomeWireRowResolution(
                    output_candidate_id=candidate.id,
                    retained_ordinal=ordinal,
                    contributor_ordinals=(ordinal,),
                )
                for ordinal, candidate in enumerate(result.candidates)
            )
        disposition = result.task_disposition
        if disposition is None:
            disposition = (
                OutcomeTaskDisposition.TERMINAL_FAILURE
                if result.failure_types or result.provider_failures
                else OutcomeTaskDisposition.STRICT_INITIAL_SUCCESS
            )
        stub = cls(
            disposition_revision=OUTCOME_DISPOSITION_REVISION,
            topic_id=partition.topic_id,
            partition_id=partition.partition_id,
            partition_hash=partition.partition_hash,
            request_kind=kind,
            model=APPROVED_TOPIC_MODEL,
            model_digest=APPROVED_TOPIC_MODEL_DIGEST,
            prompt_revision=OUTCOME_PROMPT_REVISION,
            schema_revision=OUTCOME_SCHEMA_REVISION,
            response_schema_hash=outcome_response_schema_hash(partition, kind),
            extraction_config_version=OUTCOME_EXTRACTION_CONFIG_VERSION,
            pipeline_revision=pipeline_revision,
            candidates=result.candidates,
            wire_input_row_count=provided_count,
            wire_row_resolutions=provided_resolutions,
            extraction_method=result.extraction_method,
            failure_types=result.failure_types,
            provider_failures=result.provider_failures,
            task_disposition=disposition,
            result_hash="0" * 64,
        )
        return stub.model_copy(update={"result_hash": _hash(stub.hash_input())})

    def validate_for(
        self, partition: OutcomePartition, kind: OutcomeRequestKind, pipeline_revision: str
    ) -> None:
        if (
            self.topic_id != partition.topic_id
            or self.partition_id != partition.partition_id
            or self.partition_hash != partition.partition_hash
            or self.request_kind is not kind
            or self.model != APPROVED_TOPIC_MODEL
            or self.model_digest != APPROVED_TOPIC_MODEL_DIGEST
            or self.prompt_revision != OUTCOME_PROMPT_REVISION
            or self.schema_revision != OUTCOME_SCHEMA_REVISION
            or self.response_schema_hash != outcome_response_schema_hash(partition, kind)
            or self.extraction_config_version != OUTCOME_EXTRACTION_CONFIG_VERSION
            or self.disposition_revision != OUTCOME_DISPOSITION_REVISION
            or self.pipeline_revision != pipeline_revision
            or self.result_hash != _hash(self.hash_input())
        ):
            raise OutcomePartitionError("partition_result_tamper")

    def as_topic_outcome_set(self) -> Any:
        from lingxu.worker.outcomes import TopicOutcomeSet

        extraction_method = self.extraction_method
        if self.model == APPROVED_TOPIC_MODEL:
            if extraction_method == "local_llm":
                extraction_method = "deepseek"
            elif extraction_method == "local_llm_partial":
                extraction_method = "deepseek_partial"
        return TopicOutcomeSet(
            topic_id=self.topic_id,
            candidates=self.candidates,
            extraction_method=extraction_method,
            failure_types=self.failure_types,
            provider_failures=self.provider_failures,
            wire_input_row_count=self.wire_input_row_count,
            wire_row_resolutions=self.wire_row_resolutions,
            task_disposition=self.task_disposition,
        )


class _Fix33OutcomePartitionResultV1(BaseModel):
    """Exact metadata shape written by reviewed Fix33 before disposition authority existed."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    topic_id: str = Field(min_length=1, max_length=128)
    partition_id: str = Field(min_length=1, max_length=128)
    partition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_kind: OutcomeRequestKind
    model: Literal["qwen3.5:9b", "deepseek-v4-flash", "deepseek-flash"] = "qwen3.5:9b"
    model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    extraction_config_version: int = Field(ge=1)
    pipeline_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: tuple[OutcomeCandidate, ...] = Field(max_length=20)
    wire_input_row_count: int = Field(ge=0, le=20)
    wire_row_resolutions: tuple[OutcomeWireRowResolution, ...] = Field(max_length=20)
    extraction_method: Literal["local_llm", "local_llm_partial", "deterministic_fallback"]
    failure_types: tuple[str, ...] = Field(max_length=8)
    provider_failures: tuple[OutcomeProviderFailure, ...] = Field(max_length=22)
    result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    def hash_input(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.model_dump(mode="json").items()
            if key != "result_hash"
        }

    def validate_fix33(self, partition: OutcomePartition, pipeline_revision: str) -> None:
        if (
            self.topic_id != partition.topic_id
            or self.partition_id != partition.partition_id
            or self.partition_hash != partition.partition_hash
            or self.model_digest != APPROVED_TOPIC_MODEL_DIGEST
            or self.prompt_revision != OUTCOME_PROMPT_REVISION
            or self.schema_revision != OUTCOME_SCHEMA_REVISION
            or self.response_schema_hash
            != outcome_response_schema_hash(partition, self.request_kind)
            or self.extraction_config_version != OUTCOME_EXTRACTION_CONFIG_VERSION
            or self.pipeline_revision != pipeline_revision
            or self.result_hash != _hash(self.hash_input())
        ):
            raise OutcomePartitionError("outcome_disposition_migration_required")

    def migrate(
        self,
        *,
        partition: OutcomePartition,
        journal: UnifiedLlmBudgetJournal,
        pipeline_revision: str,
    ) -> OutcomePartitionResult:
        self.validate_fix33(partition, pipeline_revision)
        task_hash = outcome_task_hash(journal.job_hash, partition, self.request_kind)
        actions = tuple(action for action in journal.actions if action.task_hash == task_hash)
        if not self.failure_types and not self.provider_failures:
            disposition = OutcomeTaskDisposition.STRICT_INITIAL_SUCCESS
        elif (
            not self.failure_types
            and self.provider_failures
            and all(
                failure.attempt is OutcomeFailureAttempt.INITIAL
                and failure.category in {OutcomeFailureCategory.JSON, OutcomeFailureCategory.SCHEMA}
                for failure in self.provider_failures
            )
            and tuple(action.kind for action in actions)
            == (LlmActionKind.OUTCOME_INITIAL, LlmActionKind.OUTCOME_REPAIR)
        ):
            disposition = OutcomeTaskDisposition.STRICT_REPAIR_SUCCESS
        else:
            disposition = OutcomeTaskDisposition.TERMINAL_FAILURE
        stub = OutcomePartitionResult(
            disposition_revision=OUTCOME_DISPOSITION_REVISION,
            topic_id=self.topic_id,
            partition_id=self.partition_id,
            partition_hash=self.partition_hash,
            request_kind=self.request_kind,
            model=self.model,
            model_digest=self.model_digest,
            prompt_revision=self.prompt_revision,
            schema_revision=self.schema_revision,
            response_schema_hash=self.response_schema_hash,
            extraction_config_version=self.extraction_config_version,
            pipeline_revision=self.pipeline_revision,
            candidates=self.candidates,
            wire_input_row_count=self.wire_input_row_count,
            wire_row_resolutions=self.wire_row_resolutions,
            extraction_method=self.extraction_method,
            failure_types=self.failure_types,
            provider_failures=self.provider_failures,
            task_disposition=disposition,
            result_hash="0" * 64,
        )
        migrated = stub.model_copy(update={"result_hash": _hash(stub.hash_input())})
        validate_outcome_task_disposition(migrated, journal, task_hash)
        return migrated


def migrate_fix33_outcome_partition_result(
    raw: object,
    *,
    partition: OutcomePartition,
    journal: UnifiedLlmBudgetJournal,
    pipeline_revision: str,
) -> OutcomePartitionResult:
    """Migrate only the exact reviewed Fix33 result shape using durable action authority."""

    try:
        legacy = _Fix33OutcomePartitionResultV1.model_validate_json(
            json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
        )
        return legacy.migrate(
            partition=partition,
            journal=journal,
            pipeline_revision=pipeline_revision,
        )
    except (TypeError, ValueError) as exc:
        raise OutcomePartitionError("outcome_disposition_migration_required") from exc


class OutcomeDispositionMetrics(BaseModel):
    """Content-safe task/fact counts derived from terminal and durable action authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider_failure_incidents: int = Field(ge=0)
    recovered_initial_incidents: int = Field(ge=0)
    recovered_task_count: int = Field(ge=0)
    terminal_provider_failure_incidents: int = Field(ge=0)
    terminal_failure_task_count: int = Field(ge=0)


def validate_outcome_task_disposition(
    result: OutcomePartitionResult,
    journal: UnifiedLlmBudgetJournal,
    task_hash: str,
) -> None:
    """Bind the result's closed disposition to the task's exact completed action trace."""

    if task_hash not in journal.result_task_hashes:
        raise OutcomePartitionError("outcome_disposition_invalid")
    expected_request_kind = outcome_failure_request_kind(result.request_kind)
    if any(
        failure.request_kind is not expected_request_kind for failure in result.provider_failures
    ):
        raise OutcomePartitionError("outcome_disposition_invalid")
    actions = tuple(action for action in journal.actions if action.task_hash == task_hash)
    if any(
        not action.completed
        or action.kind not in {LlmActionKind.OUTCOME_INITIAL, LlmActionKind.OUTCOME_REPAIR}
        for action in actions
    ):
        raise OutcomePartitionError("outcome_disposition_invalid")
    kinds = tuple(action.kind for action in actions)
    action_ordinals: Counter[LlmActionKind] = Counter()
    expected_action_hashes: list[str] = []
    for kind in kinds:
        expected_action_hashes.append(llm_action_hash(task_hash, kind, action_ordinals[kind]))
        action_ordinals[kind] += 1
    if tuple(action.action_hash for action in actions) != tuple(expected_action_hashes):
        raise OutcomePartitionError("outcome_disposition_invalid")
    initial_failures = tuple(
        failure
        for failure in result.provider_failures
        if failure.attempt is OutcomeFailureAttempt.INITIAL
    )
    repair_failures = tuple(
        failure
        for failure in result.provider_failures
        if failure.attempt is OutcomeFailureAttempt.REPAIR
    )
    preflight_failures = tuple(
        failure
        for failure in result.provider_failures
        if failure.attempt is OutcomeFailureAttempt.PREFLIGHT
    )
    if (
        len(initial_failures) > 1
        or len(repair_failures) > MAX_OUTCOME_RESPONSE_REPAIRS
        or len(preflight_failures) > 1
    ):
        raise OutcomePartitionError("outcome_disposition_invalid")

    call_boundary_categories = {
        OutcomeFailureCategory.HTTP_STATUS,
        OutcomeFailureCategory.TRANSPORT_TIMEOUT,
        OutcomeFailureCategory.TRANSPORT_ERROR,
        OutcomeFailureCategory.JSON,
        OutcomeFailureCategory.OUTPUT_LIMIT,
        OutcomeFailureCategory.ENVELOPE,
    }
    parsed_content_categories = {
        OutcomeFailureCategory.JSON,
        OutcomeFailureCategory.ENVELOPE,
        OutcomeFailureCategory.SCHEMA,
        OutcomeFailureCategory.OUTPUT_CAPACITY,
    }

    def call_matches_failure(
        action: LlmActionRecord,
        failure: OutcomeProviderFailure,
    ) -> bool:
        if action.failure_category is None:
            return failure.category in parsed_content_categories
        return (
            failure.category is action.failure_category
            and failure.category in call_boundary_categories
        )

    expected_kind_traces = tuple(
        (LlmActionKind.OUTCOME_INITIAL,) + (LlmActionKind.OUTCOME_REPAIR,) * repair_count
        for repair_count in range(MAX_OUTCOME_RESPONSE_REPAIRS + 1)
    )
    action_failures = tuple(
        failure
        for failure in result.provider_failures
        if failure.attempt is not OutcomeFailureAttempt.PREFLIGHT
    )
    expected_failure_attempts = (OutcomeFailureAttempt.INITIAL,) + (
        OutcomeFailureAttempt.REPAIR,
    ) * max(0, len(actions) - 1)
    if result.task_disposition is OutcomeTaskDisposition.STRICT_INITIAL_SUCCESS:
        valid = (
            kinds == (LlmActionKind.OUTCOME_INITIAL,)
            and actions[0].failure_category is None
            and not result.failure_types
            and not result.provider_failures
        )
    elif result.task_disposition is OutcomeTaskDisposition.STRICT_REPAIR_SUCCESS:
        valid = (
            kinds in expected_kind_traces[1:]
            and all(action.failure_category is None for action in actions)
            and len(action_failures) == len(actions) - 1
            and tuple(failure.attempt for failure in action_failures)
            == expected_failure_attempts[:-1]
            and all(
                failure.category in _RETRYABLE_OUTCOME_RESPONSE_CATEGORIES
                and call_matches_failure(action, failure)
                for action, failure in zip(actions[:-1], action_failures, strict=True)
            )
            and not preflight_failures
        )
    else:
        if result.failure_types:
            valid = not result.provider_failures and not actions
        elif tuple(failure.attempt for failure in result.provider_failures) == (
            OutcomeFailureAttempt.PREFLIGHT,
        ):
            valid = not actions
        else:
            trace_is_closed = (
                kinds in expected_kind_traces
                and bool(actions)
                and len(action_failures) == len(actions)
                and tuple(failure.attempt for failure in action_failures)
                == expected_failure_attempts
                and all(
                    call_matches_failure(action, failure)
                    for action, failure in zip(actions, action_failures, strict=True)
                )
            )
            prior_failures_are_retryable = trace_is_closed and all(
                action.failure_category is None
                and failure.category in _RETRYABLE_OUTCOME_RESPONSE_CATEGORIES
                for action, failure in zip(actions[:-1], action_failures[:-1], strict=True)
            )
            if preflight_failures:
                valid = (
                    prior_failures_are_retryable
                    and actions[-1].failure_category is None
                    and action_failures[-1].category in _RETRYABLE_OUTCOME_RESPONSE_CATEGORIES
                    and preflight_failures[0].category is OutcomeFailureCategory.INPUT_LIMIT
                )
            else:
                final_failure_is_terminal = trace_is_closed and (
                    len(actions) == 1 + MAX_OUTCOME_RESPONSE_REPAIRS
                    or actions[-1].failure_category is not None
                    or action_failures[-1].category not in _RETRYABLE_OUTCOME_RESPONSE_CATEGORIES
                )
                valid = prior_failures_are_retryable and final_failure_is_terminal
    if not valid:
        raise OutcomePartitionError("outcome_disposition_invalid")


def summarize_outcome_dispositions(
    results: dict[tuple[str, OutcomeRequestKind], OutcomePartitionResult],
    journal: UnifiedLlmBudgetJournal,
    manifest: OutcomePartitionManifest,
) -> OutcomeDispositionMetrics:
    """Validate all terminal tasks and return only closed aggregate counts."""

    partitions = {partition.partition_id: partition for partition in manifest.partitions}
    total = recovered = recovered_tasks = terminal = terminal_tasks = 0
    for (partition_id, kind), result in sorted(
        results.items(), key=lambda item: (item[0][0], item[0][1].value)
    ):
        partition = partitions.get(partition_id)
        if partition is None:
            raise OutcomePartitionError("outcome_disposition_invalid")
        task_hash = outcome_task_hash(journal.job_hash, partition, kind)
        validate_outcome_task_disposition(result, journal, task_hash)
        incident_count = sum(failure.count for failure in result.provider_failures)
        total += incident_count
        if result.task_disposition is OutcomeTaskDisposition.STRICT_REPAIR_SUCCESS:
            recovered += incident_count
            recovered_tasks += 1
        elif result.task_disposition is OutcomeTaskDisposition.TERMINAL_FAILURE:
            terminal += incident_count
            terminal_tasks += 1
    return OutcomeDispositionMetrics(
        provider_failure_incidents=total,
        recovered_initial_incidents=recovered,
        recovered_task_count=recovered_tasks,
        terminal_provider_failure_incidents=terminal,
        terminal_failure_task_count=terminal_tasks,
    )


def _safe_end(text: str, start: int, proposed: int) -> int:
    if proposed >= len(text):
        return len(text)
    floor = max(start + 1, proposed - 256)
    for index in range(proposed, floor - 1, -1):
        if text[index - 1] in "\n\r\t 。\uff01\uff1f!?\uff1b;\uff0c,":
            return index
    return proposed


def _partition(
    topic: TopicSegment,
    ordinal: int,
    core_entries: tuple[OutcomeEntryFragment, ...],
    overlap_entries: tuple[OutcomeEntryFragment, ...],
    core_events: tuple[OutcomeEventFact, ...],
    allowed_participant_ids: tuple[int, ...] = (),
    *,
    event_closure_exception_reason: Literal["event_source_closure"] | None = None,
) -> OutcomePartition:
    partition_id = f"{topic.id}-part-{ordinal + 1:02d}"
    stub = OutcomePartition(
        partition_id=partition_id,
        topic_id=topic.id,
        topic_title=topic.title,
        ordinal=ordinal,
        allowed_participant_ids=allowed_participant_ids,
        core_entries=core_entries,
        overlap_entries=overlap_entries,
        core_events=core_events,
        event_closure_exception=event_closure_exception_reason is not None,
        event_closure_exception_reason=event_closure_exception_reason,
        partition_hash="0" * 64,
    )
    return stub.model_copy(update={"partition_hash": _hash(stub.hash_input())})


def _bind_topic_partition_count(
    partitions: tuple[OutcomePartition, ...],
) -> tuple[OutcomePartition, ...]:
    """Bind the final per-topic count after planning without changing coverage or order."""

    count = len(partitions)
    if not 1 <= count <= MAX_OUTCOME_PARTITIONS_PER_TOPIC:
        raise OutcomePartitionError("topic_partition_cap")
    rebound: list[OutcomePartition] = []
    for partition in partitions:
        stub = partition.model_copy(
            update={"topic_partition_count": count, "partition_hash": "0" * 64}
        )
        rebound.append(stub.model_copy(update={"partition_hash": _hash(stub.hash_input())}))
    return tuple(rebound)


def _fits(
    partition: OutcomePartition,
    max_chars: int = OUTCOME_TARGET_CONTEXT_CHARS,
) -> bool:
    return all(len(partition.context_json(kind)) <= max_chars for kind in OutcomeRequestKind)


def _with_event_closure_exception(
    partition: OutcomePartition,
    *,
    target_context_chars: int,
) -> OutcomePartition | None:
    """Mark only an unavoidable event-source closure that fits the hard cap."""

    maximum = max(len(partition.context_json(kind)) for kind in OutcomeRequestKind)
    if (
        not partition.core_events
        or maximum <= target_context_chars
        or maximum > MAX_OUTCOME_CONTEXT_CHARS
    ):
        return None
    return _partition(
        TopicSegment(
            id=partition.topic_id,
            title=partition.topic_title,
            entry_ids=tuple(dict.fromkeys(item.entry_id for item in partition.core_entries)),
            started_ms=0,
            ended_ms=0,
            method="partition-rebuild",
        ),
        partition.ordinal,
        partition.core_entries,
        partition.overlap_entries,
        partition.core_events,
        partition.allowed_participant_ids,
        event_closure_exception_reason="event_source_closure",
    )


def _event_candidate_rank(
    candidate: tuple[int, OutcomePartition, int],
) -> tuple[int, int, int]:
    """Prefer the smallest four-kind request, then overlap growth, then source order."""

    index, partition, added_overlap_count = candidate
    maximum = max(len(partition.context_json(kind)) for kind in OutcomeRequestKind)
    return maximum, added_overlap_count, index


def _dynamic_fragments(
    topic: TopicSegment,
    entry: CanonicalTranscriptEntry,
    entry_ordinal: int,
    allowed_participant_ids: tuple[int, ...],
    target_context_chars: int,
    search_budget: _PlanningSearchBudget,
) -> tuple[OutcomeEntryFragment, ...]:
    if not entry.text:
        return ()
    fragments: list[OutcomeEntryFragment] = []
    start = 0
    fragment_index = 0
    while start < len(entry.text):
        low = start + 1
        high = len(entry.text)
        best: int | None = None
        while low <= high:
            search_budget.consume()
            proposed = (low + high) // 2
            fragment = OutcomeEntryFragment(
                entry_id=entry.entry_id,
                entry_ordinal=entry_ordinal,
                fragment_index=fragment_index,
                start_offset=start,
                end_offset=proposed,
                started_ms=entry.started_ms,
                ended_ms=entry.ended_ms,
                speaker_user_id=entry.speaker_user_id,
                speaker_name=entry.speaker_name,
                text=entry.text[start:proposed],
            )
            if _fits(
                _partition(
                    topic,
                    0,
                    (fragment,),
                    (),
                    (),
                    allowed_participant_ids,
                ),
                target_context_chars,
            ):
                best = proposed
                low = proposed + 1
            else:
                high = proposed - 1
        if best is None:
            raise OutcomePartitionError("fragment_capacity")
        end = _safe_end(entry.text, start, best)
        fragments.append(
            OutcomeEntryFragment(
                entry_id=entry.entry_id,
                entry_ordinal=entry_ordinal,
                fragment_index=fragment_index,
                start_offset=start,
                end_offset=end,
                started_ms=entry.started_ms,
                ended_ms=entry.ended_ms,
                speaker_user_id=entry.speaker_user_id,
                speaker_name=entry.speaker_name,
                text=entry.text[start:end],
            )
        )
        start = end
        fragment_index += 1
    return tuple(fragments)


def _validate_partition_inputs(
    topic: TopicSegment,
    entries: tuple[CanonicalTranscriptEntry, ...],
    events: tuple[OutcomeEventFact, ...],
    target_context_chars: int,
) -> None:
    if not 1 <= target_context_chars <= MAX_OUTCOME_CONTEXT_CHARS:
        raise OutcomePartitionError("context_target")
    entry_ids = tuple(entry.entry_id for entry in entries)
    if entry_ids != topic.entry_ids or len(entry_ids) != len(set(entry_ids)):
        raise OutcomePartitionError("entry_order")
    event_sequences = tuple(event.sequence_number for event in events)
    if event_sequences != tuple(sorted(event_sequences)) or len(event_sequences) != len(
        set(event_sequences)
    ):
        raise OutcomePartitionError("event_order")


def _build_core_fragments(
    topic: TopicSegment,
    entries: tuple[CanonicalTranscriptEntry, ...],
    allowed_participant_ids: tuple[int, ...],
    target_context_chars: int,
    search_budget: _PlanningSearchBudget,
) -> tuple[OutcomeEntryFragment, ...]:
    return tuple(
        fragment
        for entry_ordinal, entry in enumerate(entries)
        for fragment in _dynamic_fragments(
            topic,
            entry,
            entry_ordinal,
            allowed_participant_ids,
            target_context_chars,
            search_budget,
        )
    )


def _build_entry_partitions(
    topic: TopicSegment,
    core_fragments: tuple[OutcomeEntryFragment, ...],
    allowed_participant_ids: tuple[int, ...],
    target_context_chars: int,
    search_budget: _PlanningSearchBudget,
) -> tuple[OutcomePartition, ...]:
    built: list[OutcomePartition] = []
    current: list[OutcomeEntryFragment] = []
    for fragment in core_fragments:
        search_budget.consume()
        trial = _partition(
            topic, len(built), tuple((*current, fragment)), (), (), allowed_participant_ids
        )
        if _fits(trial, target_context_chars):
            current.append(fragment)
            continue
        if current:
            built.append(
                _partition(topic, len(built), tuple(current), (), (), allowed_participant_ids)
            )
            current = []
        if len(built) >= MAX_OUTCOME_PARTITIONS_PER_TOPIC:
            raise OutcomePartitionError("topic_partition_cap")
        trial = _partition(topic, len(built), (fragment,), (), (), allowed_participant_ids)
        if not _fits(trial, target_context_chars):
            raise OutcomePartitionError("fragment_capacity")
        current.append(fragment)
    if current or not built:
        built.append(_partition(topic, len(built), tuple(current), (), (), allowed_participant_ids))
    return tuple(built)


def _build_event_closures(
    core_fragments: tuple[OutcomeEntryFragment, ...],
    events: tuple[OutcomeEventFact, ...],
) -> tuple[tuple[OutcomeEntryFragment, ...], ...]:
    fragments_by_entry: dict[str, list[OutcomeEntryFragment]] = {}
    for fragment in core_fragments:
        fragments_by_entry.setdefault(fragment.entry_id, []).append(fragment)
    if len(events) > MAX_OUTCOME_PARTITIONS_PER_TOPIC * (MAX_MODEL_OUTCOME_ROWS - 1):
        raise OutcomePartitionError("event_output_capacity")
    closures: list[tuple[OutcomeEntryFragment, ...]] = []
    for event in events:
        if any(source_id not in fragments_by_entry for source_id in event.source_entry_ids):
            raise OutcomePartitionError("event_source_closure")
        closures.append(
            tuple(
                fragment
                for source_id in event.source_entry_ids
                for fragment in fragments_by_entry[source_id]
            )
        )
    return tuple(closures)


_EventPlacementCandidate = tuple[int, OutcomePartition, int]


class _OutcomeEventPlanner:
    """Deterministic bounded backtracking for event-to-partition placement."""

    def __init__(
        self,
        topic: TopicSegment,
        events: tuple[OutcomeEventFact, ...],
        event_closures: tuple[tuple[OutcomeEntryFragment, ...], ...],
        allowed_participant_ids: tuple[int, ...],
        target_context_chars: int,
        search_budget: _PlanningSearchBudget,
    ) -> None:
        self.topic = topic
        self.events = events
        self.event_closures = event_closures
        self.allowed_participant_ids = allowed_participant_ids
        self.target_context_chars = target_context_chars
        self.search_budget = search_budget
        self.memo: dict[
            tuple[bool, int, int, tuple[str, ...]],
            tuple[OutcomePartition, ...] | None,
        ] = {}

    @staticmethod
    def _available_event_slots(
        partitions: tuple[OutcomePartition, ...],
        minimum_index: int,
    ) -> int:
        return sum(
            MAX_MODEL_OUTCOME_ROWS - 1 - len(partition.core_events)
            for partition in partitions[minimum_index:]
        ) + (MAX_OUTCOME_PARTITIONS_PER_TOPIC - len(partitions)) * (MAX_MODEL_OUTCOME_ROWS - 1)

    @staticmethod
    def _objective(
        partitions: tuple[OutcomePartition, ...],
    ) -> tuple[int, int, int, tuple[int, ...], tuple[str, ...]]:
        return (
            sum(partition.event_closure_exception for partition in partitions),
            max(
                (
                    len(partition.context_json(kind))
                    for partition in partitions
                    for kind in OutcomeRequestKind
                ),
                default=0,
            ),
            sum(len(partition.overlap_entries) for partition in partitions),
            tuple(partition.ordinal for partition in partitions for _ in partition.core_events),
            tuple(partition.partition_hash for partition in partitions),
        )

    def _existing_candidates(
        self,
        event: OutcomeEventFact,
        closure: tuple[OutcomeEntryFragment, ...],
        partitions: tuple[OutcomePartition, ...],
        minimum_index: int,
    ) -> list[_EventPlacementCandidate]:
        candidates: list[_EventPlacementCandidate] = []
        for index in range(minimum_index, len(partitions)):
            target = partitions[index]
            if len(target.core_events) >= MAX_MODEL_OUTCOME_ROWS - 1:
                continue
            existing_refs = {
                item.fragment_ref for item in (*target.core_entries, *target.overlap_entries)
            }
            overlap = tuple(item for item in closure if item.fragment_ref not in existing_refs)
            trial = _partition(
                self.topic,
                target.ordinal,
                target.core_entries,
                tuple((*target.overlap_entries, *overlap)),
                tuple((*target.core_events, event)),
                self.allowed_participant_ids,
            )
            candidates.append((index, trial, len(overlap)))
        return candidates

    def _new_candidate(
        self,
        event: OutcomeEventFact,
        closure: tuple[OutcomeEntryFragment, ...],
        partitions: tuple[OutcomePartition, ...],
    ) -> _EventPlacementCandidate | None:
        if len(partitions) >= MAX_OUTCOME_PARTITIONS_PER_TOPIC:
            return None
        trial = _partition(
            self.topic,
            len(partitions),
            (),
            closure,
            (event,),
            self.allowed_participant_ids,
        )
        return len(partitions), trial, len(closure)

    def _hard_candidates(
        self,
        candidates: tuple[_EventPlacementCandidate, ...],
    ) -> list[_EventPlacementCandidate]:
        selected: list[_EventPlacementCandidate] = []
        for index, trial, added_overlap_count in candidates:
            hard_selected = _with_event_closure_exception(
                trial,
                target_context_chars=self.target_context_chars,
            )
            if hard_selected is not None:
                selected.append((index, hard_selected, added_overlap_count))
        selected.sort(key=_event_candidate_rank)
        return selected

    def _candidate_tiers(
        self,
        event: OutcomeEventFact,
        closure: tuple[OutcomeEntryFragment, ...],
        partitions: tuple[OutcomePartition, ...],
        minimum_index: int,
        *,
        allow_hard: bool,
    ) -> tuple[list[_EventPlacementCandidate], ...]:
        existing = self._existing_candidates(event, closure, partitions, minimum_index)
        normal_existing = [
            candidate for candidate in existing if _fits(candidate[1], self.target_context_chars)
        ]
        normal_existing.sort(key=_event_candidate_rank)

        new_candidate = self._new_candidate(event, closure, partitions)
        normal_new = (
            [new_candidate]
            if new_candidate is not None and _fits(new_candidate[1], self.target_context_chars)
            else []
        )
        hard = self._hard_candidates(
            (*existing, *((new_candidate,) if new_candidate is not None else ()))
        )
        if allow_hard:
            return ([*normal_existing, *normal_new, *hard],)
        return normal_existing, normal_new

    @staticmethod
    def _apply_candidate(
        partitions: tuple[OutcomePartition, ...],
        index: int,
        selected: OutcomePartition,
    ) -> tuple[OutcomePartition, ...]:
        updated = list(partitions)
        if index == len(updated):
            updated.append(selected)
        else:
            updated[index] = selected
        return tuple(updated)

    def _search(
        self,
        event_index: int,
        partitions: tuple[OutcomePartition, ...],
        minimum_index: int,
        *,
        allow_hard: bool,
    ) -> tuple[OutcomePartition, ...] | None:
        if event_index == len(self.events):
            return partitions
        state = (
            allow_hard,
            event_index,
            minimum_index,
            tuple(item.partition_hash for item in partitions),
        )
        if state in self.memo:
            return self.memo[state]
        self.search_budget.consume()

        tiers = self._candidate_tiers(
            self.events[event_index],
            self.event_closures[event_index],
            partitions,
            minimum_index,
            allow_hard=allow_hard,
        )
        for tier in tiers:
            best: tuple[OutcomePartition, ...] | None = None
            for index, selected, _ in tier:
                updated = self._apply_candidate(partitions, index, selected)
                remaining = len(self.events) - event_index - 1
                if self._available_event_slots(updated, index) < remaining:
                    continue
                completed = self._search(
                    event_index + 1,
                    updated,
                    index,
                    allow_hard=allow_hard,
                )
                if completed is not None and (
                    best is None or self._objective(completed) < self._objective(best)
                ):
                    best = completed
            if best is not None:
                self.memo[state] = best
                return best
        self.memo[state] = None
        return None

    def plan(
        self,
        partitions: tuple[OutcomePartition, ...],
    ) -> tuple[OutcomePartition, ...]:
        completed = self._search(0, partitions, 0, allow_hard=False)
        if completed is None:
            completed = self._search(0, partitions, 0, allow_hard=True)
        if completed is None:
            raise OutcomePartitionError("event_output_capacity")
        return completed


def plan_outcome_partitions(
    topic: TopicSegment,
    entries: tuple[CanonicalTranscriptEntry, ...],
    events: tuple[OutcomeEventFact, ...],
    allowed_participant_ids: tuple[int, ...] = (),
    *,
    target_context_chars: int = OUTCOME_TARGET_CONTEXT_CHARS,
    _search_budget: _PlanningSearchBudget | None = None,
) -> tuple[OutcomePartition, ...]:
    """Cover every entry codepoint and event once; overlap exists only for event closure."""

    _validate_partition_inputs(topic, entries, events, target_context_chars)
    search_budget = (
        _PlanningSearchBudget(MAX_EVENT_SEARCH_STATES) if _search_budget is None else _search_budget
    )
    core_fragments = _build_core_fragments(
        topic,
        entries,
        allowed_participant_ids,
        target_context_chars,
        search_budget,
    )
    built = _build_entry_partitions(
        topic,
        core_fragments,
        allowed_participant_ids,
        target_context_chars,
        search_budget,
    )
    completed = _OutcomeEventPlanner(
        topic,
        events,
        _build_event_closures(core_fragments, events),
        allowed_participant_ids,
        target_context_chars,
        search_budget,
    ).plan(built)
    return _bind_topic_partition_count(completed)


_TARGET_RETRY_REASONS = {
    reason.value: reason
    for reason in (
        OutcomeTargetPlanFailureReason.TOPIC_PARTITION_CAP,
        OutcomeTargetPlanFailureReason.JOB_PARTITION_CAP,
        OutcomeTargetPlanFailureReason.FRAGMENT_CAPACITY,
        OutcomeTargetPlanFailureReason.EVENT_OUTPUT_CAPACITY,
    )
}


def _plan_adaptive_whole_meeting(
    topics: tuple[
        tuple[TopicSegment, tuple[CanonicalTranscriptEntry, ...], tuple[OutcomeEventFact, ...]], ...
    ],
    allowed_participant_ids: tuple[int, ...],
) -> tuple[
    tuple[OutcomePartition, ...],
    OutcomeTargetContextChars,
    tuple[OutcomeTargetPlanAttempt, ...],
    int,
]:
    """Select the first complete whole-meeting plan under one shared search budget."""

    budget = _PlanningSearchBudget(MAX_EVENT_SEARCH_STATES)
    attempts: list[OutcomeTargetPlanAttempt] = []
    for target_context_chars in OUTCOME_TARGET_LADDER:
        started_states = budget.used
        partitions: list[OutcomePartition] = []
        reason_counts: Counter[OutcomeTargetPlanFailureReason] = Counter()
        for topic, entries, events in topics:
            try:
                partitions.extend(
                    plan_outcome_partitions(
                        topic,
                        entries,
                        events,
                        allowed_participant_ids,
                        target_context_chars=target_context_chars,
                        _search_budget=budget,
                    )
                )
            except OutcomePartitionError as exc:
                retry_reason = _TARGET_RETRY_REASONS.get(str(exc))
                if retry_reason is None:
                    raise
                reason_counts[retry_reason] += 1
        if len(partitions) > MAX_OUTCOME_PARTITIONS_PER_JOB:
            reason_counts[OutcomeTargetPlanFailureReason.JOB_PARTITION_CAP] += 1
        states_used = budget.used - started_states
        if reason_counts:
            attempts.append(
                OutcomeTargetPlanAttempt(
                    target_context_chars=target_context_chars,
                    result=OutcomeTargetPlanResult.REJECTED,
                    reason_counts=dict(
                        sorted(reason_counts.items(), key=lambda item: item[0].value)
                    ),
                    search_states_used=states_used,
                )
            )
            continue
        attempts.append(
            OutcomeTargetPlanAttempt(
                target_context_chars=target_context_chars,
                result=OutcomeTargetPlanResult.SELECTED,
                reason_counts={},
                search_states_used=states_used,
            )
        )
        return tuple(partitions), target_context_chars, tuple(attempts), budget.used
    observed_reasons = {reason.value for attempt in attempts for reason in attempt.reason_counts}
    if len(observed_reasons) == 1:
        raise OutcomePartitionError(observed_reasons.pop())
    raise OutcomePartitionError("adaptive_target_unavailable")


def build_outcome_partition_manifest(
    *,
    canonical_hash: str,
    topic_artifact_hash: str,
    event_facts_hash: str,
    participant_hash: str,
    topics: tuple[
        tuple[TopicSegment, tuple[CanonicalTranscriptEntry, ...], tuple[OutcomeEventFact, ...]], ...
    ],
    pipeline_revision: str = hashlib.sha256(
        b"outcome-partition-pipeline-speaker-claims-fix12"
    ).hexdigest(),
    allowed_participant_ids: tuple[int, ...] = (),
) -> OutcomePartitionManifest:
    (
        partitions,
        selected_target_context_chars,
        target_attempts,
        planning_search_states_used,
    ) = _plan_adaptive_whole_meeting(
        topics,
        allowed_participant_ids,
    )
    fragment_counts = Counter(
        fragment.entry_id for partition in partitions for fragment in partition.core_entries
    )
    entry_inventory = tuple(
        OutcomeInventoryEntry(
            topic_id=topic.id,
            entry_id=entry.entry_id,
            ordinal=ordinal,
            started_ms=entry.started_ms,
            ended_ms=entry.ended_ms,
            text_length=len(entry.text),
            text_hash=hashlib.sha256(entry.text.encode("utf-8")).hexdigest(),
            fragment_count=fragment_counts[entry.entry_id],
        )
        for topic, entries, _ in topics
        for ordinal, entry in enumerate(entries)
    )
    event_inventory = tuple(
        OutcomeInventoryEvent(
            topic_id=topic.id,
            sequence_number=event.sequence_number,
            ordinal=ordinal,
            event_hash=_hash(event.model_dump(mode="json")),
        )
        for topic, _, events in topics
        for ordinal, event in enumerate(events)
    )
    coverage_input = {
        "entries": [
            {
                "partition_id": partition.partition_id,
                "entry_id": fragment.entry_id,
                "fragment_index": fragment.fragment_index,
                "start_offset": fragment.start_offset,
                "end_offset": fragment.end_offset,
            }
            for partition in partitions
            for fragment in partition.core_entries
        ],
        "events": [
            {
                "partition_id": partition.partition_id,
                "sequence_number": event.sequence_number,
            }
            for partition in partitions
            for event in partition.core_events
        ],
    }
    request_hashes = {
        partition.partition_id: {
            kind.value: hashlib.sha256(serialize_outcome_request(partition, kind)).hexdigest()
            for kind in OutcomeRequestKind
        }
        for partition in partitions
    }
    response_schema_hashes = {
        partition.partition_id: {
            kind.value: outcome_response_schema_hash(partition, kind) for kind in OutcomeRequestKind
        }
        for partition in partitions
    }
    stub = OutcomePartitionManifest(
        canonical_hash=canonical_hash,
        topic_artifact_hash=topic_artifact_hash,
        event_facts_hash=event_facts_hash,
        participant_hash=participant_hash,
        model=APPROVED_TOPIC_MODEL,
        model_digest=APPROVED_TOPIC_MODEL_DIGEST,
        prompt_revision=OUTCOME_PROMPT_REVISION,
        schema_revision=OUTCOME_SCHEMA_REVISION,
        extraction_config_version=OUTCOME_EXTRACTION_CONFIG_VERSION,
        partition_config_version=OUTCOME_PARTITION_CONFIG_VERSION,
        pipeline_revision=pipeline_revision,
        entry_inventory=entry_inventory,
        event_inventory=event_inventory,
        coverage_digest=_hash(coverage_input),
        target_ladder=OUTCOME_TARGET_LADDER,
        target_ladder_revision=OUTCOME_TARGET_LADDER_REVISION,
        target_attempts=target_attempts,
        planning_search_states_used=planning_search_states_used,
        context_target_chars=selected_target_context_chars,
        max_request_chars=max(
            (
                len(partition.context_json(kind))
                for partition in partitions
                for kind in OutcomeRequestKind
            ),
            default=0,
        ),
        event_closure_exception_count=sum(
            partition.event_closure_exception for partition in partitions
        ),
        event_closure_exception_reason_counts=dict(
            sorted(
                Counter(
                    partition.event_closure_exception_reason
                    for partition in partitions
                    if partition.event_closure_exception_reason is not None
                ).items()
            )
        ),
        request_hashes=request_hashes,
        response_schema_hashes=response_schema_hashes,
        partitions=partitions,
        manifest_hash="0" * 64,
    )
    manifest = stub.model_copy(update={"manifest_hash": _hash(stub.hash_input())})
    validate_outcome_partition_manifest(manifest)
    return manifest


def validate_outcome_partition_manifest(
    manifest: OutcomePartitionManifest,
    *,
    canonical_hash: str | None = None,
    topic_artifact_hash: str | None = None,
    event_facts_hash: str | None = None,
    participant_hash: str | None = None,
    topics: tuple[
        tuple[TopicSegment, tuple[CanonicalTranscriptEntry, ...], tuple[OutcomeEventFact, ...]], ...
    ]
    | None = None,
    pipeline_revision: str | None = None,
    allowed_participant_ids: tuple[int, ...] = (),
) -> None:
    if (
        manifest.model != APPROVED_TOPIC_MODEL
        or manifest.model_digest != APPROVED_TOPIC_MODEL_DIGEST
        or manifest.extraction_config_version != OUTCOME_EXTRACTION_CONFIG_VERSION
        or manifest.partition_config_version != OUTCOME_PARTITION_CONFIG_VERSION
        or len(manifest.partitions) > MAX_OUTCOME_PARTITIONS_PER_JOB
        or max(
            Counter(partition.topic_id for partition in manifest.partitions).values(),
            default=0,
        )
        > MAX_OUTCOME_PARTITIONS_PER_TOPIC
    ):
        raise OutcomePartitionError("manifest_tamper")
    if (
        manifest.prompt_revision != OUTCOME_PROMPT_REVISION
        or manifest.schema_revision != OUTCOME_SCHEMA_REVISION
    ):
        raise OutcomePartitionError("manifest_tamper")
    if manifest.manifest_hash != _hash(manifest.hash_input()):
        raise OutcomePartitionError("manifest_tamper")
    for partition in manifest.partitions:
        observed_topic_count = sum(
            candidate.topic_id == partition.topic_id for candidate in manifest.partitions
        )
        if (
            partition.partition_hash != _hash(partition.hash_input())
            or partition.topic_partition_count != observed_topic_count
            or not (
                _fits(partition, manifest.context_target_chars)
                or (
                    partition.event_closure_exception
                    and partition.core_events
                    and max(len(partition.context_json(kind)) for kind in OutcomeRequestKind)
                    > manifest.context_target_chars
                    and _fits(partition, manifest.context_hard_cap_chars)
                )
            )
            or set(manifest.request_hashes.get(partition.partition_id, ()))
            != {kind.value for kind in OutcomeRequestKind}
            or set(manifest.response_schema_hashes.get(partition.partition_id, ()))
            != {kind.value for kind in OutcomeRequestKind}
            or any(
                manifest.request_hashes[partition.partition_id][kind.value]
                != hashlib.sha256(serialize_outcome_request(partition, kind)).hexdigest()
                for kind in OutcomeRequestKind
            )
            or any(
                manifest.response_schema_hashes[partition.partition_id][kind.value]
                != outcome_response_schema_hash(partition, kind)
                for kind in OutcomeRequestKind
            )
        ):
            raise OutcomePartitionError("manifest_tamper")
    observed_max = max(
        (
            len(partition.context_json(kind))
            for partition in manifest.partitions
            for kind in OutcomeRequestKind
        ),
        default=0,
    )
    observed_reasons = dict(
        sorted(
            Counter(
                partition.event_closure_exception_reason
                for partition in manifest.partitions
                if partition.event_closure_exception_reason is not None
            ).items()
        )
    )
    if (
        manifest.max_request_chars != observed_max
        or manifest.event_closure_exception_count
        != sum(partition.event_closure_exception for partition in manifest.partitions)
        or manifest.event_closure_exception_reason_counts != observed_reasons
    ):
        raise OutcomePartitionError("manifest_tamper")
    if topics is None:
        return
    required = (
        canonical_hash,
        topic_artifact_hash,
        event_facts_hash,
        participant_hash,
        pipeline_revision,
    )
    if any(value is None for value in required):
        raise OutcomePartitionError("manifest_current_inputs_missing")
    rebuilt = build_outcome_partition_manifest(
        canonical_hash=str(canonical_hash),
        topic_artifact_hash=str(topic_artifact_hash),
        event_facts_hash=str(event_facts_hash),
        participant_hash=str(participant_hash),
        topics=topics,
        pipeline_revision=str(pipeline_revision),
        allowed_participant_ids=allowed_participant_ids,
    )
    if _canonical_json_bytes(manifest.model_dump(mode="json")) != _canonical_json_bytes(
        rebuilt.model_dump(mode="json")
    ):
        raise OutcomePartitionError("manifest_tamper")


def expected_partition_result_keys(
    manifest: OutcomePartitionManifest,
) -> tuple[tuple[str, OutcomeRequestKind], ...]:
    return tuple(
        sorted(
            (
                (partition.partition_id, kind)
                for partition in manifest.partitions
                for kind in OutcomeRequestKind
            ),
            key=lambda item: (item[0], item[1].value),
        )
    )


def validate_partition_result_keys(
    observed: tuple[tuple[str, OutcomeRequestKind], ...],
    expected: tuple[tuple[str, OutcomeRequestKind], ...],
) -> None:
    if len(observed) != len(set(observed)):
        raise OutcomePartitionError("partition_result_tamper")
    if any(key not in expected for key in observed):
        raise OutcomePartitionError("partition_result_tamper")
    ordered = tuple(sorted(observed, key=lambda item: (item[0], item[1].value)))
    if ordered != expected:
        raise OutcomePartitionError("partition_result_incomplete")
