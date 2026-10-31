"""Small Phase 1 domain value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from lingxu.domain.enums import (
    AgendaFieldOrigin,
    AgendaImportStatus,
    AgendaStatus,
    JobStatus,
    MeetingState,
    MuteLeaseOwnership,
    ResidentAssistantStatus,
    ServiceCapacityMode,
    ServiceCapacityStatus,
    TimerStatus,
    VoteStatus,
)
from lingxu.domain.speaking import AgendaSpeakingPolicy


@dataclass(frozen=True, slots=True)
class ResidentAssistantSessionRecord:
    """Content-free durable lease for one resident voice connection."""

    id: str
    guild_id: int
    channel_id: int
    creator_user_id: int
    status: ResidentAssistantStatus
    next_sequence: int
    created_at: datetime
    updated_at: datetime
    stopped_at: datetime | None


@dataclass(frozen=True, slots=True)
class ResidentAssistantEventRecord:
    """One immutable content-free resident-assistant lifecycle event."""

    id: str
    session_id: str
    sequence_number: int
    event_type: str
    actor_user_id: int | None
    reason: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class ServiceCapacityLeaseRecord:
    """Content-free ownership of the deployment's single service slot."""

    id: str
    guild_id: int
    mode: ServiceCapacityMode
    owner_id: str
    status: ServiceCapacityStatus
    next_sequence: int
    created_at: datetime
    updated_at: datetime
    released_at: datetime | None


@dataclass(frozen=True, slots=True)
class ServiceCapacityEventRecord:
    """One immutable, content-free service-capacity lifecycle fact."""

    id: str
    lease_id: str
    sequence_number: int
    event_type: str
    reason: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class MeetingMuteLeaseRecord:
    """Recoverable ownership record for one member's Discord server mute."""

    id: str
    session_id: str
    agenda_item_id: str
    user_id: int
    original_server_mute: bool
    ownership: MuteLeaseOwnership
    created_at: datetime
    updated_at: datetime
    released_at: datetime | None


@dataclass(frozen=True, slots=True)
class MeetingSessionRecord:
    """Content-safe persisted Meeting Session metadata."""

    id: str
    guild_id: int
    creator_user_id: int
    state: MeetingState
    resume_state: MeetingState | None
    next_sequence: int
    manifest_path: Path
    created_at: datetime
    updated_at: datetime
    publication_channel_id: int | None = None
    setup_channel_id: int | None = None


@dataclass(frozen=True, slots=True)
class MeetingEventRecord:
    """One immutable event in a Meeting Session stream."""

    id: str
    session_id: str
    sequence_number: int
    event_type: str
    actor_user_id: int | None
    source: str
    payload: dict[str, Any]
    evidence_utterance_ids: tuple[str, ...]
    state_before: MeetingState
    state_after: MeetingState
    reversible: bool
    correlation_id: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class MeetingSnapshotRecord:
    """Persisted deterministic state snapshot."""

    id: str
    session_id: str
    sequence_number: int
    state: MeetingState
    payload: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ProcessingJobRecord:
    """Content-safe Worker job lease and checkpoint metadata."""

    id: str
    session_id: str
    job_type: str
    status: JobStatus
    attempt_count: int
    available_at: datetime
    locked_at: datetime | None
    locked_by: str | None
    lock_expires_at: datetime | None
    current_stage: str | None
    last_error_type: str | None
    checkpoint: dict[str, Any]
    idempotency_key: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AudioSegmentRecord:
    """Content-free metadata for one encrypted per-user recording Segment."""

    id: str
    session_id: str
    user_id: int
    segment_number: int
    encrypted_path: Path
    sha256: str
    started_ms: int
    ended_ms: int
    bytes_encrypted: int


@dataclass(frozen=True, slots=True)
class RealtimeUtteranceRecord:
    """Raw and normalised text for one consented per-user utterance."""

    id: str
    session_id: str
    user_id: int
    started_ms: int
    ended_ms: int
    raw_text: str
    normalised_text: str | None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class IntentCandidateRecord:
    """A content-minimal persisted candidate and its confirmation decision."""

    id: str
    session_id: str
    utterance_id: str | None
    intent_type: str
    actor_user_id: int
    source: str
    confidence: float | None
    accepted: bool | None


@dataclass(frozen=True, slots=True)
class ParticipantRecord:
    id: str
    session_id: str
    user_id: int
    present: bool
    display_name: str | None = None
    is_host: bool = False
    joined_at: datetime | None = None
    left_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PronunciationAliasRecord:
    """A Guild-scoped self-managed phrase for TTS name rendering only."""

    guild_id: int
    user_id: int
    alias: str
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AgendaItemRecord:
    id: str
    session_id: str
    position: int
    title: str
    status: AgendaStatus
    duration_minutes: int | None
    requires_vote: bool
    vote_content: str | None = None
    vote_duration_seconds: int | None = None
    language: str = "auto"
    speaking_policy: AgendaSpeakingPolicy = field(default_factory=AgendaSpeakingPolicy.free)


@dataclass(frozen=True, slots=True)
class AgendaImportRecord:
    """Session-scoped import state; source text is never written to operational logs."""

    id: str
    session_id: str
    channel_id: int
    message_id: int
    attachment_id: int
    actor_user_id: int
    content_sha256: str
    source_text: str
    validated_identity_user_ids: tuple[int, ...]
    status: AgendaImportStatus
    policy_version: str
    model_name: str
    model_digest: str
    result: dict[str, Any]
    last_error_type: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AgendaImportItem:
    """A validated candidate that deterministic code may atomically commit."""

    title: str
    title_origin: AgendaFieldOrigin
    source_block_ids: tuple[str, ...]
    duration_minutes: int
    duration_origin: AgendaFieldOrigin
    requires_vote: bool
    vote_origin: AgendaFieldOrigin
    vote_content: str | None
    vote_content_origin: AgendaFieldOrigin | None
    vote_duration_seconds: int | None
    vote_duration_origin: AgendaFieldOrigin | None
    language: str
    language_origin: AgendaFieldOrigin
    confidence: float
    completion_notes: tuple[str, ...] = ()
    speaking_policy: AgendaSpeakingPolicy = field(default_factory=AgendaSpeakingPolicy.free)


@dataclass(frozen=True, slots=True)
class TimerRecord:
    id: str
    session_id: str
    timer_type: str
    deadline_at: datetime
    remaining_ms: int | None
    status: TimerStatus


@dataclass(frozen=True, slots=True)
class VoteRecord:
    id: str
    session_id: str
    title: str
    status: VoteStatus
    options: tuple[dict[str, Any], ...]
    deadline_at: datetime
    opened_at: datetime
    closed_at: datetime | None
    anonymous: bool = False
    agenda_item_id: str | None = None


@dataclass(frozen=True, slots=True)
class VoteRecipientRecord:
    vote_id: str
    user_id: int
    delivery_status: str
    dm_message_id: int | None = None


@dataclass(frozen=True, slots=True)
class TranscriptEntryRecord:
    id: str
    session_id: str
    source_key: str
    user_id: int | None
    raw_text: str
    normalised_text: str | None
    started_ms: int
    ended_ms: int
    revision_json: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    id: str
    session_id: str
    artifact_type: str
    path: Path
    sha256: str
    encrypted: bool
    metadata: dict[str, Any]
