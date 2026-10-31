"""SQLAlchemy 2 mappings for independent Lingxu persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(AsyncAttrs, DeclarativeBase):
    """Declarative base shared by migrations and runtime repositories."""


class GuildSettingsRow(Base):
    __tablename__ = "guild_settings"

    guild_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class GuildWakeupSettingsRow(Base):
    """Optional wake-phrase override scoped to exactly one Discord Guild."""

    __tablename__ = "guild_wakeup_settings"

    guild_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    wake_phrase: Mapped[str] = mapped_column(Text, nullable=False)
    updated_by_user_id: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class GuildMemberConsentRow(Base):
    """Durable per-user consent scoped to exactly one Discord Guild."""

    __tablename__ = "guild_member_consents"

    guild_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    consent_realtime_asr: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    consent_recording: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    consent_minutes: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class GuildMemberPronunciationRow(Base):
    """User-owned TTS pronunciation scoped to one Discord Guild."""

    __tablename__ = "guild_member_pronunciations"

    guild_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    alias: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ResidentAssistantSessionRow(Base):
    """Materialised state derived from one append-only operational event stream."""

    __tablename__ = "resident_assistant_sessions"
    __table_args__ = (UniqueConstraint("guild_id", "active_marker"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    guild_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    channel_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    creator_user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    next_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active_marker: Mapped[int | None] = mapped_column(Integer, nullable=True, default=1)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ResidentAssistantEventRow(Base):
    """Immutable, content-free lifecycle facts for a resident assistant lease."""

    __tablename__ = "resident_assistant_events"
    __table_args__ = (UniqueConstraint("session_id", "sequence_number"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("resident_assistant_sessions.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    actor_user_id: Mapped[int | None] = mapped_column(BigInteger)
    reason: Mapped[str] = mapped_column(String(80), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ServiceCapacityLeaseRow(Base):
    """Materialised ownership of the deployment's single shared slot."""

    __tablename__ = "service_capacity_leases"
    __table_args__ = (UniqueConstraint("slot_name", "active_marker"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    slot_name: Mapped[str] = mapped_column(String(24), nullable=False, default="PRIMARY")
    guild_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    owner_id: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    next_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active_marker: Mapped[int | None] = mapped_column(Integer, nullable=True, default=1)
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ServiceCapacityEventRow(Base):
    """Immutable, content-free lifecycle facts for the shared service slot."""

    __tablename__ = "service_capacity_events"
    __table_args__ = (UniqueConstraint("lease_id", "sequence_number"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    lease_id: Mapped[str] = mapped_column(
        ForeignKey("service_capacity_leases.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    reason: Mapped[str] = mapped_column(String(80), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class MeetingSessionRow(Base):
    __tablename__ = "meeting_sessions"
    __table_args__ = (UniqueConstraint("guild_id", "active_marker"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    guild_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    creator_user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    publication_channel_id: Mapped[int | None] = mapped_column(BigInteger)
    setup_channel_id: Mapped[int | None] = mapped_column(BigInteger)
    state: Mapped[str] = mapped_column(String(40), nullable=False)
    resume_state: Mapped[str | None] = mapped_column(String(40))
    next_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active_marker: Mapped[int | None] = mapped_column(Integer, nullable=True, default=1)
    manifest_path: Mapped[str] = mapped_column(Text, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class MeetingParticipantRow(Base):
    __tablename__ = "meeting_participants"
    __table_args__ = (UniqueConstraint("session_id", "user_id"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    display_name: Mapped[str | None] = mapped_column(Text)
    is_host: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    consent_realtime_asr: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_recording: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_minutes: Mapped[bool] = mapped_column(Boolean, default=False)
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    left_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MeetingAgendaItemRow(Base):
    __tablename__ = "meeting_agenda_items"
    __table_args__ = (UniqueConstraint("session_id", "position"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class AgendaImportRow(Base):
    """Restart-safe local processing state for one Markdown attachment."""

    __tablename__ = "agenda_imports"
    __table_args__ = (
        UniqueConstraint("session_id", "message_id", "attachment_id"),
        Index(
            "uq_agenda_imports_one_processing",
            "session_id",
            unique=True,
            sqlite_where=text("status = 'PROCESSING'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    channel_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    message_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    attachment_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    actor_user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    validated_identity_user_ids_json: Mapped[list[int]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    model_name: Mapped[str] = mapped_column(String(80), nullable=False)
    model_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    last_error_type: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class MeetingEventRow(Base):
    __tablename__ = "meeting_events"
    __table_args__ = (UniqueConstraint("session_id", "sequence_number"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(80), nullable=False)
    actor_user_id: Mapped[int | None] = mapped_column(BigInteger)
    source: Mapped[str] = mapped_column(String(40), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    evidence_utterance_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    state_before: Mapped[str] = mapped_column(String(40), nullable=False)
    state_after: Mapped[str] = mapped_column(String(40), nullable=False)
    reversible: Mapped[bool] = mapped_column(Boolean, default=False)
    correlation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class MeetingStateSnapshotRow(Base):
    __tablename__ = "meeting_state_snapshots"
    __table_args__ = (UniqueConstraint("session_id", "sequence_number"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(String(40), nullable=False)
    snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MeetingTimerRow(Base):
    __tablename__ = "meeting_timers"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    timer_type: Mapped[str] = mapped_column(String(60), nullable=False)
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    remaining_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24), nullable=False)


class MeetingMuteLeaseRow(Base):
    """Materialised ownership for a Discord mute side effect."""

    __tablename__ = "meeting_mute_leases"
    __table_args__ = (UniqueConstraint("session_id", "user_id", "active_marker"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    agenda_item_id: Mapped[str] = mapped_column(String(26), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    original_server_mute: Mapped[bool] = mapped_column(Boolean, nullable=False)
    ownership: Mapped[str] = mapped_column(String(24), nullable=False)
    active_marker: Mapped[int | None] = mapped_column(Integer, nullable=True, default=1)
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class MeetingVoteRow(Base):
    __tablename__ = "meeting_votes"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Agenda rows can only be removed while pending, before a vote exists. Keep this nullable
    # association indexed and repository-controlled so SQLite upgrades do not require a table-copy
    # migration merely to add a late foreign-key constraint.
    agenda_item_id: Mapped[str | None] = mapped_column(String(26), index=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    anonymous: Mapped[bool] = mapped_column(Boolean, default=False)
    options_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MeetingVoteEntryRow(Base):
    __tablename__ = "meeting_vote_entries"
    __table_args__ = (
        UniqueConstraint("vote_id", "user_id"),
        UniqueConstraint("vote_id", "voter_token_hash"),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    vote_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_votes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    voter_token_hash: Mapped[str | None] = mapped_column(String(64))
    option_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MeetingVoteRecipientRow(Base):
    __tablename__ = "meeting_vote_recipients"
    __table_args__ = (
        UniqueConstraint("vote_id", "user_id"),
        UniqueConstraint("dm_message_id"),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    vote_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_votes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    delivery_status: Mapped[str] = mapped_column(String(24), nullable=False, default="PENDING")
    dm_message_id: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class MeetingRealtimeUtteranceRow(Base):
    __tablename__ = "meeting_realtime_utterances"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    started_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    ended_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalised_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MeetingIntentCandidateRow(Base):
    __tablename__ = "meeting_intent_candidates"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    utterance_id: Mapped[str | None] = mapped_column(
        ForeignKey("meeting_realtime_utterances.id", ondelete="SET NULL")
    )
    intent_type: Mapped[str] = mapped_column(String(80), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    confidence: Mapped[float | None] = mapped_column(Float)
    accepted: Mapped[bool | None] = mapped_column(Boolean)


class MeetingPlaybackJobRow(Base):
    __tablename__ = "meeting_playback_jobs"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AudioSegmentRow(Base):
    __tablename__ = "audio_segments"
    __table_args__ = (UniqueConstraint("session_id", "user_id", "segment_number"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    segment_number: Mapped[int] = mapped_column(Integer, nullable=False)
    encrypted_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    started_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    ended_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    bytes_encrypted: Mapped[int] = mapped_column(BigInteger, nullable=False)


class ProcessingJobRow(Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (
        UniqueConstraint("session_id", "job_type", "idempotency_key"),
        Index("ix_processing_jobs_claim", "status", "available_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_type: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    locked_by: Mapped[str | None] = mapped_column(String(128))
    lock_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(String(160))
    current_stage: Mapped[str | None] = mapped_column(String(80))
    checkpoint_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class TranscriptEntryRow(Base):
    __tablename__ = "transcript_entries"
    __table_args__ = (UniqueConstraint("session_id", "source_key"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_key: Mapped[str] = mapped_column(String(96), nullable=False)
    user_id: Mapped[int | None] = mapped_column(BigInteger)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalised_text: Mapped[str | None] = mapped_column(Text)
    revision_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    started_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    ended_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)


class ArtifactRow(Base):
    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("session_id", "artifact_type", "path"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    artifact_type: Mapped[str] = mapped_column(String(60), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    encrypted: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ScenarioDefinitionRow(Base):
    __tablename__ = "scenario_definitions"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    active_version_id: Mapped[str | None] = mapped_column(String(26))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ScenarioVersionRow(Base):
    __tablename__ = "scenario_versions"
    __table_args__ = (UniqueConstraint("scenario_id", "version"),)

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    scenario_id: Mapped[str] = mapped_column(
        ForeignKey("scenario_definitions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="DRAFT")
    schema_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    compiled_blob: Mapped[bytes | None] = mapped_column(LargeBinary)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
