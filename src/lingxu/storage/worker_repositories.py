"""Idempotent transcript and Artifact repositories for the independent Worker."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import delete, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lingxu.domain.models import ArtifactRecord, TranscriptEntryRecord
from lingxu.storage.session_files import SessionFiles
from lingxu.storage.tables import ArtifactRow, MeetingSessionRow, TranscriptEntryRow
from lingxu.utils import new_ulid


class PublicationReconciliationRequired(RuntimeError):
    """Content-free terminal failure requiring explicit operator reconciliation."""

    _ALLOWED = frozenset(
        {
            "publication_prepared_unresolved",
            "publication_record_invalid",
            "publication_concurrent_preparation",
        }
    )

    def __init__(self, reason: str) -> None:
        if reason not in self._ALLOWED:
            raise ValueError("publication reconciliation reason is not allowlisted")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class PreparationResult:
    record: ArtifactRecord
    created_by_this_call: bool


_PUBLICATION_CONTRACTS = {
    "task10-four-file-v1": (
        ("MINUTES_MARKDOWN", "minutes.md"),
        ("MINUTES_JSON", "minutes.json"),
        ("TRANSCRIPT_TEXT", "transcript.txt"),
        ("EVIDENCE_JSON", "evidence.json"),
    ),
    "task11-one-file-v1": (("MINUTES_MARKDOWN", "minutes.md"),),
}
_CURRENT_PUBLICATION_CONTRACT_REVISION = "task11-one-file-v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ULID = re.compile(r"[0-9A-HJKMNP-TV-Z]{26}\Z")
_PREPARED_KEYS = frozenset(
    {
        "intent_schema_version",
        "intent_id",
        "state",
        "session_id",
        "guild_id",
        "requested_channel_id",
        "pipeline_version",
        "outcome_pipeline_revision",
        "speech_act_policy_revision",
        "publication_contract_revision",
        "intent_nonce",
        "attachments",
        "artifact_set_sha256",
        "intent_payload_sha256",
    }
)
_FINALIZED_KEYS = _PREPARED_KEYS | {
    "selected_channel_id",
    "message_id",
    "receipt_path",
    "receipt_payload_sha256",
}
_RECEIPT_KEYS = frozenset(
    {
        "receipt_schema_version",
        "intent_id",
        "state",
        "session_id",
        "guild_id",
        "channel_id",
        "message_id",
        "pipeline_version",
        "outcome_pipeline_revision",
        "speech_act_policy_revision",
        "publication_contract_revision",
        "intent_path",
        "intent_nonce",
        "attachments",
        "artifact_set_sha256",
        "receipt_payload_sha256",
    }
)


def publication_payload_hash(value: Any, self_hash_field: str | None = None) -> str:
    payload = (
        {key: item for key, item in value.items() if key != self_hash_field}
        if self_hash_field is not None and isinstance(value, dict)
        else value
    )
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _valid_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _streaming_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publication_attachments_for_contract(
    publication_contract_revision: object,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(publication_contract_revision, str):
        raise PublicationReconciliationRequired("publication_record_invalid")
    contract = _PUBLICATION_CONTRACTS.get(publication_contract_revision)
    if contract is None:
        raise PublicationReconciliationRequired("publication_record_invalid")
    return contract


def validate_publication_attachments(
    value: object,
    *,
    publication_contract_revision: object,
) -> tuple[dict[str, Any], ...]:
    expected = _publication_attachments_for_contract(publication_contract_revision)
    if not isinstance(value, (tuple, list)) or len(value) != len(expected):
        raise PublicationReconciliationRequired("publication_record_invalid")
    result: list[dict[str, Any]] = []
    for raw, (artifact_type, filename) in zip(value, expected, strict=True):
        if not isinstance(raw, dict) or set(raw) != {
            "artifact_type",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise PublicationReconciliationRequired("publication_record_invalid")
        path = raw.get("path")
        digest = raw.get("sha256")
        size = raw.get("size_bytes")
        if (
            raw.get("artifact_type") != artifact_type
            or not isinstance(path, str)
            or not Path(path).is_absolute()
            or str(Path(path).resolve()) != path
            or Path(path).name != filename
            or not isinstance(digest, str)
            or _HEX64.fullmatch(digest) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise PublicationReconciliationRequired("publication_record_invalid")
        result.append(dict(raw))
    return tuple(result)


def validate_prepared_publication(
    record: ArtifactRecord,
    *,
    session_id: str,
    guild_id: int,
    requested_channel_id: int | None,
) -> tuple[dict[str, Any], ...]:
    metadata = record.metadata
    contract_revision = metadata.get("publication_contract_revision")
    attachments = validate_publication_attachments(
        metadata.get("attachments"),
        publication_contract_revision=contract_revision,
    )
    expected_path = Path(f"discord-publication-intent-{session_id}")
    if (
        record.artifact_type != "DISCORD_PUBLICATION_INTENT"
        or record.session_id != session_id
        or record.encrypted
        or set(metadata) != _PREPARED_KEYS
        or metadata.get("intent_schema_version") != 2
        or metadata.get("intent_id") != record.id
        or _ULID.fullmatch(record.id) is None
        or metadata.get("state") != "PREPARED"
        or metadata.get("session_id") != session_id
        or metadata.get("guild_id") != guild_id
        or metadata.get("requested_channel_id") != requested_channel_id
        or metadata.get("pipeline_version") not in {3, 4}
        or not isinstance(metadata.get("outcome_pipeline_revision"), str)
        or _HEX64.fullmatch(metadata["outcome_pipeline_revision"]) is None
        or not isinstance(metadata.get("speech_act_policy_revision"), str)
        or _HEX64.fullmatch(metadata["speech_act_policy_revision"]) is None
        or contract_revision not in _PUBLICATION_CONTRACTS
        or not isinstance(metadata.get("intent_nonce"), str)
        or _ULID.fullmatch(metadata["intent_nonce"]) is None
        or metadata.get("artifact_set_sha256") != publication_payload_hash(list(attachments))
        or record.path != expected_path
        or record.sha256 != publication_payload_hash(metadata, "intent_payload_sha256")
        or metadata.get("intent_payload_sha256") != record.sha256
    ):
        raise PublicationReconciliationRequired("publication_record_invalid")
    return attachments


def validate_finalized_publication_pair(
    intent: ArtifactRecord,
    receipt: ArtifactRecord,
    *,
    session_id: str,
    guild_id: int,
    requested_channel_id: int | None,
) -> None:
    im, rm = intent.metadata, receipt.metadata
    contract_revision = im.get("publication_contract_revision")
    attachments = validate_publication_attachments(
        im.get("attachments"),
        publication_contract_revision=contract_revision,
    )
    channel_id, message_id = rm.get("channel_id"), rm.get("message_id")
    expected_intent_path = Path(f"discord-publication-intent-{session_id}")
    expected_receipt_path = Path(f"discord-{channel_id}-{message_id}")
    receipt_hash = publication_payload_hash(rm, "receipt_payload_sha256")
    if (
        intent.artifact_type != "DISCORD_PUBLICATION_INTENT"
        or receipt.artifact_type != "DISCORD_PUBLICATION"
        or intent.session_id != session_id
        or receipt.session_id != session_id
        or intent.encrypted
        or receipt.encrypted
        or set(im) != _FINALIZED_KEYS
        or set(rm) != _RECEIPT_KEYS
        or im.get("intent_schema_version") != 2
        or rm.get("receipt_schema_version") != 2
        or im.get("intent_id") != intent.id
        or rm.get("intent_id") != intent.id
        or _ULID.fullmatch(intent.id) is None
        or im.get("state") != "FINALIZED"
        or rm.get("state") != "FINALIZED"
        or im.get("session_id") != session_id
        or rm.get("session_id") != session_id
        or im.get("guild_id") != guild_id
        or rm.get("guild_id") != guild_id
        or im.get("requested_channel_id") != requested_channel_id
        or im.get("pipeline_version") not in {3, 4}
        or rm.get("pipeline_version") != im.get("pipeline_version")
        or not isinstance(im.get("outcome_pipeline_revision"), str)
        or _HEX64.fullmatch(im["outcome_pipeline_revision"]) is None
        or not isinstance(im.get("speech_act_policy_revision"), str)
        or _HEX64.fullmatch(im["speech_act_policy_revision"]) is None
        or im.get("outcome_pipeline_revision") != rm.get("outcome_pipeline_revision")
        or im.get("speech_act_policy_revision") != rm.get("speech_act_policy_revision")
        or contract_revision not in _PUBLICATION_CONTRACTS
        or rm.get("publication_contract_revision") != contract_revision
        or im.get("intent_nonce") != rm.get("intent_nonce")
        or not isinstance(im.get("intent_nonce"), str)
        or _ULID.fullmatch(im["intent_nonce"]) is None
        or im.get("attachments") != list(attachments)
        or rm.get("attachments") != list(attachments)
        or im.get("artifact_set_sha256") != publication_payload_hash(list(attachments))
        or rm.get("artifact_set_sha256") != im.get("artifact_set_sha256")
        or not _valid_positive_int(channel_id)
        or (requested_channel_id is not None and channel_id != requested_channel_id)
        or not isinstance(message_id, str)
        or not message_id.isdecimal()
        or int(message_id) <= 0
        or im.get("selected_channel_id") != channel_id
        or im.get("message_id") != message_id
        or intent.path != expected_intent_path
        or rm.get("intent_path") != str(expected_intent_path)
        or receipt.path != expected_receipt_path
        or im.get("receipt_path") != str(expected_receipt_path)
        or receipt.sha256 != hashlib.sha256(message_id.encode()).hexdigest()
        or rm.get("receipt_payload_sha256") != receipt_hash
        or im.get("receipt_payload_sha256") != receipt_hash
        or intent.sha256 != publication_payload_hash(im, "intent_payload_sha256")
        or im.get("intent_payload_sha256") != intent.sha256
    ):
        raise PublicationReconciliationRequired("publication_record_invalid")


def _transcript_record(row: TranscriptEntryRow) -> TranscriptEntryRecord:
    return TranscriptEntryRecord(
        id=row.id,
        session_id=row.session_id,
        source_key=row.source_key,
        user_id=row.user_id,
        raw_text=row.raw_text,
        normalised_text=row.normalised_text,
        started_ms=row.started_ms,
        ended_ms=row.ended_ms,
        revision_json=tuple(dict(revision) for revision in row.revision_json),
    )


def _artifact_record(row: ArtifactRow) -> ArtifactRecord:
    return ArtifactRecord(
        id=row.id,
        session_id=row.session_id,
        artifact_type=row.artifact_type,
        path=Path(row.path),
        sha256=row.sha256,
        encrypted=row.encrypted,
        metadata=dict(row.metadata_json),
    )


class TranscriptRepository:
    """Upsert one transcript row per realtime utterance or recorded Segment."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def upsert(
        self,
        *,
        session_id: str,
        source_key: str,
        user_id: int | None,
        raw_text: str,
        normalised_text: str | None,
        started_ms: int,
        ended_ms: int,
    ) -> TranscriptEntryRecord:
        values: dict[str, Any] = {
            "id": new_ulid(),
            "session_id": session_id,
            "source_key": source_key,
            "user_id": user_id,
            "raw_text": raw_text,
            "normalised_text": normalised_text,
            "revision_json": [],
            "started_ms": started_ms,
            "ended_ms": ended_ms,
        }
        async with self._sessions() as db, db.begin():
            await db.execute(
                sqlite_insert(TranscriptEntryRow)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=("session_id", "source_key"),
                    set_={
                        "user_id": user_id,
                        "raw_text": raw_text,
                        "normalised_text": normalised_text,
                        "started_ms": started_ms,
                        "ended_ms": ended_ms,
                    },
                )
            )
            row = await db.scalar(
                select(TranscriptEntryRow).where(
                    TranscriptEntryRow.session_id == session_id,
                    TranscriptEntryRow.source_key == source_key,
                )
            )
            if row is None:
                raise RuntimeError("transcript upsert returned no row")
            return _transcript_record(row)

    async def list(self, session_id: str) -> tuple[TranscriptEntryRecord, ...]:
        async with self._sessions() as db:
            rows = await db.scalars(
                select(TranscriptEntryRow)
                .where(TranscriptEntryRow.session_id == session_id)
                .order_by(
                    TranscriptEntryRow.started_ms,
                    TranscriptEntryRow.user_id,
                    TranscriptEntryRow.id,
                )
            )
            return tuple(_transcript_record(row) for row in rows)

    async def delete_blank(self, session_id: str) -> int:
        """Remove ASR rows with no usable speech content before artifact rendering."""

        async with self._sessions() as db, db.begin():
            removed = list(
                await db.scalars(
                    delete(TranscriptEntryRow)
                    .where(
                        TranscriptEntryRow.session_id == session_id,
                        or_(
                            TranscriptEntryRow.raw_text == "",
                            TranscriptEntryRow.raw_text.is_(None),
                        ),
                    )
                    .returning(TranscriptEntryRow.id)
                )
            )
            return len(removed)

    async def delete_ids(self, *, session_id: str, entry_ids: tuple[str, ...]) -> int:
        """Delete exact transcript rows without crossing the Session boundary."""

        if not entry_ids:
            return 0
        async with self._sessions() as db, db.begin():
            removed = list(
                await db.scalars(
                    delete(TranscriptEntryRow)
                    .where(
                        TranscriptEntryRow.session_id == session_id,
                        TranscriptEntryRow.id.in_(entry_ids),
                    )
                    .returning(TranscriptEntryRow.id)
                )
            )
            return len(removed)


class ArtifactRepository:
    """Register content hashes and external publications without duplication."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        session_files: SessionFiles | None = None,
    ) -> None:
        self._sessions = sessions
        self._session_files = session_files

    def bind_session_files(self, files: SessionFiles) -> None:
        """Bind the deployment-owned Session resolver exactly once for publication checks."""

        if self._session_files is not None and self._session_files.root != files.root:
            raise ValueError("Artifact repository Session root is already bound")
        self._session_files = files

    async def add(
        self,
        *,
        session_id: str,
        artifact_type: str,
        path: Path,
        sha256: str,
        encrypted: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        values = {
            "id": new_ulid(),
            "session_id": session_id,
            "artifact_type": artifact_type,
            "path": str(path),
            "sha256": sha256,
            "encrypted": encrypted,
            "metadata_json": dict(metadata or {}),
        }
        async with self._sessions() as db, db.begin():
            await db.execute(
                sqlite_insert(ArtifactRow)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=("session_id", "artifact_type", "path"),
                    set_={
                        "path": str(path),
                        "sha256": sha256,
                        "encrypted": encrypted,
                        "metadata_json": dict(metadata or {}),
                    },
                )
            )
            row = await db.scalar(
                select(ArtifactRow).where(
                    ArtifactRow.session_id == session_id,
                    ArtifactRow.artifact_type == artifact_type,
                    ArtifactRow.path == str(path),
                )
            )
            if row is None:
                raise RuntimeError("artifact upsert returned no row")
            return _artifact_record(row)

    async def add_file(
        self,
        *,
        session_id: str,
        artifact_type: str,
        path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        payload = await asyncio.to_thread(path.read_bytes)
        digest = hashlib.sha256(payload).hexdigest()
        file_metadata = dict(metadata or {})
        file_metadata["size_bytes"] = len(payload)
        return await self.add(
            session_id=session_id,
            artifact_type=artifact_type,
            path=path,
            sha256=digest,
            metadata=file_metadata,
        )

    async def first_of_type(
        self,
        *,
        session_id: str,
        artifact_type: str,
    ) -> ArtifactRecord | None:
        async with self._sessions() as db:
            row = await db.scalar(
                select(ArtifactRow)
                .where(
                    ArtifactRow.session_id == session_id,
                    ArtifactRow.artifact_type == artifact_type,
                )
                .order_by(ArtifactRow.id)
                .limit(1)
            )
            return _artifact_record(row) if row else None

    async def all_of_type(
        self,
        *,
        session_id: str,
        artifact_type: str,
    ) -> tuple[ArtifactRecord, ...]:
        async with self._sessions() as db:
            rows = await db.scalars(
                select(ArtifactRow)
                .where(
                    ArtifactRow.session_id == session_id,
                    ArtifactRow.artifact_type == artifact_type,
                )
                .order_by(ArtifactRow.id)
            )
            return tuple(_artifact_record(row) for row in rows)

    async def prepare_discord_publication(
        self,
        *,
        session_id: str,
        pipeline_version: int,
        outcome_pipeline_revision: str,
        speech_act_policy_revision: str,
        publication_contract_revision: str,
        attachments: tuple[dict[str, Any], ...],
    ) -> PreparationResult:
        """Insert a non-overwriting PREPARED intent under an immediate SQLite writer lock."""

        checked_attachments = validate_publication_attachments(
            attachments,
            publication_contract_revision=publication_contract_revision,
        )
        if (
            pipeline_version != 4
            or _HEX64.fullmatch(outcome_pipeline_revision) is None
            or _HEX64.fullmatch(speech_act_policy_revision) is None
            or publication_contract_revision != _CURRENT_PUBLICATION_CONTRACT_REVISION
        ):
            raise PublicationReconciliationRequired("publication_record_invalid")
        async with self._sessions() as db:
            connection = await db.connection()
            await connection.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                session = await db.get(MeetingSessionRow, session_id)
                if session is None:
                    raise PublicationReconciliationRequired("publication_record_invalid")
                existing_intents = tuple(
                    await db.scalars(
                        select(ArtifactRow).where(
                            ArtifactRow.session_id == session_id,
                            ArtifactRow.artifact_type == "DISCORD_PUBLICATION_INTENT",
                        )
                    )
                )
                existing_receipts = tuple(
                    await db.scalars(
                        select(ArtifactRow).where(
                            ArtifactRow.session_id == session_id,
                            ArtifactRow.artifact_type == "DISCORD_PUBLICATION",
                        )
                    )
                )
                if existing_intents or existing_receipts:
                    raise PublicationReconciliationRequired(
                        "publication_prepared_unresolved"
                        if len(existing_intents) == 1 and not existing_receipts
                        else "publication_record_invalid"
                    )
                intent_id = new_ulid()
                path = Path(f"discord-publication-intent-{session_id}")
                metadata: dict[str, Any] = {
                    "intent_schema_version": 2,
                    "intent_id": intent_id,
                    "state": "PREPARED",
                    "session_id": session_id,
                    "guild_id": session.guild_id,
                    "requested_channel_id": session.publication_channel_id,
                    "pipeline_version": pipeline_version,
                    "outcome_pipeline_revision": outcome_pipeline_revision,
                    "speech_act_policy_revision": speech_act_policy_revision,
                    "publication_contract_revision": publication_contract_revision,
                    "intent_nonce": new_ulid(),
                    "attachments": list(checked_attachments),
                    "artifact_set_sha256": publication_payload_hash(list(checked_attachments)),
                }
                metadata["intent_payload_sha256"] = publication_payload_hash(
                    metadata, "intent_payload_sha256"
                )
                sha256 = metadata["intent_payload_sha256"]
                await self._validate_registered_publication_attachments(
                    db,
                    session_id=session_id,
                    pipeline_version=pipeline_version,
                    attachments=checked_attachments,
                    outcome_pipeline_revision=outcome_pipeline_revision,
                    speech_act_policy_revision=speech_act_policy_revision,
                    publication_contract_revision=publication_contract_revision,
                )
                inserted = await db.execute(
                    sqlite_insert(ArtifactRow)
                    .values(
                        id=intent_id,
                        session_id=session_id,
                        artifact_type="DISCORD_PUBLICATION_INTENT",
                        path=str(path),
                        sha256=sha256,
                        encrypted=False,
                        metadata_json=dict(metadata),
                    )
                    .on_conflict_do_nothing(index_elements=("session_id", "artifact_type", "path"))
                    .returning(ArtifactRow.id)
                )
                inserted_id = inserted.scalar_one_or_none()
                if inserted_id is None:
                    raise PublicationReconciliationRequired("publication_concurrent_preparation")
                row = await db.get(ArtifactRow, inserted_id)
                if (
                    row is None
                    or row.path != str(path)
                    or row.sha256 != sha256
                    or row.encrypted
                    or row.metadata_json != metadata
                ):
                    raise PublicationReconciliationRequired("publication_record_invalid")
                record = _artifact_record(row)
                validate_prepared_publication(
                    record,
                    session_id=session_id,
                    guild_id=session.guild_id,
                    requested_channel_id=session.publication_channel_id,
                )
                await db.commit()
                return PreparationResult(record, True)
            except Exception:
                await db.rollback()
                raise

    async def finalize_discord_publication(
        self,
        *,
        session_id: str,
        selected_channel_id: int,
        message_id: str,
    ) -> tuple[ArtifactRecord, ArtifactRecord]:
        """Atomically finalize one intent and insert its cross-bound receipt."""

        async with self._sessions() as db:
            connection = await db.connection()
            await connection.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                session = await db.get(MeetingSessionRow, session_id)
                if session is None:
                    raise PublicationReconciliationRequired("publication_record_invalid")
                intents = tuple(
                    await db.scalars(
                        select(ArtifactRow).where(
                            ArtifactRow.session_id == session_id,
                            ArtifactRow.artifact_type == "DISCORD_PUBLICATION_INTENT",
                        )
                    )
                )
                receipts = tuple(
                    await db.scalars(
                        select(ArtifactRow).where(
                            ArtifactRow.session_id == session_id,
                            ArtifactRow.artifact_type == "DISCORD_PUBLICATION",
                        )
                    )
                )
                if len(intents) != 1 or receipts:
                    raise PublicationReconciliationRequired("publication_record_invalid")
                intent = intents[0]
                prepared = _artifact_record(intent)
                attachments = validate_prepared_publication(
                    prepared,
                    session_id=session_id,
                    guild_id=session.guild_id,
                    requested_channel_id=session.publication_channel_id,
                )
                await self._validate_registered_publication_attachments(
                    db,
                    session_id=session_id,
                    pipeline_version=int(prepared.metadata["pipeline_version"]),
                    attachments=attachments,
                    outcome_pipeline_revision=prepared.metadata["outcome_pipeline_revision"],
                    speech_act_policy_revision=prepared.metadata["speech_act_policy_revision"],
                    publication_contract_revision=prepared.metadata[
                        "publication_contract_revision"
                    ],
                )
                if (
                    not _valid_positive_int(selected_channel_id)
                    or (
                        session.publication_channel_id is not None
                        and selected_channel_id != session.publication_channel_id
                    )
                    or not isinstance(message_id, str)
                    or not message_id.isdecimal()
                    or int(message_id) <= 0
                ):
                    raise PublicationReconciliationRequired("publication_record_invalid")
                prepared_sha256 = intent.sha256
                intent_path = Path(intent.path)
                receipt_path = Path(f"discord-{selected_channel_id}-{message_id}")
                receipt_metadata: dict[str, Any] = {
                    "receipt_schema_version": 2,
                    "intent_id": intent.id,
                    "state": "FINALIZED",
                    "session_id": session_id,
                    "guild_id": session.guild_id,
                    "channel_id": selected_channel_id,
                    "message_id": message_id,
                    "pipeline_version": prepared.metadata["pipeline_version"],
                    "outcome_pipeline_revision": prepared.metadata["outcome_pipeline_revision"],
                    "speech_act_policy_revision": prepared.metadata["speech_act_policy_revision"],
                    "publication_contract_revision": prepared.metadata[
                        "publication_contract_revision"
                    ],
                    "intent_path": str(intent_path),
                    "intent_nonce": prepared.metadata["intent_nonce"],
                    "attachments": list(attachments),
                    "artifact_set_sha256": prepared.metadata["artifact_set_sha256"],
                }
                receipt_metadata["receipt_payload_sha256"] = publication_payload_hash(
                    receipt_metadata, "receipt_payload_sha256"
                )
                receipt_sha256 = hashlib.sha256(message_id.encode()).hexdigest()
                finalized_metadata = {
                    **prepared.metadata,
                    "state": "FINALIZED",
                    "selected_channel_id": selected_channel_id,
                    "message_id": message_id,
                    "receipt_path": str(receipt_path),
                    "receipt_payload_sha256": receipt_metadata["receipt_payload_sha256"],
                }
                finalized_metadata["intent_payload_sha256"] = publication_payload_hash(
                    finalized_metadata, "intent_payload_sha256"
                )
                finalized_sha256 = finalized_metadata["intent_payload_sha256"]
                result = await db.execute(
                    update(ArtifactRow)
                    .where(
                        ArtifactRow.id == intent.id,
                        ArtifactRow.sha256 == prepared_sha256,
                    )
                    .values(
                        sha256=finalized_sha256,
                        metadata_json=dict(finalized_metadata),
                    )
                    .returning(ArtifactRow.id)
                )
                if result.scalar_one_or_none() != intent.id:
                    raise PublicationReconciliationRequired("publication_record_invalid")
                receipt_id = new_ulid()
                inserted = await db.execute(
                    sqlite_insert(ArtifactRow)
                    .values(
                        id=receipt_id,
                        session_id=session_id,
                        artifact_type="DISCORD_PUBLICATION",
                        path=str(receipt_path),
                        sha256=receipt_sha256,
                        encrypted=False,
                        metadata_json=dict(receipt_metadata),
                    )
                    .on_conflict_do_nothing(index_elements=("session_id", "artifact_type", "path"))
                    .returning(ArtifactRow.id)
                )
                if inserted.scalar_one_or_none() != receipt_id:
                    raise PublicationReconciliationRequired("publication_record_invalid")
                await db.flush()
                finalized = await db.get(ArtifactRow, intent.id)
                receipt = await db.get(ArtifactRow, receipt_id)
                if finalized is None or receipt is None:
                    raise PublicationReconciliationRequired("publication_record_invalid")
                finalized_record = _artifact_record(finalized)
                receipt_record = _artifact_record(receipt)
                validate_finalized_publication_pair(
                    finalized_record,
                    receipt_record,
                    session_id=session_id,
                    guild_id=session.guild_id,
                    requested_channel_id=session.publication_channel_id,
                )
                await db.commit()
                return finalized_record, receipt_record
            except Exception:
                await db.rollback()
                raise

    async def _validate_registered_publication_attachments(
        self,
        db: AsyncSession,
        *,
        session_id: str,
        pipeline_version: int,
        attachments: tuple[dict[str, Any], ...],
        outcome_pipeline_revision: str,
        speech_act_policy_revision: str,
        publication_contract_revision: str,
    ) -> None:
        if self._session_files is None:
            raise PublicationReconciliationRequired("publication_record_invalid")
        if pipeline_version not in {3, 4}:
            raise PublicationReconciliationRequired("publication_record_invalid")
        contract = _publication_attachments_for_contract(publication_contract_revision)
        renderer_metadata: dict[str, str] = {}
        minutes_json_path = self._session_files.resolve(
            session_id,
            Path("meeting") / "minutes.json",
        )
        try:
            minutes_payload = json.loads(minutes_json_path.read_text(encoding="utf-8"))
            provenance = minutes_payload.get("provenance")
            if isinstance(provenance, dict):
                revision = provenance.get("minutes_renderer_revision")
                input_digest = provenance.get("renderer_input_sha256")
                if (
                    isinstance(revision, str)
                    and _HEX64.fullmatch(revision) is not None
                    and isinstance(input_digest, str)
                    and _HEX64.fullmatch(input_digest) is not None
                ):
                    renderer_metadata = {
                        "minutes_renderer_revision": revision,
                        "renderer_input_sha256": input_digest,
                    }
        except (OSError, UnicodeError, json.JSONDecodeError):
            renderer_metadata = {}
        if publication_contract_revision == _CURRENT_PUBLICATION_CONTRACT_REVISION and not (
            renderer_metadata
        ):
            raise PublicationReconciliationRequired("publication_record_invalid")
        expected_paths = tuple(
            self._session_files.resolve(session_id, Path("meeting") / filename)
            for _, filename in contract
        )
        for item, expected_path in zip(attachments, expected_paths, strict=True):
            row = await db.scalar(
                select(ArtifactRow).where(
                    ArtifactRow.session_id == session_id,
                    ArtifactRow.artifact_type == item["artifact_type"],
                    ArtifactRow.path == item["path"],
                )
            )
            expected_metadata: dict[str, Any] = {
                "pipeline_version": pipeline_version,
                "outcome_pipeline_revision": outcome_pipeline_revision,
                "size_bytes": item["size_bytes"],
            }
            if item["artifact_type"] != "TRANSCRIPT_TEXT":
                expected_metadata["speech_act_policy_revision"] = speech_act_policy_revision
            if item["artifact_type"] in {"MINUTES_JSON", "MINUTES_MARKDOWN"} and (
                renderer_metadata
            ):
                expected_metadata.update(renderer_metadata)
            try:
                file_stat = await asyncio.to_thread(expected_path.lstat)
                digest = (
                    await asyncio.to_thread(_streaming_sha256, expected_path)
                    if stat.S_ISREG(file_stat.st_mode)
                    else None
                )
            except OSError:
                file_stat = None
                digest = None
            if (
                row is None
                or item["path"] != str(expected_path)
                or row.path != str(expected_path)
                or row.encrypted
                or row.sha256 != item["sha256"]
                or row.metadata_json != expected_metadata
                or file_stat is None
                or not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_size != item["size_bytes"]
                or digest != item["sha256"]
            ):
                raise PublicationReconciliationRequired("publication_record_invalid")

    async def validate_discord_publication_files(
        self,
        *,
        session_id: str,
        pipeline_version: int = 3,
        attachments: tuple[dict[str, Any], ...],
        outcome_pipeline_revision: str,
        speech_act_policy_revision: str,
        publication_contract_revision: str,
    ) -> None:
        """Revalidate a stored manifest against the deployment-owned Session root."""

        checked = validate_publication_attachments(
            attachments,
            publication_contract_revision=publication_contract_revision,
        )
        async with self._sessions() as db:
            await self._validate_registered_publication_attachments(
                db,
                session_id=session_id,
                pipeline_version=pipeline_version,
                attachments=checked,
                outcome_pipeline_revision=outcome_pipeline_revision,
                speech_act_policy_revision=speech_act_policy_revision,
                publication_contract_revision=publication_contract_revision,
            )

    async def at_path(
        self,
        *,
        session_id: str,
        artifact_type: str,
        path: Path,
    ) -> ArtifactRecord | None:
        """Return the exact artifact identity rather than an older same-type record."""

        async with self._sessions() as db:
            row = await db.scalar(
                select(ArtifactRow).where(
                    ArtifactRow.session_id == session_id,
                    ArtifactRow.artifact_type == artifact_type,
                    ArtifactRow.path == str(path),
                )
            )
            return _artifact_record(row) if row else None

    async def list(self, session_id: str) -> tuple[ArtifactRecord, ...]:
        async with self._sessions() as db:
            rows = await db.scalars(
                select(ArtifactRow)
                .where(ArtifactRow.session_id == session_id)
                .order_by(ArtifactRow.artifact_type, ArtifactRow.path)
            )
            return tuple(_artifact_record(row) for row in rows)
