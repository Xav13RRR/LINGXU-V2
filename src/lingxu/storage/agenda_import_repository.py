# ruff: noqa: RUF001
"""Restart-safe persistence and atomic commit for Markdown agenda imports."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lingxu.domain.enums import AgendaImportStatus, AgendaStatus, MeetingEventType, MeetingState
from lingxu.domain.errors import AgendaImportError
from lingxu.domain.models import AgendaImportItem, AgendaImportRecord, AgendaItemRecord
from lingxu.storage.component_repositories import _agenda_record
from lingxu.storage.repositories import _session_row, append_meeting_event_row, utc_now
from lingxu.storage.tables import AgendaImportRow, MeetingAgendaItemRow
from lingxu.utils import new_ulid


def _import_record(row: AgendaImportRow) -> AgendaImportRecord:
    created_at = row.created_at
    updated_at = row.updated_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=utc_now().tzinfo)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=utc_now().tzinfo)
    return AgendaImportRecord(
        id=row.id,
        session_id=row.session_id,
        channel_id=row.channel_id,
        message_id=row.message_id,
        attachment_id=row.attachment_id,
        actor_user_id=row.actor_user_id,
        content_sha256=row.content_sha256,
        source_text=row.source_text,
        validated_identity_user_ids=tuple(row.validated_identity_user_ids_json),
        status=AgendaImportStatus(row.status),
        policy_version=row.policy_version,
        model_name=row.model_name,
        model_digest=row.model_digest,
        result=dict(row.result_json),
        last_error_type=row.last_error_type,
        created_at=created_at,
        updated_at=updated_at,
    )


def _item_payload(import_id: str, item: AgendaImportItem) -> dict[str, object]:
    return {
        "duration_minutes": item.duration_minutes,
        "requires_vote": item.requires_vote,
        "vote_content": item.vote_content,
        "vote_duration_seconds": item.vote_duration_seconds,
        "language": item.language,
        "speaking_policy": item.speaking_policy.to_payload(),
        "import_provenance": {
            "import_id": import_id,
            "source_block_ids": list(item.source_block_ids),
            "title_origin": item.title_origin.value,
            "duration_origin": item.duration_origin.value,
            "vote_origin": item.vote_origin.value,
            "vote_content_origin": (
                item.vote_content_origin.value if item.vote_content_origin is not None else None
            ),
            "vote_duration_origin": (
                item.vote_duration_origin.value if item.vote_duration_origin is not None else None
            ),
            "language_origin": item.language_origin.value,
            "confidence": item.confidence,
            "completion_notes": list(item.completion_notes),
        },
    }


class AgendaImportRepository:
    """Persist import work and create all final agenda rows in one transaction."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create_or_get(
        self,
        *,
        session_id: str,
        channel_id: int,
        message_id: int,
        attachment_id: int,
        actor_user_id: int,
        content_sha256: str,
        source_text: str,
        validated_identity_user_ids: tuple[int, ...],
        policy_version: str,
        model_name: str,
        model_digest: str,
    ) -> tuple[AgendaImportRecord, bool]:
        async with self._sessions() as db, db.begin():
            session = await _session_row(db, session_id)
            if session.creator_user_id != actor_user_id:
                raise AgendaImportError("只有会议主持人可以导入议程文档。")
            expected_channel = session.setup_channel_id or session.publication_channel_id
            if expected_channel is None or expected_channel != channel_id:
                raise AgendaImportError("请在创建本次会议的频道中发送议程文档。")
            if MeetingState(session.state) is not MeetingState.AWAITING_PARTICIPANTS:
                raise AgendaImportError("只能在会议开始前导入议程文档。")

            existing = await db.scalar(
                select(AgendaImportRow).where(
                    AgendaImportRow.session_id == session_id,
                    AgendaImportRow.message_id == message_id,
                    AgendaImportRow.attachment_id == attachment_id,
                )
            )
            if existing is not None:
                return _import_record(existing), False

            agenda_count = await db.scalar(
                select(func.count(MeetingAgendaItemRow.id)).where(
                    MeetingAgendaItemRow.session_id == session_id
                )
            )
            if int(agenda_count or 0) != 0:
                raise AgendaImportError("当前会议已有议程；为避免重复，不会自动覆盖或追加。")
            processing = await db.scalar(
                select(AgendaImportRow.id).where(
                    AgendaImportRow.session_id == session_id,
                    AgendaImportRow.status == AgendaImportStatus.PROCESSING.value,
                )
            )
            if processing is not None:
                raise AgendaImportError("当前已有一份议程文档正在处理。")

            row = AgendaImportRow(
                id=new_ulid(),
                session_id=session_id,
                channel_id=channel_id,
                message_id=message_id,
                attachment_id=attachment_id,
                actor_user_id=actor_user_id,
                content_sha256=content_sha256,
                source_text=source_text,
                validated_identity_user_ids_json=list(validated_identity_user_ids),
                status=AgendaImportStatus.PROCESSING.value,
                policy_version=policy_version,
                model_name=model_name,
                model_digest=model_digest,
                result_json={},
            )
            db.add(row)
            await db.flush()
            await append_meeting_event_row(
                db,
                session_id=session_id,
                event_type=MeetingEventType.AGENDA_IMPORT_ACCEPTED.value,
                actor_user_id=actor_user_id,
                source="DISCORD_ATTACHMENT",
                payload={
                    "agenda_import_id": row.id,
                    "content_sha256": content_sha256,
                    "policy_version": policy_version,
                    "model_name": model_name,
                    "model_digest": model_digest,
                },
                evidence_utterance_ids=(),
                state_before=MeetingState(session.state),
                state_after=MeetingState(session.state),
                reversible=False,
                correlation_id=row.id,
            )
            await db.refresh(row)
            return _import_record(row), True

    async def get(self, import_id: str) -> AgendaImportRecord:
        async with self._sessions() as db:
            row = await db.get(AgendaImportRow, import_id)
            if row is None:
                raise AgendaImportError("议程导入记录不存在。")
            return _import_record(row)

    async def list_processing(self) -> tuple[AgendaImportRecord, ...]:
        async with self._sessions() as db:
            rows = await db.scalars(
                select(AgendaImportRow)
                .where(AgendaImportRow.status == AgendaImportStatus.PROCESSING.value)
                .order_by(AgendaImportRow.created_at)
            )
            return tuple(_import_record(row) for row in rows)

    async def has_processing(self, session_id: str) -> bool:
        async with self._sessions() as db:
            row_id = await db.scalar(
                select(AgendaImportRow.id).where(
                    AgendaImportRow.session_id == session_id,
                    AgendaImportRow.status == AgendaImportStatus.PROCESSING.value,
                )
            )
            return row_id is not None

    async def mark_failed(self, import_id: str, *, error_type: str) -> AgendaImportRecord:
        async with self._sessions() as db, db.begin():
            row = await db.get(AgendaImportRow, import_id)
            if row is None:
                raise AgendaImportError("议程导入记录不存在。")
            if row.status != AgendaImportStatus.PROCESSING.value:
                return _import_record(row)
            session = await _session_row(db, row.session_id)
            row.status = AgendaImportStatus.FAILED.value
            row.last_error_type = error_type[:80]
            row.updated_at = utc_now()
            await append_meeting_event_row(
                db,
                session_id=row.session_id,
                event_type=MeetingEventType.AGENDA_IMPORT_FAILED.value,
                actor_user_id=row.actor_user_id,
                source="SYSTEM",
                payload={
                    "agenda_import_id": row.id,
                    "error_type": row.last_error_type,
                },
                evidence_utterance_ids=(),
                state_before=MeetingState(session.state),
                state_after=MeetingState(session.state),
                reversible=False,
                correlation_id=row.id,
            )
            await db.flush()
            return _import_record(row)

    async def commit(
        self,
        import_id: str,
        *,
        items: tuple[AgendaImportItem, ...],
        summary: dict[str, object],
    ) -> tuple[AgendaImportRecord, tuple[AgendaItemRecord, ...]]:
        """Atomically create every row and append one content-minimal import event."""

        async with self._sessions() as db, db.begin():
            row = await db.get(AgendaImportRow, import_id)
            if row is None:
                raise AgendaImportError("议程导入记录不存在。")
            if row.status == AgendaImportStatus.READY.value:
                existing = await db.scalars(
                    select(MeetingAgendaItemRow)
                    .where(MeetingAgendaItemRow.session_id == row.session_id)
                    .order_by(MeetingAgendaItemRow.position)
                )
                return _import_record(row), tuple(_agenda_record(item) for item in existing)
            if row.status != AgendaImportStatus.PROCESSING.value:
                raise AgendaImportError("该议程导入已结束，不能再次提交。")
            session = await _session_row(db, row.session_id)
            if MeetingState(session.state) is not MeetingState.AWAITING_PARTICIPANTS:
                raise AgendaImportError("会议状态已改变，议程导入未写入。")
            agenda_count = await db.scalar(
                select(func.count(MeetingAgendaItemRow.id)).where(
                    MeetingAgendaItemRow.session_id == row.session_id
                )
            )
            if int(agenda_count or 0) != 0:
                raise AgendaImportError("处理期间议程已发生变化，导入未写入。")

            agenda_rows: list[MeetingAgendaItemRow] = []
            for position, item in enumerate(items, start=1):
                agenda_row = MeetingAgendaItemRow(
                    id=new_ulid(),
                    session_id=row.session_id,
                    position=position,
                    title=item.title,
                    status=AgendaStatus.PENDING.value,
                    payload_json=_item_payload(row.id, item),
                )
                db.add(agenda_row)
                agenda_rows.append(agenda_row)
            await db.flush()

            row.status = AgendaImportStatus.READY.value
            row.result_json = dict(summary)
            row.last_error_type = None
            row.updated_at = utc_now()
            await append_meeting_event_row(
                db,
                session_id=row.session_id,
                event_type=MeetingEventType.AGENDA_IMPORT_COMMITTED.value,
                actor_user_id=row.actor_user_id,
                source="SYSTEM",
                payload={
                    "agenda_import_id": row.id,
                    "agenda_item_ids": [item.id for item in agenda_rows],
                    **dict(summary),
                },
                evidence_utterance_ids=(),
                state_before=MeetingState(session.state),
                state_after=MeetingState(session.state),
                reversible=False,
                correlation_id=row.id,
            )
            await db.flush()
            return _import_record(row), tuple(_agenda_record(item) for item in agenda_rows)
