from pathlib import Path

import pytest
from sqlalchemy import update

from lingxu.bot.client import LingxuBot
from lingxu.config import Settings
from lingxu.domain.enums import JobStatus, MeetingState, ServiceCapacityMode
from lingxu.storage.repositories import ProcessingJobRepository
from lingxu.storage.tables import MeetingSessionRow, ProcessingJobRow


def _bot(tmp_path: Path, database_name: str) -> LingxuBot:
    return LingxuBot(
        Settings(
            _env_file=None,
            discord_application_id=123,
            database_url=(f"sqlite+aiosqlite:///{(tmp_path / database_name).as_posix()}"),
            data_root=tmp_path / "sessions",
            runtime_tmp=tmp_path / "runtime",
            tts_provider="cosyvoice-sft",
            tts_cache_dir=tmp_path / "tts",
        )
    )


@pytest.mark.asyncio
async def test_restart_releases_meeting_claimed_before_runtime_started(tmp_path: Path) -> None:
    bot = _bot(tmp_path, "stale-start.db")
    await bot.meetings.database.create_schema()
    try:
        session = await bot.meetings.controller.create(guild_id=101, host_user_id=201)
        assert session.state is MeetingState.AWAITING_PARTICIPANTS
        lease = await bot.capacity.claim(
            guild_id=session.guild_id,
            mode=ServiceCapacityMode.MEETING,
            owner_id=session.id,
        )

        await bot._reconcile_capacity()

        assert await bot.capacity.current() is None
        assert [event.event_type for event in await bot.capacity.events(lease.id)] == [
            "ACQUIRED",
            "RELEASED",
        ]
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_restart_retains_failed_meeting_retry_until_job_is_terminal(tmp_path: Path) -> None:
    bot = _bot(tmp_path, "retry.db")
    await bot.meetings.database.create_schema()
    try:
        session = await bot.meetings.controller.create(guild_id=102, host_user_id=202)
        async with bot.meetings.database.sessions() as db, db.begin():
            await db.execute(
                update(MeetingSessionRow)
                .where(MeetingSessionRow.id == session.id)
                .values(
                    state=MeetingState.FAILED_POST_PROCESSING.value,
                    active_marker=None,
                )
            )
        job_repository = ProcessingJobRepository(bot.meetings.database.sessions)
        job = await job_repository.create(
            session_id=session.id,
            job_type="POST_MEETING",
            idempotency_key="retry-capacity-test",
        )
        lease = await bot.capacity.claim(
            guild_id=session.guild_id,
            mode=ServiceCapacityMode.MEETING,
            owner_id=session.id,
        )

        await bot._reconcile_capacity()

        assert (await bot.capacity.current()).id == lease.id  # type: ignore[union-attr]
        assert [event.event_type for event in await bot.capacity.events(lease.id)] == [
            "ACQUIRED",
            "RECOVERED",
        ]
        claimed = await job_repository.claim(worker_id="retry-worker", lease_seconds=60)
        assert claimed is not None
        assert claimed.id == job.id
        await job_repository.complete(job_id=job.id, worker_id="retry-worker")

        await bot._release_terminal_job_capacity()

        assert await bot.capacity.current() is None
    finally:
        await bot.close()


@pytest.mark.parametrize("job_status", [JobStatus.PENDING, JobStatus.RUNNING])
@pytest.mark.asyncio
async def test_restart_retains_stopped_meeting_until_post_meeting_job_is_terminal(
    tmp_path: Path,
    job_status: JobStatus,
) -> None:
    bot = _bot(tmp_path, f"stopped-{job_status.value.lower()}.db")
    await bot.meetings.database.create_schema()
    try:
        session = await bot.meetings.controller.create(guild_id=103, host_user_id=203)
        async with bot.meetings.database.sessions() as db, db.begin():
            await db.execute(
                update(MeetingSessionRow)
                .where(MeetingSessionRow.id == session.id)
                .values(state=MeetingState.STOPPED.value, active_marker=None)
            )
        jobs = ProcessingJobRepository(bot.meetings.database.sessions)
        job = await jobs.create(
            session_id=session.id,
            job_type="POST_MEETING",
            idempotency_key=f"stopped-{job_status.value.lower()}",
        )
        if job_status is JobStatus.RUNNING:
            claimed = await jobs.claim(worker_id="stopped-worker", lease_seconds=60)
            assert claimed is not None
            assert claimed.id == job.id
        lease = await bot.capacity.claim(
            guild_id=session.guild_id,
            mode=ServiceCapacityMode.MEETING,
            owner_id=session.id,
        )

        await bot._reconcile_capacity()

        current = await bot.capacity.current()
        assert current is not None
        assert current.id == lease.id
        events = await bot.capacity.events(lease.id)
        assert [event.event_type for event in events] == ["ACQUIRED", "RECOVERED"]
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_restart_retains_stopped_meeting_while_terminal_job_action_is_pending(
    tmp_path: Path,
) -> None:
    bot = _bot(tmp_path, "stopped-action-pending.db")
    await bot.meetings.database.create_schema()
    try:
        session = await bot.meetings.controller.create(guild_id=105, host_user_id=205)
        stopped = await bot.meetings.runtime.stop(
            session.id,
            actor_user_id=session.creator_user_id,
            source="SLASH_COMMAND",
        )
        assert stopped.session.state is MeetingState.STOPPED
        jobs = ProcessingJobRepository(bot.meetings.database.sessions)
        assert await jobs.latest_for_session(session.id, job_type="POST_MEETING") is None
        lease = await bot.capacity.claim(
            guild_id=session.guild_id,
            mode=ServiceCapacityMode.MEETING,
            owner_id=session.id,
        )

        await bot._reconcile_capacity()

        current = await bot.capacity.current()
        assert current is not None
        assert current.id == lease.id
        events = await bot.capacity.events(lease.id)
        assert [event.event_type for event in events] == ["ACQUIRED", "RECOVERED"]
    finally:
        await bot.close()


@pytest.mark.parametrize(
    "job_status",
    [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED],
)
@pytest.mark.asyncio
async def test_periodic_reconciliation_releases_stopped_meeting_terminal_job_capacity(
    tmp_path: Path,
    job_status: JobStatus,
) -> None:
    bot = _bot(tmp_path, f"stopped-terminal-{job_status.value.lower()}.db")
    await bot.meetings.database.create_schema()
    try:
        session = await bot.meetings.controller.create(guild_id=104, host_user_id=204)
        async with bot.meetings.database.sessions() as db, db.begin():
            await db.execute(
                update(MeetingSessionRow)
                .where(MeetingSessionRow.id == session.id)
                .values(state=MeetingState.STOPPED.value, active_marker=None)
            )
        jobs = ProcessingJobRepository(bot.meetings.database.sessions)
        job = await jobs.create(
            session_id=session.id,
            job_type="POST_MEETING",
            idempotency_key=f"stopped-terminal-{job_status.value.lower()}",
        )
        async with bot.meetings.database.sessions() as db, db.begin():
            await db.execute(
                update(ProcessingJobRow)
                .where(ProcessingJobRow.id == job.id)
                .values(status=job_status.value)
            )
        lease = await bot.capacity.claim(
            guild_id=session.guild_id,
            mode=ServiceCapacityMode.MEETING,
            owner_id=session.id,
        )

        await bot._release_terminal_job_capacity()

        assert await bot.capacity.current() is None
        events = await bot.capacity.events(lease.id)
        assert [event.event_type for event in events] == ["ACQUIRED", "RELEASED"]
        assert events[-1].reason == f"STOPPED_POST_MEETING_{job_status.value}"
    finally:
        await bot.close()
