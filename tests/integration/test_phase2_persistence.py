import asyncio
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from lingxu.domain.enums import JobStatus, MeetingState
from lingxu.domain.errors import (
    InvalidStateTransitionError,
    JobClaimError,
    SessionAlreadyActiveError,
    SessionNotFoundError,
)
from lingxu.storage.database import Database
from lingxu.storage.repositories import (
    MeetingEventStore,
    MeetingSessionRepository,
    MeetingSnapshotStore,
    ProcessingJobRepository,
    utc_now,
)
from lingxu.storage.services import MeetingSessionService, StartupRecovery
from lingxu.storage.session_files import SessionFiles
from lingxu.worker.pipeline import POST_MEETING_JOB
from lingxu.worker.service import PERSISTENCE_PROBE_JOB, WorkerService


def _database_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.as_posix()}"


@pytest.mark.asyncio
async def test_append_only_events_have_unique_atomic_sequence(tmp_path: Path) -> None:
    database = Database.open(_database_url(tmp_path / "events.db"))
    await database.create_schema()
    sessions = MeetingSessionRepository(database.sessions)
    events = MeetingEventStore(database.sessions)
    record = await sessions.create(
        guild_id=10,
        creator_user_id=20,
        manifest_path=tmp_path / "manifest.json",
    )

    appended = await asyncio.gather(
        *(
            events.append(
                session_id=record.id,
                event_type="CONCURRENT_TEST",
                payload={"index": index},
            )
            for index in range(30)
        )
    )
    assert sorted(event.sequence_number for event in appended) == list(range(1, 31))

    async with database.sessions() as db:
        with pytest.raises(DBAPIError):
            await db.execute(
                text("UPDATE meeting_events SET event_type='MUTATED' WHERE session_id=:session_id"),
                {"session_id": record.id},
            )
            await db.commit()
        await db.rollback()
        with pytest.raises(DBAPIError):
            await db.execute(
                text("DELETE FROM meeting_events WHERE session_id=:session_id"),
                {"session_id": record.id},
            )
            await db.commit()
        await db.rollback()

    assert len(await events.list(record.id)) == 30

    await database.close()


@pytest.mark.asyncio
async def test_session_recovery_job_leases_and_synchronised_delete(tmp_path: Path) -> None:
    database_url = _database_url(tmp_path / "recovery.db")
    session_root = tmp_path / "sessions"
    database = Database.open(database_url)
    await database.create_schema()
    session_repository = MeetingSessionRepository(database.sessions)
    jobs = ProcessingJobRepository(database.sessions)
    snapshots = MeetingSnapshotStore(database.sessions)
    service = MeetingSessionService(session_repository, SessionFiles(session_root))
    record = await service.create(guild_id=11, creator_user_id=22)
    with pytest.raises(SessionAlreadyActiveError):
        await service.create(guild_id=11, creator_user_id=23)

    transitioned = await service.transition(
        session_id=record.id,
        target=MeetingState.AWAITING_PARTICIPANTS,
        actor_user_id=22,
        source="TEST",
    )
    assert transitioned.state is MeetingState.AWAITING_PARTICIPANTS
    with pytest.raises(InvalidStateTransitionError):
        await service.transition(
            session_id=record.id,
            target=MeetingState.OPENING,
            actor_user_id=22,
            source="TEST",
        )

    snapshot = await snapshots.save(
        session_id=record.id,
        state=transitioned.state,
        payload={"agenda_index": 0},
    )
    expired_at = utc_now() - timedelta(minutes=5)
    job = await jobs.create(
        session_id=record.id,
        job_type=PERSISTENCE_PROBE_JOB,
        idempotency_key="one",
        available_at=expired_at,
    )
    duplicate = await jobs.create(
        session_id=record.id,
        job_type=PERSISTENCE_PROBE_JOB,
        idempotency_key="one",
    )
    assert duplicate.id == job.id
    claimed = await jobs.claim(
        worker_id="crashed-worker",
        lease_seconds=1,
        now=expired_at,
    )
    assert claimed is not None
    assert claimed.status is JobStatus.RUNNING
    part = session_root / record.id / "meeting/snapshots.jsonl.part"
    part.write_bytes(b"incomplete")
    await database.close()

    reopened = Database.open(database_url)
    recovered_sessions = MeetingSessionRepository(reopened.sessions)
    recovered_jobs = ProcessingJobRepository(reopened.sessions)
    report = await StartupRecovery(
        sessions=recovered_sessions,
        jobs=recovered_jobs,
        files=SessionFiles(session_root),
    ).run()
    assert report.parts_removed == 1
    assert report.expired_jobs_requeued == 1
    assert report.sessions_recovered == (record.id,)
    assert (await MeetingSnapshotStore(reopened.sessions).latest(record.id)) == snapshot

    worker_results = await asyncio.gather(
        WorkerService(jobs=recovered_jobs, worker_id="independent-worker-a").run_once(),
        WorkerService(jobs=recovered_jobs, worker_id="independent-worker-b").run_once(),
    )
    assert sorted(worker_results) == [False, True]
    completed = await recovered_jobs.get(job.id)
    assert completed is not None
    assert completed.status is JobStatus.COMPLETED
    assert completed.current_stage == "PERSISTENCE_READY"

    delete_service = MeetingSessionService(
        recovered_sessions,
        SessionFiles(session_root),
    )
    await delete_service.delete(session_id=record.id, actor_user_id=22)
    assert not (session_root / record.id).exists()
    with pytest.raises(SessionNotFoundError):
        await recovered_sessions.get(record.id)
    await reopened.close()


@pytest.mark.asyncio
async def test_processing_job_owner_can_renew_lease_without_changing_claim(tmp_path: Path) -> None:
    database = Database.open(_database_url(tmp_path / "renew.db"))
    await database.create_schema()
    session = await MeetingSessionRepository(database.sessions).create(
        guild_id=31,
        creator_user_id=32,
        manifest_path=tmp_path / "manifest.json",
    )
    jobs = ProcessingJobRepository(database.sessions)
    job = await jobs.create(
        session_id=session.id,
        job_type=PERSISTENCE_PROBE_JOB,
        idempotency_key="renew",
    )
    claimed_at = utc_now()
    claimed = await jobs.claim(
        worker_id="lease-owner",
        lease_seconds=10,
        now=claimed_at,
    )
    assert claimed is not None and claimed.id == job.id

    renewed_at = claimed_at + timedelta(seconds=5)
    renewed = await jobs.renew_lease(
        job_id=job.id,
        worker_id="lease-owner",
        lease_seconds=30,
        now=renewed_at,
    )

    assert renewed.status is JobStatus.RUNNING
    assert renewed.locked_at == claimed_at
    assert renewed.locked_by == "lease-owner"
    assert renewed.lock_expires_at == renewed_at + timedelta(seconds=30)
    assert renewed.attempt_count == 1
    with pytest.raises(JobClaimError):
        await jobs.renew_lease(
            job_id=job.id,
            worker_id="different-worker",
            lease_seconds=30,
        )
    with pytest.raises(ValueError):
        await jobs.renew_lease(
            job_id=job.id,
            worker_id="lease-owner",
            lease_seconds=0,
        )
    await database.close()


@pytest.mark.asyncio
async def test_worker_heartbeats_lease_while_pipeline_stage_is_in_flight(tmp_path: Path) -> None:
    database = Database.open(_database_url(tmp_path / "heartbeat.db"))
    await database.create_schema()
    session = await MeetingSessionRepository(database.sessions).create(
        guild_id=41,
        creator_user_id=42,
        manifest_path=tmp_path / "manifest.json",
    )
    jobs = ProcessingJobRepository(database.sessions)
    job = await jobs.create(
        session_id=session.id,
        job_type=POST_MEETING_JOB,
        idempotency_key="heartbeat",
    )
    stage_started = asyncio.Event()
    stage_release = asyncio.Event()

    class SlowPipeline:
        async def run(self, claimed_job: object, checkpoint: object) -> None:
            del claimed_job, checkpoint
            stage_started.set()
            await stage_release.wait()

    service = WorkerService(
        jobs=jobs,
        worker_id="heartbeat-worker",
        lease_seconds=0.15,
        pipeline=SlowPipeline(),  # type: ignore[arg-type]
    )
    service_task = asyncio.create_task(service.run_once())
    await asyncio.wait_for(stage_started.wait(), timeout=1)
    initial = await jobs.get(job.id)
    assert initial is not None and initial.lock_expires_at is not None

    await asyncio.sleep(0.22)
    in_flight = await jobs.get(job.id)
    assert in_flight is not None and in_flight.lock_expires_at is not None
    assert in_flight.lock_expires_at > initial.lock_expires_at
    assert in_flight.status is JobStatus.RUNNING
    assert in_flight.locked_by == "heartbeat-worker"

    stage_release.set()
    assert await asyncio.wait_for(service_task, timeout=1)
    completed = await jobs.get(job.id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    await database.close()


@pytest.mark.asyncio
async def test_worker_cancels_local_pipeline_and_does_not_fail_job_after_lease_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database.open(_database_url(tmp_path / "lease-loss.db"))
    await database.create_schema()
    session = await MeetingSessionRepository(database.sessions).create(
        guild_id=51,
        creator_user_id=52,
        manifest_path=tmp_path / "manifest.json",
    )
    jobs = ProcessingJobRepository(database.sessions)
    job = await jobs.create(
        session_id=session.id,
        job_type=POST_MEETING_JOB,
        idempotency_key="lease-loss",
    )
    stage_started = asyncio.Event()
    stage_cancelled = asyncio.Event()

    class BlockingPipeline:
        async def run(self, claimed_job: object, checkpoint: object) -> None:
            del claimed_job, checkpoint
            stage_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stage_cancelled.set()
                raise

    async def lose_lease(**kwargs: object) -> object:
        del kwargs
        raise JobClaimError("simulated ownership loss")

    monkeypatch.setattr(jobs, "renew_lease", lose_lease)
    service = WorkerService(
        jobs=jobs,
        worker_id="stale-worker",
        lease_seconds=0.03,
        pipeline=BlockingPipeline(),  # type: ignore[arg-type]
    )
    service_task = asyncio.create_task(service.run_once())
    await asyncio.wait_for(stage_started.wait(), timeout=1)

    assert await asyncio.wait_for(service_task, timeout=1)
    assert stage_cancelled.is_set()
    retained = await jobs.get(job.id)
    assert retained is not None
    assert retained.status is JobStatus.RUNNING
    assert retained.locked_by == "stale-worker"
    assert retained.last_error_type is None
    await database.close()
