"""Content-safe deployed Phase 8 Worker smoke without external publication."""

from __future__ import annotations

import asyncio
import json

from lingxu.config import Settings
from lingxu.domain.enums import ConsentKind, JobStatus
from lingxu.domain.models import RealtimeUtteranceRecord
from lingxu.storage.component_repositories import AgendaRepository, ParticipantRepository
from lingxu.storage.database import Database
from lingxu.storage.repositories import (
    AudioSegmentRepository,
    MeetingEventStore,
    MeetingSessionRepository,
    ProcessingJobRepository,
    RealtimeUtteranceRepository,
)
from lingxu.storage.services import MeetingSessionService
from lingxu.storage.session_files import SessionFiles
from lingxu.storage.worker_repositories import ArtifactRepository, TranscriptRepository
from lingxu.utils import new_ulid
from lingxu.worker.pipeline import POST_MEETING_JOB, PostMeetingPipeline
from lingxu.worker.service import WorkerService


async def smoke() -> dict[str, object]:
    settings = Settings()
    database = Database.open(settings.database_url)
    sessions = MeetingSessionRepository(database.sessions)
    files = SessionFiles(settings.data_root)
    service = MeetingSessionService(sessions, files)
    session = await service.create(
        guild_id=9_000_000_000_000_008,
        creator_user_id=9_100_000_000_000_008,
    )
    try:
        participants = ParticipantRepository(database.sessions)
        await participants.join(session_id=session.id, user_id=session.creator_user_id)
        await participants.set_consent(
            session_id=session.id,
            user_id=session.creator_user_id,
            kind=ConsentKind.MINUTES,
            granted=True,
        )
        utterances = RealtimeUtteranceRepository(database.sessions)
        await utterances.add(
            RealtimeUtteranceRecord(
                id=new_ulid(),
                session_id=session.id,
                user_id=session.creator_user_id,
                started_ms=0,
                ended_ms=800,
                raw_text="<|zh|>部署验证",
                normalised_text="部署验证",
            )
        )
        events = MeetingEventStore(database.sessions)
        await events.append(
            session_id=session.id,
            event_type="VOTE_CLOSED",
            payload={"vote_id": "smoke", "counts": {"yes": 1, "no": 0}, "total": 1},
        )
        artifacts = ArtifactRepository(database.sessions)
        pipeline = PostMeetingPipeline(
            sessions=sessions,
            events=events,
            participants=participants,
            agenda=AgendaRepository(database.sessions),
            utterances=utterances,
            segments=AudioSegmentRepository(database.sessions),
            transcripts=TranscriptRepository(database.sessions),
            artifacts=artifacts,
            files=files,
            transcriber=None,
            publisher=None,
            retention_hours=settings.audio_retention_hours,
        )
        jobs = ProcessingJobRepository(database.sessions)
        job = await jobs.create(
            session_id=session.id,
            job_type=POST_MEETING_JOB,
            idempotency_key="deployed-smoke",
        )
        claimed = await WorkerService(
            jobs=jobs,
            worker_id="deployed-smoke",
            pipeline=pipeline,
        ).run_once()
        completed = await jobs.get(job.id)
        rendered = await artifacts.list(session.id)
        return {
            "claimed": claimed,
            "job_status": completed.status.value if completed else None,
            "last_stage": completed.current_stage if completed else None,
            "checkpoint_stages": len(completed.checkpoint["completed_stages"]) if completed else 0,
            "artifact_count": len(rendered),
            "publication_disabled": not any(
                artifact.artifact_type == "DISCORD_PUBLICATION" for artifact in rendered
            ),
            "completed": completed is not None and completed.status is JobStatus.COMPLETED,
        }
    finally:
        await service.delete(session_id=session.id, actor_user_id=session.creator_user_id)
        await database.close()


def main() -> None:
    print(json.dumps(asyncio.run(smoke()), sort_keys=True))


if __name__ == "__main__":
    main()
