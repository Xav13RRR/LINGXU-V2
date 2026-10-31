"""Independent Lingxu Worker process with Phase 2 persistent job leasing."""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
from pathlib import Path

import structlog

from lingxu.asr.provider import SenseVoiceProvider
from lingxu.config import Settings
from lingxu.lifecycle import ShutdownSignal
from lingxu.logging import configure_logging
from lingxu.resources import deepseek_full_transcript_minutes_template
from lingxu.storage.component_repositories import AgendaRepository, ParticipantRepository
from lingxu.storage.database import Database
from lingxu.storage.encryption import FileCipher
from lingxu.storage.migrate import require_database_current
from lingxu.storage.repositories import (
    AudioSegmentRepository,
    MeetingEventStore,
    MeetingSessionRepository,
    ProcessingJobRepository,
    RealtimeUtteranceRepository,
)
from lingxu.storage.services import StartupRecovery
from lingxu.storage.session_files import SessionFiles
from lingxu.storage.worker_repositories import ArtifactRepository, TranscriptRepository
from lingxu.worker.audio import SegmentTranscriber
from lingxu.worker.full_transcript_preview import DeepSeekFullTranscriptPreviewClient
from lingxu.worker.pipeline import PostMeetingPipeline
from lingxu.worker.publisher import DiscordArtifactPublisher
from lingxu.worker.service import WorkerService
from lingxu.worker.summary import DeepSeekPostMeetingInferenceProvider

_FULL_TRANSCRIPT_MINUTES_TEMPLATE = deepseek_full_transcript_minutes_template()


def _write_ready_marker(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part")
    with part.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ready\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, path)


async def run_worker(settings: Settings) -> None:
    """Validate the schema, recover, and poll the independent job table."""

    if not settings.summary_llm_enabled:
        raise RuntimeError("the fixed DeepSeek post-meeting path must be enabled for the Worker")
    await asyncio.to_thread(require_database_current, settings.database_url)
    database = Database.open(settings.database_url)
    sessions = MeetingSessionRepository(database.sessions)
    jobs = ProcessingJobRepository(database.sessions)
    recovery = await StartupRecovery(
        sessions=sessions,
        jobs=jobs,
        files=SessionFiles(settings.data_root),
    ).run()
    cipher = FileCipher.from_secret(settings.file_encryption_key)
    transcriber = SegmentTranscriber(
        cipher=cipher,
        provider=SenseVoiceProvider(
            model=settings.realtime_asr_model,
            model_revision=settings.realtime_asr_model_revision,
            vad_model=settings.realtime_asr_vad_model,
            vad_model_revision=settings.realtime_asr_vad_model_revision,
            model_cache_dir=settings.model_cache_dir,
            expected_model_sha256=settings.realtime_asr_model_sha256,
            expected_vad_model_sha256=settings.realtime_asr_vad_model_sha256,
        ),
        language=settings.realtime_asr_language_mode,
        hotwords=settings.realtime_asr_hotword_values,
    )
    token = settings.discord_token
    publisher = (
        DiscordArtifactPublisher(token.get_secret_value())
        if token is not None and token.get_secret_value()
        else None
    )
    summary_api_key = settings.summary_llm_api_key
    summary_api_key_value = (
        summary_api_key.get_secret_value()
        if summary_api_key is not None and summary_api_key.get_secret_value()
        else None
    )
    post_meeting_inference_provider = DeepSeekPostMeetingInferenceProvider(
        base_url=settings.summary_llm_base_url,
        model=settings.summary_llm_model,
        expected_model_digest=settings.summary_llm_model_digest,
        api_key=summary_api_key_value,
        timeout_seconds=settings.summary_llm_timeout_seconds,
        max_output_tokens=settings.summary_llm_max_output_tokens,
        max_input_chars=settings.summary_llm_max_input_chars,
        allow_remote=settings.summary_llm_allow_remote,
    )
    full_transcript_minutes_provider = DeepSeekFullTranscriptPreviewClient(
        api_key=summary_api_key_value or "",
        base_url=settings.summary_llm_base_url,
        model=settings.summary_llm_model,
        expected_model_digest=settings.summary_llm_model_digest,
        timeout_seconds=settings.summary_llm_timeout_seconds,
        max_output_tokens=settings.summary_final_minutes_max_output_tokens,
        allow_remote=settings.summary_llm_allow_remote,
        output_format="json_object",
    )
    pipeline = PostMeetingPipeline(
        sessions=sessions,
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=ArtifactRepository(database.sessions),
        files=SessionFiles(settings.data_root),
        transcriber=transcriber,
        publisher=publisher,
        retention_hours=settings.audio_retention_hours,
        post_meeting_inference_provider=post_meeting_inference_provider,
        full_transcript_minutes_provider=full_transcript_minutes_provider,
        full_transcript_minutes_template=_FULL_TRANSCRIPT_MINUTES_TEMPLATE,
        glossary_settings=settings,
    )
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    service = WorkerService(
        jobs=jobs,
        worker_id=worker_id,
        lease_seconds=settings.worker_job_lock_seconds,
        pipeline=pipeline,
        max_attempts=settings.worker_job_max_attempts,
    )
    marker = settings.runtime_tmp / "worker.ready"
    _write_ready_marker(marker)
    shutdown = ShutdownSignal()
    shutdown.install()
    structlog.get_logger(__name__).info(
        "worker_persistence_ready",
        parts_removed=recovery.parts_removed,
        expired_jobs_requeued=recovery.expired_jobs_requeued,
        sessions_recovered=len(recovery.sessions_recovered),
        tombstones_cleaned=len(recovery.tombstones_cleaned),
    )
    try:
        while True:
            await service.run_once()
            try:
                await asyncio.wait_for(
                    shutdown.wait(),
                    timeout=settings.worker_poll_interval_seconds,
                )
                break
            except TimeoutError:
                continue
    finally:
        with contextlib.suppress(FileNotFoundError):
            marker.unlink()
        await database.close()


def main() -> None:
    """Run the independently configured Worker skeleton."""

    settings = Settings()
    configure_logging(settings.log_level, secrets=settings.configured_secret_values)
    try:
        asyncio.run(run_worker(settings))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
