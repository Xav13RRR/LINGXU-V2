"""Lease-driven independent Worker with resumable post-Meeting processing."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable

import structlog

from lingxu.domain.errors import JobClaimError
from lingxu.storage.repositories import ProcessingJobRepository, utc_now
from lingxu.storage.worker_repositories import PublicationReconciliationRequired
from lingxu.worker.consistency import MinutesConsistencyError
from lingxu.worker.full_transcript_preview import FullTranscriptPreviewError
from lingxu.worker.outcome_partitions import OutcomePartitionError
from lingxu.worker.pipeline import POST_MEETING_JOB, PostMeetingPipeline

PERSISTENCE_PROBE_JOB = "PERSISTENCE_PROBE"


class WorkerService:
    """Independently claim, checkpoint, retry, and complete allowlisted jobs."""

    def __init__(
        self,
        *,
        jobs: ProcessingJobRepository,
        worker_id: str,
        lease_seconds: float = 30.0,
        pipeline: PostMeetingPipeline | None = None,
        max_attempts: int = 3,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("Worker lease duration must be positive")
        self._jobs = jobs
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._pipeline = pipeline
        self._max_attempts = max_attempts
        self._log = structlog.get_logger(__name__).bind(worker_id=worker_id)

    async def _heartbeat_lease(self, job_id: str, stop: asyncio.Event) -> None:
        interval_seconds = min(30.0, self._lease_seconds / 3.0)
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
                return
            except TimeoutError:
                await self._jobs.renew_lease(
                    job_id=job_id,
                    worker_id=self._worker_id,
                    lease_seconds=self._lease_seconds,
                )

    async def _run_with_lease_heartbeat(
        self,
        job_id: str,
        operation: Awaitable[None],
    ) -> None:
        stop = asyncio.Event()
        operation_task = asyncio.ensure_future(operation)
        heartbeat_task = asyncio.create_task(self._heartbeat_lease(job_id, stop))
        try:
            done, _ = await asyncio.wait(
                (operation_task, heartbeat_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                await heartbeat_task
                raise RuntimeError("Worker lease heartbeat stopped unexpectedly")
            stop.set()
            await heartbeat_task
            operation_task.result()
        finally:
            stop.set()
            for task in (operation_task, heartbeat_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(operation_task, heartbeat_task, return_exceptions=True)

    async def run_once(self) -> bool:
        """Claim and process at most one allowlisted job."""

        job_types = [PERSISTENCE_PROBE_JOB]
        if self._pipeline is not None:
            job_types.append(POST_MEETING_JOB)
        job = await self._jobs.claim(
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
            job_types=job_types,
        )
        if job is None:
            return False
        active_stage = job.current_stage
        try:

            async def process_claimed_job() -> None:
                if job.job_type == PERSISTENCE_PROBE_JOB:
                    await self._jobs.checkpoint(
                        job_id=job.id,
                        worker_id=self._worker_id,
                        stage="PERSISTENCE_READY",
                        checkpoint={"schema_version": 1},
                        lease_seconds=self._lease_seconds,
                    )
                    return
                if job.job_type != POST_MEETING_JOB or self._pipeline is None:
                    raise ValueError(f"unsupported Worker job type: {job.job_type}")

                async def checkpoint(stage: str, payload: dict[str, object]) -> None:
                    nonlocal active_stage
                    active_stage = stage
                    await self._jobs.checkpoint(
                        job_id=job.id,
                        worker_id=self._worker_id,
                        stage=stage,
                        checkpoint=dict(payload),
                        lease_seconds=self._lease_seconds,
                    )

                await self._pipeline.run(job, checkpoint)

            await self._run_with_lease_heartbeat(job.id, process_claimed_job())
            await self._jobs.complete(job_id=job.id, worker_id=self._worker_id)
        except JobClaimError:
            self._log.error(
                "worker_job_lease_lost",
                job_id=job.id,
                stage=active_stage,
            )
            return True
        except PublicationReconciliationRequired as exc:
            failure = f"{type(exc).__name__}@{active_stage}" if active_stage else type(exc).__name__
            await self._jobs.fail(
                job_id=job.id,
                worker_id=self._worker_id,
                error_type=failure,
                retry_at=None,
            )
            self._log.error(
                "worker_job_failed",
                job_id=job.id,
                error_type=type(exc).__name__,
                reason=exc.reason,
                stage=active_stage,
            )
            return True
        except MinutesConsistencyError as exc:
            failure = f"{type(exc).__name__}@{active_stage}" if active_stage else type(exc).__name__
            await self._jobs.fail(
                job_id=job.id,
                worker_id=self._worker_id,
                error_type=failure,
                retry_at=None,
            )
            self._log.error(
                "worker_job_failed",
                job_id=job.id,
                error_type=type(exc).__name__,
                invariant=exc.invariant,
                stage=active_stage,
            )
            return True
        except FullTranscriptPreviewError as exc:
            failure = f"{type(exc).__name__}[{exc.code}]"
            if active_stage:
                failure = f"{failure}@{active_stage}"
            await self._jobs.fail(
                job_id=job.id,
                worker_id=self._worker_id,
                error_type=failure,
                retry_at=None,
            )
            self._log.error(
                "worker_job_failed",
                job_id=job.id,
                error_type=type(exc).__name__,
                failure_code=exc.code,
                failure_diagnostics=(
                    exc.diagnostics.as_dict() if exc.diagnostics is not None else None
                ),
                validation_details=(
                    exc.validation_details.as_dict() if exc.validation_details is not None else None
                ),
                stage=active_stage,
            )
            return True
        except OutcomePartitionError as exc:
            failure = f"{type(exc).__name__}[{exc.reason}]"
            if active_stage:
                failure = f"{failure}@{active_stage}"
            await self._jobs.fail(
                job_id=job.id,
                worker_id=self._worker_id,
                error_type=failure,
                retry_at=None,
            )
            self._log.error(
                "worker_job_failed",
                job_id=job.id,
                error_type=type(exc).__name__,
                failure_code=exc.reason,
                stage=active_stage,
            )
            return True
        except Exception as exc:
            failure = type(exc).__name__
            if active_stage:
                failure = f"{failure}@{active_stage}"
            await self._jobs.fail(
                job_id=job.id,
                worker_id=self._worker_id,
                error_type=failure,
                retry_at=utc_now() if job.attempt_count < self._max_attempts else None,
            )
            self._log.error(
                "worker_job_failed",
                job_id=job.id,
                error_type=type(exc).__name__,
                stage=active_stage,
            )
            return True
        self._log.info("worker_job_completed", job_id=job.id, job_type=job.job_type)
        return True
