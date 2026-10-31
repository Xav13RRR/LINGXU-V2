"""Discord publication durability and recovery integration contracts."""

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest

from lingxu.domain.enums import JobStatus
from lingxu.storage.component_repositories import AgendaRepository, ParticipantRepository
from lingxu.storage.repositories import (
    AudioSegmentRepository,
    MeetingEventStore,
    MeetingSessionRepository,
    RealtimeUtteranceRepository,
)
from lingxu.storage.session_files import SessionFiles
from lingxu.storage.worker_repositories import (
    ArtifactRepository,
    PreparationResult,
    PublicationReconciliationRequired,
    TranscriptRepository,
)
from lingxu.utils import new_ulid
from lingxu.worker.pipeline import _OUTCOME_PIPELINE_REVISION, PostMeetingPipeline
from lingxu.worker.semantic_policy import SPEECH_ACT_POLICY_REVISION
from lingxu.worker.service import WorkerService
from tests.integration.worker_fixtures import (
    FakePublisher,
    _pipeline_fixture,
    _test_payload_hash,
)


async def _seed_historical_task10_prepared_intent(
    *,
    pipeline: PostMeetingPipeline,
    artifacts: ArtifactRepository,
    session_id: str,
) -> None:
    """Represent a valid stored Task 10 intent without authorizing new four-file sends."""

    manifest = tuple(
        await pipeline._publication_manifest(
            session_id,
            publication_contract_revision="task10-four-file-v1",
        )
    )
    for item in manifest:
        path = Path(str(item["path"]))
        record = await artifacts.at_path(
            session_id=session_id,
            artifact_type=str(item["artifact_type"]),
            path=path,
        )
        assert record is not None
        await artifacts.add(
            session_id=session_id,
            artifact_type=record.artifact_type,
            path=record.path,
            sha256=record.sha256,
            metadata={**record.metadata, "pipeline_version": 3},
        )
    placeholder = await artifacts.add(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION_INTENT",
        path=Path(f"discord-publication-intent-{session_id}"),
        sha256="0" * 64,
        metadata={},
    )
    intent_id = placeholder.id
    metadata: dict[str, object] = {
        "intent_schema_version": 2,
        "intent_id": intent_id,
        "state": "PREPARED",
        "session_id": session_id,
        "guild_id": 81,
        "requested_channel_id": 8100,
        "pipeline_version": 3,
        "outcome_pipeline_revision": _OUTCOME_PIPELINE_REVISION,
        "speech_act_policy_revision": SPEECH_ACT_POLICY_REVISION,
        "publication_contract_revision": "task10-four-file-v1",
        "intent_nonce": new_ulid(),
        "attachments": list(manifest),
        "artifact_set_sha256": _test_payload_hash(list(manifest)),
    }
    metadata["intent_payload_sha256"] = _test_payload_hash(metadata)
    await artifacts.add(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION_INTENT",
        path=Path(f"discord-publication-intent-{session_id}"),
        sha256=str(metadata["intent_payload_sha256"]),
        metadata=dict(metadata),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_legacy_publication_receipt_is_terminal_without_resend(tmp_path: Path) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await pipeline.run(job, interrupt_before_publish)
    assert publisher.calls == 0
    await artifacts.add(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION",
        path=Path("discord-8100-1001"),
        sha256=hashlib.sha256(b"1001").hexdigest(),
        metadata={"channel_id": 8100, "message_id": "1001"},
    )

    assert await WorkerService(
        jobs=jobs,
        worker_id="legacy-terminal-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert completed.current_stage is None
    assert completed.checkpoint == {}
    assert publisher.calls == 0
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_historical_task10_finalized_pair_is_terminal_without_resend(tmp_path: Path) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await pipeline.run(job, interrupt_before_publish)
    await _seed_historical_task10_prepared_intent(
        pipeline=pipeline,
        artifacts=artifacts,
        session_id=session_id,
    )
    finalized, receipt = await artifacts.finalize_discord_publication(
        session_id=session_id,
        selected_channel_id=8100,
        message_id="1001",
    )
    assert finalized.metadata["publication_contract_revision"] == "task10-four-file-v1"
    assert receipt.metadata["publication_contract_revision"] == "task10-four-file-v1"

    assert await pipeline._stage_publish_discord(session_id) == {
        "published": False,
        "deduplicated": True,
    }
    assert publisher.calls == 0
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_historical_task10_prepared_intent_requires_manual_reconciliation(
    tmp_path: Path,
) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await pipeline.run(job, interrupt_before_publish)
    await _seed_historical_task10_prepared_intent(
        pipeline=pipeline,
        artifacts=artifacts,
        session_id=session_id,
    )
    with pytest.raises(PublicationReconciliationRequired) as error:
        await pipeline._stage_publish_discord(session_id)
    assert error.value.reason == "publication_prepared_unresolved"
    assert publisher.calls == 0
    prepared = await artifacts.first_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION_INTENT",
    )
    assert prepared is not None
    assert prepared.metadata["state"] == "PREPARED"
    assert prepared.metadata["publication_contract_revision"] == "task10-four-file-v1"

    finalized, receipt = await artifacts.finalize_discord_publication(
        session_id=session_id,
        selected_channel_id=8100,
        message_id="1001",
    )
    assert finalized.metadata["state"] == receipt.metadata["state"] == "FINALIZED"
    assert publisher.calls == 0
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_crash_after_prepared_commit_before_publish_never_resends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
    )
    original_prepare = artifacts.prepare_discord_publication

    async def crash_after_prepare(
        **kwargs: Any,
    ) -> PreparationResult:
        await original_prepare(**kwargs)
        raise RuntimeError("injected post-prepare crash")

    monkeypatch.setattr(artifacts, "prepare_discord_publication", crash_after_prepare)
    assert await WorkerService(
        jobs=jobs,
        worker_id="post-prepare-crash-worker",
        pipeline=pipeline,
    ).run_once()
    pending = await jobs.get(job_id)
    assert pending is not None and pending.status is JobStatus.PENDING
    assert publisher.calls == 0
    intents = await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION_INTENT",
    )
    assert len(intents) == 1 and intents[0].metadata["state"] == "PREPARED"

    monkeypatch.setattr(artifacts, "prepare_discord_publication", original_prepare)
    recovery_publisher = FakePublisher()
    recovery_pipeline = PostMeetingPipeline(
        sessions=MeetingSessionRepository(database.sessions),
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=artifacts,
        files=SessionFiles(tmp_path / "sessions"),
        transcriber=None,
        publisher=recovery_publisher,
        retention_hours=24,
    )
    assert await WorkerService(
        jobs=jobs,
        worker_id="post-prepare-recovery-worker",
        pipeline=recovery_pipeline,
    ).run_once()
    failed = await jobs.get(job_id)
    assert failed is not None and failed.status is JobStatus.FAILED
    assert recovery_publisher.calls == 0
    assert not await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION",
    )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_publish_success_then_finalize_crash_requires_manual_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
    )
    original_finalize = artifacts.finalize_discord_publication

    async def crash_before_finalize(**kwargs: Any) -> tuple[Any, Any]:
        del kwargs
        raise RuntimeError("injected pre-finalize crash")

    monkeypatch.setattr(artifacts, "finalize_discord_publication", crash_before_finalize)
    assert await WorkerService(
        jobs=jobs,
        worker_id="pre-finalize-crash-worker",
        pipeline=pipeline,
    ).run_once()
    pending = await jobs.get(job_id)
    assert pending is not None and pending.status is JobStatus.PENDING
    assert publisher.calls == 1
    assert [
        intent.metadata["state"]
        for intent in await artifacts.all_of_type(
            session_id=session_id,
            artifact_type="DISCORD_PUBLICATION_INTENT",
        )
    ] == ["PREPARED"]

    monkeypatch.setattr(artifacts, "finalize_discord_publication", original_finalize)
    recovery_publisher = FakePublisher()
    recovery_pipeline = PostMeetingPipeline(
        sessions=MeetingSessionRepository(database.sessions),
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=artifacts,
        files=SessionFiles(tmp_path / "sessions"),
        transcriber=None,
        publisher=recovery_publisher,
        retention_hours=24,
    )
    assert await WorkerService(
        jobs=jobs,
        worker_id="pre-finalize-recovery-worker",
        pipeline=recovery_pipeline,
    ).run_once()
    failed = await jobs.get(job_id)
    assert failed is not None and failed.status is JobStatus.FAILED
    assert recovery_publisher.calls == 0
    assert not await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION",
    )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_duplicate_publication_intents_fail_without_resend(tmp_path: Path) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await pipeline.run(job, interrupt_before_publish)
    for suffix in ("one", "two"):
        await artifacts.add(
            session_id=session_id,
            artifact_type="DISCORD_PUBLICATION_INTENT",
            path=Path(f"invalid-intent-{suffix}"),
            sha256="0" * 64,
            metadata={"state": "PREPARED"},
        )

    assert await WorkerService(
        jobs=jobs,
        worker_id="duplicate-intent-worker",
        pipeline=pipeline,
    ).run_once()
    failed = await jobs.get(job_id)
    assert failed is not None and failed.status is JobStatus.FAILED
    assert publisher.calls == 0
    assert (
        len(
            await artifacts.all_of_type(
                session_id=session_id,
                artifact_type="DISCORD_PUBLICATION_INTENT",
            )
        )
        == 2
    )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_publication_preparation_has_one_creator(tmp_path: Path) -> None:
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await pipeline.run(job, interrupt_before_publish)
    manifest = tuple(await pipeline._publication_manifest(session_id))

    async def prepare() -> PreparationResult:
        repository = ArtifactRepository(
            database.sessions, session_files=SessionFiles(tmp_path / "sessions")
        )
        return await repository.prepare_discord_publication(
            session_id=session_id,
            pipeline_version=4,
            outcome_pipeline_revision=_OUTCOME_PIPELINE_REVISION,
            speech_act_policy_revision=SPEECH_ACT_POLICY_REVISION,
            publication_contract_revision="task11-one-file-v1",
            attachments=manifest,
        )

    results = await asyncio.gather(prepare(), prepare(), return_exceptions=True)
    created = [result for result in results if isinstance(result, PreparationResult)]
    failed = [result for result in results if isinstance(result, PublicationReconciliationRequired)]
    assert len(created) == 1
    assert created[0].created_by_this_call
    assert len(failed) == 1
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_repository_constructs_and_strictly_validates_publication_records(
    tmp_path: Path,
) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
    )
    job = await jobs.get(job_id)
    assert job is not None
    revision: str | None = None
    speech_revision: str | None = None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        nonlocal revision, speech_revision
        selected_revision = payload.get("outcome_pipeline_revision")
        if isinstance(selected_revision, str):
            revision = selected_revision
        selected_speech_revision = payload.get("speech_act_policy_revision")
        if isinstance(selected_speech_revision, str):
            speech_revision = selected_speech_revision
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await pipeline.run(job, interrupt_before_publish)
    assert revision is not None
    assert speech_revision is not None
    manifest = tuple(await pipeline._publication_manifest(session_id))
    historical_manifest = tuple(
        await pipeline._publication_manifest(
            session_id,
            publication_contract_revision="task10-four-file-v1",
        )
    )

    with pytest.raises(PublicationReconciliationRequired):
        await artifacts.prepare_discord_publication(
            session_id=session_id,
            pipeline_version=4,
            outcome_pipeline_revision=revision,
            speech_act_policy_revision=speech_revision,
            publication_contract_revision="task10-four-file-v1",
            attachments=historical_manifest,
        )

    with pytest.raises(PublicationReconciliationRequired):
        await artifacts.prepare_discord_publication(
            session_id=session_id,
            pipeline_version=4,
            outcome_pipeline_revision=revision,
            speech_act_policy_revision=speech_revision,
            publication_contract_revision="task11-one-file-v1",
            attachments=(*manifest, manifest[0]),
        )

    prepared = await artifacts.prepare_discord_publication(
        session_id=session_id,
        pipeline_version=4,
        outcome_pipeline_revision=revision,
        speech_act_policy_revision=speech_revision,
        publication_contract_revision="task11-one-file-v1",
        attachments=manifest,
    )
    assert prepared.created_by_this_call
    assert prepared.record.metadata["intent_id"] == prepared.record.id

    forged_metadata = {
        **prepared.record.metadata,
        "requested_channel_id": 9999,
    }
    forged_metadata["intent_payload_sha256"] = _test_payload_hash(
        {key: value for key, value in forged_metadata.items() if key != "intent_payload_sha256"}
    )
    await artifacts.add(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION_INTENT",
        path=prepared.record.path,
        sha256=forged_metadata["intent_payload_sha256"],
        metadata=forged_metadata,
    )
    with pytest.raises(PublicationReconciliationRequired):
        await artifacts.finalize_discord_publication(
            session_id=session_id,
            selected_channel_id=8100,
            message_id="1001",
        )
    assert publisher.calls == 0
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_repository_finalizer_constructs_exact_cross_bound_pair(tmp_path: Path) -> None:
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
    )
    job = await jobs.get(job_id)
    assert job is not None
    revision: str | None = None
    speech_revision: str | None = None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        nonlocal revision, speech_revision
        selected_revision = payload.get("outcome_pipeline_revision")
        if isinstance(selected_revision, str):
            revision = selected_revision
        selected_speech_revision = payload.get("speech_act_policy_revision")
        if isinstance(selected_speech_revision, str):
            speech_revision = selected_speech_revision
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await pipeline.run(job, interrupt_before_publish)
    assert revision is not None
    assert speech_revision is not None
    prepared = await artifacts.prepare_discord_publication(
        session_id=session_id,
        pipeline_version=4,
        outcome_pipeline_revision=revision,
        speech_act_policy_revision=speech_revision,
        publication_contract_revision="task11-one-file-v1",
        attachments=tuple(await pipeline._publication_manifest(session_id)),
    )

    finalized, receipt = await artifacts.finalize_discord_publication(
        session_id=session_id,
        selected_channel_id=8100,
        message_id="1001",
    )

    assert finalized.id == prepared.record.id
    assert finalized.metadata["state"] == receipt.metadata["state"] == "FINALIZED"
    assert finalized.metadata["intent_id"] == receipt.metadata["intent_id"] == finalized.id
    assert (
        finalized.metadata["receipt_payload_sha256"] == receipt.metadata["receipt_payload_sha256"]
    )
    assert set(finalized.metadata) == {
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
        "selected_channel_id",
        "message_id",
        "receipt_path",
        "receipt_payload_sha256",
        "intent_payload_sha256",
    }
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_finalized_previous_speech_revision_is_terminal_without_resend(
    tmp_path: Path,
) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path, publisher
    )
    job = await jobs.get(job_id)
    assert job is not None
    await pipeline.run(job, lambda _stage, _payload: asyncio.sleep(0))
    assert publisher.calls == 1
    old_revision = hashlib.sha256(b"lingxu-speech-act-policy-task10r-v1").hexdigest()
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    for artifact_type, name in (
        ("MINUTES_MARKDOWN", "minutes.md"),
        ("MINUTES_JSON", "minutes.json"),
        ("EVIDENCE_JSON", "evidence.json"),
    ):
        record = await artifacts.at_path(
            session_id=session_id, artifact_type=artifact_type, path=meeting_dir / name
        )
        assert record is not None
        await artifacts.add(
            session_id=session_id,
            artifact_type=artifact_type,
            path=record.path,
            sha256=record.sha256,
            metadata={**record.metadata, "speech_act_policy_revision": old_revision},
        )
    intent = await artifacts.first_of_type(
        session_id=session_id, artifact_type="DISCORD_PUBLICATION_INTENT"
    )
    receipt = await artifacts.first_of_type(
        session_id=session_id, artifact_type="DISCORD_PUBLICATION"
    )
    assert intent is not None and receipt is not None
    receipt_metadata = {**receipt.metadata, "speech_act_policy_revision": old_revision}
    receipt_metadata["receipt_payload_sha256"] = _test_payload_hash(
        {key: value for key, value in receipt_metadata.items() if key != "receipt_payload_sha256"}
    )
    intent_metadata = {
        **intent.metadata,
        "speech_act_policy_revision": old_revision,
        "receipt_payload_sha256": receipt_metadata["receipt_payload_sha256"],
    }
    intent_metadata["intent_payload_sha256"] = _test_payload_hash(
        {key: value for key, value in intent_metadata.items() if key != "intent_payload_sha256"}
    )
    await artifacts.add(
        session_id=session_id,
        artifact_type=intent.artifact_type,
        path=intent.path,
        sha256=intent_metadata["intent_payload_sha256"],
        metadata=intent_metadata,
    )
    await artifacts.add(
        session_id=session_id,
        artifact_type=receipt.artifact_type,
        path=receipt.path,
        sha256=receipt.sha256,
        metadata=receipt_metadata,
    )

    assert await WorkerService(
        jobs=jobs, worker_id="previous-revision-terminal", pipeline=pipeline
    ).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert publisher.calls == 1
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("delete", "replace", "same_size_substitute"))
async def test_finalize_revalidates_bound_files_without_mutating_prepared_intent(
    tmp_path: Path,
    mutation: str,
) -> None:
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path, FakePublisher()
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await pipeline.run(job, interrupt_before_publish)
    manifest = tuple(await pipeline._publication_manifest(session_id))
    prepared = await artifacts.prepare_discord_publication(
        session_id=session_id,
        pipeline_version=4,
        outcome_pipeline_revision=_OUTCOME_PIPELINE_REVISION,
        speech_act_policy_revision=SPEECH_ACT_POLICY_REVISION,
        publication_contract_revision="task11-one-file-v1",
        attachments=manifest,
    )
    target = Path(manifest[0]["path"])
    original = await asyncio.to_thread(target.read_bytes)
    if mutation == "delete":
        await asyncio.to_thread(target.unlink)
    elif mutation == "replace":
        replacement = target.with_name("replacement.tmp")
        replacement.write_bytes(original + b"replacement")
        replacement.replace(target)
    else:
        assert original
        await asyncio.to_thread(target.write_bytes, bytes([original[0] ^ 1]) + original[1:])

    with pytest.raises(PublicationReconciliationRequired):
        await artifacts.finalize_discord_publication(
            session_id=session_id,
            selected_channel_id=8100,
            message_id="1001",
        )
    retained = await artifacts.first_of_type(
        session_id=session_id, artifact_type="DISCORD_PUBLICATION_INTENT"
    )
    assert retained is not None
    assert retained.id == prepared.record.id
    assert retained.sha256 == prepared.record.sha256
    assert retained.metadata["state"] == "PREPARED"
    assert not await artifacts.all_of_type(
        session_id=session_id, artifact_type="DISCORD_PUBLICATION"
    )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prepare_rejects_cross_session_absolute_attachment_even_with_matching_row(
    tmp_path: Path,
) -> None:
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path, FakePublisher()
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await pipeline.run(job, interrupt_before_publish)
    manifest = list(await pipeline._publication_manifest(session_id))
    source = Path(manifest[0]["path"])
    foreign = (tmp_path / "foreign-session" / "meeting" / source.name).resolve()
    foreign.parent.mkdir(parents=True)
    await asyncio.to_thread(foreign.write_bytes, await asyncio.to_thread(source.read_bytes))
    forged = {**manifest[0], "path": str(foreign)}
    await artifacts.add(
        session_id=session_id,
        artifact_type=str(forged["artifact_type"]),
        path=foreign,
        sha256=str(forged["sha256"]),
        metadata={
            "pipeline_version": 4,
            "outcome_pipeline_revision": _OUTCOME_PIPELINE_REVISION,
            "speech_act_policy_revision": SPEECH_ACT_POLICY_REVISION,
            "size_bytes": forged["size_bytes"],
        },
    )
    manifest[0] = forged

    with pytest.raises(PublicationReconciliationRequired):
        await artifacts.prepare_discord_publication(
            session_id=session_id,
            pipeline_version=4,
            outcome_pipeline_revision=_OUTCOME_PIPELINE_REVISION,
            speech_act_policy_revision=SPEECH_ACT_POLICY_REVISION,
            publication_contract_revision="task11-one-file-v1",
            attachments=tuple(manifest),
        )
    assert not await artifacts.all_of_type(
        session_id=session_id, artifact_type="DISCORD_PUBLICATION_INTENT"
    )
    await database.close()
