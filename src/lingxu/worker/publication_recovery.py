"""Content-safe Discord publication manifest and recovery validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from lingxu.storage.repositories import MeetingSessionRepository
from lingxu.storage.worker_repositories import (
    ArtifactRepository,
    PublicationReconciliationRequired,
    validate_finalized_publication_pair,
    validate_prepared_publication,
)
from lingxu.worker.minutes_models import MinutesDocumentV2
from lingxu.worker.minutes_renderer import MINUTES_RENDERER_REVISION

PUBLICATION_CONTRACT_REVISION = "task11-one-file-v1"
PUBLICATION_ATTACHMENTS = (("MINUTES_MARKDOWN", "minutes.md"),)
LEGACY_TASK10_PUBLICATION_CONTRACT_REVISION = "task10-four-file-v1"
LEGACY_TASK10_PUBLICATION_ATTACHMENTS = (
    ("MINUTES_MARKDOWN", "minutes.md"),
    ("MINUTES_JSON", "minutes.json"),
    ("TRANSCRIPT_TEXT", "transcript.txt"),
    ("EVIDENCE_JSON", "evidence.json"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def build_publication_manifest(
    *,
    artifacts: ArtifactRepository,
    meeting_dir: Path,
    session_id: str,
    outcome_pipeline_revision: str,
    speech_act_policy_revision: str,
    artifact_pipeline_version: int,
    allow_legacy_speech_revision: bool = False,
    publication_contract_revision: str = PUBLICATION_CONTRACT_REVISION,
    pipeline_version: int | None = None,
) -> list[dict[str, Any]]:
    """Validate registered attachment authority and return a content-free manifest."""

    selected_pipeline_version = (
        artifact_pipeline_version if pipeline_version is None else pipeline_version
    )
    attachments: tuple[tuple[str, str], ...]
    if publication_contract_revision == PUBLICATION_CONTRACT_REVISION:
        attachments = PUBLICATION_ATTACHMENTS
    elif publication_contract_revision == LEGACY_TASK10_PUBLICATION_CONTRACT_REVISION:
        attachments = LEGACY_TASK10_PUBLICATION_ATTACHMENTS
    else:
        raise PublicationReconciliationRequired("publication_record_invalid")

    manifest: list[dict[str, Any]] = []
    speech_revision_presence: list[bool] = []
    renderer_revision_presence: list[bool] = []
    for artifact_type, name in attachments:
        path = meeting_dir / name
        record = await artifacts.at_path(
            session_id=session_id,
            artifact_type=artifact_type,
            path=path,
        )
        expected_metadata: dict[str, object] = {
            "pipeline_version": selected_pipeline_version,
            "outcome_pipeline_revision": outcome_pipeline_revision,
            "size_bytes": path.stat().st_size if path.is_file() else -1,
        }
        if artifact_type in {"MINUTES_MARKDOWN", "MINUTES_JSON", "EVIDENCE_JSON"}:
            has_revision = "speech_act_policy_revision" in (record.metadata if record else {})
            speech_revision_presence.append(has_revision)
            if not allow_legacy_speech_revision or has_revision:
                expected_metadata["speech_act_policy_revision"] = speech_act_policy_revision
        if artifact_type in {"MINUTES_MARKDOWN", "MINUTES_JSON"}:
            metadata = record.metadata if record else {}
            has_renderer_revision = "minutes_renderer_revision" in metadata
            has_renderer_input = "renderer_input_sha256" in metadata
            if has_renderer_revision != has_renderer_input:
                raise PublicationReconciliationRequired("publication_record_invalid")
            renderer_revision_presence.append(has_renderer_revision)
            if (
                publication_contract_revision == PUBLICATION_CONTRACT_REVISION
                or has_renderer_revision
            ):
                document = MinutesDocumentV2.model_validate_json(
                    (meeting_dir / "minutes.json").read_text(encoding="utf-8")
                )
                expected_metadata["minutes_renderer_revision"] = MINUTES_RENDERER_REVISION
                expected_metadata["renderer_input_sha256"] = (
                    document.provenance.renderer_input_sha256
                )
        if (
            record is None
            or record.encrypted
            or not path.is_file()
            or record.path != path
            or record.sha256 != _sha256(path)
            or record.metadata != expected_metadata
        ):
            raise PublicationReconciliationRequired("publication_record_invalid")
        manifest.append(
            {
                "artifact_type": artifact_type,
                "path": str(path),
                "sha256": record.sha256,
                "size_bytes": path.stat().st_size,
            }
        )
    if (
        allow_legacy_speech_revision
        and speech_revision_presence
        and any(speech_revision_presence)
        and not all(speech_revision_presence)
    ):
        raise PublicationReconciliationRequired("publication_record_invalid")
    if (
        renderer_revision_presence
        and any(renderer_revision_presence)
        and not all(renderer_revision_presence)
    ):
        raise PublicationReconciliationRequired("publication_record_invalid")
    return manifest


async def publication_is_terminal(
    *,
    sessions: MeetingSessionRepository,
    artifacts: ArtifactRepository,
    meeting_dir: Path,
    session_id: str,
    outcome_pipeline_revision: str,
    speech_act_policy_revision: str,
    artifact_pipeline_version: int,
) -> bool:
    """Validate current or legacy publication authority without republishing."""

    intents = await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION_INTENT",
    )
    receipts = await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION",
    )
    if not intents and not receipts:
        return False
    if len(intents) > 1 or len(receipts) > 1:
        raise PublicationReconciliationRequired("publication_record_invalid")
    session = await sessions.get(session_id)
    if intents:
        intent = intents[0]
        metadata = intent.metadata
        outcome_revision = metadata.get("outcome_pipeline_revision")
        speech_revision = metadata.get("speech_act_policy_revision")
        if not isinstance(outcome_revision, str) or not isinstance(speech_revision, str):
            raise PublicationReconciliationRequired("publication_record_invalid")
        attachments = metadata.get("attachments")
        if not isinstance(attachments, list):
            raise PublicationReconciliationRequired("publication_record_invalid")
        if not receipts:
            try:
                validate_prepared_publication(
                    intent,
                    session_id=session_id,
                    guild_id=session.guild_id,
                    requested_channel_id=session.publication_channel_id,
                )
                await artifacts.validate_discord_publication_files(
                    session_id=session_id,
                    pipeline_version=int(metadata["pipeline_version"]),
                    attachments=tuple(attachments),
                    outcome_pipeline_revision=outcome_revision,
                    speech_act_policy_revision=speech_revision,
                    publication_contract_revision=str(metadata["publication_contract_revision"]),
                )
            except PublicationReconciliationRequired:
                raise PublicationReconciliationRequired("publication_record_invalid") from None
            raise PublicationReconciliationRequired("publication_prepared_unresolved")
        try:
            validate_finalized_publication_pair(
                intent,
                receipts[0],
                session_id=session_id,
                guild_id=session.guild_id,
                requested_channel_id=session.publication_channel_id,
            )
            await artifacts.validate_discord_publication_files(
                session_id=session_id,
                pipeline_version=int(metadata["pipeline_version"]),
                attachments=tuple(attachments),
                outcome_pipeline_revision=outcome_revision,
                speech_act_policy_revision=speech_revision,
                publication_contract_revision=str(metadata["publication_contract_revision"]),
            )
        except PublicationReconciliationRequired:
            raise PublicationReconciliationRequired("publication_record_invalid") from None
        return True

    if len(receipts) != 1:
        raise PublicationReconciliationRequired("publication_record_invalid")
    manifest: list[dict[str, Any]] | None = None
    for pipeline_version in (3, artifact_pipeline_version):
        try:
            manifest = await build_publication_manifest(
                artifacts=artifacts,
                meeting_dir=meeting_dir,
                session_id=session_id,
                outcome_pipeline_revision=outcome_pipeline_revision,
                speech_act_policy_revision=speech_act_policy_revision,
                artifact_pipeline_version=artifact_pipeline_version,
                allow_legacy_speech_revision=True,
                publication_contract_revision=LEGACY_TASK10_PUBLICATION_CONTRACT_REVISION,
                pipeline_version=pipeline_version,
            )
            break
        except (OSError, ValueError, PublicationReconciliationRequired):
            continue
    if manifest is None:
        raise PublicationReconciliationRequired("publication_record_invalid")
    receipt = receipts[0]
    metadata = receipt.metadata
    if set(metadata) != {"channel_id", "message_id"}:
        raise PublicationReconciliationRequired("publication_record_invalid")
    channel_id = metadata.get("channel_id")
    message_id = metadata.get("message_id")
    if (
        not isinstance(channel_id, int)
        or isinstance(channel_id, bool)
        or channel_id <= 0
        or not isinstance(message_id, str)
        or not message_id.isdecimal()
        or int(message_id) <= 0
        or receipt.path != Path(f"discord-{channel_id}-{message_id}")
        or receipt.sha256 != hashlib.sha256(message_id.encode()).hexdigest()
        or receipt.encrypted
        or (
            session.publication_channel_id is not None
            and channel_id != session.publication_channel_id
        )
    ):
        raise PublicationReconciliationRequired("publication_record_invalid")
    return bool(manifest)
