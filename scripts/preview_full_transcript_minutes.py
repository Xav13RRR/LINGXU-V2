"""Run one isolated full-transcript DeepSeek minutes preview."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from lingxu.config import Settings
from lingxu.storage.session_files import atomic_write_text
from lingxu.worker.full_transcript_preview import (
    EVIDENCE_NORMALIZATION_REVISION,
    PREVIEW_SCHEMA_VERSION,
    TRANSCRIPT_RECONCILIATION_REVISION,
    DeepSeekFullTranscriptPreviewClient,
    FullTranscriptPreviewError,
    TranscriptReconciliation,
    compose_authoritative_minutes,
    load_exported_authoritative_meeting_facts,
    load_preview_template,
    load_speaker_names,
    normalize_generated_evidence,
    prepare_authoritative_meeting_facts,
    prepare_full_transcript,
    reconcile_generated_transcript,
    render_public_minutes,
    request_sha256,
    validate_generated_minutes,
    validate_public_minutes,
)
from lingxu.worker.structured_minutes import (
    STRUCTURED_MINUTES_REVISION,
    build_structured_messages,
    compose_structured_minutes,
)
from lingxu.worker.topics import (
    APPROVED_TOPIC_BASE_URL,
    APPROVED_TOPIC_MODEL,
    APPROVED_TOPIC_MODEL_DIGEST,
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _safe_output_dir(path: Path) -> None:
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise FullTranscriptPreviewError("preview_output_not_empty")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise FullTranscriptPreviewError("preview_output_invalid")


async def _run(args: argparse.Namespace) -> dict[str, object]:
    if (args.agenda_metadata is None) != (args.events is None):
        raise FullTranscriptPreviewError("authoritative_fact_inputs_incomplete")
    speaker_names = load_speaker_names(args.speaker_metadata)
    prepared = prepare_full_transcript(args.transcript, speaker_names=speaker_names)
    authoritative_facts = (
        load_exported_authoritative_meeting_facts(args.agenda_metadata, args.events)
        if args.agenda_metadata is not None and args.events is not None
        else prepare_authoritative_meeting_facts()
    )
    template, template_sha256 = load_preview_template(args.template)
    messages = build_structured_messages(
        prepared=prepared,
        authoritative_facts=authoritative_facts,
    )
    settings = Settings()
    api_key = settings.summary_llm_api_key
    if (
        not settings.summary_llm_enabled
        or settings.summary_llm_base_url.rstrip("/") != APPROVED_TOPIC_BASE_URL
        or settings.summary_llm_model != APPROVED_TOPIC_MODEL
        or settings.summary_llm_model_digest != APPROVED_TOPIC_MODEL_DIGEST
        or not settings.summary_llm_allow_remote
        or api_key is None
        or not api_key.get_secret_value()
    ):
        raise FullTranscriptPreviewError("approved_model_policy_failed")

    started_at = datetime.now(UTC)
    client = DeepSeekFullTranscriptPreviewClient(
        api_key=api_key.get_secret_value(),
        base_url=settings.summary_llm_base_url,
        model=settings.summary_llm_model,
        expected_model_digest=settings.summary_llm_model_digest,
        timeout_seconds=settings.summary_llm_timeout_seconds,
        max_output_tokens=settings.summary_final_minutes_max_output_tokens,
        allow_remote=settings.summary_llm_allow_remote,
        output_format="json_object",
    )
    result = await client.generate(messages)
    completed_at = datetime.now(UTC)
    validation_failure: str | None = None
    reconciliation: TranscriptReconciliation | None = None
    evidence_audit: dict[str, object] | None = None
    evidence_normalization = None
    validated_minutes_sha256: str | None = None
    validation_details: dict[str, str | int] | None = None
    summary_bindings: list[dict[str, object]] | None = None
    content = result.content
    try:
        if result.status != "complete" or result.finish_reason != "stop":
            raise FullTranscriptPreviewError("final_minutes_generation_incomplete")
        draft = compose_structured_minutes(
            result.content, prepared=prepared, authoritative_facts=authoritative_facts
        )
        summary_bindings = draft.summary_bindings
        content = compose_authoritative_minutes(
            draft.content,
            expected_speaker_labels=prepared.expected_speaker_labels,
            authoritative_facts=authoritative_facts,
        )
        content, reconciliation = reconcile_generated_transcript(
            content,
            prepared=prepared,
        )
        content, evidence_normalization = normalize_generated_evidence(
            content,
            expected_speaker_labels=prepared.expected_speaker_labels,
            expected_source_speaker_labels=prepared.source_speaker_labels,
            authoritative_facts=authoritative_facts,
        )
        validate_generated_minutes(
            content,
            expected_speaker_labels=prepared.expected_speaker_labels,
            expected_source_speaker_labels=prepared.source_speaker_labels,
            authoritative_facts=authoritative_facts,
        )
        validated_minutes_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        content, evidence_audit = render_public_minutes(content)
        validate_public_minutes(
            content,
            evidence_audit=evidence_audit,
            expected_speaker_labels=prepared.expected_speaker_labels,
            expected_source_speaker_labels=prepared.source_speaker_labels,
            authoritative_facts=authoritative_facts,
        )
    except FullTranscriptPreviewError as exc:
        validation_failure = exc.code
        validation_details = exc.validation_details.as_dict() if exc.validation_details else None
    _safe_output_dir(args.output_dir)
    first_heading = next(
        (line for line in template.splitlines() if line.startswith("# ")),
        "",
    )
    template_heading_matched = content.startswith(first_heading)
    accepted = result.status == "complete" and validation_failure is None
    output_name = (
        "minutes.md"
        if accepted
        else (
            "internal.structured.invalid.json"
            if summary_bindings is None
            else "internal.annotated.invalid.md"
        )
    )
    output_path = args.output_dir / output_name
    output_sha256 = atomic_write_text(output_path, content)
    evidence_audit_sha256 = None
    if accepted and evidence_audit is not None:
        evidence_audit_sha256 = atomic_write_text(
            args.output_dir / "evidence_audit.private.json",
            f"{_json(evidence_audit)}\n",
        )
    metadata: dict[str, object] = {
        "structured_minutes_revision": STRUCTURED_MINUTES_REVISION,
        "output_format": "json_object",
        "summary_bindings": summary_bindings,
        "validation_details": validation_details,
        "schema_version": PREVIEW_SCHEMA_VERSION,
        "status": "complete" if accepted else "failed",
        "model": APPROVED_TOPIC_MODEL,
        "thinking": "high",
        "max_output_tokens": client.max_output_tokens,
        "transcript_reconciliation_revision": TRANSCRIPT_RECONCILIATION_REVISION,
        "evidence_normalization_revision": EVIDENCE_NORMALIZATION_REVISION,
        "evidence_normalization": (
            evidence_normalization.as_dict() if evidence_normalization is not None else None
        ),
        "transcript_reconciliation": (
            reconciliation.as_dict() if reconciliation is not None else None
        ),
        "automatic_retries": 0,
        "model_inventory_requests": 1,
        "inference_requests": 1,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": round((completed_at - started_at).total_seconds(), 3),
        "row_count": prepared.row_count,
        "raw_text_chars": prepared.raw_text_chars,
        "cleaned_text_chars": prepared.cleaned_text_chars,
        "removed_japanese_chars": prepared.removal_counts.japanese,
        "removed_korean_chars": prepared.removal_counts.korean,
        "removed_emoji_chars": prepared.removal_counts.emoji,
        "removed_asr_control_chars": prepared.removal_counts.asr_control,
        "source_sha256": prepared.source_sha256,
        "cleaned_payload_sha256": prepared.cleaned_payload_sha256,
        "authoritative_facts_sha256": authoritative_facts.payload_sha256,
        "authoritative_participant_count": authoritative_facts.participant_count,
        "authoritative_agenda_count": authoritative_facts.agenda_count,
        "authoritative_speaking_plan_count": authoritative_facts.speaking_plan_count,
        "authoritative_mute_enforcement_count": authoritative_facts.mute_enforcement_count,
        "authoritative_vote_count": authoritative_facts.vote_count,
        "template_sha256": template_sha256,
        "request_sha256": request_sha256(
            messages,
            max_output_tokens=client.max_output_tokens,
            output_format=client.output_format,
        ),
        "finish_reason": result.finish_reason,
        "response_content_bytes": len(content.encode("utf-8")),
        "provider_response_content_bytes": result.response_content_bytes,
        "output_file": output_name,
        "output_sha256": output_sha256,
        "validated_minutes_sha256": validated_minutes_sha256,
        "evidence_audit_file": ("evidence_audit.private.json" if evidence_audit_sha256 else None),
        "evidence_audit_sha256": evidence_audit_sha256,
        "narrative_evidence_bindings": (
            evidence_audit.get("narrative_binding_count") if evidence_audit else None
        ),
        "action_evidence_bindings": (
            evidence_audit.get("action_binding_count") if evidence_audit else None
        ),
        "transcript_evidence_bindings": (
            evidence_audit.get("transcript_binding_count") if evidence_audit else None
        ),
        "template_heading_matched": template_heading_matched,
        "speaker_metadata_used": speaker_names is not None,
        "speaker_identity_count": len(prepared.expected_speaker_labels),
        "output_contract_valid": validation_failure is None,
        "output_contract_failure": validation_failure,
        "usage": result.usage,
    }
    metadata_sha256 = atomic_write_text(args.output_dir / "metadata.json", f"{_json(metadata)}\n")
    return {
        "status": "complete" if accepted else "failed",
        "model": APPROVED_TOPIC_MODEL,
        "row_count": prepared.row_count,
        "authoritative_agenda_count": authoritative_facts.agenda_count,
        "authoritative_vote_count": authoritative_facts.vote_count,
        "removed_chars": prepared.removal_counts.total,
        "response_content_bytes": result.response_content_bytes,
        "finish_reason": result.finish_reason,
        "template_heading_matched": template_heading_matched,
        "output_contract_valid": validation_failure is None,
        "output_sha256": output_sha256,
        "metadata_sha256": metadata_sha256,
        "usage": result.usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--speaker-metadata", type=Path)
    parser.add_argument("--agenda-metadata", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = asyncio.run(_run(args))
    except FullTranscriptPreviewError as exc:
        print(_json({"failure_code": exc.code, "status": "failed"}))
        raise SystemExit(2) from None
    except Exception:
        print(_json({"failure_code": "preview_unexpected_error", "status": "failed"}))
        raise SystemExit(2) from None
    print(_json(payload))
    if payload["status"] != "complete" or not payload["template_heading_matched"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
