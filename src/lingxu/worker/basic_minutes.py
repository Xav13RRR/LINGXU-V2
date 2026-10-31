"""Deterministic, explicitly limited minutes when final model output is unavailable."""

from __future__ import annotations

import hashlib
import html

from lingxu.worker.full_transcript_preview import (
    FULL_TRANSCRIPT_HEADING,
    REQUIRED_FINAL_HEADINGS,
    SYSTEM_NAME,
    PreparedAuthoritativeMeetingFacts,
    PreparedFullTranscript,
    compose_authoritative_minutes,
    reconcile_generated_transcript,
    render_public_minutes,
    validate_generated_minutes,
    validate_public_minutes,
)

BASIC_MINUTES_REVISION = hashlib.sha256(b"lingxu-basic-minutes-v1-source-excerpts").hexdigest()
BASIC_MINUTES_NOTICE = (
    "基础纪要: 智能整理未完成, 以下内容由已保存的会议记录自动生成。"
    "议程、投票和发言流程来自会议事实; 发言摘录保留原始识别措辞, 可能含听辨错误。"
    "决策、共识、负责人和截止时间需人工核对, 不能把未整理视为没有发生。"
)


def _excerpt(value: str) -> str:
    compact = " ".join(value.split())
    compact = compact[:240] + ("…(摘录)" if len(compact) > 240 else "")
    escaped = html.escape(compact, quote=False)
    for character in "`*_[]{}":
        escaped = escaped.replace(character, f"&#{ord(character)};")
    return escaped


def render_basic_minutes(
    prepared: PreparedFullTranscript,
    facts: PreparedAuthoritativeMeetingFacts,
) -> tuple[str, dict[str, object]]:
    """Render only source excerpts and authoritative facts, then validate the public artifact."""

    humans = tuple(row for row in prepared.source_rows if row.speaker_label != f"[{SYSTEM_NAME}]")
    count = min(24, len(humans))
    indexes = (
        tuple(index * (len(humans) - 1) // (count - 1) for index in range(count))
        if count > 1
        else tuple(range(count))
    )
    excerpts = [
        "以下为沿时间线选取的发言摘录, 供定位讨论; 不代表全部重点或已确认结论。完整记录见文末。"
    ]
    for index in indexes:
        row = humans[index]
        minutes, seconds = divmod(row.started_ms // 1_000, 60)
        excerpts.append(
            f"  - [{minutes:02d}:{seconds:02d}] {row.speaker_label} 原始发言摘录: "
            f"{_excerpt(row.text)} <!-- evidence_ordinals:{row.ordinal} -->"
        )
    if not humans:
        excerpts.append("没有可用于纪要的已授权真人发言记录; 请参照已记录的流程和投票事实。")
    bodies = {
        "## 会议概览": "- 参会者: 待填充\n- 文档类型: 基础纪要(需人工核对)",
        "## 执行摘要": (
            f"{BASIC_MINUTES_NOTICE}\n\n"
            f"已保留 {facts.agenda_count} 项议程、{facts.vote_count} 项已完成投票、"
            f"{facts.speaking_row_count} 条受控发言记录, 以及 {len(humans)} 条真人发言。"
        ),
        "## 议题与讨论": "\n".join(excerpts),
        FULL_TRANSCRIPT_HEADING: "基础纪要保留来源记录, 未进行模型纠错或去重。",
    }
    sections = [REQUIRED_FINAL_HEADINGS[0]]
    for heading in REQUIRED_FINAL_HEADINGS[1:]:
        sections.extend([heading, bodies.get(heading, "本节未完成智能整理, 请人工核对原始记录。")])
    annotated = compose_authoritative_minutes(
        "\n\n".join(sections) + "\n",
        expected_speaker_labels=prepared.expected_speaker_labels,
        authoritative_facts=facts,
    )
    annotated, reconciliation = reconcile_generated_transcript(annotated, prepared=prepared)
    validate_generated_minutes(
        annotated,
        expected_speaker_labels=prepared.expected_speaker_labels,
        expected_source_speaker_labels=prepared.source_speaker_labels,
        authoritative_facts=facts,
    )
    content, evidence_audit = render_public_minutes(annotated)
    validate_public_minutes(
        content,
        evidence_audit=evidence_audit,
        expected_speaker_labels=prepared.expected_speaker_labels,
        expected_source_speaker_labels=prepared.source_speaker_labels,
        authoritative_facts=facts,
    )
    return content, {
        "generation_method": "deterministic_basic",
        "basic_minutes_revision": BASIC_MINUTES_REVISION,
        "status": "fallback",
        "finish_reason": "not_applicable",
        "response_content_bytes": len(content.encode("utf-8")),
        "minutes_markdown_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "validated_minutes_sha256": hashlib.sha256(annotated.encode("utf-8")).hexdigest(),
        "evidence_audit": evidence_audit,
        "transcript_reconciliation": reconciliation.as_dict(),
    }
