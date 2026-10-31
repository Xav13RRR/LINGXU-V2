"""Deterministic one-file renderer for evidence-linked Meeting minutes."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict

from lingxu.transcript.models import CanonicalTranscript, CanonicalTranscriptEntry
from lingxu.worker.full_transcript_preview import (
    MUTE_ENFORCEMENT_HEADING,
    MUTE_ENFORCEMENT_TABLE_HEADER,
    MUTE_ENFORCEMENT_TABLE_SEPARATOR,
    SPEAKING_FACTS_HEADING,
    SPEAKING_PLAN_HEADING,
    SPEAKING_PLAN_TABLE_HEADER,
    SPEAKING_PLAN_TABLE_SEPARATOR,
    SPEAKING_TABLE_HEADER,
    SPEAKING_TABLE_SEPARATOR,
    prepare_authoritative_meeting_facts,
)
from lingxu.worker.minutes_models import MinutesDocumentV2, MinutesVote
from lingxu.worker.outcome_merge import MergedMinutesOutcome

_READABLE_EXCLUDED_ENTRY_TYPES = frozenset(
    {"backchannel", "noise", "excluded", "privacy_control", "privacy-control"}
)
MINUTES_RENDERER_REVISION = hashlib.sha256(b"lingxu-minutes-renderer-speaking-plan-v1").hexdigest()
_BIDI_CONTROLS = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)
_MARKDOWN_LITERAL_CHARS = frozenset("\\`*_{}[]()<>#+-.!|")
_AUTOLINK_SCHEME = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*):(?=//)")

_SUMMARY_METHOD_LABELS = {
    "deterministic": "确定性模板 (未使用 LLM)",
    "deterministic_fallback": "确定性安全回退",
    "deepseek_partial": "DeepSeek 部分结果与确定性回退",
    "deepseek": "DeepSeek (结果已通过确定性验证)",
    "local_llm_partial": "历史本地 LLM 部分结果与确定性回退",
    "local_llm": "历史本地 LLM (结果已通过确定性验证)",
}


def _inline(value: object) -> str:
    """Render untrusted content as a one-line non-interactive Markdown literal."""

    sanitized: list[str] = []
    for character in str(value):
        category = unicodedata.category(character)
        if character in _BIDI_CONTROLS or (category == "Cc" and character not in "\r\n\t"):
            sanitized.append("�")
        elif character in "\r\n\t" or category in {"Zl", "Zp"}:
            sanitized.append(" ")
        else:
            sanitized.append(character)
    collapsed = " ".join("".join(sanitized).split())
    escaped_html = collapsed.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    mention_safe = escaped_html.replace("@", "\N{FULLWIDTH COMMERCIAL AT}")
    autolink_safe = _AUTOLINK_SCHEME.sub(
        lambda match: f"{match.group(1)}\N{FULLWIDTH COLON}",
        mention_safe,
    )
    return "".join(
        f"\\{character}" if character in _MARKDOWN_LITERAL_CHARS else character
        for character in autolink_safe
    )


def _renderer_document_payload(document: MinutesDocumentV2) -> dict[str, object]:
    payload = document.model_dump(mode="json")
    provenance = dict(payload["provenance"])
    for field in (
        "renderer_input_sha256",
        "minutes_markdown_sha256",
        "minutes_payload_hash",
    ):
        provenance.pop(field, None)
    payload["provenance"] = provenance
    return payload


def renderer_input_sha256(
    document: MinutesDocumentV2,
    canonical: CanonicalTranscript,
    *,
    labels: Mapping[int, str],
) -> str:
    """Hash the exact renderer facts without persisting their Meeting content."""

    readable_entries = tuple(
        entry
        for entry in canonical.entries
        if entry.entry_type not in _READABLE_EXCLUDED_ENTRY_TYPES
    )
    payload = {
        "minutes_renderer_revision": MINUTES_RENDERER_REVISION,
        "document": _renderer_document_payload(document),
        "canonical_entries": [asdict(entry) for entry in readable_entries],
        "labels": [[user_id, labels[user_id]] for user_id in sorted(labels)],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(max(milliseconds, 0), 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _speaker_label(entry: CanonicalTranscriptEntry, labels: Mapping[int, str]) -> str:
    if entry.speaker_user_id is None:
        return "聆序机器人"
    return _inline(
        labels.get(
            entry.speaker_user_id,
            entry.speaker_name or f"user:{entry.speaker_user_id}",
        )
    )


def _render_outcomes(
    lines: list[str],
    heading: str,
    outcomes: Sequence[MergedMinutesOutcome],
    *,
    unverified: bool = False,
) -> None:
    lines.extend([f"## {heading}", ""])
    if not outcomes:
        lines.extend(["- 无。", ""])
        return
    for outcome in outcomes:
        details: list[str] = []
        if outcome.owner_name is not None:
            details.append(f"负责人: {_inline(outcome.owner_name)}")
        if outcome.deadline is not None:
            details.append(f"截止时间: {_inline(outcome.deadline)}")
        marker = ""
        if unverified:
            marker = " **未验证 / 待确认**"
        elif outcome.requires_confirmation:
            marker = " **待确认**"
        suffix = f" ({'; '.join(details)})" if details else ""
        lines.append(f"- {_inline(outcome.text)}{suffix}{marker}")
    lines.append("")


def _render_votes(lines: list[str], votes: Sequence[MinutesVote]) -> None:
    lines.extend(["## 投票结果", ""])
    if not votes:
        lines.extend(["- 无已完成投票。", ""])
        return
    for vote in votes:
        subject = vote.vote_content or vote.vote_id or "未命名投票"
        lines.append(f"### {_inline(subject)}")
        lines.append("")
        for option, count in vote.counts.items():
            lines.append(f"- {_inline(option)}: {count}")
        eligibility = "未知" if vote.eligible_count is None else str(vote.eligible_count)
        result = "未知" if vote.result is None else _inline(vote.result)
        lines.extend(
            [
                f"- 有效人数: {eligibility}",
                f"- 已响应人数: {vote.responded_count}",
                f"- 最终结果: {result}",
                "",
            ]
        )


def render_minutes(
    document: MinutesDocumentV2,
    canonical: CanonicalTranscript,
    *,
    labels: Mapping[int, str],
) -> str:
    """Render one stable public document from validated facts and canonical entries only."""

    lines = ["# 聆序会议纪要与逐字稿", "", "## 会议概览与参会者", ""]
    attendees = "、".join(_inline(value) for value in document.participants.attendees)
    title = _inline(document.meeting_title) if document.meeting_title else "未设置"
    lines.extend(
        [
            f"- 会议 ID: `{_inline(document.session_id)}`",
            f"- 会议标题: {title}",
            f"- 主持人: {_inline(document.participants.host)}",
            f"- 参会者: {attendees or '无其他已授权参会者'}",
            f"- 生成时间: {document.generated_at.isoformat()}",
            f"- 生成方式: {_SUMMARY_METHOD_LABELS[document.summary_method]}",
        ]
    )
    if document.summary_provider is not None:
        lines.append(f"- 总结提供方: {_inline(document.summary_provider)}")
    if document.summary_model is not None:
        lines.append(f"- 总结模型: {_inline(document.summary_model)}")
    if document.fallback_reasons:
        lines.append(
            "- 安全回退原因: " + "、".join(_inline(reason) for reason in document.fallback_reasons)
        )

    lines.extend(["", "## 执行摘要", ""])
    lines.append(
        _inline(document.executive_summary)
        if document.executive_summary
        else "当前无经过验证的执行摘要; 请参阅下方结构化结果与完整逐字稿。"
    )
    lines.extend(["", "### 会议完整总结", ""])
    lines.extend(
        [
            (
                _inline(document.full_summary)
                if document.full_summary
                else "当前没有可形成完整总结的已验证会议事实。"
            ),
            "以下结构化结果、权威投票和规范逐字稿共同构成完整会议记录。",
            "",
            "## 议程与议题摘要",
            "",
        ]
    )
    topics_by_id = {topic.topic_id: topic for topic in document.topics}
    rendered_topic_ids: set[str] = set()
    if not document.agendas:
        lines.extend(["- 本次会议未设置议程。", ""])
    else:
        for agenda in document.agendas:
            duration = (
                f"{agenda.duration_minutes} 分钟"
                if agenda.duration_minutes is not None
                else "无固定时长"
            )
            lines.extend(
                [
                    f"### {agenda.position}. {_inline(agenda.title)}",
                    "",
                    f"- 状态: {_inline(agenda.status)}",
                    f"- 语言: {_inline(agenda.language)}",
                    f"- 时长: {duration}",
                    f"- 需要投票: {'是' if agenda.requires_vote else '否'}",
                ]
            )
            if agenda.requires_vote:
                vote_content = (
                    _inline(agenda.vote_content) if agenda.vote_content is not None else "未知"
                )
                vote_duration = (
                    f"{agenda.vote_duration_seconds} 秒"
                    if agenda.vote_duration_seconds is not None
                    else "未知"
                )
                lines.extend(
                    [
                        f"- 投票内容: {vote_content}",
                        f"- 投票时长: {vote_duration}",
                    ]
                )
            agenda_summary = _inline(agenda.summary) if agenda.summary else "无经过验证的议程摘要。"
            lines.append(f"- 议程摘要: {agenda_summary}")
            for topic_id in agenda.topic_ids:
                topic = topics_by_id.get(topic_id)
                if topic is None:
                    continue
                rendered_topic_ids.add(topic_id)
                summary = _inline(topic.summary) if topic.summary else "无经过验证的议题摘要。"
                lines.extend(
                    [
                        f"- 议题: {_inline(topic.title)}",
                        f"  - 时间: {_timestamp(topic.started_ms)}-{_timestamp(topic.ended_ms)}",
                        f"  - 方法: {_inline(topic.method)}",
                        f"  - 摘要: {summary}",
                    ]
                )
            lines.append("")
    remaining_topics = tuple(
        topic for topic in document.topics if topic.topic_id not in rendered_topic_ids
    )
    if remaining_topics:
        lines.extend(["### 其他议题", ""])
        for topic in remaining_topics:
            summary = _inline(topic.summary) if topic.summary else "无经过验证的议题摘要。"
            lines.extend(
                [
                    f"- 议题: {_inline(topic.title)}",
                    f"  - 时间: {_timestamp(topic.started_ms)}-{_timestamp(topic.ended_ms)}",
                    f"  - 方法: {_inline(topic.method)}",
                    f"  - 摘要: {summary}",
                ]
            )
        lines.append("")

    authoritative_facts = prepare_authoritative_meeting_facts(
        agendas=document.agendas,
        votes=document.votes,
        speaking_plans=document.speaking_plans,
        mute_enforcement_summaries=document.mute_enforcement_summaries,
        speaking_turns=document.speaking_turns,
        speaking_followups=document.speaking_followups,
    )
    lines.extend(
        [
            SPEAKING_PLAN_HEADING,
            "",
            SPEAKING_PLAN_TABLE_HEADER,
            SPEAKING_PLAN_TABLE_SEPARATOR,
            *authoritative_facts.speaking_plan_markdown_rows,
            "",
            MUTE_ENFORCEMENT_HEADING,
            "",
            MUTE_ENFORCEMENT_TABLE_HEADER,
            MUTE_ENFORCEMENT_TABLE_SEPARATOR,
            *authoritative_facts.mute_enforcement_markdown_rows,
            "",
            SPEAKING_FACTS_HEADING,
            "",
            SPEAKING_TABLE_HEADER,
            SPEAKING_TABLE_SEPARATOR,
            *authoritative_facts.speaking_markdown_rows,
            "",
        ]
    )

    _render_outcomes(lines, "已确认决策", document.decisions)
    _render_outcomes(lines, "已确认共识", document.agreements)
    _render_outcomes(lines, "行动事项", document.actions)
    _render_outcomes(lines, "开放问题", document.questions)
    _render_outcomes(lines, "提案", document.proposals)
    _render_outcomes(lines, "风险", document.risks)
    _render_outcomes(lines, "分歧", document.disagreements)
    _render_outcomes(lines, "延期事项", document.deferred_items)
    _render_outcomes(lines, "未验证的发言者主张", document.speaker_claims, unverified=True)
    _render_votes(lines, document.votes)

    counts = document.deterministic_counts
    evidence = document.evidence_coverage
    metrics = document.canonical_metrics
    privacy = document.privacy_exclusions
    lines.extend(
        [
            "## 确定性统计与质量说明",
            "",
            f"- 议程总数: {counts.agenda_total}",
            f"- 已完成议程: {counts.agenda_completed}",
            f"- 参会者逐字稿条目: {counts.participant_transcript_entries}",
            f"- 机器人播报条目: {counts.bot_tts_entries}",
            f"- 已完成投票: {counts.closed_votes}",
            f"- 议题总数: {counts.topic_total}",
            f"- 已验证结果: {counts.validated_outcomes}",
            f"- 议程流程配置: {counts.agenda_speaking_plans}",
            f"- 静音执行摘要: {counts.mute_enforcement_summaries}",
            f"- 受控主发言轮次: {counts.controlled_speaking_turns}",
            f"- 受控后续环节: {counts.controlled_speaking_followups}",
            f"- 证据要求项: {evidence.evidence_required}",
            f"- 已链接证据项: {evidence.evidence_linked}",
            f"- 逐字稿证据来源数: {evidence.transcript_sources}",
            f"- 事件证据来源数: {evidence.event_sources}",
            f"- 原始来源条目数: {metrics.source_entry_count}",
            f"- 规范条目数: {metrics.canonical_entry_count}",
            f"- 重复合并数: {metrics.duplicate_entry_count}",
            f"- 回应词条目数: {metrics.backchannel_entry_count}",
            f"- 已移除 ASR 标签数: {metrics.asr_tag_count}",
        ]
    )
    if privacy.measurement_available:
        lines.extend(
            [
                "- 隐私排除计量: 可用",
                f"- 隐私排除逐字稿条目: {privacy.transcript_entries}",
                f"- 隐私排除音频行: {privacy.segment_rows}",
                f"- 隐私排除音频文件: {privacy.segment_files}",
            ]
        )
    else:
        lines.append("- 隐私排除计量: 不可用 (未知)")
    lines.extend(
        [
            "- 质量边界: 会议事件、投票计数、议程状态与发言流程行为为确定性事实; "
            "模型不能覆盖这些事实。",
            "",
            "## 带时间戳逐字稿",
            "",
        ]
    )
    readable_entries = tuple(
        entry
        for entry in canonical.entries
        if entry.entry_type not in _READABLE_EXCLUDED_ENTRY_TYPES
    )
    if readable_entries:
        for entry in readable_entries:
            lines.append(
                f"- [{_timestamp(entry.started_ms)}-{_timestamp(entry.ended_ms)}] "
                f"**{_speaker_label(entry, labels)}**: {_inline(entry.text)}"
            )
    else:
        lines.append("- 没有可公开的规范逐字稿条目。")
    return "\n".join(lines) + "\n"
