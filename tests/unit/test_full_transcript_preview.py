import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from lingxu.domain.models import TranscriptEntryRecord
from lingxu.resources import deepseek_full_transcript_minutes_template
from lingxu.worker.full_transcript_preview import (
    AGENDA_TABLE_HEADER,
    AGENDA_TABLE_SEPARATOR,
    MUTE_ENFORCEMENT_TABLE_HEADER,
    MUTE_ENFORCEMENT_TABLE_SEPARATOR,
    PREVIEW_SCHEMA_VERSION,
    SPEAKING_PLAN_TABLE_HEADER,
    SPEAKING_PLAN_TABLE_SEPARATOR,
    SPEAKING_TABLE_HEADER,
    SPEAKING_TABLE_SEPARATOR,
    VOTE_TABLE_HEADER,
    VOTE_TABLE_SEPARATOR,
    DeepSeekFullTranscriptPreviewClient,
    FullTranscriptFailureDiagnostics,
    FullTranscriptPreviewError,
    build_preview_messages,
    build_preview_request,
    compose_authoritative_minutes,
    load_exported_authoritative_meeting_facts,
    load_preview_template,
    load_speaker_names,
    normalize_generated_evidence,
    prepare_authoritative_meeting_facts,
    prepare_full_transcript,
    prepare_full_transcript_records,
    reconcile_generated_transcript,
    render_public_minutes,
    request_sha256,
    strip_japanese_korean_and_emoji,
    validate_generated_minutes,
    validate_public_minutes,
)
from lingxu.worker.minutes_models import (
    MinutesAgenda,
    MinutesSpeakingFollowup,
    MinutesSpeakingPlan,
    MinutesSpeakingPlanParticipant,
    MinutesSpeakingPlanTurn,
    MinutesSpeakingTurn,
    MinutesVote,
)
from lingxu.worker.topics import APPROVED_TOPIC_MODEL


def _write_transcript(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "source-1",
                    "user_id": 10,
                    "source_key": "speaker:10",
                    "started_ms": 0,
                    "ended_ms": 800,
                    "raw_text": "重复原文😀カナ한글",
                    "normalised_text": "本地规范文本不得使用",
                },
                {
                    "id": "source-2",
                    "user_id": 10,
                    "source_key": "speaker:10",
                    "started_ms": 900,
                    "ended_ms": 1_700,
                    "raw_text": "重复原文😀カナ한글",
                    "normalised_text": "另一条本地规范文本不得使用",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_strip_japanese_korean_emoji_preserves_chinese_latin_and_duplicates() -> None:
    cleaned, counts = strip_japanese_korean_and_emoji("中文ABCかなカナ한글😀❤️1️⃣中文ABC")

    assert cleaned == "中文ABC中文ABC"
    assert counts.japanese == 4
    assert counts.korean == 2
    assert counts.emoji == 6
    assert counts.total == 12


def test_prepare_removes_asr_control_tokens_before_deepseek() -> None:
    prepared = prepare_full_transcript_records(
        (
            TranscriptEntryRecord(
                id="human-1",
                session_id="session-1",
                source_key="speaker:10",
                user_id=10,
                raw_text="<|zh|><|NEUTRAL|><|Speech|><|withitn|>我同意。",
                normalised_text=None,
                started_ms=0,
                ended_ms=1_000,
            ),
        ),
        speaker_names={10: "Alice"},
    )

    assert "<|" not in prepared.payload
    assert json.loads(prepared.payload)[0]["text"] == "我同意。"
    assert prepared.source_rows[0].text == "我同意。"
    assert prepared.removal_counts.asr_control == 38


def test_prepare_uses_every_raw_row_without_local_deduplication(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.json"
    _write_transcript(transcript)

    prepared = prepare_full_transcript(transcript)
    rows = json.loads(prepared.payload)

    assert prepared.row_count == 2
    assert [row["text"] for row in rows] == ["重复原文", "重复原文"]
    assert [row["ordinal"] for row in rows] == [0, 1]
    assert [row["speaker_label"] for row in rows] == ["[user:10]", "[user:10]"]
    assert "本地规范文本不得使用" not in prepared.payload
    assert prepared.removal_counts.japanese == 4
    assert prepared.removal_counts.korean == 4
    assert prepared.removal_counts.emoji == 2


def test_request_is_thinking_high_and_binds_final_output_limit(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.json"
    template_path = tmp_path / "template.md"
    _write_transcript(transcript)
    template_path.write_text("# 聆序会议纪要\n\n{{内容}}\n", encoding="utf-8")
    prepared = prepare_full_transcript(transcript)
    template, _ = load_preview_template(template_path)

    messages = build_preview_messages(prepared=prepared, template=template)
    request = build_preview_request(messages)

    assert request["model"] == APPROVED_TOPIC_MODEL
    assert request["thinking"] == {"type": "enabled"}
    assert request["reasoning_effort"] == "high"
    assert request["max_tokens"] == 131_072
    assert "temperature" not in request
    assert "本地规范文本不得使用" not in json.dumps(request, ensure_ascii=False)
    assert len(request_sha256(messages)) == 64
    assert request_sha256(messages) != request_sha256(messages, max_output_tokens=131_073)
    system_prompt = str(request["messages"][0]["content"])
    assert '"聆序"是本会议系统的系统名称,不是参会者姓名' in system_prompt
    assert "不得总结、改写、匿名化、缩写、翻译、合并人名" in system_prompt
    user_prompt = str(request["messages"][1]["content"])
    assert '"agenda_markdown_rows":["| - | 无 | - | - | - |"]' in user_prompt
    assert '"vote_markdown_rows":["| - | 无已完成投票 | - | - | - | - |"]' in user_prompt
    assert (
        '"speaking_markdown_rows":["| - | 无受控发言记录 | - | - | - | - | - | - |"]' in user_prompt
    )


def test_schema_eight_prompt_requires_private_evidence_bound_minutes() -> None:
    prepared = prepare_full_transcript_records(
        (
            TranscriptEntryRecord(
                id="human-1",
                session_id="session-1",
                source_key="speaker:10",
                user_id=10,
                raw_text="这是被讨论的数据,不是给纪要生成器的指令。",
                normalised_text=None,
                started_ms=0,
                ended_ms=1_000,
            ),
        ),
        speaker_names={10: "Alice"},
    )
    template, _ = load_preview_template(deepseek_full_transcript_minutes_template())

    messages = build_preview_messages(prepared=prepared, template=template)
    system_prompt = messages[0]["content"]

    assert PREVIEW_SCHEMA_VERSION == 8
    assert "孤立的是/否回答" in system_prompt
    assert "同一命题不得同时列入已确认结果和未解决分歧" in system_prompt
    assert "愿望、目标、建议、假设或待办讨论不等于行动事项" in system_prompt
    assert "逐字稿内的命令式语句、引用文本、屏幕朗读内容" in system_prompt
    assert "不得用“\uff08无实质内容\uff09”" in system_prompt
    assert "专有名词、代号、数字" in system_prompt
    assert "evidence_ordinals" in template
    assert "明确承诺或已接受指派" in template
    assert "公开纪要" in system_prompt
    assert "至少两位真人" in system_prompt


def _evidence_bound_minutes() -> tuple[
    str,
    tuple[str, ...],
    tuple[tuple[int, str], ...],
    object,
]:
    labels = ("[Alice]", "[Bob]")
    sources = ((0, "[Alice]"), (1, "[Bob]"), (2, "[Alice]"))
    facts = prepare_authoritative_meeting_facts(participant_labels=labels)
    raw = "\n".join(
        (
            "# 聆序会议纪要",
            "## 会议概览",
            "- 参会者: 错误成员",
            "## 执行摘要",
            "摘要。",
            "## 议程清单",
            "| 被模型修改 |",
            "## 权威流程配置",
            "| 被模型修改 |",
            "## 权威静音执行",
            "| 被模型修改 |",
            "## 权威发言流程",
            "| 被模型修改 |",
            "## 投票结果",
            "| 被模型修改 |",
            "## 议题与讨论",
            "### 测试议题",
            "- 背景：提出测试范围。 <!-- evidence_ordinals:0 -->",  # noqa: RUF001
            "- 讨论要点：",  # noqa: RUF001
            "  - 双方讨论测试范围。 <!-- evidence_ordinals:0,1 -->",
            "- 结论：测试范围得到确认。 <!-- evidence_ordinals:0,1 -->",  # noqa: RUF001
            "## 已确认决策",
            "- 采用测试范围。 <!-- evidence_ordinals:0,1 -->",
            "## 已确认共识",
            "- 双方认可测试范围。 <!-- evidence_ordinals:0,1 -->",
            "## 行动事项",
            "| 行动事项 | 负责人 | 截止时间 | 依据 |",
            "| --- | --- | --- | --- |",
            "| 执行测试 | [Alice] | 未提及 | source_ordinals:2 |",
            "## 开放问题",
            "- 无。",
            "## 提案",
            "- 无。",
            "## 风险",
            "- 无。",
            "## 分歧",
            "- 无。",
            "## 优化去重后的完整逐字稿",
            "- [00:00:00-00:00:01] **[Alice]** <!-- source_ordinals:0 -->: 提出测试范围。",
            "- [00:00:01-00:00:02] **[Bob]** <!-- source_ordinals:1 -->: 我同意。",
            "- [00:00:02-00:00:03] **[Alice]** <!-- source_ordinals:2 -->: 我会执行测试。",
        )
    )
    content = compose_authoritative_minutes(
        raw,
        expected_speaker_labels=labels,
        authoritative_facts=facts,
    )
    return content, labels, sources, facts


def test_public_minutes_strip_ordinals_and_keep_closed_private_audit() -> None:
    content, labels, sources, facts = _evidence_bound_minutes()

    validate_generated_minutes(
        content,
        expected_speaker_labels=labels,
        expected_source_speaker_labels=sources,
        authoritative_facts=facts,
    )
    public, audit = render_public_minutes(content)

    assert "evidence_ordinals" not in public
    assert "source_ordinals" not in public
    assert "<!--" not in public
    assert "| 行动事项 | 负责人 | 截止时间 |" in public
    assert "| 执行测试 | [Alice] | 未提及 |" in public
    assert audit["narrative_binding_count"] == 5
    assert audit["action_binding_count"] == 1
    assert audit["transcript_binding_count"] == 3
    validate_public_minutes(
        public,
        evidence_audit=audit,
        expected_speaker_labels=labels,
        expected_source_speaker_labels=sources,
        authoritative_facts=facts,
    )


@pytest.mark.parametrize(
    "suffix",
    ("<!-- ordinal_audit:0 -->", "source_ordinals:0"),
)
def test_public_minutes_reject_any_residual_private_annotation(suffix: str) -> None:
    content, labels, sources, facts = _evidence_bound_minutes()
    public, audit = render_public_minutes(content)
    tampered = public.replace("摘要。", f"摘要。 {suffix}")
    audit["public_minutes_sha256"] = hashlib.sha256(tampered.encode("utf-8")).hexdigest()

    with pytest.raises(
        FullTranscriptPreviewError,
        match="public_minutes_ordinal_redaction_invalid",
    ):
        validate_public_minutes(
            tampered,
            evidence_audit=audit,
            expected_speaker_labels=labels,
            expected_source_speaker_labels=sources,
            authoritative_facts=facts,
        )


def test_topic_conclusion_without_internal_evidence_fails_closed() -> None:
    content, labels, sources, facts = _evidence_bound_minutes()
    content = content.replace(
        "- 结论：测试范围得到确认。 <!-- evidence_ordinals:0,1 -->",  # noqa: RUF001
        "- 结论：测试范围得到确认。",  # noqa: RUF001
    )

    with pytest.raises(FullTranscriptPreviewError, match="final_minutes_evidence_invalid"):
        validate_generated_minutes(
            content,
            expected_speaker_labels=labels,
            expected_source_speaker_labels=sources,
            authoritative_facts=facts,
        )


def test_multi_participant_consensus_requires_two_human_evidence_speakers() -> None:
    content, labels, sources, facts = _evidence_bound_minutes()
    content = content.replace(
        "- 双方认可测试范围。 <!-- evidence_ordinals:0,1 -->",
        "- 双方认可测试范围。 <!-- evidence_ordinals:0 -->",
    )

    with pytest.raises(
        FullTranscriptPreviewError,
        match="final_minutes_consensus_evidence_invalid",
    ):
        validate_generated_minutes(
            content,
            expected_speaker_labels=labels,
            expected_source_speaker_labels=sources,
            authoritative_facts=facts,
        )


def test_evidence_normalization_sorts_ordinals_and_downgrades_unproven_owner() -> None:
    content, labels, sources, facts = _evidence_bound_minutes()
    content = content.replace(
        "  - 双方讨论测试范围。 <!-- evidence_ordinals:0,1 -->",
        "  - 双方讨论测试范围。 <!-- evidence_ordinals:1,0,1 -->",
    ).replace(
        "| 执行测试 | [Alice] | 未提及 | source_ordinals:2 |",
        "| 执行测试 | [Bob] | 未提及 | source_ordinals:2,2 |",
    )

    normalized, report = normalize_generated_evidence(
        content,
        expected_speaker_labels=labels,
        expected_source_speaker_labels=sources,
        authoritative_facts=facts,
    )

    assert "<!-- evidence_ordinals:0,1 -->" in normalized
    assert "| 执行测试 | 待确认 | 未提及 | source_ordinals:2 |" in normalized
    assert report.as_dict() == {
        "ordinal_lines_normalized": 2,
        "action_owners_downgraded": 1,
    }
    validate_generated_minutes(
        normalized,
        expected_speaker_labels=labels,
        expected_source_speaker_labels=sources,
        authoritative_facts=facts,
    )


def test_evidence_normalization_does_not_invent_unknown_bindings() -> None:
    content, labels, sources, facts = _evidence_bound_minutes()
    content = content.replace(
        "<!-- evidence_ordinals:0 -->",
        "<!-- evidence_ordinals:999 -->",
        1,
    )

    normalized, report = normalize_generated_evidence(
        content,
        expected_speaker_labels=labels,
        expected_source_speaker_labels=sources,
        authoritative_facts=facts,
    )

    assert report.ordinal_lines_normalized == 0
    with pytest.raises(FullTranscriptPreviewError, match="final_minutes_evidence_invalid"):
        validate_generated_minutes(
            normalized,
            expected_speaker_labels=labels,
            expected_source_speaker_labels=sources,
            authoritative_facts=facts,
        )


def test_request_and_output_bind_every_authoritative_agenda_and_vote() -> None:
    facts = prepare_authoritative_meeting_facts(
        agendas=(
            MinutesAgenda(
                agenda_id="agenda-1",
                position=1,
                title="产品发布 | 准备",
                language="zh",
                status="COMPLETED",
                duration_minutes=15,
                requires_vote=True,
                vote_content="是否发布?",
                vote_duration_seconds=60,
                topic_ids=("topic-1",),
                summary=None,
            ),
            MinutesAgenda(
                agenda_id="agenda-2",
                position=2,
                title="风险复盘",
                language="zh",
                status="SKIPPED",
                duration_minutes=None,
                requires_vote=False,
                vote_content=None,
                vote_duration_seconds=None,
                topic_ids=("topic-2",),
                summary=None,
            ),
        ),
        votes=(
            MinutesVote(
                source_event_id=10,
                vote_id="vote-1",
                agenda_item_id="agenda-1",
                vote_content="是否发布?",
                counts={"yes": 2, "no": 1, "abstain": 0},
                eligible_count=3,
                responded_count=3,
                result="是",
            ),
            MinutesVote(
                source_event_id=11,
                vote_id="vote-2",
                agenda_item_id=None,
                vote_content="选择时间",
                counts={"yes": 1, "no": 0, "abstain": 1},
                eligible_count=2,
                responded_count=1,
                result="是",
            ),
        ),
    )
    prepared = prepare_full_transcript_records(
        (
            TranscriptEntryRecord(
                id="human-1",
                session_id="session-1",
                source_key="speaker:10",
                user_id=10,
                raw_text="讨论内容",
                normalised_text=None,
                started_ms=0,
                ended_ms=1_000,
            ),
        ),
        speaker_names={10: "Discord Alice"},
    )
    messages = build_preview_messages(
        prepared=prepared,
        template="# 聆序会议纪要\n\n{{内容}}\n",
        authoritative_facts=facts,
    )

    assert "<authoritative_meeting_facts_json>" in messages[1]["content"]
    assert '"agenda_count":2' in messages[1]["content"]
    assert '"vote_count":2' in messages[1]["content"]
    assert "actor_user_id" not in facts.payload
    facts_payload = json.loads(facts.payload)
    assert [item["markdown_row"] for item in facts_payload["agendas"]] == list(
        facts.agenda_markdown_rows
    )
    assert [item["markdown_row"] for item in facts_payload["votes"]] == list(
        facts.vote_markdown_rows
    )
    assert facts_payload["agenda_markdown_rows"] == list(facts.agenda_markdown_rows)
    assert facts_payload["vote_markdown_rows"] == list(facts.vote_markdown_rows)

    content = "\n".join(
        (
            "# 聆序会议纪要",
            "## 会议概览",
            "- 参会者: [Discord Alice]",
            "## 执行摘要",
            "## 议程清单",
            AGENDA_TABLE_HEADER,
            AGENDA_TABLE_SEPARATOR,
            *facts.agenda_markdown_rows,
            "## 权威流程配置",
            SPEAKING_PLAN_TABLE_HEADER,
            SPEAKING_PLAN_TABLE_SEPARATOR,
            *facts.speaking_plan_markdown_rows,
            "## 权威静音执行",
            MUTE_ENFORCEMENT_TABLE_HEADER,
            MUTE_ENFORCEMENT_TABLE_SEPARATOR,
            *facts.mute_enforcement_markdown_rows,
            "## 权威发言流程",
            SPEAKING_TABLE_HEADER,
            SPEAKING_TABLE_SEPARATOR,
            *facts.speaking_markdown_rows,
            "## 投票结果",
            VOTE_TABLE_HEADER,
            VOTE_TABLE_SEPARATOR,
            *facts.vote_markdown_rows,
            "## 议题与讨论",
            "## 已确认决策",
            "## 已确认共识",
            "## 行动事项",
            "## 开放问题",
            "## 提案",
            "## 风险",
            "## 分歧",
            "## 优化去重后的完整逐字稿",
            "- [00:00:00-00:00:01] **[Discord Alice]**: 讨论内容",
        )
    )
    validate_generated_minutes(
        content,
        expected_speaker_labels=("[Discord Alice]",),
        authoritative_facts=facts,
    )

    with pytest.raises(
        FullTranscriptPreviewError,
        match="final_minutes_authoritative_facts_invalid",
    ):
        validate_generated_minutes(
            content.replace(facts.vote_markdown_rows[1], ""),
            expected_speaker_labels=("[Discord Alice]",),
            authoritative_facts=facts,
        )

    with pytest.raises(
        FullTranscriptPreviewError,
        match="final_minutes_authoritative_facts_invalid",
    ):
        validate_generated_minutes(
            content.replace(facts.speaking_markdown_rows[0], "| 改写 |"),
            expected_speaker_labels=("[Discord Alice]",),
            authoritative_facts=facts,
        )


def test_speaking_rows_follow_agenda_then_turn_and_followup_order() -> None:
    at = datetime(2026, 8, 27, tzinfo=UTC)
    agendas = tuple(
        MinutesAgenda(
            agenda_id=f"agenda-{position}",
            position=position,
            title=f"议程 {position}",
            language="zh",
            status="COMPLETED",
            duration_minutes=10,
            requires_vote=False,
            vote_content=None,
            vote_duration_seconds=None,
            topic_ids=(),
            summary=None,
        )
        for position in (1, 2)
    )

    def turn(agenda_item_id: str, turn_index: int) -> MinutesSpeakingTurn:
        return MinutesSpeakingTurn(
            agenda_item_id=agenda_item_id,
            turn_index=turn_index,
            mode="ORDERED",
            speaker_user_ids=(turn_index + 1,),
            speaker_names=(f"成员 {turn_index + 1}",),
            planned_seconds=30,
            extension_seconds=0,
            started_at=None,
            finished_at=at,
            active_intervals=(),
            pause_intervals=(),
            actual_floor_ms=0,
            finish_reason="HOST_SKIPPED",
            finished_early=False,
            skipped=True,
            absent=False,
            overtime_ms=0,
        )

    followup = MinutesSpeakingFollowup(
        agenda_item_id="agenda-1",
        mode="SUPPLEMENT_TURNS",
        started_at=at,
        finished_at=at,
        standing_user_ids=(),
        standing_names=(),
        participations=(),
    )
    facts = prepare_authoritative_meeting_facts(
        agendas=agendas,
        speaking_turns=(turn("agenda-2", 0), turn("agenda-1", 1), turn("agenda-1", 0)),
        speaking_followups=(followup,),
    )

    assert [
        [cell.strip() for cell in row.strip("|").split("|")][:2]
        for row in facts.speaking_markdown_rows
    ] == [
        ["议程 1", "1"],
        ["议程 1", "2"],
        ["议程 1", "后续"],
        ["议程 2", "1"],
    ]


def test_exported_legacy_minutes_and_closed_vote_build_current_facts(tmp_path: Path) -> None:
    agenda_metadata = tmp_path / "minutes.json"
    events = tmp_path / "events.json"
    agenda_metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "agenda_summaries": [
                    {
                        "agenda_id": "agenda-1",
                        "position": 1,
                        "title": "发布准备",
                        "language": "zh",
                        "status": "COMPLETED",
                        "duration_minutes": 10,
                        "requires_vote": True,
                        "vote_content": "是否发布",
                        "vote_duration_seconds": 60,
                        "summary": {"legacy": "ignored"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    events.write_text(
        json.dumps(
            [
                {
                    "sequence_number": 9,
                    "event_type": "VOTE_CAST",
                    "actor_user_id": 100,
                    "payload": {"option_id": "yes"},
                },
                {
                    "sequence_number": 10,
                    "event_type": "VOTE_CLOSED",
                    "actor_user_id": 200,
                    "payload": {
                        "vote_id": "vote-1",
                        "agenda_item_id": "agenda-1",
                        "vote_content": "是否发布",
                        "counts": {"yes": 2, "no": 0, "abstain": 1},
                        "eligible_count": 3,
                        "responded_count": 2,
                        "result": "yes",
                    },
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    facts = load_exported_authoritative_meeting_facts(agenda_metadata, events)

    assert facts.agenda_count == 1
    assert facts.vote_count == 1
    assert "actor_user_id" not in facts.payload
    assert "user_id" not in facts.payload
    assert "legacy" not in facts.payload


def test_exported_speaking_plan_json_arrays_build_current_facts(tmp_path: Path) -> None:
    agenda_metadata = tmp_path / "minutes.json"
    events = tmp_path / "events.json"
    agenda_metadata.write_text(
        json.dumps(
            {
                "agendas": [
                    {
                        "agenda_id": "agenda-1",
                        "position": 1,
                        "title": "顺序发言",
                        "language": "zh",
                        "status": "COMPLETED",
                        "duration_minutes": 10,
                        "requires_vote": False,
                        "vote_content": None,
                        "vote_duration_seconds": None,
                        "topic_ids": [],
                        "summary": None,
                    }
                ],
                "speaking_plans": [
                    {
                        "policy_schema_version": 1,
                        "agenda_item_id": "agenda-1",
                        "mode": "ORDERED",
                        "order_strategy": "EXPLICIT",
                        "host_participation": "AUTO",
                        "turns": [
                            {
                                "user_id": 100,
                                "name": "Alice (user:100)",
                                "planned_seconds": 120,
                            }
                        ],
                        "default_turn_seconds": None,
                        "presenters": [],
                        "presentation_seconds": None,
                        "followup_mode": "NONE",
                        "mute_enforcement_requested": True,
                        "prep_seconds": 10,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    events.write_text("[]", encoding="utf-8")

    facts = load_exported_authoritative_meeting_facts(agenda_metadata, events)

    assert facts.agenda_count == 1
    assert facts.speaking_plan_count == 1
    assert "Alice (user:100)" in facts.payload


def test_repository_rows_bind_exact_discord_names_and_system_name() -> None:
    prepared = prepare_full_transcript_records(
        (
            TranscriptEntryRecord(
                id="human-1",
                session_id="session-1",
                source_key="speaker:10",
                user_id=10,
                raw_text="真人发言😀",
                normalised_text="不得使用",
                started_ms=0,
                ended_ms=1_000,
            ),
            TranscriptEntryRecord(
                id="system-1",
                session_id="session-1",
                source_key="BOT_TTS:agenda",
                user_id=None,
                raw_text="系统发言",
                normalised_text=None,
                started_ms=1_000,
                ended_ms=2_000,
            ),
        ),
        speaker_names={10: "Discord Alice"},
    )

    rows = json.loads(prepared.payload)
    assert rows[0]["speaker_kind"] == "participant"
    assert rows[0]["speaker_name"] == "Discord Alice"
    assert rows[0]["speaker_label"] == "[Discord Alice]"
    assert rows[1]["speaker_kind"] == "system"
    assert rows[1]["speaker_name"] == "聆序"
    assert rows[1]["speaker_label"] == "[聆序]"
    assert prepared.expected_speaker_labels == ("[Discord Alice]", "[聆序]")
    assert "不得使用" not in prepared.payload


def test_generated_minutes_accepts_only_exact_input_speaker_labels() -> None:
    headings = (
        "# 聆序会议纪要\n",
        "## 会议概览\n",
        "- 参会者: [Discord Alice]\n",
        "## 执行摘要\n",
        "## 议程清单\n",
        f"{AGENDA_TABLE_HEADER}\n",
        f"{AGENDA_TABLE_SEPARATOR}\n",
        "| - | 无 | - | - | - |\n",
        "## 权威流程配置\n",
        f"{SPEAKING_PLAN_TABLE_HEADER}\n",
        f"{SPEAKING_PLAN_TABLE_SEPARATOR}\n",
        "| - | 无可用流程配置 | - | - | - | - | - | - | - |\n",
        "## 权威静音执行\n",
        f"{MUTE_ENFORCEMENT_TABLE_HEADER}\n",
        f"{MUTE_ENFORCEMENT_TABLE_SEPARATOR}\n",
        "| - | 无可用静音执行记录 | - | - | - | - | - | - |\n",
        "## 权威发言流程\n",
        f"{SPEAKING_TABLE_HEADER}\n",
        f"{SPEAKING_TABLE_SEPARATOR}\n",
        "| - | 无受控发言记录 | - | - | - | - | - | - |\n",
        "## 投票结果\n",
        f"{VOTE_TABLE_HEADER}\n",
        f"{VOTE_TABLE_SEPARATOR}\n",
        "| - | 无已完成投票 | - | - | - | - |\n",
        "## 议题与讨论\n",
        "## 已确认决策\n",
        "## 已确认共识\n",
        "## 行动事项\n",
        "## 开放问题\n",
        "## 提案\n",
        "## 风险\n",
        "## 分歧\n",
        "## 优化去重后的完整逐字稿\n",
        "- [00:00:00-00:00:01] **[Discord Alice]**: 测试\n",
        "- [00:00:01-00:00:02] **[聆序]**: 测试\n",
    )
    content = "\n".join(headings)
    validate_generated_minutes(
        content,
        expected_speaker_labels=("[Discord Alice]", "[聆序]"),
    )

    with pytest.raises(FullTranscriptPreviewError, match="final_minutes_speaker_identity_invalid"):
        validate_generated_minutes(
            content.replace("**[Discord Alice]**", "**[主持人]**"),
            expected_speaker_labels=("[Discord Alice]", "[聆序]"),
        )


def test_plan_matrix_local_composition_and_transcript_coverage_are_fail_closed() -> None:
    agendas = tuple(
        MinutesAgenda(
            agenda_id=f"agenda-{position}",
            position=position,
            title=title,
            language="zh",
            status="COMPLETED" if position < 4 else "SKIPPED",
            duration_minutes=duration,
            requires_vote=False,
            topic_ids=(),
        )
        for position, title, duration in (
            (1, "自由讨论", 5),
            (2, "指定顺序", 10),
            (3, "首字母顺序", 10),
            (4, "演讲", 15),
        )
    )
    plans = (
        MinutesSpeakingPlan(
            agenda_item_id="agenda-1",
            mode="FREE",
            order_strategy="NONE",
            host_participation="AUTO",
            followup_mode="NONE",
            mute_enforcement_requested=False,
            prep_seconds=10,
        ),
        MinutesSpeakingPlan(
            agenda_item_id="agenda-2",
            mode="ORDERED",
            order_strategy="EXPLICIT",
            host_participation="AUTO",
            turns=(
                MinutesSpeakingPlanTurn(
                    user_id=10,
                    name="Alice (user:10)",
                    planned_seconds=120,
                ),
            ),
            followup_mode="SUPPLEMENT_TURNS",
            mute_enforcement_requested=True,
            prep_seconds=10,
        ),
        MinutesSpeakingPlan(
            agenda_item_id="agenda-3",
            mode="ORDERED",
            order_strategy="DISCORD_ALPHABETICAL",
            host_participation="MODERATOR_ONLY",
            default_turn_seconds=90,
            followup_mode="Q_AND_A",
            mute_enforcement_requested=True,
            prep_seconds=5,
        ),
        MinutesSpeakingPlan(
            agenda_item_id="agenda-4",
            mode="PRESENTATION",
            order_strategy="NONE",
            host_participation="AUTO",
            presenters=(MinutesSpeakingPlanParticipant(user_id=None, name="未获纪要授权成员"),),
            presentation_seconds=900,
            followup_mode="NONE",
            mute_enforcement_requested=True,
            prep_seconds=30,
        ),
    )
    facts = prepare_authoritative_meeting_facts(
        participant_labels=("[Alice]", "[Silent Bob]"),
        agendas=agendas,
        speaking_plans=plans,
    )
    raw = "\n".join(
        (
            "# 聆序会议纪要",
            "## 会议概览",
            "- 参会者: 错误成员",
            "## 执行摘要",
            "摘要",
            "## 议程清单",
            "| 被模型修改 |",
            "## 权威流程配置",
            "| 被模型修改 |",
            "## 权威静音执行",
            "| 被模型修改 |",
            "## 权威发言流程",
            "| 被模型修改 |",
            "## 投票结果",
            "| 被模型修改 |",
            "## 议题与讨论",
            "无",
            "## 已确认决策",
            "无",
            "## 已确认共识",
            "无",
            "## 行动事项",
            "无",
            "## 开放问题",
            "无",
            "## 提案",
            "无",
            "## 风险",
            "无",
            "## 分歧",
            "无",
            "## 优化去重后的完整逐字稿",
            "- [00:00:00-00:00:01] **[Alice]** <!-- source_ordinals:0,1 -->: 内容",
        )
    )
    composed = compose_authoritative_minutes(
        raw,
        expected_speaker_labels=("[Alice]",),
        authoritative_facts=facts,
    )

    assert "- 参会者: [Alice]、[Silent Bob]" in composed
    assert "被模型修改" not in composed
    assert "自由发言" in composed
    assert "主持人指定顺序" in composed
    assert "Discord 首字母顺序" in composed
    assert "演讲" in composed
    validate_generated_minutes(
        composed,
        expected_speaker_labels=("[Alice]",),
        expected_source_speaker_labels=((0, "[Alice]"), (1, "[Alice]")),
        authoritative_facts=facts,
    )
    with pytest.raises(
        FullTranscriptPreviewError,
        match="final_minutes_transcript_coverage_invalid",
    ):
        validate_generated_minutes(
            composed.replace("source_ordinals:0,1", "source_ordinals:0"),
            expected_speaker_labels=("[Alice]",),
            expected_source_speaker_labels=((0, "[Alice]"), (1, "[Alice]")),
            authoritative_facts=facts,
        )


def test_reconcile_generated_transcript_keeps_valid_edits_and_fills_invalid_coverage() -> None:
    records = (
        TranscriptEntryRecord(
            id="human-1",
            session_id="session-1",
            source_key="speaker:10:1",
            user_id=10,
            raw_text="原文一",
            normalised_text=None,
            started_ms=0,
            ended_ms=800,
        ),
        TranscriptEntryRecord(
            id="human-2",
            session_id="session-1",
            source_key="speaker:10:2",
            user_id=10,
            raw_text="原文二",
            normalised_text=None,
            started_ms=900,
            ended_ms=1_700,
        ),
        TranscriptEntryRecord(
            id="human-3",
            session_id="session-1",
            source_key="speaker:20:1",
            user_id=20,
            raw_text="原文三",
            normalised_text=None,
            started_ms=2_000,
            ended_ms=2_800,
        ),
        TranscriptEntryRecord(
            id="human-4",
            session_id="session-1",
            source_key="speaker:10:3",
            user_id=10,
            raw_text="原文四\n续行",
            normalised_text=None,
            started_ms=3_000,
            ended_ms=3_900,
        ),
    )
    prepared = prepare_full_transcript_records(records, speaker_names={10: "Alice", 20: "Bob"})
    raw = "\n".join(
        (
            "# 聆序会议纪要",
            "## 会议概览",
            "- 参会者: 错误成员",
            "## 执行摘要",
            "摘要",
            "## 议程清单",
            "| 被模型修改 |",
            "## 权威流程配置",
            "| 被模型修改 |",
            "## 权威静音执行",
            "| 被模型修改 |",
            "## 权威发言流程",
            "| 被模型修改 |",
            "## 投票结果",
            "| 被模型修改 |",
            "## 议题与讨论",
            "无",
            "## 已确认决策",
            "无",
            "## 已确认共识",
            "无",
            "## 行动事项",
            "无",
            "## 开放问题",
            "无",
            "## 提案",
            "无",
            "## 风险",
            "无",
            "## 分歧",
            "无",
            "## 优化去重后的完整逐字稿",
            "- [任意时间] **[Alice]** <!-- source_ordinals:0,1 -->: 模型合并后的发言",
            "- [任意时间] **[Alice]** <!-- source_ordinals:2 -->: 错误说话人",
            "- [任意时间] **[Alice]** <!-- source_ordinals:3,999 -->: 未知序号",
            "- [任意时间] **[Alice]** <!-- source_ordinals:0 -->: 重复序号",
            "- [任意时间] **[Bob]**: 缺少来源标签",
        )
    )
    composed = compose_authoritative_minutes(
        raw,
        expected_speaker_labels=prepared.expected_speaker_labels,
    )

    reconciled, audit = reconcile_generated_transcript(composed, prepared=prepared)

    assert audit.source_row_count == 4
    assert audit.model_line_count == 5
    assert audit.accepted_model_group_count == 1
    assert audit.accepted_model_source_count == 2
    assert audit.fallback_source_count == 2
    assert audit.rejected_model_line_count == 4
    assert (
        "- [00:00:00.000-00:00:01.700] **[Alice]** <!-- source_ordinals:0,1 -->: 模型合并后的发言"
    ) in reconciled
    assert (
        "- [00:00:02.000-00:00:02.800] **[Bob]** <!-- source_ordinals:2 -->: 原文三"
    ) in reconciled
    assert (
        "- [00:00:03.000-00:00:03.900] **[Alice]** <!-- source_ordinals:3 -->: 原文四 续行"
    ) in reconciled
    assert "错误说话人" not in reconciled
    assert "未知序号" not in reconciled
    assert "重复序号" not in reconciled
    assert "缺少来源标签" not in reconciled
    validate_generated_minutes(
        reconciled,
        expected_speaker_labels=prepared.expected_speaker_labels,
        expected_source_speaker_labels=prepared.source_speaker_labels,
    )


def test_reconcile_rejects_generic_placeholder_and_preserves_short_response() -> None:
    prepared = prepare_full_transcript_records(
        (
            TranscriptEntryRecord(
                id="human-1",
                session_id="session-1",
                source_key="speaker:10",
                user_id=10,
                raw_text="<|zh|><|NEUTRAL|><|Speech|><|withitn|>我同意。",
                normalised_text=None,
                started_ms=0,
                ended_ms=1_000,
            ),
        ),
        speaker_names={10: "Alice"},
    )
    raw = "\n".join(
        (
            "# 聆序会议纪要",
            "## 会议概览",
            "- 参会者: [Alice]",
            "## 执行摘要",
            "摘要",
            "## 议程清单",
            "无",
            "## 权威流程配置",
            "无",
            "## 权威静音执行",
            "无",
            "## 权威发言流程",
            "无",
            "## 投票结果",
            "无",
            "## 议题与讨论",
            "无",
            "## 已确认决策",
            "无",
            "## 已确认共识",
            "无",
            "## 行动事项",
            "无",
            "## 开放问题",
            "无",
            "## 提案",
            "无",
            "## 风险",
            "无",
            "## 分歧",
            "无",
            "## 优化去重后的完整逐字稿",
            "- [任意时间] **[Alice]** <!-- source_ordinals:0 -->: \uff08无实质内容\uff09",
        )
    )

    reconciled, audit = reconcile_generated_transcript(raw, prepared=prepared)

    assert "\uff08无实质内容\uff09" not in reconciled
    assert "<|" not in reconciled
    assert "<!-- source_ordinals:0 -->: 我同意。" in reconciled
    assert audit.accepted_model_source_count == 0
    assert audit.fallback_source_count == 1
    assert audit.rejected_model_line_count == 1


def test_reconcile_generated_transcript_scales_to_observed_1179_row_meeting() -> None:
    records = tuple(
        TranscriptEntryRecord(
            id=f"human-{ordinal}",
            session_id="session-1",
            source_key=f"speaker:10:{ordinal}",
            user_id=10,
            raw_text=f"原文 {ordinal}",
            normalised_text=None,
            started_ms=ordinal * 1_000,
            ended_ms=ordinal * 1_000 + 800,
        )
        for ordinal in range(1_179)
    )
    prepared = prepare_full_transcript_records(records, speaker_names={10: "Alice"})
    transcript_lines = tuple(
        f"- [任意时间] **[Alice]** <!-- source_ordinals:{ordinal} -->: 模型编辑 {ordinal}"
        for ordinal in range(1_178)
    )
    raw = "\n".join(
        (
            "# 聆序会议纪要",
            "## 会议概览",
            "- 参会者: 错误成员",
            "## 执行摘要",
            "摘要",
            "## 议程清单",
            "| 被模型修改 |",
            "## 权威流程配置",
            "| 被模型修改 |",
            "## 权威静音执行",
            "| 被模型修改 |",
            "## 权威发言流程",
            "| 被模型修改 |",
            "## 投票结果",
            "| 被模型修改 |",
            "## 议题与讨论",
            "无",
            "## 已确认决策",
            "无",
            "## 已确认共识",
            "无",
            "## 行动事项",
            "无",
            "## 开放问题",
            "无",
            "## 提案",
            "无",
            "## 风险",
            "无",
            "## 分歧",
            "无",
            "## 优化去重后的完整逐字稿",
            *transcript_lines,
        )
    )
    composed = compose_authoritative_minutes(
        raw,
        expected_speaker_labels=prepared.expected_speaker_labels,
    )

    reconciled, audit = reconcile_generated_transcript(composed, prepared=prepared)

    assert audit.source_row_count == 1_179
    assert audit.accepted_model_source_count == 1_178
    assert audit.fallback_source_count == 1
    assert audit.rejected_model_line_count == 0
    assert "<!-- source_ordinals:1178 -->: 原文 1178" in reconciled
    validate_generated_minutes(
        reconciled,
        expected_speaker_labels=prepared.expected_speaker_labels,
        expected_source_speaker_labels=prepared.source_speaker_labels,
    )


def test_system_name_is_reserved_from_discord_participant_labels() -> None:
    record = TranscriptEntryRecord(
        id="human-1",
        session_id="session-1",
        source_key="speaker:10",
        user_id=10,
        raw_text="真人发言",
        normalised_text=None,
        started_ms=0,
        ended_ms=1_000,
    )

    with pytest.raises(FullTranscriptPreviewError, match="speaker_name_reserved"):
        prepare_full_transcript_records((record,), speaker_names={10: "聆序"})


def test_acceptance_metadata_supplies_exact_names_to_native_transcript(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.json"
    metadata = tmp_path / "minutes.json"
    _write_transcript(transcript)
    metadata.write_text(
        json.dumps(
            {
                "participants": {
                    "host": "Discord Alice (user:10)",
                    "attendees": [],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    names = load_speaker_names(metadata)
    assert names == {10: "Discord Alice"}
    prepared = prepare_full_transcript(transcript, speaker_names=names)
    rows = json.loads(prepared.payload)
    assert {row["speaker_label"] for row in rows} == {"[Discord Alice]"}


async def _server(
    *,
    finish_reason: str = "stop",
    reasoning_content: str | None = "PRIVATE_REASONING_MUST_BE_DISCARDED",
    content: object = "# 聆序会议纪要\n\n测试输出\n",
) -> tuple[TestServer, list[dict[str, Any]], list[str | None]]:
    requests: list[dict[str, Any]] = []
    authorizations: list[str | None] = []

    async def models(request: web.Request) -> web.Response:
        authorizations.append(request.headers.get("Authorization"))
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": APPROVED_TOPIC_MODEL,
                        "object": "model",
                        "owned_by": "deepseek",
                    }
                ],
            }
        )

    async def complete(request: web.Request) -> web.Response:
        authorizations.append(request.headers.get("Authorization"))
        requests.append(await request.json())
        return web.json_response(
            {
                "id": "content-free-preview-test",
                "object": "chat.completion",
                "model": APPROVED_TOPIC_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {
                            "role": "assistant",
                            "reasoning_content": reasoning_content,
                            "content": content,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            }
        )

    app = web.Application()
    app.router.add_get("/models", models)
    app.router.add_post("/chat/completions", complete)
    server = TestServer(app)
    await server.start_server()
    return server, requests, authorizations


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["text", "json_object"])
async def test_client_verifies_model_and_discards_reasoning(output_format: Any) -> None:
    server, requests, authorizations = await _server()
    try:
        client = DeepSeekFullTranscriptPreviewClient(
            api_key="PRIVATE_TEST_KEY", output_format=output_format
        )
        client._models_endpoint = str(server.make_url("/models"))
        client._chat_endpoint = str(server.make_url("/chat/completions"))

        result = await client.generate([{"role": "user", "content": "CONTENT_FREE_TEST_INPUT"}])
    finally:
        await server.close()

    assert authorizations == ["Bearer PRIVATE_TEST_KEY", "Bearer PRIVATE_TEST_KEY"]
    assert len(requests) == 1
    assert requests[0]["thinking"] == {"type": "enabled"}
    assert requests[0]["reasoning_effort"] == "high"
    assert requests[0]["max_tokens"] == 131_072
    assert requests[0].get("response_format") == (
        {"type": "json_object"} if output_format == "json_object" else None
    )
    assert result.status == "complete"
    assert result.content == "# 聆序会议纪要\n\n测试输出\n"
    assert "PRIVATE_REASONING_MUST_BE_DISCARDED" not in result.content
    assert result.usage == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }


def test_client_accepts_1200_second_timeout_and_rejects_higher_value() -> None:
    client = DeepSeekFullTranscriptPreviewClient(
        api_key="PRIVATE_TEST_KEY",
        timeout_seconds=1_200,
    )

    assert client._timeout_seconds == 1_200
    with pytest.raises(FullTranscriptPreviewError, match="approved_model_policy_failed"):
        DeepSeekFullTranscriptPreviewClient(
            api_key="PRIVATE_TEST_KEY",
            timeout_seconds=1_200.01,
        )
    with pytest.raises(FullTranscriptPreviewError, match="approved_model_policy_failed"):
        DeepSeekFullTranscriptPreviewClient(
            api_key="PRIVATE_TEST_KEY",
            max_output_tokens=384_001,
        )


@pytest.mark.asyncio
async def test_client_retains_safe_diagnostics_for_output_limit() -> None:
    server, _, _ = await _server(
        finish_reason="length",
        reasoning_content="PRIVATE_REASONING_MUST_NEVER_BE_PERSISTED",
        content="TRUNCATED_PRIVATE_OUTPUT",
    )
    try:
        client = DeepSeekFullTranscriptPreviewClient(api_key="PRIVATE_TEST_KEY")
        client._models_endpoint = str(server.make_url("/models"))
        client._chat_endpoint = str(server.make_url("/chat/completions"))

        result = await client.generate([{"role": "user", "content": "CONTENT_FREE_TEST_INPUT"}])
    finally:
        await server.close()

    assert result.status == "output_limit"
    assert result.finish_reason == "length"
    assert result.diagnostics is not None
    assert result.diagnostics.as_dict() == {
        "phase": "response_validation",
        "elapsed_ms": result.diagnostics.elapsed_ms,
        "http_status": 200,
        "finish_reason": "length",
        "content_state": "present",
        "reasoning_state": "present",
        "reasoning_content_chars": len("PRIVATE_REASONING_MUST_NEVER_BE_PERSISTED"),
        "response_body_bytes": result.diagnostics.response_body_bytes,
        "usage_prompt_tokens": 100,
        "usage_completion_tokens": 50,
        "usage_total_tokens": 150,
    }
    assert "TRUNCATED_PRIVATE_OUTPUT" not in json.dumps(
        result.diagnostics.as_dict(), ensure_ascii=False
    )


@pytest.mark.parametrize(
    ("finish_reason", "content", "expected_content_state"),
    [("stop", "", "empty"), ("length", None, "null")],
)
@pytest.mark.asyncio
async def test_client_empty_content_exposes_only_safe_diagnostics(
    finish_reason: str,
    content: object,
    expected_content_state: str,
) -> None:
    private_reasoning = "PRIVATE_REASONING_MUST_NEVER_BE_PERSISTED"
    server, _, _ = await _server(
        finish_reason=finish_reason,
        reasoning_content=private_reasoning,
        content=content,
    )
    try:
        client = DeepSeekFullTranscriptPreviewClient(api_key="PRIVATE_TEST_KEY")
        client._models_endpoint = str(server.make_url("/models"))
        client._chat_endpoint = str(server.make_url("/chat/completions"))

        with pytest.raises(FullTranscriptPreviewError, match="preview_content_empty") as raised:
            await client.generate([{"role": "user", "content": "CONTENT_FREE_TEST_INPUT"}])
    finally:
        await server.close()

    assert raised.value.diagnostics is not None
    diagnostics = raised.value.diagnostics.as_dict()
    assert diagnostics["phase"] == "response_validation"
    assert diagnostics["http_status"] == 200
    assert diagnostics["finish_reason"] == finish_reason
    assert diagnostics["content_state"] == expected_content_state
    assert diagnostics["reasoning_state"] == "present"
    assert diagnostics["reasoning_content_chars"] == len(private_reasoning)
    assert diagnostics["usage_prompt_tokens"] == 100
    assert diagnostics["usage_completion_tokens"] == 50
    assert diagnostics["usage_total_tokens"] == 150
    assert isinstance(diagnostics["elapsed_ms"], int)
    assert isinstance(diagnostics["response_body_bytes"], int)
    assert private_reasoning not in json.dumps(diagnostics, ensure_ascii=False)


def test_failure_diagnostics_rejects_unbounded_or_unknown_values() -> None:
    with pytest.raises(ValueError):
        FullTranscriptFailureDiagnostics(phase="PRIVATE_RESPONSE_BODY")
    with pytest.raises(ValueError):
        FullTranscriptFailureDiagnostics(
            phase="response_validation",
            finish_reason="PRIVATE_FINISH_REASON",
        )
