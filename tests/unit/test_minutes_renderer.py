import hashlib
import re
from dataclasses import replace
from datetime import UTC, datetime

from lingxu.transcript.models import (
    CanonicalTranscript,
    CanonicalTranscriptEntry,
    CanonicalTranscriptMetrics,
)
from lingxu.worker.minutes_models import (
    CanonicalMetrics,
    DeterministicCounts,
    EvidenceCoverage,
    MinutesAgenda,
    MinutesDocumentV2,
    MinutesParticipants,
    MinutesProvenance,
    MinutesSpeakingInterval,
    MinutesSpeakingTurn,
    MinutesTopic,
    MinutesVote,
    PrivacyExclusionCounts,
)
from lingxu.worker.minutes_renderer import (
    MINUTES_RENDERER_REVISION,
    render_minutes,
    renderer_input_sha256,
)
from lingxu.worker.outcome_merge import merge_outcomes
from lingxu.worker.outcomes import OutcomeCandidate, OutcomeType, VerificationStatus


def _candidate(
    candidate_id: str,
    outcome_type: OutcomeType,
    text: str,
    *,
    status: VerificationStatus = VerificationStatus.EVIDENCE_LINKED,
    requires_confirmation: bool = False,
    owner_user_id: int | None = None,
    owner_name: str | None = None,
    deadline: str | None = None,
) -> OutcomeCandidate:
    return OutcomeCandidate(
        id=candidate_id,
        type=outcome_type,
        text=text,
        owner_user_id=owner_user_id,
        owner_name=owner_name,
        deadline=deadline,
        source_entry_ids=("entry-participant",),
        source_event_ids=(),
        confidence=0.9,
        requires_confirmation=requires_confirmation,
        verification_status=status,
    )


def _document(*, measurement_available: bool = True) -> MinutesDocumentV2:
    candidates = (
        _candidate("decision", OutcomeType.DECISION, "采用发布方案"),
        _candidate("agreement", OutcomeType.AGREEMENT, "同意复核流程"),
        _candidate(
            "action",
            OutcomeType.ACTION_ITEM,
            "准备发布清单",
            requires_confirmation=True,
            owner_user_id=7,
            owner_name="参会者甲",
            deadline="2026-08-21",
        ),
        _candidate("question", OutcomeType.OPEN_QUESTION, "何时开始复核?"),
        _candidate("proposal", OutcomeType.PROPOSAL, "建议增加回归测试"),
        _candidate("risk", OutcomeType.RISK, "存在发布延迟风险"),
        _candidate("disagreement", OutcomeType.DISAGREEMENT, "对发布日期存在分歧"),
        _candidate(
            "claim",
            OutcomeType.PROPOSAL,
            "外部团队已经批准",
            status=VerificationStatus.NOT_CHECKED,
            requires_confirmation=True,
        ),
    )
    merged = merge_outcomes(candidates)
    return MinutesDocumentV2(
        provenance=MinutesProvenance(
            outcome_pipeline_revision="renderer-test",
            speech_act_policy_revision="a" * 64,
            minutes_renderer_revision=MINUTES_RENDERER_REVISION,
            renderer_input_sha256="f" * 64,
            validated_outcomes_sha256="b" * 64,
            canonical_transcript_sha256="c" * 64,
        ),
        session_id="01SYNTHETICRENDERERTEST0000",
        generated_at=datetime(2026, 8, 20, 10, 30, tzinfo=UTC),
        meeting_title="发布评审会",
        participants=MinutesParticipants(
            host="主持人甲 (user:1)", attendees=("参会者甲 (user:7)",)
        ),
        agendas=(
            MinutesAgenda(
                agenda_id="agenda-1",
                position=1,
                title="发布准备",
                language="zh",
                status="COMPLETED",
                duration_minutes=10,
                requires_vote=True,
                vote_content="是否发布?",
                vote_duration_seconds=60,
                topic_ids=("topic-1",),
                summary="议程包含一个已验证议题。",
            ),
        ),
        topics=(
            MinutesTopic(
                topic_id="topic-1",
                title="发布准备",
                started_ms=1_000,
                ended_ms=4_000,
                method="deterministic",
                source_entry_ids=("entry-bot", "entry-participant"),
                summary="完成发布前复核。",
            ),
        ),
        validated_outcomes=(),
        decisions=merged.decisions,
        agreements=merged.agreements,
        actions=merged.actions,
        questions=merged.questions,
        proposals=merged.proposals,
        risks=merged.risks,
        disagreements=merged.disagreements,
        deferred_items=(),
        speaker_claims=merged.speaker_claims,
        votes=(
            MinutesVote(
                source_event_id=9,
                vote_id="vote-1",
                agenda_item_id="agenda-1",
                vote_content="是否发布?",
                counts={"approve": 2, "reject": 1, "abstain": 1},
                eligible_count=4,
                responded_count=3,
                result="approved",
            ),
        ),
        deterministic_counts=DeterministicCounts(
            agenda_total=1,
            agenda_completed=1,
            participant_transcript_entries=1,
            bot_tts_entries=1,
            closed_votes=1,
            topic_total=1,
            validated_outcomes=8,
        ),
        evidence_coverage=EvidenceCoverage(
            evidence_required=4,
            evidence_linked=4,
            transcript_sources=1,
            event_sources=1,
        ),
        privacy_exclusions=PrivacyExclusionCounts(
            transcript_entries=2,
            segment_rows=1,
            segment_files=1,
            measurement_available=measurement_available,
        ),
        canonical_metrics=CanonicalMetrics(
            source_entry_count=5,
            canonical_entry_count=4,
            duplicate_entry_count=1,
            backchannel_entry_count=1,
            asr_tag_count=2,
        ),
        summary_method="deepseek_partial",
        summary_model="deepseek-v4-flash",
        summary_provider="deepseek",
        summary_model_digest="d" * 64,
        summary_policy_identity="e" * 64,
        fallback_reasons=("topic_fallback",),
        executive_summary="会议确认了发布准备事项。",
        full_summary="会议完成了发布准备事项的核验。",
    )


def _entry(
    entry_id: str,
    text: str,
    *,
    speaker_user_id: int | None,
    entry_type: str,
    started_ms: int,
) -> CanonicalTranscriptEntry:
    return CanonicalTranscriptEntry(
        entry_id=entry_id,
        source_entry_ids=(f"source-{entry_id}",),
        speaker_user_id=speaker_user_id,
        speaker_name=None,
        started_ms=started_ms,
        ended_ms=started_ms + 500,
        raw_texts=(text,),
        text=text,
        entry_type=entry_type,
        language="zh",
        confidence=0.95,
        utterance_group_id=f"group-{entry_id}",
        normalisation_changes=(),
        asr_tags=(),
        emotion=None,
    )


def _canonical() -> CanonicalTranscript:
    entries = (
        _entry(
            "entry-bot",
            "现在进入发布准备议程。",
            speaker_user_id=None,
            entry_type="bot_announcement",
            started_ms=1_000,
        ),
        _entry(
            "entry-participant",
            "我们确认发布清单。",
            speaker_user_id=7,
            entry_type="speech",
            started_ms=2_000,
        ),
        _entry(
            "entry-backchannel",
            "嗯",
            speaker_user_id=7,
            entry_type="backchannel",
            started_ms=2_500,
        ),
        _entry(
            "entry-control",
            "synthetic privacy control phrase",
            speaker_user_id=7,
            entry_type="privacy_control",
            started_ms=3_000,
        ),
        _entry(
            "entry-noise",
            "synthetic private noise",
            speaker_user_id=7,
            entry_type="noise",
            started_ms=3_500,
        ),
    )
    return CanonicalTranscript(
        entries=entries,
        metrics=CanonicalTranscriptMetrics(
            source_entry_count=5,
            canonical_entry_count=5,
            duplicate_entry_count=0,
            backchannel_entry_count=1,
            asr_tag_count=0,
        ),
    )


def test_renderer_contains_complete_minutes_and_readable_canonical_transcript_once() -> None:
    rendered = render_minutes(_document(), _canonical(), labels={7: "参会者甲"})

    for heading in (
        "## 会议概览与参会者",
        "## 执行摘要",
        "## 议程与议题摘要",
        "## 已确认决策",
        "## 已确认共识",
        "## 行动事项",
        "## 开放问题",
        "## 提案",
        "## 风险",
        "## 分歧",
        "## 未验证的发言者主张",
        "## 投票结果",
        "## 确定性统计与质量说明",
        "## 带时间戳逐字稿",
    ):
        assert heading in rendered
    assert rendered.count("现在进入发布准备议程。") == 1
    assert rendered.count("我们确认发布清单。") == 1
    assert "[00:00:01.000-00:00:01.500] **聆序机器人**" in rendered
    assert "[00:00:02.000-00:00:02.500] **参会者甲**" in rendered
    assert "synthetic privacy control phrase" not in rendered
    assert "synthetic private noise" not in rendered
    assert "**待确认**" in rendered
    assert "**未验证 / 待确认**" in rendered
    assert "- 会议标题: 发布评审会" in rendered
    assert "- 投票内容: 是否发布?" in rendered
    assert "- 投票时长: 60 秒" in rendered
    assert "议程包含一个已验证议题。" in rendered
    assert "会议完成了发布准备事项的核验。" in rendered


def test_renderer_uses_authoritative_custom_vote_counts_and_document_metrics() -> None:
    rendered = render_minutes(_document(), _canonical(), labels={7: "参会者甲"})

    assert "approve: 2" in rendered
    assert "reject: 1" in rendered
    assert "abstain: 1" in rendered
    assert "有效人数: 4" in rendered
    assert "已响应人数: 3" in rendered
    assert "最终结果: approved" in rendered
    assert "规范条目数: 4" in rendered
    assert "重复合并数: 1" in rendered
    assert "证据要求项: 4" in rendered
    assert "已链接证据项: 4" in rendered
    assert "隐私排除逐字稿条目: 2" in rendered


def test_renderer_reports_unavailable_and_zero_voting_agenda_configuration_honestly() -> None:
    document = _document()
    unavailable_agenda = document.agendas[0].model_copy(
        update={"vote_content": None, "vote_duration_seconds": None}
    )
    unavailable = render_minutes(
        document.model_copy(update={"agendas": (unavailable_agenda,)}),
        _canonical(),
        labels={7: "参会者甲"},
    )
    zero_agenda = unavailable_agenda.model_copy(update={"vote_duration_seconds": 0})
    zero = render_minutes(
        document.model_copy(update={"agendas": (zero_agenda,)}),
        _canonical(),
        labels={7: "参会者甲"},
    )

    assert "- 投票内容: 未知" in unavailable
    assert "- 投票时长: 未知" in unavailable
    assert "- 投票内容: 未知" in zero
    assert "- 投票时长: 0 秒" in zero


def test_renderer_is_stable_and_reports_unknown_privacy_measurement() -> None:
    document = _document(measurement_available=False)
    canonical = _canonical()

    first = render_minutes(document, canonical, labels={7: "参会者甲"})
    second = render_minutes(document, canonical, labels={7: "参会者甲"})

    assert first == second
    assert "隐私排除计量: 不可用 (未知)" in first


def test_renderer_preserves_zero_metrics_without_inference() -> None:
    document = _document().model_copy(
        update={
            "agendas": (),
            "topics": (),
            "validated_outcomes": (),
            "decisions": (),
            "agreements": (),
            "actions": (),
            "questions": (),
            "proposals": (),
            "risks": (),
            "disagreements": (),
            "deferred_items": (),
            "speaker_claims": (),
            "votes": (),
            "executive_summary": None,
            "full_summary": None,
            "deterministic_counts": DeterministicCounts(
                agenda_total=0,
                agenda_completed=0,
                participant_transcript_entries=0,
                bot_tts_entries=0,
                closed_votes=0,
                topic_total=0,
                validated_outcomes=0,
            ),
            "evidence_coverage": EvidenceCoverage(
                evidence_required=0,
                evidence_linked=0,
                transcript_sources=0,
                event_sources=0,
            ),
            "canonical_metrics": CanonicalMetrics(
                source_entry_count=0,
                canonical_entry_count=0,
                duplicate_entry_count=0,
                backchannel_entry_count=0,
                asr_tag_count=0,
            ),
            "privacy_exclusions": PrivacyExclusionCounts(
                transcript_entries=0,
                segment_rows=0,
                segment_files=0,
            ),
        }
    )

    rendered = render_minutes(
        document, CanonicalTranscript(entries=(), metrics=_canonical().metrics), labels={}
    )

    assert "当前没有可形成完整总结的已验证会议事实。" in rendered
    assert "- 议程总数: 0" in rendered
    assert "- 规范条目数: 0" in rendered
    assert "- 证据要求项: 0" in rendered
    assert "- 隐私排除逐字稿条目: 0" in rendered
    assert "- 没有可公开的规范逐字稿条目。" in rendered


def test_renderer_input_digest_binds_document_canonical_entries_and_labels() -> None:
    document = _document()
    canonical = _canonical()
    labels = {7: "参会者甲"}

    digest = renderer_input_sha256(document, canonical, labels=labels)

    assert len(digest) == 64
    assert digest == renderer_input_sha256(document, canonical, labels=labels)
    assert digest != renderer_input_sha256(
        document.model_copy(update={"meeting_title": "不同标题"}),
        canonical,
        labels=labels,
    )
    assert digest != renderer_input_sha256(
        document,
        CanonicalTranscript(
            entries=(replace(canonical.entries[0], text="不同播报"), *canonical.entries[1:]),
            metrics=canonical.metrics,
        ),
        labels=labels,
    )
    assert digest != renderer_input_sha256(document, canonical, labels={7: "不同标签"})


def test_renderer_escapes_untrusted_markdown_html_mentions_controls_and_bidi() -> None:
    hostile = (
        "# @everyone <@123> <script>x</script> <!--x--> **bold** "
        "[link](x) `code`\n## injected\x00\u202e"
    )
    base = _document()
    hostile_outcome = base.decisions[0].model_copy(update={"text": hostile})
    hostile_vote = base.votes[0].model_copy(
        update={"vote_content": hostile, "counts": {hostile: 2}, "responded_count": 2}
    )
    document = base.model_copy(
        update={
            "meeting_title": hostile,
            "participants": MinutesParticipants(host=hostile, attendees=(hostile,)),
            "agendas": (base.agendas[0].model_copy(update={"title": hostile, "summary": hostile}),),
            "topics": (base.topics[0].model_copy(update={"title": hostile, "summary": hostile}),),
            "decisions": (hostile_outcome,),
            "votes": (hostile_vote,),
            "executive_summary": hostile,
            "full_summary": hostile,
        }
    )
    canonical = CanonicalTranscript(
        entries=(replace(_canonical().entries[1], text=hostile),),
        metrics=_canonical().metrics,
    )

    rendered = render_minutes(document, canonical, labels={7: hostile})

    assert "@everyone" not in rendered
    assert "<@123>" not in rendered
    assert "<script>" not in rendered
    assert "<!--" not in rendered
    assert "\x00" not in rendered
    assert "\u202e" not in rendered
    assert "\N{FULLWIDTH COMMERCIAL AT}everyone" in rendered
    assert "&lt;script&gt;" in rendered
    assert "\\#" in rendered
    assert "\\*\\*bold\\*\\*" in rendered
    assert "\\[link\\]\\(x\\)" in rendered
    assert "\\`code\\`" in rendered
    for heading in (
        "## 会议概览与参会者",
        "## 执行摘要",
        "## 议程与议题摘要",
        "## 带时间戳逐字稿",
    ):
        assert rendered.count(heading) == 1
    assert "\n## injected" not in rendered


def test_renderer_neutralizes_autolink_schemes_across_every_untrusted_field_path() -> None:
    base = _document()
    action = base.actions[0].model_copy(
        update={
            "text": "outcome://text",
            "owner_name": "owner://name",
            "deadline": "deadline://value",
        }
    )
    vote = base.votes[0].model_copy(
        update={
            "vote_content": "votecontent://subject",
            "counts": {"voteoption://yes": 2},
            "responded_count": 2,
            "result": "voteresult://accepted",
        }
    )
    document = base.model_copy(
        update={
            "session_id": "session://id",
            "meeting_title": "title://meeting",
            "participants": MinutesParticipants(
                host="host://name",
                attendees=("attendee://name",),
            ),
            "agendas": (
                base.agendas[0].model_copy(
                    update={
                        "title": "agenda://title",
                        "language": "language://value",
                        "status": "status://value",
                        "vote_content": "agendavote://content",
                        "summary": "agendasummary://text",
                    }
                ),
            ),
            "topics": (
                base.topics[0].model_copy(
                    update={
                        "title": "topic://title",
                        "method": "method://value",
                        "summary": "topicsummary://text",
                    }
                ),
            ),
            "decisions": (),
            "actions": (action,),
            "votes": (vote,),
            "summary_provider": "provider://name",
            "summary_model": "model://name",
            "fallback_reasons": ("fallback://reason",),
            "executive_summary": "executive://summary",
            "full_summary": "fullsummary://text",
        }
    )
    participant = replace(
        _canonical().entries[1],
        text="transcript://text",
        speaker_name="speakername://fallback",
    )
    fallback_speaker = replace(
        participant,
        entry_id="entry-fallback-speaker",
        source_entry_ids=("source-entry-fallback-speaker",),
        speaker_user_id=8,
        utterance_group_id="group-entry-fallback-speaker",
    )
    canonical = CanonicalTranscript(
        entries=(participant, fallback_speaker),
        metrics=_canonical().metrics,
    )

    rendered = render_minutes(document, canonical, labels={7: "label://speaker"})
    fullwidth_colon = "\N{FULLWIDTH COLON}"

    assert re.search(r"[A-Za-z][A-Za-z0-9+.-]*://", rendered, flags=re.IGNORECASE) is None
    for scheme in (
        "session",
        "title",
        "host",
        "attendee",
        "agenda",
        "language",
        "status",
        "agendavote",
        "agendasummary",
        "topic",
        "method",
        "topicsummary",
        "outcome",
        "owner",
        "deadline",
        "votecontent",
        "voteoption",
        "voteresult",
        "provider",
        "model",
        "fallback",
        "executive",
        "fullsummary",
        "transcript",
        "speakername",
        "label",
    ):
        assert f"{scheme}{fullwidth_colon}//" in rendered


def test_renderer_neutralizes_mixed_case_and_rfc_style_schemes_without_touching_colons() -> None:
    base = _document()
    values = (
        "http://x",
        "https://localhost",
        "ftp://host",
        "HtTp+Alt.Example://mixed",
        "sentence with CuStOm-1://host/path embedded",
    )
    document = base.model_copy(
        update={
            "meeting_title": values[0],
            "executive_summary": values[1],
            "full_summary": values[2],
            "fallback_reasons": values[3:],
        }
    )

    rendered = render_minutes(document, _canonical(), labels={7: "参会者甲"})
    fullwidth_colon = "\N{FULLWIDTH COLON}"

    assert re.search(r"[A-Za-z][A-Za-z0-9+.-]*://", rendered, flags=re.IGNORECASE) is None
    assert f"http{fullwidth_colon}//x" in rendered
    assert f"https{fullwidth_colon}//localhost" in rendered
    assert f"ftp{fullwidth_colon}//host" in rendered
    assert rf"HtTp\+Alt\.Example{fullwidth_colon}//mixed" in rendered
    assert rf"CuStOm\-1{fullwidth_colon}//host/path" in rendered
    assert "[00:00:01.000-00:00:01.500]" in rendered
    assert "approve: 2" in rendered


def test_renderer_revision_changes_with_autolink_semantics() -> None:
    assert (
        MINUTES_RENDERER_REVISION
        == hashlib.sha256(b"lingxu-minutes-renderer-speaking-plan-v1").hexdigest()
    )


def test_renderer_binds_authoritative_speaking_row_and_counts() -> None:
    base = _document()
    started = datetime(2026, 8, 20, 10, 31, tzinfo=UTC)
    ended = datetime(2026, 8, 20, 10, 31, 45, tzinfo=UTC)
    turn = MinutesSpeakingTurn(
        agenda_item_id="agenda-1",
        turn_index=0,
        mode="ORDERED",
        speaker_user_ids=(7,),
        speaker_names=("参会者甲 (user:7)",),
        planned_seconds=60,
        extension_seconds=30,
        started_at=started,
        finished_at=ended,
        active_intervals=(
            MinutesSpeakingInterval(started_at=started, ended_at=ended, duration_ms=45_000),
        ),
        pause_intervals=(),
        actual_floor_ms=45_000,
        finish_reason="HOST_ENDED_EARLY",
        finished_early=True,
        skipped=False,
        absent=False,
        overtime_ms=0,
    )
    document = base.model_copy(
        update={
            "speaking_turns": (turn,),
            "deterministic_counts": base.deterministic_counts.model_copy(
                update={"controlled_speaking_turns": 1}
            ),
        }
    )

    rendered = render_minutes(document, _canonical(), labels={7: "参会者甲"})

    assert "## 权威发言流程" in rendered
    assert "| 发布准备 | 1 | ORDERED | 参会者甲 (user:7) | 60 秒 | 30 秒 |" in rendered
    assert "主持人提前结束" in rendered
    assert "- 受控主发言轮次: 1" in rendered
