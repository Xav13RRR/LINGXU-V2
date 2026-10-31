from __future__ import annotations

import asyncio
import json
import struct
from collections import deque
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from lingxu.asr.models import ASRResult, UtteranceAudio
from lingxu.asr.queue import PerUserRealtimeQueue
from lingxu.asr.utterance import UtteranceBuilder
from lingxu.voice.frames import SessionAudioFrame
from lingxu.voice.sink import MeetingPcmSink
from lingxu.voice_assistant.fuzzy import split_wake_query
from lingxu.voice_assistant.response import (
    OllamaShortAnswerProvider,
    VoiceAssistantProviderError,
    limit_short_answer,
)
from lingxu.voice_assistant.service import AssistantTurnState, MeetingVoiceAssistant
from lingxu.voice_assistant.wakeword import (
    KEYWORDS_TEXT,
    SherpaOnnxKeywordSpotter,
    keyword_manifest_for_phrase,
    selected_model_digest,
)
from lingxu.voice_assistant.window import WakeWindowBuilder

_DIGEST = "a" * 64
_VOICE = struct.pack("<hh", 2_000, 2_000) * 960
_SILENCE = b"\0" * len(_VOICE)


class FakeKeywordSpotter:
    def __init__(self, detected: bool = True) -> None:
        self.detected = detected
        self.prepare_count = 0
        self.wake_phrase = "呼叫指挥中心"

    async def prepare(self) -> None:
        self.prepare_count += 1

    async def detect(self, utterance: UtteranceAudio) -> bool:
        return self.detected

    async def update_wake_phrase(self, wake_phrase: str) -> None:
        self.wake_phrase = wake_phrase


class BlockingUpdateKeywordSpotter(FakeKeywordSpotter):
    def __init__(self) -> None:
        super().__init__()
        self.update_entered = asyncio.Event()
        self.update_release = asyncio.Event()

    async def update_wake_phrase(self, wake_phrase: str) -> None:
        self.update_entered.set()
        await self.update_release.wait()
        self.wake_phrase = wake_phrase


class BlockingKeywordSpotter:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.user_calls: list[int] = []

    async def prepare(self) -> None:
        return None

    async def detect(self, utterance: UtteranceAudio) -> bool:
        self.user_calls.append(utterance.user_id)
        if len(self.user_calls) == 1:
            self.entered.set()
            await self.release.wait()
        return False

    async def update_wake_phrase(self, wake_phrase: str) -> None:
        del wake_phrase


class CompletedFallbackKeywordSpotter:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[tuple[int, int]] = []

    async def prepare(self) -> None:
        return None

    async def detect(self, utterance: UtteranceAudio) -> bool:
        self.calls.append((utterance.user_id, len(utterance.pcm_s16le_48k_stereo)))
        if len(self.calls) == 1:
            self.entered.set()
            await self.release.wait()
            return False
        return utterance.user_id == 2 and len(utterance.pcm_s16le_48k_stereo) > len(_VOICE)

    async def update_wake_phrase(self, wake_phrase: str) -> None:
        del wake_phrase


class FakeASRProvider:
    def __init__(self, *texts: str) -> None:
        self.texts = deque(texts)
        self.prepare_count = 0
        self.hotwords: list[tuple[str, ...]] = []

    async def prepare(self) -> None:
        self.prepare_count += 1

    async def transcribe(
        self,
        utterance: UtteranceAudio,
        *,
        language: str,
        hotwords: tuple[str, ...],
    ) -> ASRResult:
        self.hotwords.append(hotwords)
        text = self.texts.popleft()
        return ASRResult(text, text, "zh", 1)


class FakeAnswerProvider:
    def __init__(self) -> None:
        self.questions: list[str] = []
        self.prepare_count = 0

    async def prepare(self) -> None:
        self.prepare_count += 1

    async def answer(self, *, question: str, context: dict[str, Any]) -> str:
        self.questions.append(question)
        return "这是一个简短回答。"


class _BatchOnlyQueue(PerUserRealtimeQueue):
    def __init__(self) -> None:
        super().__init__(max_frames_per_user=20)
        self.batch_calls = 0

    def get(self) -> SessionAudioFrame | None:
        raise AssertionError("assistant consumer must not perform one thread hop per frame")

    def get_many(self, max_items: int) -> tuple[SessionAudioFrame, ...] | None:
        self.batch_calls += 1
        return super().get_many(max_items)


def _feed_utterance(assistant: MeetingVoiceAssistant, user_id: int, started_ns: int) -> None:
    assert assistant.receive(
        user_id=user_id,
        received_monotonic_ns=started_ns,
        pcm_s16le_48k_stereo=_VOICE,
    )
    assert assistant.receive(
        user_id=user_id,
        received_monotonic_ns=started_ns + 20_000_000,
        pcm_s16le_48k_stereo=_SILENCE,
    )


async def _wait_state(
    assistant: MeetingVoiceAssistant,
    expected: AssistantTurnState,
) -> None:
    async with asyncio.timeout(2):
        while assistant.state is not expected:
            await asyncio.sleep(0.01)


def _assistant(
    *,
    asr: FakeASRProvider,
    answers: FakeAnswerProvider,
    speak: Any,
    followup_seconds: float = 0.2,
    vad_end_ms: int = 20,
    endpoint_poll_seconds: float = 0.05,
    clock_ns: Callable[[], int] = lambda: 0,
    frame_queue: PerUserRealtimeQueue | None = None,
    keyword_spotter: Any | None = None,
    max_pending_utterances: int = 8,
    wake_phrase: str = "呼叫指挥中心",
) -> MeetingVoiceAssistant:
    return MeetingVoiceAssistant(
        session_id="session-1",
        wake_phrase=wake_phrase,
        frame_queue=frame_queue or PerUserRealtimeQueue(max_frames_per_user=20),
        utterance_builder=UtteranceBuilder(
            session_id="session-1",
            vad_threshold=100,
            vad_start_ms=20,
            vad_end_ms=vad_end_ms,
            min_utterance_ms=20,
            max_utterance_seconds=2,
        ),
        wake_window_builder=WakeWindowBuilder(
            session_id="session-1",
            activity_threshold=100,
            window_ms=200,
            hop_ms=20,
            min_window_ms=20,
            preroll_ms=20,
        ),
        keyword_spotter=keyword_spotter or FakeKeywordSpotter(),
        asr_provider=asr,
        answer_provider=answers,
        context_provider=lambda: asyncio.sleep(0, result={"meeting_state": "SPEAKER_ACTIVE"}),
        speak=speak,
        followup_seconds=followup_seconds,
        quarantine_ms=0,
        endpoint_poll_seconds=endpoint_poll_seconds,
        clock_ns=clock_ns,
        max_pending_utterances=max_pending_utterances,
    )


def test_dynamic_keyword_manifest_preserves_default_and_builds_exact_custom_path() -> None:
    assert keyword_manifest_for_phrase("呼叫指挥中心") == KEYWORDS_TEXT
    custom = keyword_manifest_for_phrase("灵序语音助手")
    assert custom.endswith("@灵序语音助手\n")
    assert "l íng" in custom
    assert "zh ù" in custom
    assert "呼叫指挥中心" not in custom


@pytest.mark.asyncio
async def test_kws_preflight_checks_pinned_digest_and_token_vocabulary(tmp_path: Path) -> None:
    model_dir = tmp_path / "kws-model"
    runtime_dir = tmp_path / "runtime-keywords"
    model_dir.mkdir()
    supported = "灵序语音助手"
    manifest_tokens = keyword_manifest_for_phrase(supported).partition(" @")[0].split()
    for filename in (
        "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
        "decoder-epoch-13-avg-2-chunk-8-left-64.onnx",
        "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
    ):
        (model_dir / filename).write_bytes(filename.encode())
    (model_dir / "tokens.txt").write_text(
        "\n".join(f"{token} {index}" for index, token in enumerate(manifest_tokens)),
        encoding="utf-8",
    )
    (model_dir / "lingxu-keywords.txt").write_text(KEYWORDS_TEXT, encoding="utf-8")
    spotter = SherpaOnnxKeywordSpotter(
        model_dir=model_dir,
        expected_model_digest=selected_model_digest(model_dir),
        keywords_root=runtime_dir,
    )

    await spotter.preflight(supported)
    with pytest.raises(ValueError, match="不支持的读音"):
        await spotter.preflight("麒麟凤凰饕餮")

    assert len(tuple(runtime_dir.glob("*.keywords.txt"))) == 1


@pytest.mark.asyncio
async def test_active_assistant_switches_wake_phrase_and_asr_hotword_without_restart() -> None:
    asr = FakeASRProvider("灵序语音助手，请说明当前流程。")
    answers = FakeAnswerProvider()
    spoken = asyncio.Event()
    spotter = FakeKeywordSpotter()

    async def speak(_text: str) -> None:
        spoken.set()

    selected = _assistant(
        asr=asr,
        answers=answers,
        speak=speak,
        keyword_spotter=spotter,
    )
    await selected.start()
    try:
        await selected.update_wake_phrase("灵序语音助手")
        assert selected.wake_phrase == "灵序语音助手"
        assert spotter.wake_phrase == "灵序语音助手"
        _feed_utterance(selected, 1, 1_000_000_000)
        await asyncio.wait_for(spoken.wait(), timeout=2)
        assert answers.questions == ["请说明当前流程。"]
        assert asr.hotwords == [("灵序语音助手",)]
    finally:
        await selected.stop()


@pytest.mark.asyncio
async def test_cancelled_wake_update_waits_for_spotter_and_assistant_convergence() -> None:
    spotter = BlockingUpdateKeywordSpotter()
    selected = _assistant(
        asr=FakeASRProvider(""),
        answers=FakeAnswerProvider(),
        speak=lambda _text: asyncio.sleep(0),
        keyword_spotter=spotter,
    )
    await selected.start()
    try:
        change = asyncio.create_task(selected.update_wake_phrase("灵序语音助手"))
        await asyncio.wait_for(spotter.update_entered.wait(), timeout=1)
        change.cancel()
        await asyncio.sleep(0)
        assert not change.done()

        spotter.update_release.set()
        with pytest.raises(asyncio.CancelledError):
            await change

        assert spotter.wake_phrase == "灵序语音助手"
        assert selected.wake_phrase == "灵序语音助手"
    finally:
        await selected.stop()


@pytest.mark.asyncio
async def test_explicit_prepare_then_start_does_not_repeat_dependency_preparation() -> None:
    spotter = FakeKeywordSpotter()
    asr = FakeASRProvider("")
    answers = FakeAnswerProvider()
    selected = _assistant(
        asr=asr,
        answers=answers,
        speak=lambda _text: asyncio.sleep(0),
        keyword_spotter=spotter,
    )

    await selected.prepare()
    await selected.prepare()
    await selected.start()
    await selected.stop()

    assert spotter.prepare_count == 1
    assert asr.prepare_count == 1
    assert answers.prepare_count == 1


@pytest.mark.asyncio
async def test_assistant_drains_frames_in_bounded_batches() -> None:
    queue = _BatchOnlyQueue()
    selected = _assistant(
        asr=FakeASRProvider(""),
        answers=FakeAnswerProvider(),
        speak=lambda _text: asyncio.sleep(0),
        frame_queue=queue,
    )

    await selected.start()
    await asyncio.sleep(0)
    await selected.stop()

    assert queue.batch_calls >= 1


@pytest.mark.asyncio
async def test_assistant_coalesces_overlapping_windows_and_rotates_users_fairly() -> None:
    spotter = BlockingKeywordSpotter()
    selected = _assistant(
        asr=FakeASRProvider(""),
        answers=FakeAnswerProvider(),
        speak=lambda _text: asyncio.sleep(0),
        keyword_spotter=spotter,
        max_pending_utterances=16,
    )

    await selected.start()
    try:
        assert selected.receive(
            user_id=1,
            received_monotonic_ns=1_000_000_000,
            pcm_s16le_48k_stereo=_VOICE,
        )
        await asyncio.wait_for(spotter.entered.wait(), timeout=2)
        for offset in range(1, 8):
            for user_id in range(1, 11):
                assert selected.receive(
                    user_id=user_id,
                    received_monotonic_ns=1_000_000_000 + offset * 20_000_000,
                    pcm_s16le_48k_stereo=_VOICE,
                )
        async with asyncio.timeout(2):
            while selected.metrics_snapshot().get("wake_windows_coalesced", 0) < 2:
                await asyncio.sleep(0.01)

        spotter.release.set()
        async with asyncio.timeout(2):
            while len(spotter.user_calls) < 11:
                await asyncio.sleep(0.01)

        assert spotter.user_calls[:11] == [1, *range(2, 11), 1]
        metrics = selected.metrics_snapshot()
        assert metrics["wake_windows_coalesced"] >= 2
        assert metrics.get("wake_windows_dropped_queue_full", 0) == 0
    finally:
        spotter.release.set()
        await selected.stop()


@pytest.mark.asyncio
async def test_completed_utterance_bypasses_wake_backlog_with_acoustic_fallback() -> None:
    spotter = CompletedFallbackKeywordSpotter()
    spoken = asyncio.Event()
    answers = FakeAnswerProvider()
    selected = _assistant(
        asr=FakeASRProvider(
            "\u547c\u53eb\u6307\u6325\u4e2d\u5fc3\uff0c\u8bf7\u8bf4\u660e\u5f53\u524d\u6d41\u7a0b\u3002"
        ),
        answers=answers,
        speak=lambda _text: asyncio.sleep(0, result=spoken.set()),
        keyword_spotter=spotter,
    )

    await selected.start()
    try:
        assert selected.receive(
            user_id=1,
            received_monotonic_ns=1_000_000_000,
            pcm_s16le_48k_stereo=_VOICE,
        )
        await asyncio.wait_for(spotter.entered.wait(), timeout=2)
        _feed_utterance(selected, 2, 2_000_000_000)
        spotter.release.set()

        await asyncio.wait_for(spoken.wait(), timeout=2)
        await _wait_state(selected, AssistantTurnState.IDLE)

        assert spotter.calls[1] == (2, len(_VOICE) + len(_SILENCE))
        assert answers.questions == ["\u8bf7\u8bf4\u660e\u5f53\u524d\u6d41\u7a0b\u3002"]
        metrics = selected.metrics_snapshot()
        assert metrics["utterance_wake_fallbacks_detected"] == 1
        assert metrics["answers_completed"] == 1
    finally:
        spotter.release.set()
        await selected.stop()


@pytest.mark.parametrize(
    ("text", "query"),
    [
        ("呼叫指挥中心，当前议程是什么？", "当前议程是什么？"),
        ("请问呼叫智慧中心，当前议程是什么？", "当前议程是什么？"),
        ("呼叫指中心，当前议程是什么？", "当前议程是什么？"),
        (
            "我们先讨论刚才的问题，然后呼叫指挥中心，请概括当前结论。",
            "请概括当前结论。",
        ),
    ],
)
def test_fuzzy_wake_occurrence_accepts_homophones_fillers_and_one_deletion(
    text: str,
    query: str,
) -> None:
    assert split_wake_query(text) == query


@pytest.mark.parametrize(
    "text",
    [
        "指挥中心，当前议程是什么？",
        "呼叫指挥，当前议程是什么？",
        "中心呼叫指挥，当前议程是什么？",
    ],
)
def test_fuzzy_wake_occurrence_rejects_short_or_reordered_phrases(text: str) -> None:
    assert split_wake_query(text) is None


def test_meeting_sink_routes_human_pcm_to_ephemeral_assistant_without_consent() -> None:
    calls: list[tuple[int, bytes]] = []
    pipeline = SimpleNamespace(
        metrics=SimpleNamespace(increment=lambda *_args: None),
        receive=lambda **_kwargs: None,
    )
    realtime_asr = SimpleNamespace(receive=lambda **_kwargs: False)
    assistant = SimpleNamespace(
        receive=lambda **kwargs: (
            calls.append((kwargs["user_id"], kwargs["pcm_s16le_48k_stereo"])) or True
        )
    )
    sink = MeetingPcmSink(
        pipeline=pipeline,  # type: ignore[arg-type]
        realtime_asr=realtime_asr,  # type: ignore[arg-type]
        ephemeral_assistant=assistant,  # type: ignore[arg-type]
        clock_ns=lambda: 123,
    )

    sink.write(  # type: ignore[arg-type]
        SimpleNamespace(id=42, bot=False),
        SimpleNamespace(pcm=bytearray(b"\x01\x00\x02\x00")),
    )
    sink.write(  # type: ignore[arg-type]
        SimpleNamespace(id=99, bot=True),
        SimpleNamespace(pcm=b"\x03\x00\x04\x00"),
    )

    assert calls == [(42, b"\x01\x00\x02\x00")]


def test_wake_window_builder_emits_overlapping_bounded_windows() -> None:
    builder = WakeWindowBuilder(
        session_id="session-1",
        activity_threshold=100,
        window_ms=60,
        hop_ms=20,
        min_window_ms=40,
        preroll_ms=20,
    )

    assert builder.accept(SessionAudioFrame("session-1", 1, 1_000_000_000, _VOICE)) is None
    first = builder.accept(SessionAudioFrame("session-1", 1, 1_020_000_000, _VOICE))
    second = builder.accept(SessionAudioFrame("session-1", 1, 1_040_000_000, _VOICE))
    third = builder.accept(SessionAudioFrame("session-1", 1, 1_060_000_000, _VOICE))

    assert first is not None
    assert second is not None
    assert third is not None
    assert third.started_monotonic_ns == 1_020_000_000
    assert len(third.pcm_s16le_48k_stereo) == 3 * len(_VOICE)


def test_wake_window_builder_keeps_low_energy_preroll_before_speech() -> None:
    builder = WakeWindowBuilder(
        session_id="session-1",
        activity_threshold=100,
        window_ms=100,
        hop_ms=20,
        min_window_ms=40,
        preroll_ms=40,
    )

    assert builder.accept(SessionAudioFrame("session-1", 1, 1_000_000_000, _SILENCE)) is None
    window = builder.accept(SessionAudioFrame("session-1", 1, 1_020_000_000, _VOICE))

    assert window is not None
    assert window.started_monotonic_ns == 1_000_000_000
    assert window.pcm_s16le_48k_stereo == _SILENCE + _VOICE


@pytest.mark.asyncio
async def test_assistant_drops_new_invocations_until_full_answer_finishes() -> None:
    answers = FakeAnswerProvider()
    playback_entered = asyncio.Event()
    playback_release = asyncio.Event()
    selected: MeetingVoiceAssistant

    async def speak(_text: str) -> None:
        playback_entered.set()
        assert not selected.receive(
            user_id=2,
            received_monotonic_ns=2_000_000_000,
            pcm_s16le_48k_stereo=_VOICE,
        )
        await playback_release.wait()

    selected = _assistant(
        asr=FakeASRProvider("呼叫指挥中心，现在到哪个议程？"),
        answers=answers,
        speak=speak,
    )
    await selected.start()
    try:
        _feed_utterance(selected, 1, 1_000_000_000)
        await asyncio.wait_for(playback_entered.wait(), timeout=2)
        assert selected.state is AssistantTurnState.SPEAKING
        assert answers.questions == ["现在到哪个议程？"]
        playback_release.set()
        await _wait_state(selected, AssistantTurnState.IDLE)
        assert selected.metrics_snapshot()["frames_dropped_busy"] >= 1
    finally:
        playback_release.set()
        await selected.stop()


@pytest.mark.asyncio
async def test_streaming_wake_claims_mid_sentence_before_user_finishes() -> None:
    answers = FakeAnswerProvider()
    spoken = asyncio.Event()

    async def speak(_text: str) -> None:
        spoken.set()

    selected = _assistant(
        asr=FakeASRProvider("我们先确认背景，随后呼叫指挥中心，请说明当前流程。"),
        answers=answers,
        speak=speak,
    )
    await selected.start()
    try:
        assert selected.receive(
            user_id=1,
            received_monotonic_ns=1_000_000_000,
            pcm_s16le_48k_stereo=_VOICE,
        )
        assert selected.receive(
            user_id=1,
            received_monotonic_ns=1_020_000_000,
            pcm_s16le_48k_stereo=_VOICE,
        )
        await _wait_state(selected, AssistantTurnState.CAPTURING_QUERY)
        assert answers.questions == []

        assert selected.receive(
            user_id=1,
            received_monotonic_ns=1_040_000_000,
            pcm_s16le_48k_stereo=_SILENCE,
        )
        await asyncio.wait_for(spoken.wait(), timeout=2)
        await _wait_state(selected, AssistantTurnState.IDLE)
        assert answers.questions == ["请说明当前流程。"]
        metrics = selected.metrics_snapshot()
        assert metrics["wake_windows_detected"] == 1
        assert metrics["wake_turns_claimed"] == 1
    finally:
        await selected.stop()


@pytest.mark.asyncio
async def test_assistant_answers_after_wall_clock_silence_without_another_pcm_frame() -> None:
    answers = FakeAnswerProvider()
    asr = FakeASRProvider(
        "\u547c\u53eb\u6307\u6325\u4e2d\u5fc3\uff0c\u8bf7\u8bf4\u660e\u5f53\u524d\u6d41\u7a0b\u3002"
    )
    spoken = asyncio.Event()
    now_ns = [1_000_000_000]

    async def speak(_text: str) -> None:
        spoken.set()

    selected = _assistant(
        asr=asr,
        answers=answers,
        speak=speak,
        vad_end_ms=800,
        endpoint_poll_seconds=0.005,
        clock_ns=lambda: now_ns[0],
    )
    await selected.start()
    try:
        assert asr.prepare_count == 1
        assert selected.receive(
            user_id=1,
            received_monotonic_ns=1_000_000_000,
            pcm_s16le_48k_stereo=_VOICE,
        )
        assert selected.receive(
            user_id=1,
            received_monotonic_ns=1_020_000_000,
            pcm_s16le_48k_stereo=_VOICE,
        )
        await _wait_state(selected, AssistantTurnState.CAPTURING_QUERY)

        now_ns[0] = 1_840_000_000
        await asyncio.wait_for(spoken.wait(), timeout=1)
        await _wait_state(selected, AssistantTurnState.IDLE)

        assert answers.questions == ["\u8bf7\u8bf4\u660e\u5f53\u524d\u6d41\u7a0b\u3002"]
        assert selected.metrics_snapshot()["utterances_finalized_by_clock"] == 1
    finally:
        await selected.stop()


@pytest.mark.asyncio
async def test_bare_wake_arms_one_caller_followup_without_queueing_others() -> None:
    answers = FakeAnswerProvider()
    spoken = asyncio.Event()

    async def speak(_text: str) -> None:
        spoken.set()

    selected = _assistant(
        asr=FakeASRProvider("呼叫指挥中心", "请说明当前流程"),
        answers=answers,
        speak=speak,
        followup_seconds=1,
    )
    await selected.start()
    try:
        _feed_utterance(selected, 1, 1_000_000_000)
        await _wait_state(selected, AssistantTurnState.AWAITING_QUERY)
        assert not selected.receive(
            user_id=2,
            received_monotonic_ns=2_000_000_000,
            pcm_s16le_48k_stereo=_VOICE,
        )
        _feed_utterance(selected, 1, 3_000_000_000)
        await asyncio.wait_for(spoken.wait(), timeout=2)
        await _wait_state(selected, AssistantTurnState.IDLE)
        assert answers.questions == ["请说明当前流程"]
    finally:
        await selected.stop()


def test_short_answer_hard_limits_sentences_and_length() -> None:
    chinese = "第一句。第二句！第三句？" + "很" * 100
    english = " ".join(f"word{index}" for index in range(50))

    limited_chinese = limit_short_answer(chinese)
    limited_english = limit_short_answer(english)

    assert limited_chinese == "第一句。第二句！"
    assert len(limited_english.rstrip(".").split()) == 35


async def _ollama_server(
    *,
    digest: str = _DIGEST,
) -> tuple[TestServer, list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []

    async def tags(_request: web.Request) -> web.Response:
        return web.json_response({"models": [{"name": "qwen3.5:9b", "digest": digest}]})

    async def chat(request: web.Request) -> web.Response:
        requests.append(await request.json())
        content = json.dumps({"language": "zh", "answer": "当前正在讨论第一项议程。"})
        return web.json_response({"message": {"content": content}})

    app = web.Application()
    app.router.add_get("/api/tags", tags)
    app.router.add_post("/api/chat", chat)
    server = TestServer(app)
    await server.start_server()
    return server, requests


@pytest.mark.asyncio
async def test_short_answer_provider_pins_model_schema_and_has_no_tools() -> None:
    server, requests = await _ollama_server()
    try:
        provider = OllamaShortAnswerProvider(
            base_url=str(server.make_url("/v1")).rstrip("/"),
            model="qwen3.5:9b",
            expected_model_digest=_DIGEST,
        )
        answer = await provider.answer(
            question="当前议程是什么？",
            context={"meeting_state": "SPEAKER_ACTIVE"},
        )
    finally:
        await server.close()

    assert answer == "当前正在讨论第一项议程。"
    assert requests[0]["think"] is False
    assert requests[0]["stream"] is False
    assert requests[0]["options"]["num_predict"] == 64
    assert "tools" not in requests[0]
    assert requests[0]["format"]["additionalProperties"] is False
    meeting_user = json.loads(requests[0]["messages"][1]["content"])
    assert meeting_user["meeting_context"] == {"meeting_state": "SPEAKER_ACTIVE"}


@pytest.mark.asyncio
async def test_resident_short_answer_has_only_fixed_channel_context_and_no_tools() -> None:
    server, requests = await _ollama_server()
    try:
        provider = OllamaShortAnswerProvider(
            base_url=str(server.make_url("/v1")).rstrip("/"),
            model="qwen3.5:9b",
            expected_model_digest=_DIGEST,
            role="resident",
        )
        await provider.answer(
            question="请简要回答这个问题。",
            context={"mode": "resident_voice_qa"},
        )
    finally:
        await server.close()

    payload = requests[0]
    assert "tools" not in payload
    assert "Discord操作" in payload["messages"][0]["content"]
    user_payload = json.loads(payload["messages"][1]["content"])
    assert user_payload == {
        "channel_context": {"mode": "resident_voice_qa"},
        "question": "请简要回答这个问题。",
    }
    assert "meeting_context" not in user_payload


@pytest.mark.asyncio
async def test_short_answer_provider_rejects_unapproved_digest_before_chat() -> None:
    server, requests = await _ollama_server(digest="b" * 64)
    try:
        provider = OllamaShortAnswerProvider(
            base_url=str(server.make_url("/v1")).rstrip("/"),
            model="qwen3.5:9b",
            expected_model_digest=_DIGEST,
        )
        with pytest.raises(VoiceAssistantProviderError):
            await provider.answer(question="问题", context={})
    finally:
        await server.close()
    assert requests == []
