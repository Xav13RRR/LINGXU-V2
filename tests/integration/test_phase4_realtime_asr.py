import struct
from pathlib import Path

import pytest

from lingxu.asr.models import ASRResult, UtteranceAudio
from lingxu.asr.queue import PerUserRealtimeQueue
from lingxu.asr.service import RealtimeASRMetrics, RealtimeASRService
from lingxu.asr.utterance import UtteranceBuilder
from lingxu.storage.database import Database
from lingxu.storage.repositories import (
    MeetingSessionRepository,
    RealtimeUtteranceRepository,
)
from lingxu.storage.services import MeetingSessionService
from lingxu.storage.session_files import SessionFiles
from lingxu.voice.meeting_consent import MeetingConsentRegistry

_BASE_NS = 5_000_000_000
_FRAME_NS = 20_000_000
_VOICE = struct.pack("<hh", 2_000, -2_000) * 960
_SILENCE = b"\0" * len(_VOICE)


class ReplayProvider:
    async def transcribe(
        self,
        utterance: UtteranceAudio,
        *,
        language: str,
        hotwords: tuple[str, ...],
    ) -> ASRResult:
        values = {
            101: ("<|zh|>中文 fixture", "中文 fixture", "zh"),
            202: ("<|en|>English fixture", "English fixture", "en"),
            303: ("<|zh|>中英 mixed fixture", "中英 mixed fixture", "zh"),
        }
        raw, normalised, detected = values[utterance.user_id]
        return ASRResult(raw, normalised, detected, 5)


def _database_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.as_posix()}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_overlap_replay_persists_raw_and_normalised_per_user(tmp_path: Path) -> None:
    database = Database.open(_database_url(tmp_path / "asr.db"))
    await database.create_schema()
    session = await MeetingSessionService(
        MeetingSessionRepository(database.sessions),
        SessionFiles(tmp_path / "sessions"),
    ).create(guild_id=44, creator_user_id=55)
    utterances = RealtimeUtteranceRepository(database.sessions)
    consents = MeetingConsentRegistry()
    service = RealtimeASRService(
        session_id=session.id,
        session_started_monotonic_ns=_BASE_NS,
        consents=consents,
        frame_queue=PerUserRealtimeQueue(max_frames_per_user=50),
        utterance_builder=UtteranceBuilder(
            session_id=session.id,
            vad_threshold=100,
            vad_start_ms=40,
            vad_end_ms=40,
            min_utterance_ms=40,
        ),
        provider=ReplayProvider(),
        utterances=utterances,
        metrics=RealtimeASRMetrics(session.id),
        hotwords=("聆序",),
        clock_ns=lambda: _BASE_NS + 2_000_000_000,
    )
    await service.start()
    for user_id in (101, 202, 303):
        service.grant(user_id)

    # Replay three simultaneous speakers frame-by-frame to prove stream isolation.
    for index, payload in enumerate((_VOICE, _VOICE, _VOICE, _SILENCE, _SILENCE)):
        for user_id in (101, 202, 303):
            assert service.receive(
                user_id=user_id,
                received_monotonic_ns=_BASE_NS + index * _FRAME_NS,
                pcm_s16le_48k_stereo=payload,
            )
    await service.stop()

    chinese = await utterances.list_user(session_id=session.id, user_id=101)
    english = await utterances.list_user(session_id=session.id, user_id=202)
    mixed = await utterances.list_user(session_id=session.id, user_id=303)
    assert [(row.raw_text, row.normalised_text) for row in chinese] == [
        ("<|zh|>中文 fixture", "中文 fixture")
    ]
    assert [(row.raw_text, row.normalised_text) for row in english] == [
        ("<|en|>English fixture", "English fixture")
    ]
    assert [(row.raw_text, row.normalised_text) for row in mixed] == [
        ("<|zh|>中英 mixed fixture", "中英 mixed fixture")
    ]
    assert {chinese[0].user_id, english[0].user_id, mixed[0].user_id} == {101, 202, 303}
    assert service.metrics.snapshot()["counters"]["utterances_final"] == 3

    await database.close()
