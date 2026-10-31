"""Content-safe retained-audio replay for the latest Meeting assistant wake attempts."""

from __future__ import annotations

import asyncio
import io
import json
import sqlite3
from pathlib import Path

import numpy as np
import soundfile as sf
from sqlalchemy.engine import make_url

from lingxu.asr.models import UtteranceAudio
from lingxu.config import Settings
from lingxu.storage.encryption import FileCipher
from lingxu.storage.session_files import SessionFiles
from lingxu.voice_assistant.fuzzy import split_wake_query
from lingxu.voice_assistant.wakeword import SherpaOnnxKeywordSpotter

_QUARANTINE_MS = 600
_REPLAY_CONTEXT_MS = 1_000


async def _run() -> dict[str, object]:
    settings = Settings()
    database_path = make_url(settings.database_url).database
    if not database_path:
        raise RuntimeError("assistant analysis requires the local SQLite deployment")
    connection = sqlite3.connect(database_path)
    try:
        session = connection.execute(
            "SELECT id FROM meeting_sessions ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if session is None:
            raise RuntimeError("no retained Meeting is available")
        session_id = str(session[0])
        utterances = connection.execute(
            """
            SELECT user_id, started_ms, ended_ms, normalised_text, raw_text
            FROM meeting_realtime_utterances
            WHERE session_id = ?
            ORDER BY started_ms
            """,
            (session_id,),
        ).fetchall()
        transcript = connection.execute(
            """
            SELECT started_ms, ended_ms, source_key
            FROM transcript_entries
            WHERE session_id = ?
            ORDER BY started_ms
            """,
            (session_id,),
        ).fetchall()
        segments = connection.execute(
            """
            SELECT user_id, started_ms, ended_ms, encrypted_path
            FROM audio_segments
            WHERE session_id = ?
            ORDER BY started_ms
            """,
            (session_id,),
        ).fetchall()
    finally:
        connection.close()

    wake_hits = [
        (int(user_id), int(started_ms), int(ended_ms))
        for user_id, started_ms, ended_ms, normalised_text, raw_text in utterances
        if split_wake_query(normalised_text or raw_text) is not None
    ]
    bot_intervals = [
        (int(started_ms), int(ended_ms))
        for started_ms, ended_ms, source_key in transcript
        if str(source_key).startswith("BOT_TTS:")
    ]
    blocked = [
        hit
        for hit in wake_hits
        if any(
            hit[1] < bot_end + _QUARANTINE_MS and hit[2] > bot_start
            for bot_start, bot_end in bot_intervals
        )
    ]
    eligible = [hit for hit in wake_hits if hit not in blocked]

    cipher = FileCipher.from_secret(settings.file_encryption_key)
    files = SessionFiles(settings.data_root)
    spotter = SherpaOnnxKeywordSpotter(
        model_dir=settings.meeting_assistant_wakeword_model_dir,
        expected_model_digest=settings.meeting_assistant_wakeword_model_digest,
        keywords_score=settings.meeting_assistant_wakeword_score,
        keywords_threshold=settings.meeting_assistant_wakeword_threshold,
    )
    await spotter.prepare()
    replayed = 0
    detected = 0
    for user_id, started_ms, ended_ms in eligible:
        candidates = [
            segment
            for segment in segments
            if int(segment[0]) == user_id
            and int(segment[1]) <= started_ms
            and int(segment[2]) >= ended_ms
        ]
        if not candidates:
            continue
        segment = min(candidates, key=lambda value: int(value[2]) - int(value[1]))
        segment_start = int(segment[1])
        segment_end = int(segment[2])
        encrypted_path = files.resolve(session_id, Path(str(segment[3])))
        plaintext = cipher.decrypt(encrypted_path.read_bytes())
        samples, sample_rate = sf.read(
            io.BytesIO(plaintext),
            dtype="int16",
            always_2d=True,
        )
        if sample_rate != 48_000:
            raise RuntimeError("retained assistant replay requires 48 kHz audio")
        if samples.shape[1] == 1:
            samples = np.repeat(samples, 2, axis=1)
        elif samples.shape[1] > 2:
            samples = samples[:, :2]
        selected_start = max(segment_start, started_ms - _REPLAY_CONTEXT_MS)
        selected_end = min(segment_end, ended_ms + _REPLAY_CONTEXT_MS)
        first_sample = max(0, (selected_start - segment_start) * sample_rate // 1_000)
        last_sample = min(
            len(samples),
            (selected_end - segment_start) * sample_rate // 1_000,
        )
        pcm = samples[first_sample:last_sample].astype("<i2", copy=False).tobytes()
        if not pcm:
            continue
        replayed += 1
        if await spotter.detect(
            UtteranceAudio(
                session_id=session_id,
                user_id=user_id,
                started_monotonic_ns=selected_start * 1_000_000,
                ended_monotonic_ns=selected_end * 1_000_000,
                pcm_s16le_48k_stereo=pcm,
            )
        ):
            detected += 1
    return {
        "realtime_wake_hits": len(wake_hits),
        "blocked_by_bot_output_or_quarantine": len(blocked),
        "eligible_wake_hits": len(eligible),
        "eligible_audio_replayed": replayed,
        "eligible_acoustic_detections": detected,
    }


def main() -> None:
    print(json.dumps(asyncio.run(_run()), separators=(",", ":")))


if __name__ == "__main__":
    main()
