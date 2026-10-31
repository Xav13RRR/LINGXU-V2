"""Content-free benchmark for per-frame versus bounded-batch queue draining."""

from __future__ import annotations

import asyncio
import json
import time

from lingxu.asr.queue import PerUserRealtimeQueue
from lingxu.voice.frames import SessionAudioFrame

_FRAME_COUNT = 4_000
_PCM = b"\0" * 3_840


async def _measure(batch_size: int) -> float:
    frames = PerUserRealtimeQueue(max_frames_per_user=_FRAME_COUNT)
    for index in range(_FRAME_COUNT):
        frames.put(
            SessionAudioFrame(
                session_id="benchmark",
                user_id=1,
                received_monotonic_ns=index * 20_000_000,
                pcm_s16le_48k_stereo=_PCM,
            )
        )
    frames.close()
    started = time.perf_counter()
    drained = 0
    while True:
        batch = await asyncio.to_thread(frames.get_many, batch_size)
        if batch is None:
            break
        drained += len(batch)
    if drained != _FRAME_COUNT:
        raise RuntimeError("frame-drain benchmark lost input")
    return time.perf_counter() - started


async def _run() -> dict[str, object]:
    per_frame = await _measure(1)
    batched = await _measure(64)
    return {
        "frames": _FRAME_COUNT,
        "per_frame_seconds": round(per_frame, 6),
        "batch_64_seconds": round(batched, 6),
        "speedup": round(per_frame / max(batched, 0.000001), 2),
    }


def main() -> None:
    print(json.dumps(asyncio.run(_run()), separators=(",", ":")))


if __name__ == "__main__":
    main()
