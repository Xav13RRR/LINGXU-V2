"""Content-safe Phase 5 offline Mandarin TTS benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

from lingxu.config import Settings
from lingxu.tts.factory import COMMON_PROCESS_PROMPTS, create_tts_provider


def _percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


async def _run(work_dir: Path, iterations: int) -> dict[str, Any]:
    await asyncio.to_thread(work_dir.mkdir, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lingxu-tts-benchmark-", dir=work_dir) as temporary:
        settings = Settings().model_copy(update={"tts_cache_dir": Path(temporary)})
        provider = create_tts_provider(settings)
        fixed_started = time.perf_counter_ns()
        fixed = await provider.pre_generate(COMMON_PROCESS_PROMPTS)
        fixed_ms = max(0, (time.perf_counter_ns() - fixed_started) // 1_000_000)
        latencies = []
        audio = None
        for index in range(iterations):
            started = time.perf_counter_ns()
            audio = await provider.synthesize(f"请第{index + 1}位参与者准备讨论会议议程。")
            latencies.append(max(0, (time.perf_counter_ns() - started) // 1_000_000))
        assert audio is not None
        warm_started = time.perf_counter_ns()
        warm = await provider.synthesize(f"请第{iterations}位参与者准备讨论会议议程。")
        warm_ms = max(0, (time.perf_counter_ns() - warm_started) // 1_000_000)
        p50 = _percentile(latencies, 0.50)
        p95 = _percentile(latencies, 0.95)
        return {
            "provider": settings.tts_provider,
            "voice": settings.tts_voice,
            "fixed_prompt_count": len(fixed),
            "fixed_all_valid": all(item.duration_ms > 0 for item in fixed),
            "dynamic_iterations": iterations,
            "dynamic_all_valid": audio.duration_ms > 0,
            "cold_latency_ms": {
                "p50": p50,
                "p95": p95,
                "max": max(latencies),
            },
            "warm_cache_latency_ms": warm_ms,
            "warm_cache_hit": warm.from_cache,
            "pregenerate_total_ms": fixed_ms,
            "acceptance": {
                "fixed_prompts": len(fixed) == len(COMMON_PROCESS_PROMPTS),
                "dynamic_synthesis": audio.duration_ms > 0,
                "cache": warm.from_cache,
                "cold_p95_le_20000_ms": p95 <= 20_000,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path(".runtime"))
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")
    result = asyncio.run(_run(args.work_dir, args.iterations))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not all(result["acceptance"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
