"""Run the Phase 4 acceptance replay against the real SenseVoice model."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from imageio_ffmpeg import get_ffmpeg_exe

from lingxu.asr.models import UtteranceAudio
from lingxu.asr.provider import SenseVoiceProvider

_SESSION_ID = "phase4-benchmark"
_USER_ID = 1
_SENSEVOICE_REPOSITORY_HEAD = "7bf452403abd7353a300cd760f7adae7701c92c1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_48k_stereo(path: Path) -> bytes:
    completed = subprocess.run(
        [
            get_ffmpeg_exe(),
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "2",
            "-ar",
            "48000",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _utterance(pcm: bytes) -> UtteranceAudio:
    duration_ns = len(pcm) * 1_000_000_000 // (48_000 * 4)
    return UtteranceAudio(
        session_id=_SESSION_ID,
        user_id=_USER_ID,
        started_monotonic_ns=0,
        ended_monotonic_ns=duration_ns,
        pcm_s16le_48k_stereo=pcm,
    )


def _percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


async def _run(cache_dir: Path, iterations: int) -> dict[str, Any]:
    provider = SenseVoiceProvider(model_cache_dir=cache_dir, allow_download=True)
    model = await asyncio.to_thread(provider._load_model)
    model_path = await asyncio.to_thread(Path(model.model_path).resolve)
    chinese_path = model_path / "example" / "zh.mp3"
    english_path = model_path / "example" / "en.mp3"
    if not chinese_path.is_file() or not english_path.is_file():
        raise FileNotFoundError("official SenseVoice zh/en examples are missing")
    chinese_pcm = _decode_48k_stereo(chinese_path)
    english_pcm = _decode_48k_stereo(english_path)
    fixtures = {
        "zh": chinese_pcm,
        "en": english_pcm,
        "mixed": chinese_pcm + (b"\0" * (48_000 * 4 // 2)) + english_pcm,
    }

    # Pre-warm model kernels; this result is deliberately excluded from the benchmark.
    await provider.transcribe(_utterance(chinese_pcm), language="auto", hotwords=("聆序",))

    latencies: list[int] = []
    case_results: dict[str, dict[str, Any]] = {}
    for name, pcm in fixtures.items():
        final_result = None
        case_latencies = []
        for _ in range(iterations):
            started_ns = time.perf_counter_ns()
            final_result = await provider.transcribe(
                _utterance(pcm),
                language="auto",
                hotwords=("聆序", "GRS Studio"),
            )
            latency_ms = max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
            case_latencies.append(latency_ms)
            latencies.append(latency_ms)
        assert final_result is not None
        text = final_result.normalised_text
        case_results[name] = {
            "detected_language": final_result.language,
            "duration_ms": len(pcm) * 1_000 // (48_000 * 4),
            "final_nonempty": bool(text),
            "contains_cjk": any("\u3400" <= character <= "\u9fff" for character in text),
            "contains_latin": any(
                character.isascii() and character.isalpha() for character in text
            ),
            "latency_ms": case_latencies,
        }

    p50 = _percentile(latencies, 0.50)
    p95 = _percentile(latencies, 0.95)
    return {
        "provider": "SenseVoiceProvider",
        "model": "iic/SenseVoiceSmall",
        "model_revision": "master",
        "model_repository_head_at_implementation": _SENSEVOICE_REPOSITORY_HEAD,
        "model_pt_sha256": _sha256(model_path / "model.pt"),
        "vad_model": "fsmn-vad",
        "vad_model_revision": "v2.0.4",
        "vad_model_pt_sha256": _sha256(Path(str(model.vad_kwargs["model_path"])) / "model.pt"),
        "device": "cpu",
        "iterations_per_case": iterations,
        "fixtures": {
            "zh_sha256": _sha256(chinese_path),
            "en_sha256": _sha256(english_path),
        },
        "cases": case_results,
        "latency_ms": {"p50": p50, "p95": p95, "max": max(latencies)},
        "acceptance": {
            "zh_final": case_results["zh"]["final_nonempty"],
            "en_final": case_results["en"]["final_nonempty"],
            "mixed_final": (
                case_results["mixed"]["final_nonempty"]
                and case_results["mixed"]["contains_cjk"]
                and case_results["mixed"]["contains_latin"]
            ),
            "p95_le_4000_ms": p95 <= 4_000,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")
    result = asyncio.run(_run(args.cache_dir, args.iterations))
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not all(result["acceptance"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
