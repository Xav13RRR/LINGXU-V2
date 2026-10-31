"""Pre-generate fixed Phase 5 process prompts without printing their text."""

from __future__ import annotations

import asyncio
import json

from lingxu.config import Settings
from lingxu.tts.factory import COMMON_PROCESS_PROMPTS, create_tts_provider


async def _run() -> dict[str, object]:
    settings = Settings()
    provider = create_tts_provider(settings)
    audio = await provider.pre_generate(COMMON_PROCESS_PROMPTS)
    return {
        "provider": settings.tts_provider,
        "voice": settings.tts_voice,
        "prompt_count": len(audio),
        "all_valid": all(item.duration_ms > 0 for item in audio),
        "cache_hits": sum(item.from_cache for item in audio),
    }


def main() -> None:
    result = asyncio.run(_run())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
