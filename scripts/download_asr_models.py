"""Download and verify the pinned SenseVoice and FSMN-VAD snapshots."""

from __future__ import annotations

import asyncio

from lingxu.asr.factory import create_realtime_asr_provider
from lingxu.config import Settings


async def _main() -> None:
    provider = create_realtime_asr_provider(Settings(), allow_download=True)
    prepare = getattr(provider, "prepare", None)
    if prepare is None:
        raise RuntimeError("configured realtime ASR provider cannot prepare models")
    await prepare()
    print("Lingxu SenseVoice and FSMN-VAD snapshots are installed and verified.")


if __name__ == "__main__":
    asyncio.run(_main())
