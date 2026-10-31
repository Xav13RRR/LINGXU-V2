import asyncio
import io
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from lingxu.tts.provider import CosyVoiceHttpProvider, EspeakNgProvider, normalise_tts_text


def _fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    output_path = Path(command[command.index("-w") + 1])
    with wave.open(str(output_path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(22_050)
        target.writeframes(b"\0\0" * 2_205)
    return subprocess.CompletedProcess(command, 0, "", "")


@pytest.mark.asyncio
async def test_provider_uses_stdin_atomic_wav_and_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def record_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return _fake_run(command, **kwargs)

    monkeypatch.setattr("lingxu.tts.provider.shutil.which", lambda _: "espeak-ng")
    monkeypatch.setattr("lingxu.tts.provider.subprocess.run", record_run)
    provider = EspeakNgProvider(cache_dir=tmp_path)

    first = await provider.synthesize("  聆序   开始会议  ")
    second = await provider.synthesize("聆序 开始会议")

    assert first.from_cache is False
    assert second.from_cache is True
    assert first.path == second.path
    assert first.duration_ms == 100
    assert len(calls) == 1
    assert calls[0][1]["input"] == "聆序 开始会议"
    assert "聆序" not in " ".join(calls[0][0])
    assert not await asyncio.to_thread(lambda: tuple(tmp_path.glob("*.part")))


@pytest.mark.asyncio
async def test_espeak_ephemeral_synthesis_never_enters_persistent_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setattr("lingxu.tts.provider.shutil.which", lambda _: "espeak-ng")
    monkeypatch.setattr("lingxu.tts.provider.subprocess.run", _fake_run)
    provider = EspeakNgProvider(cache_dir=cache_dir)

    audio = await provider.synthesize_ephemeral("ephemeral answer", output_dir=runtime_dir)

    assert audio.path.parent == runtime_dir.resolve()
    assert audio.path.is_file()
    assert not cache_dir.exists()
    audio.path.unlink()


@pytest.mark.asyncio
async def test_text_normalisation_and_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lingxu.tts.provider.shutil.which", lambda _: "espeak-ng")
    provider = EspeakNgProvider(cache_dir=tmp_path)

    assert normalise_tts_text("Ａ  B") == "A B"  # noqa: RUF001
    assert normalise_tts_text("共有1项\uff0c预计10分钟。") == "共有一项,预计十分钟。"
    assert normalise_tts_text("Version 21 takes 5.25 minutes.") == (
        "Version twenty one takes five point two five minutes."
    )
    assert normalise_tts_text("予定時間は15分です。") == "予定時間は十五分です。"
    assert normalise_tts_text("예정 시간은 15분입니다.") == "예정 시간은 십오분입니다."
    with pytest.raises(ValueError, match="speed"):
        EspeakNgProvider(cache_dir=tmp_path, speed=3.0)
    with pytest.raises(ValueError, match="blank"):
        await provider.synthesize("")


def _test_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24_000)
        target.writeframes(b"\0\0" * 2_400)
    return output.getvalue()


@pytest.mark.asyncio
async def test_cosyvoice_provider_verifies_digest_and_caches(
    tmp_path: Path,
) -> None:
    digest = "a" * 64
    requests: list[dict[str, object]] = []

    async def synthesize(request: web.Request) -> web.Response:
        requests.append(await request.json())
        return web.Response(
            body=_test_wav(),
            content_type="audio/wav",
            headers={"X-Lingxu-TTS-Model-Digest": digest},
        )

    application = web.Application()
    application.router.add_post("/v1/synthesize", synthesize)
    runner = web.AppRunner(application, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    server = site._server
    assert server is not None
    port = server.sockets[0].getsockname()[1]
    provider = CosyVoiceHttpProvider(
        cache_dir=tmp_path,
        base_url=f"http://127.0.0.1:{port}",
        model="FunAudioLLM/CosyVoice-300M-SFT",
        model_revision="b" * 40,
        model_digest=digest,
        voice="中文女",
    )
    try:
        first = await provider.synthesize("  聆序   开始会议  ")
        second = await provider.synthesize("聆序 开始会议")
    finally:
        await runner.cleanup()

    assert first.from_cache is False
    assert second.from_cache is True
    assert first.path == second.path
    assert first.duration_ms == 100
    assert requests == [{"text": "聆序 开始会议", "voice": "中文女", "speed": 1.0}]
    assert not await asyncio.to_thread(lambda: tuple(tmp_path.glob("*.part")))


@pytest.mark.asyncio
async def test_cosyvoice_ephemeral_synthesis_never_enters_persistent_cache(
    tmp_path: Path,
) -> None:
    digest = "a" * 64

    async def synthesize(_: web.Request) -> web.Response:
        return web.Response(
            body=_test_wav(),
            content_type="audio/wav",
            headers={"X-Lingxu-TTS-Model-Digest": digest},
        )

    application = web.Application()
    application.router.add_post("/v1/synthesize", synthesize)
    runner = web.AppRunner(application, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    server = site._server
    assert server is not None
    port = server.sockets[0].getsockname()[1]
    cache_dir = tmp_path / "cache"
    runtime_dir = tmp_path / "runtime"
    provider = CosyVoiceHttpProvider(
        cache_dir=cache_dir,
        base_url=f"http://127.0.0.1:{port}",
        model="FunAudioLLM/CosyVoice-300M-SFT",
        model_revision="b" * 40,
        model_digest=digest,
        voice="中文女",
    )
    try:
        audio = await provider.synthesize_ephemeral("临时回答", output_dir=runtime_dir)
    finally:
        await runner.cleanup()

    assert audio.path.parent == runtime_dir.resolve()
    assert audio.path.is_file()
    assert not cache_dir.exists()
    audio.path.unlink()


@pytest.mark.asyncio
async def test_cosyvoice_provider_verbalises_numbers_before_cache_and_request(
    tmp_path: Path,
) -> None:
    digest = "a" * 64
    requests: list[dict[str, object]] = []

    async def synthesize(request: web.Request) -> web.Response:
        requests.append(await request.json())
        return web.Response(
            body=_test_wav(),
            content_type="audio/wav",
            headers={"X-Lingxu-TTS-Model-Digest": digest},
        )

    application = web.Application()
    application.router.add_post("/v1/synthesize", synthesize)
    runner = web.AppRunner(application, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    server = site._server
    assert server is not None
    port = server.sockets[0].getsockname()[1]
    provider = CosyVoiceHttpProvider(
        cache_dir=tmp_path,
        base_url=f"http://127.0.0.1:{port}",
        model="FunAudioLLM/CosyVoice-300M-SFT",
        model_revision="b" * 40,
        model_digest=digest,
        voice="中文女",
    )
    try:
        numeric = await provider.synthesize("下面介绍本次会议议程, 共1项。")
        spoken = await provider.synthesize("下面介绍本次会议议程, 共一项。")
    finally:
        await runner.cleanup()

    assert numeric.path == spoken.path
    assert spoken.from_cache is True
    assert requests == [{"text": "下面介绍本次会议议程, 共一项。", "voice": "中文女", "speed": 1.0}]
