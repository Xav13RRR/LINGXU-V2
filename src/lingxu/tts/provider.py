"""Replaceable TTS provider and offline Mandarin espeak-ng implementation."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import secrets
import shutil
import subprocess
import threading
import unicodedata
import wave
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import urljoin

import aiohttp

from lingxu.tts.models import TTSAudio

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_HIRAGANA_KATAKANA = re.compile(r"[\u3040-\u30ff]")
_HANGUL = re.compile(r"[\uac00-\ud7af]")
_CJK = re.compile(r"[\u3400-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")

_ZH_DIGITS = "零一二三四五六七八九"
_KO_DIGITS = "영일이삼사오육칠팔구"
_EN_SMALL = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_EN_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)


class TTSProvider(Protocol):
    async def synthesize(self, text: str) -> TTSAudio:
        """Synthesize or load one validated cache artifact."""

    async def pre_generate(self, prompts: tuple[str, ...]) -> tuple[TTSAudio, ...]:
        """Generate a fixed prompt set through the same validated cache."""


@runtime_checkable
class EphemeralTTSProvider(Protocol):
    async def synthesize_ephemeral(self, text: str, *, output_dir: Path) -> TTSAudio:
        """Synthesize one WAV that the caller must erase immediately after playback."""


def _cjk_integer(value: str, *, korean: bool = False) -> str:
    digits = _KO_DIGITS if korean else _ZH_DIGITS
    units = ("", "십", "백", "천") if korean else ("", "十", "百", "千")
    number = int(value)
    if number == 0:
        return digits[0]
    if len(value) > 1 and value.startswith("0"):
        return "".join(digits[int(character)] for character in value)
    if number > 9_999:
        return "".join(digits[int(character)] for character in value)

    output: list[str] = []
    pending_zero = False
    for position in range(3, -1, -1):
        divisor = 10**position
        digit = number // divisor % 10
        if digit:
            if pending_zero and output:
                output.append(digits[0])
            if not (digit == 1 and position == 1 and not output):
                output.append(digits[digit])
            output.append(units[position])
            pending_zero = False
        elif output and number % divisor:
            pending_zero = True
    return "".join(output)


def _english_integer(value: str) -> str:
    number = int(value)
    if len(value) > 1 and value.startswith("0"):
        return " ".join(_EN_SMALL[int(character)] for character in value)
    if number < 20:
        return _EN_SMALL[number]
    if number < 100:
        tens, remainder = divmod(number, 10)
        return _EN_TENS[tens] + (f" {_EN_SMALL[remainder]}" if remainder else "")
    if number < 1_000:
        hundreds, remainder = divmod(number, 100)
        suffix = f" {_english_integer(str(remainder))}" if remainder else ""
        return f"{_EN_SMALL[hundreds]} hundred{suffix}"
    if number < 1_000_000:
        thousands, remainder = divmod(number, 1_000)
        suffix = f" {_english_integer(str(remainder))}" if remainder else ""
        return f"{_english_integer(str(thousands))} thousand{suffix}"
    return " ".join(_EN_SMALL[int(character)] for character in value)


def _spoken_number(value: str, *, language: str) -> str:
    integer, separator, fraction = value.partition(".")
    if language == "en":
        spoken = _english_integer(integer)
        if separator:
            spoken += " point " + " ".join(_EN_SMALL[int(character)] for character in fraction)
        return spoken
    korean = language == "ko"
    spoken = _cjk_integer(integer, korean=korean)
    if separator:
        point = " 점 " if korean else "点"
        digits = _KO_DIGITS if korean else _ZH_DIGITS
        spoken += point + "".join(digits[int(character)] for character in fraction)
    return spoken


def verbalise_tts_numbers(text: str) -> str:
    """Replace Arabic numeric tokens with deterministic language-appropriate speech."""

    if _HIRAGANA_KATAKANA.search(text):
        language = "ja"
    elif _HANGUL.search(text):
        language = "ko"
    elif _CJK.search(text) or not _LATIN.search(text):
        language = "zh"
    else:
        language = "en"
    return _NUMBER.sub(lambda match: _spoken_number(match.group(), language=language), text)


def normalise_tts_text(text: str) -> str:
    """Canonicalise cache input and verbalise numbers before synthesis."""

    canonical = " ".join(unicodedata.normalize("NFKC", text).split())
    return verbalise_tts_numbers(canonical)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EspeakNgProvider:
    """Generate Mandarin WAV files locally without network or API credentials."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        binary: str = "espeak-ng",
        voice: str = "cmn",
        speed: float = 1.0,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not 0.5 <= speed <= 2.0:
            raise ValueError("TTS speed must be between 0.5 and 2.0")
        if timeout_seconds <= 0:
            raise ValueError("TTS timeout must be positive")
        resolved_binary = shutil.which(binary)
        if resolved_binary is None:
            raise FileNotFoundError(f"TTS binary is unavailable: {binary}")
        self._binary = resolved_binary
        self._cache_dir = cache_dir.expanduser().resolve()
        self._voice = voice
        self._speed = speed
        self._timeout = timeout_seconds
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    async def synthesize(self, text: str) -> TTSAudio:
        normalised = normalise_tts_text(text)
        if not normalised:
            raise ValueError("TTS text must not be blank")
        if len(normalised) > 1_000:
            raise ValueError("TTS text exceeds the 1000-character safety limit")
        return await asyncio.to_thread(self._synthesize_sync, normalised)

    async def pre_generate(self, prompts: tuple[str, ...]) -> tuple[TTSAudio, ...]:
        """Pre-generate fixed process prompts through the same cache path."""

        return tuple(await asyncio.gather(*(self.synthesize(prompt) for prompt in prompts)))

    async def synthesize_ephemeral(self, text: str, *, output_dir: Path) -> TTSAudio:
        normalised = normalise_tts_text(text)
        if not normalised:
            raise ValueError("TTS text must not be blank")
        if len(normalised) > 1_000:
            raise ValueError("TTS text exceeds the 1000-character safety limit")
        return await asyncio.to_thread(
            self._synthesize_ephemeral_sync,
            normalised,
            output_dir,
        )

    def _cache_key(self, text: str) -> str:
        material = f"espeak-ng-v1\0{self._voice}\0{self._speed:.3f}\0{text}".encode()
        return hashlib.sha256(material).hexdigest()

    def _lock_for(self, cache_key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(cache_key, threading.Lock())

    @staticmethod
    def _inspect(path: Path, cache_key: str, *, from_cache: bool) -> TTSAudio:
        with wave.open(str(path), "rb") as source:
            frames = source.getnframes()
            sample_rate = source.getframerate()
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
        if frames <= 0 or sample_rate <= 0 or channels not in {1, 2} or sample_width != 2:
            raise ValueError("TTS provider produced an invalid PCM WAV")
        return TTSAudio(
            cache_key=cache_key,
            path=path,
            sha256=_file_sha256(path),
            duration_ms=frames * 1_000 // sample_rate,
            sample_rate=sample_rate,
            channels=channels,
            from_cache=from_cache,
        )

    def _synthesize_sync(self, text: str) -> TTSAudio:
        cache_key = self._cache_key(text)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        final_path = (self._cache_dir / f"{cache_key}.wav").resolve()
        if final_path.parent != self._cache_dir:
            raise ValueError("TTS cache path escapes configured root")
        with self._lock_for(cache_key):
            if final_path.is_file():
                return self._inspect(final_path, cache_key, from_cache=True)
            temporary_path = final_path.with_suffix(".wav.part")
            temporary_path.unlink(missing_ok=True)
            words_per_minute = max(80, min(450, round(175 * self._speed)))
            try:
                subprocess.run(
                    [
                        self._binary,
                        "--stdin",
                        "-v",
                        self._voice,
                        "-s",
                        str(words_per_minute),
                        "-w",
                        str(temporary_path),
                    ],
                    input=text,
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    check=True,
                    timeout=self._timeout,
                )
                descriptor = os.open(temporary_path, os.O_RDWR | getattr(os, "O_BINARY", 0))
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                self._inspect(temporary_path, cache_key, from_cache=False)
                os.replace(temporary_path, final_path)
                return self._inspect(final_path, cache_key, from_cache=False)
            finally:
                temporary_path.unlink(missing_ok=True)

    def _synthesize_ephemeral_sync(self, text: str, output_dir: Path) -> TTSAudio:
        selected_dir = output_dir.expanduser().resolve()
        selected_dir.mkdir(parents=True, exist_ok=True)
        ephemeral_key = secrets.token_hex(16)
        final_path = (selected_dir / f"{ephemeral_key}.wav").resolve()
        if final_path.parent != selected_dir:
            raise ValueError("ephemeral TTS path escapes configured runtime root")
        temporary_path = final_path.with_suffix(".wav.part")
        words_per_minute = max(80, min(450, round(175 * self._speed)))
        try:
            subprocess.run(
                [
                    self._binary,
                    "--stdin",
                    "-v",
                    self._voice,
                    "-s",
                    str(words_per_minute),
                    "-w",
                    str(temporary_path),
                ],
                input=text,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=True,
                timeout=self._timeout,
            )
            descriptor = os.open(temporary_path, os.O_RDWR | getattr(os, "O_BINARY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            audio = self._inspect(temporary_path, ephemeral_key, from_cache=False)
            os.replace(temporary_path, final_path)
            return TTSAudio(
                cache_key=audio.cache_key,
                path=final_path,
                sha256=audio.sha256,
                duration_ms=audio.duration_ms,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                from_cache=False,
            )
        finally:
            temporary_path.unlink(missing_ok=True)


class CosyVoiceHttpProvider:
    """Cache WAV returned by the private, pinned CosyVoice SFT service."""

    _MAX_RESPONSE_BYTES = 16 * 1024 * 1024

    def __init__(
        self,
        *,
        cache_dir: Path,
        base_url: str,
        model: str,
        model_revision: str,
        model_digest: str,
        voice: str,
        speed: float = 1.0,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not 0.5 <= speed <= 2.0:
            raise ValueError("TTS speed must be between 0.5 and 2.0")
        if timeout_seconds <= 0:
            raise ValueError("TTS timeout must be positive")
        self._cache_dir = cache_dir.expanduser().resolve()
        self._endpoint = urljoin(base_url.rstrip("/") + "/", "v1/synthesize")
        self._model = model
        self._model_revision = model_revision
        self._model_digest = model_digest
        self._voice = voice
        self._speed = speed
        self._timeout = timeout_seconds
        self._locks: dict[str, asyncio.Lock] = {}

    def _cache_key(self, text: str) -> str:
        material = (
            f"cosyvoice-sft-v1\0{self._model}\0{self._model_revision}\0"
            f"{self._model_digest}\0{self._voice}\0{self._speed:.3f}\0{text}"
        ).encode()
        return hashlib.sha256(material).hexdigest()

    async def _request_payload(self, normalised: str) -> bytes:
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as client:
            async with client.post(
                self._endpoint,
                json={
                    "text": normalised,
                    "voice": self._voice,
                    "speed": self._speed,
                },
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"CosyVoice service returned HTTP {response.status}")
                if response.headers.get("X-Lingxu-TTS-Model-Digest") != self._model_digest:
                    raise RuntimeError("CosyVoice service model digest mismatch")
                if response.content_type not in {"audio/wav", "audio/x-wav"}:
                    raise RuntimeError("CosyVoice service returned an unexpected content type")
                declared_length = response.content_length
                if declared_length is not None and declared_length > self._MAX_RESPONSE_BYTES:
                    raise RuntimeError("CosyVoice response exceeds size limit")
                payload = await response.read()
        if not payload or len(payload) > self._MAX_RESPONSE_BYTES:
            raise RuntimeError("CosyVoice returned an invalid response size")
        return payload

    async def pre_generate(self, prompts: tuple[str, ...]) -> tuple[TTSAudio, ...]:
        return tuple(await asyncio.gather(*(self.synthesize(prompt) for prompt in prompts)))

    async def synthesize(self, text: str) -> TTSAudio:
        normalised = normalise_tts_text(text)
        if not normalised:
            raise ValueError("TTS text must not be blank")
        if len(normalised) > 1_000:
            raise ValueError("TTS text exceeds the 1000-character safety limit")
        cache_key = self._cache_key(normalised)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        final_path = (self._cache_dir / f"{cache_key}.wav").resolve()
        if final_path.parent != self._cache_dir:
            raise ValueError("TTS cache path escapes configured root")
        async with self._locks.setdefault(cache_key, asyncio.Lock()):
            if final_path.is_file():
                return EspeakNgProvider._inspect(final_path, cache_key, from_cache=True)
            payload = await self._request_payload(normalised)
            temporary_path = final_path.with_suffix(".wav.part")
            try:
                temporary_path.write_bytes(payload)
                descriptor = os.open(temporary_path, os.O_RDWR | getattr(os, "O_BINARY", 0))
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                audio = EspeakNgProvider._inspect(
                    temporary_path,
                    cache_key,
                    from_cache=False,
                )
                os.replace(temporary_path, final_path)
                return TTSAudio(
                    cache_key=audio.cache_key,
                    path=final_path,
                    sha256=audio.sha256,
                    duration_ms=audio.duration_ms,
                    sample_rate=audio.sample_rate,
                    channels=audio.channels,
                    from_cache=False,
                )
            finally:
                temporary_path.unlink(missing_ok=True)

    async def synthesize_ephemeral(self, text: str, *, output_dir: Path) -> TTSAudio:
        normalised = normalise_tts_text(text)
        if not normalised:
            raise ValueError("TTS text must not be blank")
        if len(normalised) > 1_000:
            raise ValueError("TTS text exceeds the 1000-character safety limit")
        payload = await self._request_payload(normalised)
        return await asyncio.to_thread(self._write_ephemeral_payload, payload, output_dir)

    @staticmethod
    def _write_ephemeral_payload(payload: bytes, output_dir: Path) -> TTSAudio:
        selected_dir = output_dir.expanduser().resolve()
        selected_dir.mkdir(parents=True, exist_ok=True)
        ephemeral_key = secrets.token_hex(16)
        final_path = (selected_dir / f"{ephemeral_key}.wav").resolve()
        if final_path.parent != selected_dir:
            raise ValueError("ephemeral TTS path escapes configured runtime root")
        temporary_path = final_path.with_suffix(".wav.part")
        try:
            temporary_path.write_bytes(payload)
            descriptor = os.open(temporary_path, os.O_RDWR | getattr(os, "O_BINARY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            audio = EspeakNgProvider._inspect(
                temporary_path,
                ephemeral_key,
                from_cache=False,
            )
            os.replace(temporary_path, final_path)
            return TTSAudio(
                cache_key=audio.cache_key,
                path=final_path,
                sha256=audio.sha256,
                duration_ms=audio.duration_ms,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                from_cache=False,
            )
        finally:
            temporary_path.unlink(missing_ok=True)
