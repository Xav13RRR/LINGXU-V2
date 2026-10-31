"""Pinned local acoustic keyword spotting for the Meeting assistant."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pypinyin import Style, lazy_pinyin

from lingxu.asr.models import UtteranceAudio
from lingxu.asr.provider import SenseVoiceProvider
from lingxu.domain.wake_phrase import DEFAULT_WAKE_PHRASE, normalise_wake_phrase

_MODEL_FILES = (
    "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
    "decoder-epoch-13-avg-2-chunk-8-left-64.onnx",
    "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
    "tokens.txt",
)

# The long canonical phrase and each one-syllable deletion are separate acoustic paths.
# Four-syllable fragments such as "指挥中心" are deliberately absent.
KEYWORDS_TEXT = """h ū j iào zh ǐ h uī zh ōng x īn @呼叫指挥中心
j iào zh ǐ h uī zh ōng x īn @叫指挥中心
h ū zh ǐ h uī zh ōng x īn @呼指挥中心
h ū j iào h uī zh ōng x īn @呼叫挥中心
h ū j iào zh ǐ zh ōng x īn @呼叫指中心
h ū j iào zh ǐ h uī x īn @呼叫指挥心
h ū j iào zh ǐ h uī zh ōng @呼叫指挥中
"""


def keyword_manifest_for_phrase(wake_phrase: str) -> str:
    """Build one exact custom Mandarin KWS path without weakening the default variants."""

    selected = normalise_wake_phrase(wake_phrase)
    if selected == DEFAULT_WAKE_PHRASE:
        return KEYWORDS_TEXT
    initials = lazy_pinyin(selected, style=Style.INITIALS, strict=False)
    finals = lazy_pinyin(selected, style=Style.FINALS_TONE, strict=False)
    if len(initials) != len(selected) or len(finals) != len(selected):
        raise ValueError("唤醒词无法转换为本地关键词模型支持的拼音。")
    tokens: list[str] = []
    for initial, final in zip(initials, finals, strict=True):
        if initial:
            tokens.append(initial)
        if not final:
            raise ValueError("唤醒词包含本地关键词模型不支持的字符。")
        tokens.append(final)
    return f"{' '.join(tokens)} @{selected}\n"


def selected_model_digest(model_dir: Path) -> str:
    """Hash the exact runtime files in a stable, filename-bound order."""

    digest = hashlib.sha256()
    for name in _MODEL_FILES:
        path = model_dir / name
        digest.update(f"{name}\n".encode())
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class KeywordSpotter(Protocol):
    async def prepare(self) -> None:
        """Verify and load the local acoustic model."""

    async def detect(self, utterance: UtteranceAudio) -> bool:
        """Return whether the utterance contains an approved wake variant."""

    async def update_wake_phrase(self, wake_phrase: str) -> None:
        """Prepare and atomically select a new validated wake phrase."""


class SherpaOnnxKeywordSpotter:
    """Utterance-isolated sherpa-onnx KWS with pinned model verification."""

    def __init__(
        self,
        *,
        model_dir: Path,
        expected_model_digest: str,
        wake_phrase: str = DEFAULT_WAKE_PHRASE,
        keywords_root: Path | None = None,
        keywords_score: float = 1.5,
        keywords_threshold: float = 0.18,
        num_threads: int = 2,
    ) -> None:
        self._model_dir = model_dir.expanduser().resolve()
        self._keywords_root = (
            keywords_root.expanduser().resolve() if keywords_root is not None else self._model_dir
        )
        self._expected_digest = expected_model_digest.lower()
        self._wake_phrase = normalise_wake_phrase(wake_phrase)
        self._keywords_score = keywords_score
        self._keywords_threshold = keywords_threshold
        self._num_threads = max(1, num_threads)
        self._spotter: Any | None = None
        self._lock = threading.RLock()

    async def prepare(self) -> None:
        await asyncio.to_thread(self._load)

    async def detect(self, utterance: UtteranceAudio) -> bool:
        return await asyncio.to_thread(self._detect_sync, utterance)

    async def update_wake_phrase(self, wake_phrase: str) -> None:
        selected = normalise_wake_phrase(wake_phrase)
        await asyncio.to_thread(self._update_sync, selected)

    async def preflight(self, wake_phrase: str) -> None:
        """Verify a phrase against the pinned model and token vocabulary without activating it."""

        selected = normalise_wake_phrase(wake_phrase)
        await asyncio.to_thread(self._preflight_sync, selected)

    @staticmethod
    def _token_vocabulary(tokens_file: Path) -> frozenset[str]:
        vocabulary: set[str] = set()
        for raw_line in tokens_file.read_text(encoding="utf-8").splitlines():
            token, separator, identifier = raw_line.rpartition(" ")
            if separator and token and identifier.isdigit():
                vocabulary.add(token)
        if not vocabulary:
            raise RuntimeError("Meeting wake-word token vocabulary is invalid")
        return frozenset(vocabulary)

    def _keywords_file(self, wake_phrase: str) -> Path:
        manifest = keyword_manifest_for_phrase(wake_phrase)
        if wake_phrase == DEFAULT_WAKE_PHRASE:
            keywords_file = self._model_dir / "lingxu-keywords.txt"
            if not keywords_file.is_file() or keywords_file.read_text(encoding="utf-8") != manifest:
                raise RuntimeError("Meeting wake-word keyword manifest does not match")
            return keywords_file

        manifest_tokens = manifest.partition(" @")[0].split()
        vocabulary = self._token_vocabulary(self._model_dir / _MODEL_FILES[3])
        if any(token not in vocabulary for token in manifest_tokens):
            raise ValueError("唤醒词包含当前本地关键词模型不支持的读音。")
        self._keywords_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
        keywords_file = self._keywords_root / f"{digest}.keywords.txt"
        if keywords_file.is_file():
            if keywords_file.read_text(encoding="utf-8") != manifest:
                raise RuntimeError("Meeting wake-word runtime manifest hash collision")
            return keywords_file
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{digest}.",
                suffix=".part",
                dir=self._keywords_root,
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(manifest)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, keywords_file)
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
        return keywords_file

    def _build_spotter(self, wake_phrase: str) -> Any:
        keywords_file = self._verified_keywords_file(wake_phrase)

        import sherpa_onnx

        return sherpa_onnx.KeywordSpotter(
            tokens=str(self._model_dir / _MODEL_FILES[3]),
            encoder=str(self._model_dir / _MODEL_FILES[0]),
            decoder=str(self._model_dir / _MODEL_FILES[1]),
            joiner=str(self._model_dir / _MODEL_FILES[2]),
            keywords_file=str(keywords_file),
            num_threads=self._num_threads,
            max_active_paths=4,
            keywords_score=self._keywords_score,
            keywords_threshold=self._keywords_threshold,
            num_trailing_blanks=1,
            provider="cpu",
        )

    def _verified_keywords_file(self, wake_phrase: str) -> Path:
        if not self._model_dir.is_dir():
            raise RuntimeError("approved Meeting wake-word model is not installed")
        observed = selected_model_digest(self._model_dir)
        if observed != self._expected_digest:
            raise RuntimeError("Meeting wake-word model digest does not match")
        return self._keywords_file(wake_phrase)

    def _preflight_sync(self, wake_phrase: str) -> None:
        with self._lock:
            self._verified_keywords_file(wake_phrase)

    def _load(self) -> Any:
        with self._lock:
            if self._spotter is not None:
                return self._spotter
            self._spotter = self._build_spotter(self._wake_phrase)
            return self._spotter

    def _update_sync(self, wake_phrase: str) -> None:
        with self._lock:
            if wake_phrase == self._wake_phrase:
                self._load()
                return
            replacement = self._build_spotter(wake_phrase)
            self._wake_phrase = wake_phrase
            self._spotter = replacement

    def _detect_sync(self, utterance: UtteranceAudio) -> bool:
        with self._lock:
            spotter = self._load()
            stream = spotter.create_stream()
            samples = SenseVoiceProvider._mono_16k(utterance.pcm_s16le_48k_stereo)
            stream.accept_waveform(16_000, samples)
            stream.accept_waveform(16_000, np.zeros(10_560, dtype=np.float32))
            stream.input_finished()
            while spotter.is_ready(stream):
                spotter.decode_stream(stream)
                if spotter.get_result(stream):
                    spotter.reset_stream(stream)
                    return True
            return False
