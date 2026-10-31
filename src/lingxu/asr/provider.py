"""Replaceable realtime ASR boundary and the default SenseVoice implementation."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from lingxu.asr.models import ASRResult, UtteranceAudio

_LANGUAGE_TAG = re.compile(r"<\|(?P<language>zh|en|yue|ja|ko|auto)\|>", re.IGNORECASE)
_SENSEVOICE_MODEL_SHA256 = "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea"
_FSMN_VAD_MODEL_SHA256 = "b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5"
_MODEL_REPOSITORY_ALIASES = {
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _modelscope_snapshot_path(
    cache_dir: Path,
    *,
    model_id: str,
    revision: str,
) -> Path:
    """Resolve one ModelScope snapshot without permitting path traversal."""

    repository_id = _MODEL_REPOSITORY_ALIASES.get(model_id, model_id)
    parts = repository_id.split("/")
    if (
        len(parts) != 2
        or any(part in {"", ".", ".."} for part in parts)
        or any(separator in revision for separator in ("/", "\\"))
        or revision in {"", ".", ".."}
    ):
        raise RuntimeError("approved ASR model identity is invalid")
    snapshots_root = (cache_dir / "modelscope" / "models").resolve()
    snapshot = (snapshots_root / f"{parts[0]}--{parts[1]}" / "snapshots" / revision).resolve()
    if not snapshot.is_relative_to(snapshots_root):
        raise RuntimeError("approved ASR model snapshot escapes the model cache")
    return snapshot


def _verified_model_weight(
    snapshot: Path,
    *,
    expected_sha256: str,
    label: str,
) -> Path:
    weight = snapshot / "model.pt"
    resolved_snapshot = snapshot.resolve()
    resolved_weight = weight.resolve()
    if not resolved_weight.is_relative_to(resolved_snapshot) or not resolved_weight.is_file():
        raise RuntimeError(
            f"approved {label} model snapshot is not installed; run the Lingxu ASR model downloader"
        )
    if _sha256(resolved_weight) != expected_sha256:
        raise RuntimeError(f"{label} model weight hash does not match the approved snapshot")
    return resolved_weight


class RealtimeASRProvider(Protocol):
    """Provider contract; meeting runtime never imports one model implementation."""

    async def transcribe(
        self,
        utterance: UtteranceAudio,
        *,
        language: str,
        hotwords: tuple[str, ...],
    ) -> ASRResult:
        """Return raw and normalised text for one sentence."""


@runtime_checkable
class PreparableRealtimeASRProvider(Protocol):
    """Optional capability for loading an ASR provider before live audio starts."""

    async def prepare(self) -> None:
        """Load and verify the already selected local speech model."""


class SenseVoiceProvider:
    """Lazy, CPU-only FunASR provider with offline-by-default runtime loading."""

    def __init__(
        self,
        *,
        model: str = "iic/SenseVoiceSmall",
        model_revision: str = "master",
        vad_model: str = "fsmn-vad",
        vad_model_revision: str = "v2.0.4",
        model_cache_dir: Path = Path("models"),
        cpu_threads: int = 4,
        expected_model_sha256: str = _SENSEVOICE_MODEL_SHA256,
        expected_vad_model_sha256: str = _FSMN_VAD_MODEL_SHA256,
        allow_download: bool = False,
    ) -> None:
        self._model_name = model
        self._model_revision = model_revision
        self._vad_model = vad_model
        self._vad_model_revision = vad_model_revision
        self._cache_dir = model_cache_dir.expanduser().resolve()
        self._cpu_threads = max(1, cpu_threads)
        self._expected_model_sha256 = expected_model_sha256.lower()
        self._expected_vad_model_sha256 = expected_vad_model_sha256.lower()
        self._allow_download = allow_download
        self._model: Any | None = None
        self._model_lock = threading.RLock()

    async def prepare(self) -> None:
        """Load and verify SenseVoice before the first live assistant turn."""

        await asyncio.to_thread(self._load_model)

    async def transcribe(
        self,
        utterance: UtteranceAudio,
        *,
        language: str,
        hotwords: tuple[str, ...],
    ) -> ASRResult:
        return await asyncio.to_thread(
            self._transcribe_sync,
            utterance,
            language,
            hotwords,
        )

    def _load_model(self) -> Any:
        with self._model_lock:
            if self._model is not None:
                return self._model
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MODELSCOPE_CACHE", str(self._cache_dir / "modelscope"))

            if self._allow_download:
                model_argument = self._model_name
                vad_argument = self._vad_model
                revisions = {
                    "model_revision": self._model_revision,
                    "vad_model_revision": self._vad_model_revision,
                }
            else:
                model_snapshot = _modelscope_snapshot_path(
                    self._cache_dir,
                    model_id=self._model_name,
                    revision=self._model_revision,
                )
                vad_snapshot = _modelscope_snapshot_path(
                    self._cache_dir,
                    model_id=self._vad_model,
                    revision=self._vad_model_revision,
                )
                _verified_model_weight(
                    model_snapshot,
                    expected_sha256=self._expected_model_sha256,
                    label="SenseVoice",
                )
                _verified_model_weight(
                    vad_snapshot,
                    expected_sha256=self._expected_vad_model_sha256,
                    label="FSMN-VAD",
                )
                model_argument = str(model_snapshot)
                vad_argument = str(vad_snapshot)
                revisions = {}

            from funasr import AutoModel

            loaded_model = AutoModel(
                model=model_argument,
                vad_model=vad_argument,
                vad_kwargs={"max_single_segment_time": 30_000},
                device="cpu",
                ncpu=self._cpu_threads,
                disable_update=True,
                disable_pbar=True,
                trust_remote_code=False,
                check_latest=False,
                **revisions,
            )
            model_path = Path(loaded_model.model_path) / "model.pt"
            vad_path = Path(str(loaded_model.vad_kwargs["model_path"])) / "model.pt"
            if _sha256(model_path) != self._expected_model_sha256:
                raise RuntimeError(
                    "SenseVoice model weight hash does not match the approved snapshot"
                )
            if _sha256(vad_path) != self._expected_vad_model_sha256:
                raise RuntimeError(
                    "FSMN-VAD model weight hash does not match the approved snapshot"
                )
            self._model = loaded_model
            return self._model

    @staticmethod
    def _mono_16k(pcm_s16le_48k_stereo: bytes) -> np.ndarray[Any, np.dtype[np.float32]]:
        stereo = np.frombuffer(pcm_s16le_48k_stereo, dtype="<i2")
        if stereo.size == 0 or stereo.size % 2:
            raise ValueError("utterance PCM must contain complete stereo int16 frames")
        stereo = stereo.reshape((-1, 2)).astype(np.int32)
        mono_48k = (stereo[:, 0] + stereo[:, 1]) / 2.0
        return np.ascontiguousarray(mono_48k[::3] / 32_768.0, dtype=np.float32)

    def _transcribe_sync(
        self,
        utterance: UtteranceAudio,
        language: str,
        hotwords: tuple[str, ...],
    ) -> ASRResult:
        started_ns = time.perf_counter_ns()
        samples = self._mono_16k(utterance.pcm_s16le_48k_stereo)
        hotword = " ".join(hotwords)
        with self._model_lock:
            model = self._load_model()
            values = model.generate(
                input=samples,
                fs=16_000,
                language=language,
                use_itn=True,
                hotword=hotword,
                batch_size_s=60,
                disable_pbar=True,
            )
        first = values[0] if values else {}
        raw_text = str(first.get("text", "")).strip()
        language_match = _LANGUAGE_TAG.search(raw_text)
        detected_language = language_match.group("language").lower() if language_match else None
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        normalised = rich_transcription_postprocess(raw_text).strip()
        return ASRResult(
            raw_text=raw_text,
            normalised_text=normalised,
            language=detected_language,
            processing_ms=max(0, (time.perf_counter_ns() - started_ns) // 1_000_000),
        )
