import hashlib
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lingxu.asr.provider import SenseVoiceProvider


def _write_snapshot(
    cache_root: Path,
    *,
    model_id: str,
    revision: str,
    payload: bytes,
) -> Path:
    owner, name = model_id.split("/", maxsplit=1)
    snapshot = cache_root / "modelscope" / "models" / f"{owner}--{name}" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.pt").write_bytes(payload)
    return snapshot


def test_pcm_conversion_is_stereo_to_mono_and_48k_to_16k() -> None:
    pcm = b"".join(
        struct.pack("<hh", left, right)
        for left, right in [(3_000, 1_000), (6_000, 0), (9_000, -3_000)] * 2
    )

    converted = SenseVoiceProvider._mono_16k(pcm)

    assert converted.dtype == np.float32
    assert converted.tolist() == pytest.approx([2_000 / 32_768, 2_000 / 32_768])


def test_pcm_conversion_rejects_incomplete_stereo_frame() -> None:
    with pytest.raises(ValueError, match="complete stereo"):
        SenseVoiceProvider._mono_16k(b"\x01\x00")


def test_runtime_load_uses_verified_local_snapshots_without_remote_model_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_payload = b"approved-sensevoice"
    vad_payload = b"approved-fsmn-vad"
    model_snapshot = _write_snapshot(
        tmp_path,
        model_id="iic/SenseVoiceSmall",
        revision="master",
        payload=model_payload,
    )
    vad_snapshot = _write_snapshot(
        tmp_path,
        model_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        revision="v2.0.4",
        payload=vad_payload,
    )
    observed: dict[str, Any] = {}

    def fake_auto_model(**kwargs: Any) -> Any:
        observed.update(kwargs)
        return SimpleNamespace(
            model_path=kwargs["model"],
            vad_kwargs={"model_path": kwargs["vad_model"]},
        )

    monkeypatch.setattr("funasr.AutoModel", fake_auto_model)
    provider = SenseVoiceProvider(
        model_cache_dir=tmp_path,
        expected_model_sha256=hashlib.sha256(model_payload).hexdigest(),
        expected_vad_model_sha256=hashlib.sha256(vad_payload).hexdigest(),
    )

    provider._load_model()

    assert observed["model"] == str(model_snapshot.resolve())
    assert observed["vad_model"] == str(vad_snapshot.resolve())


def test_runtime_load_fails_closed_before_funasr_when_snapshot_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def unexpected_auto_model(**_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("FunASR must not run without verified local snapshots")

    monkeypatch.setattr("funasr.AutoModel", unexpected_auto_model)
    provider = SenseVoiceProvider(model_cache_dir=tmp_path)

    with pytest.raises(RuntimeError, match="approved SenseVoice model snapshot is not installed"):
        provider._load_model()

    assert not called
