"""Pinned CosyVoice model inventory and content-safe integrity verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

MODEL_REPOSITORY = "FunAudioLLM/CosyVoice-300M-SFT"
MODEL_REVISION = "fbb71de2afe387ed854eebd80b9f3d078c6b9869"
MODEL_DIGEST = "3a80af814cfd832dadbf1fe677adf08e30f64b8bdcd52b997184515fb33d2198"
TEXT_NORMALIZER_REPOSITORY = "pengzhendong/wetext"
TEXT_NORMALIZER_REVISION = "master"
TEXT_NORMALIZER_DIGEST = "a2e3a67aab0e2f9457ee61c1d3b40203d24e30d08cfdeceefdb7042f8a24a6d5"


@dataclass(frozen=True, slots=True)
class ModelFile:
    path: str
    size: int
    sha256: str


MODEL_FILES = (
    ModelFile(
        "campplus.onnx",
        28_303_423,
        "a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73",
    ),
    ModelFile(
        "cosyvoice.yaml",
        6_421,
        "5475b9b2f788f3ddd3798898041197e076d20cb2ea783ef9638e6eed36ba16db",
    ),
    ModelFile(
        "flow.pt",
        419_900_943,
        "21eae78c105b5e1c6c337b04f667843377651b4bcfb2d43247ed3ad7fd0a3470",
    ),
    ModelFile(
        "hift.pt",
        81_896_716,
        "91e679b6ca1eff71187ffb4f3ab0444935594cdcc20a9bd12afad111ef8d6012",
    ),
    ModelFile(
        "llm.pt",
        1_242_994_835,
        "d198ce56636e1eb1c9d0cb0d6e3529de8fdfd3fd45075c346296b0d6dcfc54ea",
    ),
    ModelFile(
        "speech_tokenizer_v1.onnx",
        522_624_269,
        "23b5a723ed9143aebfd9ffda14ac4c21231f31c35ef837b6a13bb9e5488abb1e",
    ),
    ModelFile(
        "spk2info.pt",
        7_772,
        "e3b1d62ca87cdcb25a9003fa0c8f2cba5c94f55b0d5f80f0b63ef8c22d919cfc",
    ),
)

TEXT_NORMALIZER_FILES = (
    ModelFile(
        "en/tn/tagger.fst",
        5_389_722,
        "ce8b28103ccf2d936d057a5cd5621ec29ccfedb5f737e48ac8096e7d6af7b2ad",
    ),
    ModelFile(
        "en/tn/verbalizer.fst",
        2_194_398,
        "16a6a2ce28b60975e501e965396edbf94a591216a644abd27c8607a2a9aa24cd",
    ),
    ModelFile(
        "zh/tn/tagger.fst",
        1_075_582,
        "4b8b6504c7effcaa067e2fd14ecac3ae2edf9130d229e11abd0ff9b85cbe0390",
    ),
    ModelFile(
        "zh/tn/verbalizer.fst",
        87_702,
        "9fcadad76cafddb2e96b92d892d18249f0ba5fbe60e4d4fc51f466cb6bc7ade9",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_inventory(
    root: Path,
    inventory: tuple[ModelFile, ...],
    expected_digest: str,
) -> str:
    aggregate = hashlib.sha256()
    resolved_root = root.resolve()
    for item in sorted(inventory, key=lambda candidate: candidate.path):
        path = (resolved_root / item.path).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError("model path escaped configured root")
        if not path.is_file() or path.stat().st_size != item.size:
            raise ValueError(f"model file size mismatch: {item.path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != item.sha256:
            raise ValueError(f"model file digest mismatch: {item.path}")
        aggregate.update(f"{item.path}\0{item.size}\0{actual_sha256}\n".encode())
    actual_digest = aggregate.hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("aggregate model digest mismatch")
    return actual_digest


def verify_model(model_dir: Path, expected_digest: str = MODEL_DIGEST) -> str:
    """Verify every approved model file and its deterministic aggregate digest."""

    return _verify_inventory(model_dir, MODEL_FILES, expected_digest)


def verify_text_normalizer(
    normalizer_dir: Path,
    expected_digest: str = TEXT_NORMALIZER_DIGEST,
) -> str:
    """Verify the pinned offline text-normalisation FST inventory."""

    return _verify_inventory(normalizer_dir, TEXT_NORMALIZER_FILES, expected_digest)
