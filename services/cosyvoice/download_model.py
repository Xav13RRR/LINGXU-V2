"""Download one immutable CosyVoice snapshot and verify it before use."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download as hf_snapshot_download
from model_integrity import (
    MODEL_DIGEST,
    MODEL_FILES,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    TEXT_NORMALIZER_DIGEST,
    TEXT_NORMALIZER_FILES,
    TEXT_NORMALIZER_REPOSITORY,
    TEXT_NORMALIZER_REVISION,
    verify_model,
    verify_text_normalizer,
)
from modelscope import snapshot_download as modelscope_snapshot_download


def _download_text_normalizer(normalizer_dir: Path) -> str:
    normalizer_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lingxu-wetext-") as temporary:
        snapshot = Path(
            modelscope_snapshot_download(
                TEXT_NORMALIZER_REPOSITORY,
                revision=TEXT_NORMALIZER_REVISION,
                cache_dir=temporary,
                allow_patterns=[item.path for item in TEXT_NORMALIZER_FILES],
            )
        )
        for item in TEXT_NORMALIZER_FILES:
            source = snapshot / item.path
            destination = normalizer_dir / item.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = destination.with_suffix(destination.suffix + ".part")
            shutil.copyfile(source, temporary_path)
            os.replace(temporary_path, destination)
    return verify_text_normalizer(normalizer_dir, TEXT_NORMALIZER_DIGEST)


def main() -> None:
    repository = os.environ.get("TTS_MODEL", MODEL_REPOSITORY)
    revision = os.environ.get("TTS_MODEL_REVISION", MODEL_REVISION)
    expected_digest = os.environ.get("TTS_MODEL_DIGEST", MODEL_DIGEST)
    if repository != MODEL_REPOSITORY or revision != MODEL_REVISION:
        raise SystemExit("unapproved model repository or revision")
    model_dir = Path(os.environ.get("TTS_MODEL_DIR", "/models/cosyvoice-300m-sft")).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    hf_snapshot_download(
        repo_id=repository,
        revision=revision,
        local_dir=model_dir,
        allow_patterns=[item.path for item in MODEL_FILES],
    )
    actual_digest = verify_model(model_dir, expected_digest)
    normalizer_dir = Path(os.environ.get("TTS_TEXT_NORMALIZER_DIR", "/models/wetext-tn")).resolve()
    normalizer_digest = _download_text_normalizer(normalizer_dir)
    manifest_path = model_dir / ".lingxu-model.json"
    temporary_path = model_dir / ".lingxu-model.json.part"
    temporary_path.write_text(
        json.dumps(
            {
                "digest": actual_digest,
                "repository": repository,
                "revision": revision,
                "text_normalizer_digest": normalizer_digest,
                "text_normalizer_repository": TEXT_NORMALIZER_REPOSITORY,
                "text_normalizer_revision": TEXT_NORMALIZER_REVISION,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary_path, manifest_path)
    print(
        json.dumps(
            {
                "model_verified": True,
                "file_count": len(MODEL_FILES),
                "text_normalizer_verified": True,
                "text_normalizer_file_count": len(TEXT_NORMALIZER_FILES),
            }
        )
    )


if __name__ == "__main__":
    main()
