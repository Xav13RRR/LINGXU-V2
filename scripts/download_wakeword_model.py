"""Install the one approved sherpa-onnx wake-word snapshot into the model volume."""

from __future__ import annotations

import hashlib
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from lingxu.voice_assistant.wakeword import KEYWORDS_TEXT, selected_model_digest

MODEL_NAME = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
ARCHIVE_URL = (
    f"https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/{MODEL_NAME}.tar.bz2"
)
ARCHIVE_SHA256 = "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6"
MODEL_DIGEST = "9604d8412a76c099ef8eecedb95fa04d523cb16df2f6a6be724bc956a92870ef"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_installed(target: Path) -> bool:
    keyword_file = target / "lingxu-keywords.txt"
    return (
        target.is_dir()
        and keyword_file.is_file()
        and keyword_file.read_text(encoding="utf-8") == KEYWORDS_TEXT
        and selected_model_digest(target) == MODEL_DIGEST
    )


def main() -> None:
    model_root = Path(os.environ.get("MODEL_CACHE_DIR", "/models")).resolve()
    target = model_root / MODEL_NAME
    if target.exists():
        if _verify_installed(target):
            print("approved Meeting wake-word model is already installed")
            return
        raise RuntimeError("existing Meeting wake-word model does not match; refusing overwrite")

    model_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".lingxu-wakeword-",
        dir=model_root,
    ) as temp_value:
        temp_dir = Path(temp_value)
        archive = temp_dir / f"{MODEL_NAME}.tar.bz2"
        request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "Lingxu/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as sink:
            while chunk := response.read(1024 * 1024):
                sink.write(chunk)
        if _sha256(archive) != ARCHIVE_SHA256:
            raise RuntimeError("Meeting wake-word archive digest does not match")
        with tarfile.open(archive, mode="r:bz2") as bundle:
            bundle.extractall(temp_dir, filter="data")
        extracted = temp_dir / MODEL_NAME
        (extracted / "lingxu-keywords.txt").write_text(KEYWORDS_TEXT, encoding="utf-8")
        if selected_model_digest(extracted) != MODEL_DIGEST:
            raise RuntimeError("Meeting wake-word runtime model digest does not match")
        extracted.replace(target)
    if not _verify_installed(target):
        raise RuntimeError("Meeting wake-word installation verification failed")
    print("approved Meeting wake-word model installed and verified")


if __name__ == "__main__":
    main()
