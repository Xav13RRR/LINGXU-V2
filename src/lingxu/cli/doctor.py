"""Content-free environment and repository diagnostics."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin

from imageio_ffmpeg import get_ffmpeg_exe

from lingxu.config import Settings
from lingxu.domain.errors import EncryptionConfigurationError
from lingxu.storage.encryption import FileCipher
from lingxu.voice_assistant.wakeword import KEYWORDS_TEXT, selected_model_digest

CheckStatus = Literal["PASSED", "WARNING", "FAILED"]


@dataclass(frozen=True, slots=True)
class Check:
    """One content-free doctor result."""

    name: str
    status: CheckStatus
    detail: str


def _dependency_check(package: str, expected: str) -> Check:
    try:
        actual = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return Check(f"dependency:{package}", "FAILED", "not installed")
    status: CheckStatus = "PASSED" if actual == expected else "FAILED"
    return Check(f"dependency:{package}", status, f"expected={expected}, actual={actual}")


def _receive_fork_supports_dave() -> bool:
    try:
        voice_recv = importlib.import_module("discord.ext.voice_recv")
    except ImportError:
        return False
    return hasattr(voice_recv, "parse_dave_payload")


def _writable_directory_check(path: Path, *, name: str) -> Check:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix="doctor-", delete=True):
            pass
    except OSError as exc:
        return Check(name, "FAILED", type(exc).__name__)
    return Check(name, "PASSED", str(path.resolve()))


def _independence_scan(project_root: Path) -> Check:
    legacy_name = "ling" + "ji"
    patterns = (
        re.compile(
            rf"^\s*(?:from|import)\s+{legacy_name}\b",
            re.MULTILINE | re.IGNORECASE,
        ),
        re.compile(rf"\b{legacy_name.upper()}_[A-Z0-9_]*\b"),
        re.compile(rf"{legacy_name}\.db", re.IGNORECASE),
        re.compile(rf"XA_Devs_2026[\\/]{legacy_name}", re.IGNORECASE),
    )
    candidates = [
        *project_root.glob("src/**/*.py"),
        *project_root.glob("scripts/**/*.py"),
        project_root / "pyproject.toml",
        project_root / ".env.example",
        project_root / "Dockerfile",
        project_root / "docker-compose.yml",
    ]
    violations: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in patterns):
            violations.append(path.relative_to(project_root).as_posix())
    if violations:
        return Check("runtime_independence", "FAILED", ",".join(sorted(violations)))
    return Check("runtime_independence", "PASSED", "no forbidden runtime reference")


def _asr_model_cache_check(cache_root: Path) -> Check:
    expected = (
        (
            cache_root
            / "modelscope"
            / "models"
            / "iic--SenseVoiceSmall"
            / "snapshots"
            / "master"
            / "model.pt",
            936_291_369,
        ),
        (
            cache_root
            / "modelscope"
            / "models"
            / "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch"
            / "snapshots"
            / "v2.0.4"
            / "model.pt",
            1_721_366,
        ),
    )
    if not any(path.exists() for path, _ in expected):
        return Check("realtime_asr_model_cache", "WARNING", "approved weights not preloaded")
    if not all(path.is_file() and path.stat().st_size == size for path, size in expected):
        return Check("realtime_asr_model_cache", "FAILED", "weight size mismatch or partial cache")
    return Check("realtime_asr_model_cache", "PASSED", "approved weights present")


def _tts_provider_check(settings: Settings, *, runtime_only: bool) -> Check:
    if settings.tts_provider == "espeak-ng":
        tts_binary = shutil.which(settings.tts_binary)
        return Check(
            "tts_provider",
            "PASSED" if tts_binary else "WARNING",
            tts_binary or "espeak-ng unavailable on this host",
        )
    endpoint = urljoin(settings.tts_base_url.rstrip("/") + "/", "healthz")
    try:
        with urllib.request.urlopen(endpoint, timeout=5) as response:
            valid = (
                response.status == 200
                and response.read() == b'{"ok":true}'
                and response.headers.get("X-Lingxu-TTS-Model-Digest") == settings.tts_model_digest
            )
    except (OSError, urllib.error.URLError):
        valid = False
    if valid:
        return Check("tts_provider", "PASSED", "CosyVoice model verified and ready")
    return Check(
        "tts_provider",
        "FAILED" if runtime_only else "WARNING",
        "CosyVoice service unavailable or model digest mismatch",
    )


def _meeting_assistant_model_check(settings: Settings) -> Check:
    if not settings.meeting_assistant_enabled:
        return Check("meeting_assistant_wakeword", "PASSED", "disabled")
    model_dir = settings.meeting_assistant_wakeword_model_dir.expanduser().resolve()
    try:
        keywords_match = (model_dir / "lingxu-keywords.txt").read_text(
            encoding="utf-8"
        ) == KEYWORDS_TEXT
        digest_match = (
            selected_model_digest(model_dir) == settings.meeting_assistant_wakeword_model_digest
        )
    except OSError as exc:
        return Check("meeting_assistant_wakeword", "FAILED", type(exc).__name__)
    if not keywords_match or not digest_match:
        return Check("meeting_assistant_wakeword", "FAILED", "model or keyword digest mismatch")
    return Check("meeting_assistant_wakeword", "PASSED", "approved local model present")


def _meeting_assistant_llm_check(settings: Settings, *, runtime_only: bool) -> Check:
    if not settings.meeting_assistant_enabled:
        return Check("meeting_assistant_llm", "PASSED", "disabled")
    root = settings.meeting_assistant_llm_base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=5) as response:
            payload = json.loads(response.read(64 * 1024))
        observed = next(
            entry["digest"]
            for entry in payload["models"]
            if entry.get("name") == settings.meeting_assistant_llm_model
            or entry.get("model") == settings.meeting_assistant_llm_model
        )
        valid = (
            isinstance(observed, str)
            and observed.lower() == settings.meeting_assistant_llm_model_digest
        )
    except (
        OSError,
        urllib.error.URLError,
        json.JSONDecodeError,
        KeyError,
        StopIteration,
        TypeError,
    ):
        valid = False
    if valid:
        return Check("meeting_assistant_llm", "PASSED", "approved local model verified")
    return Check(
        "meeting_assistant_llm",
        "FAILED" if runtime_only else "WARNING",
        "local model unavailable or digest mismatch",
    )


def _agenda_import_llm_check(settings: Settings, *, runtime_only: bool) -> Check:
    if not settings.agenda_import_llm_enabled:
        return Check("agenda_import_llm", "PASSED", "disabled")
    root = settings.agenda_import_llm_base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=5) as response:
            payload = json.loads(response.read(64 * 1024))
        observed = next(
            entry["digest"]
            for entry in payload["models"]
            if entry.get("name") == settings.agenda_import_llm_model
            or entry.get("model") == settings.agenda_import_llm_model
        )
        valid = (
            isinstance(observed, str)
            and observed.lower() == settings.agenda_import_llm_model_digest
        )
    except (
        OSError,
        urllib.error.URLError,
        json.JSONDecodeError,
        KeyError,
        StopIteration,
        TypeError,
    ):
        valid = False
    if valid:
        return Check("agenda_import_llm", "PASSED", "approved local model verified")
    return Check(
        "agenda_import_llm",
        "FAILED" if runtime_only else "WARNING",
        "local model unavailable or digest mismatch",
    )


def run_checks(
    settings: Settings,
    *,
    strict_discord: bool = False,
    runtime_only: bool = False,
) -> tuple[Check, ...]:
    """Run diagnostics without printing secrets or meeting content."""

    checks: list[Check] = []
    python_ok = sys.version_info[:2] == (3, 12)
    checks.append(
        Check(
            "python",
            "PASSED" if python_ok else "FAILED",
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )
    checks.extend(
        (
            _dependency_check("discord.py", "2.7.1"),
            _dependency_check("davey", "0.1.6"),
            _dependency_check("discord-ext-voice-recv", "0.5.3a185"),
            _dependency_check("soundfile", "0.13.1"),
            _dependency_check("SQLAlchemy", "2.0.51"),
            _dependency_check("alembic", "1.18.5"),
            _dependency_check("aiosqlite", "0.22.1"),
            _dependency_check("cryptography", "49.0.0"),
            _dependency_check("funasr", "1.3.27"),
            _dependency_check("modelscope", "1.38.1"),
            _dependency_check("numpy", "2.4.6"),
            _dependency_check("pypinyin", "0.55.0"),
            _dependency_check("sherpa-onnx", "1.13.4"),
            _dependency_check("sherpa-onnx-core", "1.13.4"),
            _dependency_check("torch", "2.11.0+cpu"),
            _dependency_check("torchaudio", "2.11.0+cpu"),
        )
    )
    checks.append(_asr_model_cache_check(settings.model_cache_dir))
    checks.append(_meeting_assistant_model_check(settings))
    checks.append(_meeting_assistant_llm_check(settings, runtime_only=runtime_only))
    checks.append(_agenda_import_llm_check(settings, runtime_only=runtime_only))
    checks.append(_tts_provider_check(settings, runtime_only=runtime_only))
    inbound_dave_available = _receive_fork_supports_dave()
    checks.append(
        Check(
            "inbound_dave_api",
            "PASSED" if inbound_dave_available else "FAILED",
            "parse_dave_payload available"
            if inbound_dave_available
            else "parse_dave_payload unavailable",
        )
    )
    try:
        ffmpeg = Path(get_ffmpeg_exe())
        ffmpeg_ok = ffmpeg.is_file()
    except Exception as exc:
        checks.append(Check("ffmpeg", "FAILED", type(exc).__name__))
    else:
        checks.append(Check("ffmpeg", "PASSED" if ffmpeg_ok else "FAILED", str(ffmpeg)))
    checks.append(
        _writable_directory_check(
            settings.runtime_tmp,
            name="recording_tmp_writable",
        )
    )
    checks.append(
        _writable_directory_check(
            settings.tts_cache_dir,
            name="tts_cache_writable",
        )
    )

    if strict_discord:
        credentials_status: CheckStatus = "PASSED" if settings.has_discord_credentials else "FAILED"
    else:
        credentials_status = "PASSED" if settings.has_discord_credentials else "WARNING"
    checks.append(
        Check(
            "independent_discord_credentials",
            credentials_status,
            "configured" if settings.has_discord_credentials else "not configured",
        )
    )

    encryption_key = settings.file_encryption_key
    if encryption_key is None:
        checks.append(Check("file_encryption_key", "WARNING", "not configured"))
    else:
        try:
            FileCipher.from_secret(encryption_key)
        except EncryptionConfigurationError as exc:
            checks.append(Check("file_encryption_key", "FAILED", type(exc).__name__))
        else:
            checks.append(Check("file_encryption_key", "PASSED", "configured"))

    if not runtime_only:
        project_root = Path(__file__).resolve().parents[3]
        checks.append(_independence_scan(project_root))
    return tuple(checks)


def main() -> None:
    """Print JSON diagnostics and fail only on required-check failures."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-discord", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    args = parser.parse_args()

    checks = run_checks(
        Settings(),
        strict_discord=args.strict_discord,
        runtime_only=args.runtime_only,
    )
    payload = {
        "ok": not any(check.status == "FAILED" for check in checks),
        "checks": [asdict(check) for check in checks],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()
