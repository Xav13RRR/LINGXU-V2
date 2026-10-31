"""Private, content-safe HTTP boundary around the pinned CosyVoice SFT model."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import threading
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import soundfile
import torch
from cosyvoice.cli.cosyvoice import AutoModel
from model_integrity import (
    MODEL_DIGEST,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    TEXT_NORMALIZER_DIGEST,
    verify_model,
    verify_text_normalizer,
)

MAX_REQUEST_BYTES = 16 * 1024
MAX_TEXT_CHARACTERS = 1_000
DEFAULT_VOICE = "中文女"
MODEL: Any = None
MODEL_LOCK = threading.Lock()
ACTIVE_DIGEST = ""
ACTIVE_TEXT_NORMALIZER_DIGEST = ""
ALLOWED_VOICE = DEFAULT_VOICE


class _DiscardWriter(io.TextIOBase):
    def write(self, text: str) -> int:
        return len(text)


@contextlib.contextmanager
def _suppress_model_output() -> Iterator[None]:
    sink = _DiscardWriter()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


class Handler(BaseHTTPRequestHandler):
    server_version = "LingxuTTS/1"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(
        self,
        status: HTTPStatus,
        payload: dict[str, object],
        *,
        include_digest: bool = False,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if include_digest:
            self.send_header("X-Lingxu-TTS-Model-Digest", ACTIVE_DIGEST)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/healthz" and MODEL is not None and ACTIVE_DIGEST:
            self._send_json(HTTPStatus.OK, {"ok": True}, include_digest=True)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/v1/synthesize":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid_size"})
            return
        try:
            request = json.loads(self.rfile.read(content_length))
            text = request["text"]
            voice = request["voice"]
            speed = request["speed"]
            if (
                not isinstance(text, str)
                or not text.strip()
                or len(text) > MAX_TEXT_CHARACTERS
                or not isinstance(voice, str)
                or voice != ALLOWED_VOICE
                or isinstance(speed, bool)
                or not isinstance(speed, (int, float))
                or not 0.5 <= float(speed) <= 2.0
            ):
                raise ValueError
            wav = _synthesize(text.strip(), voice, float(speed))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request"})
            return
        except Exception as exc:
            logging.error("tts_inference_failed error_type=%s", type(exc).__name__)
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "inference_failed"})
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Lingxu-TTS-Model-Digest", ACTIVE_DIGEST)
        self.end_headers()
        self.wfile.write(wav)


def _synthesize(text: str, voice: str, speed: float) -> bytes:
    model = MODEL
    if model is None:
        raise RuntimeError("model unavailable")
    with MODEL_LOCK, torch.inference_mode(), _suppress_model_output():
        chunks = [
            item["tts_speech"].detach().cpu()
            for item in model.inference_sft(text, voice, stream=False, speed=speed)
        ]
    if not chunks:
        raise RuntimeError("empty inference output")
    waveform = torch.cat(chunks, dim=1).squeeze(0).numpy()
    output = io.BytesIO()
    soundfile.write(output, waveform, model.sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def _load_model() -> None:
    global ACTIVE_DIGEST, ACTIVE_TEXT_NORMALIZER_DIGEST, ALLOWED_VOICE, MODEL
    repository = os.environ.get("TTS_MODEL", MODEL_REPOSITORY)
    revision = os.environ.get("TTS_MODEL_REVISION", MODEL_REVISION)
    expected_digest = os.environ.get("TTS_MODEL_DIGEST", MODEL_DIGEST)
    ALLOWED_VOICE = os.environ.get("TTS_VOICE", DEFAULT_VOICE)
    if repository != MODEL_REPOSITORY or revision != MODEL_REVISION:
        raise RuntimeError("unapproved model repository or revision")
    if ALLOWED_VOICE != DEFAULT_VOICE:
        raise RuntimeError("unapproved SFT voice")
    model_dir = Path(os.environ.get("TTS_MODEL_DIR", "/models/cosyvoice-300m-sft")).resolve()
    normalizer_dir = Path(os.environ.get("TTS_TEXT_NORMALIZER_DIR", "/models/wetext-tn")).resolve()
    ACTIVE_DIGEST = verify_model(model_dir, expected_digest)
    ACTIVE_TEXT_NORMALIZER_DIGEST = verify_text_normalizer(
        normalizer_dir,
        TEXT_NORMALIZER_DIGEST,
    )
    import wetext.wetext as wetext_module

    def pinned_snapshot_download(model_id: str) -> str:
        if model_id != "pengzhendong/wetext":
            raise RuntimeError("unapproved text normalizer repository")
        return str(normalizer_dir)

    wetext_module.snapshot_download = pinned_snapshot_download
    logging.getLogger().setLevel(logging.ERROR)
    with _suppress_model_output():
        model = AutoModel(model_dir=str(model_dir))
        speakers: Any = model.list_available_spks()
    if getattr(model.frontend, "text_frontend", "") != "wetext":
        raise RuntimeError("approved text normalizer frontend is unavailable")
    numeric_probe = model.frontend.text_normalize("共10项。", split=False)
    if numeric_probe != "共十项。":
        raise RuntimeError("approved text normalizer failed its numeric self-test")
    if ALLOWED_VOICE not in speakers:
        raise RuntimeError("configured SFT voice is unavailable")
    MODEL = model


def main() -> None:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    _load_model()
    server = ThreadingHTTPServer(("0.0.0.0", 50000), Handler)
    server.daemon_threads = True
    print(json.dumps({"ready": True, "model_verified": True, "voice": "female"}))
    server.serve_forever()


if __name__ == "__main__":
    main()
