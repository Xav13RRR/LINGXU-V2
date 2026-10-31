# Locked Dependencies

Target runtime: CPython 3.12.11 on Windows development hosts, Linux CI, and Debian Bookworm
containers. GitHub's Windows 2022 runner is pinned to CPython 3.12.10 because the official
`actions/python-versions` manifest has no Windows x64 artifact for 3.12.11.

| Package | Pin | Purpose | Licence | Verified platform | Known limitation / upgrade |
| --- | --- | --- | --- | --- | --- |
| Python | 3.12.11 runtime; 3.12.10 Windows CI | Runtime required by the plan | PSF | Windows / CI / Debian | Application image uses multi-architecture manifest `sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7`; the GitHub Windows 2022 compatibility pin must remain an exact available 3.12.x release; 3.13+ is intentionally rejected |
| uv | 0.9.2 | Environment and lock management | MIT OR Apache-2.0 | Windows / CI / Docker | Installed separately on hosts; regenerate and review `uv.lock` when upgrading |
| hatchling | 1.28.0 | PEP 517 build backend | MIT | Cross-platform | Build-time only; update exact pin and rebuild wheel |
| discord.py | 2.7.1 | Official Bot, commands, voice send and DAVE session | MIT | CPython 3.12 | Voice receive is supplied by the separate pinned fork |
| davey | 0.1.6 | Discord DAVE protocol implementation used by discord.py | MIT | CPython 3.12 | Native wheels required; run the real DAVE Gate after every upgrade |
| discord-ext-voice-recv fork | `ee160c0f36516927b6214bc9d6babe524016770f` | Per-user voice receive with inbound DAVE parsing and diagnostics | MIT | CPython 3.12 | Experimental Git dependency; upgrade only to a reviewed full SHA and rerun the real Gate |
| imageio-ffmpeg | 0.6.0 | Locate a packaged FFmpeg executable for probe playback | BSD-2-Clause | Windows / Debian | A system FFmpeg remains used in the container; verify voice playback after upgrade |
| soundfile | 0.13.1 | Per-user lossless FLAC writing | BSD-3-Clause | Windows / Debian | Writers are owned by one recorder thread; do not share handles |
| pydantic | 2.13.4 | Configuration validation | MIT | Cross-platform | Update with pydantic-settings and rerun config tests |
| pydantic-settings | 2.14.2 | Environment and `.env` loading | MIT | Cross-platform | Empty Compose optionals are explicitly normalised |
| SQLAlchemy | 2.0.51 | Async ORM, repositories, transactions, and atomic SQLite job/event operations | MIT | Windows / Debian | Keep the 2.x typed API; run migrations and persistence concurrency tests after upgrades |
| aiosqlite | 0.22.1 | Async SQLite driver used by SQLAlchemy | MIT | Windows / Debian | SQLite serialises writes; keep transactions short and rerun WAL/concurrency tests after upgrades |
| Alembic | 1.18.5 | Versioned independent Lingxu database migrations | MIT | Windows / Debian | Upgrade only with a reviewed forward migration and tested downgrade path |
| cryptography | 49.0.0 | AES-256-GCM envelopes for persisted Session files | Apache-2.0 OR BSD-3-Clause | Windows / Debian | Native wheels are required; rotate keys through an explicit data migration, never by replacing the value in place |
| FunASR | 1.3.27 | SenseVoice/FSMN-VAD model loading and inference | MIT | Windows / Debian CPU | Imports TorchAudio; rerun the real multilingual benchmark after every upgrade |
| ModelScope | 1.38.1 | Approved SenseVoice/FSMN-VAD snapshot download and cache | Apache-2.0 | Windows / Debian | SenseVoice has no release tag; runtime additionally verifies the approved weight SHA-256 |
| NumPy | 2.4.6 | PCM stereo-to-mono conversion and 48 kHz to 16 kHz decimation | BSD-3-Clause and bundled compatible notices | Windows / Debian | Provider input must remain normalised `float32`, not `int16` |
| PyTorch CPU | 2.11.0 | SenseVoice/FSMN-VAD tensor runtime | BSD-3-Clause | Windows / Debian CPU | Resolved only from the explicit PyTorch CPU index; CUDA wheels are excluded |
| TorchAudio CPU | 2.11.0 | FunASR Kaldi-compatible feature extraction | BSD | Windows / Debian CPU | Must match the PyTorch minor version |
| espeak-ng | 1.51 (`1.51+dfsg-10+deb12u2`) | Offline Mandarin TTS executable and voice data | GPL-3.0-or-later with bundled notices | Debian container | Separate stdin-driven process; no source text in argv; rerun Mandarin WAV benchmark after image upgrades |
| CosyVoice source | `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` | Local high-quality Mandarin SFT inference | Apache-2.0 | Isolated CUDA 12.1 container | Full Git SHA; runtime verifies the separate model snapshot before load |
| Wetext text-normalisation resources | `pengzhendong/wetext@master`, aggregate `a2e3a67aab0e2f9457ee61c1d3b40203d24e30d08cfdeceefdb7042f8a24a6d5` | Offline Chinese/English number verbalisation before CosyVoice inference | Apache-2.0 | Isolated CUDA 12.1 container | The remote label is mutable, so all four approved FST filenames, sizes, hashes, and their aggregate are enforced before startup; no runtime download is allowed |
| PyTorch CUDA TTS base | `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime@sha256:fc47f8018254e6df30f48c48f2db1c758d44de21a8c553de1a1c451a65baa70a` | CosyVoice tensor/CUDA runtime | BSD-3-Clause plus CUDA/container notices | Docker Desktop / NVIDIA GPU | Immutable platform digest; GPU profile only |
| CosyVoice Python bundle | Exact pins in `services/cosyvoice/requirements.txt` | Frontend, ONNX and acoustic/vocoder inference | Mixed compatible upstream licences | CosyVoice container / CPython 3.10 | Isolated from the Bot's CPython 3.12 lock; rebuild and rerun model/import/audio tests for every pin change |
| aiohttp | 3.14.3 | Worker Discord/DeepSeek HTTP plus native Ollama intent, assistant, and agenda-import HTTP | Apache-2.0 | Windows / Debian | Direct pin; never log Authorization headers, attachments, utterances, transcripts, summaries, or response content |
| PyYAML | 6.0.3 | Safe Scenario YAML loading | MIT | Windows / Debian | Use `safe_load` only; strict Pydantic validation follows parsing |
| Ollama container | `0.21.2@sha256:2038a264392af4d21f7f14e1568e8dd9aa0f2bd0f31d7d9a59a8cca1e9829663` | Private-network local `qwen3.5:9b` structured intent, short answer, and agenda import | MIT | Docker Desktop / NVIDIA GPU | API port is not published; update image and rerun digest/schema/injection/agenda-import/assistant/deployment Gates together |
| structlog | 26.1.0 | Structured content-free logging | MIT OR Apache-2.0 | Cross-platform | Redaction processors must remain before rendering |
| Ruff | 0.16.0 | Lint and format checks | MIT | Development / CI | Exact development pin |
| mypy | 2.3.0 | Strict type checking | MIT | Development / CI | Exact development pin |
| pytest | 9.1.1 | Automated tests | MIT | Development / CI | Exact development pin |
| pytest-asyncio | 1.4.0 | Strict async tests | Apache-2.0 | Development / CI | `asyncio_mode=strict` |
| actions/checkout | `v7.0.1` / `3d3c42e5aac5ba805825da76410c181273ba90b1` | CI source checkout | MIT | GitHub-hosted CI | Node 24 action pinned to a full commit SHA; review runner requirements before upgrade |
| actions/setup-python | `v7.0.0` / `5fda3b95a4ea91299a34e894583c3862153e4b97` | Exact CI Python provisioning | MIT | GitHub-hosted CI | Node 24 action pinned to a full commit SHA; keep Python patch aligned with `.python-version` and Docker |

The voice-receive fork is locked to a complete commit SHA, never a branch or pull-request ref.
`uv.lock` is the transitive dependency source of truth after `uv lock` succeeds.
Approved ASR/LLM model identities, weight hashes or manifest digests, licences, and replay fixtures
are recorded in `docs/MODELS.md`.
