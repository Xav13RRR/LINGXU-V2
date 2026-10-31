# 聆序 LINGXU Local Deployment Guide

This guide covers deployment with Windows, Docker Desktop, an
ordinary Discord voice channel, local Ollama `qwen3.5:9b`, local SenseVoice ASR, local CosyVoice
Mandarin female TTS, an independent post-meeting Worker, and the official DeepSeek
final-minutes API. Persistent database, audio, and model volumes remain on the operator's machine;
the Worker sends the consent-filtered complete transcript, fixed template, authoritative agendas,
and aggregate vote facts to DeepSeek. Ollama and TTS do not publish host ports by default.

[中文版](DEPLOYMENT.zh-CN.md) · [README](../README.md) · [English usage guide](USAGE.en.md)

## 1. Deployment components

| Service/volume | Responsibility | Network and data boundary |
| --- | --- | --- |
| `lingxu-bot` | Discord Gateway, DAVE, meeting Runtime, realtime ASR, voice Q&A, host console | Connects to Discord; mounts `/data`, `/models`, and tmpfs `/runtime` |
| `lingxu-worker` | Post-meeting verification, transcription completion, DeepSeek final minutes, publication, recovery | Shares Lingxu's independent data volume; connects to Discord and the official DeepSeek API; never calls another project's Worker |
| `lingxu-ollama` | Local `qwen3.5:9b` for intents, agenda import, and short answers | Internal `lingxu-llm` network only; no host port |
| `lingxu-tts` | CosyVoice `中文女` speech synthesis | Internal `lingxu-tts` network only; no host port |
| `lingxu-data` | SQLite/WAL, encrypted Sessions, TTS cache | Production data; back up before upgrades |
| `lingxu-models` | SenseVoice, FSMN-VAD, and sherpa-onnx KWS | Independent from application images |
| `lingxu-ollama` volume | Ollama model files | Accessible to the downloader and serving container only |
| `lingxu-cosyvoice` volume | CosyVoice and Wetext files | Mounted read-only by the serving container |

The public product name is “聆序 LINGXU”. Internal `lingxu` package, database, container, and volume
identifiers remain lowercase for upgrade and recovery compatibility. Do not rename them.

## 2. Prerequisites

### Windows GPU profile

- 64-bit Windows 10 or 11;
- Docker Desktop with the WSL 2 backend and Docker Compose v2;
- an NVIDIA driver that supports container compute;
- Git;
- a dedicated Discord Application, Bot token, and at least one target Guild where the installer
  may add the Bot;
- sufficient disk space. The pinned Qwen Ollama manifest alone is about 6.59 GB; ASR, TTS, KWS,
  and Docker images require additional space.

Check the host:

```powershell
docker version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
git --version
```

Without an NVIDIA GPU, use the CPU compatibility profile in section 10. Functionality is the same,
but Qwen and TTS cold-start and inference latency are substantially higher.

### Optional development and acceptance tools

Host quality checks require Python 3.12 and `uv 0.9.2`. A Docker-only production operator does not
need the complete Python dependency set on the host.

## 3. Create the Discord Application

1. Open the [Discord Developer Portal](https://discord.com/developers/applications) and create an
   Application dedicated to 聆序 LINGXU.
2. Create the Bot on the **Bot** page and generate a token. Store it only in local `.env`; never put
   it in chat, screenshots, issues, or Git.
3. Privileged Message Content Intent is not required. Lingxu uses Guilds, Voice States, and normal
   Guild Messages. Discord supplies attachment metadata for a real direct Bot mention.
4. Enable the `bot` and `applications.commands` installation scopes.
5. Grant View Channels, Send Messages and Attach Files in the publication text channel, and View
   Channel, Connect and Speak in the voice channel. Mute Members is optional for enforced ordered
   or presentation agendas. Do not grant Administrator. Check channel overrides as well as roles.
6. Use the Developer Portal installation link to add the same Bot to every target Guild. The
   installer needs the corresponding server-management authority. Copy the Application ID to
   `DISCORD_APPLICATION_ID`.
7. Leave `DISCORD_TEST_GUILD_ID` blank for a production multi-Guild deployment. Set one Guild ID
   only for development, where immediate test-Guild command sync is desired.

See [DISCORD_SETUP.md](DISCORD_SETUP.md) for the permission boundary. Never reuse another Bot or
project's token.

## 4. Clone and create the local configuration

Use the independent public `Xav13RRR/LINGXU-V2` repository, not the private engineering archive.
Supply your own credentials and never overwrite an existing `.env`:

```powershell
git clone https://github.com/Xav13RRR/LINGXU-V2.git LINGXU
Set-Location LINGXU
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Follow [API and credential configuration](CONFIGURATION.en.md) for the required/optional field table,
your own API-key registration, Windows DPAPI versus manual Compose, safe updates and troubleshooting.
Edit ignored `.env` and provide at least:

```dotenv
DISCORD_TOKEN=<Discord Bot token>
DISCORD_APPLICATION_ID=<Application ID>
DISCORD_TEST_GUILD_ID=
FILE_ENCRYPTION_KEY=<URL-safe base64 32-byte key>
```

With `DISCORD_TEST_GUILD_ID` blank, Lingxu synchronises global commands restricted to Guild
installation and Guild contexts. A newly installed Guild therefore needs no `.env` edit or
separate Bot process. Discord clients may need a short refresh interval after global sync. If the
local host console is enabled, set one explicit `HOST_CONSOLE_GUILD_ID`; the console remains bound
to that one Guild.

The workstation still exposes one heavy-workload slot. Formal Meetings (through post-meeting
Worker publication), resident mode, and Markdown agenda-model processing claim it
atomically. A competing Guild receives the fixed capacity-full response without learning the
owner. Lightweight deterministic setup such as Meeting creation and status may be completed before
the heavy workflow claims the slot.

Do not change pinned model names, revisions, or digests without a new model review and complete
regression. For the current workstation GPU profile, `.env` may contain:

```dotenv
COMPOSE_FILE=docker-compose.yml;docker-compose.gpu.yml
```

If the local Compose installation does not accept the semicolon form, omit it. The Windows launcher
loads both Compose files explicitly.

### Generate the file-encryption key

Docker-only operators can use the built-in PowerShell generator in
[credential configuration](CONFIGURATION.en.md#5-generate-the-encryption-key-and-update-credentials).
Alternatively, with Python 3.12 and uv installed, run from the repository root:

```powershell
uv sync --frozen --no-dev
uv run python -c "from lingxu.storage.encryption import FileCipher; print(FileCipher.generate_key())"
```

Copy the output only into `.env` as `FILE_ENCRYPTION_KEY`. It protects Session files with
AES-256-GCM. Existing encrypted audio cannot be recovered if the key is lost. Keep a secure backup
separate from the data backup, but never commit the key.

## 5. First build and pinned model downloads

Downloaders use an explicit `model-download` profile. Ordinary startup does not repeatedly download
models. Run these steps only for first installation or an explicitly approved model replacement.

### 5.1 Build application and TTS images

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build `
  lingxu-bot lingxu-worker lingxu-tts
```

### 5.2 Download the Ollama Qwen model

The serving container is on an internal network without egress. A one-shot downloader owns the
volume during pull:

```powershell
docker compose stop lingxu-ollama
docker compose --profile model-download run --rm lingxu-ollama-downloader
docker compose up -d --no-deps lingxu-ollama
docker compose exec -T lingxu-ollama ollama show qwen3.5:9b
```

The production manifest digest is
`6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7`.
The Bot checks local inventory before every request and fails closed on mismatch.

### 5.3 Download CosyVoice Mandarin female TTS

```powershell
docker compose --profile model-download run --rm lingxu-tts-downloader
docker compose up -d lingxu-tts
```

The downloader validates the pinned `FunAudioLLM/CosyVoice-300M-SFT` revision, the aggregate model
digest, and the offline Wetext normalisation files. The service rejects incomplete or changed
weights.

### 5.4 Download the wake-word model

```powershell
docker compose --profile model-download run --rm lingxu-wakeword-downloader
```

This installs the sherpa-onnx Chinese/English KWS model and validates both the official archive and
selected runtime-file digests. The default phrase is “呼叫指挥中心” with its restricted acoustic
variants. A Manage Guild member may use `/wakeup set` to select 4-16 consecutive Han characters for
that Guild. The command preflights pronunciation against the pinned `tokens.txt` before persistence;
custom phrases use one exact acoustic path and active Meeting/resident assistants switch without a
restart. `/wakeup reset` restores the default.

### 5.5 Download and validate SenseVoice and FSMN-VAD

This one-shot command writes ASR assets to the shared `lingxu-models` volume and verifies both
`model.pt` SHA-256 values before returning:

```powershell
docker compose --profile model-download run --rm lingxu-asr-downloader
```

Rerun the command after an interrupted download. Do not bypass a digest mismatch; remove the bad
volume content and recheck the approved source in [MODELS.md](MODELS.md). Bot and Worker runtime
load only these verified local snapshots and never query ModelScope during Meeting startup or
post-processing.

## 6. Start the Windows GPU deployment

After `.env` and all first-install models are ready, enter the DeepSeek API key once through the
masked prompt. The script stores only a Windows current-user DPAPI ciphertext outside the repository;
it does not write the key to `.env`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/set_deepseek_api_key.ps1
```

Then double-click the repository-root launcher:

```text
启动聆序.cmd
```

The launcher starts Docker Desktop when necessary, decrypts the DPAPI credential only into its own
process, validates required settings without printing them, builds the shared Bot/Worker/migration
application image once through `lingxu-migrate`, and rebuilds TTS only when its Dockerfile/service
source fingerprint changed. It repeats the aggregate-only idle guard, starts Bot, Worker, Ollama,
and TTS with `--no-build`, waits for four health checks, and confirms Discord
`bot_ready`. It clears the process credential after validation and does not print credentials,
transcripts, or meeting content. From a Git checkout it also derives the full HEAD revision before
Compose validation: clean trees use the exact SHA and modified trees use the SHA with a `-dirty`
suffix. An explicit process `LINGXU_BUILD_REVISION` is preserved; source bundles without `.git` use
the Compose fallback.

PowerShell equivalent:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_lingxu.ps1
```

Validation only, with no service changes:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_lingxu.ps1 -DryRun
```

Pass `-RebuildAll` only when TTS must be rebuilt even though its source fingerprint matches.

The root CMD forwards optional switches too, so this exercises the actual wrapper from a terminal:

```text
启动聆序.cmd -DryRun
```

Manual start is for operators who have separately supplied `SUMMARY_LLM_API_KEY` through the
Compose process environment or an access-restricted local `.env`; **Compose does not decrypt DPAPI**.
Windows DPAPI users should use the launcher above. See [manual credentials](CONFIGURATION.en.md#4-manual-compose--non-windows-hosts).
On an existing deployment, first follow the idle and backup procedure in section 9.

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## 7. Validate the deployment

```powershell
docker compose ps
docker compose exec -T lingxu-bot python -m lingxu.cli.doctor --strict-discord --runtime-only
docker compose exec -T lingxu-bot python scripts/smoke_meeting_assistant.py
docker compose exec -T lingxu-bot python scripts/smoke_agenda_import.py
```

Healthy production means:

- the one-shot `lingxu-migrate` container exits with code 0; Bot and Worker only read-check the
  current Alembic revision and do not race through DDL;
- `lingxu-bot`, `lingxu-worker`, `lingxu-ollama`, and `lingxu-tts` are all `healthy`;
- Bot logs contain `application_commands_synced`; verify `/meeting`, `/assistant`, and `/wakeup`
  in Discord with the default resident profile;
- Bot logs contain `bot_ready`;
- Doctor passes Discord, encryption, ASR, KWS, Ollama, TTS, and inbound DAVE checks;
- `http://127.0.0.1:8765/healthz` is reachable locally only when the host console is enabled;
- no Ollama 11434 or TTS service port is published to the host.

Never paste full `docker compose config` output into a public location because rendered environment
values can contain secrets. Use only `docker compose config --quiet`.

Compose assigns credentials by process capability: Bot receives the local-intent, meeting-
assistant, agenda-import, and Host-console credentials, while Worker receives the final-minutes
credential. Only Bot and Worker share the Discord token and file-encryption key they actually use;
`lingxu-migrate` receives none of these secrets. When restoring an older database without rebuilding
the image, run `docker compose run --rm lingxu-migrate` before starting Bot or Worker.

## 8. Optional local host console

Set in `.env`:

```dotenv
HOST_CONSOLE_ENABLED=true
HOST_CONSOLE_GUILD_ID=<the one permitted Guild ID>
```

When `HOST_CONSOLE_TOKEN` is blank, Lingxu derives a domain-separated token from the file-encryption
root. Reveal it only in the local terminal after startup:

```powershell
docker compose exec -T lingxu-bot python -m lingxu.console.token
```

Open `http://127.0.0.1:8765`. Do not screenshot, record, or forward the token. The console binds to
localhost and one Guild, with HttpOnly sessions, CSRF, and same-origin enforcement.

## 9. Stop, restart, upgrade, and back up

Normal stop:

```powershell
docker compose stop
```

Bot and Worker must exit with code 0. Do not force-remove containers while recording finalisation or
Worker publication is active.

Before an upgrade:

1. Confirm no active Meeting and no `PENDING/RUNNING` Worker job.
2. Record the shared application and TTS image IDs and add rollback tags.
3. Stop Bot and Worker cleanly.
4. Back up the complete `lingxu-data` volume.
5. Pull the new source, run the quality gates, and rebuild.
6. Require the one-shot `lingxu-migrate` service to exit successfully, then start Worker recovery
   and Bot.
7. Verify revision, Doctor, command sync, and all four health states.

Never commit or upload `backups/`. See [ROLLBACK.md](ROLLBACK.md) and
[FAILURE_RECOVERY.md](FAILURE_RECOVERY.md) for exact recovery rules.

## 10. CPU compatibility deployment

The base Compose profile requests no NVIDIA device. These manual commands require separately
provided Compose credentials as explained in section 6; DPAPI users use `-Cpu` below:

```powershell
docker compose -f docker-compose.yml config --quiet
docker compose -f docker-compose.yml build lingxu-migrate lingxu-tts
docker compose -f docker-compose.yml up -d --no-build
```

Or run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_lingxu.ps1 -Cpu
```

The Qwen, ASR, and TTS functional boundaries remain unchanged, but live short-answer and dynamic TTS
latency can rise substantially. CPU mode is intended for compatibility and recovery, not the
GPU profile's lower-latency use case.

## 11. Pinned model sources

| Role | Model | Official/upstream source |
| --- | --- | --- |
| Local LLM | `qwen3.5:9b` / Qwen3.5-9B | [Ollama](https://ollama.com/library/qwen3.5:9b) · [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Final Meeting minutes | Official DeepSeek API / `deepseek-flash` | [DeepSeek API](https://api-docs.deepseek.com/) |
| ASR | `iic/SenseVoiceSmall` | [ModelScope](https://modelscope.cn/models/iic/SenseVoiceSmall) |
| VAD | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | [ModelScope](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch) |
| Wake word | sherpa-onnx KWS Zipformer | [Model index](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html) · [Releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models) |
| TTS | `FunAudioLLM/CosyVoice-300M-SFT` | [Hugging Face](https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT) · [GitHub](https://github.com/FunAudioLLM/CosyVoice) |
| Text normalisation | `pengzhendong/wetext` | [ModelScope](https://modelscope.cn/models/pengzhendong/wetext) |

See [MODELS.md](MODELS.md) for production revisions, file digests, manifest identities, licences,
and replacement requirements. A mutable `master` or `latest` label is never sufficient production
identity by itself.

## 12. Pre-release quality gate

Developers run before a build or upgrade:

```powershell
uv sync --all-extras --frozen
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
uv run python scripts/scan_secrets.py
uv run python -m scripts.check_public_files
uv build
docker compose config --quiet
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --quiet
```

The application image starts from a pinned Python 3.12.11 multi-architecture manifest digest and
uses builder/runtime stages; Git and uv exist only in the builder. Runtime images carry OCI
`version`, `revision`, and `source` labels, and CI supplies the commit SHA as the revision. Compose
accepts the same metadata through `LINGXU_BUILD_VERSION`, `LINGXU_BUILD_REVISION`, and
`LINGXU_BUILD_SOURCE`; an omitted revision is explicitly `unknown` and is not a traceable release.

See [test coverage](TEST_REPORT.md) for CI results and outstanding manual voice acceptance checks.
Validate Discord connectivity, voice processing and final minutes delivery on each new deployment.
