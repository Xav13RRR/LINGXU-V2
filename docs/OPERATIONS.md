# Lingxu Operations

## Start

```text
uv sync --all-extras --frozen
uv run python -m lingxu.cli.doctor --strict-discord
uv run python -m lingxu.bot_main
```

## Stop

Use `/meeting stop` to drain an active formal Meeting before process shutdown. SIGINT/SIGTERM closes
the maintained Meeting, resident-assistant, console, and database resources.

## Health

`python -m lingxu.cli.doctor --runtime-only` verifies Python, fixed package versions, inbound DAVE
API availability, FFmpeg, formal recording/TTS storage, model assets, and safe credential presence.

## Data

Formal Meeting data lives under `DATA_ROOT` and uses encrypted per-user Segments plus registered
transcript, evidence, and minutes artifacts. Atomic writes use `.part`, flush/fsync, and replacement;
the retired unencrypted standalone probe is not part of the maintained runtime.

## Database migrations

Set the independent `DATABASE_URL`, then migrate before starting a process that uses persistence:

```text
uv run python -m lingxu.storage.migrate
```

For Compose, `lingxu-migrate` is the only schema owner. Bot and Worker wait for its successful
completion and then perform a read-only current-revision check. Direct process startup therefore
requires the migration command above first. A downgrade is always explicit:

```text
uv run python -m lingxu.storage.migrate <revision> --downgrade
```

Back up the SQLite database only after stopping Bot and Worker, or use SQLite's online backup API.
Keep the database, Session root, and `FILE_ENCRYPTION_KEY` backup together; the files cannot be
recovered from the database without the matching key.

Run the retained-record audit after a backup or before relying on historical Meeting evidence:

```text
python scripts/audit_retained_meetings.py \
  --database /data/lingxu.db \
  --sessions-root /data/sessions \
  --backup-root /backups
```

Mount the database, Session root, and backup directory read-only when running this in a
container. The command emits only counts and pass/fail status: it verifies SQLite integrity and
relations, append-only event continuity, consent/participant links, encrypted-segment and registered-
artifact hashes, manifest structure, and tar/gzip safety. It never decrypts audio or emits Meeting
content, identities, paths, or identifiers. Its optional legacy probe-root support exists only for
old retained data and is not a current test instruction.

## Phase 2 Worker

```text
uv run python -m lingxu.worker_main
```

Startup verifies the current schema, removes incomplete `.part` files, requeues expired job leases,
reconciles active manifests, completes interrupted tombstone cleanup, and then creates
`RUNTIME_TMP/worker.ready`. Schema mismatch fails closed; the Worker never mutates Alembic state.

For Compose:

```text
docker compose build lingxu-migrate
docker compose up -d --no-build lingxu-worker lingxu-bot
docker compose ps
docker compose logs --no-log-prefix --tail 100 lingxu-worker
```

The one-shot `lingxu-migrate` container should be `Exited (0)`. After restoring an older compatible
database without rebuilding the image, rerun it explicitly with
`docker compose run --rm lingxu-migrate` before starting Bot or Worker.

Logs contain only job IDs, job types, counts, stages, and error classes; they must never include
keys, audio, transcripts, or meeting content.

## Phase 3 recording storage

Formal recording requires a valid `FILE_ENCRYPTION_KEY` and writable `RUNTIME_TMP`. In Compose,
`/runtime` is a private tmpfs and encrypted output is under `/data/sessions`.

The following Doctor checks must pass before capture:

```text
file_encryption_key
recording_tmp_writable
```

Never redirect `RUNTIME_TMP` into the persistent Session root. Backup only `.flac.enc` and metadata;
plaintext `.flac` or `.flac.part` under persistent storage is an incident.

## Phase 4 realtime ASR

Model cache is independent Lingxu deployment data under `MODEL_CACHE_DIR` (`/models` in Compose).
Preload and validate the approved CPU models before a meeting:

```text
uv run python scripts/benchmark_realtime_asr.py --cache-dir models --iterations 3
```

The command downloads only approved public weights/fixtures, verifies runtime weight hashes, runs
Chinese, English, and mixed replay, and exits nonzero unless all three finals are present and P95 is
at most 4,000 ms. It prints only content-safe result predicates and hashes, never transcript text.

`lingxu-doctor` checks FunASR, ModelScope, NumPy, CPU PyTorch/TorchAudio pins and confirms both
expected model files are complete by path and size. Runtime model loading performs the stronger
SHA-256 validation once before serving inference.

Operational signals:

- `frames_dropped_queue_full`: one user's frame backlog exceeded its capacity;
- `utterances_dropped_queue_full`: model throughput is below offered sentence load;
- `provider_timeouts` / `provider_failures`: isolated inference problems;
- `provider_results_discarded_after_withdrawal`: expected privacy enforcement;
- `latency_ms.p50/p95/max`: stop-to-final latency, with a Phase 4 target of P95 ≤ 4,000 ms.

## Ephemeral Meeting voice assistant

Install and verify the pinned CPU wake model before enabling the assistant:

```text
docker compose --profile model-download run --rm lingxu-wakeword-downloader
docker compose build lingxu-migrate
docker compose run --rm --no-deps lingxu-bot python scripts/smoke_meeting_assistant.py
```

The smoke uses fixed synthetic TTS only. It requires the exact wake phrase, a homophone, and one
missing-syllable variant to match, requires the short fragment `指挥中心` not to match, verifies the
Ollama manifest, and prints only booleans/counts, latency, and an answer hash. At runtime,
`lingxu-doctor --runtime-only` fails health if the wake files, keyword manifest, or approved
`qwen3.5:9b` digest does not match.

Operational contract:

- all non-Bot members whose PCM is received from the selected Meeting voice channel may invoke;
- acoustic wake detection uses bounded per-user rolling windows and may match at the beginning or
  in the middle of an utterance; only text after the confirmed occurrence becomes the question;
- acoustic detection claims the caller immediately, but ASR/answering waits for that caller's
  completed utterance;
- invocation PCM and the caller question are memory-only, regardless of consent;
- persistent recording, realtime transcript, and minutes continue using their existing consent;
- only one caller owns the assistant from accepted wake through full playback and quarantine;
- later calls are dropped, never queued, and human speech never stops Bot playback;
- a bare wake allows one same-user follow-up within four seconds;
- provider/model failures speak one fixed short failure notice and never mutate Meeting state.

At Meeting voice shutdown, `meeting_assistant_stopped` logs content-free aggregate counters only.
Use `wake_windows_detected`, `wake_turns_claimed`, `wake_occurrence_rejected`,
`wake_windows_dropped_queue_full`, `frames_dropped_queue_full`, `frames_dropped_busy`, and
`answers_completed` to identify the silent gate without logging PCM or text. Frame consumers drain
fair 64-frame batches so the receive callback remains non-blocking without paying one worker-thread
wake per Discord PCM frame.

For a retained local Meeting, the following diagnostic decrypts only in memory, replays eligible
wake slices through the pinned acoustic model, and prints counts only—never transcript text, PCM,
paths, or model output:

```text
docker compose exec -T lingxu-bot python scripts/analyze_latest_assistant.py
uv run python scripts/benchmark_frame_drain.py
```

The assistant diagnostic distinguishes transcript-confirmed hits that overlap Bot output or the
600 ms echo quarantine from eligible hits. Transcript deduplication is a later Worker-only decision
and cannot remove live assistant PCM or KWS events.

## Resident voice Q&A mode

Production Compose registers this mode by default with the already approved wake, ASR, Ollama, and
CosyVoice assets:

```text
RESIDENT_ASSISTANT_ENABLED=true
RESIDENT_ASSISTANT_RECOVER_ON_RESTART=true
RESIDENT_ASSISTANT_MAX_SESSIONS=1
```

Use `/assistant join` while a Manage-Guild operator is in the target ordinary voice channel. Use
`/assistant status` to inspect the content-free lease and `/assistant stop` to terminate it. The
creator or a Manage-Guild operator may stop; moving or disconnecting the Bot also appends STOPPED
and disconnects. Probe, formal Meeting, and resident startup share one per-Guild voice lease plus
the deployment-wide persistent capacity slot, so resolve an existing mode rather than trying to
preempt it. Another Guild receives only the fixed capacity-full response.

Migration `0011_resident_assistant` adds `resident_assistant_sessions` plus the append-only
`resident_assistant_events` stream. On normal process shutdown, runtime resources are released but
an ACTIVE lease is retained; after formal Meeting recovery, Bot startup reconnects valid resident
leases. Missing channels/permissions and a formal voice conflict terminate stale leases. Do not
edit lifecycle events; SQLite rejects update and delete.

Migration `0012_service_capacity` adds one database-unique `PRIMARY` slot and an append-only
capacity event stream. A formal Meeting holds it from voice startup through Worker publication;
resident lifetimes and agenda-model imports use the same slot. A valid Meeting or resident owner
survives Bot restart. Stale legacy-probe/import owners and owners without a recoverable source row
are released. Never edit or manually delete capacity rows while services are running.

Operational logs contain only content-free session/Guild/channel IDs, lifecycle reasons, error
classes, and aggregate assistant counters. There is no recording, transcript, Worker, artifact, or
persistent TTS-cache output to inspect. Resident-mode checks do not alter the already accepted
formal-Meeting Gates.

## Phase 5 offline TTS

Production uses the pinned `FunAudioLLM/CosyVoice-300M-SFT` snapshot with its built-in `中文女`
speaker. The private service has no host port. For the first local installation, download and
verify the approved snapshot, then start the service:

```text
docker compose --profile model-download run --rm lingxu-tts-downloader
docker compose up -d lingxu-tts
docker compose exec -T lingxu-bot python scripts/pre_generate_tts.py
```

Run a cold/warm cache benchmark:

```text
docker compose exec -T lingxu-bot python scripts/benchmark_tts.py --work-dir /runtime --iterations 5
```

`TTS_CACHE_DIR` is persistent (`/data/tts_cache` in Compose). Treat cached WAV as meeting-visible
media: do not log it, commit it, or place it under the public model volume. Model weights live in
the separate `lingxu-cosyvoice` volume and are mounted read-only while serving. `lingxu-doctor`
verifies service health, the configured model digest, and cache writability. Playback failure
automatically uses the configured Discord text fallback with mention expansion disabled.

`espeak-ng` remains an optional compatibility provider only. Set `TTS_PROVIDER=espeak-ng`,
`TTS_VOICE=cmn`, and ensure the executable is installed to use it.

## Meeting, Worker, and Scenario operations

```text
docker compose up -d
docker compose ps
docker compose logs --tail 100 lingxu-bot lingxu-worker
uv run python -m lingxu.scenario.cli scenarios/fixed_meeting_v1.yaml
uv run python -m lingxu.scenario.simulate_cli scenarios/fixed_meeting_v1.yaml
uv run python scripts/deploy_fixed_scenario.py
```

Bot and Worker both upgrade Alembic on startup. The current expected revision is
`0012_service_capacity`. `deploy_fixed_scenario.py` is source-hash idempotent.

`/meeting stop` creates one `POST_MEETING` Job. Worker downtime is safe; restart resumes from the
last checkpoint. `/meeting retry` requeues only a failed Job. For a failed DeepSeek final-minutes
attempt, that host-only command also increments the durable manual-attempt generation: the old
content-free failure journal remains immutable and the next Worker claim may make exactly one new
request. Automatic Worker retries never repeat an identical final-minutes request. `/meeting delete`
removes the latest Meeting database graph and exact Session directory.

Meeting consent is scoped to one Guild and persists until self-withdrawal:

- the first `/meeting consent` in a Guild grants realtime ASR, recording, and minutes for future
  Meetings in that same Guild;
- a user in another Guild must consent there independently;
- `/meeting withdraw` removes the current Guild grant even when no Meeting is active and clears
  retained Session processing flags;
- `CONSENT_CHECK` advances automatically when every currently present participant has all three
  grants. If an unconsented participant leaves, readiness is recalculated.

Formal voice startup and shutdown:

- The host must be in the intended ordinary voice channel when running `/meeting start`.
- The Bot must have View Channel, Connect, and Speak in that channel. Runtime does not advance if
  connect, inbound DAVE readiness, recorder startup, or realtime-ASR startup fails.
- A repeated `/meeting start` for the same active Session/channel is connection-idempotent. This is
  also the operator recovery path after a Bot restart because the voice channel itself is not
  persisted.
- `/meeting stop`, the stop control button, and `/meeting delete` can take longer than an ordinary
  Discord response because they drain capture before returning. Their Discord adapters defer the
  acknowledgement and publish the final result afterward.
- Normal stop order is receive stop, ASR drain, encrypted Segment persistence, consent-registry
  clear, Discord disconnect, then post-meeting Job creation. Do not kill the Bot merely because
  finalisation takes several seconds.

## Phase 12 local structured intent

The selected deployment uses the pinned Ollama image and `qwen3.5:9b` on the private Compose
network. Port 11434 is intentionally not published.

First deployment or approved model update (stop the serving container so only the downloader owns
the model volume):

```text
docker compose stop lingxu-ollama
docker compose --profile model-download run --rm lingxu-ollama-downloader
docker compose up -d --no-deps lingxu-ollama
docker compose exec -T lingxu-ollama ollama list
```

Copy the exact 64-character digest returned by Ollama's local `/api/tags` inventory into the
ignored `.env` as `LLM_MODEL_DIGEST`. Then validate before starting the Bot:

```text
docker compose build lingxu-migrate
docker compose run --rm --no-deps lingxu-bot python scripts/smoke_structured_intent.py
docker compose up -d --no-build lingxu-worker lingxu-bot
docker compose ps
```

Required settings:

```text
LLM_INTENTS_ENABLED=true
LLM_BASE_URL=http://lingxu-ollama:11434/v1
LLM_MODEL=qwen3.5:9b
LLM_MODEL_DIGEST=<approved local manifest digest>
LLM_ALLOW_REMOTE=false
```

The production Ollama service is attached only to the `internal: true` `lingxu-llm` network. The
profile-gated downloader is the only model service with ordinary outbound networking, and it must
not run concurrently with the serving container. Do not publish the Ollama port or change
`LLM_ALLOW_REMOTE` for this deployment. Every inference rechecks the model digest. A missing
model, digest mismatch, timeout, invalid JSON, extra property, or out-of-allowlist intent leaves
state unchanged and preserves exact rule mode. Logs contain only error classes and content-free
IDs, never the utterance or provider response.

ADR-0165 final-minutes attempts use `output_format=json_object` and bind a structured-contract
revision. A failed schema/reference check records content-free `validation_details` with a closed
reason, section and optional item index/reference count. Inspect this alongside transport
`failure_diagnostics`; neither field contains the rejected prose. Any existing attempt generation
blocks another inference even when a deployment changes its prompt/request. Basic fallback can
complete such recovery, while the old attempt remains untouched. A completed publication is never
silently regenerated. Automated tests do not constitute real-model content acceptance.

The production DeepSeek summary/final-minutes transport timeout is 1,200 seconds. This is a total
request timeout, including the non-streaming thinking-high final-minutes response body. A timeout
produces labelled basic minutes from saved facts and consented records under ADR-0164. The original
failure remains in the attempt journal; no automatic model retry occurs. A failed storage,
integrity or delivery stage still requires recovery. `/meeting retry` applies to failed Jobs, not
already published basic minutes.

`SUMMARY_LLM_MAX_OUTPUT_TOKENS=32768` applies only to bounded topic/outcome requests.
`SUMMARY_FINAL_MINUTES_MAX_OUTPUT_TOKENS=131072` is the independent explicit ceiling for the sole
full-transcript final-minutes request. Both are bounded at configuration load; changing the latter
changes the attempt request digest and never authorizes an automatic retry.

The Worker renews the claimed Job lease in the background during long stages. Lease renewal is
scoped to the exact running Job and Worker and does not modify its checkpoint. If ownership is
lost, the Worker cancels its local pipeline task and does not complete or fail the Job under stale
authority; ordinary expired-lease recovery remains responsible for requeueing it.

For a failed final-minutes attempt, inspect only the attempt journal's `failure_code` and optional
`failure_diagnostics`. The latter is schema-closed and content-free: phase, elapsed milliseconds,
HTTP status, allowlisted transport category and finish reason, content/reasoning presence states,
reasoning character count, response-body byte count, and aggregate token usage. Never copy or add
raw provider responses, reasoning, prompts, transcripts, names, IDs, or exception messages. Older
attempt journals cannot be reconstructed; new diagnostics apply only after the instrumented Worker
is deployed and a host explicitly authorizes a new generation.

If a failed post-meeting Job reports `internal_artifact_expected_digest` at
`VALIDATE_MINUTES_CONSISTENCY`, do not edit `events.json`. On the deployed ADR-0112 Worker, the
host's explicit `/meeting retry` detects whether that export is stale relative to the append-only
log, rebuilds it and its deterministic dependants, and reuses a current full-transcript minutes
artifact without sending a further DeepSeek request. A successful retry then proceeds to ordinary
Discord publication, so use it only when publication is intended.

## M1+P2 Markdown agenda import

After `/meeting create`, the host sends one direct Bot mention with exactly one `.md` attachment in
the setup channel. Lingxu replies immediately with a content-free processing acknowledgement. It
accepts no second import while one is processing and `/meeting start` remains blocked until the
import either commits or fails. A successful atomic commit posts `准备完毕`, the agenda count,
configured total time, default/inference counts, and vote count. The host can then start normally.

The production defaults are deliberately bounded:

```text
AGENDA_IMPORT_LLM_ENABLED=true
AGENDA_IMPORT_LLM_BASE_URL=http://lingxu-ollama:11434/v1
AGENDA_IMPORT_LLM_MODEL=qwen3.5:9b
AGENDA_IMPORT_LLM_MODEL_DIGEST=6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7
AGENDA_IMPORT_LLM_ALLOW_REMOTE=false
AGENDA_IMPORT_MAX_ATTACHMENT_BYTES=65536
AGENDA_IMPORT_MAX_SOURCE_CHARS=30000
AGENDA_IMPORT_MAX_ITEMS=50
AGENDA_IMPORT_DEFAULT_DURATION_MINUTES=15
AGENDA_IMPORT_DEFAULT_VOTE_SECONDS=60
```

Verify the exact selected model and P2 policy with a synthetic, content-safe smoke:

```text
docker compose run --rm --no-deps lingxu-bot python scripts/smoke_agenda_import.py
```

The smoke prints only counts and a result SHA-256. Provider failures, low confidence, conflicting
structure, bad references, unsupported languages or vote shapes, and capacity limits produce a
fixed user-facing retry message and no partial agenda. Do not inspect provider payloads in logs.
An interrupted `PROCESSING` record is recovered once after Bot readiness; if its configured provider
is unavailable it is failed safely so the Meeting does not remain permanently blocked.

## Phase 13 Host console

The console is part of the Bot process and is published only on `127.0.0.1:8765`. Start the approved
GPU deployment, open `http://127.0.0.1:8765`, and reveal the login token only in the local terminal:

```text
docker compose build lingxu-migrate
docker compose up -d --no-build lingxu-bot
docker compose exec -T lingxu-bot python -m lingxu.console.token
```

The token is exchanged for an HttpOnly browser session. Refresh restores it; writes require CSRF,
and WebSocket state is limited to the configured Guild. `/healthz` and `/metrics` are content-free.
Run `scripts/smoke_host_console.py` only when the configured Guild has no real active Meeting.

## Phase 14 profiles, release, and rollback

- CPU: `docker compose -f docker-compose.yml ...`
- NVIDIA GPU: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml ...`
- Secret scan: `uv run python scripts/scan_secrets.py`
- Full deployment: `docs/DEPLOYMENT.md`
- Rollback and data restore: `docs/ROLLBACK.md`

The application build owner is `lingxu-migrate`; Bot and Worker reuse the exact same image with
independent commands. The Windows launcher performs this build once and fingerprints TTS inputs.
Use `-RebuildAll` only for an explicit full TTS rebuild.

Preview local storage retention without deleting anything:

```text
uv run python scripts/preview_storage_cleanup.py
```

The JSON report is content-safe and Lingxu-scoped. `review-candidate` is not deletion authority;
the tool intentionally has no mutation mode and protects rollback/current/container references and
unverified backup headers.

Both Bot and Worker must exit code zero under `docker compose stop`. Startup recovery must report
expired Job requeue, Session reconciliation, `.part` cleanup, and tombstone cleanup counts before
readiness.
