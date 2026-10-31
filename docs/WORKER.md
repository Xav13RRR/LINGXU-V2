# Independent Lingxu Worker

`lingxu-worker` is a separate process using only the Lingxu database, Session root, encryption key,
logs, health marker, and Docker service. It does not import, call, or require Lingji.

Compose supplies a Worker-specific environment projection and attaches the Worker only to the
default application network. Bot voice/assistant/agenda/TTS settings are absent, and the Worker is
not attached to the internal `lingxu-llm` Ollama network. Post-meeting inference uses only the
approved official DeepSeek HTTPS endpoint.

## Job lifecycle

1. Upgrade the independent database with Alembic.
2. Run startup recovery for `.part` files, expired leases, manifests, and tombstones.
3. Atomically claim one eligible job using a bounded lease.
4. Heartbeat the exact Job/Worker lease throughout in-flight processing without changing its
   checkpoint; cancel local work if ownership is lost.
5. Persist the current stage and checkpoint under the owning Worker ID.
6. Stop and join the heartbeat, then complete or fail the job only while the same Worker still owns
   it.

The unique `(session_id, job_type, idempotency_key)` constraint makes job creation idempotent. The
claim is a single SQLite `UPDATE ... RETURNING` operation, so concurrent Workers cannot both own the
same job.

`PERSISTENCE_PROBE` checkpoints `PERSISTENCE_READY` and completes without model inference.
It checks Bot-side job creation and independent Worker claiming; meetings use `POST_MEETING`.

## Run

```text
uv run python -m lingxu.worker_main
```

The process is ready after `RUNTIME_TMP/worker.ready` exists and the structured
`worker_persistence_ready` event is emitted. Shutdown removes the marker and closes the database
engine.

## Current production pipeline

The Worker allowlists both `PERSISTENCE_PROBE` and `POST_MEETING`. The latter checkpoints 22 ordered
stages:

`VERIFY_SESSION`, `BUILD_CONSENT_TIMELINE`, `APPLY_PRIVACY_CUTOFFS`,
`VERIFY_AUDIO_SEGMENTS`, `FINALISE_REALTIME_UTTERANCES`, `TRANSCRIBE_MISSING_SEGMENTS`,
`VALIDATE_LANGUAGES`, `CANONICALISE_TRANSCRIPT`, `NORMALISE_ENTITIES`, `SEGMENT_TOPICS`,
`EXTRACT_TOPIC_OUTCOMES`, `LINK_EVIDENCE`, `VALIDATE_OUTCOMES`, `MERGE_TRANSCRIPT`,
`MERGE_MEETING_EVENTS`, `CALCULATE_LANGUAGE_DISTRIBUTION`, `GENERATE_EVIDENCE`,
`GENERATE_MINUTES`, `VALIDATE_MINUTES_CONSISTENCY`, `RENDER_ARTIFACTS`, `PUBLISH_DISCORD`,
`SCHEDULE_CLEANUP`.

`PostMeetingPipeline` coordinates this order. `lingxu.worker.stages` owns the immutable stage
registry, `lingxu.worker.audio` owns segment transcription, `lingxu.worker.topic_artifacts` owns
durable topic policy/artifact/reuse orchestration, `lingxu.worker.outcome_artifacts` owns outcome
artifact codecs, fixed-provider provenance, terminal partition persistence, and unified-budget
journals, and `lingxu.worker.publication_recovery` owns publication-manifest and terminal-recovery
validation. These modules do not change the append-only event, checkpoint, provider, model, or
publication authorities.

The production Worker has one fixed post-meeting inference path: official DeepSeek API
`deepseek-flash`. It fails startup if that capability is disabled or its exact policy/key is
missing. Ollama is a Bot-side dependency for intent, bounded voice answers and agenda import.

Transcript source keys and Artifact uniqueness make internal stage retries idempotent. Discord
publication uses a stricter external-side-effect contract: the Worker commits one non-overwriting
`PREPARED` intent before its single automated send attempt, then atomically stores a cross-bound
`FINALIZED` intent and receipt. A valid legacy receipt or finalized pair is terminal. A remaining
PREPARED intent, duplicate/mixed record, invalid hash/revision, or attachment mismatch fails
nonretryably and is never resent automatically. With the approved DeepSeek final-output provider
enabled, `GENERATE_MINUTES` sends one compact privacy-filtered complete transcript. ADR-0165 uses
a speaker dictionary, stable source ordinals, five-minute human-speech windows and separately
labelled system context. Every non-empty source row remains; internal database/source keys and
repeated identity fields are removed. Only compact agenda and aggregate closed-vote context goes
to the model, never individual voter mappings. The request explicitly selects JSON-object mode
and embeds a closed output schema. Typed claims carry integer evidence arrays; local code checks
references, human support, consensus speaker count and classifications before composing Markdown.
Optional adjacent same-speaker transcript edits are reconciled; all untouched source rows remain.
Full authoritative tables and the fixed layout are composed locally, not copied by the model.
Publication still requires their exact ordered match. Summary lines have private hash-bound source
bindings in addition to the existing evidence audit. Agenda state, vote counts/results, roster,
timestamps, and the internal `minutes.json` remain deterministic facts. Thinking-high, the exact
model identity and the durable single-attempt guard apply to every request. The final request uses the
dedicated `SUMMARY_FINAL_MINUTES_MAX_OUTPUT_TOKENS=131072` ceiling; the separate
`SUMMARY_LLM_MAX_OUTPUT_TOKENS=32768` continues to bound topic/outcome partitions. Any
provider or output-contract failure writes its allowlisted error code into the content-free attempt
journal. Under ADR-0164, final-generation failures produce clearly labelled deterministic basic
minutes with authoritative facts, bounded attributed excerpts and all retained source rows. The
Worker validates and publishes this artifact without another model request. Its manifest records
`generation_method=deterministic_basic`, `status=fallback`, the original failure code, and a renderer
revision; currentness checks reconstruct and compare its complete bytes and private audit. The
failed model journal is preserved. Corrupt source/journal data, cancellation and publication
ambiguity retain their failure boundaries. An explicit host `/meeting retry` on a failed Job
increments a durable attempt generation, preserves the
prior generation's journal, and authorizes exactly one new request under a distinct attempt file.
The DeepSeek request has a 1,200-second total transport timeout for non-streaming thinking-high completion.
Failed provider attempts also retain a closed `failure_diagnostics` mapping in their generation
journal. It may contain only phase, elapsed time, HTTP status,
allowlisted transport/finish categories, content/reasoning presence states, reasoning character
count, response byte count, and aggregate usage counts. It can never contain provider text,
reasoning text, prompt/transcript content, participant identity, or arbitrary exception messages.
Structured validation additionally records `validation_details`: a closed reason/section and
optional bounded item index/reference count, never Pydantic input, arbitrary field names or prose.
The final request limit, response format and structured-contract revision bind the attempt manifest
and request digest. Any existing generation journal blocks new inference even when the request
changes; restart/upgrade must not silently spend another attempt. The preview script uses this same
structured contract. Legacy Markdown providers remain available for retained/test contracts.
JSON mode is syntax assistance, not proof of semantic correctness; human content review remains
necessary. See the [official JSON-mode contract](https://api-docs.deepseek.com/guides/json_mode/).

Before resuming a checkpoint that has completed `MERGE_MEETING_EVENTS`, the Worker verifies that
`events.json` is the exact current export of the append-only event log. A stale or symlinked file
rewinds execution to that stage and rebuilds the event export and downstream deterministic
artifacts. If the existing full-transcript minutes remain current, the Worker reuses them without
another DeepSeek request and retains their renderer-input digest for the remaining checks.

The Worker renders `minutes.md` and machine-readable `minutes.json` in addition to the canonical
transcript and evidence. Under the `task11-one-file-v1` contract, Discord receives exactly one
attachment: `minutes.md`. That document includes the complete readable canonical transcript.
`minutes.json`, `transcript.txt`, canonical transcript JSON, and evidence remain registered local
Artifacts for recovery and audit; they are never attached by the new contract.

## Discord publication reconciliation

If a post-meeting Job fails with the content-safe publication-reconciliation status, do not retry
the automated send. Inspect the configured Discord publication channel without copying Meeting
content into logs or support notes:

1. If the exact bound attachment set was already sent, run the explicit command below with the
   existing PREPARED session and the operator-verified Discord channel/message receipt, then use
   the existing failed-job retry operation. The repository validates the requested channel and
   bound manifest before atomically constructing FINALIZED plus its receipt. There is no generic
   overwrite or auto-resend command. The next Worker claim validates the finalized pair and
   completes without a send.
2. If no message was sent, manually publish the exact manifest bound in the PREPARED intent, bind
   the resulting channel/message IDs through the same controlled repository operation, and then
   retry the failed Job.
3. If the channel state or Artifact binding cannot be proven, leave the Job failed. Never delete,
   overwrite, or recreate an intent to force an automated resend.

Operational output must remain content-safe: report only the allowlisted reconciliation reason and
status, never attachment text, transcript/minutes content, private IDs beyond the operator-supplied
Discord receipt, secrets, audio, or key material.

```text
uv run python scripts/reconcile_discord_publication.py --session-id <ULID> --channel-id <DISCORD_CHANNEL_ID> --message-id <DISCORD_MESSAGE_ID>
```

The command prints only `{"status":"finalized"}` or an allowlisted content-safe reconciliation
status/reason. It never sends a Discord message.

The command and Worker bind publication validation to the deployment's trusted session root. Each
file in the stored contract must be at its exact canonical Meeting path, be a regular non-symlink
file, and match the registered byte size and streaming SHA-256 immediately before preparation or
finalization. A `task11-one-file-v1` intent binds only `minutes.md`; an already stored legacy
four-file intent continues to bind its ordered manifest. Legacy four-file `FINALIZED` pairs and strict
legacy receipts remain terminal and never resend. A legacy four-file `PREPARED` intent is never
converted to the single-file contract automatically: it requires manual reconciliation under its
stored contract. If any PREPARED file is missing or changed, finalization fails without
modifying the intent or resending. Unknown flags and malformed values also return only the
allowlisted `invalid_arguments` result; the parser never echoes raw operator input.
