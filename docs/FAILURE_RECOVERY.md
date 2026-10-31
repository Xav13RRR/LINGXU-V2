# Lingxu Failure and Recovery

- Formal `/meeting start` does not advance Meeting state unless Discord connection, inbound DAVE
  readiness, encrypted recording, and realtime ASR startup all succeed.
- A repeated `/meeting start` for the same active Session/channel is connection-idempotent and is
  the operator recovery path after Bot restart.
- An isolated Opus failure is classified by type without logging packet data or DAVE/MLS material.
- Recording queue pressure is content-safe telemetry; the receive callback remains non-blocking.
- A crash may leave `.part` files. Startup removes only contained incomplete files, and they are
  never registered or reported as completed artifacts.

## Startup recovery

- Every important Session write uses a sibling `.part`, flush, `fsync`, SHA-256 calculation, and
  atomic replacement. Startup removes orphan `.part` files only within the configured Session root.
- Active database Sessions are reconciled with `manifest.json`. A missing directory skeleton or
  manifest is recreated from the database row; a stale manifest state is updated from the database.
- A `RUNNING` processing job whose lease expired is requeued as `PENDING`. The next Worker claims
  it atomically and continues from the persisted checkpoint.
- Job ownership is enforced on checkpoint, completion, and failure updates. A stale Worker cannot
  complete a job after another Worker has claimed it.
- A Session already marked `DELETED` is treated as an interrupted tombstone: startup deletes its
  exact Session directory and cascades its database graph.
- SQLite WAL and foreign keys are enabled on every connection. Transactions remain short so Bot
  and Worker can coordinate through the same independent database.

Recovery never guesses meeting content from files. The database state and append-only event
sequence are authoritative. Meeting Runtime replays that sequence during recovery.

## Recorder recovery

- Closed encrypted Segment names are scanned on recorder start; numbering resumes at the next value
  and never overwrites a prior Segment.
- A crash can leave plaintext only in the configured runtime temporary root. Bot startup and
  recorder startup remove that exact stale recording subtree before accepting new formal audio.
- Already encrypted Segments remain authenticated and independently decryptable after restart.
- An encryption or cleanup failure fails recorder finalisation and records only the error class.
- Withdrawal controls are ordered after already accepted frames; the acknowledgement is not
  returned until the user's current Segment is encrypted or the recorder has failed.

## Realtime ASR isolation

- Frame and utterance queues are bounded. Overflow is counted per user and never blocks AudioSink.
- Provider calls have a hard async timeout. Timeout, inference error, empty output, or persistence
  error affects only that utterance; the consumer and other users continue.
- Consent generation is checked immediately before inference and again before persistence. A result
  finishing after withdrawal is discarded.
- Consent is memory-only and fail-closed after restart. Accepted but unpersisted realtime PCM is
  intentionally lost on crash; it is never reconstructed from another user's recording.
- Realtime utterances already committed to SQLite remain available after restart. Later Worker
  transcript finalisation remains a separate phase.
- A missing model cache is a Doctor warning before first use; partial/wrong-sized cache is a Doctor
  failure, and a weight hash mismatch fails model initialisation.

## Playback isolation

- The CosyVoice service verifies every approved model file before readiness. The Bot checks the
  service's model digest and PCM WAV structure before caching. Any mismatch uses text fallback.
- The serving model volume and root filesystem are read-only, the API is private-network-only, and
  the external-network downloader never receives meeting text.
- Synthesis and playback failures are caught per request, sent once through text fallback, and
  completed with status `FALLBACK`; the scheduler continues to later requests.
- A request ID and dedupe key are accepted once. `TTS_FINISHED` is emitted at most once even if a
  completion callback is repeated.
- Higher-priority interruption completes the old request as `INTERRUPTED` without sending a false
  failure fallback, then starts the queued priority request.
- Every WAV is written as `.part`, validated as signed 16-bit PCM, fsynced, and atomically renamed.
  A failed synthesis removes its partial file.
- Fixed prompts are pre-generated into the persistent cache at deployment. Cache loss causes
  deterministic local regeneration and does not affect Meeting state recovery.

## Meeting Runtime recovery

- Runtime restart replays every immutable event and cross-checks the latest snapshot and
  materialised Session state.
- A broken state-before chain, missing/extra sequence, snapshot mismatch, or row mismatch raises
  `MeetingRecoveryError`; no automatic history rewrite occurs.
- Repeated `start`, `pause`, and `stop` at their achieved states are no-ops and do not duplicate
  events or side effects.
- `back` appends a reverse transition, preserving all earlier evidence.

## Meeting services and scenario recovery

- Final-minutes model/transport/validation failures produce labelled basic minutes without another
  inference (ADR-0164). These include saved meeting facts, attributed excerpts and all consented
  source rows, and explicitly leave unverified conclusions for human review. The original attempt
  remains failed in the audit journal while the Job can complete after validated publication.
  Corrupt input/journals, storage failure and ambiguous Discord sends still stop delivery. Existing
  failed Jobs are not silently requeued; completed publications are not regenerated.

- Speaking state replays events; timers persist deadlines/remaining milliseconds; reminder and
  expiry markers prevent duplicate notices or vote close.
- Agenda vote eligibility and DM message IDs persist separately from vote entries. A Bot restart
  reuses delivered ballots and the open deadline; undeliverable DMs remain eligible and can use
  `/meeting vote cast`. Pause/resume updates both the timer and vote deadline. Early completion
  cancels only that vote timer, while the five-second result transition is itself persistent.
- Each Worker stage persists `completed_stages`. Transcript source keys, Artifact unique keys, and
  publication records make retry idempotent.
- Worker downtime does not change an active Bot Meeting. Deleted Sessions cascade their Jobs.
- Scenario versions are immutable snapshots. Simulation is deterministic from snapshot, event
  sequence, role, and clock.
- Anonymous tokens are Session+vote scoped. Private content remains encrypted at rest.
- Migration `0012_service_capacity` persists the single workstation slot and an append-only
  acquire/recover/release stream. Bot restart retains a valid formal Meeting or resident owner,
  releases non-durable probe/import owners, and releases any owner whose durable source record is
  no longer recoverable. A wrong Guild/mode/owner token cannot release the slot. Agenda imports
  that were already accepted wait asynchronously for a recovered slot instead of running beside a
  different Guild's workload.

## Model and confirmation recovery

- Exact rules run before model inference and remain available when Ollama is stopped or unhealthy.
- Provider timeout, HTTP failure, oversized/invalid JSON, schema violation, unknown intent,
  allowlist escape, or model-digest mismatch returns no executable candidate.
- Low-confidence candidates persist with an undecided value and can be confirmed after Bot
  restart. Only the original participant in the same active Session may decide them.
- Confirmation is atomically claimed once. Repeated or concurrent confirmation cannot dispatch a
  second action. Runtime and component guards remain the final authority if Meeting state changed
  while confirmation was pending.

## Console and deployment recovery

- Browser refresh reuses an unexpired HttpOnly session and rebuilds the full view from persistent
  Meeting state; WebSocket reconnect never becomes a second state authority.
- Bot restart invalidates browser sessions and in-memory Simulator state. Live Meeting state,
  timers, queue, vote, consent, utterances, and append-only events recover from SQLite.
- A console action is authorized again against the current active Session immediately before the
  deterministic Controller call. Stale UI cannot bypass state or host guards.
- SIGTERM closes console WebSockets before the Bot database/Discord client and removes readiness
  markers. Worker SIGTERM finishes the current poll boundary and releases resources.
- CPU and GPU profiles share identical storage and model identity. Switching profiles does not
  migrate or rewrite Meeting data.
- Rollback restores application images only when schema-compatible; otherwise it restores the
  complete pre-upgrade data volume. See `docs/ROLLBACK.md`.
