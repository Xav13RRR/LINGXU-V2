# Lingxu Architecture

This document describes the current production architecture. Historical phase headings below
identify when a boundary was introduced; they do not limit the currently deployed feature set.

```text
Independent Lingxu Discord Application
                |
                v
discord.py VoiceRecvClient + pinned inbound-DAVE fork
        | receive                              ^ TTS playback
        v                                      |
Discord user ID -> MeetingPcmSink -> consent-gated fan-out
                                  |                 |
                                  v                 v
                     encrypted per-user Segments   realtime ASR/voice assistant
                                  |
                                  v
                      independent post-meeting Worker
```

The Discord receive callback is deliberately narrow: authoritative user mapping, byte copy,
monotonic timestamp, and non-blocking handoff to the formal recording, ASR, and ephemeral-assistant
boundaries. Encryption, disk I/O, model/network calls, logging, and state transitions are outside it.

## Phase 2 persistence boundary

```text
Bot-side service / future Meeting Runtime
        |
        +--> MeetingSessionRepository --> SQLite WAL --> append-only events
        |                                  |              + snapshots
        |                                  + processing_jobs
        |
        +--> SessionFiles --> .part + fsync + SHA-256 + os.replace
                                   |
                                   +--> AES-256-GCM encrypted artifacts

One-shot lingxu-migrate --> exclusive Alembic DDL ownership
                                  |
                +-----------------+-----------------+
                |                                   |
        lingxu-bot exact-head check       lingxu-worker exact-head check
                                                    + startup recovery
                                                    + atomic lease/checkpoint/completion
```

The Phase 2 database is owned only by Lingxu. SQLAlchemy 2 repositories use short transactions,
SQLite foreign keys, WAL mode, a five-second busy timeout, and database constraints for one active
Session per guild, event sequence uniqueness, and job idempotency.

`meeting_events` is immutable during a Session: SQLite triggers reject update and delete. The only
deletion exception is coordinated retention cleanup after the parent Session has first entered
`DELETED`; the service removes the Session directory and then cascades the database graph.

The append-only Meeting event names and deterministic state-transition graph are owned by the pure
`lingxu.domain` layer. Meeting orchestration, storage, Scenario, and Worker code consume that
contract; legacy `lingxu.meeting.events/state` imports are compatibility-only re-exports.

The independently deployed Worker owns the complete post-meeting pipeline described below. It
never runs schema DDL: startup requires the exact configured Alembic head before recovery or job
leasing begins.

## Phase 3 recording path

```text
Discord user PCM
  -> three-purpose consent registry
  -> recording-consent linearisation
  -> bounded ordered frame queue
  -> per-user silence/max-duration Segmenter
  -> FLAC under RUNTIME_TMP tmpfs
  -> fsync + AES-256-GCM + atomic replace
  -> Session audio/<user_id>/<segment>.flac.enc
  -> audio_segments metadata
```

Queue controls and frames share one ordered structure; overflow removes only the oldest audio frame,
never a withdrawal/finalisation control. No FFmpeg, codec, encryption, file, database, or network
operation runs in the Discord receive callback.

## Phase 4 realtime sentence ASR path

```text
Discord user PCM
  -> independent realtime-ASR consent check
  -> bounded queue per Discord user ID
  -> round-robin frame consumer
  -> per-user energy VAD / UtteranceBuilder
  -> bounded utterance queue
  -> timeout-isolated SenseVoiceProvider (CPU, auto language, hotwords)
  -> consent-generation recheck
  -> raw_text + normalised_text persisted separately
```

The AudioSink performs only consent linearisation, one byte copy, timestamping, and non-blocking
queue insertion. One noisy user can overflow only that user's frame capacity. ASR workers are
bounded independently from frame capture, so model latency cannot block Discord receive.

Withdrawal increments a per-user generation, clears only that user's queued/builder PCM, and
invalidates pending or in-flight provider output before persistence. A native inference call already
running in a worker thread cannot be force-killed safely, but its result is discarded and no new
call for that consent generation may begin.

SenseVoice receives normalised mono `float32` at 16 kHz, uses automatic language selection, and is
wrapped by the replaceable `RealtimeASRProvider` protocol. Model loading is lazy and validates the
approved SenseVoice and FSMN-VAD weight hashes.

## Phase 5 TTS and playback path

```text
Meeting prompt
  -> priority + interrupt policy + dedupe key
  -> PlaybackScheduler
  -> TTSProvider
  -> private hash-verified CosyVoice SFT service
  -> atomic hashed WAV cache
  -> Discord FFmpegPCMAudio voice send
  -> one idempotent TTS_FINISHED
       \-> on failure: Discord text fallback
```

Only one artifact plays at a time. Emergency/host/process prompts can interrupt lower priority
audio; repeats cannot interrupt normal work. Source text is never placed in the executable command
line, access log, inference log, or cache filename. Fixed process prompts and dynamic names/agendas
use the same validated cache path. The default `中文女` speaker is enforced at both client and
service boundaries.

`EchoGuard` holds only canonicalised Bot speech in memory while playback is active and for a
configured 200–500 ms debounce after completion. Rule intent processing can reject similar ASR
text during that window, preventing the Bot's own playback from causing a high-risk action.

## Phase 6 deterministic Meeting Runtime

Commands from Discord, rules, or system tasks enter `MeetingRuntime`. The Runtime validates the
fixed edge and host guard, updates the materialised Session, and appends the state event in one
transaction. It then returns declarative action names to the controller; side effects never select
or rewrite state.

`back` is a new reverse event. `pause` persists the exact resume state. Periodic snapshots accelerate
recovery but are never authoritative over the append-only event stream. See
`docs/MEETING_STATE_MACHINE.md`.

## Phase 7–11 application layers

- Discord `/meeting` commands and persistent Views call `MeetingController`; only
  `MeetingRuntime` chooses Meeting state targets.
- `/meeting start` resolves the host's current ordinary voice channel, verifies View/Connect/Speak,
  makes persisted participant presence match that channel, and starts one
  `VoiceRecvClient`-backed `MeetingVoiceCoordinator` before asking Runtime to advance. A failed
  voice/DAVE/capture startup leaves Runtime unchanged.
- The formal coordinator composes the purpose-specific in-memory consent registry, non-blocking
  `MeetingPcmSink`, encrypted per-user recorder, realtime ASR service, and Discord TTS playback.
  Runtime action names drive process prompts but never choose a state.
- Runtime stop/finalisation actions are executed in declared order: timers are cancelled, receive
  stops, ASR drains, encrypted Segments are persisted, Discord disconnects, and only then is the
  idempotent post-meeting Job inserted.
- Agenda, participant identity/role, presence, timers, and votes are materialised in SQLite. The
  speaking queue replays immutable events. Opening and timed-agenda deadlines are persistent;
  after Bot restart the voice coordinator reconnects to the persisted Meeting channel before the
  Runtime re-emits idempotent current-state actions.
- Each agenda stores one optional total discussion duration. An agenda that requires a vote also
  stores its vote content and duration. Entering `VOTE_OPEN` creates an immutable eligible-recipient
  snapshot from the current non-Bot voice roster, persists DM delivery/message identifiers, and
  starts a restart-safe vote deadline. Static DM buttons resolve the ballot from message ID plus
  user ID; the same eligibility guard protects the slash-command fallback.
- Agenda votes close when every snapshotted recipient has a Yes/No row or when the deadline expires.
  Missing rows become abstentions. The append-only `VOTE_CLOSED` fact includes the agenda ID,
  aggregate counts, eligible/responded totals, and deterministic Yes/No/tie result. A persistent
  five-second `FLOW:VOTE_RESULT` timer starts only after result delivery finishes, then advances the
  Runtime to the next agenda.
- Explicit Meeting consent has a durable `(guild_id, user_id)` grant. Joining a Session copies only
  that Guild's grant into the Session participant row. Once every currently present participant has
  all three grants, the Controller asks the Runtime to idempotently advance `CONSENT_CHECK` to
  `OPENING`; it also rechecks after a participant leaves.
- `/meeting withdraw` deletes only that Guild grant and clears all three consent flags for that user
  in every retained Session belonging to the same Guild. Other Guilds are never queried or changed.
- Stop/completion inserts one idempotent `POST_MEETING` Job. The separate Worker runs 22
  checkpointed stages through Artifact rendering, Discord publication, and cleanup scheduling.
- One Bot Application may be installed in many Guilds and exposes global Guild-only commands. A
  database-unique `PRIMARY` service-capacity lease serialises formal Meetings, resident mode, and
  agenda-model imports across the entire workstation. Its append-only event stream
  records only Guild/workload ownership metadata. Formal Meeting ownership survives voice
  finalisation and is released only after publication, terminal processing failure, or deletion.
  Competing Guilds receive a content-free fixed response with no owner identity.
- Missing consented Segments may use the approved SenseVoice speech recognizer. Deterministic
  `minutes.json`, event/vote/controlled-speaking facts, evidence, privacy exclusions, speaker
  inventory, and consistency checks remain local and authoritative. Controlled-speaking facts are
  replayed from the append-only flow and timer events: active intervals exclude matching Meeting,
  enforcement-recovery, and host-intervention pauses; extension, early finish, skip, absence,
  overtime, and optional follow-up participation remain closed event facts. The current
  final-publication path sends every non-empty consented source row in compact human/system
  partitions, a fixed speaker dictionary, and compact agenda/aggregate-vote context to the
  owner-approved official DeepSeek API. ADR-0165 requests typed JSON claims and integer evidence
  arrays, not model-written Markdown tables. Local code validates the schema/references and
  composes the fixed layout and full authority tables, retaining untouched transcript rows.
  The composed Markdown is rejected unless its fixed
  structure, speaker inventory, agenda, speaking-fact, and vote tables match local authority
  exactly. LLM output cannot mutate Meeting state or any speaking row.
- Strict Scenario YAML/JSON becomes immutable published snapshots over allowlisted events/actions.
- Scenario roles fail closed. Anonymous votes store scoped pseudonyms/token hashes rather than
  Discord IDs. Role-private content uses the Session AES-GCM envelope.

## Phase 12 local structured-intent boundary

```text
untrusted text
  -> exact local RuleIntentResolver
  -> actor/state-specific allowed intent set
  -> private-network Ollama OpenAI-compatible adapter
  -> request-specific Pydantic JSON Schema
  -> application allowlist + confidence validation
  -> high confidence: Controller dispatch
       \-> low confidence: persisted candidate -> same-user confirm/reject
  -> existing permission guard + deterministic Runtime/event log
```

The provider receives no database handle, Runtime, TTS, Discord client, private role content, or
tool definitions. The persisted candidate contains only the intent enum, actor ID, source,
confidence, optional utterance ID, and decision; it does not duplicate utterance text. Ollama is a
separate GPU service with no published port. The Bot verifies the pinned model manifest digest
before each provider call.

## Phase 13 local Host console

The console runs inside the Bot process so every write reaches the same deterministic
`MeetingController` and manual TTS can use the existing Discord voice client. Compose publishes
only `127.0.0.1:8765`.

- The public HTML shell and content-free `/healthz`/`/metrics` expose no Meeting data.
- A configured operator secret is exchanged for an in-memory, eight-hour HttpOnly/SameSite session.
- Every write also requires same-origin and per-session CSRF validation.
- Authenticated WebSockets project one configured Guild only; the client cannot select a Guild or
  actor ID.
- The server derives the action actor from the active Session's host and passes
  `source=HOST_CONSOLE` through the deterministic Controller.
- Snapshots omit event payloads/actors and vote-entry rows. Anonymous and public votes expose only
  option counts.
- Final utterances are projected only while the participant's realtime-ASR consent remains true.
- Simulator state is process-local and cannot mutate the live Runtime.

## M1+P2 Markdown agenda-import boundary

```text
host direct mention + one .md attachment in setup channel
  -> host/channel/state/empty-agenda/size validation
  -> persistent idempotent PROCESSING import
  -> if one closed lingxu-agenda JSON block:
       explicit mention extraction -> Discord Guild/non-Bot validation
       -> host identity_map resolution -> built-in template compiler (no model call)
  -> otherwise:
       decoded, line-identified, bounded overlapping source blocks
       -> private-network pinned qwen3.5:9b + closed native JSON Schema
  -> source/provenance/confidence/language/vote/capacity validation
  -> deterministic P2 defaults (agenda 15 min; vote 60 sec)
  -> one transaction: all agenda rows + append-only AGENDA_IMPORT_COMMITTED event
  -> READY + content-safe “准备完毕” summary
```

The Bot enables only the standard Guild Messages intent; a direct mention exposes the required
attachment metadata without the privileged Message Content intent. The model receives no tools,
database handle, Discord client, or Runtime. It cannot partially write agenda rows. One PROCESSING
record per Session blocks `/meeting start`; restart recovery resumes it, and provider unavailability
fails it closed rather than leaving a permanent start lock.

Guild pronunciation aliases live in their own `(guild_id, user_id)` settings projection. The voice
adapter looks them up only while constructing TTS input for a controlled-flow notice. The immutable
speaking event, canonical Bot transcript row, text fallback, public/private panels, Discord-name
ordering, participant label, and Worker facts retain the Discord identity/display name.

## Authoritative controlled-speaking minutes boundary

The Worker never infers a planned or actual turn from transcript text. `SPEAKING_FLOW_STARTED`
freezes the plan and identity snapshot; completed turn events bind the result. Matching
`TIMER_PAUSED`/`TIMER_RESUMED` events partition each granted-floor interval, including timer IDs for
Meeting-wide pauses and explicit reasons for enforcement recovery or host intervention. The exact
active interval sum is `actual_floor_ms`; preparation and paused spans are excluded. Plan/start/
finish identities, planned duration, and extension totals must agree or minutes generation fails.

The closed minutes payload retains the active and pause intervals plus explicit early-finish,
skip, whole/partial absence, overtime, and follow-up participation facts. Confirmed `next` and
Meeting `stop` first append a terminal turn/follow-up event and `SPEAKING_FLOW_COMPLETED`; they do
not merely cancel the timer, so the final occupied interval remains reconstructable. `back` and
generic state `skip` fail closed while a controlled agenda is active. A member outside the Worker’s
`MINUTES`-consented identity inventory remains represented in plan/outcome counts but has both ID
and display name replaced with `null`/`未获纪要授权成员` before `minutes.json`, the remote request, or
public Markdown is built. Both deterministic and approved full-transcript renderers consume one
ordered speaking-row builder. The latter uses preview contract 3 and rejects any response whose
heading, table header, separator, row count, order, or row bytes differ from local authority.

## Current post-Meeting Worker boundary

The independently deployed Worker owns 22 ordered, checkpointed stages from Session verification
through privacy projection, transcript recovery/canonicalisation, topic/outcome evidence,
deterministic facts, final Markdown generation, consistency validation, publication, and cleanup.
Only `minutes.md` is attached to Discord under the current `task11-one-file-v1` contract. Local
machine-readable transcript, evidence, event, privacy, and minutes artifacts remain registered for
audit and recovery. A final-minutes provider or output-contract failure produces explicitly labelled
basic minutes from authoritative facts and consented source records (ADR-0164). These pass the same
privacy/evidence/publication checks; reuse requires deterministic byte-for-byte reconstruction.
The failed model attempt stays in its journal. The Worker never automatically repeats an identical
request, and input/storage corruption or ambiguous Discord delivery still require recovery.
