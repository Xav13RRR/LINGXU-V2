# Lingxu Privacy Boundary

Lingxu separates consent for recording, realtime ASR and minutes, and limits data exposure at each
processing boundary.

- Every formal Meeting participant must run `/meeting consent` in that Guild before persistent
  recording, realtime ASR, or minutes processing accepts their content.
- `/meeting withdraw` linearises with receive callbacks and rejects future persistent frames
  immediately without rewriting the append-only audit history.
- Encrypted audio Segments are separated by authoritative Discord user ID. Logs and metrics never
  contain Tokens, DAVE/MLS keys, raw PCM, transcripts, minutes, prompts, or model responses.
- The retired standalone probe's unencrypted diagnostic files are legacy local data only. They are
  ignored by Git, are not created by the maintained runtime, and must not be imported into formal
  Meeting storage.

## Persisted-file boundary

- Formal Session artifacts use an AES-256-GCM envelope with a random nonce and authenticated
  Lingxu file-format context. The encrypted bytes do not retain a playable media header.
- `FILE_ENCRYPTION_KEY` must be URL-safe base64 that decodes to exactly 32 bytes. It is required
  before formal encrypted artifacts can be produced and is included in value-based log redaction.
- The key belongs only in the ignored `.env` or deployment secret store. It must never be committed,
  logged, copied into a manifest, or stored in SQLite.
- Atomic encryption writes the encrypted envelope through `.part`; decryption is intended only for
  Worker-controlled temporary processing. Recording must put plaintext in the configured
  tmpfs and remove it after use.
- Losing the key makes encrypted Session files unrecoverable. Replacing a key does not rotate
  existing data; rotation requires an explicit decrypt/re-encrypt migration.

## Consent and formal recording

- Realtime ASR, persistent recording, and minutes inclusion are three independent self-consent
  purposes. `/meeting consent` explicitly grants all three together; internal enforcement continues
  to check the specific purpose.
- Meeting consent is persisted under the exact `(Discord Guild ID, Discord user ID)` pair. It is
  reusable only for later Meetings in that Guild and is never treated as a global Discord-account
  grant. No existing per-Session consent is inferred into this table during migration.
- `/meeting withdraw` works with or without an active Meeting. It deletes that Guild-scoped grant
  and clears the user's three consent flags from every retained Session in that Guild so pending
  Worker processing also fails closed. Grants in other Guilds remain untouched.
- Per-Session participant flags remain the processing snapshot used by capture, ASR, console
  projection, and Worker filtering. The receive-path in-memory registry still linearises frame
  acceptance and withdrawal.
- Formal capture starts only after the host selects a voice channel with `/meeting start`.
  Persisted presence is synchronised to the non-Bot members actually in that channel, and only
  currently present users' per-Session grants seed the live registry.
- Join, leave, consent, and withdrawal events update the live purpose-specific registry. Moving
  outside the selected Meeting channel suspends all three live processing paths without deleting
  the durable same-Guild grant; returning to that channel restores only the still-persisted grants.
- Recording consent is checked and queue insertion is linearised with withdrawal. Frames before
  consent or after withdrawal cannot enter formal recording.
- Each authoritative Discord user ID owns a separate directory, FLAC Segment stream, timeline, and
  database metadata set.
- Plaintext FLAC exists only inside `RUNTIME_TMP/recording`; Compose mounts `/runtime` as a 64 MiB,
  mode-0700 tmpfs. Startup removes stale formal-recording plaintext.
- Personal deletion first stops that user's capture, then removes their exact encrypted directory
  and matching `audio_segments` rows. It does not delete another user's data.

## Realtime ASR

- Realtime ASR uses only `ConsentKind.REALTIME_ASR`; recording or minutes consent never authorises
  transcription.
- Withdrawal atomically rejects future frame insertion, removes that user's pending PCM, invalidates
  queued/in-flight result generations, and prevents their output from being persisted.
- Realtime PCM remains in bounded memory queues only. It is not written to a temporary or persistent
  ASR audio file.
- `raw_text` is immutable provider output. `normalised_text` is a separate value and never overwrites
  the raw text.
- Application logs and content-free metrics may include Session/user IDs, counts, error classes,
  detected language codes, and latency, but never PCM or transcript text.
- Model cache files contain public model weights and official public fixtures, not meeting audio.

## TTS cache and fallback

- TTS source text is sent only from the Bot to the local CosyVoice service on the internal
  `lingxu-tts` network. The service has no host port or external network and suppresses upstream
  progress, inference-text, stdout, stderr, and access logging.
- The opt-in model downloader has external network access but never receives meeting text. Model
  weights are mounted read-only into the serving container.
- Cache filenames are SHA-256 keys over provider configuration and normalised text. WAV content can
  reveal the spoken prompt and therefore remains deployment data under `/data/tts_cache`.
- Text fallback deliberately posts the original prompt to the selected meeting channel with all
  mentions disabled. It is used only when spoken delivery fails.
- Echo matching is memory-only and expires after playback plus the configured debounce. It is not
  a transcript store.

## Meeting and Scenario privacy

- Worker transcript/minutes include only users with explicit `MINUTES` consent.
- Public vote rows identify voters because the vote is explicitly public.
- Agenda-linked ballots are delivered through one Discord DM per eligible participant. Eligibility
  is a snapshot of non-Bot voice participants at vote open and does not depend on Meeting consent;
  consent continues to govern recording, ASR, and minutes only. DM failure does not silently remove
  a voter and the Guild slash-command fallback applies the same eligibility check.
- Agenda-linked vote rows retain voter identity and the latest Yes/No choice for exact one-person,
  one-vote overwrite and early-close behavior. Discord channel output, Worker artifacts, and the
  Host console expose only aggregate Yes/No/abstain counts and the final result. Individual choices
  and recipient rows are not logged or projected.
- Anonymous vote rows never store Discord user IDs. They store a Session+vote HMAC-derived negative
  pseudonym and token hash. Cast events omit actor and option; only final aggregates expose counts.
- Role-private content is AES-256-GCM encrypted. Unauthorized roles fail before decryption.
- Transcript, evidence, and deterministic minutes remain Meeting content under the same deletion
  and retention boundary.

## Local LLM privacy

- Meeting text is sent only to `lingxu-ollama` over the private Compose network. The service has
  no published host or LAN port and no cloud API key.
- Prompt context is limited to the untrusted utterance, public Meeting state, and the actor's
  allowlisted intent enum. Role-private content and anonymous voter mappings are excluded.
- Provider requests/responses and utterance text are never logged. Candidate persistence does not
  duplicate the text.
- `LLM_ALLOW_REMOTE=false` rejects a non-local endpoint at configuration validation.
- The model has no tools and cannot access Discord, TTS, files, SQLite, or Runtime.

## Resident voice Q&A privacy

- The resident mode is not a Meeting and never creates Meeting, participant, consent, recording,
  realtime-utterance, transcript, processing-Job, minutes, or artifact rows.
- Any non-Bot member whose PCM is received from the selected channel may invoke the local wake path.
  PCM, ASR question text, answer text, and model output remain bounded in memory and are not logged.
- The model receives one ephemeral question and a fixed mode marker only. It receives no Meeting
  context, channel messages, member data, transcript, audio, files, or repository access.
- Synthesized answer WAVs are created only under tmpfs `RUNTIME_TMP`, deleted after playback, and
  never written to the persistent TTS cache. There is no Discord text fallback for resident output.
- Durable recovery stores only Guild ID, voice-channel ID, creator ID, lifecycle state/reason, and
  timestamps in an append-only event stream. It contains no caller identity or conversation content.
- This consent-independent path does not grant or alter formal Meeting recording, realtime-ASR, or
  minutes consent. Starting a formal Meeting remains subject to those separate controls.

## Markdown agenda-import privacy

- Only the host may submit one `.md` attachment by directly mentioning the Bot in the Meeting's
  setup channel. The Bot does not follow links or fetch referenced documents.
- The decoded Markdown is retained in the session-scoped encrypted deployment database so an
  interrupted local job can be recovered and audited. It is deleted with the Meeting graph.
- Bounded source blocks travel only to the pinned `qwen3.5:9b` on the private Compose network.
  Attachment text, prompts, responses, inferred agenda text, and Meeting content are never logged.
- A closed `lingxu-agenda` JSON block bypasses the model. Only explicit mention IDs referenced by
  that block are retained with the import after Discord verifies Guild membership and rejects Bots;
  names are never resolved heuristically. Host-authored `identity_map` entries remain inside the
  same Session-scoped source document and are deleted with that Meeting graph.
- This path does not consume voice or change audio/transcript consent. It stores only the host's
  explicitly attached setup document and the resulting agenda/provenance.
- The model has no Discord or database tools. State remains unchanged until repository validation
  atomically commits every agenda row and its append-only event; any failure commits no partial list.
- A pronunciation alias is a user-managed Guild setting. It is queried only to synthesize the
  spoken name and does not replace identity/display-name fields, sorting keys, canonical Bot
  transcript text, participant transcript labels, panels, or Worker/minutes facts.

## Final post-Meeting minutes and remote-provider privacy

- The current normal final-minutes path uses the official DeepSeek API and
  `deepseek-flash`; it is not a local-only inference path. The independent Worker makes the
  request only after applying the durable consent timeline and privacy cutoffs.
- Participant speech is eligible only when the participant granted `MINUTES` consent for the
  corresponding interval. Non-consented and post-withdrawal utterances/Segments are excluded before
  transcript construction. Bot/system transcript rows may remain as deterministic Meeting context.
- The remote final-minutes request contains every non-empty consent-filtered source row, a compact
  dictionary of exact approved speaker labels, stable source ordinals/times, separate system
  context, compact ordered agenda and aggregate closed-vote facts, and a closed JSON output schema.
  Internal database/source keys, repeated identity fields, the Markdown template and preformatted
  authority tables are not sent. Controlled-speaking tables and active/pause details remain local
  for deterministic composition; system transcript rows may still describe process events. Meeting
  text and consenting participant display labels are still disclosed remotely and must be
  disclosed before participants grant minutes consent. This is compaction, not anonymization.
- A controlled-flow identity absent from the Worker's `MINUTES`-consented speaker inventory is
  deterministically redacted before `minutes.json`, the provider request, and Markdown: the public
  fact retains the turn/outcome but carries no Discord ID or display name. Pronunciation aliases
  are never used in these facts.
- The request excludes raw audio/PCM, encryption or DAVE/MLS material, credentials, local file
  paths, role-private content, individual voter-to-option mappings, anonymous-vote token hashes, and
  database access. Aggregate vote counts/results are allowed.
- Provider prompts, responses, reasoning, transcripts, and identities are never written to
  operational logs. The validated final Markdown is retained only as a Meeting artifact under the
  configured retention/deletion boundary. Content-safe attempt journals retain hashes, counts,
  allowlisted failure codes, bounded diagnostics, and source-ordinal/hash summary bindings only.
  Structured validation errors contain closed reason/section identifiers and bounded indices/counts,
  never model values or arbitrary JSON keys. Invalid production model drafts are not retained.
- The provider has no tools and cannot access Discord, SQLite, files, or Meeting Runtime. Local code
  validates fixed structure, exact speaker inventory, and authoritative agenda/speaking/vote rows
  before publication. A rejected model output can only be replaced with labelled basic minutes
  reconstructed from the same consent-filtered source records and authoritative facts (ADR-0164).
  Unverified prose is never included. The original failed model attempt is preserved and no
  automatic model retry occurs. Input/privacy/storage corruption still prevents publication.
- There is no retained local-Qwen post-meeting summary implementation or provider selector. The
  Worker fails startup when its fixed DeepSeek path is disabled or unavailable; it does not publish
  a local-model fallback as final minutes.

## Host console boundary

- The console is bound to localhost and one configured Guild; requests cannot supply another Guild
  or impersonated host ID.
- The configured/derived login token is never returned by an API, retained from login attempts, or
  stored in browser storage. The browser receives only an HttpOnly session cookie and CSRF value.
- WebSocket data requires that authenticated cookie and same origin.
- Participant IDs and consent status are host-authorized operational data. Final utterances are
  filtered by current realtime-ASR consent.
- Event payloads, event actor IDs, vote-entry user IDs, anonymous token hashes, private role
  content, raw model traffic, and secrets are not part of the console projection.
- `/metrics` has no Guild/user labels and no Meeting content.
