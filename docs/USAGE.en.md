# 聆序 LINGXU Usage Guide

This guide covers the current Discord surfaces: formal meetings under `/meeting` and resident voice
Q&A under `/assistant`. Only one mode may own the Bot's voice connection in a Guild at a time. Stop
the current meeting or assistant before starting another mode.

[中文版](USAGE.zh-CN.md) · [README](../README.md) · [English deployment guide](DEPLOYMENT.en.md)

## 1. Before use

- The Bot is online and has View Channel, Connect, and Speak in the target voice channel.
- The Bot has View Channel and Send Messages in the relevant text surface.
- `/meeting` and `/assistant` appear in slash-command autocomplete.
- The host is present in the ordinary voice channel that will be used.
- Participants who need DM ballots should allow server-member DMs. If DMs are disabled, use the
  in-Guild `/meeting vote cast` fallback.
- Real recording, ASR, and minutes require each participant to run `/meeting consent` in this Guild.

## 2. Complete meeting workflow

### 2.1 Create the meeting

The host runs in the setup text channel or the target voice channel's built-in text chat:

```text
/meeting create
```

The Bot creates one Meeting Session and publishes the persistent host control panel. A Guild can
have only one active Meeting. The creator becomes the host. If the creator is already in voice,
current non-Bot members are included in participant presence.

### 2.2 Add agenda items with slash commands

```text
/meeting agenda add title:<agenda title> duration:<minutes> requires_vote:<true|false> language:<auto|zh|yue|en|ja|ko>
```

- `title`: announced by the Bot when the item starts;
- `duration`: total discussion minutes for this item, excluding voting; omit it for an untimed item
  that waits for host `/meeting next`;
- `requires_vote=false`: no agenda-linked ballot;
- `requires_vote=true`: Discord opens a modal requiring ballot text and a 10–3600 second duration;
- `language=auto`: automatic choice, or explicitly select Chinese, Cantonese, English, Japanese,
  or Korean.

Manage the pending agenda:

```text
/meeting agenda list
/meeting agenda remove position:<number>
/meeting agenda reorder from_position:<old> to_position:<new>
```

Removal and reordering apply only before the Meeting starts.

Built-in controlled-agenda templates are available through:

```text
/meeting agenda template kind:<round_robin|standup|pre_vote_statements|alternating_sides|presentation|presentation_q_and_a>
```

Round-robin, stand-up, and pre-vote statements snapshot Discord display-name NFKC/case-fold order.
Alternating sides requires two distinct selected members; presentation templates require one or two
presenters. Total duration, per-turn seconds, preparation, follow-up, and strict mute are explicit
options. Follow-up exists only when the host selects it or uses `presentation_q_and_a`. Strict mute
requires `Mute Members` before start and supports ordinary voice channels only. A host may also be a
selected participant, but host control authority alone never grants the floor.

### 2.3 Import an agenda from Markdown

After `/meeting create`, the host may instead use the same setup channel:

1. select the Bot using a real Discord mention;
2. attach exactly one `.md` file to that same message;
3. wait for the content-free “received” acknowledgement;
4. wait for “准备完毕”, the agenda count, and configured total duration.

This route accepts only the Meeting host, an empty pending agenda, a not-yet-started Meeting, and one
bounded Markdown attachment. Mention matching uses the immutable Bot user ID, so changing the Bot's
global name or Guild nickname does not matter. Plain typed `@name` text is not a mention.

Local `qwen3.5:9b` applies the agenda-import policy to infer source-backed titles, order, language,
explicit vote semantics, and duration. Missing discussion duration receives the fixed 15-minute
default; an explicit vote without a duration receives the fixed 60-second default. The model has no
Discord, database, or tool access. All source references, bounds, languages, votes, confidence, and
item counts are validated before one atomic commit; partial import is not allowed.

To extend the same templates without any agenda-model call, a `.md` file may contain exactly one
closed JSON block:

Replace `PRESENTER_USER_ID` and `OTHER_USER_ID` below with the intended users' IDs in your local
file before importing. They are placeholders, not usable mentions; never commit the filled file.

````markdown
```lingxu-agenda
{
  "schema_version": 1,
  "identity_map": {"presenter": "<@PRESENTER_USER_ID>"},
  "items": [
    {
      "template": "PRESENTATION_Q_AND_A",
      "title": "Product presentation",
      "duration_minutes": 20,
      "presenters": ["presenter"],
      "enforce_mute": true
    },
    {
      "template": "ALTERNATING_SIDES",
      "members": ["presenter", "<@OTHER_USER_ID>"],
      "turn_seconds": 120
    }
  ]
}
```
````

Members must be real Discord mentions or host-supplied `identity_map` keys that map to mentions.
Natural names are never matched. Lingxu verifies that every reference is a current non-Bot Guild
member before submission. Unknown fields, unmapped names, duplicate/invalid members, or parameters
that do not belong to the selected template reject the complete import without a partial write.

### 2.4 Participant consent and withdrawal

Each participant runs once per Guild:

```text
/meeting consent
```

The command explicitly grants realtime ASR, persistent recording, and minutes inclusion. The grant
is reusable across later Meetings in this Discord Guild only; it does not transfer to another
server.

Withdraw at any time:

```text
/meeting withdraw
```

Withdrawal immediately prevents future frames from entering the relevant processing and removes
the Guild grant, even when no Meeting is active. It does not falsify or erase append-only audit
events that already occurred.

Each user can manage a Guild-scoped pronunciation used only for Meeting TTS:

```text
/meeting pronunciation set alias:<spoken name>
/meeting pronunciation show
/meeting pronunciation clear
```

This never changes Discord identity, display name, alphabetical ordering, public status, transcript
speaker labels, or minutes facts. Text fallback and the persisted canonical Bot prompt retain the
Discord display name.

### 2.5 Start the Meeting

The host enters the target ordinary voice channel and runs:

```text
/meeting start
```

The Bot then:

1. validates host authority, Meeting state, agenda, permissions, and pending imports;
2. synchronises current non-Bot voice members;
3. joins the host's channel and confirms inbound DAVE readiness;
4. starts per-user recording, realtime ASR, and the Meeting voice assistant;
5. checks current participant consent;
6. announces that the Meeting starts in ten seconds;
7. automatically announces agenda count, titles, language, and duration, then enters item one.

If anyone is missing consent, state remains `CONSENT_CHECK`. It recalculates and advances when all
present participants consent or an unconsented non-participant leaves.

### 2.6 Automatic stages and host controls

Timed stages advance automatically on expiry. An untimed stage announces that the host must advance
manually. Only a stage whose original duration is greater than 120 seconds receives one reminder at
about 60 seconds remaining; openings, short stages, and votes do not produce dense countdown speech.

Host commands:

```text
/meeting status
/meeting pause
/meeting resume
/meeting next
/meeting back
/meeting skip
/meeting stop
```

The persistent host panel also exposes Pause, Resume, Next, and Stop. `back` appends a new reverse
event rather than deleting history. An LLM cannot change state directly; every operation still
passes deterministic Controller and host-permission guards.

### 2.7 Speaking queue

Participants request a turn:

```text
/meeting speak
```

Duplicate requests are idempotent. The host advances the queue:

```text
/meeting speaker next
```

The active speaker or host finishes the turn:

```text
/meeting speaker finish
```

The host can remove a waiting participant:

```text
/meeting speaker remove member:<member>
```

### 2.8 Agenda-linked private voting

When discussion ends for an item with `requires_vote=true`:

1. the Bot announces ballot text and valid duration;
2. it snapshots every non-Bot participant currently in the voice channel;
3. it sends each eligible voter a DM with Yes/No buttons;
4. the voter may click again before the deadline to change the choice;
5. all-voted completion or the deadline closes the ballot; missing responses become abstentions;
6. the Bot announces Yes, No, and abstention counts plus Yes/No/Tie outcome;
7. the next agenda item begins after about five seconds.

If DMs or buttons are unavailable, run inside the Meeting Guild:

```text
/meeting vote cast option:<yes|no>
```

The fallback enforces the same voter snapshot. Late joiners are not eligible; snapshotted voters
remain eligible after leaving voice. Minutes expose only aggregates and outcome, never the
user-to-choice mapping.

### 2.9 Independent public voting

The host can also open a standalone public ballot:

```text
/meeting vote open title:<question> duration_seconds:<10-3600>
/meeting vote status
/meeting vote close
```

Its public card has Yes, No, and Abstain buttons. This is separate from agenda-linked private
Yes/No DMs.
While a standalone public ballot remains open, `next`, `back`, and `skip` reject workflow changes.
The host must run `/meeting vote close` first or wait for that ballot to expire.

### 2.10 Real-time voice Q&A during a Meeting

Wake phrases are isolated per Discord Guild. From any channel in that Guild, run:

```text
/wakeup show
/wakeup set phrase:<4-16 consecutive Han characters>
/wakeup reset
```

Every Guild member may use `show`; `set` and `reset` require Manage Guild. The default is
“呼叫指挥中心”. Reset deletes only that Guild's override. A setting is normalized and preflighted
against the pinned local KWS pronunciation/token vocabulary before persistence, then shared by the
Guild's Meeting and resident assistants. An active session switches without a Bot restart and no
other Guild is affected. Responses are ephemeral and operational logs omit the phrase value.

Any non-Bot channel member may then say:

```text
<this Guild's current wake phrase>，<question>
```

The phrase may occur at the start or inside an utterance; ASR text confirmation allows a restricted
fuzzy syllable distance for unclear speech. A caller may also say only the current phrase and
provide the question within four seconds. After acoustic detection, Lingxu waits for that caller to
finish, performs ASR, asks
local `qwen3.5:9b`, and speaks at most two sentences with an approximately 15-second TTS hard limit.

- Only one turn exists at a time.
- New wakes during model processing or Bot playback are dropped, never queued.
- Human speech does not interrupt the Bot; a new wake is accepted only after playback finishes.
- This immediate Q&A route needs no consent because PCM, question, and answer are not persisted.
- Formal recording, transcript, and minutes still enforce their own consent.
- The model has no tools, vote, timer, Discord, or Meeting-state mutation authority.

### 2.11 Finish and receive artifacts

The final agenda item enters closing automatically, or the host runs:

```text
/meeting stop
```

The Bot announces closure and that the summary will be sent to the voice channel's text chat. It
then stops receive, drains ASR, finalises encrypted Segments, disconnects, and creates an independent
Worker job. The current normal publication contract sends exactly one Discord attachment:

- `minutes.md`: human-readable minutes containing the complete readable canonical transcript.

The Worker still registers local `minutes.json`, `transcript.txt`, canonical transcript JSON, event,
and `evidence.json` artifacts for consistency, recovery, and audit. They are not normal Discord
attachments under the current contract.

The official DeepSeek API analyzes the complete consent-filtered transcript as
compact human/system partitions with a speaker dictionary and agenda/aggregate-vote context.
It returns structured claims and evidence IDs; the Worker validates them and composes `minutes.md`
locally. Unedited source rows are preserved, and full authority tables never depend on the model
copying them. This reduces formatting burden, not the need for factual review.
The speaking table is derived only from append-only events and records
planned/extended duration, actual floor occupancy after Meeting/recovery/intervention pauses, early
finish, skip, whole or partial absence, overtime, and configured supplement/Q&A participation.
Confirmed next-agenda and stop actions close the current floor in the event log before transition;
generic back/skip-state commands are rejected during a controlled agenda. Identities outside
the `MINUTES`-consented inventory lose both Discord ID and display name while their process fact is
retained. Local validation enforces the template, exact speaker inventory, agenda, speaking, and
vote tables row-for-row; generated text cannot add, remove, reorder, or alter authoritative facts or
Meeting state. A
provider, timeout, or output-contract failure now produces explicitly labelled basic minutes:
authoritative agenda/vote/speaking facts, attributed source excerpts, and the full consented record.
Unverified decisions, consensus, and assignments require human review. Rejected model prose is
discarded; the failed attempt remains in the audit journal and is not automatically repeated.
Basic minutes pass the same privacy, evidence and publication checks and are reused after restart.
Storage/input corruption or Discord delivery problems can still prevent delivery. After resolving
such failures, the host may retry a failed Job with `/meeting retry`; completed publications remain
immutable, including successfully delivered basic minutes.

If the post-meeting job fails, the host runs:

```text
/meeting retry
```

Delete the latest Meeting and its data:

```text
/meeting delete
```

Deletion removes the database graph and exact Session directory. It is not recoverable without an
operator backup.

## 3. Resident voice command centre

Resident mode runs no Meeting behavior. A member with Manage Guild first enters the target voice
channel and runs:

```text
/assistant join
```

Any non-Bot member can then ask with this Guild's current phrase; `/wakeup show` displays it. Turn
behavior matches Meeting Q&A, but the model receives only the current question and a fixed mode
marker—never Meeting state, channel messages, member lists, or transcripts. Meeting and resident
mode always share the same Guild setting.

```text
/assistant status
/assistant stop
```

- `status` privately shows the content-safe session and channel state.
- The creator or a Manage Guild member can stop it.
- Moving or removing the Bot stops the current resident session; it does not follow to a new room.
- A normal process restart recovers from Guild/channel/creator/lifecycle metadata only.
- It creates no recording, transcript, minutes, agenda, vote, or Worker job.
- PCM, questions, answers, and model output stay in memory; dynamic WAV exists only in tmpfs and is
  deleted after playback.

## 4. Local host console

When enabled, open on the Bot host:

```text
http://127.0.0.1:8765
```

The console exposes one Guild's live state, agenda, participant consent, speaking queue, timers,
votes, events, manual TTS, and scenario simulation. Reveal the token only in the local terminal with
`docker compose exec -T lingxu-bot python -m lingxu.console.token`. Never publish the token or bind
the console to a public interface.

## 5. Troubleshooting

### Slash commands do not appear

Verify `DISCORD_APPLICATION_ID`, confirm installation includes the `applications.commands` scope,
then look for `application_commands_synced` and `bot_ready` in logs. Production multi-Guild
deployment leaves `DISCORD_TEST_GUILD_ID` blank and should report `scope=global`; setting a Guild ID
intentionally limits commands to that development Guild. Refresh the Discord client and allow a
short interval after global sync.

### “Service capacity is full” appears

This workstation is serving another Discord Guild through a Meeting/post-publication run, resident
assistant, or agenda-model import. Retry after it completes or its host stops it. The message
deliberately does not reveal the owning Guild or user. Lightweight Meeting setup and status remain
available; heavy entry points such as `/meeting start` atomically compete for the one slot.

### The Bot cannot join voice

The caller must be in an ordinary voice channel; the Bot needs View Channel, Connect, and Speak;
no other `/meeting` or `/assistant` mode may own this Guild's voice connection; and the
local deployment slot must not be owned by another Guild.

### The Meeting remains in consent_check

Every present non-Bot participant must run `/meeting consent` in this Guild, or a non-participating
unconsented member must leave. Consent is Guild-scoped; a grant from another server does not apply.

### Wake receives no answer

Run `/wakeup show` first to confirm this Guild's current phrase. Do not call during the Bot's own
playback or its short post-playback quarantine, and say the complete current phrase; an in-sentence
occurrence is allowed, but a fragment is deliberately insufficient. Check content-safe aggregate
counters and Doctor results for KWS, ASR, and the model—never upload user audio for routine
diagnostics. `/wakeup set` rejects a pronunciation absent from the pinned token vocabulary;
`/wakeup reset` restores the default.

### Minutes were not published

Check that the Meeting entered post-processing, Worker is healthy, and whether a job failed. The
host may run `/meeting retry`. Artifacts are published to the actual Meeting voice channel's
built-in text chat, not an arbitrary channel.

See [OPERATIONS.md](OPERATIONS.md), [FAILURE_RECOVERY.md](FAILURE_RECOVERY.md), and
[TEST_REPORT.md](TEST_REPORT.md) for deeper recovery and validation evidence.
