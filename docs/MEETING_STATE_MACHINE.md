# Fixed Meeting State Machine

Product Stage 1 uses one deterministic forward sequence:

```text
CREATED
-> AWAITING_PARTICIPANTS
-> CONSENT_CHECK
-> OPENING
-> AGENDA_OVERVIEW
-> AGENDA_ITEM_INTRO
-> DISCUSSION_READY
-> SPEAKER_ACTIVE
-> DISCUSSION_COMPLETE
-> VOTE_OPEN
-> VOTE_RESULT
-> NEXT_AGENDA_ITEM
-> CLOSING
-> COMPLETED
-> POST_PROCESSING
-> PUBLISHED
```

Global states are `PAUSED`, `STOPPED`, `INTERRUPTED`, the four explicit failure states, and
`DELETED`.

Only `MeetingRuntime` chooses a target state. Slash commands, buttons, ASR rule intents, TTS,
Worker jobs, and future LLM candidates submit commands; they never write state directly.

## Commands and events

| Command | Guard | Event | Effect |
| --- | --- | --- | --- |
| initialise | host/system; state `CREATED` | `MEETING_INITIALISED` | Enter participant wait |
| start | host; state `AWAITING_PARTICIPANTS` | `MEETING_STARTED` | Enter consent check; if all present participants already have this Guild's three grants, immediately and idempotently enter `OPENING`; repeated start is a no-op |
| consent-ready | system; state `CONSENT_CHECK`; every present participant fully consented | `STATE_ADVANCED` | Enter `OPENING`; reevaluated after grant, join, leave, or start |
| advance / next | host/system; fixed next exists | `STATE_ADVANCED` | Follow one fixed edge |
| skip | host; fixed next exists | `STATE_SKIPPED` | Follow the same edge with audit meaning |
| back | host; fixed prior exists | `MEETING_BACK` | Append a reverse transition; never delete history |
| pause | host; active state | `MEETING_PAUSED` | Persist resume state and emit `freeze_timers` |
| resume | host; paused/interrupted | `MEETING_RESUMED` | Restore exact persisted state and emit `resume_timers` |
| stop | host; nonterminal | `MEETING_STOPPED` | Enter terminal stop; repeated stop is a no-op |

Control sources `SLASH_COMMAND`, `BUTTON`, and `VOICE_RULE` must identify the Session creator.
System progression may use a null actor. Invalid skips, terminal transitions, wrong resume targets,
and non-host control fail closed.

## Persistence and recovery

Every actual state change and its event share one database transaction. Sequence numbers are
allocated atomically. Every configured interval, the Runtime writes a state snapshot at the exact
event sequence.

Recovery replays the immutable event chain from `CREATED` and verifies:

- each event's `state_before` equals the previous `state_after`;
- the final event sequence equals `meeting_sessions.next_sequence`;
- the latest snapshot state equals replay at its sequence;
- replay final state equals the materialised Session row.

Any mismatch raises `MeetingRecoveryError`; recovery never guesses or silently overwrites history.
