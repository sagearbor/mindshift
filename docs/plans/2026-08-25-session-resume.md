# Session resume — a live session (or call) survives a network drop

2026-08-25 · `feat/session-resume` · server/session_resume.py

Real phone calls drop WiFi/cellular for ten to thirty seconds. Everything
about a `/ws/session/{id}` session was per-CONNECTION, so a drop cost three
things — and none of them announced itself:

1. **Turns.** The fast loop keeps finalizing turns while the socket is down.
   They were kept only for the end-of-session POST, so the cloud coach — and,
   in a call, every other member's screen — silently missed whole turns.
2. **The clock.** `PcmRingBuffer`'s t = 0 is the first audio frame *of the
   connection*, advancing by audio received; `calls.Participant.offset_s` is
   re-fixed on rebind (#167). The phone's capture clock, meanwhile, never
   stopped. After a drop every `turn_local` time addressed audio the ring
   buffer did not hold: tone/identity enrichment quietly stopped producing
   anything, and in a call the resumed member's turns landed *before* turns
   already merged.
3. **Context.** Nothing replayed the merged call transcript the reconnecting
   phone missed, so its screen had a hole in the conversation.

## The protocol

On reconnect the client re-authenticates with a `config` and then sends,
BEFORE any audio and before `call_join`:

```json
{"type": "resume", "session_id": "...", "since_seq": 4, "last_local_time": 28.7}
```

* `since_seq` — the highest merged-call `seq` this client has rendered
  (0 = nothing). Solo sessions always send 0: the server keeps no transcript
  for one.
* `last_local_time` — the phone's capture-clock time of the NEXT audio byte it
  will send: captured-so-far minus what is still queued on the phone. The
  server re-anchors its session timeline (`PcmRingBuffer.rebase` + the
  Deepgram `transcriber_offset_s`) to it. The audio lost during the outage
  becomes a **hole**, never a shift — a slice over it comes back short or
  empty and every consumer already degrades to "no enrichment" rather than a
  confident wrong answer. Re-anchoring BACKWARDS is refused.

Answered with:

```json
{"type": "resume_ack", "session_id": "...", "resumed": true,
 "since_seq": 4, "last_local_time": 28.7, "known_turns": 6}
```

`resumed: false` means the server had no memory of this session (a restart, a
TTL sweep, a different process) — said plainly rather than left to guess. The
clock is still re-anchored and de-duplication still starts from here.

In a call, the `call_join` that follows is answered with the replay:

```json
{"type": "resume_replay", "call_id": "...", "since_seq": 4,
 "replayed": 2, "dropped": 0}
```

…followed by `replayed` ordinary `transcript` frames carrying `"replay": true`
(the only difference from a live one — a client that never resumes sees
exactly the pre-resume wire). Bounded by `RESUME_REPLAY_MAX` (50), oldest
dropped, count reported. **Replay renders; it does not re-coach**: a turn from
twenty seconds ago is history, and coaching it would spend tokens answering a
question the conversation has moved past.

## Exactly once

Every `turn_local` now carries a client-generated `turn_uid` (`FastLoop`: a
random per-session prefix + a counter, stamped once when the turn is
finalized, so a buffered turn keeps its id). The phone queues the turns it
cannot send (bounded, oldest dropped with a logged count and a diagnostics
line) and flushes them in order after `resume` — and after `call_join`, or a
call member's turns would be treated as a solo session's and never merged.

The server remembers processed ids **across connections**
(`session_resume.registry`, process-local like `calls.registry` and the watch
relay, TTL-swept and bounded) and ignores a repeat entirely: no coaching, no
enrichment, no merge, no delivery. `calls.Call.push_turn` de-duplicates
independently by the same id. A client that sends no `turn_uid` behaves
exactly as before — no de-duplication, and never a swallowed turn.

The session-id → state map is keyed by uid: a session id is client-chosen, so
a second ACCOUNT picking the same one gets an isolated state it can neither
read nor clobber. A graceful `stop` forgets the state at once.

## What this deliberately does not do

* **No server-side transcript.** A solo session's words are still never stored
  server-side (see `audio_pipeline._remember_utterance`), so a solo resume has
  nothing to replay — it is the clock and the de-duplication. The phone holds
  that transcript. Replaying it would pre-empt a flagged human/product
  decision.
* **No cross-process resume.** Like the calls registry and the watch relay,
  this is process-local; production runs `--max-instances 1`. A multi-instance
  deployment needs a shared store (flagged, later).

## Tests

* `server/tests/test_session_resume.py` — the pure state (de-duplication,
  bounds, TTL, per-account isolation, "a second connection is what makes a
  resume findable"), frame validation (`turn_uid` shape, `since_seq`,
  `last_local_time` rejecting inf/NaN/negative), `PcmRingBuffer.rebase`
  arithmetic and its refusal to go backwards, `Call.turns_since` /
  `push_turn` de-duplication / a resumed `bind` keeping the clock offset, and
  the wire: a solo session that drops and resumes coaches each turn exactly
  once; the re-anchor is what lets enrichment still recover a turn's audio
  after a drop (with the pre-fix behaviour pinned as the negative case); a
  call member that drops gets its missed turns replayed, un-coached, and its
  buffered turns merged.
* `apps/mobile/__tests__/useAudioStreamResume.test.tsx` — the real `FastLoop`
  through the hook, with the socket killed mid-conversation (sends throw
  first, the way a real drop does): across the drop every turn reaches the
  server in order, exactly once, with the resume frame carrying the phone's
  capture clock; the queue bound drops the oldest and says so; `since_seq`
  tracks the merged transcript across two drops.
* `server/tests/test_live_e2e_inprocess.py::test_live_e2e_inprocess_survives_a_mid_session_disconnect`
  and `scripts/live_e2e.py --drop-after-turns N [--drop-seconds S]` — the full
  phone-shaped walk against a real uvicorn with the transport aborted after
  four turns and eight seconds of audio lost: 13/13 turns reported, nothing
  coached twice, the clock re-anchored to exactly what was received plus what
  was not, and the stored episode identical to a clean run's.
