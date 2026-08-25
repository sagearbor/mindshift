# In-app calls — server contract (2026-08-25)

> **Why.** Android (10+) and iOS give third-party apps silence while the phone
> is on a cellular/VoIP call, so "phone Mom and get coached on the same Pixel"
> cannot work — unless MindShift **is** the call. Then the app owns the mic,
> each member's audio is its own stream, and every side can be coached.
>
> **Shape (owner-approved, extended tonight to three members with roles).**
> Audio is peer-to-peer over WebRTC (full mesh; the server only relays
> signaling). Each member transcribes THEMSELVES on-device (phone fast loop or
> Safari Web Speech) and sends `turn_local` exactly as a solo session does. The
> server merges every member's turns into ONE shared **call session**, pushes
> each turn to the other sockets as a `transcript` event, coaches each
> *participant* with the merged context, gives the *therapist* observer a
> read-only copy of that coaching, and at the end persists one episode per
> participant (mode `"call"`). Attribution is structural — member = speaker —
> so the remote voices never need server STT or speaker-ID.
>
> Server half: PR `feat/calls-server` (this document). Client half: the sibling
> client agent appends below the line at the bottom.

Files: `server/calls.py` (model + process-local registry + relay + merge +
persist), `server/routers/calls.py` (REST), `server/audio_pipeline.py` (the
`call_join` / `rtc_signal` frames, per-session `CallEndpoint`, role behaviour),
`server/routers/sessions.py::ingest_live` (the shared ingest the call end
uses), `server/models/audio.py::ObserverTagged` (`for_uid`), tests
`server/tests/test_calls.py`, `server/tests/test_live_e2e_inprocess.py::test_call_e2e_inprocess`,
`scripts/live_e2e.py --call`.

## Roles, slots, labels

| role | how many | slot / label | coached? | gets |
|---|---|---|---|---|
| `participant` (the host is always one) | ≤ 2 | host `A` → `"Speaker A"`, second `B` → `"Speaker B"` | yes — nudges on own turns, suggestions about **both** other people's turns | `transcript` of every other member's turns; own `suggestion`s; own episode at end |
| `therapist` (observer — Mom in Safari) | ≤ 1 | `C` → `"Speaker C"` | **never** (her own turns are transcribed + merged, nothing is generated for her) | `transcript` of everyone; every participant's `suggestion` / `tone_flag` / `speaker_identity` as a read-only copy tagged `for_uid`; no episode of her own but a share grant of each participant's |

`max_participants` on `POST /calls` is the TOTAL (2 or 3, default 3). Seats
are per role: a third `participant` → 409 `call already has two participants`;
a second `therapist` → 409 `call already has a therapist`; anything past the
cap → 409 `call is full`. Roles are fixed at join; re-joining is idempotent.

**Every turn is relabelled by slot, whatever the phone's diarizer said.** In a
call the phone hears only its owner, so `turn_local.speaker` is ignored and
`is_self` forced `true` — the existing "Speaker X" machinery (side-aware
coaching, the episode label ladder, mid-call naming) works unchanged. The
client learns its own label from `call_state.self_label` and should show its
own turns under it.

**Names (relative to each viewer), in precedence order:** the viewer's own
`speaker_label` naming of that member's slot label (persisted call-wide) →
the member's self-declared `display_name` (`POST /calls`, `/join`,
`call_join`) → the member's account email (resolved best-effort; the same
thing the therapist link shows) → the slot label. A therapist's name carries
the ` (therapist)` suffix for everyone but herself ("Mom (therapist)");
yourself is always `"You"`. `/voice/people` entries are voiceprints, not
accounts, so they cannot be mapped to a member automatically — naming the
person mid-call (`speaker_label` with a `person_id`) is how to attach one.

## REST (`/calls`, Bearer auth like every other route)

### `POST /calls` → 201 `CallOut`
```json
{ "invitee_email": "mom@example.com",   // optional; resolved to an account (404 none, 400 yourself)
  "invitee_uid": null,                    // optional alternative
  "display_name": "Sage",                 // optional, how the others see the host
  "ttl_minutes": 180,                     // optional; how long an OPEN call stays joinable (≤ 1440)
  "max_participants": 3 }                 // optional; 2 or 3 (default 3)
```
Without an invitee the call is **open**: anyone with the join code may join.

### `POST /calls/join` → 200 `CallOut` (join by code — the invite link carries only the code)
```json
{ "join_code": "VZBJBY", "display_name": "Mom", "role": "therapist" }   // role optional, default "participant"
```

### `POST /calls/{call_id}/join` → 200 `CallOut`
```json
{ "join_code": "VZBJBY", "display_name": "Dad", "role": "participant" }
```
The named invitee needs no code; anyone else must send it (403 `join code
does not match`). Codes are 6 chars from `ABCDEFGHJKLMNPQRSTUVWXYZ23456789`
(no 0/O/1/I), accepted in any case, with spaces or dashes.

### `GET /calls/{call_id}` → 200 `CallOut`
Members and the invitee only; anyone else gets 404. After the call ended it
still answers (for `CALL_RETENTION_MINUTES`, default 60) so a reconnecting
phone can read its `episode_id`.

### `POST /calls/{call_id}/end` → 200 `CallOut`
Any member. Persists the episodes, sends `call_ended` to every bound socket,
detaches them (each session keeps coaching solo). Idempotent.

### `CallOut` (the same body as the WS `call_state`, plus the join code/url)
```json
{ "call_id": "68da4269-…", "status": "open" | "active" | "ended",
  "host_uid": "uid-a", "max_participants": 3,
  "self_uid": "uid-b", "self_role": "participant" | "therapist" | null,
  "self_label": "Speaker B", "peer_label": "Speaker A",      // the OTHER participant's label (fixed by slot even before they join)
  "therapist_label": "Speaker C", "therapist_uid": "uid-c" | null,
  "participants": [
    { "uid": "uid-a", "slot": "A", "label": "Speaker A", "role": "participant",
      "display_name": "Sage", "is_self": false, "connected": true, "joined_at": "…" },
    { "uid": "uid-b", "slot": "B", "label": "Speaker B", "role": "participant",
      "display_name": "You", "is_self": true, "connected": true, "joined_at": "…" },
    { "uid": "uid-c", "slot": "C", "label": "Speaker C", "role": "therapist",
      "display_name": "Mom (therapist)", "is_self": false, "connected": false, "joined_at": "…" } ],
  "invitee": { "uid": "uid-b", "email": "dad@example.com" } | null,
  "ice_servers": [ { "urls": ["stun:stun.l.google.com:19302"] },
                   { "urls": ["turn:…"], "username": "…", "credential": "…" } ],   // TURN only when configured
  "join_code": "VZBJBY", "join_url": "https://arborfam-hub.web.app/call/VZBJBY",
  "invitee_uid": "uid-b", "invitee_email": "dad@example.com",
  "created_at": "…", "expires_at": "…", "started_at": "…" | null, "ended_at": "…" | null,
  "end_reason": "ended" | "all participants left" | "expired" | null,
  "turn_count": 13,
  "episode_id": "56b5ca5d-…" | null,          // YOUR episode once ended (always null for the therapist)
  "shared_with": ["mom@example.com"] }        // who your episode was granted to
```
`display_name` and `is_self` are relative to the caller. `status` is `open`
until a second member joins, `active` after, `ended` at the end. An open call
expires after `ttl_minutes` (410 on join, `end_reason: "expired"`); an active
call never expires by clock — it ends when its last socket leaves.

Errors: 404 unknown/foreign call or unknown invitee email · 400 calling
yourself · 403 wrong code / non-member ending · 409 seat taken / full · 410
ended or expired · 422 malformed (`role`, `max_participants`, email).

ICE: `stun:stun.l.google.com:19302` always; add TURN with
`MINDSHIFT_TURN_URLS` (comma-separated), `MINDSHIFT_TURN_USERNAME`,
`MINDSHIFT_TURN_CREDENTIAL` (read per request — no restart). Without TURN two
phones on carrier NAT may fail to connect; the client should say so.
`MINDSHIFT_CALL_JOIN_BASE` overrides the `join_url` base (default the web
app, so Mom can join from Safari; `mindshift://call` for a deep link).

## WebSocket (`/ws/session/{session_id}` — the existing socket, existing auth)

Every member opens its OWN session socket (normal `config` handshake with
`id_token`, `tts: "on-device"`, etc.) and binds it to the call. Client →
server frames:

```json
{ "type": "call_join", "call_id": "68da4269-…",
  "join_code": "VZBJBY",          // optional — lets a non-member join here without the REST call
  "display_name": "Sage",         // optional — self-declared name
  "role": "therapist" }           // optional — only used when joining here; default participant
```
→ answered with `call_state` (below) or `{"error": "call_join: <reason>"}`
(`invalid call_id`, `no such call`, `join code does not match`, `call is
full`, `call already has a therapist`, `call has ended`, …). A second socket
for the same uid replaces the first (reconnect).

```json
{ "type": "rtc_signal", "call_id": "68da4269-…", "to": "uid-c",
  "payload": { "type": "offer", "sdp": "v=0 …" } }          // or {"candidate": …, "sdpMid": …} — relayed VERBATIM
```
→ delivered to the addressed member as
`{"type": "rtc_signal", "call_id", "from": "<sender uid>", "payload": <verbatim>}`.
**`to` is required as soon as the call has more than two members** (full
mesh: every client holds one RTCPeerConnection per other member and must
address each offer/answer/candidate); in a two-member call it may be omitted
and means "the other one". Errors: `rtc_signal: not in that call`, `'to' is
required in a call with more than two members`, `peer has not joined`, `peer
not connected` (wait for `call_state` to show `connected: true`, then
(re)offer), `payload must be a non-empty object`, `payload too large`
(64 KiB). Nothing is buffered server-side.

`turn_local` — unchanged shape. In a call: relabelled to your slot label,
`is_self: true`, appended to the shared transcript, pushed to every other
socket, and (participants only) coached as your own turn (nudge). Timeline:
your `start_time`/`end_time` stay on YOUR capture clock; the server keeps
them (`local_*`) and re-bases a copy onto the call clock by a per-member
offset fixed at your first turn.

`speaker_label` — unchanged shape, now call-wide: naming another member's
slot label (`"Speaker C"` → `"Linda"`) is YOUR naming of that member
(persisted on the call, used for your transcript/prompt/episode, broadcast
in `call_state`); a real name on your OWN label (not "You"/"me") is a
self-declared name the others see. `is_self` claims are ignored in a call
(attribution is structural).

`stop` — hangs up your side. If others are still connected the call goes on
(`call_state` shows you `connected: false`); when the last socket leaves the
call ends, episodes are persisted, and that socket receives `call_ended`
BEFORE its `session_complete`. `session_complete` carries
`"call": {"call_id", "status", "episode_id"}`.

Server → client frames (additive; a solo session never sees them):

```json
{ "type": "call_state", …CallOut minus join_code/join_url/invitee_uid/invitee_email… }
```
on every bind, leave, join, name change, and end.

```json
{ "type": "transcript", "session_id": "<your session>",
  "speaker": "Speaker A", "display_name": "Sage", "role": "participant",
  "text": "You never call me back.", "start_time": 3.0, "end_time": 4.5,      // call clock
  "local_start_time": 3.0, "local_end_time": 4.5,                            // sender's clock
  "call_id": "68da4269-…", "participant_uid": "uid-a", "is_self": false, "seq": 1,
  "text_tone": { "warmth": null, …, "frustration": 70, "label": "angry" } | null,   // sender's on-device measurements
  "prosody": { "rms_dbfs": -12.0, "pitch_hz": null, "speech_rate": null } | null }
```
a turn another member's phone finalized. Your own turns are never echoed
(you rendered them locally). A participant gets a `suggestion`
(`kind: "response"`, `speaker` = that label) for it; the therapist just sees
it. `text_tone`/`prosody` let an observer run the scoreboard over everyone.

```json
{ "type": "suggestion", …, "for_uid": "uid-a" }        // THERAPIST sockets only
{ "type": "tone_flag", …, "for_uid": "uid-a" }
{ "type": "speaker_identity", …, "for_uid": "uid-a" }
```
read-only copies of each participant's coaching events (finals AND partial
previews; nudges too). `for_uid` is absent on your own events, so a
participant's wire stays byte-identical to a solo session.

```json
{ "type": "call_ended", "call_id": "68da4269-…", "reason": "ended" | "all participants left",
  "ended_by": "uid-b" | null, "episode_id": "56b5ca5d-…" | null, "recording_id": "<same>",
  "shared_with": ["mom@example.com"], "episodes": { "uid-a": "56b5…", "uid-b": "205f…" }, "turn_count": 13 }
```
`episode_id` is YOURS (null for the therapist, who gets `episodes` — every
participant's — and a share grant of each). After this the session is solo
again (no relay, still coached).

## What the server stores at the end

One episode per **participant** through `POST /sessions/live`'s ingest
(`routers.sessions.ingest_live`), never for the therapist:

* `session_id = "call-<call_id>"` (so the recording id is
  `live_recording_id(uid, "call-<call_id>")` — distinct per participant),
  `mode: "call"`, `source.type: "live"`, `media_type: "none"`, title `Call
  with Dad and Mom (therapist)`;
* `turns`: the FULL merged transcript in arrival order, every row carrying
  `call_seq`, `participant_uid`, `local_start_time`, `local_end_time` next to
  the call-clock `start_time`/`end_time`; own turns `is_self: true` +
  `speaker_person_id: "self"`, everyone else's `is_self: false` (the other
  phones' own "self" verdicts never leak in); the sender's `text_tone`/
  `prosody` preserved;
* `analysis.live.self_speaker` = your slot label; manual speaker labels
  `{"Speaker A": "You", "Speaker B": "Dad", "Speaker C": "Mom (therapist)"}`
  (+ `manual_speaker_people: {"Speaker A": "self"}`), so Replay, Growth,
  the therapist dashboard and reflections work like any live session;
* the batch analysis + "what you could have said" are scheduled as usual
  (`MINDSHIFT_CALL_ANALYZE` / `MINDSHIFT_CALL_REFLECT` = `0` turns them off);
* sharing: the therapist LINK's auto-share as for any live session, **and** a
  direct per-episode grant to the therapist who was ON the call (same
  `store.add_share` Replay's "Share with…" uses) whether or not a link
  exists — she was in the room, the participant admitted her. Granted once
  when both apply. `shared_with` on `call_ended` / `GET /calls/{id}` lists
  the emails.

**The phone must NOT POST /sessions/live for a call session** — it would
store its own half a second time.

Caps: 400 merged turns / 60 000 transcript chars per episode (oldest
dropped), the same as ingest. Registry: process-local like the watch relay;
production runs Cloud Run `--max-instances 1` so every socket of a call
lands on the same process. `MINDSHIFT_CALL_TTL_MINUTES` (180),
`MINDSHIFT_CALL_RETENTION_MINUTES` (60), `MINDSHIFT_MAX_CALLS` (500).

## Server-STT fallback

A member whose phone has no on-device STT can stream PCM as before: the
server's transcriber segments are structurally THAT member's turns
(relabelled, `is_self`, merged, `transcript_source: "cloud"`), coached and
relayed the same way (not coached for the therapist). Voiceprint identity
enrichment is skipped in a call (the speaker is known); audio tone
enrichment still runs on your own audio.

## Tests

* `server/tests/test_calls.py` — REST (create/join/code/seats/caps/expiry/end),
  signaling (2-way relay, `to`-required mesh, errors), merged transcript
  (slot labels, tone hints, both sides coached, call-wide naming), server-STT
  fallback, ending (both stop → two episodes + link auto-share; REST end
  notifies all and solo coaching continues; abrupt disconnect → peer told,
  last leaver ends; reconnect replaces the socket; no store; analysis +
  reflection scheduled), three-way (seats, mesh relay, read-only fan-out
  tagged `for_uid`, therapist never coached, per-viewer names, two
  participant episodes shared to the in-call therapist, link + in-call
  therapist granted once), pure model helpers. The sockets of one test share
  one event loop (a portal attached to the TestClient) as in production.
* `server/tests/test_live_e2e_inprocess.py::test_call_e2e_inprocess` and
  `scripts/live_e2e.py --call` — the two-phone walk against a real uvicorn /
  a deployed server: link → `POST /calls` → join by code → both sockets
  bind + exchange a fake SDP offer/answer → both stream their own halves of
  the couple scene concurrently (each phone hears only itself) → merged
  transcript on each screen, per-participant coaching → both hang up →
  `GET /calls/{id}` ended with one episode each → the patient's episode
  (analysis, reflection, detail with scene-order turns + the expected self
  escalations, growth) → the therapist's `GET /sessions` lists the patient's
  episode (shared) and its own.

## Decisions recorded

* No signaling buffer: an offer to a not-yet-connected peer is an error
  frame, the client re-offers on `call_state.connected`.
* Slot labels over phone labels (structural attribution beats the phone's
  diarizer; keeps `_SELF_SPEAKER_RE` and the label ladder intact).
* Two clocks kept; the call clock is arrival-anchored per member (offset
  fixed at the first turn, never negative).
* The therapist gets no episode of her own; she is granted the
  participants' (link or not).
* `for_uid` only serialized when set (`ObserverTagged` wrap serializer) so
  no pre-call payload changes.
* Voiceprint identity is skipped in calls; tone enrichment stays.
* Names: naming is per viewer; a self-declared name is the fallback; the
  therapist link adds no name beyond the email already resolved.

---

_Client half (appended by the client agent):_
