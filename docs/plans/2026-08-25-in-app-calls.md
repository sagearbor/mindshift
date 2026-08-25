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
A coached participant (403 for the therapist — an observer may leave, not
hang up for everyone — and for the never-joined invitee). Persists the
episodes, sends `call_ended` to every bound socket, detaches them (each
session keeps coaching solo). Idempotent.

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
until a second member joins, `active` after, `ended` at the end. A call with
no socket bound (open, or active only through REST joins nobody connected
to) expires after `ttl_minutes` (410 on join, `end_reason: "expired"`); a
call with a live socket never expires by clock — it ends when its last
socket leaves.

Errors: 404 unknown/foreign call or unknown invitee email · 400 calling
yourself · 403 wrong code / therapist or non-member ending · 409 seat taken /
full · 410 ended or expired · 422 malformed (`role`, `max_participants`,
email) · 429 the host already has `MAX_OPEN_CALLS_PER_HOST` (3) un-ended
calls · 503 the process holds `MAX_CALLS` live calls.

Brute-force guards on the 6-char code (32^6 ≈ 1e9): REST is IP-rate-limited;
a wrong code is refused BEFORE the account-email lookup; after 50 wrong
codes on one call (`JOIN_CODE_FAILURES_MAX`) the code is burned (the named
invitee never needed it; the host starts a new call).

ICE: `stun:stun.l.google.com:19302` always; add TURN with `MINDSHIFT_TURN_URLS`
(comma-separated). Two ways to authenticate to that relay, read per request —
no restart:

* **Ephemeral (preferred)** — `MINDSHIFT_TURN_SECRET` (+ optional
  `MINDSHIFT_TURN_REALM`, `MINDSHIFT_TURN_TTL_SECONDS`, default 4h): the
  server mints a standard TURN REST credential
  (draft-uberti-behave-turn-rest-00, coturn's `use-auth-secret`) **per member,
  per handout** — username `"<unix-expiry>:<uid>"`, credential
  `base64(HMAC-SHA1(secret, username))`. Nobody shares a password and a leaked
  credential expires. Takes precedence when both are configured.
* **Static** — `MINDSHIFT_TURN_USERNAME` / `MINDSHIFT_TURN_CREDENTIAL`: one
  password every client holds, forever. Still supported because some vendors
  (Cloudflare, Twilio, Metered) only issue API-minted credentials.

`GET /calls/ice` returns the same list WITHOUT creating a call (plus
`turn_configured`, `turn_credential_mode`, `ttl_seconds`) so the client can run
a connectivity pre-flight before the demo:
`apps/mobile/src/live/call/iceProbe.ts` gathers candidates against it and the
Call pre-flight panel shows one honest line — *relay ready* / *direct likely,
no TURN* / *relay needed — no TURN configured* / *TURN configured but no relay
candidate*. Vendor comparison, prices and click-paths:
`docs/research/turn-options-2026-08-25.md`. Without TURN two phones on carrier
NAT may fail to connect; the client says so rather than spinning.
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
full`, `call already has a therapist`, `call has ended`, …). After 8 wrong
codes on one socket (`JOIN_ATTEMPTS_MAX`) every further `call_join` on it is
`too many failed attempts` — a new socket costs a fresh token handshake. A
second socket for the same uid replaces the first (reconnect; the member's
clock offset is re-fixed at its next turn).

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
(64 KiB), `too many signals` (per-socket token bucket: a burst of 60, then
20/s — ICE gathering fits, a flood does not). Nothing is buffered server-side.

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
  "replaces_seq": null,                                                      // or 1: this row CORRECTS seq 1
  "text_tone": { "warmth": null, …, "frustration": 70, "label": "angry" } | null,   // sender's on-device measurements
  "prosody": { "rms_dbfs": -12.0, "pitch_hz": null, "speech_rate": null } | null }
```
a turn another member's phone finalized. Your own turns are never echoed
(you rendered them locally). A participant gets a `suggestion`
(`kind: "response"`, `speaker` = that label) for it; the therapist just sees
it. `text_tone`/`prosody` let an observer run the scoreboard over everyone.

**`seq` is the row's identity, and a client MUST key its call transcript by
it.** `seq` is the row's position in the shared merged transcript, and the
same `seq` can arrive twice: the second frame carries `replaces_seq` (equal
to its own `seq`) and is the *corrected* copy of a line you already
rendered — the sender's phone reporting words the server's transcriber had
merged for it first (see **Server-STT fallback** below). Swap that line in
place; never append it. `replaces_seq` is `null` on every ordinary turn. A
correction brings the sender's wording, `text_tone`, `prosody` and
`local_start_time`/`local_end_time`; it is deliberately **not** coached a
second time (same words — the first copy already produced a `suggestion`),
so no new `suggestion` follows it.

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
`episode_id` is YOURS (null for the therapist). `episodes` — every
participant's — is sent to the THERAPIST's socket only (she holds a share
grant of each); a participant never learns the other's episode id. After
this the session is solo again (no relay, still coached).

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
`MINDSHIFT_CALL_RETENTION_MINUTES` (60), `MINDSHIFT_MAX_CALLS` (500 — live
calls; retained ended ones are evicted first, never crowding a live one out),
`MINDSHIFT_MAX_OPEN_CALLS_PER_HOST` (3 un-ended calls per account, 429 beyond).

## Server-STT fallback

A member whose phone has no on-device STT can stream PCM as before: the
server's transcriber segments are structurally THAT member's turns
(relabelled, `is_self`, merged, `transcript_source: "cloud"`), coached and
relayed the same way (not coached for the therapist). Voiceprint identity
enrichment is skipped in a call (the speaker is known); audio tone
enrichment still runs on your own audio.

**Only while the phone has never sent `turn_local`.** A local-first phone
still streams PCM (for tone enrichment), and the server's transcriber can
finalize a span *before* the phone's `turn_local` for it — the pipeline's
local-range suppression only drops the segments that land *after*. Such a
segment used to be relayed and persisted as a second copy of the same words
(found by the 3-way production e2e, 2026-08-25). Now: once a member is
local-first its own cloud segments never enter the shared transcript (the
phone is the authority for its own voice; spans its VAD misses are still
surfaced and coached on its own socket, as in a solo session), and the one
remaining race — the transcriber beating the phone's *first* `turn_local`,
before the latch — is closed in `Call.push_turn`: a phone turn whose padded
range contains the midpoint of that member's recent cloud row replaces it in
place (same `seq`, the phone's text/tone/prosody).

That replacement is **re-relayed**, tagged `replaces_seq`. The persisted
episodes were always right (they are built from `call.turns` at the end),
but the other members had already *rendered* the transcriber's copy —
different wording, no `text_tone`, no sender clock — and only a second
delivery corrects the line on their screens. Because every member's FIRST
utterance is finalized before anything has latched that phone local-first,
this hit once per member per call: the 3-way production e2e of 2026-08-25
showed the host's opening line reaching both other viewers as the cloud
copy (`"merged transcript (per viewer)"` ❌, and 30/32 relay deliveries
timed because the phone's wording never arrived).

**Why the correction and not a hold-back.** The alternative considered was
queueing a member's cloud rows for ~3 s while it has sent no `turn_local`,
so the phone usually wins outright. Rejected: (a) it is a guess about a
race the phone loses *hardest* on its first turn, when its STT/LLM models
are still warming — the very case it must cover; (b) a member with no
on-device STT at all never latches local-first, so every one of its rows
would sit in the queue, adding ~3 s to the transcript and to the others'
coaching for exactly the users the cloud path exists for; (c) it needs a
timer, a flush on `turn_local`/hang-up/end, and ordering across the two
paths, against one extra frame here. The correction is deterministic — it
lands however late the phone is — and costs one frame per member per call.
`Call.push_turn` logs how far the phone trailed (`… replaces its cloud
duplicate seq N (X ms later)`), so the size of the race stays visible.

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
* `server/tests/test_live_e2e_inprocess.py::test_call_e2e_inprocess_three_way`
  and `scripts/live_e2e.py --call --participants 3` — the three-phone walk
  (a third account: `--peer-*`, or a third `--signup` throwaway): `POST
  /calls` (Dad invited) → Dad joins over REST without a code → three sockets
  bind, Mom on hers with the code as `role: "therapist"` → `call_state`
  roles/labels/relative names ("You" / "Dad" / "Sage" / "Mom (therapist)")
  → full-mesh signaling (an unaddressed `rtc_signal` is refused with three
  members; six addressed offers delivered verbatim with `from`) → host =
  Speaker A turns, Dad = Speaker B turns, Mom three short lines in the gaps,
  concurrently → merged `transcript` per viewer, coaching per participant
  only (nudges on own turns, suggestions about both others), Mom's socket
  gets every participant's `suggestion`/`tone_flag` copy tagged `for_uid`
  and never a suggestion of her own → turn_local → other-viewer transcript
  p50/p95 → hang-up host → Dad → Mom, so the participants leave an ACTIVE
  call (no `call_ended`; episode via `GET /calls/{id}`) and Mom's socket ends
  it (`call_ended` with `episodes` for exactly the two participants,
  `episode_id: null`) → both episodes shared with her directly (no link),
  both on her `GET /sessions`, none of her own → cleanup deletes both
  episodes and the accounts.

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

## Client half (PR `feat/calls-client`, 2026-08-25)

Files: `apps/mobile/src/live/call/` (`types.ts` wire types, `callSession.ts`
the mesh state machine, `callApi.ts` REST, `rtcNative.ts` react-native-webrtc
adapter, `callWeb.ts` browser adapter, `invite.ts` share sheet / Web Share /
clipboard, `rtc.ts` the adapter interface), `src/nav/callLink.ts` (invite
links), `src/components/CallPanel.tsx` (the Call-mode UI), the Call branches
in `src/hooks/useAudioStream.ts` and `src/screens/LiveCoachScreen.tsx`,
`App.tsx` (routes). Tests: `__tests__/callSession.test.ts`,
`callLink.test.ts`, `useAudioStreamCall.test.tsx`, `LiveCoachCall.test.tsx`,
`AppCallRoute.test.tsx`, `diagnostics.test.ts`.

### Modes
* **"Speaker-phone" is now "In person"** (both people in the room, one
  mic). The wire/stored value stays `speaker` so `POST /sessions/live`,
  episodes and the per-account mode pref (`modePrefs.ts`) are unchanged.
* **"Call"** is new (`LiveMode = "call"`): MindShift places the call. The
  pre-flight explainer reads *"Your phone can't listen during a normal phone
  call — MindShift places the call itself."*

### What the client does, in order
1. **Start a call** (host): `POST /calls {display_name, max_participants: 3}`
   → the `CallOut` (`self_label` "Speaker A", `ice_servers`, `join_code`,
   `join_url`). The session starts in Call mode (`session_id =
   "call-<call_id>"`, the same fast loop + WebSocket as any live session,
   `config … tts: "on-device"`), and on every socket open sends
   `call_join {call_id, join_code, display_name, role}`.
   *Share invite*: "Invite a participant" → `https://arborfam-hub.web.app/call/<code>`;
   "Invite my therapist" → the same link with `?role=therapist`. Both also
   appear as `mindshift://call/<code>[?role=therapist]` for the app.
2. **Join** (Dad, Mom): the link opens the app (`mindshift://call/…`) or the
   web app (`/call/<code>[?role=…]`) on Live Coach in Call mode with ONE
   **Answer** button; typing a code under "Join with code" does the same
   (participant). Answer → `POST /calls/join {join_code, display_name,
   role}` → session start → `call_join`. On Safari everything the tap must
   unlock (AudioContext, SpeechRecognition, getUserMedia, the remote
   `<audio autoplay playsinline>` elements) happens synchronously inside the
   Answer tap, before the REST round-trip (#152's findings).
3. **Mesh**: `call_state` is the roster. `CallSession` keeps one
   `RTCPeerConnection` per OTHER member (peer map by uid). On each link the
   **lexicographically-lower uid offers**, the other answers — symmetric,
   glare-free, and independent of who joined first. Every `rtc_signal` is
   addressed (`to` always set); `payload` is the W3C init dict verbatim
   (`{type, sdp}` / `{candidate, sdpMid, sdpMLineIndex}`). Candidates that
   beat the remote description are queued per link. The server buffers
   nothing, so the offerer re-offers when `call_state` shows a peer
   `connected: true` and the link still has no remote description. ICE
   `failed` (or `disconnected` > 4 s) → the offerer sends an `iceRestart`
   offer, up to 4 per link; the answerer just reports "reconnecting". A member
   leaving tears down only their link; a returning member gets a fresh one.
4. **Mic**: WebRTC sends the phone's mic at 16 kHz mono with echo
   cancellation ON. On the Pixel that is a SECOND `AudioRecord`
   (react-native-webrtc's) next to expo-audio's — the fast loop keeps
   consuming expo-audio's frames exactly as before, because react-native-webrtc
   exposes no PCM tap on its track; Android allows concurrent capture within
   one app. On the web the call sends the SAME `MediaStream` the fast loop
   already captures (`WebAudioCapture.mediaStream`) — one getUserMedia.
   Mute flips only the WebRTC track (the coach keeps hearing you).
5. **Remote audio**: native — react-native-webrtc renders and mixes every
   remote track itself; the output route is expo-audio's audio mode
   (`setCallAudioRoute`: speaker by default, Earpiece/Speaker toggle in the
   call panel; Android only). Web — one `<audio autoplay playsinline>`
   element per peer. The remote voices are NEVER run through STT or
   speaker-ID: the other members' turns arrive as `transcript` events.
6. **Transcript**: in a call every `transcript` event is another member's
   turn — shown under `display_name` ("Dad", "Mom (therapist)"), keyed by
   their slot `speaker` label (so tapping the chip → "Who is this?" sends
   `speaker_label {speaker: "Speaker B", …}` = call-wide naming). The phone's
   own turns render as "You", keyed by `call_state.self_label`. The sender's
   `text_tone`/`prosody` feed the local scoreboard, so an observer scores
   everyone. Relayed lines are stored by `seq` (`TranscriptEntry.callSeq`):
   a frame whose `seq`/`replaces_seq` is already on screen replaces that
   line in place — a member's first turn arrives twice when the server's
   transcriber beat that phone to it, and must show once.
7. **Coaching**: a participant's wire is byte-identical to a solo session.
   Events carrying `for_uid` (therapist sockets) render as read-only cards
   tagged *"for Dad"*, are never voiced and never counted as escalations.
   A therapist-role client runs the fast loop in `therapist` mode (STT +
   `turn_local`, no local LLM speech, any local suggestion dropped) and
   `speakSuggestion` is a no-op for her — nothing is ever spoken to the
   therapist. She always sees the scoreboard.
8. **End**: Hang up → `POST /calls/{id}/end` + the normal `stop` drain; the
   other side's `call_ended` (or the server's) stops the session too.
   **The phone does NOT `POST /sessions/live` for a call** — `call_ended`
   carries this participant's `episode_id` + `shared_with`, which land in
   `lastEpisode` (the summary card's "shared with…") and Your Day.
9. **No TURN configured** (`ice_servers` has no `turn:`) → the in-call panel says
   two phones on mobile data may not connect; Wi-Fi usually works. And BEFORE
   the call, the pre-flight panel's **Peer connection** row runs a real ICE
   gathering pass (`iceProbe.ts` against `GET /calls/ice`) and reads
   *relay needed — no TURN configured* — so this is found at the kitchen table,
   not mid-demo.

### Send diagnostics (bonus item)
Settings → *Diagnostics* → **Send diagnostics** (and automatically when a
session ended with errors: mic/STT/transcription failures, WS reconnects, a
failed `POST /sessions/live`, a failed call or ICE restarts) POSTs one
`client_diagnostics` event to the existing `/telemetry` with a structured
`data` payload (additive `TelemetryEvent.data`): the capability probe, the
last session's latency summary (median/p90 segment-end→speak, per-provider
counts, held), STT restarts + failure, WS reconnects, mic error, live status,
POST outcome, call outcome, app version / build / runtime / OTA update id /
channel, platform / OS / device model (or UA). The screen shows the id —
**`dx-XXXX-XXXX`** (Crockford-ish, no 0/O/1/I) — for the owner to read out;
`python scripts/diagnostics_tail.py --email … | --uid … | --id dx-…` prints
the latest record(s). Note `GET /telemetry` is unauthenticated (inherited),
so the payload carries uid + email in the clear.

### Decisions recorded (client)
* Offerer = lower uid (not "who joined second"): reconnect-safe and needs
  no server hint.
* `to` always sent, even in a two-member call.
* Native remote audio via react-native-webrtc's own rendering; route via
  expo-audio (`shouldRouteThroughEarpiece`) — no react-native-incall-manager.
  Unverified on hardware until the new build is installed.
* The therapist's own client never speaks and drops local suggestions; the
  server already never generates for her — belt and braces.
* `app.json`: `@config-plugins/react-native-webrtc` added, `expo.version`
  1.17.0 → **1.18.0**, `versionCode` 33 → **34** (a NEW NATIVE MODULE:
  the next Play/preview build must be an `eas build`, not an OTA; the 1.17.0
  runtime keeps taking its own OTAs).
* Android App Links (`https://arborfam-hub.web.app/call/…` opening the app
  directly) need `assetlinks.json` with the signing cert on the site — an
  owner item; until then the web page shows the `mindshift://` link.

