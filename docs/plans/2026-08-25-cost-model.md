# What MindShift costs to run, and what stops it running away

**Status:** cost model + guardrails shipped 2026-08-25 (`feat/cost-guardrails`).
**Why now:** if this reaches real therapists, the Anthropic / Deepgram / GCP
bills land on the owner's personal cards. Before this branch, nothing in the
server bounded what one account could spend in a day, and nothing said where
the money went. This doc is the arithmetic; `server/usage_meter.py` is the
enforcement.

Everything below is either a **measured** number from this repo or a **stated
assumption**, labelled as such. Nothing is remembered or guessed.

---

## 1. Unit prices (public list prices, read 2026-08-25)

| Item | Price | Source |
|---|---|---|
| Claude Haiku 4.5 input | $1.00 / MTok | [Anthropic pricing](https://docs.claude.com/en/docs/about-claude/pricing) |
| Claude Haiku 4.5 output | $5.00 / MTok | same |
| Claude Sonnet 5 input / output | $3.00 / $15.00 per MTok | same |
| Claude Opus 5 input / output | $5.00 / $25.00 per MTok | same |
| Deepgram Nova-3 streaming | **$0.0077/min** regular (a $0.0048/min promo is live today) | [Deepgram pricing](https://deepgram.com/pricing) |
| Deepgram Nova-3 pre-recorded | $0.0043/min | same |
| GCS Standard, us-central1 | $0.020 / GB-month | [Cloud Storage pricing](https://cloud.google.com/storage/pricing) |
| GCS Class A ops (write/list) | $0.005 / 1,000 | same |
| GCS Class B ops (read) | $0.0004 / 1,000 | same |
| GCS egress to internet (0–10 TiB) | $0.12 / GB | same |
| Cloud Run, CPU always allocated | $0.000018 / vCPU-s | [Cloud Run pricing](https://cloud.google.com/run/pricing) |
| Cloud Run, CPU always allocated | $0.000002 / GiB-s | same |

We budget Deepgram at the **regular** streaming price, not the promo. A promo
expiring is not a cost surprise anyone should discover from an invoice.

The same numbers live in one constant block in `scripts/usage_report.py`
(`PRICES_READ_ON`, `LLM_PRICES`, `DEEPGRAM_*`, `GCS_*`, `CLOUD_RUN_*`). Update
both together.

**The model matters more than anything else here.** The live loop runs on
`MINDSHIFT_MODEL`; measurements below are Haiku 4.5. Moving the per-turn
coaching call to Opus 5 multiplies the largest line item by 5×.

---

## 2. What one 30-minute 3-way coached call costs

"3-way" is the shipped in-app call: **two coached participants + one therapist
observer** (`server/calls.py`, `MAX_PARTICIPANT_ROLE = 2`,
`MAX_THERAPIST_ROLE = 1`).

### Measured inputs

| Quantity | Value | Where it was measured |
|---|---|---|
| Input tokens per live coaching call | **339.5** (30,554 over 90 calls) | `docs/plans/2026-08-24-live-e2e.md` § Hedged streaming, real API, 2026-08-24 |
| Hedge surcharge (the losing request's identical prompt, billed too) | **+6.9 %** (+2,115 tokens on 30,554) | same run |
| Output tokens, suggestion turn | **~80** (3 sentences + one integer) | `server/audio_pipeline.py` § `SUGGESTION_MAX_TOKENS` comment |
| Output tokens, nudge turn | **~25** (≤6 words + one integer; cap 60) | `NUDGE_MAX_TOKENS` |
| Conversational turn rate | **12–14 turns/min** | the repo's own scene fixtures: `scene_couple_escalation` 13 turns / 64.8 s, `scene_family3` 15 / 63.6 s, `scene_meeting4` 17 / 76.5 s |
| LLM calls per turn | **~1** | the same e2e run: 90 calls over 6 scene runs of 13–17 turns |

### Stated assumptions

* **10 turns/minute** across all three speakers — below the 12–14 of the
  gapless synthetic scenes, because real conversation has pauses. 30 min → 300 turns.
* Speaking split 45 % / 45 % / 10 % (participant A / participant B / therapist).
* Each turn costs **two** LLM calls, because each coached phone runs its own
  session: a *nudge* in the speaker's session (their own delivery) and a
  *suggestion* in the listener's session (`_CallSessionEndpoint.on_remote_turn`
  → `enqueue_job`). The therapist observes and is never coached. → **600 calls**
  (270 nudges, 330 suggestions).
* One Cloud Run instance (the deployed shape: `--cpu 4 --memory 2Gi
  --no-cpu-throttling`) is alive for the call and nothing else is on it. This
  is the honest worst case for a small deployment; with *N* concurrent calls
  per instance, divide that line by *N*.
* Post-call: `MINDSHIFT_CALL_ANALYZE` and `MINDSHIFT_CALL_REFLECT` are both on
  by default → one batch analysis + one reflection **per participant**.

### The bill

| Line | Arithmetic | Cost |
|---|---|---|
| Live coaching, input | 600 × 339.5 × 1.069 = 217,800 tok × $1.00/MTok | **$0.218** |
| Live coaching, output | (270 × 25) + (330 × 80) = 33,150 tok × $5.00/MTok | **$0.166** |
| Post-call analysis + reflection | 2 participants × 2 passes; ~21,200 in + ~9,000 out | **$0.066** |
| Cloud Run | 1,800 s × (4 × $0.000018 + 2 × $0.000002) | **$0.137** |
| Deepgram STT — **local-first (shipped)** | phones transcribe on-device; the server stops feeding Deepgram on the first `turn_local` | **$0.00** |
| GCS storage + ops | 3 episodes × (meta + turns + analysis), no media (`media_type: "none"`), ~350 KB + ~12 Class A ops | **<$0.001** |
| **Total, local-first** | | **≈ $0.59** |
| Deepgram STT — *if all three legs use cloud STT* | 3 × 30 min × $0.0077/min | +$0.693 |
| **Total, cloud STT** | | **≈ $1.28** |

**One 30-minute 3-way coached call costs about $0.59 on the shipped
local-first path, or about $1.28 if every leg falls back to cloud STT.**

Not counted, because they are one-time or negligible per call: the ~80 MB
ECAPA ONNX download a new device makes once a day at most
(0.078 GB × $0.12 = **$0.94 per 100 downloads**), TTS (server-side TTS is off
for local-first clients — the phone speaks), and the usage counters themselves
(one ~1 KB shard write per active user per 30 s ≈ 2,900 Class A ops/user/day =
$0.014/user/day at the extreme of a user talking all day; in practice a few
writes per session).

---

## 3. Per therapist, per month

A therapy session is 50 minutes, so one session = 5/3 of the call above,
recomputed rather than scaled blindly (500 turns → 1,000 LLM calls; 3,000 s of
instance time):

| Line | 50-min session, local-first | with cloud STT |
|---|---|---|
| Live coaching LLM | $0.639 | $0.639 |
| Post-call LLM | $0.110 | $0.110 |
| Cloud Run | $0.228 | $0.228 |
| Deepgram | $0.00 | $1.155 |
| **Per session** | **$0.98** | **$2.13** |

| Sessions / month | Local-first | Cloud STT |
|---|---|---|
| 10 | **$9.80** | $21.30 |
| 50 | **$48.90** | $106.50 |
| 200 | **$195.60** | $426.00 |

Two caveats that cut the real number:

* The Cloud Run line assumes one call per instance. A therapist doing 200
  sessions a month has overlapping load with everyone else on the service;
  at an average of 3 concurrent calls per instance that line drops from
  $45.60 to $15.20 at the 200-session tier.
* Storage grows, it does not repeat: 200 transcript-only episodes a month is
  ~70 MB → **$0.0014/month** in GCS. Storing *media* changes this a lot — a
  50-minute AAC derivative is ~24 MB, so 200 stored recordings/month adds
  ~4.7 GB/month, and by month 12 the accumulated 56 GB costs $1.12/month.
  Still small; it is the LLM and the STT that matter.

**Rule of thumb: ~$1 per coached session, so ~$50/therapist/month at 50
sessions — as long as the phones transcribe on-device.**

---

## 4. The three biggest levers

Ranked by measured share of the local-first bill.

### 1. Stop generating a coaching call for every single turn — 65 % of the bill

Live coaching is $0.384 of $0.59. It is one LLM call per turn, per listening
participant, unconditionally. The interject threshold
(`SuggestionEvent.speak`, `job.interject_level`) already decides whether a
suggestion is *voiced* — but only **after** the model has been paid to write
it. Deciding *before* the call, from signals the phone already computes for
free (on-device tone, prosody, the `turn_local` `text_tone` block), would cut
this line by whatever fraction of turns do not need coaching. In the
scene fixtures that is most of them.

Cheaper variants of the same lever, in order of effort:
* Don't coach the observer's turns at all when the therapist is speaking to
  the room rather than to a participant.
* Raise `MINDSHIFT_LLM_HEDGE_AFTER_MS` (default 1500 ms): the hedge costs a
  measured **+6.9 % of all input tokens** to buy back a p95 of 1.2 s instead of
  5.8 s. That is a good trade today; it is the first thing to give back if
  cost bites.
* Prompt caching is *not* a lever here: the live prompts are ~200 tokens, far
  below Haiku 4.5's 4,096-token cacheable minimum, and measurement showed a
  worse latency tail with the marker on (`server/llm_client.py`,
  `PROMPT_CACHE_ENABLED`).

### 2. Keep speech-to-text on the phone — the difference between $0.59 and $1.28

Cloud STT is 54 % of the bill the moment it engages, and it engages silently:
any leg whose on-device STT fails falls back to Deepgram, and a 3-way call
bills **three** simultaneous streams. The server already stops feeding
Deepgram the instant a `turn_local` arrives (`ctx.local_first` in
`audio_pipeline._run_session`), so the lever is entirely about the *phones*:
on-device STT actually working on real hardware, and the fallback being
visible when it doesn't. `GET /admin/usage` now shows `stt_seconds` per uid —
a non-zero number for a Track-3 phone is the alarm.

### 3. Right-size Cloud Run and batch the post-call tail — 23 % + 11 %

* **Cloud Run is $0.137 of a $0.59 call** purely because the service is
  deployed `--cpu 4 --memory 2Gi --no-cpu-throttling`, and 4 vCPU exists for
  *video transcode*, not for realtime coaching. Splitting the realtime WS onto
  a 1-vCPU service would take that line to $0.036 — a **17 % cut to the whole
  call** for a deploy-script change.
* The post-call analysis + reflection ($0.066) is not latency-sensitive: it
  runs in a background task after the call has ended. The Message Batches API
  is 50 % off, which halves that line.

Explicitly *not* a lever: storage. Transcript-only episodes cost fractions of
a cent per therapist-year. Do not spend engineering there.

---

## 5. The guardrails that now exist

### Counters (`server/usage_meter.py`)

Per uid, per UTC day, a flat counter dict:

* `llm.<site>.{calls,input_tokens,output_tokens,cache_read_input_tokens,cache_creation_input_tokens,hedge_extra_input_tokens}`
  — sites: `live_suggestion`, `live_nudge`, `live_repair`, `batch_analysis`,
  `counterfactual`, `reflection`, `respond`, `score`, `export`,
  `watch_summary`, `unattributed`.
* `stt.seconds`, `live.minutes`, `model.downloads`, `model.bytes`, `calls.started`.

Attribution rides a `ContextVar` (`usage_meter.attribute` /
`usage_scope`) that `LLMClient._record_usage` reads, so one process-wide LLM
client still bills the right user — `asyncio.to_thread` and
`asyncio.create_task` both copy the context, which is what every LLM call site
in this codebase uses.

Recording is an in-memory dict add under a lock (safe on the per-utterance hot
path). A background flusher writes one blob per process per uid per day to
`usage/{day}/{uid}/{instance}.json` every 30 s. Each process owns its shard
exclusively — no read-modify-write race, no lost update across Cloud Run
instances, and a uid's total is the sum of its shards.

**Accuracy contract, stated plainly:** counters are best-effort. Up to 30 s of
usage is lost if a process dies, and with no bucket configured they live only
in memory. Quotas are enforced from one process's view (its own counters plus
a snapshot of other shards refreshed every 60 s), so with *N* instances a
determined user can overrun a cap by about one refresh window before every
instance sees it. That is deliberate: a synchronous GCS read-modify-write per
utterance would put the bucket in the coaching hot path.

### Quotas — degrade, never break

| Env var | Default | Gates |
|---|---|---|
| `MINDSHIFT_DAILY_LLM_TOKENS` | 1,500,000 | cloud suggestions, batch analysis |
| `MINDSHIFT_DAILY_STT_SECONDS` | 21,600 (6 h) | cloud transcription |
| `MINDSHIFT_DAILY_LIVE_MINUTES` | 480 (8 h) | both of the above |
| `MINDSHIFT_DAILY_MODEL_DOWNLOADS` | 25 | `GET /models/ecapa.onnx` |
| `MINDSHIFT_DAILY_CALLS` | 50 | `POST /calls` |

`0` disables a cap. The defaults are generous on purpose: 1.5 M tokens is
roughly **37 back-to-back 30-minute coached calls in one day** — far past
honest use, close enough to matter to a runaway client.

What happens at the cap:

* **Live session** — cloud suggestions stop. The transcript keeps flowing, the
  phone's on-device loop is untouched, the socket stays open, and the client
  gets exactly one `quota_notice` frame naming the limit, the numbers, the
  reset time, what stopped, and what still works. Same shape for the STT cap:
  the vendor stream stops, `turn_local` from the phone keeps producing
  transcript *and* coaching.
* **Live-session ingest (`POST /sessions/live`)** — never blocked. The phone
  has already recorded that transcript; refusing it would be data loss. The
  LLM tail is skipped instead and the reason is stored on the episode as
  `analysis.live.quota_notice`.
* **REST spend paths** (`/analyze*`, `/respond`, `/score`, `/episodes/{id}/reflect`
  with `force`, `/calls`, `/models/ecapa.onnx`) — HTTP **429** carrying the
  same `quota_notice` body plus `Retry-After`. There is no half of `/analyze`
  worth returning without the model, so these fail closed — but loudly, with a
  reset time. A *cached* reflection stays readable: refusing a free read would
  be theatre.

Nothing is ever silently dropped.

### Owner visibility

`GET /admin/usage?since=YYYY-MM-DD` — per-uid rollups, sorted by spend,
restricted to a `MINDSHIFT_ADMIN_UIDS` comma-separated allowlist. **Unset
means closed**, and a signed-in non-admin gets 404 rather than 403.

```bash
export MINDSHIFT_ADMIN_ID_TOKEN=<a firebase id token for an allowlisted uid>
python scripts/usage_report.py --since 2026-08-01
python scripts/usage_report.py --since 2026-08-01 --csv > usage.csv
python scripts/usage_report.py --json saved.json --model claude-opus-5
```

The script prints the per-uid table, a run rate, and a projected monthly bill
at the prices in §1.

---

## 6. What is still uncounted

Honest gaps, so nobody reads the report as an invoice:

* **Cloud Run and GCS are not metered per uid.** They are not attributable to
  a user from inside the request, so §2 models them and the report says so.
* **TTS** (`TTSClient`) is not counted. Server-side TTS is off for local-first
  clients, so it is zero on the shipped path; a legacy client would spend
  there unmetered.
* **Recording storage bytes** are not counted per uid. The `/recordings` list
  already knows the sizes; wiring them into the meter is the obvious next
  counter if media storage is ever turned on broadly.
* **Deepgram pre-recorded** (the `/analyze/upload` path) is not metered — only
  the live streaming seconds are. An upload's minutes are known at
  transcription time and should be added.
