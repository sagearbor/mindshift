# Nudge quality — offline measurement and the top-two fixes (2026-08-30)

MVP item #2: *the quality of the nudges* — the real-time coaching lines the
Live Coach shows/speaks. This note measures them offline on data we already
had, names the failure modes with evidence, fixes the top two in the
prompt/policy layer, and re-measures on the same data. No live tests; the
pleasantness math and the diarization are untouched.

Everything here is reproducible with `grade.py` (see *Method*). The LLM
judgments are cached under `tmp/nudge-eval/llm_cache/` (private, gitignored)
so re-runs are free; the owner's session data lives under `tmp/nudge-eval/<id>/`
and never leaves `tmp/`.

## TL;DR

| metric (same data, same judge rubric) | before | after |
|---|---|---|
| **Scenes — expected nudges hit** (3 fixtures, 6 hand-authored nudges) | 6/6, strong 6/6 | 6/6, strong 6/6 |
| **Scenes — false-positive nudges on calm/repairing self turns** | **4/11 (36%)** | **1/11 (9%)** |
| Scenes — response lines ≤ 12 words (speakable in ~4 s) | 54% (mean 12.6 w) | 96% (mean 10.3 w) |
| Scenes — response judge overall (1–5) | 4.21 | 4.47 |
| Real sessions (8, 77 turns) replayed through the cloud coach — speakable | 69% (mean 11.7 w) | 97% (mean 10.1 w) |
| Real replay — judge overall / specific / not preachy | 3.87 / 3.29 / 4.31 | 4.05 / 3.55 / 4.52 |
| On-device prompt (v1→v2, cloud model standing in for Nano) — self turns nudged when nothing should change | **11/11 (100%)** | **0/11** |
| On-device prompt — response lines ≤ 12 words / addressed to the user | 29% / 4.38 | 100% / 4.90 |
| On-device prompt — expected nudges | 6/6 hit | 5/6 hit + 1 late (next self turn) |

The two fixes: **(1) the coach now sees the exchange and itself** — the last
six turns and its own recent lines go into the prompt, with per-kind
guidance (stay silent on a calm/repairing self turn; a verbatim, first-person,
≤10-word line for the other side) — and **(2) a policy gate after the model**
— a nudge below importance 40 or one that re-issues a nudge from the last 45 s
is silence; a first suggestion that repeats a recent line is demoted behind
the model's first different alternative. The on-device prompt got the same
two rules in its own words, plus a phone-side repeat gate ready to wire.

Spend: 1,181 real Haiku 4.5 calls (coach + judge, all runs incl. an
interrupted first pass), 522k input / 101k output tokens ≈ **$1.03**.

## Method

`grade.py` runs the **real cloud coaching functions** — `audio_pipeline
._generate_nudge` (self turn) and `._generate_suggestions` (other turn), the
same prompt bytes the WebSocket worker sends, plus the same session helpers
(`_remember_utterance`, `_history_for_prompt`, `_gate_nudge`,
`_gate_suggestions`, `_remember_coaching`) in the order `process_segment`
uses them — over two data sets:

* **Scene fixtures** (`server/tests/fixtures/audio/test_recording_scene_*`):
  `scripts/live_e2e.build_turn_locals` turns each scene into the exact
  `turn_local` frames a phone would send (is_self oracle, prosody measured
  from the wav, the fixed text_tone table) and the coach is run turn by turn.
  Graded against the hand-authored `expected_nudges` timeline. Empathy 50,
  interject level 0 (the app's defaults); role Husband / Parent / Team lead.
* **The owner's real live sessions** (8 from 2026-08-26, `gs://…/recordings/
  <uid>/<id>/turns.json`): the stored on-device suggestions graded **as-is**,
  and the transcript **replayed** through the cloud coach exactly as
  production would have coached it (every turn as an OTHER turn — see
  finding 3).
* **The on-device prompt** (`apps/mobile/src/live/localLlm.ts buildPrompt`),
  ported verbatim to Python and run on the same transcripts with the cloud
  model standing in for Gemini Nano — this grades the *prompt*, not Nano.

Metrics, deterministic first:

* **TIMING** (scenes): hit = a non-empty nudge on the expected self turn;
  late = on the next self turn within two turns; miss otherwise. A *strong*
  expectation must clear importance 60. **False positive** = a non-empty
  nudge on a self turn the script marks calm/neutral/repair/warm. Non-self
  turns can never get a nudge in this pipeline (structural; reported as 0).
* **LENGTH**: words; *speakable* = ≤ 12 words (~4 s). **REPETITION**:
  word-bigram Jaccard against the previous coaching line in the session
  (≥ 0.5 = repeat).
* **RELEVANCE**: `claude-haiku-4-5-20251001` as judge, temperature 0, one
  fixed rubric for every run (in `grade.py: JUDGE_SYSTEM`): five 1–5 scores —
  `addressed_to_self`, `specific`, `actionable`, `not_preachy`,
  `not_repeating` — given the last 4 turns, the coached turn, the previous
  coaching line and the line to grade. Cached to disk by prompt hash.
* **LATENCY**: from the owner's diagnostics records (`scripts/
  diagnostics_tail.py --uid …`, saved under `tmp/nudge-eval/diagnostics/`)
  and the `suggestion_source` split of the stored turns.

```
set -a; source .env; set +a
MINDSHIFT_COACH_CONTEXT=0 tmp/venv-voice/bin/python docs/research/2026-08-30-nudge-quality/grade.py --tag before --sources scenes,real,replay,ondevice_v1
tmp/venv-voice/bin/python docs/research/2026-08-30-nudge-quality/grade.py --tag after  --sources scenes,real,replay,ondevice_v2
```

`MINDSHIFT_COACH_CONTEXT=0` restores the pre-fix single-turn prompt byte for
byte, so before/after is the same code with one switch. Results land in
`tmp/nudge-eval/results/<tag>.json` (per-turn records + judge notes) and a
Markdown summary on stdout.

## What the data said (before)

### 1. Nudges fire when nothing should change — praise and "drop the apology"

Cloud coach on the scenes: every expected nudge hit (6/6, all strong ones
≥ 85), but **4 of the 11 calm/repairing self turns also got one**:

| scene / turn (scripted emotion) | nudge | importance |
|---|---|---|
| couple #10 (repair_apology: "You're right. I'm sorry…") | "good — hold that tone" | 25 |
| family3 #5 (calm_neutral: "It was probably me, honestly…") | "you're over-owning — be matter-of-fact" | 62 |
| family3 #11 (repair_apology: "I know. I snapped too…") | "good — hold that tone" | 15 |
| meeting4 #15 (repair_apology: "Yeah. Sorry, everyone…") | "drop the apology, own it straight" | 72 |

Two mechanisms: the self-turn system prompt's own example list contains
"good — hold that tone", and the app's default interject level is 0, so a
nudge the model itself rated 15–25 is voiced anyway; and the balanced stance
("nudge them firmer when they over-apologize") reads a sincere repair as
hedging. The on-device prompt is worse: with no "say nothing" clause it
nudged **every** self turn (11/11 calm turns: "Smile warmly. You've turned
this around beautifully.", "Add a please—soften the command.").

### 2. The coach repeats itself and can't see the exchange

The cloud user turn was a single line — `Transcript turn: "…"` — no prior
turns, no memory of what it had already said. `not_repeating` is the lowest
judge dimension on every data set (scenes 3.3, real replay 2.7, stored
on-device 2.3): "I hear you—let's figure out what's not working and fix it
together" → "let's figure out who did it…" → "let's figure this out together"
across three consecutive turns; the judge's notes say *"Reuses 'I hear the
[X] concern' frame"*, *"same core move"*. On the phone the same words were
delivered **twice in a row** (two fragments of one sentence, 3 s apart),
and one stored line was a meta-instruction rather than something to say
("Acknowledge their concern, then firmly reiterate the agreed-upon meeting
spot" — `addressed_to_self` 2). Lines were also too long to whisper: 46% of
cloud responses and 71% of on-device responses ran over 12 words (the prompts
allowed 15 and 18).

### 3. Structural findings (not fixable in the prompt layer — reported)

* **No real session carries a self identity.** All 8 sessions have
  `is_self = null` on every turn (6 of them label every turn "Unknown" —
  the speaker labeler produced nothing; the other two have A/B/C/D labels but
  no enrolled print). So in production the **nudge path never ran**: every
  turn, including the owner's own, was coached as an OTHER turn ("what to say
  back"). The phone's "Speaker A speaks first" fallback only helps when the
  labeler emits a label. Timing can therefore not be graded on real sessions;
  the scene fixtures (identity oracle) carry that measurement.
* **The coach hears itself.** Turns contain the coach's own spoken lines:
  one turn ends in "…Be firmer, Don't back down" (the self-feedback prompt's
  example text, spoken by TTS and transcribed back), another is "I hear you,
  what would help you feel better right now?" — a suggestion re-coached as
  if the other person said it. Echo suppression / matching against
  just-spoken lines belongs in the audio/fast-loop layer.
* **Fragments and cumulative-STT duplicates** in the pre-fix sessions
  (12:xx–14:xx): "I need you to stop interrupting me so I can finish my
  point. I was definitely not yell" / "…not yelling you, you?" / "was
  definitely not yelling you." — three turns, one sentence. The dedup landed
  on 2026-08-26 (78ffac2); the 20:55 session no longer shows it, but VAD
  still splits sentences ("always there." as its own turn). Context in the
  prompt (fix 1) is what lets the coach read a fragment against the turn
  before it.
* **Latency.** Diagnostics for the 20:55 session (on-device, Nano answering):
  median LLM **5.5 s**, median segment-end→speak **15.9 s**, 3 of 7
  suggestions held (someone was talking) and 5 spoken. Earlier sessions fell
  through to cloud after the os rung's 4 s timeout on every turn (median
  "LLM" 4.0 s was the timeout itself). By the time a line is spoken the
  conversation has moved on 2–3 turns; the latest-wins queue then coaches
  the newest turn, but nothing shortens Nano itself.
* Stored `suggestion_source` split: 7 on-device, 0 cloud — the phone stores
  only its own line, so the cloud's suggestions are not in `turns.json` (the
  replay above is how they were graded).

## The fixes

### Fix 1 — the coach sees the exchange and itself (`server/audio_pipeline.py`, `localLlm.ts`)

* `_history_for_prompt(ctx, utterance, is_self=…)` builds, at enqueue time,
  the last `COACH_CONTEXT_TURNS` (6) turns with "You" for any label the
  phone has called self and the mid-call display name otherwise, plus the
  coach's own recent lines (`ctx.coaching_log`, filled by
  `_remember_coaching` after each event is sent). `_render_history` prints
  it deterministically after the transcript turn and the tone hints:

  ```
  Transcript turn: "I SAID you never listen!"

  Recent exchange before this turn (oldest first):
  - You: "You never listen to me."
  - (coach whispered to the user: "ease up")
  - Mom: "I do listen."
  Nudge ONLY if something about HOW the user just came across should change right now. A calm, sincere, or repairing turn (an apology, owning a mistake, agreeing) needs no nudge: return "" — never praise, never tell them to drop a sincere apology. Never re-issue a nudge the coach already whispered above.
  ```

  For an OTHER turn the trailing line is instead: *each suggestion is
  something the user can say verbatim, first person, 10 words or fewer,
  grounded in what was just said; do not reword a line the coach already
  gave, and do not open with the same words it opened with.*
* `_turn_prompt(..., history=None)`: byte-identical without history (the
  existing prompt tests still pin `'Transcript turn: "hi"'`);
  `MINDSHIFT_COACH_CONTEXT=0` turns the whole block off.
* On-device prompt v2 (`SUGGESTION_SYSTEM_PROMPT` + `buildPrompt`): one line
  ≤ 10 words in the coached person's own voice, never an instruction to be
  translated first; a self turn may return an empty suggestion ("calm,
  sincere, apologizing, agreeing"), never praise. The Python port in
  `grade.py` is byte-identical and is what the v2 rows above measure.

### Fix 2 — policy after the model (`server/audio_pipeline.py`, `nudgePolicy.ts`)

* `_gate_nudge`: importance < `NUDGE_MIN_IMPORTANCE` (40) → silence; a
  nudge whose word-bigram Jaccard with a nudge from the last
  `COACH_REPEAT_COOLDOWN_S` (45 s of session time) is ≥ 0.5 → silence. The
  expected nudges in the fixtures scored 45–95; the false positives 15–25
  (praise) — the floor separates them; the prompt guidance handles the two
  higher-scored ones.
* `_gate_suggestions`: when the first (voiced) suggestion repeats a recent
  coaching line, the first alternative that does not is promoted; nothing is
  dropped.
* `CoachRepeatGate` in `nudgePolicy.ts` (+ `coachingOverlap`) is the
  phone-side mirror with the same constants, tested against the exact line
  the 20:55 session repeated. **Not yet wired** into `fastLoop.finalizeTurn`
  (another owner) — a three-line change: `admit(suggestion, span.end, kind)`
  before `speakNow`/`held`.

Tests: `server/tests/test_nudge_quality.py` (13: history/rendering,
byte-identical-off, both gates, bounded log, and one run through the real
WebSocket worker showing the second turn's prompt carries the exchange and a
repeated nudge is silenced); `apps/mobile/__tests__/liveNudgePolicy.test.ts`
(+4) and `liveLocalLlm.test.ts` (+2). `server/tests/test_calls.py`'s `env`
fixture disables the repeat gate (its LLM double answers every self turn
with the same "ease up" and the tests wait on each event).

## Before → after, in full

### Scenes (cloud coach, real pipeline functions)

| scene | expected | hit | late | miss | strong ok | FP / calm self turns | nudge words | nudge judge | response words | response ≤12w | response judge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| couple_escalation before | 3 | 3 | 0 | 0 | 3 | 1/4 | 5.0 | 4.60 | 13.3 | — | 4.67 |
| couple_escalation after | 3 | 3 | 0 | 0 | 3 | **0/4** | 5.3 | 4.33 | 10.5 | — | 4.70 |
| family3 before | 1 | 1 | 0 | 0 | 1 | 2/4 | 5.3 | 4.40 | 11.8 | — | 3.88 |
| family3 after | 1 | 1 | 0 | 0 | 1 | **1/4** | 6.0 | 4.40 | 10.6 | — | 4.32 |
| meeting4 before | 2 | 2 | 0 | 0 | 2 | 1/3 | 5.0 | 4.73 | 12.8 | — | 4.25 |
| meeting4 after | 2 | 2 | 0 | 0 | 2 | **0/3** | 4.5 | 4.60 | 9.9 | — | 4.48 |
| **all, nudges** | 6 | 6 | 0 | 0 | 6 | 4/11 → **1/11** | 5.1 → 5.3 | 4.58 → 4.43 (n 10 → 7) | | | |
| **all, responses** (n=28) | | | | | | | | | 12.6 → 10.3 | 54% → **96%** | 4.21 → **4.47** |

Response judge dimensions (scenes, before → after): addressed 5.0 → 4.89,
specific 3.93 → 4.21, actionable 4.46 → 4.89, not preachy 4.32 → 4.82,
not repeating 3.32 → 3.54. The remaining FP (family3 #2, "soften the edge —
ask, don't command", importance 65, on "Then it's my turn. Grab the plates
for me, would you?") is arguable — the line *is* a command.

The nudge judge mean dipped 4.58 → 4.43 because the three dropped lines were
the praise ("good — hold that tone", scored 4.6–4.8 by the judge, which does
not know the turn needed no nudge) — the timing table is the right measure
for those.

Expected nudges after: couple #4 "ease up — you sound angry" (72), #6
"you're attacking, not discussing — reset" (92), #8 "stop — you're
punishing, not problem-solving" (95); family3 #9 "You sound
defensive—soften, listen first" (75); meeting4 #11 "that sounded blaming —
soften it" (72), #13 "ease up — you're losing them" (92).

### Real sessions — stored on-device suggestions, as-is (unchanged by definition)

7 lines in 2 of 8 sessions, all `suggestion_source: on-device`; 6 responses:
mean 10.7 words, 1 exact repeat (17%), judge addressed 3.33 / specific 2.83 /
actionable 3.33 / not preachy 3.0 / not repeating 2.33 / overall **2.96**;
the 1 nudge ("Pause, let them respond. Don't rush.") overall 4.0.

### Real sessions — transcript replayed through the cloud coach (77 coached turns)

| | mean words | ≤12 words | addressed | specific | actionable | not preachy | not repeating | overall |
|---|---|---|---|---|---|---|---|---|
| before | 11.7 | 69% | 4.79 | 3.29 | 4.31 | 4.31 | 2.66 | 3.87 |
| after | 10.1 | **97%** | 4.86 | **3.55** | **4.64** | **4.52** | 2.68 | **4.05** |

Per session the overall moved 3.55–4.18 → 3.73–4.34 (every session up or
flat). `not_repeating` did **not** move on the real transcripts: the judge
penalises the *stance* repeating ("You're right, I…" on four consecutive
fragments of the owner's test monologue), which is what a balanced coach
will keep saying to the same complaint; the bigram repeat rate is 0 in both
runs. See *What's next*.

### On-device prompt, v1 → v2 (cloud model standing in for Gemini Nano; 122 turns)

| | nudges on calm self turns (scenes) | expected nudges | response words | ≤12 words | addressed | specific | actionable | not preachy | not repeating | overall |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | 11/11 | 6/6 hit | 13.6 | 29% | 4.38 | 3.35 | 3.96 | 3.96 | 2.66 | 3.66 |
| v2 | **0/11** | 5/6 hit, 1 late | 8.7 | **100%** | **4.90** | 3.52 | 4.59 | 4.55 | 3.03 | **4.12** |

Same turn, v1 → v2 (owner's 20:55 session, paraphrased turn): *"…I'm right
here, but you're not…"* → v1 "I hear you. Let's be specific about location
and time so we're both there together." (15 w) → v2 "I know, and I'm heading
there now." (7 w). The late hit is couple #4 (mild, tense_rising): v2 stayed
silent there and fired "Take a breath. You're escalating fast." on #6 — the
price of the "say nothing when fine" clause on a borderline turn; the two
strong turns hit.

### Latency (recorded; unchanged by this work)

| session | mode | turns | spoken | held | median LLM | median to-speak | outcomes |
|---|---|---|---|---|---|---|---|
| 2026-08-26 20:55 (dx-XXXX-XXXX) | earpiece, on-device | 7 | 5 | 3 | 5.5 s | 15.9 s (p90 15.9) | os:ok ×6 |
| 2026-08-26 14:02 (dx-XXXX-XXXX) | earpiece | 6 | 0 | 0 | — | — | os:timeout 3, os:error 3 → cloud ×6 |
| 2026-08-26 13:52 (dx-XXXX-XXXX) | therapist | 10 | 0 | 0 | 4.0 s (the os timeout) | — | os:timeout 6, os:error 4 → cloud ×10 |

## Deviations from the brief

* The scenes are pushed through the real coaching **functions** (same
  prompt bytes, same gates, same session helpers) rather than the uvicorn
  WebSocket harness in `test_live_e2e_inprocess.py` — that harness depends on
  the pytest conftest's auth fakes. The WebSocket wiring is covered by the
  worker test in `test_nudge_quality.py` and the (green) e2e suite.
* Real sessions have no self identity, so TIMING on real data is not
  measurable; the replay coaches every turn as OTHER, as production did.
* Gemini Nano itself is not measured; the on-device rows grade the prompt
  with Haiku 4.5 as the model. Judge and coach are the same model family
  (as specified), so absolute judge scores carry self-preference; the
  deltas are on a fixed rubric.
* `server/tests/test_calls.py` (fixture only) was touched to disable the
  repeat gate for its always-"ease up" LLM double.

## What's next

1. **Wire `CoachRepeatGate` into `fastLoop.finalizeTurn`** and treat an empty
   v2 `suggestion` as "nothing to say" (today `parseSuggestionJson` returns
   null → the chain logs it as unparseable and falls through to the next
   provider; the fast loop should short-circuit on an empty string for self
   turns). Both are in `fastLoop.ts` (other owner).
2. **Identity is the real blocker for nudges in production** — no real
   session had a self turn, so the nudge path (and everything above about
   it) has never run for the owner. Enrollment at session start, or an
   explicit "I'm Speaker A" tap when the labeler says Unknown, would turn
   the nudge coach on.
3. Echo suppression: drop a finalized turn whose text overlaps the line the
   phone spoke in the last few seconds (the fast loop knows both).
4. Residual repetition on real transcripts is stance-level ("You're right…"
   ×4). Options: pass the *opening words* of the last two lines as an
   explicit avoid-list, or diversify by asking for three suggestions with
   different moves (acknowledge / ask / commit) and rotating the voiced one.
5. Latency: Nano at 5.5 s + a 16 s median to-speak makes even a perfect line
   late. Measure the v2 prompt on the device (it is ~40% fewer output
   tokens), and consider cloud-first with the on-device rung as the
   private-mode opt-in.
6. Cross-check the judge with a different model family on the cached
   records (`tmp/nudge-eval/results/*.json`) — a one-flag change in
   `grade.py` (`JUDGE_MODEL`).

## Files

* `docs/research/2026-08-30-nudge-quality/grade.py` — the grader (this note's numbers).
* `server/audio_pipeline.py` — `COACH_CONTEXT*`, `NUDGE_MIN_IMPORTANCE`,
  `COACH_REPEAT_*`; `SessionContext.coaching_log/self_labels`;
  `_history_for_prompt`, `_render_history`, `_remember_coaching`,
  `_coaching_overlap`, `_is_repeat_coaching`, `_gate_nudge`,
  `_gate_suggestions`; `_turn_prompt(history=)`; `SuggestionJob.history`;
  `process_segment` wiring.
* `server/tests/test_nudge_quality.py` (new), `server/tests/test_calls.py` (fixture).
* `apps/mobile/src/live/localLlm.ts` (prompt v2), `apps/mobile/src/live/nudgePolicy.ts`
  (`CoachRepeatGate`, `coachingOverlap`), `apps/mobile/__tests__/liveLocalLlm.test.ts`,
  `apps/mobile/__tests__/liveNudgePolicy.test.ts`.
* Private, under `tmp/nudge-eval/`: the 8 sessions' `turns.json`/`analysis.json`/`meta.json`,
  `diagnostics/owner-latest.json`, `llm_cache/` (1,181 cached calls), `results/{before,after}.json`.
