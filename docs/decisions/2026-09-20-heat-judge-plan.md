# Decision: hysteresis on the ladder, WavLM+SpeechBrain as the cloud "heated" judge

*2026-09-20. Owner: "i think im good with your 1-4 suggestions, probably
including SpeechBrain." Evidence: `docs/research/2026-09-20-tone-model-options.md`,
`docs/plans/2026-09-19-heat-map-rounds.md`.*

## What was decided

1. **Hold-5s hysteresis on the loudness ladder.** The +6 dB first rung must
   hold for 5 consecutive 1 s windows before the ladder may escalate; reminders
   and decay unchanged. Measured dose: AMI 97 → 18/h, SBCSAE 80 → 12/h. 3 s is
   the fallback knob if the wrist feels sluggish. This is a three-runtime
   contract change (watch Kotlin `NudgePolicy`/`SentinelDetector`, phone
   `fastLoop.ts`, server `relay.py`) and must land in all three in one PR with
   the shared fixture updated.
2. **Flip Cloud Run `MINDSHIFT_TONE_AUDIO` from `off` to `dark`.** Logs
   arousal/valence/dominance beside every relayed session; no user-visible
   change. **Prerequisite:** bake the pinned 1.27 GB snapshot into the
   `INSTALL_VOICE=1` image layer (or GCS mount) so cold starts do not
   re-download it.
3. **Cloud "heated" judge on the relayed lane: WavLM odyssey_dim + SpeechBrain
   IEMOCAP.** Extend the PR #186 veto from valence-only to an arousal+valence
   confirm/veto within ~3 s of the instant tap. Stack AUC 0.941 / recall 0.66
   vs 0.897 / 0.59 for WavLM alone; +360 MB, +~100 ms. Gate: CONFER agreement
   must stay ≥ +0.55 and AMI/SBCSAE dose must not rise.
4. **On-device tier later:** eGeMAPS linear model (AUC 0.83, 36 ms) on the
   phone; distil our own Wav2Small-class student (MIT/Apache teachers only)
   for the watch. Published Wav2Small weights do not exist and are NC-licensed.

## Not decided / owner's call

- Streaming watch audio to the cloud when no phone is present (privacy posture).
- $50 OpenAI top-up to bench gpt-audio as a labeller (~$25 for a full bench).

## Order of work

1 (dose win, zero model risk) → 2 (data starts flowing) → 3 (quality) → 4.
Each step is verified from files: replay report + jest/pytest gate, never the
owner (`verify-from-files-never-owner`).

## Next steps and where each runs (2026-09-20, owner asked for the short form)

Latency = speech event → nudge on that device. Watch column assumes a phone is
present (relayed lane); watch-alone gets only rows 1 and 6. "5 s" earlier
meant two different things: the hold-N hysteresis rule (a deliberate delay on
the FIRST buzz of an episode, tunable 3 s) and the tone model's audio window
(2 s is enough: 130 ms compute measured; it was trained on 3–11 s podcast
segments, shorter than 2 s untested).

| # | step | watch | phone | web/cloud | CC effort |
|---|---|---|---|---|---|
| 1 | Hold-3s hysteresis on the loudness ladder, all runtimes | 3 s | 3 s | 3 s | ~4 h (3 runtimes + shared fixture + dose gate) |
| 2 | Bake WavLM weights into image; flip cloud flag off → dark | NO (logs only) | NO | ~0.5 s compute, logging | ~2 h (Dockerfile layer, build, deploy, verify logs) |
| 3 | Sliding 2 s tone window every 1 s; arousal+valence confirm/veto; flag dark → on | ~3 s via cloud | ~3 s via cloud | ~2.5 s | ~1 day (relay, gate on CONFER ≥ +0.55 and no dose rise) |
| 4 | Stack SpeechBrain angry-vs-happy vote on the judge | ~3 s via cloud | ~3 s via cloud | ~2.7 s | ~3 h (backend exists; wire + bench gate) |
| 5 | eGeMAPS + linear model on device (instant tier) | NO until Kotlin port (~2 s) | ~2 s | ~2 s | ~3 days phone (no openSMILE in RN: port features) + ~2 days watch |
| 6 | Distil 72 K student on our 31 h; run on watch | ~2 s | ~2 s | ~2 s | ~5 days (teacher labelling, GPU rental, ONNX on watch) |
| 7 | Regenerate CHiME-6 speaker labels from transcripts | — | — | offline | ~2 h |
| 8 | $50 OpenAI top-up; bench gpt-audio as labeller | NO | NO | 3–5 s, offline only | ~3 h after credits |
| 9 | Review and push eval/real-conversation-audit | — | — | — | ~30 min + owner review |

"Dark" is the middle of a three-state flag (`MINDSHIFT_TONE_AUDIO`): **off** =
not computed; **dark** = computed and logged beside every session, never
changes a buzz; **on** = drives the confirm/veto. Step 2 goes off → dark so
real-session data accumulates with zero user-visible risk; step 3 goes dark → on.

---

# Step 3 built (2026-09-20)

`server/watch/heat_judge.py`, calibrated by `scripts/heat_judge_calibrate.py`,
pinned by `server/tests/test_heat_judge.py` (44 tests). The phone's relayed
audio is buffered into a rolling **2 s window, scored once per second**,
exponentially smoothed over the last **3** scores, and the verdict confirms or
vetoes the loudness ladder's escalation.

## The rules, and the number behind each

| constant | value | evidence |
|---|---|---|
| `WINDOW_S` / `HOP_S` | 2.0 s / 1.0 s | 130 ms of CPU per 2 s clip (measured); 1 s is `SentinelDetector`'s own window cadence, so a verdict always lines up with the dB window that asked for it |
| `SMOOTHING_N` / `ALPHA` | 3 / 0.5 | weights 0.571 / 0.286 / 0.143 — the current window still dominates, but one odd window cannot flip a verdict on its own |
| `MIN_SCORES` | 2 | under this the judge says `unknown`, and `unknown` escalates |
| `VALENCE_VETO_MAX` | 0.48 | inherited unchanged from PR #186 (leave-one-speaker-out over 85 held-out CREMA-D speakers) |
| `CALM_AROUSAL_FLOOR` | **0.40** | highest floor that still keeps ≥ 90% of over-rung acted anger |
| `CONFIRM_AROUSAL` | **0.78** | above the 90th percentile of over-rung acted HAPPY arousal (0.756), far below CONFER's human-rated heated mean (0.839 ± 0.048) |
| `JUDGE_WAIT_S` | 2.0 s | one hop plus one model call, rounded up; only the FIRST rung ever waits |

**Verdict order — and the one place this departs from the brief.** `confirm`
is evaluated **before** the valence veto, not after it. Measured reason: the
inherited 0.48 valence bar, applied veto-first, vetoes **20.6%** of CONFER's
human-rated heated windows, and *no choice of either arousal knob can move that
number* because the veto never consults them. Letting a window at CONFER-heated
activation escape a borderline pleasantness reading takes it to **8.8%** and
costs 0.7 points of the happy veto. Pinned by
`test_the_rule_order_is_what_the_confer_numbers_chose`, against the corpus that
measured it.

## Calibration (`scripts/heat_judge_calibrate.py`)

Three gates at once; **77 of 1,140** threshold pairs clear all three. Chosen:
the pair vetoing the most acted HAPPY.

| gate | bar | at (0.40, 0.78) |
|---|---|---|
| CONFER human-rated heated windows vetoed | ≤ 10% | **8.8%** (89.7% confirmed) |
| over-rung acted HAPPY vetoed | ≥ 50% | **56.9%** |
| over-rung acted ANGRY vetoed (PR #186's recall gate) | ≤ 10% | **9.6%** — 90.4% of anger kept |

CONFER calm windows confirm at 26.7% against heated at 89.7% (separation 0.63).

Corpora, and the caveats kept in the open:

* **CONFER** — 345 scored 5 s windows over 24 TV-debate recordings, 68 heated
  (ten raters' mean conflict ≥ 400/1000). **Zero** of those 68 are also ≥ +6 dB
  over baseline, i.e. the shipped ladder nudges on none of them — consistent
  with round 2's "0 of 25 human-rated spans caught". So the CONFER gate bounds
  a cost the ladder cannot currently incur; it is kept as the guard against a
  future ladder that can, and as the only human-rated y-axis we have.
* **CREMA-D** — the same two artefacts PR #186 used (`tmp/corpora/valence_probe.json`
  + `tmp/corpora/heat_rubric_cremad.csv`), so the numbers are directly
  comparable: 116 happy and 208 angry clips clear the +6 dB rung. Valence alone
  = 60.3% happy / 9.1% angry, exactly reproducing that record. (The feature
  bank carries the same dims but its own dB estimator, which puts 99 happy
  clips over the rung instead of 116 — worth knowing before anyone re-derives
  these numbers from `tmp/feature-bank/` and finds different ones.)
* The cached reference windows are **5 s** and the judge runs on **2 s**. Both
  are inside the model's trained span, and the smoother gives the judge ~4 s of
  context, but the thresholds should be re-checked against real sessions once
  `dark` has accumulated `arousal`/`valence` series from live lanes — which is
  exactly why those series are now persisted (below).
* CONFER is smoothed along contiguous runs (a time series, which is what the
  judge sees); CREMA-D clips are treated as one steady state (the judge's three
  windows would all be that clip). The asymmetry biases the calibration toward
  vetoing more acted-happy than heated conversation — the safe direction.

## Dose: AMI and SBCSAE, replayed

Full audio, shipped nudge chain, gate built from the cached 5 s tone series
(unscored windows left OPEN — fail-open, same rule as the shipped veto):

| corpus | n | median buzzes/h before | after | fell | rose |
|---|---|---|---|---|---|
| AMI | 12 | 97.0 | **72.0** | 11 | 1 (`IS1000a`, 102.4 → 104.6) |
| SBCSAE | 60 | 80.2 | **72.2** | 50 | 6 |

A recording *can* come out marginally worse: removing an escalation can leave
the ladder at a lower level that later re-escalates — a second first-rung buzz
with a fresh PRD §6 reminder cycle, where the ungated replay sat at one high
level and reminded slowly. `test_the_judge_gate_never_raises_the_dose` bounds
this: median must not rise, total must not rise, risers must stay under a
quarter of fallers.

This does **not** reach the 12/h product target on its own; step 1's hold-3s
hysteresis is still the dominant lever (AMI 97 → 18/h).

## Mode × rung × verdict → action

`MINDSHIFT_TONE_AUDIO` is the only switch. **Production is `dark` now; the
merge deploy sets `on`.** Everything that changes at that flip is in this table.

### Wrist — the relayed loudness lane (`watch/relay.py`)

| mode | rung | verdict | action |
|---|---|---|---|
| `off` | any | — (no judge) | escalate exactly as today |
| `dark` | any | computed + logged | escalate exactly as today — **nothing the wearer can feel changes** |
| `on` | 1 (+6 dB) | `confirm` | escalate, no wait |
| `on` | 1 | `unknown` | **wait up to 2 s** for a verdict, then escalate anyway |
| `on` | 1 | `veto` | **no escalation this window** |
| `on` | 2 (+10 dB) / 3 (+14 dB) | `confirm` / `unknown` | escalate immediately — higher rungs are **never delayed** |
| `on` | 2 / 3 | `veto` | no escalation (vetoed, but never delayed) |

`aggressive_tone` — the words — is never touched, in any mode. A model that is
missing, cold, or broken yields `unknown`, which escalates: **a model outage
can never mute the product.** The waited escalation keeps its ORIGINAL stream
clock, so `NudgePolicy`'s cooldown arithmetic still sees the moment the ladder
crossed its rung.

### Phone + in-call surfacing (`audio_pipeline._enrich_tone`)

| mode | backend | verdict | `tone_flag` to the phone, call fan-out, and on to the relay |
|---|---|---|---|
| `off` | any | — | nothing computed |
| `dark` | any | computed + logged | **never sent** |
| `on` | dimensional (`odyssey_dim`, the default) | `confirm` | sent |
| `on` | dimensional | `unknown` / `veto` | **not sent** — logged only, exactly like `dark` |
| `on` | categorical (`superb_er`, `iemocap`) | n/a | sent, unchanged — a 4-class softmax has no arousal/valence axis, so the judge cannot have an opinion about one, and gating it to death would switch off a working configuration nobody asked to change |

This gate **fails closed** for a dimensional reading: no judge module, or a
judge that raised, means no surfacing. Showing a user "you sounded angry" is a
claim the server has to be able to support; escalating the wrist without a
verdict only costs a buzz they can attribute to their own volume.

Note the practical consequence of `CONFIRM_AROUSAL = 0.78`: roughly a quarter
of over-rung acted anger reaches it, so surfaced tone flags will be **rare**.
That is the intended posture for a first `on` deploy, not an accident.

### Stored analysis (`live_sessions.turn_tone_rows` → Growth / Replay / YourDay)

| `audio_allowed` | flag's `heat_verdict` | `audio_label` | `audio_escalated` |
|---|---|---|---|
| False | any | `None` | `None` |
| True | `confirm` | the model's label | per the label |
| True | `unknown` / `veto` / absent | the model's label | **False** |

The verdict travels on the flag's `scores` dict (`heat_verdict`: −1 veto /
0 unknown / +1 confirm), so a session stored tonight can still be told apart
from an unjudged one months from now. Dims are always kept; only the
escalation verdict needs a confirm.

### `MINDSHIFT_HEAT_VALENCE_GATE` — leave it OFF

The judge **supersedes** PR #186's per-turn `valence_veto`. Both read the same
valence axis at the same 0.48 bar, but the judge reads a rolling 2 s window
instead of one whole turn and carries the arousal floor as well, so when it has
an opinion the per-turn veto has nothing to add. One mechanism, not two: the
deploy must **not** set `MINDSHIFT_HEAT_VALENCE_GATE`. (`valence_veto` stays in
the code and stays tested — it is still the fallback for a build with no judge.)

`relay.tone_level` is inert for dimensional backends by construction: the
dimensional result reports `confidence 0.0`, below `TONE_FLAG_MIN_CONFIDENCE`,
so flipping the flag to `on` cannot make AUDIO tone drive the `aggressive_tone`
lane. Pinned by `test_tone_level_stays_inert_for_a_dimensional_backend`.

## Where the judge runs, and what it costs

Attached by `relay.register_live_session` — the judge exists to confirm or veto
a buzz on a **wrist**, so it runs exactly while a wrist is live, and never for a
phone-only session. It is also a no-op when the pinned snapshot is not on disk
(a filesystem check, no download): a server that would have to fetch 1.3 GB
first gets no judge rather than a queue of stalled windows, which makes step 2's
baked image layer a hard prerequisite for step 3 rather than a nicety. One score
at a time — a model slower than the hop drops windows rather than queueing them,
because a 4 s-old verdict is worse than no verdict. Scoring runs on a worker
thread; the phone's receive loop only appends bytes. One structured log line per
verdict (arousal, valence, verdict, latency ms), which in `dark` mode IS the
feature.

## Per-second series, persisted

`watch/routers/ws.py`'s `series` now carries `arousal`, `valence` and `judge`
beside `rms_db` / `hr_bpm` / `hr_t` — one sample per second, the judge's whole
stream and not just the moments it acted. `judge` is a code (−1 / 0 / +1)
because `LiveSession.series` is typed `dict[str, list[float]]`; the ordering
veto < unknown < confirm is meaningful and plots. The keys carry no `_series`
suffix, matching the existing `rms_db` / `hr_bpm` convention. No keys at all are
added when the flag is off or nothing was judged, so a stored document stays
byte-identical to today's.

This is the answer to item 4 of the 2026-09-17 record's "what I would need to
be more confident": live shadow data, so the next calibration can use real
sessions instead of CONFER and CREMA-D.

## Step 4 (SpeechBrain) — designed for, deliberately not wired

`tone_id.angry_vote` / `stacked_heat_score` landed on a parallel branch while
this was being built. The judge stores the full dims dict per window and
`verdict_from` ignores unknown keys, so the second vote is a scorer change plus
one constant. It is **not** wired here for one measurable reason: the cached
CONFER reference series carry arousal and valence but **no dominance**, and
`stacked_heat_score` needs all three — so the stack cannot be evaluated against
the human-rated corpus without re-running the model over 24 recordings. Wiring
a signal that has not been measured on the strongest y-axis in the repo is
exactly what the rest of this document argues against. The `dark`-mode series
above will supply that data.

## Reproduce

```bash
python scripts/heat_judge_calibrate.py --dose     # every number above
pytest server/tests/test_heat_judge.py -q         # 44 tests
```

The corpus-scale tests skip cleanly without the gitignored `tmp/` data (that
includes CI); the rules, thresholds, wiring, degrade and latency all run
everywhere.
