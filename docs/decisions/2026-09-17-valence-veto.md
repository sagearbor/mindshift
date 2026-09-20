# Decision: add valence as a veto on the loudness nudge (Option B)

**Date:** 2026-09-17
**Status:** implemented, shipped OFF. No production behaviour changes until
the owner sets two env vars.
**Decides:** the question left open in
`docs/plans/2026-09-10-heat-rubric-and-buzz-dose.md` — *what should buzz?*
**Code:** `server/watch/relay.py` (`valence_veto`, `VALENCE_VETO_MAX`),
tests in `server/tests/watch/test_relay.py`.

---

## The decision

**Option B, scoped to the server-relayed loudness lane.** A loudness nudge is
**suppressed** when the server's own audio-tone reading says the turn was
*pleasant* (valence > 0.48). It only ever removes a buzz. It never adds one,
and it never touches the `aggressive_tone` (text) lane.

Option C (raise the ladder) was rejected on the prior night's own evidence:
the +2…+14 dB sweep trades recall for false alarms and never separates anger
from joy, and it is a cross-language contract change for no benefit. Option A
(require text-tone confirmation) remains the real fix and is **not** done here
— see *What this does not fix*.

## Why

The shipped +6 dB rung measures **arousal**. Anger and joy are both
high-arousal, so the ladder cannot tell shouting at your wife from laughing
with your friends. `tone_id`'s default backend has been returning **valence**
— the pleasant/unpleasant axis — all along, and every caller has only ever
read `arousal`. The cheapest available improvement was a dimension we were
already computing and throwing away.

## The data

Joined two artefacts the 2026-09-11 session left on disk: the heat rubric
(`tmp/corpora/heat_rubric_cremad.csv`, dB over each speaker's **own** neutral)
and the valence probe (`tmp/corpora/valence_probe.json`, `tone_id`'s own
dimensional output). Overlap: **1,000 CREMA-D clips, balanced 250 each**
angry / happy / neutral / sad.

The veto can only act on clips that **already** clear the +6 dB rung — 325 of
the 1,000, of which 208 angry, 116 happy, 1 sad. (Neutral and sad essentially
never reach +6 dB, which is why sad's low valence — 0.345, statistically
indistinguishable from angry's 0.355 — costs nothing here. The veto is a
second gate behind loudness, never a trigger of its own.)

| | angry recall | fires on HAPPY | fires on all non-angry |
| --- | --- | --- | --- |
| shipped (+6 dB alone) | 83.2% | 46.4% | 15.6% |
| **+ valence veto @ 0.48** | **75.6%** | **18.4%** | **6.3%** |

Read as a share of what the rung already caught: the veto **keeps 90.9% of
the anger** and **removes 60.3% of the happy false alarms**.

**Threshold choice.** 0.48 is the highest threshold that still keeps ≥90% of
the caught anger. Validated **leave-one-speaker-out across 85 held-out
speakers** (threshold re-chosen on the other speakers each fold, applied to
the held-out one): 90.9% anger kept, 60.3% of happy false alarms removed —
identical to in-sample, so it is not overfit to these speakers.

**Applied at every rung, including the loudest**, because that is where it
pays best:

| rung | angry vetoed | happy vetoed |
| --- | --- | --- |
| level 1 (+6…10 dB) | 11% | 59% |
| level 2 (+10…14 dB) | 7% | 48% |
| level 3 (≥ +14 dB) | 9% | **93%** |

Loud *and* happy concentrates at the top rung — cheering, laughing hard.
Exempting level 3 to "let unmistakable shouting through" would have preserved
3.3 points of anger and given up 11 points of false-alarm removal.

## The cost, stated plainly

**Angry recall drops 83.2% → 75.6%.** Roughly one in eleven genuinely angry
turns that buzzes today will stop buzzing. That is a real regression in
sensitivity, accepted deliberately: a coach that buzzes at 46% of happy speech
teaches the wearer to ignore the buzz, and a nudge that is ignored has no
recall at all.

## How it is gated (two switches, both the owner's)

1. `MINDSHIFT_HEAT_VALENCE_GATE` — **new, defaults OFF.** Deliberately its own
   flag rather than reusing `tone_id.is_enabled()`: `MINDSHIFT_TONE_AUDIO`
   defaults to `dark`, and `dark` means *compute and log, never change what
   the user experiences*. A suppressed buzz is something the wearer can feel,
   so it must not ride in on `dark`.
2. `MINDSHIFT_TONE_AUDIO=on` — required as well, and currently `off` in
   production. Below `on`, `audio_pipeline._enrich_tone` returns `None` and no
   `ToneFlagEvent` reaches the relay at all, so there is no valence to read.

The veto **fails open** at every step: gate off, no flag, a categorical
backend with no `valence` key, a text-lane flag, or an unparseable/NaN score
all mean *do not suppress*. A missing signal is not evidence that a turn was
pleasant.

To roll back: unset `MINDSHIFT_HEAT_VALENCE_GATE`. No deploy needed.

## What this does not fix

- **The watch's own offline mic path is untouched.** `SentinelController` /
  `NudgeStateMachine` / `PulseEngine` still nudge on loudness alone when the
  phone is not in the loop. Valence is computed by a >1 GB WavLM model on the
  server and cannot run on the wrist; reaching that lane needs a new
  server→watch wire datum and Kotlin changes, which could not be device-tested
  tonight. The pulse train — the densest offender there — is already default
  **off** as of `d140928`.
- **Audio alone still buzzes ~18% of happy speech.** The residual signal is
  the **words**. Option A (require server text-tone confirmation before
  channel A escalates) is still the real fix, and costs the instant ~1 s tier.
- **No fusion exists.** Channels A and B still run as independent detectors
  with their own levels and cooldowns. This veto is the first place in the
  system where two signals take a joint decision — and it is a veto, not a
  fusion.

## What I would need to be more confident

1. **A corpus that is not acted.** CREMA-D and RAVDESS are actors reading
   fixed sentences. Real conversational anger is quieter, longer and more
   ambiguous; the acoustic lane has still only ever been validated on ~30 s of
   real speech (`family_real`).
2. **The owner's own recordings, labelled independently** — the gpt-audio
   labeller is written and tested and runs the moment OpenAI credits land.
   Everything above is `tone_id` grading a corpus's own labels; an independent
   listener would break that circularity.
3. **The 1,000-clip overlap is the limit here.** The rubric covers 7,442
   clips but the valence probe only scored 1,000 of them. Re-running the probe
   over the full set would tighten every interval — it is a compute cost, not
   a research problem.
4. **Live shadow data.** Running with `MINDSHIFT_TONE_AUDIO=on` and the gate
   **off** would log what the veto *would* have suppressed, on real sessions,
   without any wearer feeling a change. That is the recommended next step
   before the gate is ever switched on.

## Reproduce

```bash
python scripts/valence_gate_calib.py              # the tables above
pytest server/tests/watch/test_relay.py -q        # the behaviour, 29 tests
```

> **Note (2026-09-20):** the "flip `MINDSHIFT_TONE_AUDIO=on` with the valence gate off is invisible to wearers" claim above is incomplete. `on` also enables `tone_id.surface_allowed()`, which pushes `ToneFlagEvent` to the phone and to call participants (`audio_pipeline._enrich_tone`) and writes `audio_escalated` into the stored recap (`live_sessions.audio_tone_allowed`). Production went `off → dark` on 2026-09-20 (revision 00094); `on` lands only with the heat judge gating those paths — see `2026-09-20-heat-judge-plan.md`.
