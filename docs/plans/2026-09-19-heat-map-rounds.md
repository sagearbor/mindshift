# How heated is each recording, and how well do we measure it — 2026-09-19

The owner's ask, verbatim in spirit: *test a lot of files and show me
graphically — how heated they are on one axis, how well we measure on the
other — so we can see how we do on calm conversations versus up-and-down ones.
Diverse data, graphed, three rounds of improvement if it isn't good.*

Everything here is reproducible with `scripts/heat_round.sh`; the graph is
`tmp/heat-map/heat_map.html` (one dot per recording, both axes selectable,
hover for detail).

## The axes

**x — how volatile the conversation is.** From the *same* 1 s window / running
baseline machinery the watch runs (a faithful port): the standard deviation of
dB-over-baseline across voiced windows. Alternatives selectable on the graph:
share of windows over the +6 dB first rung; the tone model's own arousal
volatility (independent of loudness); and, for CONFER, how heated ten human
raters scored the clip.

**y — how well we measure.** Several metrics, each only where its ground truth
exists, never estimated:

| metric | needs | who has it |
| --- | --- | --- |
| agreement with **human** conflict ratings (Spearman per clip) | continuous human heat labels | CONFER |
| agreement with the **tone model's** arousal (Spearman per recording) | nothing — computed for every recording | all |
| loud-but-pleasant: share of loud windows the tone model calls pleasant | nothing | all |
| dose — buzzes/hour the shipped chain would deliver | nothing | all |
| attribution accuracy | speaker truth + a scored run | AMI, scenes, (CHiME-6, SBCSAE, VoxConverse) |
| nudge hit / false rate | per-turn heat labels | scenes, MELD |

The tone-model reference is a model, not a person. Agreement with it is
evidence, not proof — which is why CONFER's human ratings are also used to
check the reference itself (`reference_vs_human_conflict`).

## Round 1 — what we already had (19 recordings)

12 AMI office meetings, the owner's two real recordings, five scripted TTS
scenes.

| corpus | n | volatility sd | agreement w/ tone model | loud-but-pleasant | dose |
| --- | --- | --- | --- | --- | --- |
| AMI meetings | 12 | 5.2 dB | **+0.48** | 0.07 | 92/h |
| TTS scenes | 5 | 4.5 dB | **−0.27** | 0.01 | 78/h |
| owner (poker6) | 1 | 6.6 dB | +0.77 | **0.50** | 478/h |

Across recordings: Spearman(volatility, dose) = **0.88**;
Spearman(volatility, loud-but-pleasant) = **0.78**;
Spearman(volatility, agreement) = +0.45.

Three things visible at a glance:

1. **Every real recording sits above the zero line; four of five synthetic
   scenes sit below it.** On real audio our loudness-heat moves *with* an
   independent tone model; on the TTS scenes it moves *against* it. That is
   the acoustic-flatness finding from 2026-09-10 (synthetic anger is wording
   and pacing, not amplitude), now as a per-recording measurement — and a
   reason to stop treating those scenes as acoustic ground truth.
2. **Dose tracks volatility almost linearly** (0.88). Expected — the ladder
   *is* a loudness measure — but it means the more animated a conversation,
   the more it buzzes, whether or not anyone is angry.
3. **The poker game is the anger-vs-joy failure in one dot:** half of all its
   loud windows are ones the tone model calls *pleasant*. Excited, happy
   speech — and 478 buzzes/hour.

On calm vs volatile: agreement is *lower* on the calm recordings (+0.45
correlation with volatility). Part of that is honest — a flat signal has
little to track, so any correlation is unstable — and part is the real
problem: on calm conversation the thing loudness is picking up is not heat.

Round 1's diversity is poor by construction: office meetings and a poker game
are both "nobody is angry". Round 2 exists to fix that.

## Round 1b — candidate fixes, measured in replay on the same 19 recordings

Before any new data, the knobs that exist in the shipped chain were measured
against each other. Nothing here changed product behaviour; each is a column on
the graph (`dose_identity_ceiling`, `dose_valence_veto`, …).

| variant | mean dose | duration-weighted |
| --- | --- | --- |
| shipped today | 117/h | **89.7/h** |
| only the wearer's own windows may buzz (perfect identity — Option A's *ceiling*) | 82/h | 72.0/h |
| valence veto extended to the watch lane | 100/h | 80.0/h |
| identity **and** valence | 73/h | **64.1/h** |
| first rung +8 dB | 80/h | 71.8/h |
| first rung +10 dB | 63/h | 51.3/h |

By corpus (weighted, shipped → identity → valence → both): AMI 89 → 72 → 79 →
64; the owner's recordings 361 → 181 → 241 → 122.

What this says, and it is the most important result so far:

- **Perfect identity removes only ~30% of the dose.** About 70% of buzzes are
  the wearer's *own* loud windows. Option A is necessary — a buzz should never
  be about someone else — but it is nowhere near sufficient.
- **The valence veto removes ~10%** on real audio and *adds* buzzes on the TTS
  scenes (79 → 90), because there the model reads loud synthetic anger as
  unpleasant and un-vetoes it. Correct behaviour, small effect.
- **Both together reach 64/h — still five times the 12/h target.** No
  combination of the existing knobs gets close. The remaining dose is the
  wearer, genuinely louder than their own baseline, in conversations where
  nobody is angry. That is not a gating problem; it is the +6 dB rung
  measuring animation, not heat. Raising it to +10 dB (51/h) buys the most of
  any single knob, at the cost of the recall the CREMA-D rubric already
  measured (61% of angry clips at +10 vs 84% at +6).

So the honest conclusion of round 1b: **the dose target is unreachable by
tuning the loudness lane. Something other than loudness has to decide when a
loud moment is worth a buzz** — words, valence with a stricter bar, or a
learned model — and the corpus below is what it gets calibrated on.

## Round 2 — diverse real conversation

_(filled in as corpora land — see below)_

Sources, chosen by three parallel researchers who verified licences and live
download links rather than working from memory:

| corpus | what it adds | heat truth | speaker truth | licence |
| --- | --- | --- | --- | --- |
| **CONFER** | 120 clips of real televised debates — people genuinely arguing | continuous, 10 raters, 25 fps | none | research + citation |
| **MELD** | 1,400 multi-party dialogues, per-line anger/joy/sadness | per utterance (acted sitcom) | named speakers | GPL-3 repo / TV footage |
| **SBCSAE** | ~60 everyday American conversations — dinners, arguments, calls | none (tone model) | per-speaker CHAT timestamps | CC BY-ND 3.0 |
| **CHiME-6** | a real dinner party at home, per-person headsets | none (tone model) | headset channels — free overlap truth | CC BY-SA 4.0 |
| **VoxConverse** | broadcast debates and panels | none (tone model) | official RTTM | CC BY 4.0 |

Deliberately not pursued: anything requiring YouTube scraping (ToS), the
crisis-line and clinical therapy corpora (real patients, correctly locked
down), and the couples-therapy corpus (IRB + identifiable voices; worth an
email from the owner, not a script). Two strong candidates need a signature
from the owner rather than a download: **MSP-Conversation** (77 h of real
podcast conversation with continuous arousal, free academic licence) and
**VAM** (real German talk-show arguments, ELRA fee).
