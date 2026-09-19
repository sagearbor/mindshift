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

---

## Feature bench, round 1 — what signal, not what threshold (2026-09-20)

The owner's correction: loudness was only ever an example. What actually
separates anger from everything else — and from *happy*, which is the failure
that matters — is an empirical question with a literature. Two researchers
surveyed it; the headline is that the energy/pitch blind spot for angry-vs-happy
is **published and structural** (the discriminating information is valence),
and that SSL embeddings lose ~0.5% cross-corpus where eGeMAPS-style features
lose ~18% (Pepino et al. 2021).

`scripts/feature_bank.py` extracts every candidate once per clip (8,882 clips,
115 speakers); `scripts/feature_bench.py` compares any combination under
speaker-grouped CV and, the test that matters, **cross-corpus** (train on
CREMA-D's 91 speakers, test on RAVDESS's 24 — different rooms, scripts, and
label protocols).

| signal | in-corpus angry-vs-happy | **cross-corpus angry-vs-happy** | recall @ 5% false alarms |
| --- | --- | --- | --- |
| loudness over own baseline (shipped) | 0.797 | **0.733** | 0.33 |
| our prosody features | 0.815 | 0.708 | 0.30 |
| **eGeMAPS** (88 openSMILE functionals, 24 ms/clip) | 0.854 | **0.832**¹ | **0.51** |
| eGeMAPS + our prosody | 0.875 | 0.773 | 0.42 |

Two things worth more than the numbers:

- **eGeMAPS barely degrades across corpora** (0.854 → 0.832). It is measuring
  something about the voice, not about the recording. And at 24 ms per clip it
  is plausibly on-device.
- **Adding our own prosody features makes it worse cross-corpus** (0.832 →
  0.773). They carry recording-setup information — absolute level, clip
  shape — and the model learns it. This is the activation-v1 lesson again, on
  a different feature set.

Neural groups (WavLM arousal/valence/dominance, wav2vec2-base-superb-er,
SpeechBrain's IEMOCAP model, emotion2vec+) are extracted next; they are the
candidates the literature expects to win on valence.

**Two negative results worth as much as the positive one** (same bench,
cross-corpus angry-vs-happy AUC):

| variant | eGeMAPS | eGeMAPS + prosody |
| --- | --- | --- |
| linear probe (above) | **0.832** | 0.773 |
| gradient-boosted trees | 0.785 | 0.820 |
| per-speaker normalised (z-score vs own neutral clips) | 0.825, recall@5fa 0.51 → **0.29** | — |

- **A nonlinear model generalises *worse*.** Given room to fit, it fits the
  corpus. The linear probe on eGeMAPS is the best cross-corpus number on the
  board and the simplest thing on it.
- **Normalising against the speaker's own baseline generalises *worse* too.**
  This is the shipped philosophy — "dB over *your* baseline" — applied to
  all 88 features, and it costs 4 points of AUC and half the recall. The
  acoustics that mark anger (spectral tilt, harmonic structure, the shape of
  the energy envelope) are largely *absolute* properties of an angry voice,
  not deviations from that person's calm one. Loudness needed a baseline
  because loudness is the one feature that is mostly about the microphone;
  the rest do not, and subtracting the baseline throws away signal.

Implication for the product: the next detector should be eGeMAPS-class
features into a *linear* model, *not* re-referenced to the wearer — and the
per-person baseline kept only for the one feature that needs it.

## Round 2 — 98 recordings: everyday conversation and real debates

Corpora added: **SBCSAE** (60 real everyday American conversations, 23.3 h —
dinners, phone calls, arguments, a town meeting; per-speaker CHAT timestamps)
and **CONFER** (19 real televised Greek debates salvaged from the truncated
folds, 24 min, ten raters' continuous conflict intensity). CHiME-6 lands in
round 3.

Dose per hour, duration-weighted, shipped chain and each candidate rule:

| corpus | n | shipped | identity | valence | both | **hold 3 s** | **hold 5 s** | rising | hold+rise | overlap | id+hold+ovlp | rung +10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AMI meetings | 12 | 88.8 | 71.9 | 79.1 | 63.7 | 38.3 | **16.7** | 77.7 | 37.5 | 83.8 | 30.9 | 51.5 |
| SBCSAE everyday | 60 | 83.8 | 79.7 | 78.6 | 74.2 | 43.5 | **24.7** | 78.8 | 44.5 | 76.9 | 37.6 | 49.7 |
| CONFER debates | 19 | **2.5** | — | — | — | 0 | 0 | 0 | 0 | — | — | 0 |
| owner recordings | 2 | 360.8 | 181.4 | 241.2 | 121.6 | 180.4 | 180.4 | 300.0 | 180.4 | 121.6 | 60.8 | 240.2 |

Three findings:

1. **Hysteresis is the single biggest lever, by far.** Requiring the first
   rung to *hold* for 5 s before a buzz takes ordinary conversation from
   84–89/h to 17–25/h. Identity, valence and a rising-trend rule each remove
   10–30%; holding for five seconds removes 70–80%. This is the alarm-design
   literature's prediction (instantaneous threshold crossing → mostly noise)
   confirmed on 27 hours of real talk. It costs nothing in hardware and no
   model. Still not 12/h — but the first knob that gets within reach of it.
2. **On real everyday conversation, identity buys less than on meetings**
   (84 → 80). SBCSAE is mostly two- and three-person talk where the wearer
   speaks half the time; the buzzes really are theirs. Confirms round 1b:
   Option A is about *correctness*, not dose.
3. **The ladder is nearly blind to CONFER — 2.5 buzzes/hour on genuine
   arguments.** The people in these clips are rated as in conflict by ten
   annotators, and the wrist would almost never buzz. Loudness volatility on
   these clips is 1.7 dB against 5 dB elsewhere: broadcast audio is
   level-controlled, so the dynamics the ladder depends on are compressed
   away. Any phone call routed through a carrier codec, any earbud with
   automatic gain, has the same property. **Loudness over-fires on calm
   conversation and under-fires on level-controlled heated conversation** —
   wrong in both directions, for the same reason: it measures the audio
   chain as much as the person.

Round 3 adds CHiME-6, the CONFER human-rating agreement (tone-model scoring
in progress at the time of writing), and the neural feature groups.

**Confirmed, not inferred:** CONFER's audio has a peak-to-RMS (crest factor)
of **12.7 dB** against **23.2 dB** for SBCSAE's home recordings — the
signature of broadcast limiting — only 0.1% of its windows clear the +6 dB
rung (19–20% elsewhere), and the shipped ladder catches **0 of 25** spans that
ten human raters marked as heated. Pinned by
`test_the_ladder_is_blind_to_level_controlled_arguments` so that fixing it
fails the test and forces this write-up to move.

### The human-rated result (CONFER, 19 clips)

The one corpus where people, not a model, said how heated each second was:

| signal | Spearman vs the ten raters' conflict intensity (per clip) |
| --- | --- |
| our loudness-heat (dB over own baseline) | **0.00** (median +0.03, n=19) |
| the tone model's arousal (WavLM, `tone_id`) | **+0.55** (median +0.56, n=11 clips with ≥10 scored windows) |

Two conclusions. **Loudness carries no information about human-rated
conflict on real arguments** — not weak, none. And **the tone-model
reference is now validated against people** (+0.55 on level-controlled
broadcast audio it was never tuned for), so its disagreement with loudness on
the other 79 recordings means what it looks like it means.

This closes the question the owner opened: the thing that should decide
whether a loud moment is heated is not loudness. The candidates are now
measured on the bench.


¹ *Correction:* an earlier draft of this section reported eGeMAPS at 0.869
cross-corpus. That run happened while the eGeMAPS extraction was still in
progress and scored the partially filled bank; on all 8,882 clips the number
is 0.832. The ranking and every conclusion stand; the size of the gap to
loudness is 10 points, not 14.

## Feature bench, final — the neural groups (2026-09-20, early morning)

All groups scored on the **same 2,940 clips** (the stratified subset the
WavLM group was extracted for — 250 per corpus × emotion), so this is
apples-to-apples. Cross-corpus = train on CREMA-D's speakers, test on
RAVDESS's; angry-vs-**happy** is the column that matters.

| signal | in-corpus angry-vs-happy | **cross-corpus angry-vs-happy** | recall @ 5% FA |
| --- | --- | --- | --- |
| loudness over own baseline (shipped) | 0.792 | 0.733 | 0.33 |
| eGeMAPS (88 hand-crafted) | 0.840 | 0.829 | 0.43 |
| wav2vec2-base-superb-er (4-class logits) | 0.772 | 0.809 | 0.45 |
| SpeechBrain wav2vec2-IEMOCAP (4 probs + 64-d embedding) | 0.884 | 0.867 | 0.50 |
| **WavLM arousal / valence / dominance — three numbers** (`tone_id`, MIT) | 0.844 | **0.897** | 0.59 |
| eGeMAPS + WavLM dims | 0.880 | 0.903 | 0.60 |
| WavLM dims + wav2vec2-er | 0.876 | 0.928 | 0.63 |
| **WavLM dims + SpeechBrain IEMOCAP** | 0.909 | **0.941** | **0.66** |

What it says:

- **Three dimensional numbers beat every 4-class emotion classifier and every
  hand-crafted set on their own** — and beat them *more* cross-corpus than
  in-corpus, the opposite of loudness. This is the literature's prediction
  (the missing axis is valence) landing exactly. The model producing them is
  already in the codebase, permissively licensed, and — per the CONFER
  result — the only signal here validated against human conflict ratings.
- **Stacking a categorical SER model on top adds ~4 points and doubles
  recall at 5% false alarms** versus what ships (0.33 → 0.66). Both models
  are Apache/MIT and CPU-runnable; neither is watch-runnable, which is the
  latency trade that keeps the instant tier a separate question.
- **eGeMAPS is the best thing that could plausibly run on a watch** (0.83,
  24 ms/clip), ten points above loudness. Wav2Small (72K params) remains the
  candidate to distil toward if its weights are obtainable.

Every number here is from `scripts/feature_bench.py --restrict-to tone`, on
`tmp/feature-bank/*.parquet`; the bank rebuilds with `scripts/feature_bank.py`.

## Round 3 — a dinner party at home (CHiME-6), and the final graph

**121 recordings**, 55 hours of real conversation across five settings.
CHiME-6 adds two real dinner parties (four friends cooking and eating, 4.5 h,
eighteen 15-minute segments) recorded on each person's own binaural headset —
the home counterpart to AMI's office, with the same free speaker truth.

| dinner party (CHiME-6) | buzzes / hour |
| --- | --- |
| shipped | 92.1 |
| perfect identity | 92.1 |
| valence veto | 86.7 |
| **hold 5 s** | **55.3** |
| overlap (ground-truth ceiling) | 68.1 |
| identity + hold 3 s + overlap | **38.1** |
| rung +10 dB | 77.1 |

What the dinner party adds that the other corpora could not:

- **It is the hardest setting for the loudness ladder, and the one closest to
  the product's use.** Four people at a table talk over each other, laugh and
  get animated for two hours straight. The tone model calls **21% of the loud
  windows pleasant** — three times the meetings' 7% — and the ladder's dose is
  the highest of any real corpus.
- **The identity numbers for CHiME-6 are not trustworthy — do not read
  them.** The gate "removed nothing" (92.1 → 92.1) because the headset-derived
  speaker labels credit one participant with ~85% of all speech (297 s vs 37,
  3 and 1 s in the first segment). CHiME-6's binaural headsets are not
  gain-matched the way AMI's are, so the "loudest headset wins" bleed test
  hands nearly every frame to the loudest microphone. AMI's labels passed the
  8.2%-overlap sanity check; these would not. The dose, valence and hysteresis
  columns for CHiME-6 do not depend on speaker labels and stand; the
  identity and overlap columns for it should be regenerated from the
  transcript JSONs (a separate CHiME-6 download) before anyone cites them.
- **Hysteresis helps less at a dinner party** (92 → 55) than at a meeting
  (89 → 17) or everyday talk (84 → 25), because sustained animation is the
  normal state of a dinner party, not an event. The combination that gets
  furthest on the columns that ARE trustworthy here is hold-5s at 55/h — still four times the target. (The 38/h identity+hold+overlap figure rests on the bad labels above.)

Across all 121 recordings, dose still tracks loudness volatility at Spearman
**+0.88** — the ladder buzzes in proportion to how lively a conversation is,
whoever is in it and whatever they feel. On every real recording the tone
model and loudness agree only moderately (median +0.50); on the one corpus
with human ratings, loudness agrees with the humans at 0.00 and the tone model
at +0.55.

That is where the three rounds end: **no combination of the existing knobs
reaches the 12/h target on any real setting, and the two signals the
literature said would work — a dimensional tone model, and hysteresis on
whatever fires — are the two that measured best here.** Both are now in the
bench and the graph with numbers a decision can be made on.
