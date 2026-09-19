# Tone-model options for "heated": what we have, how big, where it runs, what it costs

*2026-09-20. Answers the owner's questions after the heat-map rounds
(`docs/plans/2026-09-19-heat-map-rounds.md`). Every number here was measured
in this repo on the stated date or is cited; estimates are marked.*

## 1. The model we already ship dark

| | |
|---|---|
| Model | `3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes` — `tone_id` backend `odyssey_dim` |
| Published | June 2024, baseline for the Odyssey 2024 SER Challenge (Goncalves, Salman, Naini, Moro-Velazquez, Thebaud, Garcia, Dehak, Sisman, Busso) |
| Licence | MIT |
| Architecture | WavLM-Large encoder (24 layers, 1024-d, ~316 M params) + attentive-stats pooling + small regression head → arousal, dominance, valence in ≈0…1 |
| Training data | MSP-Podcast release 1.11 subset: **68,360 training segments from 1,405 speakers** (dev 19,815 / 454 speakers). The full 1.11 release is 151,654 utterances, **~237 h**, natural podcast speech, crowd-rated. Train split ≈ 100 h (estimate: 68k × ~5.6 s). |
| Reported quality | CCC dev: arousal 0.579, valence 0.652, dominance 0.688. Test3: 0.405 / 0.577 / 0.577. |
| Size on disk | **1.27 GB** fp32 safetensors (`server/.tone_cache/odyssey-dim/model.safetensors`, measured) |
| Where it can run | **Cloud only** today. Watch: impossible (RAM). Phone: not attempted; int8 would be ~320 MB and still heavy. |
| Production state | Cloud Run env `MINDSHIFT_TONE_AUDIO=off` (checked 2026-09-20 via gcloud). Module default is `dark` (compute + log, never act). |
| Our measurements | CONFER human-rated conflict: loudness 0.00, this model's arousal **+0.55**. Cross-corpus angry-vs-happy AUC **0.897**, recall@5 % FA 0.59 (`server/tests/test_feature_bench.py`). |

### Latency — the numbers, and what "Mac vs cloud" meant

The Mac is **not** in the production path; it is only where the benchmarks
ran. The model code runs inside the Cloud Run container (`server/tone_id.py`),
the same one that runs voiceprints. Measured 2026-09-20, this Mac (M-series
CPU), warm model:

| Clip | Threads | WavLM odyssey_dim | SpeechBrain IEMOCAP | eGeMAPS |
|---|---|---|---|---|
| 2 s | 4 | 130 ms | — | — |
| 5 s | 4 | **264 ms** | **99 ms** | 36 ms |
| 5 s | 1 | 305 ms | — | — |

Cloud Run service: 4 vCPU, 8 GiB, CPU always allocated, max 1 instance.
A Cloud Run vCPU is ~1.5–2.5× slower per thread than this Mac (estimate), so
**compute in the cloud ≈ 0.4–0.8 s per 5 s window** for WavLM alone,
≈ 0.6–1.0 s with SpeechBrain stacked. That is what "~1 s cloud" meant:
compute time there, not the round trip.

End-to-end from "someone starts shouting" to the wrist:

| stage | time |
|---|---|
| audio window the model needs before it can judge | 2–5 s (the dominant term) |
| phone → server (already streaming for STT) | ~0.1–0.3 s |
| model compute on Cloud Run | ~0.5–1 s |
| server → watch push (existing nudge socket) | ~0.1–0.3 s |
| **total** | **≈ 3–6 s** |

So the tone model is a *confirm-or-veto* judge behind the instant loudness
tap, not a replacement for it. That is exactly the shape of the PR #186
valence veto, which is why the plan extends that mechanism rather than
inventing a new one.

**Cold-start caveat (must fix before flipping to `dark`):** the weights are
not in the Docker image; `tone_id._snapshot` downloads 1.27 GB from Hugging
Face on first use into the container's ephemeral disk (which on Cloud Run
counts against the 8 GiB memory). With max-scale 1 and no min instance,
every cold start re-downloads (~30–90 s). Bake the pinned snapshot into the
`INSTALL_VOICE=1` image layer, or mount it from GCS.

## 2. Could we train a better one on our audio?

Not a better *encoder*, no; possibly a better *small* one, yes.

What we hold, all in gitignored `tmp/corpora/`:

| corpus | hours | labels | licence |
|---|---|---|---|
| CREMA-D | ~5 h (7,442 clips) | acted emotion, 91 speakers | ODbL |
| RAVDESS | ~1.5 h (1,440 clips) | acted emotion, 24 actors | CC BY-NC-SA (tmp only) |
| CONFER | 0.5 h (24 clips) | **10-rater continuous conflict**, real arguments | research only |
| AMI + SBCSAE + CHiME-6 | ~31 h | none for heat (speaker/turn truth only) | CC BY / BY-ND / BY-SA |

Against MSP-Podcast's ~100 h of naturally-spoken, crowd-rated training speech,
our ~7 h of *acted* labels cannot train a large encoder that generalises —
acted-speech models are known to collapse on natural speech (that is the
CREMA-D→RAVDESS gap in our own bench). Three things we *can* do, cheapest
first:

1. **Fine-tune only the head** of the Odyssey model on CONFER + CREMA-D
   (minutes on CPU). Risk: overfits acted speech; gate it on CONFER
   agreement, which must stay ≥ +0.55.
2. **Stack** WavLM + SpeechBrain IEMOCAP (Apache-2.0): measured AUC 0.941,
   recall 0.66 — the best number we have, no training. Adds 360 MB and
   ~100 ms.
3. **Distil a tiny student on our 31 h of real conversation** — the Wav2Small
   recipe (below). This is what "newer methods" actually buys us: not a
   better big model but a watch-sized one trained on *our* kind of audio,
   labelled by the big model instead of by humans. Teacher must be
   permissively licensed: Odyssey (MIT) + SpeechBrain (Apache) qualify; the
   audEERING `wav2vec2-large-robust-12-ft-emotion-msp-dim` teacher is
   CC BY-NC-SA and must not be in the chain.

## 3. Shrinking 316 M parameters

WavLM-Large is large because it is a general speech encoder (ASR, speaker,
emotion); the emotion head itself is tiny. Options, with the quality cost the
literature reports:

| method | size | speed-up | quality cost | notes |
|---|---|---|---|---|
| int8 dynamic quantisation | 1.27 GB → ~320 MB | ~2× | usually < 1 CCC point | one afternoon; still cloud/phone-class |
| truncate to layers 1–12 | ~half | ~2× | small for arousal; SER information peaks in the middle layers | needs head re-fit |
| swap to WavLM-Base+ (94 M) | ~380 MB | ~3.5× | a few CCC points | comparable to SpeechBrain's wav2vec2-base |
| **distil to Wav2Small (72 K)** | **120 KB** int8 ONNX, 9 MB peak RAM | 9 ms / 5 s on a Xeon core | arousal CCC 0.66 vs teacher ~0.76; **valence 0.37 vs 0.68** | arousal survives, valence mostly does not |

For MindShift the CONFER-correlated dimension is **arousal**, so a distilled
student could keep most of what the watch needs; the valence veto would stay
a cloud judgement.

### Why the Wav2Small row said "weights unverified"

Checked 2026-09-20: the Hugging Face repo `dkounadis/wav2small` contains
**only `README.md` and `.gitattributes`** — no student weights are published
(created 2024-08-16, last modified 2024-10-10). The README's code loads the
*teacher* ensemble, and the repo is licensed **CC BY-NC-SA 4.0** (non-commercial).
The paper (arXiv 2408.13920, audEERING) reports the student at arousal CCC
0.66 / valence 0.37, 0.4 GMACs, 9 ms CPU latency, trained 17 days / 50 M steps
on MSP-Podcast v1.7 + AudioSet + noise + movie audio with mixup. So the
architecture is public and reproducible, the weights are not, and the
published ones could not ship anyway. **To fill the table row we must
distil our own** (item 2.3). Budget: the paper's 17 GPU-days is for their
full recipe; a first student on our 31 h + corpora with an MIT teacher is a
few GPU-hours on a rented instance — worth a dedicated session.

## 4. Can the watch use the cloud judge the way the phone does?

Yes, technically. The watch already holds a websocket to the server
(`EpisodeWsClient`) for nudges, positives and companion heart-rate, so a
watch-audio-to-cloud lane is plumbing, not research. The reasons it is not
on today are product ones, listed so the decision is explicit:

- The watch-alone lane was designed to keep audio on the wrist (no identity,
  no valence, no upload). Streaming it is a privacy posture change — owner's call.
- Bandwidth/battery: raw 16 kHz mono is ~115 MB/h; Opus at 24 kb/s is ~11 MB/h.
  Over BT-to-phone it is cheap; over the watch's own Wi-Fi/LTE it costs battery.
- When the phone is present (every session so far), the phone already streams
  and the cloud verdict reaches the watch through the existing socket, so the
  watch gains the judge for free. Watch-alone sessions are the only gap.

Recommendation: ship the cloud judge on the relayed lane first (phone present);
add an opt-in "stream from watch when no phone" later, Opus-encoded.

## 5. Hysteresis, and whether 5 s is too long

*Hysteresis* (from thermostats): the condition to switch **on** is stricter
than the condition to stay on, so noise at the threshold cannot make the
system chatter. Today the ladder escalates on a **single 1 s window** ≥ +6 dB
over baseline — every alarm system in the literature regrets that design
(Cvach 2012: 80–99 % of instantaneous-threshold alarms are insignificant).

How it is measured: `scripts/conversation_audit.py::sustained(over, 6.0, N)`
gates the replayed chain so a window may escalate only if the +6 dB rung has
held for **N consecutive seconds**. Nothing else changes: reminders and decay
run exactly as shipped. Median buzzes/hour on the real-conversation corpora:

| corpus | shipped | hold 3 s | hold 5 s |
|---|---|---|---|
| AMI (12 meetings) | 97.0 | 46.4 | **18.4** |
| SBCSAE (60 conversations) | 80.2 | 36.0 | **11.8** |
| CHiME-6 (18 segments) | 86.3 | 92.5 | 66.0 |

The 5 s is a delay on the **first buzz of an episode only**: the wearer is not
told "you just got loud" but "you have been loud for five seconds" — and once
at level 1 the ladder continues with no extra delay. For scale, the tone
model itself needs 2–5 s of audio before it can say anything, and the
continuous-arousal literature uses 3–5 s windows. If 5 s still feels long,
3 s exists as the same knob and gets roughly half the win. Recommendation:
ship 5 s, with 3 s as the fallback if the owner finds it sluggish on the
wrist. (CHiME-6 barely moves because a dinner party is *continuously* loud;
that corpus is the argument for the tone judge, not for a longer hold.)

## 6. SpeechBrain: where, and what it costs

`speechbrain/emotion-recognition-wav2vec2-IEMOCAP` — Apache-2.0, wav2vec2-base
(94 M params, 360 MB), 4-class (neutral/angry/happy/sad). Cloud only, same
container as WavLM. Measured 99 ms per 5 s window on the Mac; ~0.2–0.3 s on
Cloud Run (estimate). Memory: fp32 WavLM + SpeechBrain ≈ 2 GB resident, fits
the 8 GiB service.

Cost: Cloud Run bills the instance while it is up, not per model. With CPU
always allocated, 4 vCPU + 8 GiB ≈ $0.09/instance-hour (list price, estimate);
the service scales to zero when idle. Per **hour of live conversation** the
two models add ~720 windows × ~1 s ≈ 12 CPU-minutes ≈ **cents**. If a min
instance is later pinned to avoid cold starts: ≈ $65/month. Adding SpeechBrain
on top of WavLM is effectively free; the cost is the 360 MB image layer.

## 7. OpenAI gpt-audio as a labeller

Verified 2026-09-20: the key in `.env` answers **HTTP 429
`credit_balance_exhausted`** ("You have no credits remaining"). The
organisation does have access to `gpt-audio`, `gpt-audio-1.5`, `gpt-audio-mini`
and `gpt-4o-transcribe-diarize`.

Cost to bench (estimate from list prices, audio input ≈ $0.06/min for
gpt-audio, ~¼ of that for gpt-audio-mini):

| set | audio | ≈ cost (gpt-audio) |
|---|---|---|
| bench set (2,940 CREMA-D + RAVDESS clips) | ~2.5 h | ~$9 |
| CONFER (human-rated, the one that matters) | 0.5 h | ~$2 |
| AMI slice for false-alarm rate | ~4 h | ~$14 |
| **total** | | **~$25**, half that with gpt-audio-mini |

A $50 top-up covers a full bench with headroom. What it would tell us: whether
a general audio LLM beats a 1.2 GB specialist on CONFER agreement (+0.55) —
useful as a *labeller* for distillation, never as the live path (2–5 s,
per-call cost, third-party audio).

## 8. Options table (revised 2026-09-20)

AUC = cross-corpus angry-vs-happy, train CREMA-D → test RAVDESS; recall at 5 %
false alarms. Latency = compute per 5 s window on this Mac (cloud ≈ 2×).

| option | size | latency | AUC / recall | CONFER | watch | phone | cloud |
|---|---|---|---|---|---|---|---|
| loudness ladder (shipped) | — | < 1 s | 0.73 / 0.33 | 0.00 | ✅ | ✅ | ✅ |
| eGeMAPS + linear | ~5 MB lib | 36 ms | 0.83 / 0.43 | untested | ⚠️ Kotlin port | ✅ | ✅ |
| Wav2Small (own distillation) | 120 KB | 9 ms | **unknown until trained**; paper arousal CCC 0.66 | — | ✅ | ✅ | ✅ |
| wav2vec2 SUPERB-ER | 360 MB | ~120 ms | 0.81 / 0.45 | — | ❌ | ⚠️ | ✅ |
| SpeechBrain IEMOCAP | 360 MB | 99 ms | 0.87 / 0.50 | — | ❌ | ⚠️ | ✅ |
| **WavLM odyssey_dim (dark)** | 1.27 GB | 264 ms | **0.90 / 0.59** | **+0.55** | ❌ | ❌ | ✅ |
| **WavLM + SpeechBrain** | 1.63 GB | ~360 ms | **0.94 / 0.66** | — | ❌ | ❌ | ✅ |
| OpenAI gpt-audio | API | 2–5 s | unbenched (credits: 0) | — | ❌ | ❌ | ✅ |

Did not help (measured): GBM instead of linear, per-speaker normalisation,
adding our prosody features to eGeMAPS (0.83 → 0.77), heart-rate fusion
(literature +2 points).

## Sources

- Model card: https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes
- Goncalves et al., Odyssey 2024 SER Challenge: https://www.isca-archive.org/odyssey_2024/goncalves24_odyssey.pdf
- MSP-Podcast corpus (1.11: 151,654 utterances, 237 h): https://www.researchgate.net/publication/395474359_The_MSP-Podcast_Corpus
- Wav2Small paper: https://arxiv.org/abs/2408.13920 · repo (README only): https://huggingface.co/dkounadis/wav2small
- Survey that framed the rounds: `docs/research/2026-09-20-ser-and-conflict-survey.md`
