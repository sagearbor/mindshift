# Audio tone eval — wav2vec2-IEMOCAP on our fixtures (2026-08-24)

Produced by `scripts/tone_eval.py` (numbers in the JSON sidecar next to this file). Model: `speechbrain/emotion-recognition-wav2vec2-IEMOCAP@117a9c3dff08be81a3628eecf6a66b547ec1659b`, backbone `facebook/wav2vec2-base@0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8`, CPU only.

## Headline

| metric | gptaudio | openai | **combined** |
|---|---|---|---|
| 4-class accuracy (accepted set) | 40% | 40% | **40%** |
| 4-class accuracy (primary only) | 40% | 40% | **40%** |
| arousal accuracy (angry vs not) | 50% | 50% | **50%** |
| angry precision | 40% | 40% | **40%** |
| angry recall | 50% | 50% | **50%** |

- Per-slice CPU latency (32 slices, 5–9 s each): mean 184 ms, median 205 ms, max 292 ms — realtime factor mean 0.030, max 0.041 (1.0 = as slow as the audio).
- Cold model load (first call in a fresh process, snapshots already on disk): 1.5 s.
- Model size on disk: `wav2vec2.ckpt` 378 MB, `model.ckpt` 13 KB, `wav2vec2-base/pytorch_model.bin` 380 MB — 758 MB total (the base backbone weights are fully overwritten by the fine-tune at load; they are on disk only because the recipe loader insists).
- Machine: arm64 / macOS-15.6-arm64-arm-64bit / python 3.12.14 / torch 2.13.0 (4 threads).

**Decision: `MINDSHIFT_TONE_AUDIO` default = `dark`.** Combined 4-class accuracy 40% is under the ~60% bar on our own scripted ground truth, so per the owner's rule it ships dark: computed and logged on every analysis, never surfaced, until a better model or a calibration on real couple audio lifts it.

## Label mapping

`scripted_emotion` → accepted IEMOCAP classes (first = primary, used for the confusion matrix and for arousal truth). Reasoning per line is in the script docstring.

| scripted | accepted |
|---|---|
| calm_open | neutral |
| calm_guarded | neutral |
| calm_close | neutral / happy |
| repair_hopeful | neutral / happy |
| tense_rising | angry |
| defensive_rising | angry |
| shout_angry | angry |
| cold_contempt | angry |
| hurt_sad | sad |
| scared_shaky | sad |

## gptaudio (`test_recording_gptaudio.wav`, gpt-audio-1.5)

Strict 40% · primary 40% · arousal 50% on 10 turns.

| # | spk | scripted | accepted | predicted | conf | neu / ang / hap / sad | s | ms | ok |
|---|-----|----------|----------|-----------|------|-----------------------|---|----|----|
| 0 | A | calm_open | neutral | angry | 1.00 | neu 0.00 / ang 1.00 / hap 0.00 / sad 0.00 | 7.2 | 256 | **no** |
| 1 | B | calm_guarded | neutral | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 6.0 | 232 | yes |
| 2 | A | tense_rising | angry | angry | 1.00 | neu 0.00 / ang 1.00 / hap 0.00 / sad 0.00 | 6.8 | 213 | yes |
| 3 | B | defensive_rising | angry | neutral | 0.99 | neu 0.99 / ang 0.01 / hap 0.00 / sad 0.00 | 6.4 | 221 | **no** |
| 4 | A | shout_angry | angry | angry | 1.00 | neu 0.00 / ang 1.00 / hap 0.00 / sad 0.00 | 7.2 | 184 | yes |
| 5 | B | cold_contempt | angry | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 5.8 | 161 | **no** |
| 6 | A | hurt_sad | sad | angry | 0.97 | neu 0.00 / ang 0.97 / hap 0.00 / sad 0.03 | 7.8 | 196 | **no** |
| 7 | B | scared_shaky | sad | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 8.6 | 205 | **no** |
| 8 | A | repair_hopeful | neutral/happy | angry | 1.00 | neu 0.00 / ang 1.00 / hap 0.00 / sad 0.00 | 7.9 | 273 | **no** |
| 9 | B | calm_close | neutral/happy | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 7.2 | 272 | yes |

Confusion (primary truth):

| truth \ pred | neutral | angry | happy | sad |
|---|---|---|---|---|
| **neutral** | 2 | 2 | 0 | 0 |
| **angry** | 2 | 2 | 0 | 0 |
| **happy** | 0 | 0 | 0 | 0 |
| **sad** | 1 | 1 | 0 | 0 |

Gain invariance (same turns, level reduced before classification):

| gain | strict | arousal | labels changed vs 0 dB |
|---|---|---|---|
| -6 dB | 40% | 50% | 0/10 |
| -20 dB | 40% | 50% | 0/10 |

## openai (`test_recording_openai.wav`, gpt-4o-mini-tts-2025-12-15)

Strict 40% · primary 40% · arousal 50% on 10 turns.

| # | spk | scripted | accepted | predicted | conf | neu / ang / hap / sad | s | ms | ok |
|---|-----|----------|----------|-----------|------|-----------------------|---|----|----|
| 0 | A | calm_open | neutral | angry | 1.00 | neu 0.00 / ang 1.00 / hap 0.00 / sad 0.00 | 7.2 | 279 | **no** |
| 1 | B | calm_guarded | neutral | neutral | 0.59 | neu 0.59 / ang 0.41 / hap 0.00 / sad 0.00 | 5.5 | 223 | yes |
| 2 | A | tense_rising | angry | angry | 1.00 | neu 0.00 / ang 1.00 / hap 0.00 / sad 0.00 | 6.0 | 246 | yes |
| 3 | B | defensive_rising | angry | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 7.2 | 264 | **no** |
| 4 | A | shout_angry | angry | angry | 1.00 | neu 0.00 / ang 1.00 / hap 0.00 / sad 0.00 | 7.1 | 258 | yes |
| 5 | B | cold_contempt | angry | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 5.5 | 206 | **no** |
| 6 | A | hurt_sad | sad | angry | 0.69 | neu 0.00 / ang 0.69 / hap 0.31 / sad 0.01 | 7.1 | 257 | **no** |
| 7 | B | scared_shaky | sad | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 7.0 | 233 | **no** |
| 8 | A | repair_hopeful | neutral/happy | angry | 1.00 | neu 0.00 / ang 1.00 / hap 0.00 / sad 0.00 | 8.3 | 292 | **no** |
| 9 | B | calm_close | neutral/happy | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 5.8 | 219 | yes |

Confusion (primary truth):

| truth \ pred | neutral | angry | happy | sad |
|---|---|---|---|---|
| **neutral** | 2 | 2 | 0 | 0 |
| **angry** | 2 | 2 | 0 | 0 |
| **happy** | 0 | 0 | 0 | 0 |
| **sad** | 1 | 1 | 0 | 0 |

Gain invariance (same turns, level reduced before classification):

| gain | strict | arousal | labels changed vs 0 dB |
|---|---|---|---|
| -6 dB | 40% | 50% | 0/10 |
| -20 dB | 40% | 50% | 0/10 |

## Combined confusion (both acted fixtures, primary truth)

| truth \ pred | neutral | angry | happy | sad |
|---|---|---|---|---|
| **neutral** | 4 | 4 | 0 | 0 |
| **angry** | 4 | 4 | 0 | 0 |
| **happy** | 0 | 0 | 0 | 0 |
| **sad** | 2 | 2 | 0 | 0 |

## Real recordings (no emotion labels — distribution sanity check)

### family_real

REAL recording (not synthesized) — the project owner and his son, alternating in strict 5-second turns for ~30s.

Distribution: `{'neutral': 5, 'angry': 1, 'happy': 0, 'sad': 0, 'skipped': 2}`

| # | speaker | s | predicted | conf | neu / ang / hap / sad | ms |
|---|---|---|---|---|---|---|
| 0 | Sage | 5.1 | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 114 |
| 1 | Asher | 2.7 | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 69 |
| 2 | Asher | 0.7 | too_short |  | — | 0 |
| 3 | Sage | 4.6 | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 106 |
| 4 | Asher | 0.5 | too_short |  | — | 0 |
| 5 | Asher | 2.7 | angry | 1.00 | neu 0.00 / ang 1.00 / hap 0.00 / sad 0.00 | 71 |
| 6 | Sage | 5.3 | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 112 |
| 7 | Asher | 3.4 | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 81 |

### poker6_real

REAL recording (not synthesized) — the owner's poker night, 6 different real men each speaking for roughly 5 seconds in strict turn order (~30.12s total).

Distribution: `{'neutral': 5, 'angry': 0, 'happy': 1, 'sad': 0, 'skipped': 0}`

| # | speaker | s | predicted | conf | neu / ang / hap / sad | ms |
|---|---|---|---|---|---|---|
| 0 | Player1 | 5.0 | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 104 |
| 1 | Player2 | 5.0 | happy | 1.00 | neu 0.00 / ang 0.00 / hap 1.00 / sad 0.00 | 106 |
| 2 | Player3 | 5.0 | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 107 |
| 3 | Player4 | 5.0 | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 110 |
| 4 | Player5 | 5.0 | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 110 |
| 5 | Player6 | 5.1 | neutral | 1.00 | neu 1.00 / ang 0.00 / hap 0.00 / sad 0.00 | 107 |

## Reading the numbers

- **Strict 4-class: 40%** over 20 acted turns (primary-only 40%). Chance on 4 classes is 25%; the model card's 78.7% is on IEMOCAP's own acted-in-lab test split, so a drop on out-of-domain TTS-acted audio is expected — the question is how far.
- **Arousal (angry vs not): 50%**, angry precision 40% / recall 50% (4 of 8 angry-truth turns caught; 6 false 'angry' calls).
- Per scripted emotion (✗ = outside accepted set): calm_open: gptaudio=angry✗, openai=angry✗; calm_guarded: gptaudio=neutral, openai=neutral; tense_rising: gptaudio=angry, openai=angry; defensive_rising: gptaudio=neutral✗, openai=neutral✗; shout_angry: gptaudio=angry, openai=angry; cold_contempt: gptaudio=neutral✗, openai=neutral✗; hurt_sad: gptaudio=angry✗, openai=angry✗; scared_shaky: gptaudio=neutral✗, openai=neutral✗; repair_hopeful: gptaudio=angry✗, openai=angry✗; calm_close: gptaudio=neutral, openai=neutral.
- **Per speaker** (a real emotion reader must vary its label within a voice): gptaudio — Speaker A: angry×5 (ONE label for all its turns); Speaker B: neutral×5 (ONE label for all its turns) | openai — Speaker A: angry×5 (ONE label for all its turns); Speaker B: neutral×5 (ONE label for all its turns).
- Calibration: 18/20 acted turns are called at confidence ≥ 0.95, and 11 of those are WRONG. The softmax is saturated, not informative — a confidence floor cannot rescue a selective 'on'; the flag is all-or-nothing.
- Gain invariance: gptaudio -6 dB → 0 label(s) changed, strict 40%; gptaudio -20 dB → 0 label(s) changed, strict 40%; openai -6 dB → 0 label(s) changed, strict 40%; openai -20 dB → 0 label(s) changed, strict 40%. Labels that survive a -20 dB cut are being read from the voice (pitch/rate/quality), not the level — this is the owner's 'not just a yelling detector' check.
- Real `family_real` (calm, ordinary speech, no emotion labels): {'neutral': 5, 'angry': 1, 'happy': 0, 'sad': 0, 'skipped': 2}.
- Real `poker6_real` (calm, ordinary speech, no emotion labels): {'neutral': 5, 'angry': 0, 'happy': 1, 'sad': 0, 'skipped': 0}.
- Cost: ~205 ms median per 5–9 s turn on CPU (RTF 0.030) — cheap enough to run per utterance inside the existing asyncio.to_thread analysis path without touching realtime budgets; the one-off 1.5 s load happens once per process.
- Text-alone comparison (a judgment, not a measurement — we did not run an LLM text baseline here): a reader of the TRANSCRIPT alone gets the emotion of 8/10 scripted turns from the words (the shout is written in caps, 'I'm actually scared' names its fear, 'I'm sorry … fix this with you' names the repair, 'tired of keeping score' / 'so now I'm the villain' name the escalation). The audio model got 4/10 right on BOTH fixtures (calm_close, calm_guarded, shout_angry, tense_rising) — every one of which text already covers. The two turns where audio could add lift over text — cold_contempt (polite words, hostile delivery) and calm_guarded (the words are defensive, the delivery calm) — it called 'neutral' both times, i.e. it did not read the delivery. Net lift over text: none measurable on this data.
- Caveats on the ground truth itself: 20 turns is a small set; both fixtures are TTS *acting* (two synthetic voices), not real couples; IEMOCAP's 4 classes cannot express contempt/fear/hope, so the mapping above is doing real work. None of these caveats cut in the model's favor though — a per-voice constant label is a failure on any labeling.
- What would move it out of dark (in order of cheapness): (1) per-speaker normalization — subtract each diarized speaker's median logit over the recording so a voice's timbre bias cancels and only within-speaker CHANGE is reported (dark mode logs the raw scores needed to try this offline); (2) a model trained on naturalistic rather than acted speech with continuous arousal/valence/dominance outputs (e.g. the audeering wav2vec2 MSP-Podcast dimensional model) — arousal-as-a-number suits the coaching layer better than a 4-way label anyway; (3) a small labeled set of REAL recordings from the owner (the family/poker clips show the plumbing works on real phone audio) to re-run this exact script against.
