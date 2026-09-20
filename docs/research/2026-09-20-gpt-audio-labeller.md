# Benchmarking OpenAI's gpt-audio as a "how heated is this moment" labeller

*2026-09-20. Answers §7 of `docs/research/2026-09-20-tone-model-options.md`
("OpenAI gpt-audio as a labeller") now that the $50 credit top-up landed.
Every number below was measured tonight; nothing is estimated except the
final cost projection, which is a straight-line extrapolation and is marked
as such.*

## Verdict, up front

**gpt-audio-mini is a usable offline labeller for distillation; the full
`gpt-audio` model is not, despite scoring higher on the windows it actually
answers — its ~24% parse/refusal rate on real heated speech makes it
unreliable as a bulk labeller.** Neither is a candidate for the live path
(cost per call, 3rd-party network round trip, and the full model's
unreliability rule that out regardless of latency). As a **teacher for
distillation** (the question that actually matters — training a small
on-device model), gpt-audio-mini beats today's shipped signal (loudness) by
a wide margin and is roughly comparable to the WavLM tone model already in
the codebase, at a fraction of WavLM's model size but with a real per-call
dollar cost this doesn't have.

## Setup

- **Corpus:** CONFER — 24 real televised Greek political debates, ten human
  annotators' continuous 0–1000 conflict-intensity rating, resampled to
  1 Hz (`scripts/confer_corpus.py`).
- **Windows:** the SAME 5 s windows already scored by the cached WavLM
  reference (`tmp/heat-map/reference_series/confer_*.json`'s `window_idx`),
  so gpt-audio, WavLM, and the human raters line up on identical audio.
  345 windows total across the 24 clips (voiced-only, up to 120/clip,
  subsampled by `scripts/heat_reference.py`).
- **Models:** `gpt-audio-mini` and `gpt-audio` (full). `gpt-audio-1.5` was
  available but not run — it is priced identically to `gpt-audio` per
  `developers.openai.com/api/docs/pricing` (verified 2026-09-20), so it adds
  no new cost/quality point over the full model already benched.
- **Endpoint — deviation from plan:** the task asked for the Responses API
  (`POST /v1/responses`, `input_audio` content parts). Live-tested with all
  three models: every one returned
  `HTTP 400 {"message": "Audio input is not available.", ...}`. This is an
  account/endpoint-level gate, not a model limitation — `POST
  /v1/chat/completions` with `modalities: ["text"]` (the same endpoint
  `scripts/audio_tone_probe.py` already uses successfully elsewhere in this
  repo) works perfectly and returns full token-level usage detail
  (`prompt_tokens_details.audio_tokens`), which is what actual cost is
  computed from below. `scripts/gpt_audio_heat.py` uses that endpoint.

### Prompt (verbatim, CONFER)

```
SYSTEM:
You are an acoustic tone analyzer. The speech in this clip may be in a
language you do not understand (Greek). You MUST ignore word meaning
entirely — judge ONLY from tone of voice: pitch, loudness, pace, tremor,
harshness, overlapping voices. Do not try to translate or guess the words.

Return ONLY strict JSON, no markdown fences, no other text:
{"heat": 0-100, "arousal": 0-100, "valence": -100..100,
 "angry_vs_excited": -100..100, "one_line_reason": "short phrase"}

heat = how heated/conflictual this moment sounds (0 calm, 100 an extremely
heated argument). arousal = vocal energy/intensity (0 low, 100 high).
valence = how negative vs positive the tone sounds (-100 very
hostile/negative, +100 very warm/positive). angry_vs_excited disambiguates
high-arousal moments that loudness alone confuses: -100 clearly
angry/hostile, +100 clearly excited/joyful/enthusiastic, 0 ambiguous or
neutral.

USER: Rate this audio clip's vocal tone only, ignoring words/language.
Judge purely from how it sounds.
[+ input_audio, 5s wav, 16kHz mono]
```

A second, CREMA-D-specific prompt was used for the sanity check below — see
"A live bug found and worked around" for why.

## The agreement table

Protocol replicated exactly from `scripts/heat_map.py`'s
`human_conflict_agreement()`: **within-clip** = median of the per-clip
Spearman correlation (one number per clip with ≥10 scored windows);
**across-clip** = Spearman of the clip-level means (mean signal vs mean
human conflict, one point per clip, all 24 clips). This reproduces the
documented WavLM (+0.51 / +0.85) and loudness (0.05 / −0.41) numbers exactly
from `tmp/heat-map/heat_map.json`, so the new rows are directly comparable.

| signal | within-clip (median Spearman) | across-clip (Spearman of clip means) |
| --- | --- | --- |
| loudness (dB over own baseline) | 0.049 | −0.417 |
| WavLM arousal (`tone_id`, already shipped dark) | 0.51 | 0.85 |
| **gpt-audio-mini heat** | **0.277** (n=13 clips) | **0.798** |
| gpt-audio-mini arousal | 0.401 | 0.822 |
| **gpt-audio (full) heat** | **0.319** (n=10 clips) | **0.91** |
| gpt-audio (full) arousal | 0.319 | 0.913 |

Heated-window recall / false-positive rate, on the same 5 s grid (human
mean ≥400 of 1000 = "heated"; gpt heat ≥50 = "called heated"):

| model | heated-hit-rate (human ≥400 → heat ≥50) | false-positive-rate (human <200 → heat ≥50) |
| --- | --- | --- |
| gpt-audio-mini | 0.612 (n=67 windows) | 0.344 (n=215) |
| gpt-audio (full) | **1.00** (n=51) | 0.400 (n=165) |

Read the full model's 1.00 hit-rate with the reliability caveat below in
mind: it is measured only on the 76% of windows it actually answered, and
those windows were not randomly selected.

### What this says

- **Both gpt-audio models beat loudness by a wide margin** and are in the
  same neighbourhood as WavLM (mini slightly below, full slightly above,
  on across-clip agreement — the cleaner of the two numbers since it
  isn't sensitive to a single noisy clip the way a 10-13-clip median is).
- **The full model is NOT simply "better"** — its higher agreement comes
  with a much higher false-positive rate too (0.40 vs 0.34) and, critically,
  a much higher failure rate (next section) that likely biases which
  windows survive into that agreement number in the first place.

## The reliability finding (the one that actually matters)

Parse/refusal rate on CONFER, same 345-window run for both models:

| model | windows attempted | usable | **error rate** |
| --- | --- | --- | --- |
| gpt-audio-mini | 345 | 342 | **0.9%** |
| gpt-audio (full) | 345 | 262 | **24.1%** |

Nearly 1 in 4 windows sent to the full model came back as a short (~14–50
completion tokens), non-JSON reply rather than a rating — cheap in dollars
(still billed for the audio input tokens) but useless as data, and the
failures were **not** uniformly distributed: one clip
(`confer_20111031_seq5`) failed on ~48 of its 54 windows for the full model
while `gpt-audio-mini` scored 53 of the same 54 windows cleanly. That
clustering means the full model's surviving-window agreement numbers above
are computed on a **non-random subset** of the heated spectrum — exactly
the kind of selection effect that would make a labeller's numbers look
better than a production pipeline built on it would actually perform.
**This is the main reason gpt-audio-mini, not the full model, is the
recommended labeller** despite the full model's higher raw agreement.

### A live bug found and worked around (CREMA-D)

The first CREMA-D pass (see below) came back **0/40** usable for the full
model and only 22/40 for mini. Diagnosis: on these much shorter (~2.5 s vs
5 s) clips, `gpt-audio` (full) reliably replied *"I can analyze the vocal
tone of the audio clip for you. Please provide the audio clip..."* — i.e.
it acknowledged the text instruction but behaved as if no audio had been
attached, despite `prompt_tokens_details.audio_tokens` confirming the audio
WAS received (19 tokens, proportional to the ~2.5 s clip). Two changes
fixed it: putting the `input_audio` content part **before** the text part
in the message, and dropping the CONFER prompt's "language you do not
understand (Greek)" framing (nonsensical for English CREMA-D clips, and
plausibly part of what confused the full model). After the fix: full model
31/40, mini 20/20. The CONFER prompt/ordering was left as originally run
(that data was already collected and not re-paid-for); the CREMA-D-specific
prompt is documented in `scripts/gpt_audio_heat.py`.

## CREMA-D sanity check (20 angry / 20 happy, after the prompt fix)

40 clips picked deterministically (seed 0) from `tmp/corpora/cremad`. AUC =
P(a random angry clip scores higher than a random happy clip); 0.5 = chance.

| model | n scored | AUC(heat) | AUC(angry_vs_excited) |
| --- | --- | --- | --- |
| gpt-audio-mini | 20/20 | **0.365** | 0.44 |
| gpt-audio (full) | 31/40 | 0.581 | **0.70** |

Both are weaker than expected for a task usually considered easy (angry vs
happy). `angry_vs_excited` — the field designed specifically to disambiguate
high-arousal-but-not-hostile speech from anger — does discriminate for the
full model (0.70) but not for mini (0.44, worse than the plain `heat` score
in the other direction). Two honest caveats: n=20/model is small (wide
confidence intervals — do not read 0.365 as a precise number), and CREMA-D's
"acted" emotion is a different task than CONFER's real-argument tone, so
this is a sanity check, not a confirmation, exactly as scoped. **It does not
change the verdict** — the number that matters for this app is agreement
with real heated-conversation ratings (CONFER), where mini scores 0.798/0.85
across-clip, not short acted-clip angry-vs-happy discrimination.

## Cost, latency, and the projection to 22,000 windows

Measured, not estimated — from `prompt_tokens_details`/`completion_tokens_
details` on every real call, priced at OpenAI's published per-1M-token
rates (verified 2026-09-20, `developers.openai.com/api/docs/pricing`):

| model | audio-in | text-in | text-out |
| --- | --- | --- | --- |
| gpt-audio-mini | $10.00 | $0.60 | $2.40 |
| gpt-audio | $32.00 | $2.50 | $10.00 |

| model | $/5s window (measured avg, incl. errors) | median latency/window |
| --- | --- | --- |
| gpt-audio-mini | **$0.00077** | 0.71 s |
| gpt-audio | $0.00268 | 0.86 s |

**Total spent tonight: $1.39** (860 API calls: 345 CONFER × 2 models, the
CREMA-D failed-then-fixed passes, and 10 latency-only calls) — well under
the $20 budget, and under the $4 mini-only checkpoint that gated running
the full model at all.

**Projected cost to label all 22,320 windows of the app's 31 h corpus**
(31 h ÷ 5 s = 22,320 windows; straight-line from the measured per-window
average, **not** netted for silence or voiced-only subsampling — the actual
production run would cost less):

| model | projected cost |
| --- | --- |
| gpt-audio-mini | **≈ $17.22** |
| gpt-audio (full) | ≈ $59.82 |

## Is gpt-audio a better teacher than WavLM for distillation?

**For gpt-audio-mini: roughly comparable, at a real per-call dollar cost
WavLM doesn't have, but with two practical advantages WavLM lacks —**
it needs no 1.27 GB model resident anywhere, and its labels come with a
free-text `one_line_reason` that's useful for spot-checking a distilled
model's training set by hand. Its across-clip CONFER agreement (0.798)
trails WavLM's (0.85) by a modest amount; its within-clip median (0.277) is
noticeably below WavLM's (0.51), though both are computed on small clip
counts (13 vs some larger n for WavLM) and noisy. **For the full
`gpt-audio` model: no** — despite the highest raw agreement (0.91
across-clip), its 24% failure rate on real heated speech, clustered on
specific clips, makes its output an unreliable teacher signal without a lot
more engineering (retries, better prompting per the CREMA-D fix, or
accepting a smaller labelled set).

## Is it usable as an offline labeller for the 31 h corpus?

**Yes, gpt-audio-mini, at ≈$17 for the full corpus, in well under
gpt-audio-mini's ~5 hours of wall-clock time at ~0.7 s/window** (22,320 ×
0.71 s ≈ 4.4 h if run serially; trivially parallelizable since each window
is an independent call). That is a one-time cost to build a distillation
training set, not a recurring one, and it's affordable at this project's
scale. **No, not the full `gpt-audio` model** as currently prompted — the
24% failure rate would need to be driven down (better prompting, retry
logic, or accepting the CREMA-D-style reordering fix across the board and
re-measuring) before it's worth 3.5× the cost of mini for a labeller whose
extra agreement may partly be a selection artifact of what it refuses to
answer.

## Files

- `scripts/gpt_audio_heat.py` — the bench harness (CONFER + CREMA-D stages,
  spend ledger, analysis/report).
- `tmp/heat-map/gpt_audio/` (main repo, gitignored) — every raw per-window
  response, cached; `spend_ledger.json`; `summary.json`.
- `server/tests/fixtures/gpt_audio_bench_summary.json` — the small (<5 KB)
  aggregated summary, committed, read by the pytest gate below.
- `server/tests/test_gpt_audio_bench.py` — pins the measured agreement
  numbers (floor, 0.05 below measured) and the full model's error rate
  (ceiling, ceiling above measured) as a regression gate.
