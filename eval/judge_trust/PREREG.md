# Judge-trust pre-registration (written 2026-11-05, BEFORE any judge/labeller result was seen)

Committed to git before the first model run; the commit time is the proof.
The only numbers known when this was written are the human yardsticks below
(computed from CREMA-D crowd votes alone, no model involved) and the existing
17-line result (kappa 0.14, Sonnet 5.5 vs gpt-5-mini, PR #215).

## Human yardsticks (inputs, not results)
CREMA-D voice-only crowd, frozen 300-clip sample (50 per crowd-plurality class, seed 20261105).
Single human rater vs plurality of the other raters, pooled over all raters (2,658 pairs):
overall kappa 0.48; per class: anger 0.62, disgust 0.38, fear 0.45, happy 0.60, neutral 0.35, sad 0.50.
Call these H_e. Natural-speech crowd kappa in the literature is ~0.36 (MSP-Podcast), so acted
CREMA-D is an EASIER test than real conversations: passing here is necessary, not sufficient.

## A. Emotion labellers (CREMA-D, audio-only condition is the primary condition)
Per labeller x emotion e (one-vs-rest vs the crowd plurality, 95% bootstrap CI over clips, B=2000):
* TRUSTWORTHY: kappa >= H_e AND lower CI >= 0.75*H_e AND precision >= 0.60.
* USABLE WITH CARE: kappa >= 0.75*H_e AND lower CI >= 0.50*H_e.
* NOT TRUSTWORTHY: otherwise.
Overall: 6-class Cohen kappa vs the crowd plurality, compared with 0.48.
Soft labels: mean total-variation distance (TVD) between the labeller's probability vector and
the crowd vote distribution. Reference = mean TVD between two random halves of the crowd on the same
clips (split-half). Soft-label PASS if labeller mean TVD <= split-half TVD + 0.05.

Text-leakage tests (a labeller FAILS leakage if any of these fail):
1. Transcript-only (CREMA-D sentences carry no emotion words): 6-class kappa <= 0.10.
2. Audio + a lexically CONTRADICTORY transcript (strong emotion words of a different emotion than the
   crowd label): share of clips whose label moves to the TEXT emotion (vs that labeller's audio-only
   label) <= 0.10 PASS; 0.10-0.25 WARN; > 0.25 FAIL ("reads the words").
3. Flat TTS voice reading strongly emotional sentences (macOS `say`, generated here, $0): share of
   clips labelled with the WORD emotion <= 0.20 PASS; > 0.40 FAIL.
Audio + true transcript ("both") is reported but not gated.

## B. Nudge-quality judges (rubric v4 from PR #215, binary helpful + P(helpful) + checklist)
No human labels exist for nudge lines (owner has no labelling time). Bars are agreement-based:
* A judge is ADMISSIBLE if its Cohen kappa vs the leave-one-out majority of the other judges is >= 0.40
  with lower 95% CI > 0.20, on the frozen real case set.
* Panel TRUSTWORTHY if Fleiss kappa >= 0.40 (lower CI >= 0.20). Rationale: 0.40 is above the ~0.36
  human crowd kappa on natural emotional speech, the closest human ceiling we have for a subjective
  conversational judgement.
* Synthetic cases with labels known BY CONSTRUCTION from the owner's written rules (internal speaker
  labels, invented names, scripting the wearer, praise/engagement prompts in calm talk, cue during
  silence/laughter-only context) must be graded helpful=0 in >= 90% of cases by an admissible judge.
  Positive controls (generic regulate cue in a heated context where the wearer is speaking) are
  reported only (expected helpful=1 is an owner preference, not a hard rule).
* Hypothesis H-checklist: the checklist-derived verdict (real trigger present AND no forced-zero flag)
  has higher mean pairwise kappa than the holistic `helpful` field. Reported either way.
* Minimum panel: the smallest set of judges spanning >= 2 model families whose majority (ties ->
  not helpful) agrees with the full-panel majority at kappa >= 0.80, preferring cheaper judges.
* Owner's 8 calibration cards (2026-10-11) are an anchor shown for information only (n=8, no gating).

## C. Frozen nudge case set
Held-out sessions only (landscape/splits.json "held_out"; never tuned on). Excluded: any id cited in
RUBRIC_v3/RUBRIC_v4 worked examples, the owner label cards, v3-grades.jsonl / v3dev (rubric
development), and the 17 PR #215 lines. Exact-text leakage check against those sources by SHA-256 of
the normalised (coach_line, answered_turn) pair. Hindsight (context_after) removed. Manifest stores a
SHA-256 per case and for the whole set.

## B2. Confirmation addendum (written 2026-11-05 AFTER the section-B results, BEFORE the confirmation run)
Section-B result: the pre-registered bars FAILED (Fleiss 0.13 on the real frozen set), mostly because
almost every held-out line is judged not helpful (kappa prevalence paradox) and two lenient judges
(gpt-5-mini, gemini-3.5-flash-lite) disagree with the rest. A core panel was then picked POST HOC:
{claude-haiku-5-5, gpt-6.1-terra, gemini-3.8-flash} (Fleiss 0.60 [0.39, 0.76] on real+17 lines).
Because it was chosen on the same data, it is only confirmed if, on a FRESH confirmation set
(70 held-out lines never judged before: 15 heated|wearer = all that remain, 30 heated|other, 25 calm|wearer,
seed 20261106),
its Fleiss kappa on `helpful` is >= 0.40 with lower 95% CI >= 0.20 AND its pairwise percent agreement
is >= 0.85 for every pair. Otherwise: "not confirmed". Same prompt, same models, no changes.

## D. Spend
Hard cap USD 5.00 (run cap 4.50) on estimated cost of every call, logged per call; stop on breach.
GPT/OpenAI outputs are used only as evaluation judges, never as training labels.
