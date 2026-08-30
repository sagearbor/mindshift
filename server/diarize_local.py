"""Local speaker diarization — vendor-independent "who said each utterance".

Why this exists: Deepgram's nova-3 (model 2025-07-31.0) regressed prerecorded
diarization — it merged two distinct voices into one speaker on real recordings
(verified 2026-08-05 by direct nova-2/nova-3 comparison on identical bytes).
Renting diarization means a silent vendor model swap can corrupt every
per-speaker feature downstream (talk-share, heat, report cards). This module
re-derives speaker labels ON OUR OWN COMPUTE with the same PINNED ECAPA model
that already powers voice enrollment (``speaker_id``).

Algorithm — calibrated on the real recording that exposed the regression plus
the TTS fixture (2026-08-06), extended to N-way k-selection on the real
3-person recording that exposed the forced-2 limit (2026-08-14):

1. Embed each transcript utterance that is long enough to carry voice signal.
   Per-utterance embeddings alone are NOISY — same-speaker cosine can dip
   below cross-speaker — so a plain similarity-threshold clustering
   over-fragments (it heard 4-5 "speakers" in a 2-person recording).
2. For each candidate speaker count k = 2 .. :data:`MAX_SPEAKERS_LOCAL`:
   average-linkage merge to exactly k clusters, then REFINE: embed each
   cluster's POOLED audio (pooled embeddings are what ``speaker_id``'s
   calibration table trusts: same voice ≈0.73, different ≈0.19) and reassign
   every utterance to its closest pooled centroid, a few rounds until stable.
3. VALIDATE each k before believing it: EVERY pair of pooled centroids must be
   clearly different voices (cosine ≤ :data:`MAX_POOLED_COSINE`); for k > 2
   the pair(s) the marginal split CREATED (vs the refined k-1 partition) must
   be VERY clearly different voices (≤ :data:`STRONG_SEPARATION_COSINE`) —
   measured on the real couple recording, ONE voice heard in calm + shouting
   registers forms two well-fed clusters at pooled cosine 0.359 that no
   seconds floor can reject, while the real 3-person recording's genuine
   third voice split off at 0.267 — AND the split must be ANCHORED by a half
   that is wildly unlike some other cluster (≤
   :data:`NEW_VOICE_ANCHOR_COSINE`; the TTS fixture's phantom split-pair
   0.277 is indistinguishable from the genuine 0.267, but its halves anchor
   at 0.216+ where the real child anchors at -0.017); and each cluster must
   carry enough pooled
   speech to be trustworthy — the full :data:`MIN_CLUSTER_SECONDS` normally,
   relaxed to :data:`MIN_CLUSTER_SECONDS_STRONG` only for a cluster that is
   VERY clearly distinct from every other centroid (all its pairwise cosines
   ≤ :data:`STRONG_SEPARATION_COSINE`) — a quiet third participant with one
   clean utterance is real evidence, a moderately-separated sliver is not.
   The LARGEST fully-validating k wins; a genuine monologue measures ≈0.73
   pooled self-similarity, validates at NO k, and is REJECTED — we never
   invent a speaker.

Scope + honesty:

* Speaker counts up to :data:`MAX_SPEAKERS_LOCAL` are attempted — enough for
  the recordings this app targets (family/couple conversations), not general
  N-speaker diarization. Every k tried is reported in ``k_evaluated`` so logs
  show why a count was chosen.
* Segmentation starts from the transcript's utterance boundaries, PLUS a
  word-level pass for the transcriber-welded case (two voices merged into ONE
  utterance): utterances longer than :data:`SPLIT_MIN_UTTERANCE_SECONDS` that
  carry per-word timings are scanned for a SUSTAINED voice-change point —
  windows on either side of a sliding candidate boundary are scored by their
  affinity MARGIN against the two POOLED cluster centroids from the first
  clustering pass (window-to-window cosine is useless on real speech; see the
  SPLIT_* constants), and a change requires consecutive opposite-sign margins
  clearing :data:`SPLIT_MIN_MARGIN`. The utterance is split at the nearest
  word boundary and everything is re-clustered + validated over the finer
  segments. No sustained evidence → no split.
* Requires the optional voice deps (torch + speechbrain). When they are
  missing, validation fails, or there is too little embeddable speech,
  :func:`diarize_turns` returns ``None`` and the caller keeps the transcript's
  labels.
* Pure math (merging, agreement) is torch-free; the orchestrator takes an
  injectable ``embed_fn(pcm_slice, sr)`` (and ``embed_batch_fn(chunks, sr)``
  for the window pass) so the unit suite runs without torch.

2026-08-29 — the transcript-free WINDOW PASS (voice-separation bake-off,
docs/research/2026-08-29-voice-separation/; the ``WINDOW_PASS_*`` constants
and :class:`_WindowPass`). The owner's private 3-person restaurant clip
("maggiano3", scored against his own per-second rubric) showed the shipped
pipeline "finds three voices but mixes them" (0.64 / 0.57 frame accuracy on
its two Deepgram transcripts) and, fed the rubric's own boundaries, collapsed
to k=2 (0.52): all three pooled voiceprints are ≤0.24 apart, yet average
linkage's 3-way split peeled a 1 s sliver and the duration floor rightly
rejected it. Four bake-off approaches later (pyannote, acoustic features,
sliding windows + spectral clustering, coherence), the ECAPA model and the
validation rules stay; what was added is approach B's window pass, used as
EVIDENCE, never as the final attribution (1.5 s windows in a noisy room carry
too little voice — maggiano3's window-level ceiling is 0.84):

1. 1.5 s / 0.25 s windows over the clip's SPEECH (noise-floor-relative gate,
   :func:`speaker_id.speech_mask`: max(0.003, 1.5 x p10 frame RMS); measured
   gates 0.003-0.005 on every fixture, and poker6's quietest player — median
   RMS 0.0036 — is no longer gated out) embedded in one pass, refined cosine
   affinity (Wang et al. 2018, row percentile 0.80), EIGENGAP speaker count.
   Measured: family_real 2, poker6 4 (six similar men; B's max-8 sweep said
   7), openai/gptaudio/couple 2, family3 3, meeting4 3 (D absorbed into B,
   pooled 0.32 — the documented ceiling), maggiano3 3. Never over-counted.
2. The eigengap count is a LOWER BOUND on what :func:`_select_k` TRIES: when
   a k-way linkage partition fails on a duration floor (a sliver), or k is
   the eigengap count, the SPECTRAL route (:func:`_spectral_route`: every
   turn to the nearest pooled spectral centroid, then the usual pooled
   refinement) proposes an alternative k-way partition of the same turns,
   validated by the same rules. maggiano3 on the rubric's boundaries: the
   linkage k=3 sliver (1.0 s) fails, the spectral k=3 partition validates
   (min cluster 8.1 s, marginal 0.222, anchor 0.136) → k=3, 0.52 → 0.83.
   The post-split re-selection cap is likewise lifted to the eigengap count.
   meeting4: the spectral k=3 partition is the identical 0.339 pair — no
   change (still 2/4, 0.597), as predicted by the bake-off.
3. Boundary PROPOSALS inside every utterance over
   WORD_SPLIT_MIN_UTTERANCE_SECONDS, with or without word timings: the
   utterance's windows carry the whole-clip spectral labels; smoothed label
   runs → candidate cuts (pieces ≥ one window); each cut confirmed by
   embedding its two sides against the pooled spectral centroids (different
   centroids, margins ≥ WORD_MIN_MARGIN). Union'd with the per-word cuts.
   Zero cuts on the 83 pure single-voice utterances of the seven checked-in
   fixtures (ladder + scenes pins unchanged: family_real 8/8, poker6 6/6,
   openai/gptaudio 10/10, couple 13/13, family3 15/15, meeting4 11/17).
   Costs no extra window embeddings (re-used from the whole-clip pass) and
   one short embed per candidate piece.

Result on maggiano3 (frame accuracy vs the rubric; dad-cluster purity):
rubric boundaries 0.519/k=2 → 0.833/k=3 (purity 0.47 → 0.84); Deepgram
7-utterance transcript 0.644 → 0.687 (0.76 → 0.79); 8-utterance 0.574 →
0.671 (0.59 → 0.79). The transcript variants sit below B's transcript-free
0.76 for reasons the constants forbid fixing: two welded utterances under
the 3 s scan floor (5.44-8.02 s asher→dad, 8.88-10.74 s dad→mom: 4.4 s,
12 %) whose other voice lasts under MIN_SECONDS, 7-8 % of rubric speech the
transcript never covered, and sub-second interjections ("See", "Okay",
"No.") that inherit a neighbour. Oracle attribution on OUR pieces reaches
0.83 / 0.79, so segmentation is no longer the bottleneck.

2026-08-29 ROUND 2 — attacking exactly that list (constants under
"Round 2"; nothing above changed value):

1. Scan floor 3.0 → SCAN_MIN_UTTERANCE_SECONDS 2.0 for the per-word AND
   window-proposal passes, and a per-word piece may be as short as
   CONFIRMED_PIECE_MIN_SECONDS 0.8 when the window pass's pooled spectral
   centroids independently confirm it (:meth:`_WindowPass.confirms_two_voices`;
   one source → the MIN_SECONDS floor as before). Such a piece, and every
   uncovered turn, is attributed by its OWN embedding against the final
   centroids (measured: the nearest-window spectral label — the obvious
   alternative — is right on 1 of the fixtures' 11 sub-second turns vs 4
   for the neighbour rule and 5 for the own embedding; transcript short
   turns keep the neighbour rule).
2. Speech no utterance covers becomes "(untranscribed)" turns
   (:func:`_uncovered_speech`; capped, never overlapping, labelled by own
   embedding, NEVER part of k-selection — see diarize_turns for the
   measured reason).

Measured: 7-utterance 0.687 → 0.702 (purity 0.79 → 0.80): the 5.44-8.02 s
weld now splits at the "What|do" gap (rubric 6-9 s dad 4 % → 42 %) and a
confirmed 0.89 s piece ("Hey. Settle", 22.63-23.52 s) lands on dad by its
own embedding (margin 0.31). 8-utterance 0.671 → 0.671, unchanged: against
THAT variant's round-1 centroids the per-word pass confidently labels all
ten words of the same weld one voice (margins 0.15-0.29, so a conclusive
"no split"), and a 2.58 s utterance holds 5 windows — under
WINDOW_PASS_MIN_WINDOWS (6) — so the proposal pass is structurally silent
below 2.75 s; the 8.88-10.74 s weld (1.86 s) is under the new floor on
both variants, and its "Come on" is a lone confident word ("Come" is
ambiguous at margin 0.07 and inherits the earlier confident word) that
WORD_MIN_RUN rightly rejects. Rubric boundaries 0.833 (held), every
checked-in fixture unchanged (family_real 8/8, poker6 6/6, openai/gptaudio
10/10, couple 13/13, family3 15/15, meeting4 11/17); wall time within
+0.3 s of round 1 on every fixture (the post-split re-embedding now reuses
cached turn embeddings). Uncovered turns on the transcripts: none — the
gaps Deepgram left are quiet under the gate; the rubric's "Whoa"/"See" lie
inside its utterances. The 41.9 s "No." is wrong under every rule.

Cost (this Mac, torch at 4 threads, back-to-back against the pre-change
code): the window pass embeds ~15 ms per 1.5 s window whatever the batch
size (1-15 windows per call all measure 14-18 ms/window; 16+ per call hits a
torch path 10x slower — WINDOW_PASS_BATCH), i.e. ≈ 60 ms per second of
speech at the 0.25 s hop. Whole pipeline: family_real 0.9 → 3.1 s, poker6
1.2 → 3.4 s, openai (70 s) 2.4 → 7.4 s, meeting4 (83 s) 2.8 → 9.5 s,
maggiano3 transcripts 3.5 → 7.4 s, a 4.7-minute concatenation (48 turns,
hop widened to 0.5 s by WINDOW_PASS_MAX_WINDOWS) 6.5 → 16.3 s — 2.1-3.4x.
A 1.5 s / 0.5 s grid halves the pass but was measured and rejected: the
eigengap over-counted openai (3; its lambda_2/lambda_3 and lambda_3/lambda_4
ratios are 2.82 vs 2.83 at that hop) and minted a phantom cut there, and
maggiano3's 8-utterance transcript fell back to 0.574.

2026-08-30 — "UNKNOWN" FOR UNCLAIMED SPEECH (flag MINDSHIFT_DIARIZE_UNKNOWN,
default OFF; constants under "Unclaimed speech"; measurements in
docs/research/2026-08-30-unknown-and-transcript/). The hypothesis: a word run
(or a whole embeddable turn) that sounds like NONE of the found voices — best
cosine to every pooled centroid, k-selection's and the window pass's spectral
ones, under UNCLAIMED_COSINE — was inheriting a neighbour / the least-unlike
centroid, and that is how the son's quiet lines landed on the owner. Built:
such a run (≥ WORD_MIN_RUN words, ≥ UNKNOWN_MIN_SECONDS) becomes its own
piece labelled UNKNOWN_SPEAKER ("Unknown"), excluded from every k-selection
pass, never inherited from, not a speaker (num_speakers, talk share, heat
stats / report cards, coupling and enrollment matching all skip it —
speaker_id.UNKNOWN_SPEAKER, dynamics, speaker_id.identify_speakers_multi);
the same floor is applied to every embeddable turn after the final
refinement and to self-attributed short turns.

Measured, and NOT shipped (default OFF). Calibration first (see
UNCLAIMED_COSINE): on both maggiano3 transcripts the words production gets
WRONG are claimed confidently by the wrong centroid (min best-cosine
0.10-0.13, median 0.33-0.36) — they are not unclaimed; the only words under
any floor are the OVERLAPPED chant at 28-32 s (both voices at once) where
production is right; the son's quiet "Because I wanna do my Duolingo, dad"
that motivated this is claimed at 0.74 / 0.71 since the window pass. With
the flag on at 0.12 (frame accuracy vs the rubric / dad-cluster purity /
Unknown seconds; Unknown scores as UNLABELLED = wrong): rubric boundaries
0.833 / 0.84 / 0 s (unchanged); 7-utterance transcript 0.702 → 0.491,
purity 0.796 → 0.508, 0.85 s Unknown ("I wanna go. No.", 30.04-30.89 —
mom's chant under the son); 8-utterance 0.671 → 0.592, purity 0.792 →
0.609, the same 0.85 s. At 0.15 the transcripts read 0.491 / 0.592 again
and the rubric boundaries gain 1.48 s of Unknown (0.833, purity 0.90). The
loss is not the 0.85 s itself: cutting it re-shapes the neighbouring pieces
("I'd like a solid yellow" / "one. I don't wanna go. I wanna go." / "It's
it. Yay..."), the post-split re-selection then validates a DIFFERENT
linkage k=3 whose third cluster is the 2.1 s overlapped-chant piece
(marginal 0.119, anchor 0.062 — the 2026-08-27 phantom again) and mom folds
into dad. Every checked-in fixture is unchanged with the flag on (family_real
8/8, poker6 6/6, openai / gptaudio / couple / family3 1.00, meeting4 0.597;
0 s Unknown everywhere). Ship criterion (purity +0.05 on both transcripts,
accuracy -0.03 at most, no pinned fixture gaining Unknown seconds): purity
-0.29 / -0.18, accuracy -0.21 / -0.08 — failed on the first two counts. The
feature stays behind the flag, fully tested, for a recording that actually
contains an unfound voice; on this one there is none left to find.

Same day — DOES THE TRANSCRIPT'S OWN LABELLING PUSH US WRONG? The never-
reduce guard in main.py keeps Deepgram's labels whenever it heard more
speakers than our validated k. Measured on the owner's three stored
recordings (maggiano3, poker6, family_real; GCS copies byte-/duration-
identical to the fixtures), each transcribed three times by today's
Deepgram: it heard ONE speaker in 8 of 9 runs and two (accuracy 0.384) in
the ninth; its own labels score 0.397 / 0.146 / 0.601 where ours score
0.671 / 0.467 / 0.974 and win every time (fallback path on 1-speaker
transcripts; k=3 > 2 on the one 2-speaker maggiano run). The guard never
fired and never hurt on these three; it is unchanged.

2026-08-30 — the WINDOWS ENGINE (:func:`diarize_windows_first`, the
``WINDOWS_FIRST_*`` constants; main.py's MINDSHIFT_DIARIZE_ENGINE, default
``windows``). Everything above keeps the transcript's utterances as the unit
that decides who spoke and uses the window pass as evidence. On the owner's
REAL Deepgram transcripts that ceiling is the transcript itself: Deepgram
hears ONE speaker on 8 of 9 runs and welds voices into utterances (poker:
7 utterances for 6 men, 36 % of the speech never transcribed), and the
utterance engine scored poker 0.447 (4 of 6 voices) — the owner watched
"Re-analyze with the latest engine" turn a good 7-voice result back into 4.
The windows engine runs the bake-off's approach B end to end as the
labelling: the same whole-clip window pass → spectral labels at the eigengap
k (max k 8, B's range) → mode filter → label runs → segments, then the
transcript's WORDS are regrouped by those segments (the regrouping POST
…/reanalyze-with-segments applies to the phone's timeline, one shared
implementation: :func:`regroup_transcript_by_segments`) and speech no
utterance covered becomes "(untranscribed)" turns labelled by the same
timeline. It returns None (→ the utterance engine runs as before) when the
eigengap says one voice or there is too little speech.

Measured 2026-08-30 (frame accuracy vs ground truth, score.py; windows vs
utterance engine; the windows engine's raw segment timeline in brackets):
real Deepgram transcripts — poker 0.720 / k=7 (7 clusters: one player's
turn splits at the fixture's ±1-2 s slop) vs 0.447 / k=4 [0.809];
maggiano3 7utt 0.694 vs 0.702, 8utt 0.681 vs 0.671, both k=3, dad-cluster
purity 0.775 vs 0.80 / 0.79 [0.761 — B's number, reproduced]; family
0.949 vs 0.974 [0.959]. Ground-truth boundaries — family_real 0.980 (7/8
utterances: the son's 0.5 s "And") vs 1.000, poker6 1.000 (6/6) vs 1.000,
openai / gptaudio / couple / family3 1.000 vs 1.000, meeting4 0.818 / k=3
(14/17) vs 0.597 / k=2 (11/17), maggiano3 rubric boundaries 0.865 vs 0.833.
What keeps maggiano3's transcripts under B's 0.76 is coverage: 7-8 % of
the rubric's speech has no words and sits under the speech gate or in
sub-0.4 s gaps, so no turn can carry it. Wall time at 4 torch threads, this
Mac: windows 2-6 s per fixture vs utterances 4-10 s.
"""

from __future__ import annotations

import bisect
import logging
import os

import numpy as np

import diarize_sliding_window as _dsw
import speaker_id

logger = logging.getLogger(__name__)

# Accept a k-way split only when EVERY pair of clusters' POOLED embeddings is
# at most this similar. Calibration (2026-08-06, pinned ECAPA): different
# people pooled ≈0.19-0.26; the same real voice split in half ≈0.73;
# speaker_id's table puts merged/degraded artifacts at ≈0.48-0.56. 0.45 sits
# under all observed same-voice values with margin. Env-overridable for
# recalibration.
MAX_POOLED_COSINE = float(os.getenv("MINDSHIFT_DIARIZE_MAX_POOLED_COSINE", "0.45"))

# Candidate speaker counts run k = 2 .. MAX_SPEAKERS_LOCAL (also capped by the
# number of embeddable utterances). Raised 4 -> 6 on 2026-08-21: three
# parallel experiments investigating a real 6-speaker recording confirmed
# raising this cap is safe (full regression ladder: 61 passed, 2 skipped, 1
# pre-existing test updated for the new constant, no accuracy regressions)
# and costs nothing on the app's primary 2-4-person use case, since those
# recordings never validate past their real k anyway. A higher ceiling does
# genuinely help recordings with more real, well-separated speakers and
# enough speech per person to clear MIN_CLUSTER_SECONDS_STRONG.
MAX_SPEAKERS_LOCAL = 6

# An utterance shorter than this is not embedded (too little voice signal); it
# inherits the nearest embedded utterance's cluster (nearest by midpoint).
MIN_SECONDS = 1.0

# Each cluster of an accepted split must have at least this much pooled
# speech — a "second voice" carried by one breath of audio is not evidence.
MIN_CLUSTER_SECONDS = 3.0

# "VERY clearly a different voice" — a stricter bar than the accept gate,
# used twice:
#
# 1. MARGINAL-SPLIT RULE: claiming k+1 speakers over k asserts that one of
#    k's clusters is really TWO voices; the pair that split creates must
#    measure at or below this bar, else it is one voice in two registers.
#    Calibration (2026-08-14, both real recordings, pinned ECAPA): the
#    GENUINE marginal split (the 3-person recording's third voice, a child
#    with 1.9s of solo speech) measured 0.267 against the cluster it split
#    from, while every SPURIOUS split measured 0.359+ (see historical values
#    below). Any bar in (0.267, 0.359) separates them.
# 2. EVIDENCE-FLOOR RELAXATION: a cluster whose EVERY pairwise cosine is at
#    or below this bar may carry MIN_CLUSTER_SECONDS_STRONG of speech
#    instead of the full MIN_CLUSTER_SECONDS (the real third voice above:
#    pairwise -0.017 / 0.267, only 1.9s of solo speech — real evidence).
#
# RECALIBRATED 0.30 -> 0.32 (2026-08-24) with a THIRD real recording: a real
# 6-speaker poker-night clip (server/tests/fixtures/audio/
# test_recording_poker6_real.wav) has a genuine 6th-voice marginal split at
# 0.301 — a hair over the old 0.30 bar, so the split was wrongly rejected
# and the pipeline undercounted 5 real voices instead of 6. 0.32 admits this
# (0.267, 0.301) genuine range while staying well clear of the historical
# spurious values (couple recording k=3/k=4: 0.359/0.391 — NOT a checked-in
# fixture, numbers only, unverifiable today; 3-person k=4: 0.402). Verified
# against every REAL, checked-in, currently-listenable fixture (openai,
# gptaudio, family_real, poker6) before changing — see
# docs/research/poker6-sliding-window/ for the investigation. A prior
# "0.28/0.20 phantom split" unit test and a "TTS fixture" live test that
# motivated the old 0.30 turned out to rely on tmp/test_recording.wav, a
# fixture server/tests/fixtures/audio/README.md explicitly says NOT to use
# for diarization (physics-modulated single voice, not real acted speech) —
# both were repaired to use real fixtures instead of retired outright.
#
# RECALIBRATED 0.32 -> 0.33 (2026-08-27) with a FOURTH real recording, the
# owner's 3-person family clip ("maggiano's", 42s, owner + wife + son, NOT
# checked in — private family audio; see tests/test_diarize_private.py for
# the opt-in local regression). The owner reported his son merged into HIS
# speaker on the app. Reproduced: Deepgram returns two transcript variants
# for that file run to run, and on the 7-utterance variant (two welded
# multi-voice utterances covering 27 of 42s) the genuine third voice's
# marginal split measures 0.325 — rejected by 0.32 by 0.005, so k=2 won and
# the son's turns landed in the owner's cluster (time-weighted owner-turn
# purity 0.52). At 0.33 k=3 validates (purity 0.79) and the welded first
# utterance splits owner/son at the word level. (On the 8-utterance variant
# the same split measures 0.199 and validates at either bar.)
#
# THIS IS THE CEILING OF THE SINGLE-THRESHOLD INSTRUMENT, measured, not
# assumed: family_real's one-voice-two-registers split (the son, calm vs
# shouting, 3.4s) measures 0.337 — a bar of 0.34 mints a phantom third
# speaker on the owner's own calibration fixture (7/8) — while the scene
# pack's meeting4 genuine third voice measures 0.339 (unreachable without
# breaking family_real). Genuine splits on record: 0.199/0.267/0.301/0.325/
# 0.339; spurious: 0.337/0.359/0.391/0.402. The 0.33 placement keeps the
# full real-fixture ladder green (openai, gptaudio, family_real, poker6, the
# three scenes) — but the margin on either side is now ~0.005-0.007, inside
# the +/-0.02 run-to-run variance noted below for the anchor. Do NOT nudge
# this further; the next gain needs a second discriminator (e.g. within-
# cluster coherence of a NEW cluster's pieces), not a threshold.
#
# KNOWN REMAINING FAILURE on that same recording (not fixable here): the
# child's QUIET register ("Because I wanna do my Duolingo, dad") embeds at
# cosine ~0.0 to EVERY centroid — including his own loud register ("Woah,
# dude, you're a jerk": 0.01 pooled) — while being self-consistent
# (0.33-0.53 among its own segments). The per-word smoother has no
# confident word to inherit from, so those words fall to the surrounding
# owner speech. Marking such "unclaimed" runs as their own pieces was tried
# (2026-08-27) and minted a phantom 4th cluster out of OVERLAPPED speech
# ("I don't wanna go" / "I wanna go") — the exact phantom the round-2 k cap
# in diarize_turns exists for — so it was not shipped.
STRONG_SEPARATION_COSINE = float(
    os.getenv("MINDSHIFT_DIARIZE_STRONG_SEPARATION_COSINE", "0.33")
)
MIN_CLUSTER_SECONDS_STRONG = 1.5

# ANCHOR RULE for a marginal split: the split-pair bar alone cannot separate
# a genuine new voice from a noisy same-voice split — measured split pairs
# are 0.267 (the real third voice, a child) vs 0.277 (a historical phantom
# split), a 0.010 window no honest threshold fits. What separates them
# robustly: a GENUINE new voice announces itself by being wildly unlike at
# least one established cluster (the child vs her father: -0.017), while
# BOTH halves of a phantom split sit moderately far from everything. So at
# least one half of a marginal split must have ALL its cosines to
# NON-sibling clusters at or below this anchor bar.
#
# RECALIBRATED 0.20 -> 0.24 (2026-08-24): poker6's genuine 6th-voice split
# (see STRONG_SEPARATION_COSINE above) anchors at 0.231 — worse (higher)
# than the historical phantom-split anchor this bar was built to reject
# (0.216), so 0.20 undercounted poker6 by one real voice. Investigated
# whether 0.216 is still a live threat with the CURRENT pipeline (it
# predates several since-shipped improvements: N-way k-detection, word-level
# splitting) by re-running that exact real fixture (gptaudio.wav, the source
# of the 0.216/0.238 numbers — see test_diarize_regression_ladder.py) through
# today's code: it holds at 100% accuracy at 0.24, same as it does at 0.20 —
# the improvements since 2026-08-15 already prevent that phantom split by a
# different mechanism before this bar is even reached. 0.24 was chosen with
# margin above poker6's measured 0.231 (this bar has shown +/-0.02ish
# run-to-run variance before — see the 2026-08-15 note below) while staying
# under the next real spurious value on record (0.277). Re-verified against
# every real, checked-in fixture (openai, gptaudio, family_real, poker6)
# before changing, NOT just the one poker6 case.
#
# 0.15 was the first placement; PRODUCTION taught otherwise within hours
# (2026-08-15): re-transcription draws different utterance boundaries run to
# run, and the SAME real third voice that anchored at -0.017 one night
# measured 0.173 the next morning (its clean solo line got smeared into
# adjacent speech) — k=3 was falsely rejected by 0.023. CONSEQUENCE (honest
# tradeoff, still true at 0.24): three typical ADULTS (different-people
# pooled pairs ≈0.19-0.34) may still fail to anchor and stay a 2-way split —
# the conservative failure direction, since the transcript's own diarization
# usually hears 3 adults and the never-reduce guard keeps them.
# Env-overridable.
NEW_VOICE_ANCHOR_COSINE = float(
    os.getenv("MINDSHIFT_DIARIZE_NEW_VOICE_ANCHOR_COSINE", "0.24")
)

# Pooled-centroid reassignment rounds (converges in 1-2 on calibration data).
REFINE_ROUNDS = 3

# Cap pooled audio per cluster centroid, mirroring speaker_id.MAX_POOL_SECONDS.
MAX_POOL_SECONDS = 60.0

SOURCE = "local-ecapa"

# Label for speech no found voice claims (see the UNCLAIMED_* constants).
UNKNOWN_SPEAKER = speaker_id.UNKNOWN_SPEAKER

# --- Word-level speaker-change splitting -----------------------------------
# A transcriber can weld a speaker handoff into ONE utterance; per-word
# timings let us split it at the change. Calibrated on the real recording
# (2026-08-07): comparing two short windows TO EACH OTHER is useless — on real
# speech, same-speaker 1.5-3s windows score cosine ≈0.0-0.35 against each
# other, indistinguishable from a genuine change. What separates cleanly is
# each window's affinity MARGIN against the two POOLED cluster centroids
# (margin = cos(win, c0) - cos(win, c1)): pure utterances keep both sides of
# every candidate boundary on ONE sign (measured: no flip anywhere), while a
# welded handoff shows a SUSTAINED run of opposite-sign margins (measured
# weaker-margin values ≥0.19 inside genuine runs; edge candidates ≈0.02-0.03).

# Only utterances longer than this get scanned (bounds compute; a shorter
# utterance can't yield two trustworthy sides anyway).
SPLIT_MIN_UTTERANCE_SECONDS = 5.0

# Window on each side of a candidate boundary. Below ~1.5s ECAPA embeddings
# carry too little voice signal (2026-08-06 calibration).
SPLIT_WINDOW_SECONDS = 1.5

# Candidate boundaries are tried every SPLIT_HOP_SECONDS.
SPLIT_HOP_SECONDS = 0.25

# A change point is believed only when at least this many CONSECUTIVE
# candidates flip with margin — a lone flip is noise.
SPLIT_SUSTAIN = 2

# Both sides of a flip candidate must clear this margin. Measured floor:
# genuine-change candidates ≥0.19, edge noise ≤0.03; 0.15 splits the gap.
SPLIT_MIN_MARGIN = float(os.getenv("MINDSHIFT_DIARIZE_SPLIT_MIN_MARGIN", "0.15"))

# --- Per-word rapid-exchange splitting (utterances WITH word timings) -------
# The sustained-flip scan above needs SPLIT_SUSTAIN consecutive 1.5s windows
# of flipped margin on EACH side of a boundary — structurally blind to a ~1s
# interjection inside a welded utterance. Measured live (2026-08-14, real
# 3-person recording): k-selection correctly heard 3 voices but the scan
# split NOTHING ("0 utterance(s) split") and attribution came out 89%/5%/5%
# because rapid multi-voice exchanges were welded into single utterances.
# Word timings allow a finer instrument: score a short window centered on
# EACH WORD against ALL k pooled centroids, label every word by its nearest
# centroid, smooth (ambiguous words inherit the nearest confident label; runs
# too short to trust merge into their larger neighbor), and split at the
# surviving run boundaries.

# Word-timed utterances longer than this get the per-word scan. Lower than
# SPLIT_MIN_UTTERANCE_SECONDS because two ~1.5s sides are not required —
# each piece only needs MIN_SECONDS to be attributable. SUPERSEDED as the
# applied floor by SCAN_MIN_UTTERANCE_SECONDS (round 2, 2026-08-29); the
# value is kept as the documented round-1 calibration.
WORD_SPLIT_MIN_UTTERANCE_SECONDS = 3.0

# Audio window centered on each word's midpoint (clamped to the utterance).
# Below ~0.8s ECAPA carries too little voice signal; ~1s keeps a one-second
# interjection from blending into its neighbors.
WORD_WINDOW_SECONDS = 0.9

# A word whose best-vs-second-best centroid margin is below this floor is
# AMBIGUOUS: it inherits the nearest confident word's label instead of
# guessing. Calibrated on the real recordings (2026-08-14, pinned ECAPA) —
# see the test-history docstring section.
WORD_MIN_MARGIN = float(os.getenv("MINDSHIFT_DIARIZE_WORD_MIN_MARGIN", "0.10"))

# A label run must carry at least this many words — a single flipped word is
# noise, not a voice change.
WORD_MIN_RUN = 2

# --- Transcript-free window pass (2026-08-29) --------------------------------
# From the voice-separation bake-off (docs/research/2026-08-29-voice-
# separation/, approach B): 1.5 s windows every 0.25 s over the SPEECH of the
# clip (noise-floor-relative gate, speaker_id.speech_mask), embedded in
# batches, clustered spectrally with an eigengap speaker count. Used three
# ways — never as the final word on WHO said what (1.5 s windows in a noisy
# room carry too little voice: maggiano3's window-level ceiling is 0.84):
#   1. boundary PROPOSALS inside long utterances (with or without word
#      timings) — union'd with the per-word pass;
#   2. the eigengap k as a LOWER BOUND on the speaker count that _select_k
#      must at least TRY (never accepted without validating);
#   3. spectral centroids as an ALTERNATIVE k-way partition when average
#      linkage peels a sliver (the maggiano3 failure).
WINDOW_PASS_SECONDS = 1.5
WINDOW_PASS_HOP_SECONDS = 0.25

# Cap on windows embedded for the whole-clip pass; beyond it the hop is
# widened in 0.25 s steps (logged). 600 windows = 150 s of speech at the
# dense grid; the affinity math is O(n^2) memory / O(n^3) eigensolve, both
# trivial at this size.
WINDOW_PASS_MAX_WINDOWS = 600

# A window counts as speech when at least this fraction of its 30 ms frames
# clear the speech gate (B's calibration: every speaker keeps >= 76 % of his
# windows; the TTS fixtures' digital-silence gaps are dropped).
WINDOW_PASS_MIN_SPEECH_FRAC = 0.3

# Windows per model call. Measured 2026-08-29 (this Mac, torch 4 threads):
# 8-15 windows of 1.5 s embed at ~15 ms each, but a batch of 16+ hits a
# pathological torch path (16 windows: 7 s; 162: 69 s) — 10x SLOWER than
# one call per window. 12 keeps every call on the fast path.
WINDOW_PASS_BATCH = 12

# Fewer speech windows than this inside an utterance → no spectral pass
# (an eigengap over 3-4 windows is noise, not evidence).
WINDOW_PASS_MIN_WINDOWS = 6

# Boundaries proposed by the window pass and by the per-word pass that fall
# within this many seconds of each other are the same boundary.
BOUNDARY_DEDUPE_SECONDS = 0.3

# How a boundary is PROPOSED inside an utterance (measured 2026-08-29): the
# utterance's windows carry the labels of the WHOLE-CLIP spectral partition
# (eigengap k), smoothed; a label change is a candidate cut. Running the
# eigengap on the utterance's own 7-27 windows instead was tried first and
# is useless — it returned k=4-6 on 80 of 83 PURE single-voice utterances
# across the fixtures (a percentile-thresholded affinity over that few
# windows is a chain, and every link is an "eigengap") — and a pooled-
# cosine check of a cut's two 1-1.5 s sides cannot rescue it (same-voice
# pools that short measure 0.0-0.4, indistinguishable from two voices).
# Each candidate cut is then CONFIRMED the way the per-word pass confirms a
# word: the pieces on its two sides are embedded and must land on DIFFERENT
# pooled spectral centroids, each with margin ≥ WORD_MIN_MARGIN.

# --- Round 2 (2026-08-29): lower scan floors + uncovered speech ---------------
# With the window pass as a second, transcript-free cut source (above), the
# scan floors the per-word pass shipped with can come down — the two
# constants above (WORD_SPLIT_MIN_UTTERANCE_SECONDS 3.0, MIN_SECONDS 1.0)
# keep their values and meaning; these are the NEW floors the split pass
# actually applies. Driven by maggiano3's per-segment loss after round 1:
# two welded utterances UNDER the 3 s scan floor (5.44-8.02 s asher→dad,
# 8.88-10.74 s dad→mom, 12 % of the rubric's speech) that no instrument
# even looked at, and the mom piece of the second (0.82 s) under MIN_SECONDS.

# Utterances at least this long get the per-word pass AND the window-pass
# proposals (was 3.0 via WORD_SPLIT_MIN_UTTERANCE_SECONDS). Two 1 s pieces
# are the least a split can yield, so 2.0 is the floor at which a scan can
# still say anything. The window pass needs WINDOW_PASS_MIN_WINDOWS (6)
# windows fully inside an utterance — 2.75 s at the dense grid — so on the
# shortest scanned utterances only the per-word pass can speak.
SCAN_MIN_UTTERANCE_SECONDS = 2.0

# A split piece may be this short (instead of MIN_SECONDS) ONLY when BOTH
# instruments confirm it: the per-word label run says it is a different
# voice (confident words, margin ≥ WORD_MIN_MARGIN against the chosen pooled
# centroids) AND the window pass's pooled SPECTRAL centroids place the piece
# and the neighbour it would otherwise merge into on DIFFERENT centroids,
# each with margin ≥ WORD_MIN_MARGIN (the same confirmation
# _WindowPass.propose_boundaries applies to its own cuts). A piece confirmed
# by ONE source keeps the MIN_SECONDS floor. Such a piece is still under
# MIN_SECONDS, so it is NOT embedded for clustering (MIN_SECONDS is
# unchanged for that); it is attributed by its own embedding against the
# final pooled centroids (margin ≥ WORD_MIN_MARGIN) — the very measurement
# that confirmed it — else it inherits like any short turn. 0.8 s is where
# WORD_WINDOW_SECONDS' calibration says ECAPA still carries voice.
CONFIRMED_PIECE_MIN_SECONDS = 0.8

# Speech the transcript never covered: after the utterances (and their
# pieces) are placed, runs of window-VAD speech (speaker_id.speech_mask —
# the same noise-relative gate the window pass uses) outside EVERY utterance
# for at least UNCOVERED_MIN_SECONDS become their own turns with the text
# UNTRANSCRIBED_TEXT. maggiano3's rubric holds ~7 % of speech Deepgram
# returned no utterance for ("Whoa" 18.0-18.4, "See" 21.0-21.6); nothing can
# be labelled where no turn exists. Sub-gate holes up to
# UNCOVERED_BRIDGE_SECONDS inside a run (unvoiced consonants) are bridged; a
# run may never overlap an existing utterance by more than
# UNCOVERED_MAX_OVERLAP_SECONDS (built from frames outside every utterance,
# so in practice 0); and at most UNCOVERED_MAX_FRACTION x the transcript's
# turn count + UNCOVERED_MAX_EXTRA such turns are created (longest first) so
# a bad VAD cannot flood a transcript. Uncovered turns never take part in
# k-selection (measured: they flipped maggiano3's rubric-boundary partition
# 0.833 → 0.562 — see diarize_turns); each is labelled by its own embedding
# against the final pooled centroids.
UNCOVERED_MIN_SECONDS = 0.4
UNCOVERED_BRIDGE_SECONDS = 0.15
UNCOVERED_MAX_OVERLAP_SECONDS = 0.1
UNCOVERED_MAX_FRACTION = 0.2
UNCOVERED_MAX_EXTRA = 3
UNTRANSCRIBED_TEXT = "(untranscribed)"

# How the turns that are NOT embedded for clustering (under MIN_SECONDS)
# get a label. Measured 2026-08-29 on every sub-second turn of the eight
# fixtures (11 turns; truth from the fixtures' own boundaries / the rubric):
# nearest embedded neighbour by midpoint (the shipped rule) 4/11; the
# nearest WINDOW's spectral label 1/11 — a 1.5 s window centred on a
# half-second interjection is two-thirds neighbour audio, and the mode
# filter makes it more so; the turn's OWN embedding against the final
# pooled centroids 5/11 (margins 0.01-0.18). No rule resolves "No." at
# 41.9 s on maggiano3. So: a transcript short turn keeps the neighbour rule
# (unchanged behaviour); an UNCOVERED turn (its neighbours are other
# utterances across a gap — the rubric's "Whoa"/"See" both embed nearest
# their real speaker while both neighbours are wrong) and a both-source-
# CONFIRMED split piece (one neighbour is the very voice it was cut from,
# so inheriting is at least half wrong by construction) take the nearest
# final centroid by their own embedding. Window labels are not used for
# attribution.

# --- Unclaimed speech -> "Unknown" (2026-08-30) -------------------------------
# A word window (or a whole embeddable turn) whose best cosine to EVERY pooled
# centroid — the k-selection winner's (linkage or spectral route) AND the
# window pass's pooled spectral centroids at the eigengap k — is under
# UNCLAIMED_COSINE sounds like NONE of the found voices. Before this, such a
# word simply inherited the nearest confident word (per-word smoothing) and a
# whole turn went to whichever centroid was least unlike it, so speech by an
# unfound voice landed on whoever spoke before. With the flag ON, a run of at
# least WORD_MIN_RUN unclaimed words lasting UNKNOWN_MIN_SECONDS becomes its
# own piece labelled UNKNOWN_SPEAKER: never a cluster (it is excluded from
# every k-selection pass), never an inheritance source, never a neighbour a
# short turn can inherit, never a speaker (num_speakers excludes it; dynamics,
# report cards and enrollment matching skip it — see speaker_id.UNKNOWN_SPEAKER).
# The same floor is applied to every embeddable turn after the final
# refinement and to every turn attributed by its own embedding.
#
# CALIBRATION (2026-08-30, maggiano3, both Deepgram transcripts, the round-1
# instrument exactly as the per-word pass sees it; docs/research/2026-08-30-
# unknown-and-transcript/calibrate_unclaimed.py): per-word best cosine to any
# centroid for words production labels RIGHT — p5 0.096 / 0.099, p10 0.141 /
# 0.153, median 0.41 / 0.39 — vs words it labels WRONG — min 0.129 / 0.102,
# p10 0.147 / 0.139, median 0.33 / 0.36. The mislabelled words are NOT
# unclaimed: they are claimed, confidently, by the wrong centroid. No floor
# separates the two populations; the words under any floor are the
# OVERLAPPED chant at 28-32 s ("I wanna go" / "No." / "Yay.", both voices at
# once), where production happens to be right. At 0.10 the 9 words below are
# all right; at 0.12, 13 below, 1 wrong; at 0.15, 21 below, 5 wrong; at 0.18,
# 34 below, 7 wrong. The son's quiet line that motivated this ("Because I
# wanna do my Duolingo, dad", the 2026-08-27 note under
# STRONG_SEPARATION_COSINE) is claimed at cosine 0.74 / 0.71 by the current
# pipeline — the window pass (2026-08-29) fixed it. Every embeddable final
# turn measures >= 0.22 to its centroid on both transcripts, so the whole-
# turn rule has nothing to fire on either. 0.12 is kept as the documented
# starting point; the flag defaults OFF (UNKNOWN_DEFAULT) because the ship
# criterion was not met — see the module docstring's 2026-08-30 entry.
UNCLAIMED_COSINE = float(os.getenv("MINDSHIFT_DIARIZE_UNCLAIMED_COSINE", "0.12"))

# An unclaimed word run must last this long to stand as its own Unknown piece
# (the same floor as CONFIRMED_PIECE_MIN_SECONDS: where ECAPA still carries
# voice); a shorter run inherits its neighbour exactly as before.
UNKNOWN_MIN_SECONDS = 0.8

# Internal per-word / per-turn cluster id for "no cluster" (real ids are
# 0..k-1).
UNKNOWN_LABEL = -1

# Feature flag, read at CALL time (diarize_turns) so a deploy can flip it
# without a restart of the test process; the default is the measured verdict.
UNKNOWN_FLAG_ENV = "MINDSHIFT_DIARIZE_UNKNOWN"
UNKNOWN_DEFAULT = False

# --- Windows-first engine (2026-08-30) ---------------------------------------
# The bake-off's approach B run END TO END as the speaker labelling (the
# ``windows`` engine, MINDSHIFT_DIARIZE_ENGINE — main.py's cross-check block):
# the whole-clip window pass above (same 1.5 s / 0.25 s grid, same cap and
# hop widening, same noise-floor speech gate) → spectral labels at the
# eigengap k → mode filter → label runs (< SPECTRAL_MIN_RUN_SECONDS absorbed)
# → speaker SEGMENTS, and the transcript's WORDS are regrouped by those
# segments (:func:`regroup_transcript_by_segments`, the same regrouping
# POST …/reanalyze-with-segments applies to the phone's own timeline). The
# transcript's utterance boundaries never decide who spoke — they only
# decide where a turn may ALSO break, so the welded-utterance failure
# (poker: Deepgram hears ONE speaker in seven utterances, the utterance
# engine 4 of 6 voices, 0.467) cannot happen. Why (2026-08-29/30, frame
# accuracy vs GT under score.py): on the owner's real recordings the
# utterance engine scores maggiano3 0.70 / 0.67 (two Deepgram transcripts),
# poker 0.47 (k=4), family 0.97; B scores 0.76 / 0.81 (k=7) / 0.96. Today
# the owner watched "Re-analyze with the latest engine" turn a good 7-voice
# poker result back into 4 voices. The measurements the engine ships with
# are in :func:`diarize_windows_first`'s docstring.
#
# Eigengap search range for THIS engine. B's headline (poker6 k=7, 0.81)
# was measured with max k 8; the window pass's own k_eigengap is clamped to
# MAX_SPEAKERS_LOCAL (6) because there it is only a lower bound for the
# utterance engine's validated k-selection (which returned 4 on poker6).
# Here the eigengap IS the count, so it gets B's range. Measured 2026-08-30
# at max 8: family_real 2, poker6 7, openai/gptaudio/couple 2, family3 3,
# meeting4 3, maggiano3 3 — the same counts as the bake-off.
WINDOWS_FIRST_MAX_K = 8

# The windows engine's ``source`` tag (the utterance engine's is SOURCE).
SOURCE_WINDOWS = "local-ecapa-windows"

# Speech the transcript never covered (the UNCOVERED_* rules) is labelled by
# this engine's segment timeline, so a run may bridge pauses up to this long
# (the utterance engine bridges UNCOVERED_BRIDGE_SECONDS 0.15 because it can
# only give such a run a NEIGHBOUR'S label). Measured 2026-08-30 on the
# owner's poker transcript (Deepgram covered 64 % of the speech): at 0.15 s
# the runs 0.48-1.14, 4.74-5.4 and 7.14-8.34 s — three players' speech with
# breath pauses — stayed unlabelled; at 0.5 s they are recovered, and no
# checked-in fixture gains a run it should not (their gaps are digital
# silence or under UNCOVERED_MIN_SECONDS).
WINDOWS_FIRST_UNCOVERED_BRIDGE_SECONDS = 0.5

# Word regrouping by speaker segments (moved here from main.py 2026-08-30 so
# the windows engine and POST …/reanalyze-with-segments share ONE
# implementation): a word further than this from every segment is not
# snapped to one; it stays with its utterance's neighbours instead.
SPEAKER_SEGMENT_SNAP_S = 0.5


def unknown_enabled() -> bool:
    """Is the Unknown-speaker behaviour on? ``MINDSHIFT_DIARIZE_UNKNOWN``
    (1/true/yes/on vs 0/false/no/off), else :data:`UNKNOWN_DEFAULT`."""
    raw = (os.getenv(UNKNOWN_FLAG_ENV) or "").strip().lower()
    if not raw:
        return UNKNOWN_DEFAULT
    return raw in {"1", "true", "yes", "on"}


def partition_agreement(a: list, b: list) -> float:
    """Pairwise (Rand) agreement of two labelings of the same items, in [0, 1].

    Label NAMES don't matter — only whether each pair of items is grouped
    together or apart in both labelings. Fewer than two items → 1.0 (nothing to
    disagree about). Used to log how far a local relabeling diverged from the
    transcript's own diarization.
    """
    if len(a) != len(b):
        raise ValueError(f"labelings differ in length: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        return 1.0
    agree = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if (a[i] == a[j]) == (b[i] == b[j]):
                agree += 1
    return agree / total


def _merge_to_k(embeddings: list[np.ndarray], k: int) -> list[int]:
    """Average-linkage merging until exactly ``k`` clusters remain.

    Returns a 0..k-1 label per embedding, 0 = the cluster containing the
    earliest input. O(n^3) worst case — fine at transcript scale (≤ 400
    turns).
    """
    n = len(embeddings)
    sim = np.array([[float(np.dot(a, b)) for b in embeddings] for a in embeddings])
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > k:
        best = (-2.0, 0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                avg = float(np.mean([sim[a][b] for a in clusters[i] for b in clusters[j]]))
                if avg > best[0]:
                    best = (avg, i, j)
        _, i, j = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
    labels = [0] * n
    for cid, members in enumerate(sorted(clusters, key=min)):
        for m in members:
            labels[m] = cid
    return labels


def _slice(pcm: np.ndarray, sr: int, turn: dict) -> np.ndarray:
    i0 = max(0, int(float(turn.get("start_time") or 0.0) * sr))
    i1 = min(pcm.size, int(float(turn.get("end_time") or 0.0) * sr))
    return pcm[i0:i1]


def _pooled(
    pcm: np.ndarray, sr: int, turns: list[dict], idxs: list[int],
) -> np.ndarray:
    """Concatenate the given turns' audio, capped at MAX_POOL_SECONDS."""
    cap = int(MAX_POOL_SECONDS * sr)
    chunks, total = [], 0
    for i in idxs:
        chunk = _slice(pcm, sr, turns[i])
        chunks.append(chunk)
        total += chunk.size
        if total >= cap:
            break
    pooled = np.concatenate(chunks)
    return np.ascontiguousarray(pooled[:cap])


def _default_embed(pcm_slice: np.ndarray, sr: int) -> np.ndarray:
    """Embed an audio slice with the pinned ECAPA model."""
    if not speaker_id.is_available():
        raise speaker_id.SpeakerIdUnavailable(
            "voice deps not installed (torch + speechbrain)"
        )
    return speaker_id.embed_pcm(np.ascontiguousarray(pcm_slice), sr)


def _sustained_flip(
    times: list[float], left_margins: list[float], right_margins: list[float],
    min_margin: float, min_run: int = SPLIT_SUSTAIN,
) -> float | None:
    """Time of the strongest candidate inside a sustained opposite-sign run.

    Pure math. A candidate qualifies when its left and right margins have
    OPPOSITE signs and the weaker one still clears ``min_margin``; a change
    point is believed only when at least ``min_run`` CONSECUTIVE candidates
    qualify (calibration: lone flips and sub-margin flips appear only at the
    edges of genuine changes and in noise). Returns the qualifying candidate
    with the largest weaker-margin (the clearest separation), or None.
    """
    best_time: float | None = None
    best_q = -np.inf
    run: list[int] = []

    def flush(run: list[int]) -> None:
        nonlocal best_time, best_q
        if len(run) < min_run:
            return
        for j in run:
            q = min(abs(left_margins[j]), abs(right_margins[j]))
            if q > best_q:
                best_q = q
                best_time = times[j]

    for i in range(len(times)):
        opposite = left_margins[i] * right_margins[i] < 0
        strong = min(abs(left_margins[i]), abs(right_margins[i])) >= min_margin
        if opposite and strong:
            run.append(i)
            continue
        flush(run)
        run = []
    flush(run)
    return best_time


def find_change_point(
    pcm: np.ndarray, sr: int, start: float, end: float, embed,
    centroids: tuple[np.ndarray, np.ndarray],
    *,
    min_margin: float = SPLIT_MIN_MARGIN,
) -> float | None:
    """Scan [start, end] for ONE sustained speaker-change point, or None.

    Every SPLIT_HOP_SECONDS, embed the SPLIT_WINDOW_SECONDS of audio on each
    side of the candidate boundary and score each window's affinity margin
    against the two POOLED cluster centroids (margin = cos(win, c0) -
    cos(win, c1)). Window-to-window cosine is NOT used — on real speech it is
    noise (see module constants). A change point is a sustained run of
    opposite-sign margins (:func:`_sustained_flip`); no sustained flip, no
    change point, no split. Never fabricates.
    """
    c0, c1 = centroids
    times: list[float] = []
    lefts: list[float] = []
    rights: list[float] = []
    b = start + SPLIT_WINDOW_SECONDS
    while b <= end - SPLIT_WINDOW_SECONDS + 1e-9:
        left = pcm[int((b - SPLIT_WINDOW_SECONDS) * sr):int(b * sr)]
        right = pcm[int(b * sr):int((b + SPLIT_WINDOW_SECONDS) * sr)]
        e_left = speaker_id.l2_normalize(embed(np.ascontiguousarray(left), sr))
        e_right = speaker_id.l2_normalize(embed(np.ascontiguousarray(right), sr))
        times.append(b)
        lefts.append(float(np.dot(e_left, c0) - np.dot(e_left, c1)))
        rights.append(float(np.dot(e_right, c0) - np.dot(e_right, c1)))
        b += SPLIT_HOP_SECONDS
    return _sustained_flip(times, lefts, rights, min_margin)


def split_turn_at_word_boundary(
    turn: dict, change_point: float, *, min_seconds: float = MIN_SECONDS,
) -> list[dict] | None:
    """Split ``turn`` at the word boundary nearest ``change_point``.

    Returns ``[left, right]`` turn dicts — text divided by the turn's own
    words, times meeting at the midpoint of the chosen inter-word gap — or
    ``None`` when the split cannot be made honestly: no word timings, no
    interior boundary, or a resulting piece shorter than ``min_seconds``
    (a sliver piece carries too little voice signal to attribute).
    """
    words = turn.get("words")
    if not isinstance(words, list) or len(words) < 2:
        return None
    # Candidate boundaries: midpoints of the gaps between consecutive words.
    boundaries = [
        (i, (float(words[i]["end_time"]) + float(words[i + 1]["start_time"])) / 2)
        for i in range(len(words) - 1)
    ]
    i, boundary = min(boundaries, key=lambda ib: abs(ib[1] - change_point))
    start = float(turn.get("start_time") or 0.0)
    end = float(turn.get("end_time") or 0.0)
    if boundary - start < min_seconds or end - boundary < min_seconds:
        return None
    base = {k: v for k, v in turn.items() if k != "words"}
    left = dict(
        base,
        text=" ".join(w["word"] for w in words[: i + 1]),
        start_time=start, end_time=boundary,
    )
    right = dict(
        base,
        text=" ".join(w["word"] for w in words[i + 1:]),
        start_time=boundary, end_time=end,
    )
    return [left, right]


def _label_words(
    pcm: np.ndarray, sr: int, turn: dict, embed,
    centroids: list[np.ndarray], *, window: float = WORD_WINDOW_SECONDS,
    extra_centroids: list[np.ndarray] | None = None,
) -> tuple[list[int], list[float], list[float]]:
    """Nearest-centroid label + confidence margin + best cosine per word.

    Each word is scored by embedding ``window`` seconds of audio centered on
    the word's midpoint (clamped to the utterance bounds) against ALL pooled
    centroids. The margin is best-minus-second-best cosine — low margin means
    the window does not clearly favor any one voice. The third list is each
    word's BEST cosine to any of ``centroids`` OR ``extra_centroids`` (the
    window pass's pooled spectral centroids — they never LABEL a word, they
    only get a say in whether ANY found voice claims it; see
    UNCLAIMED_COSINE).
    """
    start = float(turn.get("start_time") or 0.0)
    end = float(turn.get("end_time") or 0.0)
    extras = list(extra_centroids or [])
    labels: list[int] = []
    margins: list[float] = []
    bests: list[float] = []
    for w in turn["words"]:
        mid = (float(w["start_time"]) + float(w["end_time"])) / 2
        lo = max(start, mid - window / 2)
        hi = min(end, mid + window / 2)
        e = speaker_id.l2_normalize(
            embed(np.ascontiguousarray(pcm[int(lo * sr):int(hi * sr)]), sr)
        )
        scored = sorted(
            (float(np.dot(e, c)), i) for i, c in enumerate(centroids)
        )
        labels.append(scored[-1][1])
        margins.append(scored[-1][0] - scored[-2][0])
        bests.append(max([scored[-1][0], *(float(np.dot(e, c)) for c in extras)]))
    return labels, margins, bests


def _smooth_word_labels(
    labels: list[int], margins: list[float],
    *, min_margin: float = WORD_MIN_MARGIN,
    unclaimed: list[bool] | None = None,
) -> list[int] | None:
    """Ambiguous words inherit the nearest confident word's label.

    Pure math. A word is confident when its margin clears ``min_margin``;
    every other word takes the label of the nearest confident word (by word
    index; tie → the earlier one — voices persist forward). No confident word
    at all → ``None``: the scan has nothing trustworthy to say.

    ``unclaimed`` (one bool per word; 2026-08-30) marks words whose window
    sounds like NONE of the voices (best cosine under UNCLAIMED_COSINE):
    such a word takes :data:`UNKNOWN_LABEL`, is never an inheritance source
    and never inherits — a claimed neighbour says nothing about a voice
    nobody found. Confidence is judged among CLAIMED words only.
    """
    uncl = list(unclaimed) if unclaimed is not None else [False] * len(labels)
    conf = [i for i, m in enumerate(margins) if m >= min_margin and not uncl[i]]
    if not conf:
        return None
    return [
        UNKNOWN_LABEL if uncl[i]
        else labels[i] if margins[i] >= min_margin
        else labels[min(conf, key=lambda c: (abs(c - i), c))]
        for i in range(len(labels))
    ]


def _word_runs(labels: list[int]) -> list[list[int]]:
    """Maximal same-label runs as ``[label, first_idx, last_idx]`` triples."""
    runs: list[list[int]] = []
    for i, lab in enumerate(labels):
        if runs and runs[-1][0] == lab:
            runs[-1][2] = i
        else:
            runs.append([lab, i, i])
    return runs


def _run_pieces(
    words: list[dict], runs: list[list[int]], start: float, end: float,
) -> list[tuple[float, float]]:
    """Piece (start, end) per run: utterance bounds outside, midpoints of the
    inter-word gaps between adjacent runs inside."""
    bounds = [start]
    for r in runs[:-1]:
        j = r[2]
        bounds.append(
            (float(words[j]["end_time"]) + float(words[j + 1]["start_time"])) / 2
        )
    bounds.append(end)
    return list(zip(bounds[:-1], bounds[1:]))


def _merge_target(
    idx: int, pieces: list[tuple[float, float]],
    run_labels: list[int] | None = None,
) -> int:
    """The run ``idx`` would merge into: its longer-piece neighbour (tie →
    the earlier one). With ``run_labels`` (one label per run), an
    :data:`UNKNOWN_LABEL` neighbour is never chosen while a claimed
    NEIGHBOUR exists — a short claimed run beside unclaimed speech keeps a
    real voice. A sliver claimed run BETWEEN two unclaimed runs merges into
    one of them by the plain rule (it is part of a stretch nobody claims;
    relabelling it to a non-adjacent run would never terminate the
    collapse)."""
    n = len(pieces)
    if run_labels is not None:
        real = [
            j for j in (idx - 1, idx + 1)
            if 0 <= j < n and run_labels[j] != UNKNOWN_LABEL
        ]
        if len(real) == 1:
            return real[0]
    if idx == 0:
        return 1
    if idx == n - 1:
        return idx - 1
    prev_d = pieces[idx - 1][1] - pieces[idx - 1][0]
    next_d = pieces[idx + 1][1] - pieces[idx + 1][0]
    return idx - 1 if prev_d >= next_d else idx + 1


def _collapse_word_runs(
    words: list[dict], labels: list[int], start: float, end: float,
    *, min_run: int = WORD_MIN_RUN, min_seconds: float = MIN_SECONDS,
    min_seconds_confirmed: float | None = None, confirm=None,
    confirmed_out: set[tuple[float, float]] | None = None,
    min_seconds_unknown: float | None = None,
    unknown_out: set[tuple[float, float]] | None = None,
) -> list[int]:
    """Merge untrustworthy label runs into their surrounding dominant voice.

    Pure math. A run is untrustworthy when it carries fewer than ``min_run``
    words OR its piece would last under ``min_seconds`` (too little voice
    signal to attribute honestly). The weakest such run (fewest words, then
    shortest) inherits the label of its longer-piece neighbor (tie → the
    earlier neighbor) until every surviving run is trustworthy.

    Round 2 (2026-08-29): a run with enough words whose piece lasts at
    least ``min_seconds_confirmed`` (but under ``min_seconds``) SURVIVES when
    ``confirm(p0, p1, q0, q1)`` — the piece and the neighbour piece it would
    otherwise merge into — says a second, independent instrument also hears
    two voices there (the window pass's pooled spectral centroids; see
    CONFIRMED_PIECE_MIN_SECONDS). Without ``confirm`` the old floor applies
    unchanged. Every confirmed sub-``min_seconds`` piece's ``(p0, p1)`` is
    added to ``confirmed_out`` when given.

    Unknown (2026-08-30): a run labelled :data:`UNKNOWN_LABEL` (unclaimed
    words, see :func:`_smooth_word_labels`) is trustworthy when it has
    ``min_run`` words and lasts ``min_seconds_unknown``; otherwise it merges
    like any other bad run. With ``min_seconds_unknown`` None every such run
    merges. A bad run never merges INTO an unknown run while a claimed run
    exists (:func:`_merge_target`). Surviving unknown runs' ``(p0, p1)`` go
    to ``unknown_out``.
    """
    labels = list(labels)
    verdict: dict[tuple[int, int], bool] = {}
    while True:
        runs = _word_runs(labels)
        if len(runs) == 1:
            return labels
        pieces = _run_pieces(words, runs, start, end)
        run_labels = [r[0] for r in runs]
        bad: list[tuple[int, float, int]] = []
        for idx, (r, (p0, p1)) in enumerate(zip(runs, pieces)):
            n_words = r[2] - r[1] + 1
            d = p1 - p0
            if r[0] == UNKNOWN_LABEL:
                if (
                    min_seconds_unknown is not None
                    and n_words >= min_run and d >= min_seconds_unknown
                ):
                    continue
                bad.append((n_words, d, idx))
                continue
            if n_words >= min_run and d >= min_seconds:
                continue
            if (
                n_words >= min_run and confirm is not None
                and min_seconds_confirmed is not None and d >= min_seconds_confirmed
            ):
                key = (r[1], r[2])
                if key not in verdict:
                    q0, q1 = pieces[_merge_target(idx, pieces, run_labels)]
                    verdict[key] = bool(confirm(p0, p1, q0, q1))
                if verdict[key]:
                    continue
            bad.append((n_words, d, idx))
        if not bad:
            if confirmed_out is not None:
                for r, (p0, p1) in zip(runs, pieces):
                    if p1 - p0 < min_seconds and verdict.get((r[1], r[2])):
                        confirmed_out.add((p0, p1))
            if unknown_out is not None:
                for r, (p0, p1) in zip(runs, pieces):
                    if r[0] == UNKNOWN_LABEL:
                        unknown_out.add((p0, p1))
            return labels
        idx = min(bad)[2]
        tgt = _merge_target(idx, pieces, run_labels)
        for i in range(runs[idx][1], runs[idx][2] + 1):
            labels[i] = runs[tgt][0]


def split_turn_at_word_runs(turn: dict, labels: list[int]) -> list[dict] | None:
    """Split ``turn`` into one piece per (smoothed) label run.

    ``labels`` is one smoothed+collapsed label per word. Returns the pieces —
    text divided by the turn's own words, times meeting at the midpoints of
    the inter-run word gaps — or ``None`` when everything is one run (no
    change to make).
    """
    words = turn["words"]
    runs = _word_runs(labels)
    if len(runs) < 2:
        return None
    start = float(turn.get("start_time") or 0.0)
    end = float(turn.get("end_time") or 0.0)
    base = {k: v for k, v in turn.items() if k != "words"}
    return [
        dict(
            base,
            text=" ".join(w["word"] for w in words[r[1]:r[2] + 1]),
            start_time=p0, end_time=p1,
        )
        for r, (p0, p1) in zip(runs, _run_pieces(words, runs, start, end))
    ]


class _WindowPass:
    """Speech-gated window embeddings over the clip + the spectral tools on
    them (see the WINDOW_PASS_* constants). One instance per recording.

    ``embed_batch(chunks, sr) -> [emb, ...]`` embeds a list of PCM chunks
    (injectable for the torch-free unit suite; production uses
    :func:`speaker_id.embed_pcm_batch` in WINDOW_PASS_BATCH-sized calls).
    ``embed(pcm_slice, sr)`` embeds the POOLED audio of a window cluster.
    Every embedded window is cached by its start sample, so the windows of a
    long utterance are re-used from the whole-clip pass rather than
    embedded twice; ``embedded`` counts model work actually done.
    """

    def __init__(
        self, pcm: np.ndarray, sr: int, embed_batch, embed, *,
        window: float = WINDOW_PASS_SECONDS, hop: float = WINDOW_PASS_HOP_SECONDS,
        max_windows: int = WINDOW_PASS_MAX_WINDOWS,
    ):
        self.pcm, self.sr = pcm, sr
        self.embed_batch, self.embed = embed_batch, embed
        self.window, self.hop = window, hop
        self.win_samples = int(round(window * sr))
        self.hop_samples = int(round(hop * sr))
        self.mask, self.gate, self.frame_s = speaker_id.speech_mask(pcm, sr)
        self._speech: dict[int, bool] = {}
        self._cache: dict[int, np.ndarray] = {}
        self.embedded = 0
        self.global_hop = hop
        self.starts: list[float] = []
        self.embs = np.zeros((0, 0), dtype=np.float32)
        self.affinity: np.ndarray | None = None
        self.k_eigengap: int | None = None
        self.eigenvalues: list[float] = []
        self._labels_at: dict[int, np.ndarray] = {}
        self._centroids_at: dict[int, dict[int, np.ndarray] | None] = {}
        self.max_windows = max_windows

    # -- windows -------------------------------------------------------------

    def _is_speech(self, start_sample: int) -> bool:
        a = int(start_sample / self.sr / self.frame_s)
        b = max(a + 1, int((start_sample + self.win_samples) / self.sr / self.frame_s))
        seg = self.mask[a:b]
        return bool(seg.size) and float(seg.mean()) >= WINDOW_PASS_MIN_SPEECH_FRAC

    def _grid(self, lo: float, hi: float, hop_samples: int) -> list[int]:
        """Start samples of every window on the clip-anchored grid that lies
        fully inside [lo, hi]."""
        lo_s = max(0, int(round(lo * self.sr)))
        hi_s = min(self.pcm.size, int(round(hi * self.sr)))
        first = -(-lo_s // hop_samples) * hop_samples
        return list(range(first, hi_s - self.win_samples + 1, hop_samples))

    def windows(self, lo: float, hi: float, *, hop_samples: int | None = None,
                ) -> tuple[list[float], np.ndarray]:
        """``(starts, embeddings)`` of the SPEECH windows inside [lo, hi],
        embedding only the ones not already cached."""
        grid = self._grid(lo, hi, hop_samples or self.hop_samples)
        keep: list[int] = []
        for s in grid:
            if s not in self._speech:
                self._speech[s] = self._is_speech(s)
            if self._speech[s]:
                keep.append(s)
        missing = [s for s in keep if s not in self._cache]
        for i in range(0, len(missing), WINDOW_PASS_BATCH):
            batch = missing[i:i + WINDOW_PASS_BATCH]
            chunks = [np.ascontiguousarray(self.pcm[s:s + self.win_samples]) for s in batch]
            vecs = self.embed_batch(chunks, self.sr)
            for s, v in zip(batch, vecs):
                self._cache[s] = speaker_id.l2_normalize(np.asarray(v, dtype=np.float32))
            self.embedded += len(batch)
        starts = [s / self.sr for s in keep]
        embs = (
            np.stack([self._cache[s] for s in keep]).astype(np.float32)
            if keep else np.zeros((0, 0), dtype=np.float32)
        )
        return starts, embs

    def run_global(self) -> None:
        """Embed the whole clip's speech windows (hop widened past
        WINDOW_PASS_MAX_WINDOWS) and compute the eigengap speaker count."""
        duration = self.pcm.size / self.sr
        hop_samples = self.hop_samples
        n = len(self._grid(0.0, duration, hop_samples))
        if n > self.max_windows:
            factor = -(-n // self.max_windows)
            hop_samples *= factor
            logger.info(
                "window pass: %d windows at %.2fs hop exceed the %d cap — "
                "widening the hop to %.2fs",
                n, self.hop, self.max_windows, hop_samples / self.sr,
            )
        self.global_hop = hop_samples / self.sr
        self.starts, self.embs = self.windows(0.0, duration, hop_samples=hop_samples)
        if len(self.starts) >= 3:
            self.affinity = _dsw.refine_affinity(self.embs)
            k, self.eigenvalues = _dsw.eigengap_k(self.affinity, MAX_SPEAKERS_LOCAL)
            self.k_eigengap = max(2, min(MAX_SPEAKERS_LOCAL, k))
        logger.info(
            "window pass: %d speech windows (%.1fs grid, gate %.4f RMS, %d embedded), "
            "eigengap k=%s (raw eigenvalues %s)",
            len(self.starts), self.global_hop, self.gate, self.embedded,
            self.k_eigengap, self.eigenvalues,
        )

    # -- spectral partition of the whole clip --------------------------------

    def labels_at(self, k: int) -> np.ndarray | None:
        if self.affinity is None or k > len(self.starts):
            return None
        if k not in self._labels_at:
            self._labels_at[k] = _dsw.spectral_labels(self.affinity, k)
        return self._labels_at[k]

    def pooled_centroids(self, k: int) -> dict[int, np.ndarray] | None:
        """Pooled-audio centroid per spectral cluster at ``k`` (the union of
        each cluster's window intervals, capped at MAX_POOL_SECONDS, embedded
        with ``embed``) — ``None`` when the spectral pass has fewer than k
        populated clusters."""
        if k in self._centroids_at:
            return self._centroids_at[k]
        labels = self.labels_at(k)
        out: dict[int, np.ndarray] | None = None
        if labels is not None and len(set(labels.tolist())) == k:
            out = {}
            cap = int(MAX_POOL_SECONDS * self.sr)
            for c in sorted(set(labels.tolist())):
                spans: list[list[int]] = []
                for s in sorted(int(round(st * self.sr)) for st, lab in zip(self.starts, labels) if lab == c):
                    e = s + self.win_samples
                    if spans and s <= spans[-1][1]:
                        spans[-1][1] = max(spans[-1][1], e)
                    else:
                        spans.append([s, e])
                pooled = np.concatenate([self.pcm[a:b] for a, b in spans])[:cap]
                out[c] = speaker_id.l2_normalize(self.embed(np.ascontiguousarray(pooled), self.sr))
        self._centroids_at[k] = out
        return out

    def _score_piece(
        self, lo: float, hi: float, cents: dict[int, np.ndarray],
    ) -> tuple[int, float]:
        """``(best spectral centroid, best-minus-second-best cosine)`` of the
        pooled audio in [lo, hi] — one short embed."""
        v = speaker_id.l2_normalize(self.embed(
            np.ascontiguousarray(self.pcm[int(lo * self.sr):int(hi * self.sr)]), self.sr,
        ))
        scored = sorted((float(np.dot(v, cents[c])), c) for c in sorted(cents))
        margin = scored[-1][0] - (scored[-2][0] if len(scored) > 1 else -1.0)
        return scored[-1][1], margin

    def confirms_two_voices(
        self, p0: float, p1: float, q0: float, q1: float,
        *, min_margin: float = WORD_MIN_MARGIN,
    ) -> bool:
        """Does the window pass's instrument hear DIFFERENT voices in the
        pieces [p0, p1] and [q0, q1]? Both are embedded against the pooled
        spectral centroids at the eigengap k and must land on different
        centroids, each with margin ≥ ``min_margin`` — the same confirmation
        :meth:`propose_boundaries` applies to its own cuts. The second
        source behind CONFIRMED_PIECE_MIN_SECONDS. ``False`` whenever the
        spectral pass has nothing (no eigengap, < k populated clusters)."""
        k = self.k_eigengap
        cents = self.pooled_centroids(k) if k is not None else None
        if not cents or len(cents) < 2:
            return False
        lab_p, m_p = self._score_piece(p0, p1, cents)
        lab_q, m_q = self._score_piece(q0, q1, cents)
        return lab_p != lab_q and min(m_p, m_q) >= min_margin

    # -- boundary proposals inside one utterance -----------------------------

    def propose_boundaries(self, lo: float, hi: float) -> tuple[list[float], dict]:
        """Voice-change times inside [lo, hi] from the WHOLE-CLIP spectral
        partition (eigengap k) restricted to that span's windows: the
        windows' cluster labels are mode-filtered, turned into runs (runs
        under SPECTRAL_MIN_RUN_SECONDS absorbed), every piece is held to
        MIN_SECONDS, and each remaining cut is CONFIRMED at the pooled level:
        the pieces on its two sides are embedded and must land on DIFFERENT
        pooled spectral centroids, each with margin ≥ WORD_MIN_MARGIN (the
        per-word pass's confidence bar). The 1.5 s windows FIND a candidate;
        pooled embeddings — the instrument production trusts — CONFIRM it.
        Costs no window embeddings (the whole-clip pass already holds them)
        and one short embed per piece. Returns ``(boundaries, info)``."""
        info: dict = {"windows": 0, "k": self.k_eigengap, "raw": 0, "kept": 0}
        k = self.k_eigengap
        if k is None or self.affinity is None:
            return [], info
        labels_all = self.labels_at(k)
        cents = self.pooled_centroids(k)
        if labels_all is None or cents is None:
            return [], info
        idx = [i for i, s in enumerate(self.starts) if s >= lo and s + self.window <= hi]
        info["windows"] = len(idx)
        if len(idx) < WINDOW_PASS_MIN_WINDOWS:
            return [], info
        starts = [self.starts[i] for i in idx]
        labels = _dsw.mode_filter(labels_all[idx], starts, self.global_hop)
        runs = _dsw.window_label_runs(labels, starts, self.window, lo, hi)
        info["raw"] = len(runs) - 1
        if len(runs) < 2:
            return [], info
        # A run boundary is a cut only where two ADJACENT windows disagree —
        # a label change across a speech gap (the timeline inherits the
        # nearest window on either side of it) is not a place to cut.
        centres = np.asarray(starts, dtype=np.float64) + self.window / 2.0
        cuts = []
        for t in (r[0] for r in runs[1:]):
            left = centres[centres < t]
            right = centres[centres >= t]
            if left.size and right.size and right.min() - left.max() <= 2 * self.global_hop + 1e-6:
                cuts.append(t)
        # Pieces must be at least one WINDOW long: the window pass cannot
        # resolve a shorter piece (its timeline there is inherited from
        # windows that straddle the neighbour). Measured 2026-08-29: at
        # MIN_SECONDS (1.0) the pass cut a 1.13 s sliver off poker6's third
        # turn and a 1.13 s sliver off maggiano3's rubric turn at 9 s — both
        # inside the fixtures' own +/-1-2 s boundary slop, but slivers the
        # word pass (0.9 s word windows) is the right instrument for.
        cuts = _enforce_min_pieces(lo, hi, cuts, min_seconds=self.window)
        if not cuts:
            return [], info
        bounds = [lo, *cuts, hi]
        piece_label: list[int] = []
        piece_margin: list[float] = []
        for b, e in zip(bounds[:-1], bounds[1:]):
            lab, margin = self._score_piece(b, e, cents)
            piece_label.append(lab)
            piece_margin.append(margin)
        kept = [
            cuts[i] for i in range(len(cuts))
            if piece_label[i] != piece_label[i + 1]
            and min(piece_margin[i], piece_margin[i + 1]) >= WORD_MIN_MARGIN
        ]
        info["pieces"] = [
            (round(m, 3), int(lab)) for m, lab in zip(piece_margin, piece_label)
        ]
        info["kept"] = len(kept)
        return kept, info


def _dedupe_boundaries(
    primary: list[float], secondary: list[float], *, tol: float = BOUNDARY_DEDUPE_SECONDS,
) -> list[float]:
    """Sorted union of two boundary lists; a ``secondary`` boundary within
    ``tol`` of a kept one is the same boundary and is dropped."""
    out = sorted(set(primary))
    for b in sorted(secondary):
        if all(abs(b - o) > tol for o in out):
            out.append(b)
    return sorted(out)


def _snap_to_word_gaps(words: list[dict], boundaries: list[float]) -> list[float]:
    """Each boundary → the midpoint of the nearest inter-word gap (duplicates
    collapse)."""
    if not isinstance(words, list) or len(words) < 2:
        return list(boundaries)
    gaps = [
        (float(words[i]["end_time"]) + float(words[i + 1]["start_time"])) / 2
        for i in range(len(words) - 1)
    ]
    return sorted({min(gaps, key=lambda g: abs(g - b)) for b in boundaries})


def _enforce_min_pieces(
    start: float, end: float, boundaries: list[float], *,
    min_seconds: float = MIN_SECONDS, primary: list[float] | None = None,
    confirmed: set[tuple[float, float]] | None = None,
    min_seconds_confirmed: float | None = None,
    floors: dict[tuple[float, float], float] | None = None,
) -> list[float]:
    """Drop boundaries until every piece of [start, end] lasts at least
    ``min_seconds``. For the shortest sliver piece, the boundary dropped is a
    NON-``primary`` one when the sliver has one on each kind of side (a
    word-pass cut outranks a window-pass proposal), else the one whose
    removal merges the sliver into its shorter neighbour — so a sliver never
    survives as its own piece.

    A piece whose ``(start, end)`` is in ``confirmed`` (both-source-confirmed
    by the per-word pass — see :func:`_collapse_word_runs`) is held to
    ``min_seconds_confirmed`` instead; a piece that merely happens to be
    short — bounded by a window proposal, or by a word cut that was not
    confirmed — still faces the full floor. ``floors`` maps a piece's
    ``(start, end)`` to its own floor (the Unknown pieces,
    UNKNOWN_MIN_SECONDS) and takes precedence."""
    prim = set(primary or [])
    conf = confirmed or set()
    per_span = floors or {}
    bounds = [start, *sorted(b for b in boundaries if start < b < end), end]

    def floor_of(a: float, b: float) -> float:
        if (a, b) in per_span:
            return per_span[(a, b)]
        if min_seconds_confirmed is not None and (a, b) in conf:
            return min_seconds_confirmed
        return min_seconds

    while len(bounds) > 2:
        lens = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        short = [
            i for i in range(len(lens))
            if lens[i] < floor_of(bounds[i], bounds[i + 1])
        ]
        if not short:
            break
        i = min(short, key=lambda j: lens[j])
        left_ok = i > 0
        right_ok = i < len(lens) - 1
        if left_ok and right_ok and (bounds[i] in prim) != (bounds[i + 1] in prim):
            del bounds[i + 1 if bounds[i] in prim else i]
        elif left_ok and (not right_ok or lens[i - 1] <= lens[i + 1]):
            del bounds[i]
        else:
            del bounds[i + 1]
    return bounds[1:-1]


def _pieces_from_boundaries(turn: dict, boundaries: list[float]) -> list[dict]:
    """Split ``turn`` at ``boundaries`` (already snapped to word gaps when the
    turn has words). Text follows the words; without word timings the text
    is divided proportionally to the pieces' durations — approximate, and
    logged as such by the caller."""
    start = float(turn.get("start_time") or 0.0)
    end = float(turn.get("end_time") or 0.0)
    bounds = [start, *boundaries, end]
    base = {k: v for k, v in turn.items() if k != "words"}
    words = turn.get("words")
    pieces: list[dict] = []
    if isinstance(words, list) and len(words) >= 2:
        # Boundary → index of the last word before it.
        cuts = []
        for b in boundaries:
            gaps = [
                (float(words[i]["end_time"]) + float(words[i + 1]["start_time"])) / 2
                for i in range(len(words) - 1)
            ]
            cuts.append(min(range(len(gaps)), key=lambda i: abs(gaps[i] - b)))
        idx = [0, *[c + 1 for c in cuts], len(words)]
        for j in range(len(bounds) - 1):
            pieces.append(dict(
                base,
                text=" ".join(w["word"] for w in words[idx[j]:idx[j + 1]]),
                start_time=bounds[j], end_time=bounds[j + 1],
            ))
        return pieces
    tokens = str(turn.get("text") or "").split()
    durs = [bounds[j + 1] - bounds[j] for j in range(len(bounds) - 1)]
    total = sum(durs) or 1.0
    counts = [int(len(tokens) * d / total) for d in durs]
    for _ in range(len(tokens) - sum(counts)):  # largest-remainder top-up
        j = max(range(len(durs)), key=lambda j: len(tokens) * durs[j] / total - counts[j])
        counts[j] += 1
    pos = 0
    for j in range(len(bounds) - 1):
        pieces.append(dict(
            base, text=" ".join(tokens[pos:pos + counts[j]]),
            start_time=bounds[j], end_time=bounds[j + 1],
        ))
        pos += counts[j]
    return pieces


def split_long_utterances(
    pcm: np.ndarray, sr: int, turns: list[dict], embed,
    centroids: tuple[np.ndarray, np.ndarray],
    word_centroids: list[np.ndarray] | None = None,
    *, proposals: dict[int, list[float]] | None = None,
    confirm=None,
    unclaimed_centroids: list[np.ndarray] | None = None,
    unknown: bool = False,
) -> tuple[list[dict], dict]:
    """Split long word-timed utterances at voice changes.

    Two instruments, in order:

    1. PER-WORD pass (utterances over WORD_SPLIT_MIN_UTTERANCE_SECONDS with
       word timings): label every word against ``word_centroids`` (ALL k
       pooled centroids from the k-selection pass; defaults to ``centroids``),
       smooth, and split at label-run boundaries — this catches rapid
       multi-voice exchanges (even a ~1s interjection) the sustained scan is
       structurally blind to, and can yield MORE than two pieces.
    2. SUSTAINED-FLIP fallback (utterances over SPLIT_MIN_UTTERANCE_SECONDS
       the per-word pass had NOTHING CONFIDENT to say about — every word
       margin under the floor): the original two-centroid margin scan
       (``centroids``), split at the nearest word boundary. A confident
       per-word verdict of "no split" (e.g. a lone flipped word smoothed
       away) is respected, never overruled.

    3. WINDOW-PASS PROPOSALS (2026-08-29; ``proposals`` maps a turn index to
       voice-change times found by the transcript-free spectral pass over
       that utterance's 1.5 s windows — :meth:`_WindowPass.propose_boundaries`):
       an ADDITIONAL source of cuts for every utterance over
       WORD_SPLIT_MIN_UTTERANCE_SECONDS, INCLUDING ones without word
       timings. Proposals are snapped to the nearest word gap when words
       exist, union'd with the per-word cuts (a proposal within
       BOUNDARY_DEDUPE_SECONDS of a word-pass cut is the same cut), and
       every piece must still last MIN_SECONDS. A no-words utterance's text
       is divided proportionally to the pieces' durations (approximate —
       logged).

    Round 2 (2026-08-29): utterances of SCAN_MIN_UTTERANCE_SECONDS or more
    are scanned by passes 1 and 3 (was WORD_SPLIT_MIN_UTTERANCE_SECONDS),
    and a per-word piece as short as CONFIRMED_PIECE_MIN_SECONDS survives
    when ``confirm(p0, p1, q0, q1)`` (the window pass's
    :meth:`_WindowPass.confirms_two_voices`) independently hears two voices
    in the piece and the neighbour it would otherwise merge into — both
    sources, or the MIN_SECONDS floor.

    Returns ``(finer_turns, stats)`` where stats counts ``scanned``,
    ``split``, ``skipped_short`` (too short for any instrument — bounded
    compute, by design), ``skipped_no_words`` (long enough for the sustained
    scan but no word timings AND no window proposal — logged, never hidden)
    and ``window_boundaries`` (cuts that came from the window pass alone),
    and lists ``confirmed_short_pieces`` (the ``(start, end)`` of every
    both-source-confirmed piece under MIN_SECONDS that survived). Turns that
    yield no trustworthy change pass through unchanged.

    Unknown (2026-08-30, ``unknown=True``): in the per-word pass a word whose
    best cosine to every centroid in ``word_centroids``/``centroids`` and
    ``unclaimed_centroids`` (the window pass's pooled spectral centroids) is
    under UNCLAIMED_COSINE is UNCLAIMED; a run of WORD_MIN_RUN+ such words
    lasting UNKNOWN_MIN_SECONDS becomes its own piece, listed in
    ``unknown_pieces`` — the caller labels it UNKNOWN_SPEAKER and keeps it
    out of clustering. Off (the default), nothing here changes.
    """
    stats: dict = {
        "scanned": 0, "split": 0, "skipped_short": 0, "skipped_no_words": 0,
        "window_boundaries": 0, "confirmed_short_pieces": [],
        "unknown_pieces": [],
    }
    cents = list(word_centroids) if word_centroids is not None else list(centroids)
    proposals = proposals or {}
    out: list[dict] = []
    for idx, t in enumerate(turns):
        start = float(t.get("start_time") or 0.0)
        end = float(t.get("end_time") or 0.0)
        words = t.get("words")
        has_words = isinstance(words, list) and len(words) >= 2
        long_enough = end - start >= SCAN_MIN_UTTERANCE_SECONDS
        scanned = False
        conclusive = False
        word_bounds: list[float] = []
        confirmed: set[tuple[float, float]] = set()
        unknown_spans: set[tuple[float, float]] = set()
        if has_words and long_enough:
            scanned = True
            labels, margins, bests = _label_words(
                pcm, sr, t, embed, cents,
                extra_centroids=unclaimed_centroids if unknown else None,
            )
            unclaimed = [b < UNCLAIMED_COSINE for b in bests] if unknown else None
            smoothed = _smooth_word_labels(labels, margins, unclaimed=unclaimed)
            if smoothed is not None:
                conclusive = True
                runs = _word_runs(_collapse_word_runs(
                    words, smoothed, start, end,
                    min_seconds_confirmed=(
                        CONFIRMED_PIECE_MIN_SECONDS if confirm is not None else None
                    ),
                    confirm=confirm, confirmed_out=confirmed,
                    min_seconds_unknown=UNKNOWN_MIN_SECONDS if unknown else None,
                    unknown_out=unknown_spans,
                ))
                if len(runs) >= 2:
                    word_bounds = [
                        p1 for _, p1 in _run_pieces(words, runs, start, end)[:-1]
                    ]
        win_bounds: list[float] = []
        if long_enough and proposals.get(idx):
            scanned = True
            win_bounds = list(proposals[idx])
            if has_words:
                win_bounds = _snap_to_word_gaps(words, win_bounds)
        bounds = _enforce_min_pieces(
            start, end, _dedupe_boundaries(word_bounds, win_bounds),
            primary=word_bounds, confirmed=confirmed,
            min_seconds_confirmed=CONFIRMED_PIECE_MIN_SECONDS,
            floors={span: UNKNOWN_MIN_SECONDS for span in unknown_spans},
        )
        if (
            not bounds and not conclusive
            and end - start > SPLIT_MIN_UTTERANCE_SECONDS
        ):
            if not words:
                stats["skipped_no_words"] += 1
                logger.info(
                    "split scan skipped %.1fs utterance at %.2fs: no word timings "
                    "(and no window-pass proposal)",
                    end - start, start,
                )
                out.append(t)
                continue
            scanned = True
            change = find_change_point(pcm, sr, start, end, embed, centroids)
            two = (
                split_turn_at_word_boundary(t, change)
                if change is not None else None
            )
            if two is not None:
                bounds = [float(two[0]["end_time"])]
        if not scanned:
            stats["skipped_short"] += 1
            out.append(t)
            continue
        stats["scanned"] += 1
        if not bounds:
            out.append(t)
            continue
        pieces = _pieces_from_boundaries(t, bounds)
        from_windows = sum(
            1 for b in bounds
            if all(abs(b - w) > BOUNDARY_DEDUPE_SECONDS for w in word_bounds)
        )
        kept_short = [
            (float(p["start_time"]), float(p["end_time"])) for p in pieces
            if (float(p["start_time"]), float(p["end_time"])) in confirmed
        ]
        kept_unknown = [
            (float(p["start_time"]), float(p["end_time"])) for p in pieces
            if (float(p["start_time"]), float(p["end_time"])) in unknown_spans
        ]
        stats["split"] += 1
        stats["window_boundaries"] += from_windows
        stats["confirmed_short_pieces"].extend(kept_short)
        stats["unknown_pieces"].extend(kept_unknown)
        logger.info(
            "split %.1fs utterance at %.2fs into %d pieces (boundaries %s; "
            "%d from the window pass%s%s%s)",
            end - start, start, len(pieces),
            ", ".join(f"{p['end_time']:.2f}" for p in pieces[:-1]),
            from_windows,
            "" if has_words else "; no word timings — text divided by duration",
            (
                f"; {len(kept_short)} piece(s) under {MIN_SECONDS:.1f}s kept on "
                "both-source confirmation" if kept_short else ""
            ),
            (
                f"; {len(kept_unknown)} unclaimed piece(s) -> {UNKNOWN_SPEAKER}: "
                + ", ".join(f"{a:.2f}-{b:.2f}" for a, b in kept_unknown)
                if kept_unknown else ""
            ),
        )
        out.extend(pieces)
    return out, stats


def _speaker_name(index: int) -> str:
    """Cluster index → display label, matching the transcript convention."""
    if index < 26:
        return f"Speaker {chr(ord('A') + index)}"
    return f"Speaker {index + 1}"


def _embed_turns(
    pcm: np.ndarray, sr: int, turns: list[dict], embed, min_seconds: float,
    *, cache: dict[tuple[int, int], np.ndarray] | None = None,
    exclude: set[tuple[float, float]] | None = None,
) -> tuple[list[int], list[np.ndarray]] | None:
    """Embed every turn long enough to carry voice signal.

    Returns ``(order, embs)`` — the embeddable turn indices and their
    normalized embeddings — or ``None`` with fewer than two embeddable turns
    (nothing trustworthy to cluster). ``cache`` (keyed by the slice's sample
    bounds) lets the post-split re-embedding reuse every unchanged turn's
    embedding instead of running the model on it twice. Turns whose
    ``(start, end)`` is in ``exclude`` (the uncovered-speech turns) are left
    out of the clustering set whatever their length.
    """
    embedded: dict[int, np.ndarray] = {}
    for idx, t in enumerate(turns):
        start = float(t.get("start_time") or 0.0)
        end = float(t.get("end_time") or 0.0)
        if end - start < min_seconds or (exclude and (start, end) in exclude):
            continue
        key = (max(0, int(start * sr)), min(pcm.size, int(end * sr)))
        if cache is not None and key in cache:
            embedded[idx] = cache[key]
            continue
        embedded[idx] = speaker_id.l2_normalize(embed(_slice(pcm, sr, t), sr))
        if cache is not None:
            cache[key] = embedded[idx]

    if len(embedded) < 2:
        return None
    order = sorted(embedded)
    return order, [embedded[i] for i in order]


def _refine_k(
    pcm: np.ndarray, sr: int, turns: list[dict], embed,
    order: list[int], embs: list[np.ndarray], k: int,
) -> tuple[list[int], dict[int, np.ndarray]] | None:
    """Force-k split (average linkage) + pooled-centroid refinement.

    Returns ``(labels, centroids)`` — one cluster label per embeddable turn
    and the k pooled cluster centroids — or ``None`` when refinement empties
    a cluster (the audio does not actually hold k distinct voices).
    """
    return _refine_labels(pcm, sr, turns, embed, order, embs, _merge_to_k(embs, k), k)


def _refine_labels(
    pcm: np.ndarray, sr: int, turns: list[dict], embed,
    order: list[int], embs: list[np.ndarray], labels: list[int], k: int,
) -> tuple[list[int], dict[int, np.ndarray]] | None:
    """Pooled-centroid refinement of an initial k-way ``labels`` partition:
    embed each cluster's POOLED audio, reassign every turn to its nearest
    pooled centroid, up to REFINE_ROUNDS rounds or until stable. Shared by
    the average-linkage route (:func:`_refine_k`) and the spectral route
    (:func:`_spectral_route`). ``None`` when a cluster empties."""
    labels = list(labels)
    centroids: dict[int, np.ndarray] = {}
    for _ in range(REFINE_ROUNDS):
        centroids = {}
        for c in set(labels):
            idxs = [order[i] for i in range(len(order)) if labels[i] == c]
            centroids[c] = speaker_id.l2_normalize(
                embed(_pooled(pcm, sr, turns, idxs), sr)
            )
        if len(centroids) < k:
            return None
        new = [
            max(centroids, key=lambda c: float(np.dot(embs[i], centroids[c])))
            for i in range(len(embs))
        ]
        if new == labels:
            break
        labels = new
    if len(set(labels)) < k:
        return None
    return labels, centroids


def _marginal_pairs(
    labels: list[int], prev_labels: list[int], weights: list[float],
) -> list[tuple[int, int]]:
    """Cluster pairs the k-th split CREATED, vs the k-1 partition.

    Each current cluster's parent is the previous cluster holding the
    majority of its speech (weighted by ``weights``, seconds per embeddable
    turn); every pair of current clusters sharing a parent is a pair that
    exists only because of the extra split — the pair k must justify.
    """
    parent: dict[int, int] = {}
    for c in set(labels):
        w: dict[int, float] = {}
        for i, lab in enumerate(labels):
            if lab == c:
                w[prev_labels[i]] = w.get(prev_labels[i], 0.0) + weights[i]
        parent[c] = max(w, key=lambda p: w[p])
    by_parent: dict[int, list[int]] = {}
    for c in sorted(parent):
        by_parent.setdefault(parent[c], []).append(c)
    return [
        (a, b)
        for siblings in by_parent.values()
        for i, a in enumerate(siblings) for b in siblings[i + 1:]
    ]


def _validate_k(
    pcm: np.ndarray, sr: int, turns: list[dict],
    order: list[int], labels: list[int], centroids: dict[int, np.ndarray],
    max_pooled_cosine: float, strict_pairs: list[tuple[int, int]],
) -> dict:
    """Judge one refined k-way split; returns a ``k_evaluated`` entry.

    Always contains ``k``, ``ok``, ``max_pairwise_cosine`` and
    ``min_cluster_seconds``; ``reason`` says what failed. Rules:

    * every PAIR of pooled centroids must be clearly different voices
      (cosine ≤ ``max_pooled_cosine``);
    * every pair in ``strict_pairs`` (the pair(s) the marginal split created
      — see :func:`_marginal_pairs`) must be VERY clearly different voices
      (cosine ≤ :data:`STRONG_SEPARATION_COSINE`), else the split carved one
      voice's registers apart rather than finding a new voice;
    * every marginal split must also be ANCHORED: at least one of its halves
      must have all its cosines to non-sibling clusters ≤
      :data:`NEW_VOICE_ANCHOR_COSINE` — a genuine new voice is wildly unlike
      an existing one, a phantom split is moderately far from everything;
    * every cluster must carry ≥ :data:`MIN_CLUSTER_SECONDS` of speech, OR
      ≥ :data:`MIN_CLUSTER_SECONDS_STRONG` when it is VERY clearly distinct
      from every other centroid (all its pairwise cosines ≤
      :data:`STRONG_SEPARATION_COSINE`).
    """
    cids = sorted(centroids)
    pair_cos = {
        (a, b): float(np.dot(centroids[a], centroids[b]))
        for i, a in enumerate(cids) for b in cids[i + 1:]
    }
    seconds = {
        c: sum(
            _slice(pcm, sr, turns[order[i]]).size
            for i in range(len(order)) if labels[i] == c
        ) / sr
        for c in cids
    }
    entry = {
        "k": len(cids),
        "ok": True,
        "max_pairwise_cosine": round(max(pair_cos.values()), 3),
        "min_cluster_seconds": round(min(seconds.values()), 2),
    }
    worst = max(pair_cos.values())
    if worst > max_pooled_cosine:
        entry["ok"] = False
        entry["failed"] = "centroids"
        entry["reason"] = (
            f"centroids not clearly distinct (worst pair cosine "
            f"{worst:.3f} > {max_pooled_cosine:.2f})"
        )
        return entry
    if strict_pairs:
        worst_split = max(
            pair_cos[tuple(sorted(p))] for p in strict_pairs
        )
        entry["marginal_pair_cosine"] = round(worst_split, 3)
        if worst_split > STRONG_SEPARATION_COSINE:
            entry["ok"] = False
            entry["failed"] = "marginal_pair"
            entry["reason"] = (
                f"marginal split pair too similar (cosine {worst_split:.3f} "
                f"> {STRONG_SEPARATION_COSINE:.2f} — one voice in two "
                "registers, not a new voice)"
            )
            return entry

        def _outside(c: int, sibling: int) -> float:
            """Worst (highest) cosine from cluster ``c`` to any non-sibling."""
            others = [
                pair_cos[tuple(sorted((c, o)))]
                for o in cids if o not in (c, sibling)
            ]
            return max(others) if others else -1.0

        for a, b in strict_pairs:
            anchor = min(_outside(a, b), _outside(b, a))
            entry["split_anchor_cosine"] = round(anchor, 3)
            if anchor > NEW_VOICE_ANCHOR_COSINE:
                entry["ok"] = False
                entry["failed"] = "anchor"
                entry["reason"] = (
                    "marginal split not anchored by a clearly new voice "
                    f"(both halves ≥ cosine {anchor:.3f} from every other "
                    f"cluster; needs ≤ {NEW_VOICE_ANCHOR_COSINE:.2f})"
                )
                return entry
    for c in cids:
        own_pairs = [v for (a, b), v in pair_cos.items() if c in (a, b)]
        floor = (
            MIN_CLUSTER_SECONDS_STRONG
            if max(own_pairs) <= STRONG_SEPARATION_COSINE
            else MIN_CLUSTER_SECONDS
        )
        if seconds[c] < floor:
            entry["ok"] = False
            entry["failed"] = "duration_floor"
            entry["reason"] = (
                f"cluster has only {seconds[c]:.1f}s pooled speech "
                f"(needs {floor:.1f}s at separation "
                f"{max(own_pairs):.3f})"
            )
            return entry
    return entry


def _select_k(
    pcm: np.ndarray, sr: int, turns: list[dict], embed,
    order: list[int], embs: list[np.ndarray],
    max_pooled_cosine: float,
    *, pass2: tuple[list[int], dict[int, np.ndarray]] | None = None,
    max_k: int = MAX_SPEAKERS_LOCAL,
    window_pass: _WindowPass | None = None,
) -> tuple[list[dict], tuple[list[int], dict[int, np.ndarray], dict] | None]:
    """Refine + validate every candidate k; keep the LARGEST that validates.

    Each k > 2 must additionally justify the pair(s) its extra split created
    against the refined k-1 partition (the marginal-split rule — see
    :data:`STRONG_SEPARATION_COSINE`). ``pass2`` is an already-refined k=2
    partition to reuse; ``max_k`` caps the candidate range below
    :data:`MAX_SPEAKERS_LOCAL`.

    Two ROUTES to a k-way partition (2026-08-29; each ``k_evaluated`` entry
    says which in ``route``): average LINKAGE first, and — when
    ``window_pass`` is given — a SPECTRAL alternative seeded from the window
    pass's pooled spectral centroids (:func:`_spectral_route`) whenever the
    linkage partition failed on a DURATION FLOOR (it peeled a sliver — the
    maggiano3 failure, where all three voices are ≤0.24 apart yet linkage's
    3-way split carved a 1 s sliver) or when ``k`` is the window pass's
    EIGENGAP count (the lower bound: the window evidence says at least k
    voices, so the k-way spectral partition is at least TRIED before a
    smaller k is settled on). The eigengap never RAISES k on its own — every
    partition, whichever route, faces the same :func:`_validate_k`, and the
    spectral route is only consulted for k ≤ the eigengap count. Returns
    ``(k_evaluated, chosen)`` where ``chosen`` is ``(labels, centroids,
    entry)`` or ``None`` when no k validates.
    """
    weights = [_slice(pcm, sr, turns[i]).size / sr for i in order]
    k_evaluated: list[dict] = []
    chosen: tuple[list[int], dict[int, np.ndarray], dict] | None = None
    prev_labels: list[int] | None = None
    k_eig = window_pass.k_eigengap if window_pass is not None else None
    for k in range(2, min(max_k, MAX_SPEAKERS_LOCAL, len(order)) + 1):
        refined = (
            pass2 if k == 2 and pass2 is not None
            else _refine_k(pcm, sr, turns, embed, order, embs, k)
        )
        labels_k: list[int] | None = None
        entry: dict | None = None
        if refined is None:
            k_evaluated.append({
                "k": k, "ok": False, "route": "linkage",
                "reason": "refinement collapsed clusters "
                          f"(audio does not hold {k} distinct voices)",
            })
        else:
            labels_k, centroids_k = refined
            if k == 2:
                strict_pairs: list[tuple[int, int]] | None = []
            elif prev_labels is None:
                strict_pairs = None
                k_evaluated.append({
                    "k": k, "ok": False, "route": "linkage",
                    "reason": f"no refined k={k - 1} partition to justify "
                              "the marginal split against",
                })
            else:
                strict_pairs = _marginal_pairs(labels_k, prev_labels, weights)
            if strict_pairs is not None:
                entry = _validate_k(
                    pcm, sr, turns, order, labels_k, centroids_k,
                    max_pooled_cosine, strict_pairs,
                )
                entry["route"] = "linkage"
                k_evaluated.append(entry)
                if entry["ok"]:
                    chosen = (labels_k, centroids_k, entry)
                    prev_labels = labels_k
                    continue
        floor_fail = entry is not None and entry.get("failed") == "duration_floor"
        if (
            window_pass is not None and k_eig is not None and k <= k_eig
            and (floor_fail or k == k_eig)
        ):
            alt_entry, alt = _spectral_route(
                pcm, sr, turns, embed, order, embs, k, window_pass,
                prev_labels, weights, max_pooled_cosine,
            )
            k_evaluated.append(alt_entry)
            if alt is not None and alt_entry["ok"]:
                chosen = (alt[0], alt[1], alt_entry)
                prev_labels = alt[0]
                continue
        prev_labels = labels_k
    return k_evaluated, chosen


def _spectral_route(
    pcm: np.ndarray, sr: int, turns: list[dict], embed,
    order: list[int], embs: list[np.ndarray], k: int, window_pass: _WindowPass,
    prev_labels: list[int] | None, weights: list[float], max_pooled_cosine: float,
) -> tuple[dict, tuple[list[int], dict[int, np.ndarray]] | None]:
    """The SPECTRAL k-way partition of the embeddable turns: every turn goes
    to the nearest of the window pass's k pooled spectral centroids, then
    the same pooled-centroid refinement rounds as the linkage route
    (:func:`_refine_labels`), then the same :func:`_validate_k` (with the
    marginal-split pairs measured against ``prev_labels``, the k-1
    partition). Returns ``(k_evaluated entry, (labels, centroids) | None)``.
    """
    entry: dict = {"k": k, "ok": False, "route": "spectral"}
    cents = window_pass.pooled_centroids(k)
    if cents is None:
        entry["reason"] = (
            f"window spectral pass has fewer than {k} populated clusters"
        )
        return entry, None
    keys = sorted(cents)
    seed = [
        keys[int(np.argmax([float(np.dot(e, cents[c])) for c in keys]))]
        for e in embs
    ]
    if len(set(seed)) < k:
        entry["reason"] = (
            f"spectral centroids claim only {len(set(seed))} of {k} clusters "
            "among the embeddable turns"
        )
        return entry, None
    refined = _refine_labels(pcm, sr, turns, embed, order, embs, seed, k)
    if refined is None:
        entry["reason"] = (
            "refinement collapsed clusters from the spectral seed "
            f"(audio does not hold {k} distinct voices)"
        )
        return entry, None
    labels_s, cents_s = refined
    if k == 2:
        strict: list[tuple[int, int]] = []
    elif prev_labels is None:
        entry["reason"] = (
            f"no refined k={k - 1} partition to justify the marginal split against"
        )
        return entry, None
    else:
        strict = _marginal_pairs(labels_s, prev_labels, weights)
    entry = _validate_k(
        pcm, sr, turns, order, labels_s, cents_s, max_pooled_cosine, strict,
    )
    entry["route"] = "spectral"
    return entry, (labels_s, cents_s)


# --- Uncovered speech + window-label inheritance (round 2, 2026-08-29) -------

def _uncovered_speech(
    mask: np.ndarray, frame_s: float, turns: list[dict], duration: float,
    *, min_seconds: float = UNCOVERED_MIN_SECONDS,
    bridge: float = UNCOVERED_BRIDGE_SECONDS,
) -> list[tuple[float, float]]:
    """``(start, end)`` runs of speech frames (``mask``, one bool per
    ``frame_s``) that lie OUTSIDE every turn for at least ``min_seconds``.

    Pure numpy. A frame touching any turn counts as covered, so a run never
    overlaps a turn; sub-gate holes up to ``bridge`` seconds between speech
    frames of the same gap are bridged (a covered frame always ends a run).
    """
    n = int(mask.size)
    if n == 0 or duration <= 0:
        return []
    covered = np.zeros(n, dtype=bool)
    for t in turns:
        s = float(t.get("start_time") or 0.0)
        e = float(t.get("end_time") or 0.0)
        if e <= s:
            continue
        a = max(0, int(np.floor(s / frame_s)))
        b = min(n, int(np.ceil(e / frame_s)))
        covered[a:b] = True
    speech = np.asarray(mask, dtype=bool) & ~covered
    max_hole = int(round(bridge / frame_s))
    out: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if not speech[i]:
            i += 1
            continue
        j = i
        last = i
        while j < n and not covered[j]:
            if speech[j]:
                last = j
            elif j - last > max_hole:
                break
            j += 1
        s, e = i * frame_s, (last + 1) * frame_s
        if e - s >= min_seconds:
            out.append((round(s, 3), round(min(e, duration), 3)))
        i = max(last + 1, j)
    return out


def _overlap_seconds(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _uncovered_turn_dicts(
    candidates: list[tuple[float, float]], turns: list[dict], *, cap: int,
    max_overlap: float = UNCOVERED_MAX_OVERLAP_SECONDS,
) -> list[dict]:
    """Turn dicts (text UNTRANSCRIBED_TEXT, the nearest utterance's input
    speaker) for the ``cap`` LONGEST candidate runs that overlap no existing
    turn by more than ``max_overlap`` seconds. Pure Python."""
    if cap <= 0 or not candidates:
        return []
    spans = [
        (float(t.get("start_time") or 0.0), float(t.get("end_time") or 0.0), t)
        for t in turns
    ]
    ok = [
        (s, e) for s, e in candidates
        if e > s and all(_overlap_seconds(s, e, a, b) <= max_overlap for a, b, _ in spans)
    ]
    ok.sort(key=lambda se: (-(se[1] - se[0]), se[0]))
    out: list[dict] = []
    for s, e in sorted(ok[:cap]):
        mid = (s + e) / 2
        nearest = (
            min(spans, key=lambda sp: abs((sp[0] + sp[1]) / 2 - mid))[2]
            if spans else {}
        )
        out.append({
            "speaker": nearest.get("speaker") or "Speaker A",
            "text": UNTRANSCRIBED_TEXT,
            "start_time": s, "end_time": e,
        })
    return out


def _insert_chronological(
    turns: list[dict], extras: list[dict],
) -> tuple[list[dict], list[int]]:
    """``turns`` with ``extras`` merged in by start time (existing order kept;
    an extra goes before the first turn starting at or after it). Returns
    ``(merged, new_index_of_old)``."""
    merged: list[dict] = []
    index_map: list[int] = []
    pending = sorted(extras, key=lambda t: float(t.get("start_time") or 0.0))
    for t in turns:
        start = float(t.get("start_time") or 0.0)
        while pending and float(pending[0].get("start_time") or 0.0) < start:
            merged.append(pending.pop(0))
        index_map.append(len(merged))
        merged.append(t)
    merged.extend(pending)
    return merged, index_map


def diarize_turns(
    pcm: np.ndarray,
    sr: int,
    turns: list[dict],
    *,
    embed_fn=None,
    embed_batch_fn=None,
    min_seconds: float = MIN_SECONDS,
    max_pooled_cosine: float = MAX_POOLED_COSINE,
) -> dict | None:
    """Attempt a validated k-voice relabeling of ``turns`` from the audio.

    Every k = 2 .. :data:`MAX_SPEAKERS_LOCAL` (capped by the embeddable turn
    count) is merged, refined and validated; the LARGEST fully-validating k
    wins. Returns ``None`` whenever local diarization has nothing TRUSTWORTHY
    to say: voice model unavailable, fewer than two embeddable utterances, or
    NO k validates (a monologue, or clusters with too little pooled speech /
    insufficiently distinct centroids). Otherwise returns::

        {
          "turns": [...],              # speaker relabeled; a turn that was
                                       # split arrives as TWO turns with the
                                       # text divided at a word boundary
          "num_speakers": int,         # the chosen k (2..MAX_SPEAKERS_LOCAL)
          "source": "local-ecapa",
          "model": "<hf-source>@<pinned-revision>",
          "segments_total": int,       # after any word-level splitting
          "segments_embedded": int,
          "split_utterances": int,     # utterances split at a voice change
          "confirmed_short_pieces": int,   # pieces under MIN_SECONDS kept on
                                       # both-source confirmation
          "uncovered_turns": int,      # "(untranscribed)" turns added for
                                       # speech no utterance covered
          "unknown_turns": int,        # turns labelled UNKNOWN_SPEAKER
          "unknown_seconds": float,    # (0 / 0.0 unless MINDSHIFT_DIARIZE_
                                       # UNKNOWN is on — see UNCLAIMED_COSINE)
          "short_turn_attribution": {self, neighbour},  # how the
                                       # un-embedded turns got their label
          "pooled_cosine": float,      # WORST (highest) pairwise centroid
                                       # cosine of the chosen k (low=distinct)
          "k_evaluated": [...],        # every k tried: {k, ok,
                                       # max_pairwise_cosine,
                                       # min_cluster_seconds,
                                       # marginal_pair_cosine? (k>2),
                                       # reason?}
          "agreement_with_input": float,   # Rand agreement vs input labels
          "window_pass": {...},        # the transcript-free window pass:
                                       # windows, hop, speech gate, windows
                                       # embedded, k_eigengap, eigenvalues,
                                       # proposed_boundaries
        }

    Turns may carry an optional ``words`` list ([{word, start_time,
    end_time}, ...]); it enables the word-level split pre-pass and is never
    propagated to the output turns. ``embed_batch_fn(chunks, sr)`` embeds a
    list of window chunks for the window pass (defaults to
    :func:`speaker_id.embed_pcm_batch`, or to a loop over ``embed_fn`` when
    only that is injected).
    """
    embed, embed_batch = _resolve_embedders(embed_fn, embed_batch_fn)
    n_transcript = len(turns)
    embed_cache: dict[tuple[int, int], np.ndarray] = {}
    unknown = unknown_enabled()
    spectral_cents: list[np.ndarray] = []
    try:
        embedded = _embed_turns(pcm, sr, turns, embed, min_seconds, cache=embed_cache)
        if embedded is None:
            return None
        order, embs = embedded

        # Transcript-free window pass over the whole clip's speech (see the
        # WINDOW_PASS_* constants): supplies the eigengap speaker-count lower
        # bound + spectral centroids to k-selection and voice-change
        # proposals inside long utterances to the split pass.
        window_pass = _WindowPass(pcm, sr, embed_batch, embed)
        window_pass.run_global()

        # First k-selection pass over the transcript's own utterances: its
        # winner supplies the pooled centroids the word-level splitter scores
        # against — ALL k of them, so a rapid exchange between ANY two of the
        # recording's voices shows up (a 2-way margin contrast is blind to a
        # third voice). When no k validates yet (heavy welding can hide a
        # voice), the 2-way refinement's centroids still anchor the scan.
        pass1 = _refine_k(pcm, sr, turns, embed, order, embs, 2)
        if pass1 is None:
            return None
        k_evaluated, chosen = _select_k(
            pcm, sr, turns, embed, order, embs, max_pooled_cosine, pass2=pass1,
            window_pass=window_pass,
        )
        word_centroids = (
            [c for _, c in sorted(chosen[1].items())] if chosen is not None
            else [pass1[1][0], pass1[1][1]]
        )

        # Window-pass boundary proposals inside every long utterance (with
        # or without word timings) — cost proportional to the long
        # utterances' seconds only (their windows were embedded by the
        # whole-clip pass above and are re-used from its cache).
        proposals: dict[int, list[float]] = {}
        n_proposed = 0
        for idx, t in enumerate(turns):
            start = float(t.get("start_time") or 0.0)
            end = float(t.get("end_time") or 0.0)
            if end - start < SCAN_MIN_UTTERANCE_SECONDS:
                continue
            bounds, info = window_pass.propose_boundaries(start, end)
            logger.info(
                "window pass on %.1fs utterance at %.2fs: %d windows, %d raw "
                "cut(s), %d confirmed (pieces [margin, centroid] %s)",
                end - start, start, info["windows"], info["raw"],
                info["kept"], info.get("pieces", []),
            )
            if bounds:
                proposals[idx] = bounds
                n_proposed += len(bounds)

        # Word-level split pass: a transcriber can weld a rapid multi-voice
        # exchange into ONE utterance. Split word-timed utterances at
        # per-word voice-run boundaries (sustained-flip scan as fallback),
        # union'd with the window-pass proposals, then re-embed the finer
        # segments and REDO k-selection so every piece is attributed and
        # validated like any other turn.
        # Unknown (2026-08-30, flag): the window pass's pooled spectral
        # centroids join the k-selection centroids as the set a word or a
        # turn must be claimed by (UNCLAIMED_COSINE). Unclaimed pieces are
        # NEVER clustered — see the exclude below — so they cannot seed a k.
        if unknown and window_pass.k_eigengap:
            sc = window_pass.pooled_centroids(window_pass.k_eigengap)
            spectral_cents = [sc[c] for c in sorted(sc)] if sc else []
        turns, split_stats = split_long_utterances(
            pcm, sr, turns, embed, (pass1[1][0], pass1[1][1]),
            word_centroids=word_centroids, proposals=proposals,
            confirm=window_pass.confirms_two_voices,
            unclaimed_centroids=spectral_cents, unknown=unknown,
        )
        confirmed_short = set(split_stats["confirmed_short_pieces"])
        unknown_spans = set(split_stats["unknown_pieces"])

        # Speech the transcript never covered (round 2, 2026-08-29; see the
        # UNCOVERED_* constants): runs of window-VAD speech outside every
        # utterance become their own "(untranscribed)" turns — nothing can
        # be attributed where no turn exists. Capped, never overlapping,
        # inserted chronologically. They are NOT part of k-selection:
        # measured on maggiano3's rubric boundaries, letting the two runs
        # over MIN_SECONDS (35.5-37.0 s and 39.0-41.0 s, real speech the
        # rubric's slop leaves outside its segments) into the re-selection
        # let the LINKAGE k=3 partition validate (min cluster 2.0 s) ahead
        # of the spectral route — the old mom-inside-dad partition, 0.833 →
        # 0.562. Whole transcript utterances decide HOW MANY voices and how
        # they partition; uncovered speech only gets labelled (by its own
        # embedding against the final centroids — see the attribution rule
        # below).
        duration = pcm.size / sr
        uncovered = _uncovered_turn_dicts(
            _uncovered_speech(window_pass.mask, window_pass.frame_s, turns, duration),
            turns,
            cap=int(UNCOVERED_MAX_FRACTION * n_transcript) + UNCOVERED_MAX_EXTRA,
        )
        if uncovered:
            logger.info(
                "uncovered speech: %d run(s) outside every utterance become "
                "%r turns (%s)",
                len(uncovered), UNTRANSCRIBED_TEXT,
                ", ".join(f"{t['start_time']:.2f}-{t['end_time']:.2f}" for t in uncovered),
            )
            turns, index_map = _insert_chronological(turns, uncovered)
            order = [index_map[i] for i in order]
        uncovered_spans = {
            (float(t["start_time"]), float(t["end_time"])) for t in uncovered
        }
        if split_stats["split"]:
            embedded = _embed_turns(
                pcm, sr, turns, embed, min_seconds, cache=embed_cache,
                exclude=uncovered_spans | unknown_spans,
            )
            if embedded is None:
                return None
            order, embs = embedded
            # The re-selection may not claim MORE speakers than the
            # whole-utterance pass validated: the split pieces were measured
            # WITH round 1's centroids as the instrument, and a piece of
            # genuinely OVERLAPPED speech embeds far from every real voice —
            # on the real 3-person recording such a piece (two voices
            # chanting over each other) minted a phantom 4th cluster that
            # slipped under the marginal bars (0.294 < 0.30, anchor
            # 0.14 < 0.15). Whole utterances are the trustworthy evidence
            # for HOW MANY voices there are; pieces only redistribute WHO
            # said what. (No round-1 winner → no cap, as before.)
            #
            # 2026-08-29: the cap is lifted to the window pass's EIGENGAP
            # count when that is higher — independent, transcript-free
            # evidence of how many voices there are (it never over-counted a
            # 2-4-voice fixture in the bake-off). A whole utterance that
            # welds three voices embeds as a BLEND and can fail round 1's
            # marginal-split rule, which would otherwise forbid the very
            # pieces the window pass just cut from claiming the third voice.
            # Validation still gates every k; the cap only decides what is
            # TRIED.
            k_evaluated, chosen = _select_k(
                pcm, sr, turns, embed, order, embs, max_pooled_cosine,
                max_k=(
                    max(len(chosen[1]), window_pass.k_eigengap or 0)
                    if chosen is not None else MAX_SPEAKERS_LOCAL
                ),
                window_pass=window_pass,
            )
        if chosen is None:
            logger.info(
                "local diarization heard one voice (no k validated: %s)",
                "; ".join(
                    f"k={e['k']}: {e.get('reason', 'ok')}" for e in k_evaluated
                ),
            )
            return None
        labels, centroids, chosen_entry = chosen
        pooled_cosine = chosen_entry["max_pairwise_cosine"]
    except speaker_id.SpeakerIdUnavailable as exc:
        logger.info("local diarization unavailable: %s", exc)
        return None

    cluster_of = dict(zip(order, labels))

    # Whole-turn Unknown rule (2026-08-30, flag): an embeddable turn the
    # refinement placed in a cluster it does not resemble — best cosine to
    # EVERY final centroid and every spectral centroid under
    # UNCLAIMED_COSINE — is unclaimed. Never all of them (a one-turn cluster
    # measures 1.0 to its own centroid; the guard is belt and braces).
    all_cents = list(centroids.values()) + spectral_cents
    if unknown:
        flagged = [
            i for i, e in enumerate(embs)
            if max(float(np.dot(e, c)) for c in all_cents) < UNCLAIMED_COSINE
        ]
        if flagged and len(flagged) < len(order):
            for i in flagged:
                cluster_of[order[i]] = UNKNOWN_LABEL
                t = turns[order[i]]
                logger.info(
                    "turn %.2f-%.2fs claimed by no voice (best cosine %.3f < %.2f) -> %s",
                    float(t.get("start_time") or 0.0), float(t.get("end_time") or 0.0),
                    max(float(np.dot(embs[i], c)) for c in all_cents),
                    UNCLAIMED_COSINE, UNKNOWN_SPEAKER,
                )

    # Un-embedded (too-short) turns (round 2, 2026-08-29 — see the
    # attribution comment above CONFIRMED_PIECE_MIN_SECONDS): a both-source-
    # confirmed split piece or an uncovered turn takes the nearest final
    # pooled centroid by its OWN embedding; every other short turn inherits
    # the nearest embedded turn's cluster, by utterance midpoint. An Unknown
    # piece (unclaimed word run) is Unknown; a self-attributed turn under
    # the floor is Unknown; an Unknown turn is never inherited from.
    def midpoint(t: dict) -> float:
        return (float(t.get("start_time") or 0.0) + float(t.get("end_time") or 0.0)) / 2

    cids = sorted(centroids)
    claimed_order = [e for e in order if cluster_of[e] != UNKNOWN_LABEL] or list(order)
    attributed = {"self": 0, "neighbour": 0}
    for idx, t in enumerate(turns):
        if idx in cluster_of:
            continue
        span = (float(t.get("start_time") or 0.0), float(t.get("end_time") or 0.0))
        if span in unknown_spans:
            cluster_of[idx] = UNKNOWN_LABEL
            continue
        if span in confirmed_short or span in uncovered_spans:
            try:
                v = speaker_id.l2_normalize(embed(_slice(pcm, sr, t), sr))
            except speaker_id.SpeakerIdUnavailable:
                v = None
            if v is not None:
                scored = sorted((float(np.dot(v, centroids[c])), c) for c in cids)
                best_any = max(
                    [scored[-1][0], *(float(np.dot(v, c)) for c in spectral_cents)]
                )
                if unknown and best_any < UNCLAIMED_COSINE:
                    cluster_of[idx] = UNKNOWN_LABEL
                    logger.info(
                        "%s %.2f-%.2fs claimed by no voice (best cosine %.3f < %.2f) -> %s",
                        "uncovered turn" if span in uncovered_spans else "confirmed piece",
                        span[0], span[1], best_any, UNCLAIMED_COSINE, UNKNOWN_SPEAKER,
                    )
                    continue
                cluster_of[idx] = scored[-1][1]
                attributed["self"] += 1
                logger.info(
                    "%s %.2f-%.2fs attributed by its own embedding (margin %.3f)",
                    "uncovered turn" if span in uncovered_spans else "confirmed piece",
                    span[0], span[1],
                    scored[-1][0] - (scored[-2][0] if len(scored) > 1 else -1.0),
                )
                continue
        nearest = min(claimed_order, key=lambda e: abs(midpoint(turns[e]) - midpoint(t)))
        cluster_of[idx] = cluster_of[nearest]
        attributed["neighbour"] += 1

    # Name clusters in order of first appearance across the full transcript;
    # Unknown is a label, not a speaker (num_speakers excludes it).
    name_of: dict[int, str] = {}
    for idx in range(len(turns)):
        cid = cluster_of[idx]
        if cid != UNKNOWN_LABEL and cid not in name_of:
            name_of[cid] = _speaker_name(len(name_of))
    num_speakers = len(name_of)
    name_of[UNKNOWN_LABEL] = UNKNOWN_SPEAKER
    unknown_idx = [i for i in range(len(turns)) if cluster_of[i] == UNKNOWN_LABEL]
    unknown_seconds = sum(
        float(turns[i].get("end_time") or 0.0) - float(turns[i].get("start_time") or 0.0)
        for i in unknown_idx
    )
    if unknown_idx:
        logger.info(
            "%d turn(s) / %.2fs labelled %s (claimed by none of the %d voices)",
            len(unknown_idx), unknown_seconds, UNKNOWN_SPEAKER, num_speakers,
        )

    # Output turns are plain {speaker, text, start_time, end_time, ...} — the
    # internal ``words`` plumbing is not propagated.
    new_turns = [
        dict({k: v for k, v in t.items() if k != "words"},
             speaker=name_of[cluster_of[i]])
        for i, t in enumerate(turns)
    ]
    return {
        "turns": new_turns,
        "num_speakers": num_speakers,
        "source": SOURCE,
        "model": f"{speaker_id.ECAPA_SOURCE}@{speaker_id.ECAPA_REVISION}",
        "segments_total": len(turns),
        "segments_embedded": len(order),
        "split_utterances": split_stats["split"],
        "confirmed_short_pieces": len(confirmed_short),
        "uncovered_turns": len(uncovered),
        "unknown_turns": len(unknown_idx),
        "unknown_seconds": round(unknown_seconds, 2),
        "short_turn_attribution": attributed,
        "pooled_cosine": pooled_cosine,
        "k_evaluated": k_evaluated,
        "agreement_with_input": partition_agreement(
            [t.get("speaker") for t in turns],
            [t["speaker"] for t in new_turns],
        ),
        "window_pass": {
            "windows": len(window_pass.starts),
            "hop": window_pass.global_hop,
            "speech_gate": round(window_pass.gate, 4),
            "embedded": window_pass.embedded,
            "k_eigengap": window_pass.k_eigengap,
            "eigenvalues": window_pass.eigenvalues,
            "proposed_boundaries": n_proposed,
            "window_boundaries_used": split_stats["window_boundaries"],
        },
    }


# ---------------------------------------------------------------------------
# Word regrouping by speaker segments (shared by the windows-first engine and
# POST …/reanalyze-with-segments; moved from main.py 2026-08-30)
# ---------------------------------------------------------------------------

def _word_span(w) -> "tuple[float, float] | None":
    """A stored word's (start, end) in seconds, or None when either is
    unusable. Accepts the transcriber's ``start_time``/``end_time``
    (audio_ingest) and the bare ``start``/``end`` spelling defensively."""
    if not isinstance(w, dict):
        return None
    try:
        start = float(w.get("start_time", w.get("start")))
        end = float(w.get("end_time", w.get("end")))
    except (TypeError, ValueError):
        return None
    if end < start:
        return None
    return start, end


def _segment_triple(sg) -> tuple[float, float, str]:
    """``(start, end, label)`` of a segment given as a dict or as an object
    with ``start`` / ``end`` / ``label`` attributes (main.SpeakerSegment)."""
    if isinstance(sg, dict):
        return float(sg["start"]), float(sg["end"]), str(sg["label"]).strip()
    return float(sg.start), float(sg.end), str(sg.label).strip()


def _regroup_tokens(
    rows: list[dict], segments: list, *, snap_s: float = SPEAKER_SEGMENT_SNAP_S,
) -> list[dict]:
    """:func:`regroup_transcript_by_segments` with each turn's source
    utterance index kept under ``utterance`` (the windows engine needs it
    for its agreement-with-input diagnostic)."""
    segs = [_segment_triple(sg) for sg in segments]
    segs.sort(key=lambda t: (t[0], t[1]))
    starts = [a for a, _, _ in segs]

    def _locate(mid: float) -> "str | None":
        i = bisect.bisect_right(starts, mid) - 1
        if i >= 0 and segs[i][0] <= mid <= segs[i][1]:
            return segs[i][2]
        best: str | None = None
        best_d = snap_s
        for j in (i, i + 1):
            if 0 <= j < len(segs):
                a, b, lab = segs[j]
                d = a - mid if mid < a else max(mid - b, 0.0)
                if d <= best_d:
                    best_d, best = d, lab
        return best

    def _nearest_any(mid: "float | None") -> str:
        if mid is None:
            return segs[0][2]
        return min(
            segs,
            key=lambda t: (t[0] - mid if mid < t[0] else max(mid - t[1], 0.0)),
        )[2]

    # tokens: [utterance index, text, start|None, end|None, label|None]
    tokens: list[list] = []
    for ui, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        words = row.get("words")
        spans: list[tuple[str, "float | None", "float | None"]] = []
        if isinstance(words, list) and words:
            for w in words:
                text = str((w or {}).get("word") or "").strip() if isinstance(w, dict) else ""
                if not text:
                    continue
                span = _word_span(w)
                spans.append((text, span[0], span[1]) if span else (text, None, None))
        if not spans:
            # Proportional fallback: spread the row's words evenly over its span.
            pieces = str(row.get("text") or "").split()
            span = _word_span(row)
            n = len(pieces)
            for k, piece in enumerate(pieces):
                if span and span[1] > span[0]:
                    st = span[0] + (span[1] - span[0]) * k / n
                    en = span[0] + (span[1] - span[0]) * (k + 1) / n
                    spans.append((piece, st, en))
                elif span:
                    spans.append((piece, span[0], span[1]))
                else:
                    spans.append((piece, None, None))
        for text, st, en in spans:
            label = _locate((st + en) / 2.0) if st is not None and en is not None else None
            tokens.append([ui, text, st, en, label])

    # Neighbour fill within each utterance: previous word, then next word.
    for idx, tok in enumerate(tokens):
        if tok[4] is None and idx > 0 and tokens[idx - 1][0] == tok[0]:
            tok[4] = tokens[idx - 1][4]
    for idx in range(len(tokens) - 2, -1, -1):
        tok = tokens[idx]
        if tok[4] is None and tokens[idx + 1][0] == tok[0]:
            tok[4] = tokens[idx + 1][4]
    # A whole utterance nobody could place: the last label placed before it,
    # else the nearest segment however far.
    last: str | None = None
    for tok in tokens:
        if tok[4] is None:
            mid = (tok[2] + tok[3]) / 2.0 if tok[2] is not None and tok[3] is not None else None
            tok[4] = last if last is not None else _nearest_any(mid)
        last = tok[4]

    turns: list[dict] = []
    for ui, text, st, en, label in tokens:
        cur = turns[-1] if turns else None
        if cur is not None and cur["speaker"] == label and cur["utterance"] == ui:
            cur["_words"].append(text)
            if en is not None:
                cur["end_time"] = en if cur["end_time"] is None else max(cur["end_time"], en)
            continue
        turns.append({
            "speaker": label, "_words": [text], "utterance": ui,
            "start_time": st, "end_time": en,
        })
    return [
        {
            "speaker": t["speaker"],
            "text": " ".join(t["_words"]),
            "start_time": t["start_time"],
            "end_time": t["end_time"],
            "utterance": t["utterance"],
        }
        for t in turns
    ]


def regroup_transcript_by_segments(
    rows: list[dict], segments: list, *, snap_s: float = SPEAKER_SEGMENT_SNAP_S,
) -> list[dict]:
    """Rebuild a transcript so its SPEAKERS follow ``segments`` while its
    WORDS stay the transcriber's.

    ``segments`` are ``{start, end, label}`` dicts (or objects with those
    attributes), seconds from the start of the audio, any order, ideally
    non-overlapping. Every word of every row (the row's stored per-word
    timings when it has them; otherwise the row's text split on whitespace
    and spread evenly over the row's own [start_time, end_time] — the
    proportional fallback for turns.json rows and word-less transcribers)
    is assigned to the segment containing its midpoint, else the nearest
    segment within ``snap_s``, else it stays with its neighbours in the
    same utterance (previous word first, then next, then the last word
    placed anywhere before it, then the nearest segment however far — a
    word is never dropped). Consecutive words with the same label form one
    turn ``{speaker, text, start_time, end_time}``; a turn ALSO breaks at
    the original utterance boundary so the transcriber's turn granularity
    — what per-turn heat is scored on — is kept, and a welded utterance is
    split exactly at the voice change.

    Blank rows contribute nothing; a transcript with no words at all yields
    ``[]``."""
    return [
        {k: v for k, v in t.items() if k != "utterance"}
        for t in _regroup_tokens(rows, segments, snap_s=snap_s)
    ]


# ---------------------------------------------------------------------------
# Windows-first engine (2026-08-30; see the WINDOWS_FIRST_* constants)
# ---------------------------------------------------------------------------

def _uncovered_segment_turns(
    candidates: list[tuple[float, float]], segments: list[dict], *, cap: int,
    min_seconds: float = UNCOVERED_MIN_SECONDS,
) -> list[dict]:
    """Turn dicts (text UNTRANSCRIBED_TEXT) for the ``cap`` LONGEST
    uncovered speech runs, each cut where the segment timeline changes
    label; a piece under ``min_seconds`` is absorbed into its longer
    neighbour inside the run. Pure Python."""
    if cap <= 0 or not candidates:
        return []
    ranked = sorted(
        [(s, e) for s, e in candidates if e > s],
        key=lambda se: (-(se[1] - se[0]), se[0]),
    )[:cap]
    out: list[dict] = []
    for s, e in sorted(ranked):
        pieces = [
            [max(s, sg["start"]), min(e, sg["end"]), sg["label"]]
            for sg in segments if min(e, sg["end"]) > max(s, sg["start"])
        ]
        if not pieces:
            continue
        while len(pieces) > 1:
            lens = [b - a for a, b, _ in pieces]
            i = int(np.argmin(lens))
            if lens[i] >= min_seconds:
                break
            cand = [j for j in (i - 1, i + 1) if 0 <= j < len(pieces)]
            j = max(cand, key=lambda j: pieces[j][1] - pieces[j][0])
            pieces[i][2] = pieces[j][2]
            merged: list[list] = []
            for a, b, lab in pieces:
                if merged and merged[-1][2] == lab:
                    merged[-1][1] = b
                else:
                    merged.append([a, b, lab])
            pieces = merged
        for a, b, lab in pieces:
            out.append({
                "speaker": lab, "text": UNTRANSCRIBED_TEXT,
                "start_time": round(a, 3), "end_time": round(b, 3),
            })
    return out


def _resolve_embedders(embed_fn, embed_batch_fn):
    """``(embed, embed_batch)`` — the injected functions, or the real model
    (a loop over ``embed_fn`` stands in for the batch when only that is
    injected). Shared by both engines."""
    embed = embed_fn or _default_embed
    if embed_batch_fn is not None:
        embed_batch = embed_batch_fn
    elif embed_fn is not None:
        def embed_batch(chunks, sr_):
            return [embed_fn(c, sr_) for c in chunks]
    else:
        embed_batch = speaker_id.embed_pcm_batch
    return embed, embed_batch


def diarize_windows_first(
    pcm: np.ndarray,
    sr: int,
    turns: list[dict],
    *,
    embed_fn=None,
    embed_batch_fn=None,
) -> dict | None:
    """Transcript-FREE speaker labelling (the ``windows`` engine): label the
    audio from the whole-clip window pass alone, then regroup the
    transcript's words by the resulting speaker segments.

    Pipeline (bake-off approach B, docs/research/2026-08-29-voice-separation/
    B-sliding-window/, run end to end): :class:`_WindowPass` over the clip's
    speech (WINDOW_PASS_* — same grid, cap, hop widening and gate as the
    utterance engine's pass) → refined affinity → eigengap k over
    1..WINDOWS_FIRST_MAX_K → spectral labels → mode filter
    (SPECTRAL_SMOOTH_HOPS) → label runs, runs under SPECTRAL_MIN_RUN_SECONDS
    absorbed into their longer neighbour → segments named in order of first
    appearance → :func:`regroup_transcript_by_segments` (a turn breaks at a
    label change AND at the original utterance boundary). Costs the window
    embeddings plus one pooled embed per found voice (the ``pooled_cosine``
    diagnostic).

    Returns ``None`` — the caller falls back to :func:`diarize_turns` — when
    the voice model is unavailable, fewer than two speech windows exist, the
    eigengap says ONE voice, or smoothing leaves a single label. Otherwise
    the same shape as :func:`diarize_turns` (``turns``, ``num_speakers``,
    ``k_evaluated`` with the eigenvalues, ``agreement_with_input``, …) with
    ``source`` = SOURCE_WINDOWS and ``segments`` = the
    ``[{start, end, label}]`` timeline the words were regrouped by.

    Measured 2026-08-30 (frame accuracy vs ground truth, score.py; this
    engine vs the utterance engine): the owner's real Deepgram transcripts —
    poker 0.720 (k=7) vs 0.447 (k=4), maggiano3 0.694 / 0.681 vs 0.702 /
    0.671 (k=3 both), family 0.949 vs 0.974; ground-truth boundaries —
    family_real 0.980 vs 1.000, poker6 1.000 vs 1.000, the five TTS fixtures
    1.000 vs 1.000 except meeting4 0.818 (k=3) vs 0.597 (k=2), maggiano3's
    rubric 0.865 vs 0.833. Pins: tests/test_diarize_regression_ladder.py,
    tests/test_diarize_scenes.py, tests/test_diarize_private.py
    (``test_windows_first_*``); the full two-engine table is produced by
    docs/research/2026-08-29-voice-separation/baseline/run.py
    (results_windows.json). 2-6 s per 30-40 s fixture at 4 torch threads.
    """
    embed, embed_batch = _resolve_embedders(embed_fn, embed_batch_fn)
    try:
        window_pass = _WindowPass(pcm, sr, embed_batch, embed)
        window_pass.run_global()
        if window_pass.affinity is None or len(window_pass.starts) < 2:
            logger.info(
                "windows engine: %d speech window(s) — nothing to cluster",
                len(window_pass.starts),
            )
            return None
        k_raw, eigenvalues = _dsw.eigengap_k(window_pass.affinity, WINDOWS_FIRST_MAX_K)
        if k_raw < 2:
            logger.info(
                "windows engine heard one voice (eigengap k=1, eigenvalues %s)",
                eigenvalues,
            )
            return None
        k = min(k_raw, len(window_pass.starts))
        labels = _dsw.spectral_labels(window_pass.affinity, k)
        smoothed = _dsw.mode_filter(labels, window_pass.starts, window_pass.global_hop)
        duration = pcm.size / sr
        runs = _dsw.window_label_runs(
            smoothed, window_pass.starts, window_pass.window, 0.0, duration,
        )
        name_of: dict[int, str] = {}
        segments: list[dict] = []
        for b, e, lab in runs:
            if lab not in name_of:
                name_of[lab] = _speaker_name(len(name_of))
            segments.append({
                "start": round(float(b), 3), "end": round(float(e), 3),
                "label": name_of[lab],
            })
        if len(name_of) < 2:
            logger.info(
                "windows engine: eigengap k=%d but one label survives smoothing "
                "(%d windows) — nothing to say", k_raw, len(window_pass.starts),
            )
            return None
        # Pooled centroid per found voice (its segments' audio, capped at
        # MAX_POOL_SECONDS) → the worst pairwise cosine, the utterance
        # engine's acceptance quantity, reported for the logs only.
        cap = int(MAX_POOL_SECONDS * sr)
        seconds_of: dict[str, float] = {}
        pooled: dict[str, np.ndarray] = {}
        for lab in name_of.values():
            spans = [(int(s["start"] * sr), int(s["end"] * sr)) for s in segments if s["label"] == lab]
            seconds_of[lab] = round(sum((b - a) / sr for a, b in spans), 2)
            audio = np.concatenate([pcm[a:b] for a, b in spans])[:cap]
            pooled[lab] = speaker_id.l2_normalize(embed(np.ascontiguousarray(audio), sr))
        labs = sorted(pooled)
        pooled_cosine = max(
            (float(np.dot(pooled[a], pooled[b])) for i, a in enumerate(labs) for b in labs[i + 1:]),
            default=0.0,
        )
    except speaker_id.SpeakerIdUnavailable as exc:
        logger.info("windows engine unavailable: %s", exc)
        return None

    regrouped = _regroup_tokens(turns, segments)
    # A turn with a time span but no words at all (boundary-only input, as
    # the regression fixtures feed) is still a turn: it takes the label of
    # the segment it overlaps most and keeps its (empty) text.
    placed = {t["utterance"] for t in regrouped}
    for ui, row in enumerate(turns):
        span = _word_span(row) if isinstance(row, dict) else None
        if ui in placed or span is None or span[1] <= span[0]:
            continue
        best = max(
            segments,
            key=lambda sg: _overlap_seconds(span[0], span[1], sg["start"], sg["end"]),
        )
        regrouped.append({
            "speaker": best["label"], "text": str(row.get("text") or ""),
            "start_time": span[0], "end_time": span[1], "utterance": ui,
        })
    regrouped.sort(key=lambda t: (t["utterance"], float(t["start_time"] or 0.0)))
    if not regrouped:
        logger.info("windows engine: the transcript has nothing to regroup")
        return None
    new_turns = [{k_: v for k_, v in t.items() if k_ != "utterance"} for t in regrouped]

    # Speech the transcript never covered becomes "(untranscribed)" turns
    # labelled by the segment timeline — the same UNCOVERED_* rules as the
    # utterance engine, but labelled by the instrument that labelled
    # everything else rather than by the nearest utterance, and capped at
    # the transcript's own utterance count + UNCOVERED_MAX_EXTRA rather than
    # a fifth of it: on the owner's poker recording Deepgram transcribed
    # 7 utterances covering 64 % of the speech (whole players missing), and
    # the missing 36 % is exactly the speech this engine can label.
    uncovered = _uncovered_segment_turns(
        _uncovered_speech(
            window_pass.mask, window_pass.frame_s, turns, duration,
            bridge=WINDOWS_FIRST_UNCOVERED_BRIDGE_SECONDS,
        ),
        segments, cap=len(turns) + UNCOVERED_MAX_EXTRA,
    )
    if uncovered:
        logger.info(
            "windows engine: %d run(s) of speech outside every utterance become "
            "%r turns (%s)",
            len(uncovered), UNTRANSCRIBED_TEXT,
            ", ".join(
                f"{t['start_time']:.2f}-{t['end_time']:.2f} {t['speaker']}" for t in uncovered
            ),
        )
        new_turns, index_map = _insert_chronological(new_turns, uncovered)
    speakers_out = []
    for t in new_turns:
        if t["speaker"] not in speakers_out:
            speakers_out.append(t["speaker"])
    num_speakers = len(speakers_out)
    if num_speakers < 2:
        logger.info(
            "windows engine: %d voices found but the transcript's words all "
            "fall on one — nothing to say", len(name_of),
        )
        return None
    from_utterance = [t["utterance"] for t in regrouped]
    n_split = sum(1 for ui in set(from_utterance) if from_utterance.count(ui) > 1)
    for t in uncovered:
        if t["speaker"] not in speakers_out:
            speakers_out.append(t["speaker"])
    num_speakers = len(speakers_out)
    entry = {
        "k": k, "ok": True, "route": "spectral-windows",
        "k_eigengap": k_raw, "eigenvalues": eigenvalues,
        "voices_after_smoothing": len(name_of),
        "max_pairwise_cosine": round(pooled_cosine, 3),
        "min_cluster_seconds": min(seconds_of.values()),
        "cluster_seconds": seconds_of,
    }
    logger.info(
        "windows engine: %d speech windows (%.2fs hop, gate %.4f RMS), eigengap "
        "k=%d (eigenvalues %s) -> %d segment(s), %d voice(s) in the transcript, "
        "%d utterance(s) broken at a voice change, worst pooled cosine %.3f, "
        "cluster seconds %s",
        len(window_pass.starts), window_pass.global_hop, window_pass.gate, k_raw,
        eigenvalues, len(segments), num_speakers, n_split, pooled_cosine, seconds_of,
    )
    return {
        "turns": new_turns,
        "num_speakers": num_speakers,
        "source": SOURCE_WINDOWS,
        "model": f"{speaker_id.ECAPA_SOURCE}@{speaker_id.ECAPA_REVISION}",
        "segments_total": len(new_turns),
        "segments_embedded": len(window_pass.starts),
        "split_utterances": n_split,
        "confirmed_short_pieces": 0,
        "uncovered_turns": len(uncovered),
        "unknown_turns": 0,
        "unknown_seconds": 0.0,
        "short_turn_attribution": {"self": 0, "neighbour": 0},
        "pooled_cosine": round(pooled_cosine, 3),
        "k_evaluated": [entry],
        "agreement_with_input": partition_agreement(
            [turns[ui].get("speaker") for ui in from_utterance],
            [t["speaker"] for t in regrouped],
        ),
        "window_pass": {
            "windows": len(window_pass.starts),
            "hop": window_pass.global_hop,
            "speech_gate": round(window_pass.gate, 4),
            "embedded": window_pass.embedded,
            "k_eigengap": k_raw,
            "eigenvalues": eigenvalues,
            "proposed_boundaries": len(segments) - 1,
            "window_boundaries_used": len(segments) - 1,
        },
        "segments": segments,
    }
