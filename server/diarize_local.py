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
"""

from __future__ import annotations

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
# each piece only needs MIN_SECONDS to be attributable.
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
) -> tuple[list[int], list[float]]:
    """Nearest-centroid label + confidence margin for each word of ``turn``.

    Each word is scored by embedding ``window`` seconds of audio centered on
    the word's midpoint (clamped to the utterance bounds) against ALL pooled
    centroids. The margin is best-minus-second-best cosine — low margin means
    the window does not clearly favor any one voice.
    """
    start = float(turn.get("start_time") or 0.0)
    end = float(turn.get("end_time") or 0.0)
    labels: list[int] = []
    margins: list[float] = []
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
    return labels, margins


def _smooth_word_labels(
    labels: list[int], margins: list[float],
    *, min_margin: float = WORD_MIN_MARGIN,
) -> list[int] | None:
    """Ambiguous words inherit the nearest confident word's label.

    Pure math. A word is confident when its margin clears ``min_margin``;
    every other word takes the label of the nearest confident word (by word
    index; tie → the earlier one — voices persist forward). No confident word
    at all → ``None``: the scan has nothing trustworthy to say.
    """
    conf = [i for i, m in enumerate(margins) if m >= min_margin]
    if not conf:
        return None
    return [
        labels[i] if margins[i] >= min_margin
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


def _collapse_word_runs(
    words: list[dict], labels: list[int], start: float, end: float,
    *, min_run: int = WORD_MIN_RUN, min_seconds: float = MIN_SECONDS,
) -> list[int]:
    """Merge untrustworthy label runs into their surrounding dominant voice.

    Pure math. A run is untrustworthy when it carries fewer than ``min_run``
    words OR its piece would last under ``min_seconds`` (too little voice
    signal to attribute honestly). The weakest such run (fewest words, then
    shortest) inherits the label of its longer-piece neighbor (tie → the
    earlier neighbor) until every surviving run is trustworthy.
    """
    labels = list(labels)
    while True:
        runs = _word_runs(labels)
        if len(runs) == 1:
            return labels
        pieces = _run_pieces(words, runs, start, end)
        bad = [
            (r[2] - r[1] + 1, p1 - p0, idx)
            for idx, (r, (p0, p1)) in enumerate(zip(runs, pieces))
            if (r[2] - r[1] + 1) < min_run or (p1 - p0) < min_seconds
        ]
        if not bad:
            return labels
        idx = min(bad)[2]
        if idx == 0:
            tgt = 1
        elif idx == len(runs) - 1:
            tgt = idx - 1
        else:
            prev_d = pieces[idx - 1][1] - pieces[idx - 1][0]
            next_d = pieces[idx + 1][1] - pieces[idx + 1][0]
            tgt = idx - 1 if prev_d >= next_d else idx + 1
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
        keys = sorted(cents)
        piece_label: list[int] = []
        piece_margin: list[float] = []
        for b, e in zip(bounds[:-1], bounds[1:]):
            v = speaker_id.l2_normalize(self.embed(
                np.ascontiguousarray(self.pcm[int(b * self.sr):int(e * self.sr)]), self.sr,
            ))
            scored = sorted((float(np.dot(v, cents[c])), c) for c in keys)
            piece_label.append(scored[-1][1])
            piece_margin.append(scored[-1][0] - (scored[-2][0] if len(scored) > 1 else -1.0))
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
) -> list[float]:
    """Drop boundaries until every piece of [start, end] lasts at least
    ``min_seconds``. For the shortest sliver piece, the boundary dropped is a
    NON-``primary`` one when the sliver has one on each kind of side (a
    word-pass cut outranks a window-pass proposal), else the one whose
    removal merges the sliver into its shorter neighbour — so a sliver never
    survives as its own piece."""
    prim = set(primary or [])
    bounds = [start, *sorted(b for b in boundaries if start < b < end), end]
    while len(bounds) > 2:
        lens = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        i = int(np.argmin(lens))
        if lens[i] >= min_seconds:
            break
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

    Returns ``(finer_turns, stats)`` where stats counts ``scanned``,
    ``split``, ``skipped_short`` (too short for any instrument — bounded
    compute, by design), ``skipped_no_words`` (long enough for the sustained
    scan but no word timings AND no window proposal — logged, never hidden)
    and ``window_boundaries`` (cuts that came from the window pass alone).
    Turns that yield no trustworthy change pass through unchanged.
    """
    stats = {
        "scanned": 0, "split": 0, "skipped_short": 0, "skipped_no_words": 0,
        "window_boundaries": 0,
    }
    cents = list(word_centroids) if word_centroids is not None else list(centroids)
    proposals = proposals or {}
    out: list[dict] = []
    for idx, t in enumerate(turns):
        start = float(t.get("start_time") or 0.0)
        end = float(t.get("end_time") or 0.0)
        words = t.get("words")
        has_words = isinstance(words, list) and len(words) >= 2
        long_enough = end - start > WORD_SPLIT_MIN_UTTERANCE_SECONDS
        scanned = False
        conclusive = False
        word_bounds: list[float] = []
        if has_words and long_enough:
            scanned = True
            labels, margins = _label_words(pcm, sr, t, embed, cents)
            smoothed = _smooth_word_labels(labels, margins)
            if smoothed is not None:
                conclusive = True
                runs = _word_runs(_collapse_word_runs(words, smoothed, start, end))
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
            primary=word_bounds,
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
        stats["split"] += 1
        stats["window_boundaries"] += from_windows
        logger.info(
            "split %.1fs utterance at %.2fs into %d pieces (boundaries %s; "
            "%d from the window pass%s)",
            end - start, start, len(pieces),
            ", ".join(f"{p['end_time']:.2f}" for p in pieces[:-1]),
            from_windows,
            "" if has_words else "; no word timings — text divided by duration",
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
) -> tuple[list[int], list[np.ndarray]] | None:
    """Embed every turn long enough to carry voice signal.

    Returns ``(order, embs)`` — the embeddable turn indices and their
    normalized embeddings — or ``None`` with fewer than two embeddable turns
    (nothing trustworthy to cluster).
    """
    embedded: dict[int, np.ndarray] = {}
    for idx, t in enumerate(turns):
        start = float(t.get("start_time") or 0.0)
        end = float(t.get("end_time") or 0.0)
        if end - start < min_seconds:
            continue
        embedded[idx] = speaker_id.l2_normalize(embed(_slice(pcm, sr, t), sr))

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
    embed = embed_fn or _default_embed
    if embed_batch_fn is not None:
        embed_batch = embed_batch_fn
    elif embed_fn is not None:
        def embed_batch(chunks, sr_):
            return [embed_fn(c, sr_) for c in chunks]
    else:
        embed_batch = speaker_id.embed_pcm_batch
    try:
        embedded = _embed_turns(pcm, sr, turns, embed, min_seconds)
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
            if end - start <= WORD_SPLIT_MIN_UTTERANCE_SECONDS:
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
        turns, split_stats = split_long_utterances(
            pcm, sr, turns, embed, (pass1[1][0], pass1[1][1]),
            word_centroids=word_centroids, proposals=proposals,
        )
        if split_stats["split"]:
            embedded = _embed_turns(pcm, sr, turns, embed, min_seconds)
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

    # Un-embedded (too-short) turns inherit the nearest embedded turn's
    # cluster, nearest by utterance midpoint.
    def midpoint(t: dict) -> float:
        return (float(t.get("start_time") or 0.0) + float(t.get("end_time") or 0.0)) / 2

    for idx, t in enumerate(turns):
        if idx in cluster_of:
            continue
        nearest = min(order, key=lambda e: abs(midpoint(turns[e]) - midpoint(t)))
        cluster_of[idx] = cluster_of[nearest]

    # Name clusters in order of first appearance across the full transcript.
    name_of: dict[int, str] = {}
    for idx in range(len(turns)):
        cid = cluster_of[idx]
        if cid not in name_of:
            name_of[cid] = _speaker_name(len(name_of))

    # Output turns are plain {speaker, text, start_time, end_time, ...} — the
    # internal ``words`` plumbing is not propagated.
    new_turns = [
        dict({k: v for k, v in t.items() if k != "words"},
             speaker=name_of[cluster_of[i]])
        for i, t in enumerate(turns)
    ]
    return {
        "turns": new_turns,
        "num_speakers": len(name_of),
        "source": SOURCE,
        "model": f"{speaker_id.ECAPA_SOURCE}@{speaker_id.ECAPA_REVISION}",
        "segments_total": len(turns),
        "segments_embedded": len(order),
        "split_utterances": split_stats["split"],
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
