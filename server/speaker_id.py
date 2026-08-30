"""Speaker identity via ECAPA-TDNN voice embeddings — an OPTIONAL dependency.

This powers two features, both grounded in the existing diarization the pipeline
already runs (Deepgram gives us *which turn belongs to which speaker within one
recording*; this module gives those anonymous "Speaker A/B/…" clusters a
*persistent* identity across recordings):

* **Enrollment** ("This is me"): pool one diarized speaker's turns from a stored
  recording, embed them into a single 192-d voiceprint, and average it into the
  user's stored profile. Multiple enrollments refine the print (a running mean).
* **Auto-labeling** ("You" / a named person): during analysis, embed each
  diarized speaker's pooled turns and cosine-match against EVERY voiceprint the
  account holds — the owner's own ("self" → "You") plus any partners the user
  named ("alex" → "Alex"). Greedy one-to-one assignment, best score first; a
  pair is labeled only above :data:`MATCH_THRESHOLD` (``label_source="enrolled"``
  — the TOP rung of the display-label ladder). Below threshold → NO label, ever.
  See :func:`identify_speakers_multi`; :func:`identify_speakers` is the
  original single-print entry point, kept as a thin wrapper.

Honesty / availability notes (house rule: report unavailable, never fabricate):

* ``torch`` + ``speechbrain`` are heavy (hundreds of MB) and are kept OUT of the
  base requirements — see ``requirements-voice.txt``. They are imported LAZILY
  inside the functions that need them. :func:`is_available` is the load-bearing
  "can we do voice ID at all?" check; when it is False the router returns an
  honest 503 and the analysis pipeline skips matching cleanly (no crash, no
  label). The base test suite therefore stays green WITHOUT torch installed.
* The loaded model is cached process-wide (one copy shared by every request),
  loaded once under a lock — the SpeechBrain checkpoint is ~20MB but the first
  load (import torch + build the graph, plus a first-run HF download) is slow, so
  it must never happen more than once.
* The model revision is PINNED (:data:`ECAPA_REVISION`) so a silent upstream
  reweight can't move the embedding space — and thus the threshold — under us.

The pure vector math (pool / cosine / running-mean) has NO torch dependency and
is unit-tested directly; only :func:`embed_pcm` (and the two orchestrators that
call it) touch the model.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid

import numpy as np

logger = logging.getLogger(__name__)

# ECAPA-TDNN on VoxCeleb emits a 192-d speaker embedding.
EMBEDDING_DIM = 192

# The pretrained checkpoint + a PINNED revision. Pinning the revision keeps the
# embedding space (and therefore the calibrated threshold below) stable against a
# silent upstream reweight. Overridable via env for a controlled model bump.
# NOTE: the default is the current `main` commit of the HF repo at integration
# time; the empirical validation script records which revision produced the
# reported cosine scores.
ECAPA_SOURCE = os.getenv("MINDSHIFT_ECAPA_SOURCE", "speechbrain/spkrec-ecapa-voxceleb")
ECAPA_REVISION = os.getenv(
    "MINDSHIFT_ECAPA_REVISION", "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
)

# Cosine-similarity floor for calling a speaker "You". CHOSEN FROM EMPIRICAL
# EVIDENCE against the owner's real recordings + the two-speaker fixtures
# (tmp/voice_validate.py in the PR, ECAPA @ the pinned revision):
#
#   same person, clean cross-recording ....... 0.727   (owner in two clips)
#   same voice, clean fixture split-half ..... 0.72–0.90
#   different people, same clip .............. 0.252   (owner vs spouse)
#   different people, cross-recording ........ 0.11–0.16
#   merged/degraded-diarization artifact ..... 0.477–0.558
#
# 0.65 sits in the clean gap between the ~0.55 ambiguous/merged artifacts and the
# 0.72+ genuine same-voice matches: it accepts a true match with margin while
# rejecting the spouse-in-a-merged-clip case. A FALSE "You" (mislabeling another
# person) is the cardinal sin here, so we bias toward misses — below this floor a
# speaker keeps its generic label; we NEVER force a match. Overridable via env.
MATCH_THRESHOLD = float(os.getenv("MINDSHIFT_VOICE_MATCH_THRESHOLD", "0.65"))

# The diarizer's label for speech that sounds like NONE of a recording's found
# voices (diarize_local, 2026-08-30). Defined here — the torch-free module
# every consumer already imports — so enrollment matching, dynamics and the
# diarizer agree on the one string without a circular import. Never a real
# speaker: excluded from enrollment matching, talk share, coupling, report
# cards.
UNKNOWN_SPEAKER = "Unknown"

# CROSS-RECORDING (contrast) match — a second, narrower way to clear the bar.
#
# MEASURED 2026-08-27 on the owner's REAL recordings (pinned ECAPA; the
# family_real + poker6 fixtures and the private 3-person family clip): the
# same person's clean voice across DIFFERENT settings scores only 0.24-0.45
# against a print built from ONE other setting (restaurant vs kitchen vs
# poker table — room, mic distance and register all move the embedding),
# while different people score 0.11-0.28. The 0.65 bar above was calibrated
# on a same-setting pair (0.727) and is simply never reached across settings
# — the owner's own voice scored 0.36 (poker night) and 0.36 (family clip)
# against the print he enrolled from the restaurant clip. No single absolute
# threshold separates 0.24-0.45 from 0.11-0.28.
#
# Two things DO separate them, and both are required here:
#   1. a print pooled from at least CROSS_MATCH_MIN_SETTINGS distinct
#      recordings (see blend_samples — one centroid per recording, so three
#      taps on the same clip don't outvote a different room). A 2-setting
#      print lifted the owner's out-of-sample poker score 0.36 -> 0.39 and
#      the family clip to 0.55; non-owners stayed <= 0.22.
#   2. CONTRAST inside the recording: the matched speaker must beat every
#      OTHER speaker's score for that person by CROSS_MATCH_MARGIN. Measured
#      owner-vs-runner-up gaps: 0.16 (poker), 0.37 (family), 0.63
#      (restaurant); different-people gaps sit within ~0.1 of each other.
#      A solo recording has no contrast and stays on the 0.65 bar.
# CROSS_MATCH_THRESHOLD (0.40) sits above every different-person score on
# record (max 0.28, n≈12 voices) with a margin the +/-0.05 run-to-run
# variance needs; the owner's single-setting misses below it (0.24-0.39) are
# the honest failure direction — a false "You" is still the cardinal sin.
# Every match records its ``match_basis`` ("absolute" | "contrast") so a
# contrast match is never mistaken for a 0.65 one. All env-overridable.
CROSS_MATCH_THRESHOLD = float(os.getenv("MINDSHIFT_VOICE_CROSS_MATCH_THRESHOLD", "0.40"))
CROSS_MATCH_MARGIN = float(os.getenv("MINDSHIFT_VOICE_CROSS_MATCH_MARGIN", "0.15"))
CROSS_MATCH_MIN_SETTINGS = int(os.getenv("MINDSHIFT_VOICE_CROSS_MATCH_MIN_SETTINGS", "2"))

# "Learn this voice from a recording" guard (people labeling). Before a pooled
# speaker embedding is appended to person P's print, it is scored against
# EVERY other enrolled person. If it clears MATCH_THRESHOLD against someone
# ELSE — i.e. the matcher would already call this voice "Mom" — AND it is not
# at least this much closer to P (or P has no print yet), the enrollment is
# refused: the user has most likely tapped the wrong speaker, and appending a
# second person's voice to P's print would poison it (every later match would
# then confuse the two). 0.10 is the width of the gap between the ~0.55
# merged-diarization artifacts and the 0.72+ genuine same-voice scores in the
# calibration table above: a voice genuinely closer to P by that much is P's,
# even if it also resembles someone else (siblings, parent/child). Overridable.
ENROLL_CONFLICT_MARGIN = float(os.getenv("MINDSHIFT_VOICE_ENROLL_CONFLICT_MARGIN", "0.10"))

# ECAPA is trained on 16 kHz audio; our stored derivatives + live contract are
# already 16 kHz mono, so no resample is normally needed. Kept explicit so a
# mismatched input is caught, not silently mis-embedded.
TARGET_SR = 16000

# A speaker cluster shorter than this (total pooled speech) is too little signal
# for a trustworthy embedding — we skip it (no score, no label) rather than
# guess. ~1s is plenty for pooled matching; enrollment wants a touch more.
MIN_MATCH_SECONDS = 1.0
MIN_ENROLL_SECONDS = 3.0

# Guided (direct-upload) enrollment measures ACTUAL speech, not clip length —
# a long silent clip must not enroll. Frames whose RMS clears the floor count
# as speech; 0.01 (~ -40 dBFS) sits well below quiet speech (~0.03-0.1 RMS)
# and above room-tone/handling noise on phone mics.
SPEECH_FRAME_MS = 30.0
SPEECH_RMS_THRESHOLD = 0.01

# NOISE-FLOOR-RELATIVE speech gate (2026-08-29, voice-separation bake-off,
# docs/research/2026-08-29-voice-separation/B-sliding-window/README.md):
# the effective gate is ``max(absolute floor, SPEECH_RMS_FLOOR_MULT x the
# SPEECH_NOISE_FLOOR_PERCENTILE-th percentile of frame RMS)``. Measured: the
# real 6-speaker poker clip's quietest player has a MEDIAN frame RMS of
# 0.0036 against a room floor of 0.0032, so any absolute gate near 0.01 (or a
# peak-relative one) silently drops a whole real speaker, while the TTS
# fixtures' gaps are digital silence (RMS ~0, p10 ~0 → the absolute floor
# rules) and family_real's gaps sit at 0.0035-0.005 against the child's
# 0.012 median, so 1.5 x floor still separates them. With this gate every
# speaker on every fixture keeps >= 76 % of his 1.5 s windows.
#   * :func:`speech_seconds` (enrollment) keeps SPEECH_RMS_THRESHOLD (0.01)
#     as its absolute floor — the relative term only RAISES the gate in a
#     noisy room (room tone no longer counts as speech); a silent upload
#     still measures 0 s.
#   * The diarizer's window pass (``diarize_local``) uses the lower
#     SPEECH_RMS_FLOOR (0.003, B's calibration) as its absolute floor so a
#     quiet-but-present speaker is windowed, not gated out.
#   * The relative term is capped at SPEECH_RMS_GATE_CEILING: p10 is only a
#     NOISE-floor estimate while at least a tenth of the clip is quiet; a
#     clip with no pauses at all (a sustained tone, wall-to-wall speech) has
#     a p10 at speech level and must not gate itself out. 0.03 is the bottom
#     of the "quiet speech" range in the comment above. Measured 2026-08-29:
#     every fixture's gate lands at 0.003-0.005 (p10 0.0003-0.0035), and even
#     the frames INSIDE continuous-speech utterances have p10 <= 0.004, so
#     the cap is never reached on real speech.
SPEECH_RMS_FLOOR = 0.003
SPEECH_RMS_FLOOR_MULT = 1.5
SPEECH_NOISE_FLOOR_PERCENTILE = 10.0
SPEECH_RMS_GATE_CEILING = 0.03
# Cap pooled audio per speaker so a very long recording can't make one embed call
# unbounded; the first ~60s of a voice is more than enough identity signal.
MAX_POOL_SECONDS = 60.0

# Stored voiceprint document version. v2 keeps each enrollment's INDIVIDUAL
# embedding (``samples``) alongside the blended voiceprint, so single samples
# can be inspected and deleted; v1 stored only the running-mean blend. A v1 doc
# migrates via :func:`as_v2` — its blend becomes ONE legacy sample.
PROFILE_VERSION = 2

# The deterministic sample id a migrated v1 blend gets. Deterministic on purpose:
# GET /voice/profile serves a v1 doc through the same v2 view WITHOUT rewriting
# it (reads stay side-effect free), so the id must be identical on every read for
# a later DELETE to find it.
LEGACY_SAMPLE_ID = "legacy-blend"

# The label the top rung of the ladder assigns, and its source tag. A concurrent
# feature adds display_label/label_source with sources "name"/"voice"/"generic";
# "enrolled" is designed as the HIGHEST-precedence source.
YOU_LABEL = "You"
LABEL_SOURCE = "enrolled"

# Multi-person voiceprints (Foundation B). An account holds N named people,
# each with its own v2 profile document; EXACTLY ONE of them is the account
# owner ("this is me"), identified by the reserved ``person_id`` below and
# displayed as :data:`YOU_LABEL` (second person, see main.ENROLLED_DISPLAY_LABEL).
# The invariant "one self per account" is enforced STRUCTURALLY rather than by
# a flag scan: ``is_self`` is true if and only if ``person_id == SELF_PERSON_ID``
# (see :func:`as_person`), so two documents can never both claim to be the
# owner — there is only one "self" key. Every other person is a partner the
# user named ("Alex", "Mom") whose display label IS that name; we never invent
# one. ``person_id`` doubles as a storage path segment, hence the tight
# pattern (client-chosen slugs, lowercase, no separators GCS could mis-parse).
SELF_PERSON_ID = "self"
SELF_DISPLAY_NAME = YOU_LABEL
PERSON_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,39}$"
DISPLAY_NAME_MAX = 60

# Process-wide model cache (see module docstring). A threading.Lock (not asyncio)
# because loads happen inside asyncio.to_thread worker threads.
_model = None
_model_lock = threading.Lock()


class SpeakerIdUnavailable(RuntimeError):
    """Voice embedding is not available on this server (torch/speechbrain absent,
    or the model could not be loaded). The router maps this to a 503; the
    analysis pipeline treats it as "skip matching" (no label)."""


# ---------------------------------------------------------------------------
# Availability + model loading (the ONLY torch-touching code)
# ---------------------------------------------------------------------------

def cache_dir() -> str:
    """Where the pinned checkpoint (and, since the on-device seam, the ONNX
    export of it — ``ecapa_onnx.default_onnx_path``) live on disk:
    ``MINDSHIFT_ECAPA_CACHE`` or ``server/.ecapa_cache`` (gitignored). Read
    at call time, not import time, so a test can point it at ``tmp_path``
    and a deploy can point it at a persistent volume."""
    return os.getenv(
        "MINDSHIFT_ECAPA_CACHE",
        os.path.join(os.path.dirname(__file__), ".ecapa_cache"),
    )


def is_available() -> bool:
    """True when the optional voice deps import. Cheap import probe (no model
    load), used by the router (→ honest 503) and the pipeline (→ skip cleanly).
    """
    try:
        import speechbrain  # noqa: F401
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 — any import failure means "not available"
        return False
    return True


def _load_model():
    """Return the shared, cached ECAPA classifier, loading it once under a lock.

    Lazy + guarded exactly like the Whisper model cache: two concurrent requests
    can't both pay the load cost, and the base install (no torch) never imports
    speechbrain at module top. Raises :class:`SpeakerIdUnavailable` when the deps
    are missing or the checkpoint can't be loaded — the caller degrades honestly.
    """
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception as exc:  # noqa: BLE001
            raise SpeakerIdUnavailable(
                "voice enrollment not available on this server — install "
                "requirements-voice.txt (torch + speechbrain)"
            ) from exc
        savedir = cache_dir()
        # Enforce the revision PIN by pre-fetching that exact snapshot to a local
        # dir, then loading from it. This is version-robust: it does NOT rely on
        # SpeechBrain's from_hparams forwarding a `revision=` kwarg (some releases
        # raise TypeError on it) and it sidesteps SpeechBrain's own HF fetch path
        # (whose `use_auth_token=` arg breaks against newer huggingface_hub) — a
        # LOCAL source loads via copy/symlink, not hf_hub_download. If the pinned
        # snapshot can't be fetched we fall back to an UNPINNED load from the Hub
        # (logged, so the pin gap is never silent), rather than failing the whole
        # feature.
        source = ECAPA_SOURCE
        try:
            from huggingface_hub import snapshot_download

            source = snapshot_download(
                repo_id=ECAPA_SOURCE, revision=ECAPA_REVISION, local_dir=savedir,
            )
        except Exception as exc:  # noqa: BLE001 — degrade to unpinned, but say so
            logger.warning(
                "Could not pre-fetch pinned ECAPA revision %s (%s); loading "
                "unpinned from the Hub", ECAPA_REVISION, exc,
            )
        try:
            _model = EncoderClassifier.from_hparams(
                source=source, savedir=savedir, run_opts={"device": "cpu"},
            )
        except Exception as exc:  # noqa: BLE001
            raise SpeakerIdUnavailable(
                f"could not load speaker-embedding model {ECAPA_SOURCE!r}: {exc}"
            ) from exc
        logger.info("Loaded ECAPA speaker model %s @ %s", ECAPA_SOURCE, ECAPA_REVISION)
        return _model


def embed_pcm(pcm: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Embed mono float32 PCM into an L2-normalized 192-d voiceprint (blocking).

    Runs the pretrained model on CPU. Callers off the event loop (they wrap this
    in ``asyncio.to_thread``). Raises :class:`SpeakerIdUnavailable` when the model
    can't be loaded. ``pcm`` is expected at :data:`TARGET_SR` (16 kHz), matching
    our stored derivative + live contract; a different rate is an honest error
    rather than a silent mis-embedding.
    """
    if sr != TARGET_SR:
        raise SpeakerIdUnavailable(
            f"speaker embedding expects {TARGET_SR} Hz audio, got {sr} Hz"
        )
    import torch

    model = _load_model()
    audio = np.ascontiguousarray(pcm, dtype=np.float32)
    with torch.no_grad():
        wav = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
        emb = model.encode_batch(wav)  # (1, 1, 192)
    vec = emb.squeeze().detach().cpu().numpy().astype(np.float32)
    return l2_normalize(vec)


def embed_pcm_batch(chunks: list[np.ndarray], sr: int = TARGET_SR) -> list[np.ndarray]:
    """Embed MANY mono float32 PCM chunks in ONE forward pass (blocking).

    NEW entry point, additive — :func:`embed_pcm` (single-chunk) is UNCHANGED
    and still used by every existing caller (enrollment, per-turn/per-word
    diarization embeds). This exists for callers that need many embeddings of
    SHORT, roughly-fixed-length audio at once (e.g. a sliding-window scan) and
    would otherwise pay the model's fixed per-call overhead once per chunk.

    Measured on this machine (2026-08-22, CPU, isolated ``.ecapa_cache``): a
    single ``embed_pcm`` call costs ~15-20s regardless of chunk length — model
    call overhead dominates, not audio duration or batch size. Batching N
    chunks into one ``encode_batch`` call amortizes that fixed overhead across
    all of them: the speedup approaches N as batch size grows (see
    ``server/tests/test_speaker_id_batch.py`` and
    ``.superpowers/sdd/2026-08-22-poker6-v3-sliding-window-refine/report.md``
    for real before/after wall-clock numbers).

    Chunks may be DIFFERENT lengths (zero-padded to the batch's longest, with
    ``wav_lens`` telling the model each chunk's real relative length so
    padding never leaks into an embedding — this is exactly what
    ``encode_batch``'s ``wav_lens`` parameter is for). Returns one
    L2-normalized 192-d vector per input chunk, in the same order. An empty
    ``chunks`` list returns ``[]`` without touching the model.
    """
    if sr != TARGET_SR:
        raise SpeakerIdUnavailable(
            f"speaker embedding expects {TARGET_SR} Hz audio, got {sr} Hz"
        )
    if not chunks:
        return []
    import torch

    model = _load_model()
    arrays = [np.ascontiguousarray(c, dtype=np.float32) for c in chunks]
    max_len = max(a.size for a in arrays)
    if max_len == 0:
        raise SpeakerIdUnavailable("cannot embed a zero-length audio chunk")
    batch = np.zeros((len(arrays), max_len), dtype=np.float32)
    rel_lens = np.empty(len(arrays), dtype=np.float32)
    for i, a in enumerate(arrays):
        batch[i, : a.size] = a
        rel_lens[i] = a.size / max_len
    with torch.no_grad():
        wavs = torch.from_numpy(batch)  # (batch, max_len)
        wav_lens = torch.from_numpy(rel_lens)  # (batch,) relative lengths
        embs = model.encode_batch(wavs, wav_lens=wav_lens)  # (batch, 1, 192)
    vecs = embs.squeeze(1).detach().cpu().numpy().astype(np.float32)
    return [l2_normalize(v) for v in vecs]


# ---------------------------------------------------------------------------
# Pure vector math — NO torch. Unit-tested directly.
# ---------------------------------------------------------------------------

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit L2 norm (a zero vector is returned as-is —
    cosine against it is 0, which is the honest "no similarity")."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors in [-1, 1]. Operates on L2-normalized
    inputs in practice (stored voiceprints + :func:`embed_pcm` outputs are
    normalized), so this is just their dot product — but it normalizes defensively
    so a caller passing raw vectors still gets a correct cosine."""
    a = l2_normalize(a)
    b = l2_normalize(b)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def pool_speaker_pcm(
    pcm: np.ndarray,
    sr: int,
    turns: list[dict],
    speaker: str,
    *,
    max_seconds: float = MAX_POOL_SECONDS,
) -> np.ndarray:
    """Concatenate the PCM under ``speaker``'s diarized turns into one array.

    Pooling ALL of a speaker's turns (often tens of seconds) is what makes this
    robust: we never match a noisy 1-second utterance, we match a long pooled
    sample. Slices are clamped to the audio bounds; the total is capped at
    ``max_seconds`` (identity signal saturates well before a minute). Returns an
    empty array when the speaker has no usable audio."""
    if pcm.size == 0 or sr <= 0:
        return np.zeros(0, dtype=np.float32)
    max_samples = int(max_seconds * sr)
    chunks: list[np.ndarray] = []
    total = 0
    for t in turns:
        if t.get("speaker") != speaker:
            continue
        start = t.get("start_time")
        end = t.get("end_time")
        if start is None or end is None:
            continue
        i0 = max(0, int(float(start) * sr))
        i1 = min(pcm.size, int(float(end) * sr))
        if i1 <= i0:
            continue
        chunk = pcm[i0:i1]
        chunks.append(chunk)
        total += chunk.size
        if total >= max_samples:
            break
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    pooled = np.concatenate(chunks)
    if pooled.size > max_samples:
        pooled = pooled[:max_samples]
    return np.ascontiguousarray(pooled, dtype=np.float32)


def frame_rms(pcm: np.ndarray, sr: int, *, frame_ms: float = SPEECH_FRAME_MS) -> np.ndarray:
    """RMS of each full ``frame_ms`` frame of ``pcm`` (pure numpy). The
    partial tail frame is dropped; empty audio → an empty array."""
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.size == 0 or sr <= 0:
        return np.zeros(0, dtype=np.float64)
    frame = max(1, int(sr * frame_ms / 1000.0))
    usable = (pcm.size // frame) * frame
    if usable == 0:
        return np.zeros(0, dtype=np.float64)
    frames = pcm[:usable].reshape(-1, frame)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))


def speech_rms_threshold(
    rms: np.ndarray, *, rms_floor: float,
    floor_mult: float = SPEECH_RMS_FLOOR_MULT,
    floor_percentile: float = SPEECH_NOISE_FLOOR_PERCENTILE,
    ceiling: float = SPEECH_RMS_GATE_CEILING,
) -> float:
    """The NOISE-FLOOR-RELATIVE speech gate for a clip whose frame RMS values
    are ``rms``: ``max(rms_floor, min(ceiling, floor_mult x percentile(rms,
    floor_percentile)))``. A silent clip (p10 ~ 0) is gated by the absolute
    ``rms_floor`` alone; a noisy room raises the gate above its own floor;
    a clip with no quiet frames at all is capped at ``ceiling`` (see the
    SPEECH_RMS_FLOOR comment for the measurements)."""
    rms = np.asarray(rms, dtype=np.float64)
    if rms.size == 0:
        return float(rms_floor)
    relative = floor_mult * float(np.percentile(rms, floor_percentile))
    return max(float(rms_floor), min(float(ceiling), relative))


def speech_mask(
    pcm: np.ndarray, sr: int, *,
    frame_ms: float = SPEECH_FRAME_MS,
    rms_floor: float = SPEECH_RMS_FLOOR,
    floor_mult: float = SPEECH_RMS_FLOOR_MULT,
) -> tuple[np.ndarray, float, float]:
    """``(mask, threshold, frame_seconds)`` — one bool per ``frame_ms`` frame
    saying whether it clears the noise-floor-relative gate
    (:func:`speech_rms_threshold` with ``rms_floor`` as the absolute floor).
    Pure numpy; the diarizer's window pass gates its windows with this."""
    rms = frame_rms(pcm, sr, frame_ms=frame_ms)
    thr = speech_rms_threshold(rms, rms_floor=rms_floor, floor_mult=floor_mult)
    return rms >= thr, thr, frame_ms / 1000.0


def speech_seconds(
    pcm: np.ndarray,
    sr: int,
    *,
    frame_ms: float = SPEECH_FRAME_MS,
    rms_threshold: float = SPEECH_RMS_THRESHOLD,
    floor_mult: float = SPEECH_RMS_FLOOR_MULT,
) -> float:
    """Seconds of ACTUAL speech-level audio in ``pcm`` (pure numpy, no torch).

    A simple energy gate: the clip is cut into ``frame_ms`` frames and every
    frame whose RMS clears the gate counts as speech. The gate is
    NOISE-FLOOR-RELATIVE (2026-08-29): ``max(rms_threshold, floor_mult x the
    clip's 10th-percentile frame RMS)`` — ``rms_threshold`` is the absolute
    floor (a silent clip, p10 ~ 0, is still gated at it, so a silent upload
    can never enroll), and in a noisy room the gate rises above the room's
    own floor so room tone no longer counts as speech. Pass
    ``floor_mult=0`` for the old purely absolute gate. Deliberately
    conservative and honest — it distinguishes "a long clip" from "a long
    clip with enough speech in it". Returns 0.0 for empty audio or a
    nonsensical sample rate rather than guessing."""
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.size == 0 or sr <= 0:
        return 0.0
    frame = max(1, int(sr * frame_ms / 1000.0))
    usable = (pcm.size // frame) * frame
    rms = frame_rms(pcm, sr, frame_ms=frame_ms)
    gate = speech_rms_threshold(rms, rms_floor=rms_threshold, floor_mult=floor_mult)
    voiced_s = float(np.count_nonzero(rms >= gate)) * frame / sr if rms.size else 0.0
    tail = pcm[usable:]
    if tail.size > 0:
        tail_rms = float(np.sqrt(np.mean(tail.astype(np.float64) ** 2)))
        if tail_rms >= gate:
            voiced_s += tail.size / sr
    return voiced_s


def running_mean_embedding(
    existing: np.ndarray | None, existing_count: int, new: np.ndarray,
) -> np.ndarray:
    """Fold ``new`` into a running mean voiceprint and renormalize.

    Continuous enrollment: each confident enrollment refines the print, so it
    tracks a new phone/mic and even a seasonal voice. The mean is weighted by the
    number of prior enrollments so early prints aren't dominated by the latest
    sample. The result is L2-normalized (all matching is cosine)."""
    new = l2_normalize(new)
    if existing is None or existing_count <= 0:
        return new
    existing = np.asarray(existing, dtype=np.float32)
    blended = (existing * existing_count + new) / (existing_count + 1)
    return l2_normalize(blended)


# ---------------------------------------------------------------------------
# Orchestrators (torch via embed_pcm; still sync — callers use to_thread)
# ---------------------------------------------------------------------------

def embed_speaker(
    pcm: np.ndarray, sr: int, turns: list[dict], speaker: str,
    *, min_seconds: float = MIN_MATCH_SECONDS,
) -> np.ndarray | None:
    """Pool ``speaker``'s turns and embed them, or ``None`` when there is too
    little audio to be trustworthy (< ``min_seconds`` of pooled speech)."""
    pooled = pool_speaker_pcm(pcm, sr, turns, speaker)
    if pooled.size < int(min_seconds * sr):
        return None
    return embed_pcm(pooled, sr)


def _person_meta(person_id: str, people: dict[str, dict] | None) -> dict:
    """``{display_name, is_self}`` for one voiceprint's person, from the
    caller-supplied ``people`` metadata when present, else derived from the
    reserved self id. A non-self person with no known display name gets
    ``None`` — the ladder then skips labeling that speaker rather than
    inventing a name (the report still carries the match for debugging)."""
    meta = (people or {}).get(person_id) or {}
    is_self = bool(meta.get("is_self", person_id == SELF_PERSON_ID))
    name = meta.get("display_name")
    if is_self:
        name = SELF_DISPLAY_NAME
    elif not (isinstance(name, str) and name.strip()):
        name = None
    else:
        name = name.strip()
    return {"display_name": name, "is_self": is_self}


def identify_speakers_multi(
    pcm: np.ndarray,
    sr: int,
    turns: list[dict],
    voiceprints: dict[str, np.ndarray],
    *,
    threshold: float = MATCH_THRESHOLD,
    people: dict[str, dict] | None = None,
) -> dict:
    """Match every diarized speaker against EVERY enrolled person (blocking).

    Embeds each speaker ONCE (pooled turns — the expensive step) and hands the
    embeddings to :func:`identify_from_embeddings` for the scoring, which is
    pure and documents the report shape. ``voiceprints`` maps ``person_id`` ->
    blended embedding (the account owner is :data:`SELF_PERSON_ID`);
    ``people`` optionally carries each person's ``{display_name, is_self,
    settings}`` (``settings`` = distinct recordings pooled into the print —
    see :func:`profile_settings`; it gates the contrast match).
    """
    speakers: list[str] = []
    for t in turns:
        s = t.get("speaker")
        # UNKNOWN_SPEAKER is speech no found voice claimed — pooling it and
        # matching it to an enrolled print would name a voice the diarizer
        # itself refused to; it can never be "You".
        if s is not None and s != UNKNOWN_SPEAKER and s not in speakers:
            speakers.append(s)
    embeddings: dict[str, np.ndarray] = {}
    for speaker in speakers:
        emb = embed_speaker(pcm, sr, turns, speaker)
        if emb is None:
            continue  # too little audio — no score, honestly omitted
        embeddings[speaker] = emb
    return identify_from_embeddings(
        embeddings, voiceprints, threshold=threshold, people=people,
    )


def identify_from_embeddings(
    speaker_embeddings: dict[str, np.ndarray],
    voiceprints: dict[str, np.ndarray],
    *,
    threshold: float = MATCH_THRESHOLD,
    people: dict[str, dict] | None = None,
) -> dict:
    """Score already-computed per-speaker embeddings against every enrolled
    person (pure — no audio, no torch). Also what ``/voice/catch-up`` runs
    over the embeddings a stored analysis carries, so re-matching a past
    recording after the print improves costs a few dot products, not a
    decode + re-embed.

    Two ways a (speaker, person) pair clears the bar:

    * ABSOLUTE — cosine ≥ ``threshold`` (:data:`MATCH_THRESHOLD`).
    * CONTRAST — cosine ≥ :data:`CROSS_MATCH_THRESHOLD`, AND the person's
      print pools ≥ :data:`CROSS_MATCH_MIN_SETTINGS` distinct recordings
      (``people[pid]["settings"]``; unknown counts as 1), AND at least two
      speakers were scored, AND this speaker beats every other speaker's
      score for that person by ≥ :data:`CROSS_MATCH_MARGIN`. See the
      constants' calibration note for the real-recording numbers.

    Assignment is a greedy one-to-one matching, highest score first: each
    speaker gets at most one person (a voice is one person), each person wins
    at most one speaker (a person is one voice — two diarized clusters can't
    both be "Alex"; if the diarizer split one voice in two, only the stronger
    half is labeled and the other stays generic, honestly). Ties break
    deterministically (speaker id, then person id). Below both bars → no
    label, ever; the scores are always kept so a near-miss is inspectable::

        {
          "matched_speaker": "Speaker A" | None,      # the SELF match (legacy key)
          "match_threshold": 0.65,
          "cross_match_threshold": 0.40,
          "cross_match_margin": 0.15,
          "model": "speechbrain/spkrec-ecapa-voxceleb@<rev>",
          "matched": {"Speaker A": "self", "Speaker B": "alex"},
          "people": {"self": {"display_name": "You", "is_self": true},
                     "alex": {"display_name": "Alex", "is_self": false}},
          "speakers": {
            "Speaker A": {"scores": {"self": 0.71, "alex": 0.12},
                          "matched_person_id": "self", "is_self": true,
                          "display_name": "You", "match_basis": "absolute",
                          "embedding": [...192 floats...],
                          "score": 0.71, "is_you": true},   # legacy self keys
            "Speaker B": {"scores": {"self": 0.09, "alex": 0.45},
                          "matched_person_id": "alex", "is_self": false,
                          "display_name": "Alex", "match_basis": "contrast",
                          "embedding": [...],
                          "score": 0.09, "is_you": false},
          },
        }

    ``embedding`` is the speaker's pooled ECAPA vector (unit norm) — stored
    with the analysis so a later re-match needs no audio. ``matched_speaker``
    / per-speaker ``score`` + ``is_you`` are kept so every pre-existing reader
    of the single-voiceprint report keeps working unchanged; they describe
    the SELF person only and are omitted/None when no self print was supplied.
    """
    prints = {pid: l2_normalize(vec) for pid, vec in voiceprints.items()}
    meta = {pid: _person_meta(pid, people) for pid in prints}
    has_self = SELF_PERSON_ID in prints

    scored: dict[str, dict] = {}
    for speaker, emb in speaker_embeddings.items():
        emb = l2_normalize(np.asarray(emb, dtype=np.float32))
        scores = {pid: round(cosine(emb, vec), 4) for pid, vec in prints.items()}
        entry: dict = {
            "scores": scores,
            "matched_person_id": None,
            "is_self": False,
            "display_name": None,
            "match_basis": None,
            "embedding": [float(x) for x in emb.tolist()],
        }
        if has_self:
            entry["score"] = scores[SELF_PERSON_ID]
            entry["is_you"] = False
        scored[speaker] = entry

    candidates: list[tuple[float, str, str, str]] = []
    for speaker, entry in scored.items():
        for pid, score in entry["scores"].items():
            if score >= threshold:
                candidates.append((score, speaker, pid, "absolute"))
                continue
            if score < CROSS_MATCH_THRESHOLD or len(scored) < 2:
                continue
            settings = int(((people or {}).get(pid) or {}).get("settings") or 1)
            if settings < CROSS_MATCH_MIN_SETTINGS:
                continue
            runner_up = max(
                other["scores"][pid] for sp, other in scored.items() if sp != speaker
            )
            if score - runner_up >= CROSS_MATCH_MARGIN:
                candidates.append((score, speaker, pid, "contrast"))

    # Greedy one-to-one: best pair first; a taken speaker or person is skipped.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    matched: dict[str, str] = {}
    taken_people: set[str] = set()
    for _score, speaker, pid, basis in candidates:
        if speaker in matched or pid in taken_people:
            continue
        matched[speaker] = pid
        taken_people.add(pid)
        entry = scored[speaker]
        entry["matched_person_id"] = pid
        entry["is_self"] = meta[pid]["is_self"]
        entry["display_name"] = meta[pid]["display_name"]
        entry["match_basis"] = basis
        if has_self:
            entry["is_you"] = pid == SELF_PERSON_ID

    self_speaker = next(
        (sp for sp, pid in matched.items() if pid == SELF_PERSON_ID), None,
    )
    return {
        "matched_speaker": self_speaker,
        "match_threshold": threshold,
        "cross_match_threshold": CROSS_MATCH_THRESHOLD,
        "cross_match_margin": CROSS_MATCH_MARGIN,
        "model": f"{ECAPA_SOURCE}@{ECAPA_REVISION}",
        "matched": matched,
        "people": meta,
        "speakers": scored,
    }


def stored_speaker_embeddings(speaker_identity: object) -> dict[str, np.ndarray]:
    """Pure reader: the per-speaker ``embedding`` vectors a stored identity
    report carries (``{speaker: unit vector}``); empty for the legacy shape
    or anything malformed — a caller then falls back to the audio."""
    if not isinstance(speaker_identity, dict):
        return {}
    speakers = speaker_identity.get("speakers")
    if not isinstance(speakers, dict):
        return {}
    out: dict[str, np.ndarray] = {}
    for speaker, entry in speakers.items():
        vec = entry.get("embedding") if isinstance(entry, dict) else None
        if not (isinstance(speaker, str) and isinstance(vec, list) and vec):
            continue
        try:
            arr = np.asarray(vec, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 1 and arr.size == EMBEDDING_DIM and np.isfinite(arr).all():
            out[speaker] = l2_normalize(arr)
    return out


def identify_speakers(
    pcm: np.ndarray,
    sr: int,
    turns: list[dict],
    voiceprint: np.ndarray,
    *,
    threshold: float = MATCH_THRESHOLD,
) -> dict:
    """Match every diarized speaker against ONE voiceprint — the user's own.

    A thin wrapper over :func:`identify_speakers_multi` with the self print
    only, returning the ORIGINAL single-voiceprint report shape (nothing that
    consumed it can regress)::

        {
          "matched_speaker": "Speaker A" | None,
          "match_threshold": 0.5,
          "model": "speechbrain/spkrec-ecapa-voxceleb@<rev>",
          "speakers": {
            "Speaker A": {"score": 0.71, "is_you": true},
            "Speaker B": {"score": 0.09, "is_you": false},
          },
        }

    The single best speaker whose cosine clears ``threshold`` is the user
    ("You"); at most ONE speaker is "You" (a person is one voice); everyone else
    keeps their generic label. No label is forced below threshold.
    """
    report = identify_speakers_multi(
        pcm, sr, turns, {SELF_PERSON_ID: voiceprint}, threshold=threshold,
    )
    return {
        "matched_speaker": report["matched_speaker"],
        "match_threshold": report["match_threshold"],
        "model": report["model"],
        "speakers": {
            speaker: {"score": entry["score"], "is_you": entry["is_you"]}
            for speaker, entry in report["speakers"].items()
        },
    }


def enrolled_display_labels(speaker_identity: object) -> dict[str, str]:
    """Pure reader of an identity report (either shape) → ``{speaker:
    display_label}`` for every speaker the enrolled rung should label.

    The ONE place the ladder's consumers (main's resolver, episodes'
    participants, the voice router's relabel) learn who matched whom, so the
    two report shapes are handled once:

    * multi report — ``matched`` (speaker → person_id) joined with ``people``
      (person_id → display_name/is_self): self renders as :data:`YOU_LABEL`,
      a named partner as their name. A match to a person with NO usable
      display name is skipped (never an invented label; the score stays in
      the report for debugging).
    * legacy single-print report (stored analyses from before multi-person)
      — ``matched_speaker`` alone → "You".

    Defensive throughout: anything that isn't a dict of the expected shape
    contributes nothing (the enrolled rung is simply skipped)."""
    if not isinstance(speaker_identity, dict):
        return {}
    out: dict[str, str] = {}
    matched = speaker_identity.get("matched")
    if isinstance(matched, dict):
        people = speaker_identity.get("people")
        people = people if isinstance(people, dict) else {}
        for speaker, pid in matched.items():
            if not (isinstance(speaker, str) and speaker.strip()):
                continue
            if not isinstance(pid, str):
                continue
            meta = _person_meta(pid, people)
            if meta["display_name"]:
                out[speaker] = meta["display_name"]
        return out
    # Legacy shape: the self match only.
    legacy = speaker_identity.get("matched_speaker")
    if isinstance(legacy, str) and legacy.strip():
        out[legacy] = SELF_DISPLAY_NAME
    return out


def without_matches_for(speaker_identity: dict, speakers) -> dict:
    """A copy of an identity report with every match for ``speakers`` removed
    (both the multi ``matched`` map and the legacy ``matched_speaker``) — used
    to suppress the enrolled rung for speakers a human has manually relabeled,
    so manual truly wins. Pure; never mutates the stored report."""
    drop = set(speakers)
    out = dict(speaker_identity)
    matched = out.get("matched")
    if isinstance(matched, dict):
        out["matched"] = {sp: pid for sp, pid in matched.items() if sp not in drop}
    if out.get("matched_speaker") in drop:
        out["matched_speaker"] = None
    return out


def enrollment_conflict(
    embedding: np.ndarray,
    profiles: list[dict],
    person_id: str,
    *,
    threshold: float = MATCH_THRESHOLD,
    margin: float = ENROLL_CONFLICT_MARGIN,
) -> dict | None:
    """Would appending ``embedding`` to ``person_id``'s print mislabel someone
    ELSE's voice? Pure (no torch): scores the candidate against every stored
    person document in ``profiles`` (the ``list_voiceprints`` shape —
    ``person_id`` + blended ``embedding``).

    Returns ``None`` when the enrollment is safe, else a dict describing the
    conflict — ``{"person_id", "display_name", "score", "own_score"}`` —
    where ``score`` is the cosine to the closest OTHER person and
    ``own_score`` the cosine to ``person_id``'s existing print (``None`` for
    a brand-new person). The rule (see :data:`ENROLL_CONFLICT_MARGIN`): a
    conflict exists iff some other person's print clears ``threshold`` (the
    matcher would call this voice theirs) AND the candidate is NOT closer to
    ``person_id`` by at least ``margin``. Documents without a usable vector
    contribute nothing."""
    candidate = l2_normalize(np.asarray(embedding, dtype=np.float32))
    own_score: float | None = None
    best: tuple[float, str, str | None] | None = None
    for doc in profiles or ():
        if not isinstance(doc, dict):
            continue
        raw = doc.get("embedding")
        if not isinstance(raw, list) or not raw:
            continue
        pid = doc.get("person_id") or SELF_PERSON_ID
        score = cosine(candidate, np.asarray(raw, dtype=np.float32))
        if pid == person_id:
            own_score = score
            continue
        if best is None or score > best[0]:
            best = (score, pid, _person_meta(pid, {pid: doc})["display_name"])
    if best is None or best[0] < threshold:
        return None
    if own_score is not None and own_score - best[0] >= margin:
        return None
    return {
        "person_id": best[1],
        "display_name": best[2] or best[1],
        "score": round(best[0], 4),
        "own_score": None if own_score is None else round(own_score, 4),
    }


def sample_setting_key(sample: dict) -> str:
    """Which RECORDING a sample came from — the unit :func:`blend_samples`
    averages over. Samples with no source recording (guided enrollment) are
    each their own setting, keyed by sample id."""
    rid = sample.get("recording_id")
    if isinstance(rid, str) and rid.strip():
        return f"rec:{rid.strip()}"
    return f"sample:{sample.get('id')}"


def blend_samples(samples: list[dict]) -> np.ndarray:
    """The blended voiceprint for a v2 sample list: one centroid PER RECORDING
    (the normalized mean of that recording's samples), then the L2-normalized
    mean of those centroids. Per-recording, not per-sample, so three "This is
    me" taps on one clip don't outvote a single clip from a different room —
    what the cross-setting match (see :data:`CROSS_MATCH_THRESHOLD`) needs is
    breadth of SETTINGS, and the owner's real print had 3 of 5 samples from
    the same restaurant clip. With per-sample storage the blend is always
    recomputable, so deleting a sample simply re-runs this."""
    groups: dict[str, list[np.ndarray]] = {}
    for s in samples:
        groups.setdefault(sample_setting_key(s), []).append(
            l2_normalize(np.asarray(s["embedding"], dtype=np.float32))
        )
    centroids = [l2_normalize(np.mean(vecs, axis=0)) for vecs in groups.values()]
    return l2_normalize(np.mean(centroids, axis=0))


def current_blend(profile: dict | None) -> "np.ndarray | None":
    """The voiceprint a stored profile matches with TODAY: re-blended from its
    per-sample vectors under the current :func:`blend_samples` rule when it
    has any (so a print written under an older blend rule is served/matched
    correctly without waiting for a rewrite), else the stored blend (v1
    prints carry no samples). ``None`` when the document has no usable
    vector. Shared by main's matcher and ``GET /voice/people`` so the phone
    holds exactly the vector the server scores with."""
    if not isinstance(profile, dict):
        return None
    stored = profile.get("embedding")
    if not isinstance(stored, list) or not stored:
        return None
    samples = profile.get("samples")
    if isinstance(samples, list) and samples:
        try:
            return blend_samples([s for s in samples if isinstance(s, dict)])
        except (KeyError, TypeError, ValueError):
            pass
    return np.asarray(stored, dtype=np.float32)


def profile_settings(profile: dict | None) -> int:
    """How many distinct recordings a print pools (≥1 for any usable print;
    a v1 print with no samples counts as one setting). Gates the contrast
    match — see :data:`CROSS_MATCH_MIN_SETTINGS`."""
    if not isinstance(profile, dict):
        return 0
    samples = profile.get("samples")
    if not isinstance(samples, list) or not samples:
        return 1 if isinstance(profile.get("embedding"), list) else 0
    return len({sample_setting_key(s) for s in samples if isinstance(s, dict)})


def as_v2(profile: dict | None) -> dict | None:
    """A v2 VIEW of any stored profile document (pure — never persists).

    * ``None`` (unenrolled) passes through.
    * A v2 doc is returned as-is (same object).
    * A v1 doc is migrated: its running-mean blend becomes ONE legacy sample
      (``recording_id: None`` — the blend has no single source recording) with a
      plain note saying what it is. The legacy sample is deletable WHOLE, like
      any other; ``enroll_count`` counts SAMPLES from here on. GETs serve this
      view without rewriting the doc; the first write persists it.
    """
    if profile is None:
        return None
    if int(profile.get("version", 1) or 1) >= 2 and isinstance(
        profile.get("samples"), list,
    ):
        return profile
    n = int(profile.get("enroll_count", 0) or 0)
    legacy = {
        "id": LEGACY_SAMPLE_ID,
        "embedding": [float(x) for x in (profile.get("embedding") or [])],
        "recording_id": None,
        "speaker": None,
        "at": profile.get("updated_at"),
        "note": f"pre-v2 blend of {n} enrollment{'' if n == 1 else 's'}",
    }
    return {
        "version": PROFILE_VERSION,
        "embedding": list(legacy["embedding"]),
        "dim": len(legacy["embedding"]),
        "enroll_count": 1,
        "model": profile.get("model"),
        "created_at": profile.get("created_at"),
        "updated_at": profile.get("updated_at"),
        "samples": [legacy],
    }


def as_person(
    profile: dict | None,
    *,
    person_id: str | None = None,
    display_name: str | None = None,
) -> dict | None:
    """A PERSON view of any stored profile document (pure — never persists).

    Fills in the multi-person fields for a document that predates them (or
    that a caller built without them), on top of the v2 sample view:

    * ``person_id`` — the explicit argument, else the document's own, else
      :data:`SELF_PERSON_ID` (a pre-multi-person document can only ever have
      been the account owner's own voice — the legacy layout stored nothing
      else).
    * ``display_name`` — the explicit argument (stripped) when given, else the
      document's own, else :data:`SELF_DISPLAY_NAME` for self. A non-self
      person with no name stays ``None`` (the ladder then never labels that
      speaker — an honest gap, not an invented name).
    * ``is_self`` — DERIVED, never trusted from the document: true iff
      ``person_id == SELF_PERSON_ID``. That is what makes "exactly one self per
      account" structural (one reserved key) rather than a scan for flags.

    ``None`` (unenrolled) passes through. A v2 doc that already carries all
    three fields consistently is returned as the same object.
    """
    v2 = as_v2(profile)
    if v2 is None:
        return None
    pid = person_id or v2.get("person_id") or SELF_PERSON_ID
    is_self = pid == SELF_PERSON_ID
    if isinstance(display_name, str) and display_name.strip():
        name: str | None = display_name.strip()
    else:
        stored = v2.get("display_name")
        name = stored.strip() if isinstance(stored, str) and stored.strip() else None
    if is_self:
        name = SELF_DISPLAY_NAME
    if (
        v2.get("person_id") == pid
        and v2.get("display_name") == name
        and v2.get("is_self") is is_self
    ):
        return v2
    return {**v2, "person_id": pid, "display_name": name, "is_self": is_self}


def new_profile(
    embedding: np.ndarray,
    existing: dict | None,
    *,
    recording_id: str | None,
    speaker: str | None,
    now_iso: str,
    sample_id: str | None = None,
    note: str | None = None,
    person_id: str | None = None,
    display_name: str | None = None,
    seconds: float | None = None,
) -> dict:
    """Build the stored v2 voiceprint document: append this enrollment as an
    individual sample and recompute the blend over ALL samples. Pure (no I/O) so
    the store just persists what this returns — and it is unit-testable without
    torch. A v1 ``existing`` is migrated in the same step (see :func:`as_v2`).
    ``sample_id`` is generated when not given (tests pass one for determinism).

    ``recording_id``/``speaker`` are None for a sample with no stored source
    recording (guided direct enrollment); ``note`` then carries the honest
    provenance the client shows (e.g. "guided enrollment").

    ``person_id``/``display_name`` name WHOSE voice this is (multi-person
    voiceprints, see :func:`as_person`); both default to what ``existing``
    already says, and ultimately to the account owner ("self"/"You"), so every
    pre-existing caller keeps building the owner's own print unchanged.

    ``seconds`` — how much pooled speech the sample was embedded from — is
    stored on the sample as provenance when known (the People screen shows
    "12 s from <recording>"); omitted (not null) otherwise so older samples
    are byte-identical.
    """
    existing_v2 = as_person(existing, person_id=person_id, display_name=display_name)
    person = as_person(
        existing_v2 or {"version": PROFILE_VERSION, "samples": []},
        person_id=person_id, display_name=display_name,
    )
    samples = list((existing_v2 or {}).get("samples", []))
    new_vec = l2_normalize(embedding)
    sample = {
        "id": sample_id or uuid.uuid4().hex,
        "embedding": [float(x) for x in new_vec.tolist()],
        "recording_id": recording_id,
        "speaker": speaker,
        "at": now_iso,
    }
    if note is not None:
        sample["note"] = note
    if seconds is not None:
        sample["seconds"] = round(float(seconds), 1)
    samples.append(sample)
    blended = blend_samples(samples)
    created_at = (existing_v2 or {}).get("created_at") or now_iso
    return {
        "version": PROFILE_VERSION,
        "person_id": person["person_id"],
        "display_name": person["display_name"],
        "is_self": person["is_self"],
        "embedding": [float(x) for x in blended.tolist()],
        "dim": int(blended.size),
        "enroll_count": len(samples),
        "model": f"{ECAPA_SOURCE}@{ECAPA_REVISION}",
        "created_at": created_at,
        "updated_at": now_iso,
        "samples": samples,
    }


def remove_sample(profile: dict, sample_id: str, *, now_iso: str) -> dict | None:
    """The profile with ``sample_id`` removed and the blend recomputed (pure).

    Returns ``None`` when the last sample was removed — an empty profile is the
    SAME state as "forget my voice" (the caller deletes the stored doc), never a
    hollow document with no signal in it. Raises :class:`KeyError` when the
    sample isn't in the profile. A v1 ``profile`` is migrated first, so its
    legacy blend sample is deletable whole."""
    v2 = as_v2(profile)
    samples = [s for s in v2["samples"] if s.get("id") != sample_id]
    if len(samples) == len(v2["samples"]):
        raise KeyError(sample_id)
    if not samples:
        return None
    blended = blend_samples(samples)
    return {
        **v2,
        "embedding": [float(x) for x in blended.tolist()],
        "dim": int(blended.size),
        "enroll_count": len(samples),
        "updated_at": now_iso,
        "samples": samples,
    }
