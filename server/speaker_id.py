"""Speaker identity via ECAPA-TDNN voice embeddings — an OPTIONAL dependency.

This powers two features, both grounded in the existing diarization the pipeline
already runs (Deepgram gives us *which turn belongs to which speaker within one
recording*; this module gives those anonymous "Speaker A/B/…" clusters a
*persistent* identity across recordings):

* **Enrollment** ("This is me"): pool one diarized speaker's turns from a stored
  recording, embed them into a single 192-d voiceprint, and average it into the
  user's stored profile. Multiple enrollments refine the print (a running mean).
* **Auto-labeling** ("You"): during analysis, embed each diarized speaker's
  pooled turns and cosine-match against the user's voiceprint. The best speaker
  above :data:`MATCH_THRESHOLD` is labeled "You" (``label_source="enrolled"`` —
  the TOP rung of the display-label ladder). Below threshold → NO label, ever.

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
        savedir = os.getenv(
            "MINDSHIFT_ECAPA_CACHE",
            os.path.join(os.path.dirname(__file__), ".ecapa_cache"),
        )
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


def speech_seconds(
    pcm: np.ndarray,
    sr: int,
    *,
    frame_ms: float = SPEECH_FRAME_MS,
    rms_threshold: float = SPEECH_RMS_THRESHOLD,
) -> float:
    """Seconds of ACTUAL speech-level audio in ``pcm`` (pure numpy, no torch).

    A simple energy gate: the clip is cut into ``frame_ms`` frames and every
    frame whose RMS clears ``rms_threshold`` counts as speech. Deliberately
    conservative and honest — it distinguishes "a long clip" from "a long clip
    with enough speech in it", so a silent upload can never enroll. Returns 0.0
    for empty audio or a nonsensical sample rate rather than guessing."""
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.size == 0 or sr <= 0:
        return 0.0
    frame = max(1, int(sr * frame_ms / 1000.0))
    usable = (pcm.size // frame) * frame
    voiced_s = 0.0
    if usable > 0:
        frames = pcm[:usable].reshape(-1, frame)
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        voiced_s = float(np.count_nonzero(rms >= rms_threshold)) * frame / sr
    tail = pcm[usable:]
    if tail.size > 0:
        tail_rms = float(np.sqrt(np.mean(tail.astype(np.float64) ** 2)))
        if tail_rms >= rms_threshold:
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


def identify_speakers(
    pcm: np.ndarray,
    sr: int,
    turns: list[dict],
    voiceprint: np.ndarray,
    *,
    threshold: float = MATCH_THRESHOLD,
) -> dict:
    """Match every diarized speaker against the user's voiceprint (blocking).

    Returns a debuggable identity report; the single best speaker whose cosine
    clears ``threshold`` is the user ("You"). At most ONE speaker is "You" (a
    person is one voice); everyone else keeps their generic label. The per-speaker
    cosine scores are ALWAYS included so a near-miss is inspectable::

        {
          "matched_speaker": "Speaker A" | None,
          "match_threshold": 0.5,
          "model": "speechbrain/spkrec-ecapa-voxceleb@<rev>",
          "speakers": {
            "Speaker A": {"score": 0.71, "is_you": true},
            "Speaker B": {"score": 0.09, "is_you": false},
          },
        }

    The label ladder consumes ``matched_speaker`` as its top rung: that speaker's
    ``display_label`` becomes "You" / ``label_source`` "enrolled". No label is
    forced below threshold — an honest "unknown" stays unknown.
    """
    voiceprint = l2_normalize(voiceprint)
    speakers = []
    for t in turns:
        s = t.get("speaker")
        if s is not None and s not in speakers:
            speakers.append(s)

    scored: dict[str, dict] = {}
    best_speaker: str | None = None
    best_score = -1.0
    for speaker in speakers:
        emb = embed_speaker(pcm, sr, turns, speaker)
        if emb is None:
            continue  # too little audio — no score, honestly omitted
        score = cosine(emb, voiceprint)
        scored[speaker] = {"score": round(score, 4), "is_you": False}
        if score > best_score:
            best_score = score
            best_speaker = speaker

    matched = best_speaker if best_score >= threshold else None
    if matched is not None:
        scored[matched]["is_you"] = True

    return {
        "matched_speaker": matched,
        "match_threshold": threshold,
        "model": f"{ECAPA_SOURCE}@{ECAPA_REVISION}",
        "speakers": scored,
    }


def blend_samples(samples: list[dict]) -> np.ndarray:
    """The blended voiceprint for a v2 sample list: the L2-normalized mean of
    the (normalized) per-sample embeddings. With per-sample storage the blend is
    always recomputable, so deleting a sample simply re-runs this."""
    vecs = [
        l2_normalize(np.asarray(s["embedding"], dtype=np.float32))
        for s in samples
    ]
    return l2_normalize(np.mean(vecs, axis=0))


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


def new_profile(
    embedding: np.ndarray,
    existing: dict | None,
    *,
    recording_id: str | None,
    speaker: str | None,
    now_iso: str,
    sample_id: str | None = None,
    note: str | None = None,
) -> dict:
    """Build the stored v2 voiceprint document: append this enrollment as an
    individual sample and recompute the blend over ALL samples. Pure (no I/O) so
    the store just persists what this returns — and it is unit-testable without
    torch. A v1 ``existing`` is migrated in the same step (see :func:`as_v2`).
    ``sample_id`` is generated when not given (tests pass one for determinism).

    ``recording_id``/``speaker`` are None for a sample with no stored source
    recording (guided direct enrollment); ``note`` then carries the honest
    provenance the client shows (e.g. "guided enrollment").
    """
    existing_v2 = as_v2(existing)
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
    samples.append(sample)
    blended = blend_samples(samples)
    created_at = (existing_v2 or {}).get("created_at") or now_iso
    return {
        "version": PROFILE_VERSION,
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
