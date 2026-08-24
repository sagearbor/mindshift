"""Vocal tone (emotion) classification from AUDIO — an OPTIONAL dependency.

PRD Tier 2 asks for tone/nuance detection from the *sound* of a turn, not just
its transcript: the same words land very differently said warmly, flatly, or
shouted. This module classifies each diarized turn's audio into one of the
four IEMOCAP classes — ``neutral`` / ``angry`` / ``happy`` / ``sad`` — with a
softmax probability per class, using SpeechBrain's pretrained
``speechbrain/emotion-recognition-wav2vec2-IEMOCAP`` (Apache 2.0, ~79% on the
IEMOCAP test split per the model card). It deliberately mirrors
:mod:`speaker_id` (lazy model, pinned revision, blocking functions, honest
"unavailable" path) so the two optional-voice modules read the same way.

Why a learned acoustic model and not "loudness = anger": the owner explicitly
does not want a yelling detector. wav2vec2 features carry pitch contour,
rate, voice quality, and spectral shape — the model can (in principle) call a
quiet, cold, contemptuous line "angry" and a loud, excited one "happy". Whether
it actually does on OUR audio is a measured question, not an assumption —
see ``scripts/tone_eval.py`` and the report it writes under
``docs/research/tone-audio/``.

Ship mode — the ``MINDSHIFT_TONE_AUDIO`` flag (:func:`mode`):

* ``off``  — never load the model; :func:`classify_pcm` raises
  :class:`ToneUnavailable`. Zero cost.
* ``dark`` — (DEFAULT) compute + log per-turn results, but callers must NOT
  surface them to users (:func:`surface_allowed` is False). This is the
  owner's rule for a signal that measured weak on our own labeled fixtures:
  keep it running so it accumulates evidence, never show it.
* ``on``   — compute AND surface.

The default is set from the measured eval (see the research report): it goes
``on`` only when the model clears ~60% on our scripted ground truth AND adds
lift over text alone; otherwise it ships dark.

Honesty / availability notes (house rule: report unavailable, never fabricate):

* ``torch`` + ``speechbrain`` + ``transformers`` are heavy and kept OUT of the
  base requirements — see ``requirements-voice.txt``. They are imported LAZILY
  inside the functions that need them. :func:`is_available` is the cheap
  "can we classify tone at all?" probe; when False, callers skip cleanly.
* The loaded model is cached process-wide, loaded once under a lock — the
  fine-tuned wav2vec2 checkpoint is ~377MB and the first load (import torch,
  build the graph, first-run HF download) is slow, so it must never happen
  more than once per process.
* BOTH model repos are PINNED (:data:`TONE_REVISION` for the SpeechBrain
  fine-tune, :data:`TONE_BASE_REVISION` for the ``facebook/wav2vec2-base``
  backbone the recipe's YAML references) so a silent upstream reweight can't
  move the class probabilities — and thus any surfacing decision — under us.
  After the first fetch the load is fully offline from the local cache dir.
* Every model-touching function is BLOCKING (CPU inference). Callers on the
  event loop wrap them in ``asyncio.to_thread`` exactly as they do for
  :mod:`speaker_id`; the pure slicing / flag logic has NO torch dependency
  and is unit-tested directly.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# The pretrained SpeechBrain fine-tune + a PINNED revision (the HF repo's
# `main` commit at integration time, 2026-08-24). The recipe's hyperparams.yaml
# builds its backbone from `facebook/wav2vec2-base`, whose weights are then
# fully overwritten by the fine-tuned `wav2vec2.ckpt` — so the base pin only
# protects the model CONFIG (layer count, dims, norm placement), but a config
# drift would still break the load, so it is pinned too. All overridable via
# env for a controlled model bump; the eval report records which revisions
# produced its numbers.
TONE_SOURCE = os.getenv(
    "MINDSHIFT_TONE_SOURCE", "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
)
TONE_REVISION = os.getenv(
    "MINDSHIFT_TONE_REVISION", "117a9c3dff08be81a3628eecf6a66b547ec1659b"
)
TONE_BASE_SOURCE = os.getenv("MINDSHIFT_TONE_BASE_SOURCE", "facebook/wav2vec2-base")
TONE_BASE_REVISION = os.getenv(
    "MINDSHIFT_TONE_BASE_REVISION", "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
)

# The only files of the base repo the backbone loader needs (config + weights).
# Excluding the tokenizer/vocab files keeps the snapshot honest about what it
# is for — this is a feature encoder, never an ASR model here.
_BASE_ALLOW_PATTERNS = ("*.json", "*.bin", "README.md")

# Ship-mode flag. Read at CALL time (not import time) so a test or an operator
# can flip it without a restart; see :func:`mode`.
TONE_AUDIO_ENV = "MINDSHIFT_TONE_AUDIO"
TONE_MODES = ("off", "dark", "on")
# DEFAULT = "dark", set by the owner's rule from the measured eval
# (docs/research/tone-audio/2026-08-24-wav2vec2-iemocap-eval.md): the model
# did NOT clear the ~60% bar on our scripted ground truth, so it computes and
# logs but is never surfaced until a better model / calibration lands.
DEFAULT_TONE_MODE = "dark"

# IEMOCAP's 4 classes in the checkpoint's label-encoder order
# (label_encoder.txt: neu=0, ang=1, hap=2, sad=3), mapped to the plain words
# the rest of the product uses. Order matters: softmax column i is LABELS[i].
_MODEL_LABELS = ("neu", "ang", "hap", "sad")
LABELS = ("neutral", "angry", "happy", "sad")
_LABEL_MAP = dict(zip(_MODEL_LABELS, LABELS))

# wav2vec2 is trained on 16 kHz audio; our stored derivatives + live contract
# are already 16 kHz mono. Kept explicit so a mismatched input is caught, not
# silently mis-classified (mirrors speaker_id.TARGET_SR).
TARGET_SR = 16000

# A turn shorter than this is too little signal for a trustworthy tone call —
# the model's pooled features over a fraction of a second are dominated by
# onset noise. We SKIP it (result carries ``skipped="too_short"``, no label)
# rather than guess. Same floor speaker_id uses for a pooled embedding.
MIN_TURN_SECONDS = 1.0
# Cap a single slice so one very long turn can't make a forward pass
# unbounded (wav2vec2 attention is O(n^2) in frames; 30s ~ 1500 frames is
# comfortable on CPU). The FIRST ``MAX_TURN_SECONDS`` of the turn are scored
# and the result says so (``truncated=True``) — never a silent cut.
MAX_TURN_SECONDS = 30.0

# Process-wide model cache (see module docstring). A threading.Lock (not
# asyncio) because loads happen inside asyncio.to_thread worker threads.
_model = None
_model_lock = threading.Lock()


class ToneUnavailable(RuntimeError):
    """Audio tone classification is not available (deps absent, flag ``off``,
    or the model could not be loaded). Callers treat this as "skip tone" —
    no label is ever fabricated."""


# ---------------------------------------------------------------------------
# Ship-mode flag — pure, no torch
# ---------------------------------------------------------------------------

def mode() -> str:
    """The effective ``MINDSHIFT_TONE_AUDIO`` mode: ``off`` | ``dark`` | ``on``.

    Case/whitespace-insensitive; unset or empty → :data:`DEFAULT_TONE_MODE`.
    An UNKNOWN value also falls back to the default (with a warning) rather
    than crashing the server on a typo — and the default is the SAFE side
    (``dark`` computes but never surfaces), so a misconfiguration can't
    accidentally show users an unvalidated signal.
    """
    raw = os.getenv(TONE_AUDIO_ENV)
    if raw is None:
        return DEFAULT_TONE_MODE
    value = raw.strip().lower()
    if value == "":
        return DEFAULT_TONE_MODE
    if value not in TONE_MODES:
        logger.warning(
            "%s=%r is not one of %s; using default %r",
            TONE_AUDIO_ENV, raw, "|".join(TONE_MODES), DEFAULT_TONE_MODE,
        )
        return DEFAULT_TONE_MODE
    return value


def is_enabled() -> bool:
    """True when the flag allows COMPUTING tone at all (``dark`` or ``on``)."""
    return mode() != "off"


def surface_allowed() -> bool:
    """True ONLY in ``on`` mode. ``dark`` computes and logs but must not be
    shown to users; ``off`` computes nothing. Callers gate every user-visible
    use of a tone result on this."""
    return mode() == "on"


# ---------------------------------------------------------------------------
# Availability + model loading (the ONLY torch-touching code)
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """True when the optional tone deps import. Cheap import probe (no model
    load), independent of the flag — "could we?" not "should we?". Callers
    combine it with :func:`is_enabled`.
    """
    try:
        import speechbrain  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:  # noqa: BLE001 — any import failure means "not available"
        return False
    return True


def cache_dir() -> str:
    """Local dir the pinned snapshots are fetched into (``MINDSHIFT_TONE_CACHE``,
    default ``server/.tone_cache`` — gitignored next to ``.ecapa_cache``)."""
    return os.getenv(
        "MINDSHIFT_TONE_CACHE",
        os.path.join(os.path.dirname(__file__), ".tone_cache"),
    )


def model_id() -> str:
    """The provenance string stamped on every result (source@revision)."""
    return f"{TONE_SOURCE}@{TONE_REVISION}"


def snapshot_present(savedir: str | None = None) -> bool:
    """True when BOTH pinned snapshots are already in the cache dir, i.e. a
    load would be fully offline. Used by the live-gated test so it never
    triggers a ~750MB download in CI."""
    d = savedir or cache_dir()
    return (
        os.path.isfile(os.path.join(d, "wav2vec2.ckpt"))
        and os.path.isfile(os.path.join(d, "custom_interface.py"))
        and os.path.isfile(os.path.join(d, "wav2vec2-base", "config.json"))
        and os.path.isfile(os.path.join(d, "wav2vec2-base", "pytorch_model.bin"))
    )


def _load_model():
    """Return the shared, cached tone classifier, loading it once under a lock.

    Lazy + guarded exactly like :func:`speaker_id._load_model`: two concurrent
    requests can't both pay the load cost, and the base install (no torch)
    never imports speechbrain at module top. Raises :class:`ToneUnavailable`
    when the flag is ``off``, the deps are missing, or the checkpoint can't be
    loaded — the caller degrades honestly.
    """
    global _model
    if mode() == "off":
        raise ToneUnavailable(f"audio tone disabled ({TONE_AUDIO_ENV}=off)")
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from speechbrain.inference.interfaces import foreign_class
        except Exception as exc:  # noqa: BLE001
            raise ToneUnavailable(
                "audio tone not available on this server — install "
                "requirements-voice.txt (torch + speechbrain + transformers)"
            ) from exc
        savedir = os.path.abspath(cache_dir())
        base_dir = os.path.join(savedir, "wav2vec2-base")
        # Enforce the revision PINS by pre-fetching both exact snapshots to the
        # local dir, then loading ONLY from there (a local `source` loads via
        # copy/symlink, not hf_hub_download, so the recipe's own unpinned
        # fetch paths are never exercised). Same rationale as speaker_id: this
        # does not depend on SpeechBrain forwarding a `revision=` kwarg, and
        # sidesteps its HF fetch path's huggingface_hub version coupling. If a
        # pinned snapshot can't be fetched we fall back to an UNPINNED load
        # from the Hub (logged, so the pin gap is never silent).
        source = TONE_SOURCE
        base_source = TONE_BASE_SOURCE
        overrides: dict = {}
        try:
            from huggingface_hub import snapshot_download

            source = snapshot_download(
                repo_id=TONE_SOURCE, revision=TONE_REVISION, local_dir=savedir,
            )
            base_source = snapshot_download(
                repo_id=TONE_BASE_SOURCE, revision=TONE_BASE_REVISION,
                local_dir=base_dir, allow_patterns=list(_BASE_ALLOW_PATTERNS),
            )
            # Point the recipe's YAML at the pinned local copies: the backbone
            # (`wav2vec2_hub`), the fine-tuned ckpt paths (`pretrained_path`),
            # and where transformers may cache anything (`save_path` — the
            # recipe's default is a RELATIVE `wav2vec2_checkpoints`, i.e. a
            # stray dir in whatever the process CWD is; keep it in the cache).
            overrides = {
                "wav2vec2_hub": base_source,
                "pretrained_path": source,
                "wav2vec2": {"save_path": os.path.join(savedir, "hf_cache")},
            }
        except Exception as exc:  # noqa: BLE001 — degrade to unpinned, but say so
            logger.warning(
                "Could not pre-fetch pinned tone model %s@%s / %s@%s (%s); "
                "loading unpinned from the Hub",
                TONE_SOURCE, TONE_REVISION, TONE_BASE_SOURCE, TONE_BASE_REVISION,
                exc,
            )
        try:
            # The recipe ships its own Pretrained subclass (custom_interface.py)
            # rather than a stock EncoderClassifier — foreign_class loads it.
            _model = foreign_class(
                source=source,
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier",
                savedir=savedir,
                run_opts={"device": "cpu"},
                overrides=overrides,
            )
        except Exception as exc:  # noqa: BLE001
            raise ToneUnavailable(
                f"could not load audio tone model {TONE_SOURCE!r}: {exc}"
            ) from exc
        logger.info("Loaded audio tone model %s @ %s", TONE_SOURCE, TONE_REVISION)
        return _model


def _probs_to_result(probs: np.ndarray) -> dict:
    """Turn one softmax row (model label order) into the public result dict.
    Pure — unit-tested against a fake model without the checkpoint."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if probs.size != len(LABELS):
        raise ToneUnavailable(
            f"tone model emitted {probs.size} classes, expected {len(LABELS)}"
        )
    scores = {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))}
    best = int(np.argmax(probs))
    return {
        "label": LABELS[best],
        "scores": scores,
        "confidence": round(float(probs[best]), 4),
        "model": model_id(),
    }


def classify_pcm(pcm: np.ndarray, sr: int = TARGET_SR) -> dict:
    """Classify mono float32 PCM into one of :data:`LABELS` (blocking).

    Returns ``{"label", "scores": {neutral, angry, happy, sad}, "confidence",
    "model"}`` — ``scores`` are the softmax probabilities (sum to 1),
    ``confidence`` is the winning class's probability. Runs the pretrained
    model on CPU; callers off the event loop wrap this in
    ``asyncio.to_thread``. Raises :class:`ToneUnavailable` when the flag is
    ``off`` or the model can't be loaded. ``pcm`` must be at
    :data:`TARGET_SR` — a different rate is an honest error, never a silent
    mis-classification.
    """
    if mode() == "off":
        raise ToneUnavailable(f"audio tone disabled ({TONE_AUDIO_ENV}=off)")
    if sr != TARGET_SR:
        raise ToneUnavailable(
            f"audio tone expects {TARGET_SR} Hz audio, got {sr} Hz"
        )
    audio = np.ascontiguousarray(pcm, dtype=np.float32)
    if audio.size == 0:
        raise ToneUnavailable("cannot classify a zero-length audio chunk")
    import torch

    model = _load_model()
    with torch.no_grad():
        wav = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
        out_prob, _score, _index, _text = model.classify_batch(wav)
    probs = out_prob.squeeze().detach().cpu().numpy()
    return _probs_to_result(probs)


# ---------------------------------------------------------------------------
# Per-turn orchestration — pure slicing, model only via the injected fn
# ---------------------------------------------------------------------------

def slice_turn(pcm: np.ndarray, sr: int, turn: dict) -> tuple[np.ndarray, bool]:
    """The PCM under one diarized turn, clamped to the audio bounds and capped
    at :data:`MAX_TURN_SECONDS`. Returns ``(samples, truncated)``; an empty
    array when the turn has no usable audio (missing/inverted times, outside
    the clip)."""
    if pcm.size == 0 or sr <= 0:
        return np.zeros(0, dtype=np.float32), False
    start = turn.get("start_time")
    end = turn.get("end_time")
    if start is None or end is None:
        return np.zeros(0, dtype=np.float32), False
    i0 = max(0, int(float(start) * sr))
    i1 = min(pcm.size, int(float(end) * sr))
    if i1 <= i0:
        return np.zeros(0, dtype=np.float32), False
    max_samples = int(MAX_TURN_SECONDS * sr)
    truncated = (i1 - i0) > max_samples
    if truncated:
        i1 = i0 + max_samples
    return np.ascontiguousarray(pcm[i0:i1], dtype=np.float32), truncated


def classify_turns(
    pcm: np.ndarray,
    sr: int,
    turns: list[dict],
    *,
    min_seconds: float = MIN_TURN_SECONDS,
    classify_fn=None,
) -> list[dict]:
    """Classify the tone of EVERY diarized turn (blocking; callers use
    ``asyncio.to_thread``).

    Returns ONE entry per input turn, in order, so results align with the
    transcript by index::

        {
          "index": 3, "speaker": "Speaker B",
          "start_time": 12.4, "end_time": 18.8, "seconds": 6.4,
          "tone": {"label": "angry", "scores": {...}, "confidence": 0.91,
                   "model": "..."} | None,
          "skipped": None | "too_short" | "no_audio",
          "truncated": False,
          "latency_ms": 118.0,
        }

    A turn under ``min_seconds`` is SKIPPED honestly (``tone=None``,
    ``skipped="too_short"``) rather than scored on too little signal; a turn
    with no usable audio is ``skipped="no_audio"``. ``latency_ms`` is the
    wall-clock of the model call for that slice (0 when skipped) — the
    realtime budget this feature lives under is measured, not assumed.

    ``classify_fn`` (default :func:`classify_pcm`) is injectable so the slicing
    / skipping logic is unit-testable without the model; a
    :class:`ToneUnavailable` from it propagates — the caller decides whether
    that means "skip tone for this recording".

    Always logs a one-line label distribution per call at INFO: in ``dark``
    mode that log line IS the feature's output.
    """
    fn = classify_fn or classify_pcm
    results: list[dict] = []
    for i, t in enumerate(turns):
        samples, truncated = slice_turn(pcm, sr, t)
        entry = {
            "index": i,
            "speaker": t.get("speaker"),
            "start_time": t.get("start_time"),
            "end_time": t.get("end_time"),
            "seconds": round(samples.size / sr, 3) if sr > 0 else 0.0,
            "tone": None,
            "skipped": None,
            "truncated": truncated,
            "latency_ms": 0.0,
        }
        if samples.size == 0:
            entry["skipped"] = "no_audio"
        elif samples.size < int(min_seconds * sr):
            entry["skipped"] = "too_short"
        else:
            t0 = time.perf_counter()
            entry["tone"] = fn(samples, sr)
            entry["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        results.append(entry)
    logger.info(
        "audio tone (%s mode): %d turns, distribution %s",
        mode(), len(results), label_distribution(results),
    )
    return results


def label_distribution(results: list[dict]) -> dict[str, int]:
    """Count of predicted labels over a :func:`classify_turns` result (skipped
    turns counted under ``"skipped"``). Pure; used for the dark-mode log line
    and the eval's real-recording sanity check."""
    counts: dict[str, int] = {label: 0 for label in LABELS}
    counts["skipped"] = 0
    for r in results:
        tone = r.get("tone")
        if tone is None:
            counts["skipped"] += 1
        else:
            counts[tone["label"]] = counts.get(tone["label"], 0) + 1
    return counts
