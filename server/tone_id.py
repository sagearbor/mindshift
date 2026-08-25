"""Vocal tone from AUDIO — an OPTIONAL dependency, with swappable backends.

PRD Tier 2 asks for tone/nuance detection from the *sound* of a turn, not just
its transcript: the same words land very differently said warmly, flatly, or
shouted. Round 1 (docs/research/tone-audio/2026-08-24-wav2vec2-iemocap-eval.md)
measured the obvious model — SpeechBrain's wav2vec2 IEMOCAP 4-class
classifier — at 40% 4-class / 50% arousal on our own acted fixtures: it read
voice IDENTITY, not emotion (every turn of one voice "angry", every turn of
the other "neutral"). Round 2 (docs/research/tone-audio/2026-08-24-round2.md)
found the fix is not a better 4-class model but a different QUESTION: the
product needs *escalation of a speaker's own voice*, and that is a per-speaker
DELTA — this turn's arousal minus that speaker's running baseline — not an
absolute label. On a dimensional (arousal/dominance/valence) model trained on
naturalistic podcast speech, that delta reaches 0.81 AUC / 83% leave-one-scene
-out accuracy on the labeled pack, versus ~0.55 AUC for every backend's raw
per-turn output. So this module now has three parts:

1. **Backends** (``MINDSHIFT_TONE_BACKEND``, :func:`backend`), each pinned to
   an exact HF revision, loaded lazily once per process, CPU only:

   * ``odyssey_dim`` (DEFAULT) — ``3loi/SER-Odyssey-Baseline-WavLM-Multi-
     Attributes`` (MIT; WavLM-large + attentive pooling, the Odyssey-2024
     SER challenge baseline trained on MSP-Podcast). Outputs arousal /
     dominance / valence, each ≈ 0..1. The only backend whose per-speaker
     delta cleared the bar with a clear margin over "volume only". Cost:
     1.27 GB on disk, ~3 GB RSS, ~270 ms per 5 s turn on an M-series CPU.
   * ``superb_er`` — ``superb/wav2vec2-base-superb-er`` (Apache-2.0; IEMOCAP
     4-class, plain ``transformers``). 378 MB, ~0.5 GB RSS, ~110 ms/turn. Its
     delta (0.75 AUC / 74%) is no better than the RMS-volume delta — kept as
     the "fits a 2 GiB box" option, with that caveat on record.
   * ``iemocap`` — the round-1 SpeechBrain model, kept selectable so the
     round-1 numbers stay reproducible. Delta 0.66 AUC.

   The audeering MSP-dim model that round 2 also measured is CC-BY-NC-SA
   ("research purpose only") and is deliberately NOT a backend here.

2. **Escalation** (:class:`EscalationTracker`, :func:`annotate_escalation`):
   pure numpy, no torch. Every backend's result carries a scalar ``arousal``
   (the arousal dimension, or for a categorical model the angry-vs-best-other
   logit margin). The tracker keeps each speaker's running baseline (median
   of their PREVIOUS turns, causal — exactly what a live session can know)
   and flags a turn as ``"escalating"`` when arousal − baseline ≥ the
   backend's measured :data:`ESCALATION_DELTA_THRESHOLD`. A speaker's first
   turn has no baseline and is honestly ``"unscored"``, never guessed.

3. **Ship mode** — ``MINDSHIFT_TONE_AUDIO`` (:func:`mode`): ``off`` / ``dark``
   (DEFAULT: compute + log, never surface) / ``on``. Round 2's rule stays the
   owner's: the audio signal must clear ~60% AND add lift over text-tone to
   go ``on``. Escalation-delta clears 60% but its lift over the text-tone
   baseline (86% on the same turns) is +3 points with a 95% CI that includes
   zero, so the default remains ``dark`` — see the report for what would
   flip it.

Honesty / availability notes (house rule: report unavailable, never fabricate):

* ``torch`` + ``transformers`` (+ ``speechbrain`` for ``iemocap`` only) are
  heavy and kept OUT of the base requirements — see ``requirements-voice.txt``.
  They are imported LAZILY inside the functions that need them.
  :func:`is_available` is the cheap "can we classify tone at all?" probe.
* The loaded model is cached process-wide, loaded once under a lock.
* Every model repo is PINNED (revision hashes below) so a silent upstream
  reweight can't move a surfacing decision under us. After the first fetch
  the load is fully offline from the local cache dir.
* Every model-touching function is BLOCKING (CPU inference). Callers on the
  event loop wrap them in ``asyncio.to_thread``; the flag / tracker /
  slicing logic has NO torch dependency and is unit-tested directly.
"""

from __future__ import annotations

import logging
import os
import statistics
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend registry + pins
# ---------------------------------------------------------------------------

TONE_BACKEND_ENV = "MINDSHIFT_TONE_BACKEND"
TONE_BACKENDS = ("odyssey_dim", "superb_er", "iemocap")
# DEFAULT = the backend the round-2 report justifies: the only one whose
# per-speaker escalation delta cleared the ~60% bar with a clear margin over
# a volume-only signal (0.81 AUC / 83% LOSO vs 0.68 / 74% for RMS delta).
DEFAULT_TONE_BACKEND = "odyssey_dim"
DIMENSIONAL_BACKENDS = frozenset({"odyssey_dim"})

# --- odyssey_dim: WavLM-large SER baseline (MIT). The repo's `pipeline_utils`
# would fetch `microsoft/wavlm-large` UNPINNED via trust_remote_code at
# construction; we vendor the (tiny) head + pooling code below instead and
# build the WavLM backbone from a PINNED config-only snapshot, then load the
# fine-tuned safetensors over it. Revisions = the repos' `main` at
# integration time (2026-08-24).
ODYSSEY_SOURCE = os.getenv(
    "MINDSHIFT_TONE_ODYSSEY_SOURCE", "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes"
)
ODYSSEY_REVISION = os.getenv(
    "MINDSHIFT_TONE_ODYSSEY_REVISION", "00d0e12ba9bf957f5aeea36e8663c8c61cb50ac9"
)
ODYSSEY_SSL_SOURCE = os.getenv("MINDSHIFT_TONE_ODYSSEY_SSL_SOURCE", "microsoft/wavlm-large")
ODYSSEY_SSL_REVISION = os.getenv(
    "MINDSHIFT_TONE_ODYSSEY_SSL_REVISION", "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
)
_ODYSSEY_ALLOW_PATTERNS = ("config.json", "model.safetensors", "preprocessor_config.json", "README.md")
_ODYSSEY_SSL_ALLOW_PATTERNS = ("config.json", "preprocessor_config.json", "README.md")

# --- superb_er: wav2vec2-base fine-tuned on IEMOCAP (Apache-2.0), stock
# `transformers` Wav2Vec2ForSequenceClassification.
SUPERB_SOURCE = os.getenv("MINDSHIFT_TONE_SUPERB_SOURCE", "superb/wav2vec2-base-superb-er")
SUPERB_REVISION = os.getenv(
    "MINDSHIFT_TONE_SUPERB_REVISION", "441a7599c3b22107314dcbd9166621c5c83f2cc5"
)
_SUPERB_ALLOW_PATTERNS = ("config.json", "preprocessor_config.json", "pytorch_model.bin", "README.md")

# --- iemocap: the round-1 SpeechBrain fine-tune. The recipe's hyperparams.yaml
# builds its backbone from `facebook/wav2vec2-base`, whose weights are then
# fully overwritten by the fine-tuned `wav2vec2.ckpt` — so the base pin only
# protects the model CONFIG, but a config drift would still break the load,
# so it is pinned too.
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
_BASE_ALLOW_PATTERNS = ("*.json", "*.bin", "README.md")

BACKEND_INFO: dict[str, dict] = {
    "odyssey_dim": {
        "source": ODYSSEY_SOURCE, "revision": ODYSSEY_REVISION, "license": "MIT",
        "kind": "dimensional", "subdir": "odyssey-dim",
    },
    "superb_er": {
        "source": SUPERB_SOURCE, "revision": SUPERB_REVISION, "license": "Apache-2.0",
        "kind": "categorical", "subdir": "superb-er",
    },
    "iemocap": {
        "source": TONE_SOURCE, "revision": TONE_REVISION, "license": "Apache-2.0",
        "kind": "categorical", "subdir": "",
    },
}

# ---------------------------------------------------------------------------
# Ship-mode flag
# ---------------------------------------------------------------------------

TONE_AUDIO_ENV = "MINDSHIFT_TONE_AUDIO"
TONE_MODES = ("off", "dark", "on")
# DEFAULT = "dark", set by the owner's rule from the measured evals (round 1
# AND round 2, see module docstring): computes and logs, never surfaced.
DEFAULT_TONE_MODE = "dark"

# IEMOCAP's 4 classes mapped to the plain words the rest of the product uses.
# For the SpeechBrain checkpoint the label-encoder order is neu/ang/hap/sad
# (softmax column i is LABELS[i]); the SUPERB checkpoint's order comes from
# its config.id2label and is remapped through _SUPERB_LABEL_MAP at load.
_MODEL_LABELS = ("neu", "ang", "hap", "sad")
LABELS = ("neutral", "angry", "happy", "sad")
_LABEL_MAP = dict(zip(_MODEL_LABELS, LABELS))
DIMS = ("arousal", "dominance", "valence")

# Labels the escalation layer adds on top of the backends' own vocabularies.
ESCALATION_LABEL = "escalating"   # delta over the speaker's baseline ≥ threshold
STEADY_LABEL = "steady"           # dimensional backend, baseline known, under threshold
UNSCORED_LABEL = "unscored"       # dimensional backend, no baseline yet (speaker's first turn)

# Per-backend escalation thresholds on (arousal − speaker baseline), MEASURED
# by leave-one-fixture-out threshold selection in scripts/tone_eval.py (round
# 2). Units differ per backend: odyssey_dim is the arousal dimension (≈0..1,
# folds chose 0.015–0.03; 0.03 taken for precision: 85% acc, 74% prec / 74%
# rec, self nudges 5 hit / 1 miss / 1 false), superb_er and iemocap are logit
# margins (folds 0.48–0.73 → 0.57; 1.1–4.9 → 1.5). HUMAN-TUNABLE, but re-run
# the eval and update the report if you touch them.
ESCALATION_DELTA_THRESHOLD: dict[str, float] = {
    "odyssey_dim": 0.03,
    "superb_er": 0.57,
    "iemocap": 1.5,
}

# wav2vec2 / WavLM are trained on 16 kHz audio; our stored derivatives + live
# contract are already 16 kHz mono. Kept explicit so a mismatched input is
# caught, not silently mis-classified (mirrors speaker_id.TARGET_SR).
TARGET_SR = 16000

# A turn shorter than this is too little signal for a trustworthy tone call —
# pooled features over a fraction of a second are dominated by onset noise.
# We SKIP it (``skipped="too_short"``, no label) rather than guess.
MIN_TURN_SECONDS = 1.0
# Cap a single slice so one very long turn can't make a forward pass
# unbounded (transformer attention is O(n^2) in frames; 30 s ≈ 1500 frames
# is comfortable on CPU). The FIRST ``MAX_TURN_SECONDS`` are scored and the
# result says so (``truncated=True``) — never a silent cut.
MAX_TURN_SECONDS = 30.0

# Process-wide model cache: one loaded model per backend name (so a test or
# an operator flipping MINDSHIFT_TONE_BACKEND gets the right model without a
# restart, and never pays a load twice). A threading.Lock (not asyncio)
# because loads happen inside asyncio.to_thread worker threads.
_models: dict[str, object] = {}
_model_lock = threading.Lock()


class ToneUnavailable(RuntimeError):
    """Audio tone classification is not available (deps absent, flag ``off``,
    unknown backend, or the model could not be loaded). Callers treat this as
    "skip tone" — no label is ever fabricated."""


# ---------------------------------------------------------------------------
# Flags — pure, no torch
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


def backend() -> str:
    """The effective ``MINDSHIFT_TONE_BACKEND``: one of :data:`TONE_BACKENDS`.

    Same parsing rules as :func:`mode`: case/whitespace-insensitive, unset or
    empty → :data:`DEFAULT_TONE_BACKEND`, unknown → default with a warning
    (a typo must not take the server down or silently pick a random model).
    """
    raw = os.getenv(TONE_BACKEND_ENV)
    if raw is None:
        return DEFAULT_TONE_BACKEND
    value = raw.strip().lower()
    if value == "":
        return DEFAULT_TONE_BACKEND
    if value not in TONE_BACKENDS:
        logger.warning(
            "%s=%r is not one of %s; using default %r",
            TONE_BACKEND_ENV, raw, "|".join(TONE_BACKENDS), DEFAULT_TONE_BACKEND,
        )
        return DEFAULT_TONE_BACKEND
    return value


def _check_backend(name: str | None) -> str:
    name = name or backend()
    if name not in BACKEND_INFO:
        raise ToneUnavailable(f"unknown tone backend {name!r} (choose from {TONE_BACKENDS})")
    return name


def is_dimensional(name: str | None = None) -> bool:
    """True when the backend emits arousal/dominance/valence rather than a
    4-class softmax."""
    return _check_backend(name) in DIMENSIONAL_BACKENDS


def escalation_threshold(name: str | None = None) -> float:
    """The measured per-backend delta threshold (see
    :data:`ESCALATION_DELTA_THRESHOLD`)."""
    return ESCALATION_DELTA_THRESHOLD[_check_backend(name)]


# ---------------------------------------------------------------------------
# Availability + cache paths (the ONLY torch-touching code is further down)
# ---------------------------------------------------------------------------

def is_available(name: str | None = None) -> bool:
    """True when the optional deps for ``name`` (default: the configured
    backend) import. Cheap import probe (no model load), independent of the
    flag — "could we?" not "should we?". Callers combine it with
    :func:`is_enabled`. ``iemocap`` additionally needs ``speechbrain``.
    """
    try:
        name = _check_backend(name)
    except ToneUnavailable:
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        if name == "iemocap":
            import speechbrain  # noqa: F401
        else:
            import safetensors  # noqa: F401
    except Exception:  # noqa: BLE001 — any import failure means "not available"
        return False
    return True


def cache_dir() -> str:
    """Local dir the pinned snapshots are fetched into (``MINDSHIFT_TONE_CACHE``,
    default ``server/.tone_cache`` — gitignored next to ``.ecapa_cache``).
    Each backend lives in its own subdir (:data:`BACKEND_INFO`)."""
    return os.getenv(
        "MINDSHIFT_TONE_CACHE",
        os.path.join(os.path.dirname(__file__), ".tone_cache"),
    )


def model_id(name: str | None = None) -> str:
    """The provenance string stamped on every result (source@revision)."""
    info = BACKEND_INFO[_check_backend(name)]
    return f"{info['source']}@{info['revision']}"


def snapshot_present(savedir: str | None = None, name: str | None = None) -> bool:
    """True when the pinned snapshot(s) for ``name`` are already in the cache
    dir, i.e. a load would be fully offline. Used by the live-gated tests so
    they never trigger a multi-hundred-MB download in CI."""
    d = savedir or cache_dir()
    name = _check_backend(name)
    isfile = os.path.isfile
    if name == "odyssey_dim":
        od = os.path.join(d, "odyssey-dim")
        return (
            isfile(os.path.join(od, "config.json"))
            and isfile(os.path.join(od, "model.safetensors"))
            and isfile(os.path.join(d, "wavlm-large", "config.json"))
        )
    if name == "superb_er":
        sd = os.path.join(d, "superb-er")
        return isfile(os.path.join(sd, "config.json")) and isfile(os.path.join(sd, "pytorch_model.bin"))
    return (
        isfile(os.path.join(d, "wav2vec2.ckpt"))
        and isfile(os.path.join(d, "custom_interface.py"))
        and isfile(os.path.join(d, "wav2vec2-base", "config.json"))
        and isfile(os.path.join(d, "wav2vec2-base", "pytorch_model.bin"))
    )


# ---------------------------------------------------------------------------
# Model loading — per backend
# ---------------------------------------------------------------------------

def _snapshot(repo: str, revision: str, local_dir: str, patterns: tuple[str, ...]) -> str:
    """Pre-fetch one exact pinned snapshot into ``local_dir`` (offline after
    the first call). Raises on failure — the caller decides whether an
    unpinned fallback is acceptable (it is for iemocap's SpeechBrain path,
    with a logged warning; the transformers backends require the pin)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo, revision=revision, local_dir=local_dir, allow_patterns=list(patterns),
    )


def _load_iemocap():
    from speechbrain.inference.interfaces import foreign_class

    savedir = os.path.abspath(cache_dir())
    base_dir = os.path.join(savedir, "wav2vec2-base")
    # Enforce the revision PINS by pre-fetching both exact snapshots to the
    # local dir, then loading ONLY from there (a local `source` loads via
    # copy/symlink, not hf_hub_download, so the recipe's own unpinned fetch
    # paths are never exercised). If a pinned snapshot can't be fetched we
    # fall back to an UNPINNED load from the Hub (logged, never silent).
    source, base_source, overrides = TONE_SOURCE, TONE_BASE_SOURCE, {}
    try:
        source = _snapshot(TONE_SOURCE, TONE_REVISION, savedir, ("*",))
        base_source = _snapshot(TONE_BASE_SOURCE, TONE_BASE_REVISION, base_dir, _BASE_ALLOW_PATTERNS)
        # Point the recipe's YAML at the pinned local copies: the backbone
        # (`wav2vec2_hub`), the fine-tuned ckpt paths (`pretrained_path`), and
        # where transformers may cache anything (`save_path` — the recipe's
        # default is a RELATIVE dir in the process CWD; keep it in the cache).
        overrides = {
            "wav2vec2_hub": base_source,
            "pretrained_path": source,
            "wav2vec2": {"save_path": os.path.join(savedir, "hf_cache")},
        }
    except Exception as exc:  # noqa: BLE001 — degrade to unpinned, but say so
        logger.warning(
            "Could not pre-fetch pinned tone model %s@%s / %s@%s (%s); loading unpinned from the Hub",
            TONE_SOURCE, TONE_REVISION, TONE_BASE_SOURCE, TONE_BASE_REVISION, exc,
        )
    # The recipe ships its own Pretrained subclass (custom_interface.py)
    # rather than a stock EncoderClassifier — foreign_class loads it.
    return foreign_class(
        source=source,
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        savedir=savedir,
        run_opts={"device": "cpu"},
        overrides=overrides,
    )


def _load_superb():
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

    d = _snapshot(SUPERB_SOURCE, SUPERB_REVISION, os.path.join(cache_dir(), "superb-er"), _SUPERB_ALLOW_PATTERNS)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(d)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(d).eval()
    # The checkpoint's id2label uses IEMOCAP's short names in ITS order
    # (neu/hap/ang/sad) — remap to LABELS by name, never by position.
    order = [_LABEL_MAP[model.config.id2label[i]] for i in range(model.config.num_labels)]
    return {"model": model, "extractor": extractor, "order": order}


def _build_odyssey_model(wavlm_config_dir: str, weights_path: str):
    """The Odyssey baseline's head + pooling (vendored from the repo's MIT
    ``pipeline_utils.py`` so nothing runs via ``trust_remote_code`` and the
    backbone is built from a PINNED config, not an unpinned Hub fetch)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from transformers import WavLMConfig, WavLMModel

    class AttentiveStatisticsPooling(nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.sap_linear = nn.Linear(input_size, input_size)
            self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))

        def forward(self, xs, mask):
            wav_lens = torch.sum(mask, dim=1)
            feat_lens = (torch.div(wav_lens - 1, 16000 * 0.02, rounding_mode="floor") + 1).int().tolist()
            pooled = []
            for x, fl in zip(xs, feat_lens):
                x = x[:fl].unsqueeze(0)
                h = torch.tanh(self.sap_linear(x))
                w = torch.matmul(h, self.attention).squeeze(2)
                w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
                mu = torch.sum(x * w, dim=1)
                rh = torch.sqrt((torch.sum((x ** 2) * w, dim=1) - mu ** 2).clamp(min=1e-5))
                pooled.append(torch.cat((mu, rh), 1).squeeze(0))
            return torch.stack(pooled)

    class EmotionRegression(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.5):
            super().__init__()
            block = lambda i, o: nn.Sequential(  # noqa: E731
                nn.Linear(i, o), nn.LayerNorm(o), nn.ReLU(), nn.Dropout(dropout)
            )
            self.fc = nn.ModuleList([block(input_dim, hidden_dim)])
            for _ in range(num_layers - 1):
                self.fc.append(block(hidden_dim, hidden_dim))
            self.out = nn.Sequential(nn.Linear(hidden_dim, output_dim))
            self.inp_drop = nn.Dropout(dropout)

        def forward(self, x):
            h = self.inp_drop(x)
            for fc in self.fc:
                h = fc(h)
            return self.out(h)

    class SERModel(nn.Module):
        def __init__(self, cfg, hidden, layers, ncls, p):
            super().__init__()
            self.ssl_model = WavLMModel(cfg)
            self.pool_model = AttentiveStatisticsPooling(hidden)
            self.ser_model = EmotionRegression(hidden * 2, hidden, layers, ncls, dropout=p)

        def forward(self, x, mask):
            ssl = self.ssl_model(x, attention_mask=mask).last_hidden_state
            return self.ser_model(self.pool_model(ssl, mask))

    import json

    with open(os.path.join(os.path.dirname(weights_path), "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    wavlm_cfg = WavLMConfig.from_pretrained(wavlm_config_dir)
    model = SERModel(
        wavlm_cfg, int(cfg["hidden_size"]), int(cfg["classifier_hidden_layers"]),
        int(cfg["num_classes"]), float(cfg["classifier_dropout_prob"]),
    )
    state = load_file(weights_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # WavLM's `masked_spec_embed` (training-only) is the one key the fine-tune
    # legitimately lacks; anything else missing means the checkpoint and the
    # vendored architecture drifted apart — refuse rather than run garbage.
    real_missing = [k for k in missing if not k.endswith("masked_spec_embed")]
    if real_missing or unexpected:
        raise ToneUnavailable(
            f"odyssey checkpoint/architecture mismatch: missing {real_missing[:3]}, unexpected {list(unexpected)[:3]}"
        )
    order = [cfg["id2label"][str(i)] for i in range(int(cfg["num_classes"]))]
    if tuple(order) != DIMS:
        raise ToneUnavailable(f"odyssey id2label order {order} != {DIMS}")
    return {"model": model.eval(), "mean": float(cfg["mean"]), "std": float(cfg["std"])}


def _load_odyssey():
    d = _snapshot(ODYSSEY_SOURCE, ODYSSEY_REVISION, os.path.join(cache_dir(), "odyssey-dim"), _ODYSSEY_ALLOW_PATTERNS)
    ssl_dir = _snapshot(
        ODYSSEY_SSL_SOURCE, ODYSSEY_SSL_REVISION, os.path.join(cache_dir(), "wavlm-large"), _ODYSSEY_SSL_ALLOW_PATTERNS,
    )
    return _build_odyssey_model(ssl_dir, os.path.join(d, "model.safetensors"))


_LOADERS = {"iemocap": _load_iemocap, "superb_er": _load_superb, "odyssey_dim": _load_odyssey}


def _load_model(name: str | None = None):
    """Return the shared, cached model for ``name`` (default: the configured
    backend), loading it once under a lock.

    Lazy + guarded exactly like :func:`speaker_id._load_model`: two concurrent
    requests can't both pay the load cost, and the base install (no torch)
    never imports a model library at module top. Raises
    :class:`ToneUnavailable` when the flag is ``off``, the deps are missing,
    or the checkpoint can't be loaded — the caller degrades honestly.
    """
    if mode() == "off":
        raise ToneUnavailable(f"audio tone disabled ({TONE_AUDIO_ENV}=off)")
    name = _check_backend(name)
    cached = _models.get(name)
    if cached is not None:
        return cached
    with _model_lock:
        cached = _models.get(name)
        if cached is not None:
            return cached
        if not is_available(name):
            raise ToneUnavailable(
                f"audio tone backend {name!r} not available on this server — install "
                "requirements-voice.txt (torch + transformers"
                + (" + speechbrain)" if name == "iemocap" else ")")
            )
        t0 = time.perf_counter()
        try:
            model = _LOADERS[name]()
        except ToneUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToneUnavailable(f"could not load audio tone backend {name!r} ({model_id(name)}): {exc}") from exc
        _models[name] = model
        logger.info("Loaded audio tone backend %s (%s) in %.1fs", name, model_id(name), time.perf_counter() - t0)
        return model


# ---------------------------------------------------------------------------
# Result shaping — pure, unit-tested without a checkpoint
# ---------------------------------------------------------------------------

def arousal_margin(logits: dict[str, float]) -> float:
    """A categorical model's scalar "how angry": the ``angry`` logit minus the
    best OTHER logit. Unlike the (saturated) softmax this stays informative
    when the model is over-confident, which is what the per-speaker delta
    needs."""
    others = [float(v) for k, v in logits.items() if k != "angry"]
    return float(logits["angry"]) - (max(others) if others else 0.0)


def _probs_to_result(probs: np.ndarray, logits: np.ndarray | None = None, name: str | None = None) -> dict:
    """One softmax row (LABELS order) → the public categorical result dict.
    ``logits`` (same order) feed :func:`arousal_margin`; without them the
    margin is computed on log-probabilities (monotone-equivalent for a
    softmax, just less numerically comfortable when saturated)."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if probs.size != len(LABELS):
        raise ToneUnavailable(f"tone model emitted {probs.size} classes, expected {len(LABELS)}")
    scores = {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))}
    best = int(np.argmax(probs))
    if logits is None:
        logits = np.log(np.clip(probs, 1e-12, None))
    lg = {LABELS[i]: float(np.asarray(logits).reshape(-1)[i]) for i in range(len(LABELS))}
    return {
        "label": LABELS[best],
        "scores": scores,
        "confidence": round(float(probs[best]), 4),
        "arousal": round(arousal_margin(lg), 4),
        # The unnormalized 4-vector, kept for offline per-speaker analysis
        # (scripts/tone_eval.py); the softmax above saturates, this doesn't.
        "logits": {k: round(v, 4) for k, v in lg.items()},
        "kind": "categorical",
        "backend": _check_backend(name),
        "model": model_id(name),
    }


def _dims_to_result(dims: dict[str, float], name: str | None = None) -> dict:
    """arousal/dominance/valence (each ≈ 0..1, NOT clipped — the delta wants
    the raw number) → the public dimensional result dict. A raw dimensional
    result carries NO emotion label (raw arousal measured near chance for
    escalation, 0.61 AUC): ``label`` is :data:`UNSCORED_LABEL` with
    confidence 0 until :func:`annotate_escalation` compares it with the
    speaker's baseline."""
    try:
        scores = {d: round(float(dims[d]), 4) for d in DIMS}
    except (KeyError, TypeError, ValueError) as exc:
        raise ToneUnavailable(f"dimensional tone model emitted {dims!r}, expected {DIMS}") from exc
    return {
        "label": UNSCORED_LABEL,
        "scores": scores,
        "confidence": 0.0,
        "arousal": scores["arousal"],
        "kind": "dimensional",
        "backend": _check_backend(name),
        "model": model_id(name),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def classify_pcm(pcm: np.ndarray, sr: int = TARGET_SR) -> dict:
    """Classify mono float32 PCM with the configured backend (blocking).

    Returns ``{"label", "scores", "confidence", "arousal", "kind", "backend",
    "model"}``. Categorical backends: ``scores`` are the 4-class softmax
    (sum to 1), ``confidence`` the winner's probability, ``arousal`` the
    angry-vs-best-other logit margin. Dimensional backends: ``scores`` are
    ``{arousal, dominance, valence}`` (≈0..1), ``label`` is ``"unscored"``
    (see :func:`_dims_to_result`) and ``arousal`` the arousal dimension.
    Per-speaker escalation is a separate, pure step — :func:`annotate_escalation`.

    Runs on CPU; callers off the event loop wrap this in ``asyncio.to_thread``.
    Raises :class:`ToneUnavailable` when the flag is ``off`` or the model
    can't be loaded. ``pcm`` must be at :data:`TARGET_SR`.
    """
    if mode() == "off":
        raise ToneUnavailable(f"audio tone disabled ({TONE_AUDIO_ENV}=off)")
    if sr != TARGET_SR:
        raise ToneUnavailable(f"audio tone expects {TARGET_SR} Hz audio, got {sr} Hz")
    audio = np.ascontiguousarray(pcm, dtype=np.float32)
    if audio.size == 0:
        raise ToneUnavailable("cannot classify a zero-length audio chunk")
    name = backend()
    model = _load_model(name)
    import torch

    with torch.no_grad():
        if name == "odyssey_dim":
            y = (audio - model["mean"]) / (model["std"] + 1e-6)
            wav = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).unsqueeze(0)
            mask = torch.ones(1, wav.shape[1], dtype=torch.long)
            pred = model["model"](wav, mask).squeeze().detach().cpu().numpy().reshape(-1)
            return _dims_to_result(dict(zip(DIMS, pred.tolist())), name)
        if name == "superb_er":
            inputs = model["extractor"](audio, sampling_rate=sr, return_tensors="pt")
            logits = model["model"](**inputs).logits.squeeze().detach().cpu().numpy().reshape(-1)
            by_name = dict(zip(model["order"], logits.tolist()))
            lg = np.array([by_name[lbl] for lbl in LABELS])
            probs = np.exp(lg - lg.max())
            return _probs_to_result(probs / probs.sum(), lg, name)
        # iemocap: the SpeechBrain custom interface — encoder → output MLP →
        # softmax, run stepwise so the raw logits are available for the margin.
        wav = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
        emb = model.encode_batch(wav)
        logits = model.mods.output_mlp(emb).squeeze().detach().cpu().numpy().reshape(-1)
        probs = np.exp(logits - logits.max())
        return _probs_to_result(probs / probs.sum(), logits, name)


# ---------------------------------------------------------------------------
# Escalation — per-speaker delta over a causal baseline (pure numpy)
# ---------------------------------------------------------------------------

class EscalationTracker:
    """Per-speaker running baseline of ``arousal`` over a session.

    Causal by construction: a turn is compared with the median of that
    speaker's PREVIOUS turns only (at most ``max_history`` of them, all when
    None) — exactly what a live session can know at that moment. The round-2
    eval measured the delta's AUC at k=1/2/3/5/all previous turns as
    0.76/0.80/0.79/0.81/0.81 for odyssey_dim: usable from the speaker's
    second turn, saturating around five. A speaker's FIRST turn has no
    baseline (``delta=None``) — never guessed from other speakers' voices,
    because cross-speaker level differences were exactly round 1's failure.
    """

    def __init__(self, max_history: int | None = None):
        if max_history is not None and max_history < 1:
            raise ValueError("max_history must be >= 1 or None")
        self.max_history = max_history
        self._history: dict[str, list[float]] = {}

    def history(self, speaker: str) -> int:
        """How many previous turns of ``speaker`` back the baseline."""
        return len(self._history.get(speaker, ()))

    def baseline(self, speaker: str) -> float | None:
        """Median arousal of the speaker's previous ≤ ``max_history`` turns, or
        None before the first one."""
        h = self._history.get(speaker)
        if not h:
            return None
        window = h[-self.max_history:] if self.max_history else h
        return float(statistics.median(window))

    def observe(self, speaker: str, arousal: float) -> dict:
        """Record ``arousal`` for ``speaker`` and return
        ``{"delta", "baseline", "history"}`` for THIS turn, computed against
        the baseline BEFORE recording it (a turn never baselines itself)."""
        base = self.baseline(speaker)
        n = self.history(speaker)
        self._history.setdefault(speaker, []).append(float(arousal))
        delta = None if base is None else round(float(arousal) - base, 4)
        return {"delta": delta, "baseline": None if base is None else round(base, 4), "history": n}


def escalation_confidence(delta: float, threshold: float) -> float:
    """A monotone squash of how far over the threshold the delta is (0.5 AT
    the threshold, 1.0 at twice it) — a comparable ``confidence`` for the
    ToneFlagEvent contract, explicitly NOT a calibrated probability."""
    if threshold <= 0:
        return 1.0 if delta > 0 else 0.0
    return float(min(1.0, max(0.0, 0.5 + 0.5 * (delta - threshold) / threshold)))


def annotate_escalation(
    result: dict, speaker: str, tracker: EscalationTracker, threshold: float | None = None,
) -> dict:
    """Add the per-speaker escalation verdict to one :func:`classify_pcm`
    result (mutates and returns it).

    Sets ``result["escalation"] = {"delta", "baseline", "history", "flag",
    "threshold"}``. When ``flag`` (delta ≥ threshold, default the backend's
    :data:`ESCALATION_DELTA_THRESHOLD`) the ``label`` becomes
    :data:`ESCALATION_LABEL` and ``confidence`` :func:`escalation_confidence`.
    Otherwise a dimensional result's label becomes :data:`STEADY_LABEL`
    (baseline known, under threshold) or stays :data:`UNSCORED_LABEL` (no
    baseline yet); a categorical result keeps the model's own label and
    confidence — that is still an honest report of what the model said.
    """
    name = result.get("backend") or backend()
    t = escalation_threshold(name) if threshold is None else float(threshold)
    obs = tracker.observe(speaker, float(result["arousal"]))
    flag = obs["delta"] is not None and obs["delta"] >= t
    result["escalation"] = {**obs, "flag": bool(flag), "threshold": t}
    if flag:
        result["label"] = ESCALATION_LABEL
        result["confidence"] = round(escalation_confidence(obs["delta"], t), 4)
    elif result.get("kind") == "dimensional":
        result["label"] = STEADY_LABEL if obs["delta"] is not None else UNSCORED_LABEL
        result["confidence"] = 0.0
    return result


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
    tracker: EscalationTracker | None = None,
    threshold: float | None = None,
) -> list[dict]:
    """Classify the tone of EVERY diarized turn, in order, and run the
    per-speaker escalation tracker over them causally (blocking; callers use
    ``asyncio.to_thread``).

    Returns ONE entry per input turn, in order, so results align with the
    transcript by index::

        {
          "index": 3, "speaker": "Speaker B",
          "start_time": 12.4, "end_time": 18.8, "seconds": 6.4,
          "tone": {"label": ..., "scores": {...}, "confidence": ..., "arousal": ...,
                   "escalation": {"delta", "baseline", "history", "flag", "threshold"},
                   "kind": ..., "backend": ..., "model": ...} | None,
          "skipped": None | "too_short" | "no_audio",
          "truncated": False,
          "latency_ms": 118.0,
        }

    A turn under ``min_seconds`` is SKIPPED honestly (``tone=None``,
    ``skipped="too_short"``) rather than scored on too little signal; a turn
    with no usable audio is ``skipped="no_audio"``. Skipped turns do not
    touch the speaker's baseline. ``latency_ms`` is the wall-clock of the
    model call for that slice (0 when skipped).

    ``classify_fn`` (default :func:`classify_pcm`) is injectable so the
    slicing / skipping / escalation logic is unit-testable without a model —
    it must return a result with an ``"arousal"`` number (a result without
    one is kept as-is, with no escalation entry). ``tracker`` defaults to a
    fresh :class:`EscalationTracker` for this call; pass one to continue a
    session's baselines. A :class:`ToneUnavailable` from ``classify_fn``
    propagates — the caller decides whether that means "skip tone".

    Always logs a one-line label distribution per call at INFO: in ``dark``
    mode that log line IS the feature's output.
    """
    fn = classify_fn or classify_pcm
    tracker = tracker or EscalationTracker()
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
            tone = fn(samples, sr)
            entry["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            if isinstance(tone, dict) and isinstance(tone.get("arousal"), (int, float)):
                tone = annotate_escalation(tone, str(t.get("speaker")), tracker, threshold)
            entry["tone"] = tone
        results.append(entry)
    logger.info(
        "audio tone (%s mode, %s backend): %d turns, distribution %s",
        mode(), backend(), len(results), label_distribution(results),
    )
    return results


def label_distribution(results: list[dict]) -> dict[str, int]:
    """Count of predicted labels over a :func:`classify_turns` result (skipped
    turns counted under ``"skipped"``; escalation/steady/unscored labels
    counted under their own names). Pure; used for the dark-mode log line and
    the eval's real-recording sanity check."""
    counts: dict[str, int] = {label: 0 for label in LABELS}
    counts["skipped"] = 0
    for r in results:
        tone = r.get("tone")
        if tone is None:
            counts["skipped"] += 1
        else:
            counts[tone["label"]] = counts.get(tone["label"], 0) + 1
    return counts
