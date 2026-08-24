"""Local speaker diarization via pyannote.audio — an alternate engine to
:mod:`diarize_local`'s ECAPA k-search, added specifically for recordings
with MORE real speakers than that pipeline's transcript-utterance-relabeling
approach can find (see ``docs/research/poker6-sliding-window/README.md`` and
``docs/handoff/2026-08-24-mac-transition-and-poker6-status.md`` for the full
investigation: a real 6-speaker poker-night recording where the shipped
ECAPA pipeline only ever found 4).

Why a SEPARATE engine rather than replacing diarize_local outright: on the
project's existing regression fixtures (2-speaker real + scripted-TTS
recordings) the ECAPA pipeline already measures 100% exact per-turn
accuracy (see ``server/tests/test_diarize_regression_ladder.py``) — there is
no accuracy problem to fix there, only a heavier optional dependency to
avoid paying for by default. pyannote.audio needs its own torch/torchaudio/
huggingface_hub pins (see ``requirements-pyannote.txt`` for exactly why —
version conflicts with requirements-voice.txt's speechbrain setup) and pulls
a second multi-hundred-MB model, so it stays fully opt-in: install
``requirements-pyannote.txt`` (its own venv/image layer) AND set
``MINDSHIFT_DIARIZE_ENGINE=pyannote`` to use it. Absent either, callers get
``is_available() == False`` and the caller falls back to the default engine
— never a hard failure.

Algorithm: pyannote's own segmentation + embedding + agglomerative
clustering pipeline (``pyannote/speaker-diarization-3.1``) diarizes the RAW
AUDIO directly — it does not need or use the transcript's utterance
boundaries at all, unlike diarize_local's relabel-existing-turns approach.
Each transcript turn is then assigned the pyannote speaker with the most
TIME-OVERLAP with that turn (majority vote), so the output keeps the same
turn boundaries/text the caller passed in — same output contract as
:func:`diarize_local.diarize_turns`.

Clustering tuning: pyannote's ``clustering.min_cluster_size`` defaults to 12
(embedding-window count, not seconds) — calibrated for longer recordings.
On a 30s clip with ~5s per speaker there are too few embedding windows per
speaker to ever clear that floor, so it silently caps out at 4-5 clusters
no matter how high ``num_speakers`` is set ("Found only 4 clusters. Using a
smaller value than 5 for `min_cluster_size` might help" — pyannote's own
diagnostic). :data:`MIN_CLUSTER_SIZE` lowers this specifically for short
multi-speaker clips; see ``docs/research/poker6-sliding-window/
pyannote_result_*.json`` for the per-value measurements that produced the
chosen default (checked against poker6 AND the project's other real/
scripted fixtures for regressions before picking it).

KNOWN LIMITATION (measured 2026-08-24, not yet solved): lowering
``min_cluster_size`` to separate poker6's 6 close-together short turns has a
real cost on the OPPOSITE case — a normal 2-speaker recording with more
speech per person. In fully-automatic mode (no ``num_speakers`` hint),
``MIN_CLUSTER_SIZE=3`` over-segments ``test_recording_family_real.wav``
(2 real speakers) into 5 phantom clusters, 25% per-utterance accuracy —
much worse than diarize_local's 100% on the same file. This is NOT fixed by
bounding ``min_speakers``/``max_speakers`` (tried; a wide [2, 6] bound still
returns 5, since pyannote's own speaker-COUNT selection is a separate
mechanism from ``min_cluster_size`` and isn't forced down by the bound alone
when 5 already satisfies it). Given ``num_speakers`` as an explicit hint,
this engine is excellent across the board — poker6 and every other measured
fixture — but that hint is exactly what production does NOT have ahead of
time. Until that's resolved, this engine is intended for cases where a
higher speaker count is already suspected (the caller passes ``num_speakers``
or a deliberately higher count is expected), NOT as an auto-detecting
drop-in replacement for diarize_local's default 2-4-person path — hence the
opt-in ``MINDSHIFT_DIARIZE_ENGINE`` gate rather than a default-engine swap.
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)

SOURCE = "local-pyannote"
PIPELINE_NAME = "pyannote/speaker-diarization-3.1"

# See the module docstring's "Clustering tuning" section — 12 is pyannote's
# own default, calibrated for longer recordings than this app's typical
# clips. Env-overridable for recalibration without a code change.
MIN_CLUSTER_SIZE = int(os.getenv("MINDSHIFT_PYANNOTE_MIN_CLUSTER_SIZE", "3"))

_pipeline = None
_pipeline_lock = threading.Lock()


class PyannoteUnavailable(RuntimeError):
    """pyannote diarization is not available on this server (deps absent,
    HF_TOKEN missing, or the model could not be loaded)."""


def is_available() -> bool:
    """True when the optional pyannote deps import AND an HF_TOKEN is set.
    Cheap import probe (no model load) — mirrors speaker_id.is_available()."""
    if not os.environ.get("HF_TOKEN"):
        return False
    try:
        import pyannote.audio  # noqa: F401
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 — any import failure means "not available"
        return False
    return True


def _load_pipeline():
    """Return the shared, cached pyannote pipeline, loading it once under a
    lock (same lazy+guarded pattern as speaker_id._load_model)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise PyannoteUnavailable("HF_TOKEN not set")
        try:
            import torch

            # pyannote.audio 3.3.2's lightning-based checkpoint loader
            # predates torch>=2.6's weights_only=True default. These are
            # official pyannote checkpoints from HF (trusted source), so
            # restore the pre-2.6 default for this load only.
            _orig_load = torch.load

            def _patched_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return _orig_load(*args, **kwargs)

            torch.load = _patched_load
            try:
                from pyannote.audio import Pipeline

                pipeline = Pipeline.from_pretrained(
                    PIPELINE_NAME, use_auth_token=hf_token,
                )
            finally:
                torch.load = _orig_load
        except Exception as exc:  # noqa: BLE001
            raise PyannoteUnavailable(
                f"could not load pyannote pipeline {PIPELINE_NAME!r}: {exc}"
            ) from exc
        if pipeline is None:
            raise PyannoteUnavailable(
                f"pyannote pipeline {PIPELINE_NAME!r} requires accepting its "
                "license on huggingface.co (gated model)"
            )
        params = pipeline.parameters(instantiated=True)
        params["clustering"]["min_cluster_size"] = MIN_CLUSTER_SIZE
        pipeline.instantiate(params)
        _pipeline = pipeline
        logger.info(
            "Loaded pyannote pipeline %s (min_cluster_size=%d)",
            PIPELINE_NAME, MIN_CLUSTER_SIZE,
        )
        return _pipeline


def _speaker_name(index: int) -> str:
    """Cluster index -> display label, matching diarize_local's convention."""
    if index < 26:
        return f"Speaker {chr(ord('A') + index)}"
    return f"Speaker {index + 1}"


def _majority_overlap_speaker(seg_start: float, seg_end: float, pyannote_turns) -> str | None:
    """The pyannote speaker with the most time-overlap with [seg_start, seg_end]."""
    best_speaker, best_overlap = None, 0.0
    for turn, _, speaker in pyannote_turns:
        overlap = max(0.0, min(seg_end, turn.end) - max(seg_start, turn.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker


def diarize_turns(
    pcm: np.ndarray,
    sr: int,
    turns: list[dict],
    *,
    num_speakers: int | None = None,
) -> dict | None:
    """Diarize ``pcm`` with pyannote and relabel ``turns`` by majority time-
    overlap. Returns ``None`` when pyannote has nothing trustworthy to say
    (deps/token missing, model load failure, or fewer than 2 turns) — same
    "degrade honestly, never invent a speaker" contract as
    :func:`diarize_local.diarize_turns`. Output shape matches that function
    (``turns``, ``num_speakers``, ``source``, ...) so callers can swap engines
    without touching call sites.

    ``num_speakers``, when given, is passed straight through to pyannote as
    a hint (NOT a hard requirement — pyannote's own clustering may still
    return fewer if it can't separate that many voices at the configured
    ``MIN_CLUSTER_SIZE``). Left unset, pyannote auto-detects the count.
    """
    if len(turns) < 2:
        return None
    try:
        pipeline = _load_pipeline()
    except PyannoteUnavailable as exc:
        logger.info("pyannote diarization unavailable: %s", exc)
        return None

    import torch

    waveform = torch.from_numpy(np.ascontiguousarray(pcm, dtype=np.float32)).unsqueeze(0)
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    diarization = pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)
    pyannote_turns = list(diarization.itertracks(yield_label=True))
    if not pyannote_turns:
        logger.info("pyannote diarization heard no speech")
        return None

    # Assign each transcript turn to its majority-overlap pyannote speaker,
    # then rename clusters in order of first appearance (diarize_local's
    # display convention) rather than pyannote's own raw SPEAKER_00 labels.
    raw_label_of: dict[int, str | None] = {
        i: _majority_overlap_speaker(
            float(t.get("start_time") or 0.0), float(t.get("end_time") or 0.0),
            pyannote_turns,
        )
        for i, t in enumerate(turns)
    }
    name_of: dict[str, str] = {}
    new_turns = []
    for i, t in enumerate(turns):
        raw = raw_label_of[i]
        if raw is None:
            # No pyannote speech overlapped this turn (e.g. a silence-only
            # transcript artifact) — keep the transcript's own label rather
            # than inventing one.
            new_turns.append(dict({k: v for k, v in t.items() if k != "words"}))
            continue
        if raw not in name_of:
            name_of[raw] = _speaker_name(len(name_of))
        new_turns.append(dict(
            {k: v for k, v in t.items() if k != "words"}, speaker=name_of[raw],
        ))

    return {
        "turns": new_turns,
        "num_speakers": len({t["speaker"] for t in new_turns}),
        "source": SOURCE,
        "model": f"{PIPELINE_NAME}@min_cluster_size={MIN_CLUSTER_SIZE}",
        "segments_total": len(turns),
        "pyannote_turns_total": len(pyannote_turns),
    }
