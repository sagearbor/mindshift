"""Live-gated parity test: the ONNX export of the pinned ECAPA embedder must
produce the SAME embeddings as ``speaker_id.embed_pcm`` on REAL speech.

Why parity is the load-bearing property: Foundation B's whole premise is that
a voiceprint enrolled on the server (torch/speechbrain) can be matched on the
phone (onnxruntime-react-native) and vice versa. That only holds if the two
runtimes produce the same point in embedding space for the same audio — a
subtle front-end drift (a different window, a missing top_db clip, no
per-utterance mean subtraction) would still yield "an" embedding, just one
whose cosine against a server-enrolled print is silently lower, eroding the
calibrated ``MATCH_THRESHOLD`` for no visible reason. So this test compares
the two on real recordings (the owner's family clip and the 6-speaker poker
night), across several slice lengths (the time axis is dynamic — every
length must round-trip, not just the trace length), and requires
cosine > 0.999.

Gating mirrors ``test_diarize_regression_ladder.py``: skipped honestly (never
fake-passed) when torch/speechbrain, onnx/onnxruntime, or the real fixtures
are absent. The export itself runs into ``tmp_path`` so the test never
depends on (or overwrites) a previously exported file.

The per-slice onnxruntime CPU latency is PRINTED (run with ``-s``), not
asserted — it is a number we want to record, not a contract (phone CPUs
differ from this machine's).
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest

import audio_ingest
import speaker_id

_AUDIO_DIR = Path(__file__).resolve().parent / "fixtures" / "audio"
_FAMILY = _AUDIO_DIR / "test_recording_family_real.wav"
_POKER6 = _AUDIO_DIR / "test_recording_poker6_real.wav"
_EXPORT_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_ecapa_onnx.py"


def _onnx_available() -> bool:
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception:  # noqa: BLE001 — any import failure means "not available"
        return False
    return True


pytestmark = [
    pytest.mark.skipif(
        not speaker_id.is_available(),
        reason="voice deps (torch + speechbrain) not installed",
    ),
    pytest.mark.skipif(not _onnx_available(), reason="onnx/onnxruntime not installed"),
    pytest.mark.skipif(
        not (_FAMILY.exists() and _POKER6.exists()),
        reason="real-recording fixtures missing",
    ),
]


def _load_export_module():
    """Import scripts/export_ecapa_onnx.py by path (pytest's pythonpath already
    includes scripts/, but an explicit path import keeps this test honest
    about WHICH file it is exercising)."""
    spec = importlib.util.spec_from_file_location("export_ecapa_onnx", _EXPORT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_16k(path: Path) -> np.ndarray:
    """Decode via the SAME production path the enrollment endpoint uses
    (``decode_to_pcm_16k``) so the test audio is exactly what the embedder
    sees in production."""
    pcm, sr = audio_ingest.decode_to_pcm_16k(path.read_bytes(), path.name)
    assert sr == speaker_id.TARGET_SR
    return pcm


# (fixture, label, start_s, end_s) — real human speech slices. Lengths are
# deliberately varied (1.5 s .. 10 s) so the dynamic time axis is exercised
# on both sides of the 1.5 s trace length.
_SLICES = [
    ("family", "Sage 0-5s", 0.0, 5.0),
    ("family", "Asher 5-10s", 5.0, 10.0),
    ("family", "Sage 10-11.5s (1.5 s)", 10.0, 11.5),
    ("family", "Sage+Asher 0-10s (10 s)", 0.0, 10.0),
    ("poker6", "Player1 0-5s", 0.0, 5.0),
    ("poker6", "Player6/owner 25-30s", 25.0, 30.0),
    ("poker6", "Player3 10-11.5s (1.5 s)", 10.0, 11.5),
]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    export_mod = _load_export_module()
    out = tmp_path_factory.mktemp("onnx") / "ecapa_test.onnx"
    path = export_mod.export(out)
    assert path.exists() and path.stat().st_size > 1_000_000
    return export_mod, path


@pytest.fixture(scope="module")
def audio():
    return {"family": _load_16k(_FAMILY), "poker6": _load_16k(_POKER6)}


def test_onnx_matches_torch_on_real_speech(exported, audio):
    export_mod, path = exported
    session = export_mod.onnx_session(path)
    sr = speaker_id.TARGET_SR
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"\nONNX file size: {size_mb:.1f} MB ({path.stat().st_size} bytes)")

    # Warm the session once so the first timed run isn't paying setup cost.
    export_mod.onnx_embed(session, audio["family"][: int(1.5 * sr)])

    worst = 1.0
    for fixture, label, start, end in _SLICES:
        clip = audio[fixture][int(start * sr): int(end * sr)]
        assert clip.size >= int(1.0 * sr), f"{label}: fixture shorter than expected"

        ref = speaker_id.embed_pcm(clip, sr)
        t0 = time.perf_counter()
        got = export_mod.onnx_embed(session, clip)
        onnx_ms = (time.perf_counter() - t0) * 1000.0

        assert got.shape == (speaker_id.EMBEDDING_DIM,)
        assert np.isclose(float(np.linalg.norm(got)), 1.0, atol=1e-4)
        cos = speaker_id.cosine(got, ref)
        worst = min(worst, cos)
        print(
            f"{fixture:7s} {label:28s} {clip.size / sr:5.2f}s "
            f"cosine(onnx,torch)={cos:.6f} onnxruntime CPU={onnx_ms:.1f} ms"
        )
        assert cos > 0.999, f"{label}: ONNX drifted from torch (cosine {cos:.5f})"
    print(f"worst-case cosine across {len(_SLICES)} slices: {worst:.6f}")


def test_onnx_preserves_speaker_separation(exported, audio):
    """The exported graph must keep the embedding space's DISCRIMINATION, not
    just match torch pointwise: the owner's two family slices should be far
    closer to each other than to his son's slice (the same relationship the
    server's threshold calibration relies on)."""
    export_mod, path = exported
    session = export_mod.onnx_session(path)
    sr = speaker_id.TARGET_SR
    fam = audio["family"]
    sage_a = export_mod.onnx_embed(session, fam[0: 5 * sr])
    sage_b = export_mod.onnx_embed(session, fam[10 * sr: 15 * sr])
    asher = export_mod.onnx_embed(session, fam[5 * sr: 10 * sr])
    same = speaker_id.cosine(sage_a, sage_b)
    diff = speaker_id.cosine(sage_a, asher)
    print(f"\nonnx same-speaker cosine={same:.3f} different-speaker cosine={diff:.3f}")
    assert same > diff + 0.2
