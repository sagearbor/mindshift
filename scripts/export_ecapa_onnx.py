#!/usr/bin/env python3
"""CLI over ``server/ecapa_onnx.py``: export the PINNED ECAPA-TDNN speaker
embedder to a single ONNX graph (Foundation B of the on-device voice-ID plan).

The exporter itself lives in ``server/ecapa_onnx.py`` — read that module's
docstring for the graph contract (``waveform`` f32 [1, T] @ 16 kHz ->
``embedding`` f32 [1, 192], L2-normalized) and for every design decision
(conv-DFT STFT, baked-in front end, opset 17). It moved there because
``GET /models/ecapa.onnx`` (``server/routers/models.py``) generates the file
on demand inside the running API, and the Docker image ships ``server/`` only.
This script is the operator's hand-run entry point and re-exports the
module's functions so ``server/tests/test_ecapa_onnx_parity.py`` (which loads
THIS file by path) keeps exercising exactly what production calls.

Usage::

    tmp/venv-voice/bin/python scripts/export_ecapa_onnx.py            # default path
    tmp/venv-voice/bin/python scripts/export_ecapa_onnx.py --out x.onnx --check
    tmp/venv-voice/bin/python scripts/export_ecapa_onnx.py --reference-json \\
        apps/mobile/__tests__/fixtures/ecapa_reference.json

The default output is ``<MINDSHIFT_ECAPA_CACHE or server/.ecapa_cache>/
ecapa_<revision>.onnx`` — the very file the server serves — so running this
once on a box pre-warms the endpoint (no first-request export stall).

``--check`` runs a torch-vs-onnxruntime parity check on synthetic audio and
prints the file size + CPU latency (the parity test does the same on REAL
speech fixtures).

``--reference-json`` writes the SERVER'S (torch) embeddings for a fixed set
of real-speech slices of the committed test recordings, as a small JSON
fixture. The mobile Jest suite (``apps/mobile/__tests__/liveSpeakerId.test.ts``)
runs the exported graph on the same slices under ``onnxruntime-node`` and
asserts cosine > 0.999 against these — the true cross-runtime parity the
phone depends on (server-enrolled print vs. phone-computed embedding). The
JSON is ~30 KB and IS committed; regenerate it only when the pinned revision
changes (the file records which revision produced it, and the test skips
with a clear reason on a mismatch rather than comparing apples to oranges).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVER_DIR = _REPO_ROOT / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

import numpy as np  # noqa: E402

import speaker_id  # noqa: E402
from ecapa_onnx import (  # noqa: E402, F401 — re-exported for the parity test
    DEFAULT_OPSET,
    INPUT_NAME,
    OUTPUT_NAME,
    build_export_module,
    default_onnx_path,
    export,
    onnx_embed,
    onnx_session,
)

# Backwards-compatible alias: the parity test's docs and older notes call
# this ``default_output_path``.
default_output_path = default_onnx_path

_FIXTURES = _REPO_ROOT / "server" / "tests" / "fixtures" / "audio"

# (fixture wav, label, start_s, end_s) — real human speech, lengths varied on
# both sides of the 1.5 s trace length so the dynamic time axis is exercised.
# Ground truth per the fixtures' *_meta.json (owner-stated turn schedule).
REFERENCE_SLICES = [
    ("test_recording_family_real.wav", "Sage 0-5s", 0.0, 5.0),
    ("test_recording_family_real.wav", "Asher 5-10s", 5.0, 10.0),
    ("test_recording_family_real.wav", "Sage 10-15s", 10.0, 15.0),
    ("test_recording_family_real.wav", "Sage 10-11.5s (1.5 s)", 10.0, 11.5),
    ("test_recording_poker6_real.wav", "Player1 0-5s", 0.0, 5.0),
    ("test_recording_poker6_real.wav", "Player3 10-11.5s (1.5 s)", 10.0, 11.5),
]


def _read_wav_16k(path: Path) -> np.ndarray:
    """Plain RIFF read of a canonical 16 kHz mono int16 wav -> float32 in
    [-1, 1) via ``/ 32768`` — the SAME arithmetic the Jest test applies, so
    the two sides embed identical samples (no ffmpeg in the loop)."""
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != speaker_id.TARGET_SR:
            raise RuntimeError(f"{path.name}: expected 16 kHz mono int16 wav")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def write_reference_json(out: Path) -> Path:
    sr = speaker_id.TARGET_SR
    slices = []
    embeddings: list[list[float]] = []
    for fname, label, start, end in REFERENCE_SLICES:
        pcm = _read_wav_16k(_FIXTURES / fname)
        clip = pcm[int(start * sr): int(end * sr)]
        emb = speaker_id.embed_pcm(clip, sr)
        embeddings.append([round(float(x), 7) for x in emb.tolist()])
        slices.append({
            "fixture": fname,
            "label": label,
            "start_s": start,
            "end_s": end,
            # Swapped for the compact one-line vector below.
            "embedding": f"@@EMB{len(embeddings) - 1}@@",
        })
    doc = {
        "_note": (
            "Server-side (torch + speechbrain) ECAPA embeddings of real-speech "
            "slices of server/tests/fixtures/audio/*_real.wav, generated by "
            "scripts/export_ecapa_onnx.py --reference-json. The mobile Jest "
            "suite embeds the same slices with the ONNX export under "
            "onnxruntime-node and asserts cosine > 0.999. Regenerate when "
            "ECAPA_REVISION changes."
        ),
        "model": f"{speaker_id.ECAPA_SOURCE}@{speaker_id.ECAPA_REVISION}",
        "revision": speaker_id.ECAPA_REVISION,
        "sample_rate": sr,
        "dim": speaker_id.EMBEDDING_DIM,
        "slices": slices,
    }
    # One line per embedding (not one per float): a reviewable ~40-line file
    # instead of a 1200-line one, and a revision bump diffs as six lines.
    text = json.dumps(doc, indent=1)
    for i, emb in enumerate(embeddings):
        text = text.replace(f'"@@EMB{i}@@"', json.dumps(emb))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    return out


def _check(path: Path) -> None:
    """Synthetic-audio parity + size/latency report (real-speech parity lives
    in server/tests/test_ecapa_onnx_parity.py)."""
    import onnx

    onnx.checker.check_model(str(path))
    size_mb = path.stat().st_size / (1024 * 1024)
    session = onnx_session(path)
    ops = sorted({node.op_type for node in onnx.load(str(path)).graph.node})
    print(f"onnx file: {path} ({size_mb:.1f} MB)")
    print(f"op types ({len(ops)}): {', '.join(ops)}")

    rng = np.random.default_rng(0)
    sr = speaker_id.TARGET_SR
    for seconds in (1.0, 1.5, 4.0):
        # Band-limited noise + a tone so it isn't pathological silence.
        t = np.arange(int(seconds * sr)) / sr
        clip = (0.2 * np.sin(2 * np.pi * 180 * t) + 0.05 * rng.standard_normal(t.size)).astype(np.float32)
        ref = speaker_id.embed_pcm(clip, sr)
        onnx_embed(session, clip)  # warm-up
        t0 = time.perf_counter()
        got = onnx_embed(session, clip)
        dt = (time.perf_counter() - t0) * 1000.0
        print(
            f"  {seconds:.1f}s clip: cosine(onnx, torch)={speaker_id.cosine(got, ref):.6f} "
            f"onnxruntime CPU latency={dt:.1f} ms"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path, default=None,
        help=f"output .onnx path (default: {default_onnx_path()})",
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument(
        "--check", action="store_true",
        help="after export, run a torch-vs-onnxruntime parity + latency check",
    )
    parser.add_argument(
        "--reference-json", type=Path, default=None,
        help="ONLY write the torch reference-embedding fixture for the mobile "
             "parity test to this path (no ONNX export)",
    )
    args = parser.parse_args(argv)

    if args.reference_json is not None:
        out = write_reference_json(args.reference_json)
        print(f"wrote {len(REFERENCE_SLICES)} reference embeddings -> {out}")
        return 0

    out = export(args.out, opset=args.opset)
    print(f"exported {speaker_id.ECAPA_SOURCE}@{speaker_id.ECAPA_REVISION} -> {out}")
    if args.check:
        _check(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
