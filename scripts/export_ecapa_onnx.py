#!/usr/bin/env python3
"""Export the PINNED ECAPA-TDNN speaker embedder to a single ONNX graph.

Why this exists (Foundation B of the on-device voice-ID plan): today the only
way to compute a voiceprint is the server's ``speaker_id.embed_pcm`` (torch +
speechbrain, hundreds of MB of deps, ~15-20 s cold per call on CPU). To match
"who is speaking" on the phone itself we need the SAME embedding space in a
runtime the phone can host — ``onnxruntime-react-native``. This script turns
the exact pinned checkpoint ``speaker_id.ECAPA_SOURCE @ speaker_id.ECAPA_REVISION``
into one self-contained ``.onnx`` whose output is bit-for-bit the same
*space* as the server's voiceprints, so a print enrolled on the server can be
matched on-device and vice versa (parity is asserted by
``server/tests/test_ecapa_onnx_parity.py``: cosine > 0.999 on real speech).

Contract of the exported graph::

    input  "waveform"  float32 [1, T]   mono PCM at 16 kHz, T dynamic
    output "embedding" float32 [1, 192] L2-normalized speaker embedding

i.e. exactly what ``speaker_id.embed_pcm`` returns for the same samples.

What is baked in, and why:

* **The whole speechbrain pipeline**, not just the TDNN. ``EncoderClassifier
  .encode_batch`` is ``compute_features`` (Fbank: STFT -> power spectrum ->
  80 mel filters -> dB + top_db clip) -> ``mean_var_norm`` (per-utterance
  mean subtraction, no std scaling) -> ``embedding_model`` (ECAPA-TDNN). A
  phone can't reproduce the front end "by hand" without drifting from the
  server, so all three stages live inside the graph and the phone feeds raw
  waveform. The final L2 normalization ``embed_pcm`` applies is baked in too.
* **The STFT is a fixed-kernel Conv1d, NOT the ONNX ``STFT`` op.** ``torch.stft``
  exports to the opset-17 ``STFT`` op, but that op (and the ``DFT`` it lowers
  to) is a niche signal-processing kernel whose availability in reduced/mobile
  onnxruntime builds is not something to bet a product on. A 400-tap Conv1d
  with a hamming-windowed DFT basis (cos rows + sin rows, stride = hop) is
  mathematically the identical STFT and uses only the most universally
  supported op in existence. Cost: ~130 KB of constant weights.
* **wav_lens is a constant 1.0.** ``encode_batch`` takes relative lengths to
  mask zero-padding in a batch; the phone embeds ONE clip at a time so the
  clip is always full-length. Baking ``lengths = ones(1)`` removes an input
  the caller could only ever set to 1.0 and keeps every length-derived mask
  a no-op (the masks still exist in the graph — ``length_to_mask``'s
  ``Range``/``Less`` ops — but they select everything).
* **Opset 17**, the newest opset onnxruntime 1.29 accepts that keeps
  ``ReduceMax``/``ReduceMean`` axes as attributes (opset 18 moves them to a
  dynamic input, which some mobile EPs handle less gracefully). Nothing here
  needs anything newer.

The ONNX file is NOT committed (it lands under the gitignored ``tmp/``): it is
~80 MB of weights that are fully reproducible from the pinned HF revision by
re-running this script.

Usage::

    tmp/venv-voice/bin/python scripts/export_ecapa_onnx.py            # default path
    tmp/venv-voice/bin/python scripts/export_ecapa_onnx.py --out x.onnx --check

``--check`` runs a torch-vs-onnxruntime parity check on synthetic audio and
prints the file size + CPU latency (the parity test does the same on REAL
speech fixtures).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVER_DIR = _REPO_ROOT / "server"
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

import numpy as np  # noqa: E402

import speaker_id  # noqa: E402

# Graph I/O names — the phone-side loader binds to these; keep them stable.
INPUT_NAME = "waveform"
OUTPUT_NAME = "embedding"

# See the module docstring for why 17 and not newer.
DEFAULT_OPSET = 17


def default_output_path() -> Path:
    """``tmp/models/ecapa_voxceleb_<revision>.onnx`` under the repo root — the
    revision is in the filename so two exports from different pins can never
    be confused for one another."""
    return _REPO_ROOT / "tmp" / "models" / f"ecapa_voxceleb_{speaker_id.ECAPA_REVISION}.onnx"


def build_export_module(classifier):
    """Wrap the loaded speechbrain ``EncoderClassifier`` in a plain
    ``torch.nn.Module`` whose ``forward(waveform)`` reproduces
    ``encode_batch`` + L2 normalization using only export-friendly ops.

    Everything numeric is PULLED FROM the loaded classifier (its STFT window,
    filterbank centre/band parameters, dB constants, and the TDNN weights) —
    nothing is re-derived from documentation, so the graph can't silently
    disagree with the checkpoint it was exported from.
    """
    import torch
    import torch.nn.functional as F

    fbank_mod = classifier.mods.compute_features  # speechbrain Fbank
    stft = fbank_mod.compute_STFT
    fb = fbank_mod.compute_fbanks
    if fbank_mod.deltas or fbank_mod.context:
        raise RuntimeError("unexpected Fbank config (deltas/context on)")
    if not (stft.center and stft.pad_mode == "constant" and stft.onesided):
        raise RuntimeError("unexpected STFT config; conv-DFT assumes center+constant pad")
    if stft.normalized_stft:
        raise RuntimeError("unexpected STFT config: normalized_stft")

    n_fft = int(stft.n_fft)
    win_length = int(stft.win_length)
    hop = int(stft.hop_length)
    n_freqs = n_fft // 2 + 1

    # --- DFT basis as a conv kernel. torch.stft with win_length == n_fft
    # multiplies each frame by the window then takes the one-sided rFFT:
    #   X[k] = sum_n x[n] w[n] (cos(2*pi*k*n/N) - i*sin(2*pi*k*n/N)).
    # Only |X|^2 = re^2 + im^2 is consumed downstream, so the sign of the
    # imaginary basis is irrelevant. A shorter window would be centre-padded
    # to n_fft by torch.stft; assert equality so that case can't sneak in.
    if win_length != n_fft:
        raise RuntimeError("conv-DFT export assumes win_length == n_fft")
    window = stft.window.detach().to(torch.float64)  # hamming(400)
    n = torch.arange(n_fft, dtype=torch.float64)
    k = torch.arange(n_freqs, dtype=torch.float64)
    angle = 2.0 * math.pi * torch.outer(k, n) / n_fft  # (n_freqs, n_fft)
    basis = torch.cat([torch.cos(angle), -torch.sin(angle)], dim=0) * window
    dft_weight = basis.to(torch.float32).unsqueeze(1)  # (2*n_freqs, 1, n_fft)

    # --- Mel filterbank matrix, built EXACTLY the way Filterbank.forward does
    # for a frozen filterbank (param_rand_factor only applies in training).
    with torch.no_grad():
        f_central_mat = fb.f_central.repeat(fb.all_freqs_mat.shape[1], 1).transpose(0, 1)
        band_mat = fb.band.repeat(fb.all_freqs_mat.shape[1], 1).transpose(0, 1)
        fbank_matrix = fb._create_fbank_matrix(f_central_mat, band_mat).to(torch.float32)
    if not fb.log_mel:
        raise RuntimeError("unexpected Filterbank config: log_mel off")

    norm = classifier.mods.mean_var_norm
    if norm.norm_type != "sentence" or norm.std_norm:
        raise RuntimeError(
            f"unexpected mean_var_norm config: {norm.norm_type}/std_norm={norm.std_norm}"
        )

    class EcapaEmbedder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("dft_weight", dft_weight)
            self.register_buffer("fbank_matrix", fbank_matrix)
            self.n_freqs = n_freqs
            self.hop = hop
            self.pad = n_fft // 2
            self.multiplier = float(fb.multiplier)
            self.db_offset = float(fb.multiplier * fb.db_multiplier)
            self.amin = float(fb.amin)
            self.top_db = float(fb.top_db)
            self.embedding_model = classifier.mods.embedding_model

        def forward(self, waveform):  # (1, T) float32
            x = waveform.unsqueeze(1)  # (1, 1, T)
            # center=True, pad_mode="constant": zero-pad n_fft//2 each side.
            x = F.pad(x, (self.pad, self.pad))
            spec = F.conv1d(x, self.dft_weight, stride=self.hop)  # (1, 2F, frames)
            re = spec[:, : self.n_freqs, :]
            im = spec[:, self.n_freqs:, :]
            power = (re * re + im * im).transpose(1, 2)  # (1, frames, F)
            # Filterbank: power @ mel, then amplitude_to_DB with top_db clip
            # (max over BOTH time and mel of THIS utterance).
            mel = torch.matmul(power, self.fbank_matrix)  # (1, frames, n_mels)
            x_db = self.multiplier * torch.log10(torch.clamp(mel, min=self.amin))
            x_db = x_db - self.db_offset
            floor = x_db.amax(dim=(-2, -1), keepdim=True) - self.top_db
            x_db = torch.maximum(x_db, floor)
            # mean_var_norm (sentence, std_norm=False): subtract the
            # per-utterance mean over time; no std scaling.
            feats = x_db - x_db.mean(dim=1, keepdim=True)
            # ECAPA-TDNN with a full-length (1.0) relative length baked in.
            lengths = torch.ones(feats.shape[0], dtype=feats.dtype)
            emb = self.embedding_model(feats, lengths)  # (1, 1, 192)
            emb = emb.squeeze(1)  # (1, 192)
            # embed_pcm's l2_normalize.
            return emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    module = EcapaEmbedder()
    module.eval()
    return module


def export(out_path: Path | str | None = None, *, opset: int = DEFAULT_OPSET) -> Path:
    """Load the pinned classifier (via ``speaker_id._load_model`` — same cache,
    same revision pin, same fallback logging as production) and write the ONNX
    graph to ``out_path``. Returns the path written."""
    import torch

    out = Path(out_path) if out_path is not None else default_output_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    classifier = speaker_id._load_model()
    module = build_export_module(classifier)

    # A 1.5 s dummy clip for tracing; the time axis is marked dynamic so the
    # traced graph accepts any T >= one frame.
    dummy = torch.zeros(1, int(1.5 * speaker_id.TARGET_SR), dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            module,
            (dummy,),
            str(out),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {1: "samples"}},
            opset_version=opset,
            do_constant_folding=True,
            # The TorchScript-tracing exporter: the graph is a plain feed-forward
            # trace with dynamic shapes handled via dynamic_axes, which is the
            # well-trodden path for onnxruntime-mobile consumers.
            dynamo=False,
        )
    return out


def onnx_session(path: Path | str):
    """A CPU onnxruntime session for the exported graph (shared by --check and
    the parity test)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def onnx_embed(session, pcm: np.ndarray) -> np.ndarray:
    """Run the exported graph on one mono float32 clip -> (192,) embedding."""
    wav = np.ascontiguousarray(pcm, dtype=np.float32).reshape(1, -1)
    (out,) = session.run([OUTPUT_NAME], {INPUT_NAME: wav})
    return out.reshape(-1).astype(np.float32)


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
        help=f"output .onnx path (default: {default_output_path().relative_to(_REPO_ROOT)})",
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument(
        "--check", action="store_true",
        help="after export, run a torch-vs-onnxruntime parity + latency check",
    )
    args = parser.parse_args(argv)

    out = export(args.out, opset=args.opset)
    print(f"exported {speaker_id.ECAPA_SOURCE}@{speaker_id.ECAPA_REVISION} -> {out}")
    if args.check:
        _check(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
