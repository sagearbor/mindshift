"""Live-gated regression tests: local ECAPA diarization on the two OpenAI-TTS
acted fixtures (``server/tests/fixtures/audio/``), scored with EXACT
best-permutation per-turn accuracy against the scripted ground truth.

Ground truth for these two fixtures was independently verified before these
tests were written (see .superpowers/sdd/2026-08-17-diarization-regression/
report.md for the full writeup):

  1. ``scripts/make_test_recording_openai.py``'s ``TURNS`` list diffed
     byte-for-byte against both ``*_meta.json`` files — exact match, all 10
     turns, both fixtures.
  2. Independent local Whisper (faster-whisper) transcription of both WAVs,
     diffed against the meta's ``text`` fields — >=98.75% word-sequence
     similarity on both, zero dropped/garbled/truncated turns (the only diffs
     are trivial ASR artifacts: "touch"/"touched", "want to"/"wanna").
  3. Total file duration vs ``sum(duration_sec) + silence_gap_sec *
     (n_turns-1)`` — exact match (0.0000s diff) on both.

These fixtures were NEVER run through the current diarize_local pipeline
before this file was added (the README's "2 turns misattributed to a
phantom Speaker C" claim for the gptaudio fixture predates 2026-08-14/15's
N-way k-detection, anchor recalibration, and word-level rapid-exchange
splitting). Freshly measured here: BOTH fixtures score 10/10 = 100% exact
per-turn accuracy on the current pipeline.

IMPORTANT sample-rate note: the fixtures are 24 kHz (OpenAI TTS' native
output rate). ``speaker_id.embed_pcm`` hard-requires ``speaker_id.TARGET_SR``
(16 kHz) — feeding it raw native-rate PCM raises ``SpeakerIdUnavailable`` and
``diarize_turns`` returns ``None``. This is NOT a fixture-specific hack: it is
simply honoring diarize_local's documented input contract, the same way
``audio_ingest.decode_to_pcm_16k`` guarantees 16 kHz for the (also
ECAPA-consuming) voice-enrollment endpoint in routers/voice.py. This test
resamples via that SAME production ffmpeg path.

Note this surfaced a separate, real finding worth a follow-up: the
``/analyze/upload`` cross-check in main.py (~line 2416) decodes uploads with
plain ``decode_to_pcm`` (preserves native rate) rather than
``decode_to_pcm_16k`` (guarantees 16 kHz) before calling
``diarize_local.diarize_turns`` — so on any real WAV upload whose native rate
isn't already 16 kHz, the ECAPA cross-check raises internally and is
swallowed by the broad ``except Exception: local = None`` a few lines later,
silently no-opping the very cross-check the code exists to run. See the
report for detail; fixing that is out of scope here (a main.py decode-path
change deserves its own dedicated review, not a drive-by in a fixture test).
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

import audio_ingest
import diarize_local
import speaker_id

_AUDIO_DIR = Path(__file__).resolve().parent / "fixtures" / "audio"

_FIXTURES = {
    "test_recording_openai": _AUDIO_DIR / "test_recording_openai.wav",
    "test_recording_gptaudio": _AUDIO_DIR / "test_recording_gptaudio.wav",
}

pytestmark = [
    pytest.mark.skipif(
        not speaker_id.is_available(),
        reason="voice deps (torch + speechbrain) not installed",
    ),
    pytest.mark.skipif(
        not all(p.exists() for p in _FIXTURES.values()),
        reason=(
            "OpenAI-TTS fixtures missing — generate with "
            "scripts/make_test_recording_openai.py"
        ),
    ),
]


def _load_16k_pcm(wav_path: Path) -> tuple[np.ndarray, int]:
    """Decode via the SAME production path real uploads use for ECAPA
    (``audio_ingest.decode_to_pcm_16k`` -> ffmpeg, forced mono 16 kHz) rather
    than reading the WAV's native rate — ``speaker_id.embed_pcm`` requires
    exactly :data:`speaker_id.TARGET_SR`."""
    pcm, sr = audio_ingest.decode_to_pcm_16k(wav_path.read_bytes(), wav_path.name)
    assert sr == speaker_id.TARGET_SR
    return pcm, sr


def _build_turns(meta: dict) -> list[dict]:
    """Reconstruct each turn's (start_time, end_time) from ``duration_sec`` +
    ``silence_gap_sec`` — these two fixtures' meta.json carries no explicit
    timestamps (unlike the physics fixture), but
    ``make_test_recording_openai.py`` concatenates turns back-to-back with a
    fixed silence gap between them (verified against actual file duration in
    the ground-truth check), so this reconstruction is exact."""
    gap = meta["silence_gap_sec"]
    turns = []
    t = 0.0
    for m in meta["turns"]:
        dur = m["duration_sec"]
        turns.append({
            "speaker": m["speaker"],
            "text": m["text"],
            "start_time": round(t, 4),
            "end_time": round(t + dur, 4),
        })
        t += dur + gap
    return turns


def _best_permutation_accuracy(
    truth: list[str], pred: list[str],
) -> tuple[float, int, dict[str, str]]:
    """Exact per-turn accuracy under the best bijection pred-label ->
    truth-label — the coarser ``partition_agreement`` (pairwise Rand index)
    can score high while individual turns are wrong; this answers "did we
    get every turn right" directly, trying every label permutation (small
    speaker counts, 2..4) and keeping the best."""
    truth_labels = sorted(set(truth))
    pred_labels = sorted(set(pred))
    best_acc, best_correct, best_map = -1.0, 0, {}
    width = min(len(truth_labels), len(pred_labels))
    for perm in itertools.permutations(truth_labels, width):
        mapping = dict(zip(pred_labels, perm))
        correct = sum(1 for t, p in zip(truth, pred) if mapping.get(p) == t)
        acc = correct / len(truth)
        if acc > best_acc:
            best_acc, best_correct, best_map = acc, correct, mapping
    return best_acc, best_correct, best_map


def _diarize_and_score(name: str) -> tuple[float, int, int, list[dict]]:
    """Run diarize_turns on fixture ``name`` and return (accuracy, correct,
    total, detail-rows-for-diagnostics)."""
    wav_path = _FIXTURES[name]
    meta_path = wav_path.with_name(wav_path.stem + "_meta.json")
    meta = json.loads(meta_path.read_text())
    pcm, sr = _load_16k_pcm(wav_path)
    turns = _build_turns(meta)
    truth = [t["speaker"] for t in turns]

    got = diarize_local.diarize_turns(pcm, sr, [dict(t) for t in turns])
    assert got is not None, (
        f"{name}: local diarization returned nothing trustworthy on a "
        "2-speaker fixture with 10 embeddable turns"
    )
    pred = [t["speaker"] for t in got["turns"]]
    assert len(pred) == len(truth), (
        f"{name}: predicted {len(pred)} turns, expected {len(truth)} "
        f"(word-level splitting fired unexpectedly: "
        f"split_utterances={got['split_utterances']})"
    )

    acc, correct, mapping = _best_permutation_accuracy(truth, pred)
    detail = [
        {
            "turn": i,
            "emotion": meta["turns"][i]["scripted_emotion"],
            "truth": t,
            "pred": p,
            "mapped": mapping.get(p, "???"),
            "ok": mapping.get(p) == t,
        }
        for i, (t, p) in enumerate(zip(truth, pred))
    ]
    return acc, correct, len(truth), detail


def _format_detail(name: str, got_acc: float, correct: int, total: int,
                    detail: list[dict]) -> str:
    lines = [
        f"{name}: exact per-turn accuracy {correct}/{total} = {got_acc:.4f}",
        "  turn  emotion               truth        pred         mapped       ok",
    ]
    for d in detail:
        lines.append(
            f"  {d['turn']:>4}  {d['emotion']:<20}  {d['truth']:<11} "
            f"{d['pred']:<12} {d['mapped']:<12} {d['ok']}"
        )
    return "\n".join(lines)


def test_ecapa_clustering_openai_fixture_full_accuracy():
    """Clean 2-speaker TTS fixture: freshly measured at 10/10 = 100% exact
    per-turn accuracy (2026-08-17, current pipeline). Pinned at the true
    ceiling — if this regresses below 1.0, investigate rather than lower the
    bar (see report for the sample-rate contract this depends on)."""
    name = "test_recording_openai"
    acc, correct, total, detail = _diarize_and_score(name)
    assert acc == 1.0, _format_detail(name, acc, correct, total, detail)


def test_ecapa_clustering_gptaudio_fixture_full_accuracy():
    """Harder acted-TTS fixture (more expressive engine; README's 2026-07-12
    claim of a phantom Speaker C predates several k-detection/anchor
    improvements). Freshly measured at 10/10 = 100% exact per-turn accuracy
    (2026-08-17, current pipeline) — the old regression is gone. Pinned at
    the actual measured floor per the brief (not force-forgiven if it drops)."""
    name = "test_recording_gptaudio"
    acc, correct, total, detail = _diarize_and_score(name)
    assert acc == 1.0, _format_detail(name, acc, correct, total, detail)
