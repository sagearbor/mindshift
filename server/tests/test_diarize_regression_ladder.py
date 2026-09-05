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

_REAL_FIXTURE = _AUDIO_DIR / "test_recording_family_real.wav"

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


@pytest.mark.skipif(
    not _REAL_FIXTURE.exists(),
    reason="real family fixture missing (test_recording_family_real.wav)",
)
def test_ecapa_clustering_family_real_fixture_full_accuracy():
    """The project's FIRST real (not synthesized) calibration fixture: the
    owner and his son alternating in strict ~5-second turns for ~30s
    (2026-08-18). Deepgram's own diarization heard only ONE voice for the
    entire clip on this file — a real vendor diarization failure on real,
    very distinct human voices (an adult and a child), not a synthetic-
    fixture edge case. The local ECAPA cross-check (this function) is what
    actually produces the correct split; this test exists to make sure it
    keeps doing so.

    Ground truth here is the OWNER'S OWN STATED SCHEDULE (see the meta
    file's ``_note``), not independently re-verified the way a synthetic
    fixture's metadata needs to be — the owner directly authored it by
    recording the clip himself.

    Freshly measured 2026-08-18: 8/8 = 100% exact per-turn accuracy (best-
    permutation matching; the pipeline's own turn segmentation naturally
    produced 8 turns from 8 speech-pause boundaries, not a rigid re-cut to
    6 — two turns straddle a 5s boundary by ~1s but still landed on the
    majority-correct speaker). Pinned at that measured ceiling."""
    wav_path = _REAL_FIXTURE
    meta = json.loads(wav_path.with_name(wav_path.stem + "_meta.json").read_text())
    pcm, sr = _load_16k_pcm(wav_path)
    turns = [dict(t) for t in meta["turns"]]
    truth = [t["speaker"] for t in turns]

    got = diarize_local.diarize_turns(pcm, sr, [dict(t) for t in turns])
    assert got is not None, (
        "family_real: local diarization returned nothing trustworthy on a "
        "2-speaker real recording"
    )
    pred = [t["speaker"] for t in got["turns"]]
    assert len(pred) == len(truth), (
        f"family_real: predicted {len(pred)} turns, expected {len(truth)} "
        f"(word-level splitting behaved differently than the 2026-08-18 "
        f"measurement: split_utterances={got['split_utterances']})"
    )

    acc, correct, mapping = _best_permutation_accuracy(truth, pred)
    detail = [
        {
            "turn": i,
            "truth": t,
            "pred": p,
            "mapped": mapping.get(p, "???"),
            "ok": mapping.get(p) == t,
        }
        for i, (t, p) in enumerate(zip(truth, pred))
    ]
    lines = [
        f"family_real: exact per-turn accuracy {correct}/{len(truth)} = {acc:.4f}",
        "  turn  truth        pred         mapped       ok",
    ]
    for d in detail:
        lines.append(
            f"  {d['turn']:>4}  {d['truth']:<11} {d['pred']:<12} "
            f"{d['mapped']:<12} {d['ok']}"
        )
    assert acc == 1.0, "\n".join(lines)


_POKER6_FIXTURE = _AUDIO_DIR / "test_recording_poker6_real.wav"


@pytest.mark.skipif(
    not _POKER6_FIXTURE.exists(),
    reason="poker6 real fixture missing (test_recording_poker6_real.wav)",
)
def test_ecapa_clustering_poker6_fixture_full_accuracy():
    """A real 6-speaker recording (owner's poker night, ~30s, 6 real men in
    strict turn order) that exposed a genuine gap: the pipeline correctly
    finds up to MAX_SPEAKERS_LOCAL=6 candidate speakers, but the shipped
    STRONG_SEPARATION_COSINE=0.30 / NEW_VOICE_ANCHOR_COSINE=0.20 thresholds
    rejected the real 6th voice's split by a hair (marginal cosine 0.301 vs
    the 0.30 bar; anchor 0.231 vs the 0.20 bar) — undercounting 5 real
    voices instead of 6.

    RECALIBRATED 2026-08-24 (STRONG_SEPARATION_COSINE 0.30->0.32,
    NEW_VOICE_ANCHOR_COSINE 0.20->0.24 — see diarize_local.py's comments on
    both constants for the full investigation, including why the fixture
    that originally justified the OLD anchor value doesn't regress at the
    new one). Freshly measured 2026-08-24: 6/6 = 100% exact per-turn
    accuracy, num_speakers=6 (previously 5). Pinned at that measured
    ceiling — if this regresses, investigate rather than lower the bar.

    Ground truth here is the OWNER'S OWN STATED SCHEDULE (see the meta
    file's ``_note``), approximate turn boundaries (+/- 1-2s slop per the
    owner), not independently re-verified.
    """
    wav_path = _POKER6_FIXTURE
    meta = json.loads(wav_path.with_name(wav_path.stem + "_meta.json").read_text())
    pcm, sr = _load_16k_pcm(wav_path)
    turns = [
        {"speaker": t["speaker"], "start_time": t["approx_start"], "end_time": t["approx_end"]}
        for t in meta["approx_turns"]
    ]
    truth = [t["speaker"] for t in turns]

    got = diarize_local.diarize_turns(pcm, sr, [dict(t) for t in turns])
    assert got is not None, (
        "poker6: local diarization returned nothing trustworthy on a "
        "6-speaker real recording"
    )
    pred = [t["speaker"] for t in got["turns"]]
    assert len(pred) == len(truth), (
        f"poker6: predicted {len(pred)} turns, expected {len(truth)} "
        f"(word-level splitting behaved differently than the 2026-08-24 "
        f"measurement: split_utterances={got['split_utterances']})"
    )

    acc, correct, mapping = _best_permutation_accuracy(truth, pred)
    detail = [
        {
            "turn": i,
            "truth": t,
            "pred": p,
            "mapped": mapping.get(p, "???"),
            "ok": mapping.get(p) == t,
        }
        for i, (t, p) in enumerate(zip(truth, pred))
    ]
    lines = [
        f"poker6: exact per-turn accuracy {correct}/{len(truth)} = {acc:.4f}, "
        f"num_speakers={got['num_speakers']}",
        "  turn  truth        pred         mapped       ok",
    ]
    for d in detail:
        lines.append(
            f"  {d['turn']:>4}  {d['truth']:<11} {d['pred']:<12} "
            f"{d['mapped']:<12} {d['ok']}"
        )
    assert got["num_speakers"] == 6, "\n".join(lines)
    assert acc == 1.0, "\n".join(lines)


# ---------------------------------------------------------------------------
# Windows-first engine counterparts (2026-08-30) — diarize_windows_first labels
# the audio transcript-free and regroups the words; production's default
# engine since 2026-08-30 (MINDSHIFT_DIARIZE_ENGINE). Same fixtures, same
# best-permutation scoring, scored PER INPUT UTTERANCE: an utterance's label is
# the duration-weighted majority of the engine's output turns inside it
# ("(untranscribed)" turns for uncovered speech are not scored).
# ---------------------------------------------------------------------------

def _windows_first_utterance_labels(turns: list[dict], got: dict) -> list[str]:
    out = []
    for t in turns:
        s, e = float(t["start_time"]), float(t["end_time"])
        weight: dict[str, float] = {}
        for o in got["turns"]:
            if o["text"] == diarize_local.UNTRANSCRIBED_TEXT:
                continue
            mid = (float(o["start_time"]) + float(o["end_time"])) / 2
            if s <= mid <= e:
                weight[o["speaker"]] = weight.get(o["speaker"], 0.0) + (
                    float(o["end_time"]) - float(o["start_time"])
                )
        out.append(max(weight, key=weight.get) if weight else "???")
    return out


def _windows_first_score(name: str, pcm, sr, turns: list[dict]) -> tuple:
    truth = [t["speaker"] for t in turns]
    got = diarize_local.diarize_windows_first(pcm, sr, [dict(t) for t in turns])
    assert got is not None, f"{name}: the windows engine returned nothing"
    assert got["source"] == diarize_local.SOURCE_WINDOWS
    pred = _windows_first_utterance_labels(turns, got)
    acc, correct, mapping = _best_permutation_accuracy(truth, pred)
    lines = [
        f"{name} (windows engine): num_speakers={got['num_speakers']}, "
        f"{correct}/{len(truth)} = {acc:.4f}, eigengap k="
        f"{got['k_evaluated'][0]['k_eigengap']} (eigenvalues "
        f"{got['k_evaluated'][0]['eigenvalues']}), {len(got['turns'])} turn(s), "
        f"{got['split_utterances']} split, {got['uncovered_turns']} untranscribed, "
        f"segments={[(s['start'], s['end'], s['label']) for s in got['segments']]}",
        "  turn  truth        pred         mapped       ok",
    ]
    for i, (t, p) in enumerate(zip(truth, pred)):
        lines.append(
            f"  {i:>4}  {t:<11} {p:<12} {mapping.get(p, '???'):<12} {mapping.get(p) == t}"
        )
    return got, acc, correct, len(truth), "\n".join(lines)


def _tts_turns(name: str) -> tuple:
    wav_path = _FIXTURES[name]
    meta = json.loads(wav_path.with_name(wav_path.stem + "_meta.json").read_text())
    pcm, sr = _load_16k_pcm(wav_path)
    return pcm, sr, _build_turns(meta)


def test_windows_first_openai_fixture_full_accuracy():
    """Measured 2026-08-30: eigengap k=2, 10/10 exact per-utterance accuracy
    (frame accuracy 1.000 vs the scripted boundaries). Pinned at the
    ceiling."""
    pcm, sr, turns = _tts_turns("test_recording_openai")
    got, acc, correct, total, msg = _windows_first_score("openai", pcm, sr, turns)
    assert got["num_speakers"] == 2, msg
    assert acc == 1.0, msg


def test_windows_first_gptaudio_fixture_full_accuracy():
    """Measured 2026-08-30: eigengap k=2, 10/10 (frame accuracy 1.000)."""
    pcm, sr, turns = _tts_turns("test_recording_gptaudio")
    got, acc, correct, total, msg = _windows_first_score("gptaudio", pcm, sr, turns)
    assert got["num_speakers"] == 2, msg
    assert acc == 1.0, msg


@pytest.mark.skipif(
    not _REAL_FIXTURE.exists(),
    reason="real family fixture missing (test_recording_family_real.wav)",
)
def test_windows_first_family_real_fixture_full_accuracy():
    """Owner + son (real). Measured 2026-08-30: eigengap k=2, frame accuracy
    0.980 on the owner's boundaries (0.949 on his real Deepgram transcript,
    which heard ONE voice; the utterance engine 1.000 / 0.974), 7/8 per
    utterance: the one miss is the son's 0.5 s "And" at 15.43-15.93 s, an
    interjection a 1.5 s-window timeline cannot resolve (its segment runs
    10.38-15.88 s for the owner) — 2 % of the speech, and the reason the
    utterance engine keeps the last word on short turns (SPECTRAL_MIN_RUN_
    SECONDS). Pinned at 7/8 with the miss named; every other turn must land."""
    meta = json.loads(_REAL_FIXTURE.with_name(_REAL_FIXTURE.stem + "_meta.json").read_text())
    pcm, sr = _load_16k_pcm(_REAL_FIXTURE)
    turns = [dict(t) for t in meta["turns"]]
    got, acc, correct, total, msg = _windows_first_score("family_real", pcm, sr, turns)
    assert got["num_speakers"] == 2, msg
    assert correct == 7 and total == 8, msg
    pred = _windows_first_utterance_labels(turns, got)
    wrong = [i for i, (t, p) in enumerate(zip(turns, pred))]
    # The miss is turn 4 (the 0.5 s "And") and nothing else.
    truth = [t["speaker"] for t in turns]
    _, _, mapping = _best_permutation_accuracy(truth, pred)
    wrong = [i for i, (t, p) in enumerate(zip(truth, pred)) if mapping.get(p) != t]
    assert wrong == [4], msg


@pytest.mark.skipif(
    not _POKER6_FIXTURE.exists(),
    reason="poker6 real fixture missing (test_recording_poker6_real.wav)",
)
def test_windows_first_poker6_fixture_full_accuracy():
    """Six real men. Measured 2026-08-30: the eigengap (max k 8, B's
    range) says 7 — one player's ~5 s turn splits into two clusters at the
    fixture's ±1-2 s boundary slop (the bake-off's 0.809 / k=7 timeline,
    reproduced here to the frame) — but on the owner's boundaries every
    utterance lands on its own player and the seventh cluster claims no
    words, so num_speakers=6 and 6/6. On the owner's real Deepgram
    transcript (ONE speaker, 7 utterances, 36 % of the speech never
    transcribed) the same engine scores 0.720 with 7 speakers (the
    untranscribed stretches carry the seventh) vs the utterance engine's
    0.447 / 4 voices. Pinned: 6/6 and 6 speakers on these boundaries."""
    meta = json.loads(_POKER6_FIXTURE.with_name(_POKER6_FIXTURE.stem + "_meta.json").read_text())
    pcm, sr = _load_16k_pcm(_POKER6_FIXTURE)
    turns = [
        {"speaker": t["speaker"], "start_time": t["approx_start"], "end_time": t["approx_end"]}
        for t in meta["approx_turns"]
    ]
    got, acc, correct, total, msg = _windows_first_score("poker6", pcm, sr, turns)
    assert got["k_evaluated"][0]["k_eigengap"] == 7, msg
    assert got["num_speakers"] == 6, msg
    assert acc == 1.0, msg
