#!/usr/bin/env python3
"""Stitch RAVDESS clips into a two-speaker scene fixture — REAL human voices.

Why this exists (2026-09-06). The scene pack's three fixtures are steerable
TTS: an acted performance of an instruction, at a synthesized dynamic range.
Two things the product needs cannot be proven on them:

  1. **Code E ("you let them finish")** needs somebody to hold the floor for
     more than twelve seconds. Every TTS scene turn is one sentence, five
     seconds at most, so E could only ever be tested on a hand-written vector
     — never end to end through the real loop.
  2. **Instant loudness / vocal activation** needs a real dynamic range. The
     TTS "shout" is about +1 dB over the same voice's calm line (measured, see
     the fixture README), which is why the pack cannot exercise the instant
     tier at all.

RAVDESS (Livingstone & Russo 2018, CC BY-NC-SA 4.0, 24 professional actors)
gives both: real shouting, with per-clip ground truth for emotion AND
intensity. What it cannot give is MEANING — every clip is one of two fixed
sentences ("Kids are talking by the door" / "Dogs are sitting by the door").
So this scene is honest about what it is:

  * the AUDIO is real human affect with labelled intensity;
  * the STRUCTURE (who speaks when, for how long) is composed here;
  * the WORDS are RAVDESS's own two sentences, verbatim — never rewritten
    into dialogue that the audio does not say. Anything that needs meaning
    (code R, "you apologised and they softened") therefore CANNOT be proven
    on this fixture, and is not claimed by it.

The long turn is built by concatenating several clips from the SAME actor
with short in-turn pauses, which is exactly what a monologue is; the loop's
VAD will still cut it into fragments, and coalescing them back into one turn
is the behaviour under test (positiveNudges.ts `coalesceTurns`).

Licence: RAVDESS is CC BY-NC-SA 4.0 — non-commercial. The stitched WAV is a
TEST FIXTURE only; it must never ship inside the app or be used to train
anything shipped. tmp/ravdess/ holds the source corpus (already downloaded).

USAGE
  python scripts/make_ravdess_scene.py                 # writes the fixture
  python scripts/make_ravdess_scene.py --force         # overwrite
  python scripts/make_ravdess_scene.py --dry-run       # plan + measured dB only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RAVDESS = REPO / "tmp" / "ravdess" / "audio"
#: NOT under server/tests/fixtures/audio, and not committed. RAVDESS is
#: CC BY-NC-SA 4.0 — non-commercial and share-alike — so a derivative WAV must
#: not live inside a commercial product repo. It is regenerated from the
#: owner's local corpus in a few seconds by running this script, and every
#: test that uses it skips honestly when it is absent (the same posture as the
#: ECAPA model). tmp/ is gitignored.
OUT_WAV = REPO / "tmp" / "ravdess-scene" / "test_recording_scene_ravdess_pair.wav"
TARGET_SR = 16000

SELF_LABEL = "Speaker A"
PARTNER_LABEL = "Speaker B"
#: Actor 01 is male, actor 02 female — the widest voiceprint separation the
#: corpus offers without hunting, which keeps this fixture about the NUDGE
#: behaviour rather than about diarization difficulty.
SELF_ACTOR = 1
PARTNER_ACTOR = 2

#: RAVDESS emotion code -> (label, our coarse class). 01 neutral, 02 calm,
#: 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised.
EMOTION = {
    "01": ("neutral", "neutral"),
    "02": ("calm", "neutral"),
    "03": ("happy", "happy"),
    "04": ("sad", "sad"),
    "05": ("angry", "angry"),
}
#: Statement code -> the exact words the actor says. Never paraphrased.
STATEMENT = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}

#: One clip: (emotion, intensity, statement, repetition). intensity 01 =
#: normal, 02 = strong (RAVDESS has no strong "neutral").
Clip = tuple[str, str, str, str]

#: The scene, as turns. Each turn is one speaker and a list of clips played
#: back to back with IN_TURN_GAP between them. The whole point of turn 1 is
#: that it is LONG.
SCENE: list[dict] = [
    dict(speaker=SELF_LABEL, emotion="01", clips=[("01", "01", "01", "01")],
         note="calm opener"),
    dict(speaker=PARTNER_LABEL, emotion="02",
         clips=[("02", "01", "01", "01"), ("02", "01", "01", "02"),
                ("02", "02", "01", "01"), ("02", "02", "01", "02"),
                ("01", "01", "01", "01"), ("01", "01", "01", "02")],
         note="THE long turn: six clips from one actor, ~16 s of holding the "
              "floor. Code E exists for exactly this and no TTS scene has one."),
    dict(speaker=SELF_LABEL, emotion="01", clips=[("01", "01", "01", "02")],
         note="short acknowledgement — you took the floor back cleanly"),
    dict(speaker=PARTNER_LABEL, emotion="02", clips=[("02", "01", "01", "01")],
         note="calm reply"),
    dict(speaker=SELF_LABEL, emotion="05",
         clips=[("05", "02", "02", "01"), ("05", "02", "02", "02")],
         note="THE spike: angry at STRONG intensity — real shouting, the "
              "dynamic range synthesized speech does not have. Deliberately the "
              "ONLY turn in the scene that uses RAVDESS statement 02 ('Dogs are "
              "sitting by the door'): the replay's scripted LLM matches a turn's "
              "tone by its WORDS, and with a two-sentence corpus that is the only "
              "way one turn can carry a tone of its own"),
    dict(speaker=PARTNER_LABEL, emotion="04", clips=[("04", "02", "01", "01")],
         note="they go hurt, not angry (strong sad)"),
    dict(speaker=SELF_LABEL, emotion="02", clips=[("02", "01", "01", "02")],
         note="THE recovery: back to calm within one turn of the spike — code D"),
    dict(speaker=PARTNER_LABEL, emotion="03", clips=[("03", "01", "01", "01")],
         note="relieved"),
]

#: Silence between two clips inside one speaker's turn. Above the segmenter's
#: 300 ms cut on purpose: the loop SHOULD split a monologue into fragments,
#: and stitching them back into one turn is the behaviour under test.
IN_TURN_GAP = 0.45
#: Silence between turns.
TURN_GAP = 0.6


def clip_path(actor: int, clip: Clip) -> Path:
    emotion, intensity, statement, repetition = clip
    name = f"03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor:02d}.wav"
    return RAVDESS / f"Actor_{actor:02d}" / name


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg not on PATH — brew install ffmpeg")
    return exe


def read_16k(ffmpeg: str, path: Path) -> np.ndarray:
    """Decode any RAVDESS clip to mono int16 at 16 kHz with a real resampler."""
    out = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(TARGET_SR),
         "-f", "s16le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, dtype="<i2").copy()


#: How far below a clip's own loudest 20 ms frame still counts as speech. A
#: RELATIVE floor, not an absolute one: RAVDESS clips sit around -43 dBFS
#: overall and a fixed -50 dB gate ate the quiet onsets and tails of every
#: calm line (measured 2026-09-06 — the "long" turn came out at 13 s instead
#: of 18). 35 dB below the peak keeps the whole utterance and still drops the
#: room tone.
TRIM_BELOW_PEAK_DB = 35.0
#: Kept either side of the detected speech, so a trimmed clip still starts and
#: ends on silence rather than clipping the first consonant.
TRIM_PAD_S = 0.08


def trim_edge_silence(samples: np.ndarray, sr: int) -> np.ndarray:
    """Drop the lead-in/lead-out silence RAVDESS clips carry (~1 s each end).

    Without this the "long turn" would be half silence and the loop's VAD would
    cut it into far more fragments than a real monologue produces. Measured on
    20 ms frames against a floor [TRIM_BELOW_PEAK_DB] below the clip's own peak
    frame; a clip that is silent throughout is returned unchanged rather than
    emptied.
    """
    frame = max(1, sr // 50)
    n = samples.size // frame
    if n == 0:
        return samples
    frames = samples[: n * frame].astype(np.float64).reshape(n, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    with np.errstate(divide="ignore"):
        dbfs = 20.0 * np.log10(np.maximum(rms, 1e-9) / 32768.0)
    peak = float(dbfs.max())
    voiced = np.flatnonzero(dbfs > peak - TRIM_BELOW_PEAK_DB)
    if voiced.size == 0:
        return samples
    pad = int(TRIM_PAD_S * sr)
    start = max(0, voiced[0] * frame - pad)
    end = min(samples.size, (voiced[-1] + 1) * frame + pad)
    return samples[start:end]


def rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    return -np.inf if rms <= 0 else 20.0 * float(np.log10(rms / 32768.0))


def write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.astype("<i2").tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if OUT_WAV.exists() and not args.force and not args.dry_run:
        print(f"✓ {OUT_WAV.name} already exists — pass --force to regenerate")
        return 0
    if not RAVDESS.exists():
        sys.exit(f"RAVDESS corpus not found at {RAVDESS} (see tmp/ravdess/)")

    ffmpeg = find_ffmpeg()
    turns_meta: list[dict] = []
    pieces: list[np.ndarray] = []
    turn_gap = np.zeros(int(TURN_GAP * TARGET_SR), dtype="<i2")
    in_gap = np.zeros(int(IN_TURN_GAP * TARGET_SR), dtype="<i2")

    for i, turn in enumerate(SCENE):
        actor = SELF_ACTOR if turn["speaker"] == SELF_LABEL else PARTNER_ACTOR
        segs: list[np.ndarray] = []
        clip_names: list[str] = []
        words: list[str] = []
        for j, clip in enumerate(turn["clips"]):
            path = clip_path(actor, clip)
            if not path.exists():
                sys.exit(f"missing RAVDESS clip: {path}")
            audio = trim_edge_silence(read_16k(ffmpeg, path), TARGET_SR)
            if j > 0:
                segs.append(in_gap)
            segs.append(audio)
            clip_names.append(path.name)
            words.append(STATEMENT[clip[2]])
        body = np.concatenate(segs)
        if i > 0:
            pieces.append(turn_gap)
        pieces.append(body)
        label, coarse = EMOTION[turn["emotion"]]
        intensity = "strong" if any(c[1] == "02" for c in turn["clips"]) else "normal"
        turns_meta.append({
            "speaker": turn["speaker"],
            # Verbatim RAVDESS sentences — this fixture never invents words.
            "text": " ".join(words),
            "scripted_emotion": f"{label}_{intensity}",
            "emotion_coarse": coarse,
            "instruction": f"RAVDESS actor {actor:02d} performing {label} at {intensity} intensity",
            "ravdess_clips": clip_names,
            "ravdess_intensity": intensity,
            "duration_sec": round(body.size / TARGET_SR, 4),
            "rms_dbfs": round(rms_dbfs(body), 2),
            "_note": turn["note"],
        })

    audio = np.concatenate(pieces)
    total = audio.size / TARGET_SR

    # The measurement that justifies the whole fixture: how much louder the
    # spike actually is than the same actor's calm turns.
    self_turns = [t for t in turns_meta if t["speaker"] == SELF_LABEL]
    calm = [t["rms_dbfs"] for t in self_turns if t["emotion_coarse"] != "angry"]
    angry = [t["rms_dbfs"] for t in self_turns if t["emotion_coarse"] == "angry"]
    baseline = float(np.median(calm)) if calm else float("nan")
    spike_over = round(max(angry) - baseline, 2) if angry and calm else None

    print(f"scene: {len(turns_meta)} turns, {total:.1f}s")
    for i, t in enumerate(turns_meta):
        print(f"  [{i:2d}] {t['speaker']} {t['scripted_emotion']:<16} "
              f"{t['duration_sec']:5.1f}s  {t['rms_dbfs']:7.2f} dBFS  {t['_note'][:52]}")
    print(f"  self calm baseline (median) {baseline:.2f} dBFS · "
          f"angry spike +{spike_over} dB over it "
          f"(the loudness ladder's rungs are +6 / +10 / +14)")
    longest = max(t["duration_sec"] for t in turns_meta if t["speaker"] == PARTNER_LABEL)
    print(f"  longest partner turn {longest:.1f}s (code E needs >= 12 s)")
    if args.dry_run:
        return 0

    write_wav(OUT_WAV, audio, TARGET_SR)
    meta = {
        "_note": (
            "REAL human affect (RAVDESS, Livingstone & Russo 2018, CC BY-NC-SA 4.0), "
            "stitched into a two-speaker scene by scripts/make_ravdess_scene.py. The AUDIO is "
            "real acted emotion with the corpus's own per-clip emotion AND intensity labels; the "
            "STRUCTURE (who speaks when) is composed; the WORDS are RAVDESS's two fixed sentences "
            "VERBATIM — this fixture never invents dialogue the audio does not say. So it can "
            "prove anything acoustic (loudness spikes, holding the floor, de-escalation) and "
            "cannot prove anything about MEANING (code R, 'you apologised and they softened'), "
            "which is not claimed here. NON-COMMERCIAL source: test fixture only, never shipped "
            "in the app, never training data."
        ),
        "variant": "ravdess-stitch",
        "scene": "ravdess_pair",
        "source_corpus": "RAVDESS (Audio_Speech_Actors_01-24), CC BY-NC-SA 4.0",
        "sample_rate": TARGET_SR,
        "silence_gap_sec": TURN_GAP,
        "in_turn_gap_sec": IN_TURN_GAP,
        "num_speakers_true": 2,
        "self_speaker": SELF_LABEL,
        "speakers": {
            SELF_LABEL: {"voice": f"ravdess_actor_{SELF_ACTOR:02d}", "is_self": True, "role": "self"},
            PARTNER_LABEL: {"voice": f"ravdess_actor_{PARTNER_ACTOR:02d}", "is_self": False, "role": "partner"},
        },
        "turns": turns_meta,
        "measured": {
            "self_calm_baseline_dbfs": round(baseline, 2),
            "angry_spike_db_over_baseline": spike_over,
            "longest_partner_turn_sec": longest,
            "note": (
                "The two numbers this fixture exists for: a real shout is this far over the same "
                "actor's own calm baseline (the instant tier's rungs are +6 / +10 / +14 dB), and "
                "the partner holds the floor this long (code E needs >= 12 s). The TTS pack "
                "manages about +1 dB and 5 s."
            ),
        },
        "expected_nudges": [
            {
                "after_turn_index": i,
                "level": "strong",
                "reason": "self angry at RAVDESS STRONG intensity — a real shout, not an acted one",
            }
            for i, t in enumerate(turns_meta)
            if t["speaker"] == SELF_LABEL and t["emotion_coarse"] == "angry"
        ],
        "expected_nudges_note": (
            "after_turn_index is 0-based into `turns` and always names a SELF turn. Hand-authored "
            "from the composition, like the TTS pack's: a nudge on a non-self turn is a false "
            "positive by construction."
        ),
        "expected_positive_nudges": [
            {
                "code": "E",
                "after_turn_index": 1,
                "reason": "the partner holds the floor for the whole long turn and the user never "
                          "takes it back mid-turn",
            },
            {
                "code": "D",
                "after_turn_index": 6,
                "reason": "self goes from a strong shout to calm within one of their own turns",
            },
        ],
        "expected_positive_nudges_note": (
            "The positive half of the vocabulary (apps/mobile/src/live/positiveNudges.ts). Codes R "
            "and K are deliberately absent: R needs repair LANGUAGE, which this corpus's two fixed "
            "sentences cannot carry, and K is a summary badge over a five-minute quiet run that a "
            "40-second fixture cannot contain."
        ),
    }
    meta_path = OUT_WAV.with_name(OUT_WAV.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"✓ wrote {OUT_WAV} ({total:.1f}s) + {meta_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
