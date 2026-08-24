#!/usr/bin/env python3
"""Generate the multi-speaker ESCALATION SCENE fixtures (Foundation D).

Three acted scenes, each with a designated "self" speaker, known per-turn
emotions, and a hand-authored expected-nudge timeline, so the three product
tracks (watch nudge, phone post-analysis, realtime phone coaching incl.
speaker-phone mode) all test against ONE shared pack:

  couple_escalation  2 voices (self + partner), ~70s, 13 turns.
                     calm -> tense -> SELF loses temper (tense_rising,
                     shout_angry, cold_contempt) -> partner hurt -> self
                     repairs -> calm close.
  family3            3 voices (self + 2), ~67s, 15 turns, rapid turn-taking.
                     One escalation by a NON-self speaker (nudge must NOT
                     fire) and one brief self flare (defensive_rising) that
                     should produce exactly one mild nudge.
  meeting4           4 voices (self + 3), ~80s, 17 turns, mostly neutral,
                     with one shout_angry by self late — exercises speaker-
                     count discovery at k=4 plus a late escalation over a
                     long neutral stretch (a false-nudge trap).

The SELF speaker is "Speaker A" in every scene and always uses the SAME
OpenAI voice (SELF_VOICE) so a voiceprint enrolled from one scene can be
matched against the others (the cross-scene enrollment test in
server/tests/test_diarize_scenes.py depends on this).

Engine: the same steerable TTS as scripts/make_test_recording_openai.py
(gpt-4o-mini-tts, pinned snapshot with alias fallback) — this module REUSES
that script's request/decode/write helpers rather than copying them, so a
fix there is a fix here. Output differs in one deliberate way: scenes are
written at 16 kHz (speaker_id.TARGET_SR) instead of the TTS engine's native
24 kHz. Reasons: (1) the local ECAPA diarizer hard-requires 16 kHz, and the
README's "Lessons" record how a native-rate fixture once masked a silent
no-op in the upload cross-check; (2) size — the pack budget is ~15 MB and
16 kHz mono int16 is 1.92 MB/min. The resample is done PER TURN with a real
ffmpeg resampler (anti-aliased), never the linear-interp helper in the
sibling script (that one is fine for 24k->24k no-ops, not for downsampling),
so per-turn `duration_sec` is measured on the exact 16 kHz samples that get
concatenated and `sum(duration_sec) + gap*(n-1)` equals the file length.

Voice selection was MEASURED, not guessed: a scratch probe embedded one
neutral + one shouted line per OpenAI voice (all 13, incl. marin/cedar)
with the production ECAPA model and searched for the most separable voice
sets containing a common hub voice. `onyx` was the best hub (probe cosine
to ballad 0.09, fable 0.15, nova 0.16, coral 0.18, sage 0.18, marin 0.19),
which is why it is the self voice. couple (onyx+coral) and family3
(onyx+ballad+nova) diarize at 100%. meeting4 is the documented ceiling:
FOUR 4-voice sets were generated and measured (alloy/ballad/sage,
echo/ballad/nova, cedar/ballad/nova, marin/ballad/nova) and none cleared
diarize_local.STRONG_SEPARATION_COSINE (0.32) at k=4 — real pooled
centroids run 0.03-0.09 above the probe's estimate, and OpenAI's female
voices sit 0.45-0.74 from one another. The shipped set (marin/ballad/nova)
is the one whose failure mode is least harmful for the pack's purpose: the
three colleagues merge into one cluster but SELF stays perfectly isolated
(pooled cosine to the others 0.09-0.19), so the nudge timeline is still
testable. Measured numbers + the full story: the fixture README's table
and server/tests/test_diarize_scenes.py.

GROUND TRUTH CAVEATS (also in each meta's `_note`):
  * Emotion is ACTED (the model performing `instruction`), not a DSP-derived
    measurement — no expected pitch/energy labels, same as the sibling script.
  * `expected_nudges` is HAND-AUTHORED from the script: it says which self
    turns a coach SHOULD react to and how hard, by construction of the
    scene. It is a spec for the product tracks, not a measurement of any
    current model's output.
  * `emotion_coarse` collapses each `scripted_emotion` into one of
    neutral|angry|sad|happy so a tone-scoring test never has to re-derive the
    mapping (see EMOTION_COARSE below for the exact table).

USAGE
  python3 scripts/make_test_recording_scenes.py [--scene all] \
      [--out-dir server/tests/fixtures/audio] [--force]

REQUIREMENTS
  OPENAI_API_KEY in the repo-root .env (or exported). ffmpeg on PATH (or the
  imageio-ffmpeg static binary). No key / no ffmpeg -> exit 1 with a clear
  message; never a faked recording.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np

# Reuse the sibling generator's helpers (same directory, so a plain import
# works both when run as a script and when imported by tests).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_test_recording_openai import (  # noqa: E402
    REPO_ROOT, VARIANTS, read_env, synth_turn, write_wav,
)

TARGET_SR = 16_000  # == server/speaker_id.TARGET_SR; kept literal so this
                    # script has no server/ import dependency.

# The one voice shared by every scene's self speaker (see docstring).
SELF_VOICE = "onyx"
SELF_LABEL = "Speaker A"

# scripted_emotion -> coarse class. Every label used in SCENES must appear
# here (enforced at import time below, and again by the pure fixture test).
# The four classes are the ones the tone track scores; "scared" collapses to
# sad because it is low-valence/high-uncertainty and — per the README's
# listener probe — even gpt-audio confuses the two, so a finer split would
# be pretending to a ground truth we can't act.
EMOTION_COARSE: dict[str, str] = {
    "calm_open": "neutral",
    "calm_guarded": "neutral",
    "calm_neutral": "neutral",
    "calm_deescalate": "neutral",
    "calm_close": "neutral",
    "repair_apology": "neutral",
    "repair_hopeful": "neutral",
    "tense_rising": "angry",
    "defensive_rising": "angry",
    "shout_angry": "angry",
    "cold_contempt": "angry",
    "hurt_sad": "sad",
    "sullen_sad": "sad",
    "scared_shaky": "sad",
    "warm_happy": "happy",
}

# Delivery directions, one per scripted_emotion, shared across scenes so the
# same label always means the same acting instruction.
INSTRUCTIONS: dict[str, str] = {
    "calm_open": "Calm, warm, and a little tentative — opening a conversation gently.",
    "calm_guarded": "Calm but guarded, a hint of defensiveness under the surface.",
    "calm_neutral": "Even, matter-of-fact, conversational — no strong emotion.",
    "calm_deescalate": "Calm and steady, deliberately lowering the temperature of the room.",
    "calm_close": "Calm and warm, relieved, sincere.",
    "repair_apology": "Quiet and sincere, a little sheepish — a genuine apology.",
    "repair_hopeful": "Soft, hopeful, still raw but reaching out to repair things.",
    "tense_rising": "Tense and clipped, patience visibly running out.",
    "defensive_rising": "Defensive and sharp, volume and pace rising.",
    "shout_angry": "Shout this, furious and losing control.",
    "cold_contempt": "Deliver flat, cold, quietly contemptuous — no warmth at all.",
    "hurt_sad": "Hurt and vulnerable, on the edge of tears.",
    "sullen_sad": "Sullen and deflated, quiet, a little wounded.",
    "scared_shaky": "Frightened, voice shaking, close to panic.",
    "warm_happy": "Warm, upbeat, genuinely pleased.",
}


def T(speaker: str, emotion: str, text: str) -> dict:
    """One scripted turn. The acting instruction is looked up from the
    emotion label so every scene uses identical directions per label."""
    return dict(speaker=speaker, scripted_emotion=emotion,
                instruction=INSTRUCTIONS[emotion], text=text)


def N(after_turn_index: int, level: str, reason: str) -> dict:
    return dict(after_turn_index=after_turn_index, level=level, reason=reason)


# ---------------------------------------------------------------------------
# The three scenes. Speaker A is always self. `expected_nudges` indexes are
# 0-based into `turns` and MUST point at a self turn — the fixture schema
# test enforces that, plus "every self turn with emotion_coarse == angry has
# a nudge and no other turn does", so the hand-authored timeline can't
# silently drift from the script.
# ---------------------------------------------------------------------------
SCENES: dict[str, dict] = {
    "couple_escalation": dict(
        filename="test_recording_scene_couple_escalation.wav",
        summary=("2 voices, self + partner. Credit-card-statement argument: "
                 "calm -> tense -> self loses temper over three escalating "
                 "turns -> partner hurt -> self repairs -> warm close."),
        silence_gap_sec=0.4,
        speakers={
            "Speaker A": dict(voice=SELF_VOICE, is_self=True, role="self"),
            "Speaker B": dict(voice="coral", is_self=False, role="partner"),
        },
        turns=[
            T("Speaker A", "calm_open",
              "Hey, I looked at the credit card statement tonight. Can we go over it?"),
            T("Speaker B", "calm_guarded",
              "Sure. I know it's higher than usual, but most of that was the car repair."),
            T("Speaker A", "calm_neutral",
              "The car was six hundred. There's another nine hundred I can't place."),
            T("Speaker B", "calm_guarded",
              "Some was groceries, some was your mom's birthday gift. I didn't hide anything."),
            T("Speaker A", "tense_rising",
              "I'm not saying you hid it. I'm saying we agreed to talk before spending like that."),
            T("Speaker B", "defensive_rising",
              "We talked about it in the car. You just weren't listening, like always."),
            T("Speaker A", "shout_angry",
              "LIKE ALWAYS? I AM THE ONLY ONE IN THIS HOUSE WHO EVEN LOOKS AT THE BILLS!"),
            T("Speaker B", "hurt_sad",
              "Please don't yell at me. I'm right here, I can hear you."),
            T("Speaker A", "cold_contempt",
              "Forget it. Clearly budgeting is beneath you. I'll just handle it, like everything else."),
            T("Speaker B", "hurt_sad",
              "That's not fair. I'm trying, and you talk to me like I'm a child."),
            T("Speaker A", "repair_apology",
              "You're right. I'm sorry. That came out way harsher than I meant."),
            T("Speaker B", "calm_close",
              "Thank you. Can we just sit down Sunday and do it together, no phones?"),
            T("Speaker A", "warm_happy",
              "Yeah. Sunday. I'd like that. And I'll bring the good coffee."),
        ],
        expected_nudges=[
            N(4, "mild", "self tense_rising: first sign of self escalating — a light "
                         "'watch your tone' nudge, not an intervention"),
            N(6, "strong", "self shout_angry: shouted spike — strongest possible nudge"),
            N(8, "strong", "self cold_contempt: contempt is the most corrosive "
                           "delivery (Gottman's 'four horsemen'); still strong even "
                           "though the volume dropped"),
        ],
    ),
    "family3": dict(
        filename="test_recording_scene_family3.wav",
        summary=("3 voices, self (parent) + teen + other parent. Dinner-prep "
                 "squabble with rapid turn-taking: the TEEN escalates (non-self "
                 "shout, no nudge expected), self flares once "
                 "(defensive_rising -> one mild nudge), everyone resets."),
        # Shorter gap than the 2-voice scenes on purpose: this is the
        # rapid-turn-taking case for the realtime track.
        silence_gap_sec=0.25,
        speakers={
            "Speaker A": dict(voice=SELF_VOICE, is_self=True, role="self (parent)"),
            "Speaker B": dict(voice="ballad", is_self=False, role="teen"),
            "Speaker C": dict(voice="nova", is_self=False, role="other parent"),
        },
        turns=[
            T("Speaker C", "calm_open",
              "Okay, dinner's in ten minutes. Who's setting the table tonight?"),
            T("Speaker B", "calm_guarded",
              "Not me, I did it yesterday. And the day before, actually."),
            T("Speaker A", "calm_neutral",
              "Then it's my turn. Grab the plates for me, would you?"),
            T("Speaker C", "calm_neutral",
              "Also, someone left the milk out again. All afternoon."),
            T("Speaker B", "defensive_rising",
              "Why do you always assume it's me? Maybe ask Dad for once."),
            T("Speaker A", "calm_neutral",
              "It was probably me, honestly. I made coffee around two."),
            T("Speaker C", "tense_rising",
              "That's the third time this week. It's not funny anymore."),
            T("Speaker B", "shout_angry",
              "OH MY GOD, IT'S MILK! WHY IS EVERYTHING IN THIS HOUSE A CRISIS?"),
            T("Speaker C", "hurt_sad",
              "Don't shout at me. I'm tired and I'm just asking for a little help."),
            T("Speaker A", "defensive_rising",
              "Hey! Don't talk to your mother like that, and don't drag me into it either!"),
            T("Speaker B", "sullen_sad",
              "Fine. Sorry. I'm just sick of getting blamed for everything."),
            T("Speaker A", "repair_apology",
              "I know. I snapped too. Let's all just reset, okay?"),
            T("Speaker C", "calm_close",
              "Reset. Plates, forks, and somebody put the milk in the fridge."),
            T("Speaker B", "warm_happy",
              "On it. And for the record, I'm making the garlic bread."),
            T("Speaker A", "warm_happy",
              "Now we're talking. Best part of the week, right here."),
        ],
        expected_nudges=[
            N(9, "mild", "self defensive_rising: a brief flare in defence of the "
                         "other parent — one mild nudge. NOTE the teen's shout_angry "
                         "at turn 7 and the other parent's tense_rising at turn 6 are "
                         "NOT self turns: a nudge there is a false positive"),
        ],
    ),
    "meeting4": dict(
        filename="test_recording_scene_meeting4.wav",
        summary=("4 voices, self + 3 colleagues. Status meeting, mostly neutral "
                 "for ~45s, then a budget item: self gets tense, then shouts "
                 "(late strong nudge), a colleague de-escalates, self apologises."),
        silence_gap_sec=0.4,
        speakers={
            "Speaker A": dict(voice=SELF_VOICE, is_self=True, role="self"),
            "Speaker B": dict(voice="marin", is_self=False, role="chair"),
            "Speaker C": dict(voice="ballad", is_self=False, role="engineering lead"),
            "Speaker D": dict(voice="nova", is_self=False, role="marketing lead"),
        },
        turns=[
            T("Speaker B", "calm_open",
              "Okay, let's start. First item: the March launch timeline."),
            T("Speaker C", "calm_neutral",
              "Engineering's on track. Integration tests finish by the twentieth."),
            T("Speaker D", "calm_neutral",
              "Marketing needs final screenshots by the fifteenth, or the press kit slips."),
            T("Speaker A", "calm_neutral",
              "Tight, but doable. I'll have design prioritize the screenshots."),
            T("Speaker B", "calm_neutral",
              "Good. Second item: the support backlog is up thirty percent."),
            T("Speaker A", "calm_neutral",
              "I'll pull the top ten tickets and send them round this afternoon."),
            T("Speaker B", "calm_neutral",
              "Third item, budget. Finance wants every team to trim eight percent."),
            T("Speaker D", "tense_rising",
              "Eight percent? We already cut the contractor budget last quarter."),
            T("Speaker C", "calm_guarded",
              "We could delay the second hire until June. That covers most of it."),
            T("Speaker B", "calm_neutral",
              "That's one option. What about the conference sponsorship?"),
            T("Speaker D", "calm_guarded",
              "We signed that in December. Backing out costs almost as much as going."),
            T("Speaker A", "tense_rising",
              "So we delay my hire again? That's the third time this year."),
            T("Speaker B", "calm_neutral",
              "Nobody's decided anything. We're just listing options."),
            T("Speaker A", "shout_angry",
              "THEN LIST A DIFFERENT ONE! MY TEAM IS DROWNING AND EVERYONE HERE KNOWS IT!"),
            T("Speaker C", "calm_deescalate",
              "Okay. Let's take that seriously. Can we table this and come back with real numbers?"),
            T("Speaker A", "repair_apology",
              "Yeah. Sorry, everyone. That was out of line. Real numbers by Thursday."),
            T("Speaker B", "calm_close",
              "Thursday works. Thanks, everyone. That's all for today."),
        ],
        expected_nudges=[
            N(11, "mild", "self tense_rising: first crack after ~45s of neutral "
                          "meeting talk — mild. Speaker D's tense_rising at turn 7 "
                          "is NOT self: no nudge there"),
            N(13, "strong", "self shout_angry: the late shouted spike this scene "
                            "exists for — strong"),
        ],
    ),
}

# Every label used must have a coarse class + instruction; fail loudly at
# import time so a typo can't reach the API.
for _name, _scene in SCENES.items():
    for _t in _scene["turns"]:
        assert _t["scripted_emotion"] in EMOTION_COARSE, (_name, _t["scripted_emotion"])
        assert _t["speaker"] in _scene["speakers"], (_name, _t["speaker"])
    _selfs = [k for k, v in _scene["speakers"].items() if v["is_self"]]
    assert _selfs == [SELF_LABEL], (_name, _selfs)
    assert all(v["voice"] == SELF_VOICE for k, v in _scene["speakers"].items() if v["is_self"])

# Public list price of gpt-4o-mini-tts at time of writing (OpenAI quotes
# "~$0.015 per minute of audio"). Only used to PRINT an estimate — the API
# does not return spend, so this is a labelled estimate, never a measurement.
EST_USD_PER_AUDIO_MINUTE = 0.015


# ---------------------------------------------------------------------------
# ffmpeg resample: 24 kHz TTS WAV bytes -> 16 kHz mono int16 samples.
# ---------------------------------------------------------------------------

def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 — honest failure below
        print("ERROR: ffmpeg not found on PATH and imageio-ffmpeg not installed; "
              "cannot resample TTS output to 16 kHz.", file=sys.stderr)
        sys.exit(1)


# Edge-silence trim. The TTS engine pads each clip with 0-1.3 s of near-
# digital silence (measured on the first cut of these scenes), which (a)
# inflated every scene 20-50% past its target length and (b) put up to a
# second of dead air INSIDE a "turn", blurring the nudge timeline's turn
# boundaries. Trimming to the first/last sample above TRIM_THRESHOLD (about
# -41 dBFS — TTS padding sits near 0, speech onsets are far above it) and
# keeping TRIM_PAD_SEC on each side makes `silence_gap_sec` the ONLY silence
# between turns, so the reconstructed timeline is as tight as the meta says.
TRIM_THRESHOLD = 300        # int16 units (~-41 dBFS)
TRIM_PAD_SEC = 0.08


def trim_edge_silence(samples: np.ndarray, sr: int) -> np.ndarray:
    loud = np.nonzero(np.abs(samples.astype(np.int32)) > TRIM_THRESHOLD)[0]
    if loud.size == 0:          # a silent clip — leave it alone, never drop it
        return samples
    pad = int(round(TRIM_PAD_SEC * sr))
    start = max(0, int(loud[0]) - pad)
    end = min(len(samples), int(loud[-1]) + 1 + pad)
    return samples[start:end]


def wav_bytes_to_16k(ffmpeg: str, wav_bytes: bytes) -> np.ndarray:
    """Decode + properly resample arbitrary WAV bytes to mono 16 kHz int16
    (same target as audio_ingest's ffmpeg path: -ac 1 -ar 16000 s16le)."""
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(TARGET_SR),
         "-f", "s16le", "pipe:1"],
        input=wav_bytes, capture_output=True, check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        print(f"ERROR: ffmpeg resample failed: {proc.stderr.decode(errors='replace')[:400]}",
              file=sys.stderr)
        sys.exit(1)
    return np.frombuffer(proc.stdout, dtype=np.int16)


# ---------------------------------------------------------------------------
# Per-scene driver
# ---------------------------------------------------------------------------

def generate_scene(client: httpx.Client, api_key: str, ffmpeg: str, name: str,
                   out_dir: Path, force: bool) -> float:
    """Synthesize + concatenate one scene; returns seconds of audio billed."""
    scene = SCENES[name]
    out_path = out_dir / scene["filename"]
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    if out_path.exists() and not force:
        print(f"✓ [{name}] {out_path} already exists — skipping (use --force).")
        return 0.0

    models: tuple[str, str] = VARIANTS["tts"]["models"]
    resolved: dict = {"model": None}
    turns = scene["turns"]
    gap_sec = scene["silence_gap_sec"]
    print(f"→ [{name}] {len(turns)} turns, {len(scene['speakers'])} voices "
          f"(model {models[0]}, fallback {models[1]})")

    segments: list[np.ndarray] = []
    durations: list[float] = []
    for i, turn in enumerate(turns, start=1):
        voice = scene["speakers"][turn["speaker"]]["voice"]
        print(f"  [{i:2d}/{len(turns)}] {turn['speaker']:9s} {voice:7s} "
              f"{turn['scripted_emotion']:16s} ...", end="", flush=True)
        wav_bytes = synth_turn(client, api_key, "tts", models, resolved, i,
                               dict(turn, voice=voice))
        samples = trim_edge_silence(wav_bytes_to_16k(ffmpeg, wav_bytes), TARGET_SR)
        segments.append(samples)
        durations.append(len(samples) / TARGET_SR)
        print(f" {durations[-1]:5.2f}s")

    gap = np.zeros(int(round(gap_sec * TARGET_SR)), dtype=np.int16)
    pieces: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        if i > 0:
            pieces.append(gap)
        pieces.append(seg)
    audio = np.concatenate(pieces)
    write_wav(out_path, audio, TARGET_SR)

    # duration_sec is rounded to the sample (exact at 16 kHz to 1/16000 s)
    # so sum(duration_sec) + gap*(n-1) matches the file length to <0.01 s.
    meta = {
        "_note": (
            "Acted emotion, not physical ground truth: each turn is the model "
            "performing `instruction`; no expected pitch/energy labels on purpose "
            "(see scripts/make_test_recording_scenes.py). `expected_nudges` is "
            "HAND-AUTHORED from the script (which self turns a coach should react "
            "to, and how hard) — a spec for the product tracks, NOT a measurement "
            "of any model's output. `emotion_coarse` is the fixed "
            "scripted_emotion->{neutral,angry,sad,happy} table from the same script."
        ),
        "variant": "tts-scene",
        "scene": name,
        "summary": scene["summary"],
        "model_used": resolved["model"],
        "sample_rate": TARGET_SR,
        "silence_gap_sec": gap_sec,
        "num_speakers_true": len(scene["speakers"]),
        "self_speaker": SELF_LABEL,
        "speakers": scene["speakers"],
        "turns": [
            {
                "speaker": t["speaker"],
                "text": t["text"],
                "scripted_emotion": t["scripted_emotion"],
                "emotion_coarse": EMOTION_COARSE[t["scripted_emotion"]],
                "instruction": t["instruction"],
                "duration_sec": round(d, 4),
            }
            for t, d in zip(turns, durations)
        ],
        "expected_nudges": scene["expected_nudges"],
        "expected_nudges_note": (
            "after_turn_index is 0-based into `turns` and always names a SELF "
            "turn (Speaker A). level: mild = tense/defensive rising, strong = "
            "shout or contempt. Non-self escalations deliberately have NO entry: "
            "a nudge on them is a false positive."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    total = len(audio) / TARGET_SR
    print(f"\n[{name}] model responded: {resolved['model']}")
    print(f"{'#':>2}  {'Speaker':9s} {'Voice':7s} {'Emotion':16s} {'Coarse':7s} {'Dur':>6s}")
    for i, (t, d) in enumerate(zip(turns, durations)):
        print(f"{i:>2}  {t['speaker']:9s} {scene['speakers'][t['speaker']]['voice']:7s} "
              f"{t['scripted_emotion']:16s} {EMOTION_COARSE[t['scripted_emotion']]:7s} "
              f"{d:5.2f}s")
    print(f"\nTotal: {len(turns)} turns, {total:.2f}s (incl. {gap_sec}s gaps) "
          f"@ {TARGET_SR} Hz, {out_path.stat().st_size / 1e6:.2f} MB")
    print(f"→ {out_path}\n→ {meta_path}\n")
    return float(sum(durations))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene", choices=[*SCENES, "all"], default="all")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "tmp",
                   help="Directory for the WAV + _meta.json (default tmp/; the "
                        "committed pack lives in server/tests/fixtures/audio/)")
    p.add_argument("--force", action="store_true",
                   help="Regenerate even if the output WAV already exists")
    args = p.parse_args()

    names = list(SCENES) if args.scene == "all" else [args.scene]
    if not args.force and all((args.out_dir / SCENES[n]["filename"]).exists() for n in names):
        for n in names:
            print(f"✓ [{n}] {args.out_dir / SCENES[n]['filename']} already exists — "
                  f"skipping (use --force).")
        return 0

    api_key = os.environ.get("OPENAI_API_KEY") or read_env("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set (add to repo-root .env)", file=sys.stderr)
        return 1
    ffmpeg = find_ffmpeg()

    billed_sec = 0.0
    with httpx.Client() as client:
        for n in names:
            billed_sec += generate_scene(client, api_key, ffmpeg, n, args.out_dir, args.force)
    if billed_sec:
        print(f"Synthesized {billed_sec / 60:.2f} min of audio — estimated spend "
              f"~${billed_sec / 60 * EST_USD_PER_AUDIO_MINUTE:.3f} at "
              f"${EST_USD_PER_AUDIO_MINUTE}/min list price (estimate, not billed figure).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
