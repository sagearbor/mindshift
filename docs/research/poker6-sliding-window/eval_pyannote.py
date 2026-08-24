"""Evaluate pyannote.audio (speaker-diarization-3.1) against the poker6
fixture and the 3 safety-check fixtures named in
docs/handoff/2026-08-24-mac-transition-and-poker6-status.md, scored the
same way as the round-3 sliding-window experiment (see score.py) so the
numbers are directly comparable to the 71% baseline in README.md.

Run from repo root with the venv-voice interpreter (has torch +
pyannote.audio installed):
    tmp/venv-voice/bin/python docs/research/poker6-sliding-window/eval_pyannote.py
"""
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import score_against_reference  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

FIXTURES_DIR = REPO_ROOT / "server" / "tests" / "fixtures" / "audio"

FIXTURES = [
    ("poker6", "test_recording_poker6_real.wav", "test_recording_poker6_real_meta.json", "approx_turns"),
    ("family_real", "test_recording_family_real.wav", "test_recording_family_real_meta.json", None),
    ("openai", "test_recording_openai.wav", "test_recording_openai_meta.json", None),
    ("gptaudio", "test_recording_gptaudio.wav", "test_recording_gptaudio_meta.json", None),
]


def load_ref_turns(meta_path, turns_key):
    meta = json.loads(meta_path.read_text())
    turns = None
    if turns_key and turns_key in meta:
        turns = meta[turns_key]
    else:
        for key in ("approx_turns", "turns", "expected_turns"):
            if key in meta:
                turns = meta[key]
                break
    if turns is None:
        raise KeyError(f"no turns key found in {meta_path}; keys={list(meta.keys())}")

    # Scripted-TTS fixtures (openai/gptaudio) give duration_sec per turn +
    # a fixed silence_gap_sec instead of absolute start/end — reconstruct
    # cumulative timing so the shared scorer's overlap math works.
    if turns and "start_time" not in turns[0] and "duration_sec" in turns[0]:
        gap = meta.get("silence_gap_sec", 0.0)
        t = 0.0
        rebuilt = []
        for turn in turns:
            start, end = t, t + turn["duration_sec"]
            rebuilt.append({**turn, "start_time": start, "end_time": end})
            t = end + gap
        turns = rebuilt

    num_speakers_true = meta.get("num_speakers_true")
    if num_speakers_true is None:
        num_speakers_true = len({t["speaker"] for t in turns})
    return turns, num_speakers_true


def run_pyannote(pipeline, wav_path, num_speakers=None):
    t0 = time.time()
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    diarization = pipeline(str(wav_path), **kwargs)
    elapsed = time.time() - t0
    pred_turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        pred_turns.append({
            "start_time": turn.start,
            "end_time": turn.end,
            "speaker": speaker,
        })
    num_speakers = len({t["speaker"] for t in pred_turns})
    return pred_turns, num_speakers, elapsed


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not set in .env", file=sys.stderr)
        sys.exit(1)

    # pyannote.audio 3.3.2's lightning-based checkpoint loader predates
    # torch 2.6's weights_only=True default; these are official pyannote
    # checkpoints from HF (trusted source), so restore the old default.
    import torch
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load

    from pyannote.audio import Pipeline
    print("Loading pyannote/speaker-diarization-3.1 pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )

    results = {}
    for name, wav_name, meta_name, turns_key in FIXTURES:
        wav_path = FIXTURES_DIR / wav_name
        meta_path = FIXTURES_DIR / meta_name
        if not wav_path.exists() or not meta_path.exists():
            print(f"SKIP {name}: fixture missing ({wav_path} / {meta_path})")
            continue

        ref_turns, num_speakers_true = load_ref_turns(meta_path, turns_key)

        print(f"\n=== {name} (auto speaker count) ===")
        pred_turns, num_speakers_pred, elapsed = run_pyannote(pipeline, wav_path)
        score = score_against_reference(pred_turns, ref_turns, speaker_key="speaker")
        auto_result = {
            "num_speakers_true": num_speakers_true,
            "num_speakers_pred": num_speakers_pred,
            "elapsed_sec": round(elapsed, 1),
            "per_turn_accuracy": round(score["per_turn_accuracy"], 3),
            "per_turn_correct": score["per_turn_correct"],
            "per_turn_total": score["per_turn_total"],
            "duration_weighted_accuracy": round(score["duration_weighted_accuracy"], 3),
        }
        print(json.dumps(auto_result, indent=2))

        print(f"=== {name} (num_speakers={num_speakers_true} hint) ===")
        pred_turns_h, num_speakers_pred_h, elapsed_h = run_pyannote(
            pipeline, wav_path, num_speakers=num_speakers_true
        )
        score_h = score_against_reference(pred_turns_h, ref_turns, speaker_key="speaker")
        hinted_result = {
            "num_speakers_true": num_speakers_true,
            "num_speakers_pred": num_speakers_pred_h,
            "elapsed_sec": round(elapsed_h, 1),
            "per_turn_accuracy": round(score_h["per_turn_accuracy"], 3),
            "per_turn_correct": score_h["per_turn_correct"],
            "per_turn_total": score_h["per_turn_total"],
            "duration_weighted_accuracy": round(score_h["duration_weighted_accuracy"], 3),
        }
        print(json.dumps(hinted_result, indent=2))

        result = {"auto": auto_result, "num_speakers_hint": hinted_result}
        results[name] = result

        out_path = Path(__file__).resolve().parent / f"pyannote_result_{name}.json"
        out_path.write_text(json.dumps({
            **result,
            "pred_turns_auto": pred_turns,
            "pred_turns_hinted": pred_turns_h,
        }, indent=2))

    print("\n=== SUMMARY ===")
    print(json.dumps(results, indent=2))
    summary_path = Path(__file__).resolve().parent / "pyannote_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
