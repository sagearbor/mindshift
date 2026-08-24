"""Round-3 full pipeline runner: batched sliding-window embed pass +
centroid-margin change detection -> diarize_local.diarize_turns -> score
against each fixture's real ground truth. Real ECAPA calls (isolated
MINDSHIFT_ECAPA_CACHE). Writes a JSON result + prints a summary line so
repeated invocations are cheap to inspect without re-running.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/sophie.arborbot/PROJECTS/github_repos/mindshift/.claude/worktrees/poker6-v3-refine/server")

import numpy as np  # noqa: E402
import audio_ingest  # noqa: E402
import diarize_local  # noqa: E402
import diarize_sliding_window as dsw  # noqa: E402
import speaker_id  # noqa: E402

sys.path.insert(0, "/Users/sophie.arborbot/PROJECTS/github_repos/mindshift/.claude/worktrees/poker6-v3-refine/tmp/exp")
import score as scoring  # noqa: E402

AUDIO_DIR = Path("/Users/sophie.arborbot/PROJECTS/github_repos/mindshift/.claude/worktrees/poker6-v3-refine/server/tests/fixtures/audio")
OUT_DIR = Path("/Users/sophie.arborbot/PROJECTS/github_repos/mindshift/.claude/worktrees/poker6-v3-refine/tmp/exp")

FIXTURES = {
    "family": AUDIO_DIR / "test_recording_family_real.wav",
    "poker6": AUDIO_DIR / "test_recording_poker6_real.wav",
    "openai": AUDIO_DIR / "test_recording_openai.wav",
    "gptaudio": AUDIO_DIR / "test_recording_gptaudio.wav",
}


def build_ref_turns(fixture: str, meta: dict) -> tuple[list[dict], str]:
    if "approx_turns" in meta:
        return meta["approx_turns"], "speaker"
    if "turns" in meta and "start_time" in meta["turns"][0]:
        return meta["turns"], "speaker"
    if "turns" in meta and "duration_sec" in meta["turns"][0]:
        # openai/gptaudio: reconstruct start/end from duration_sec + silence_gap_sec
        # (same method as test_diarize_regression_ladder.py's _build_turns).
        gap = meta["silence_gap_sec"]
        turns = []
        t = 0.0
        for m in meta["turns"]:
            dur = m["duration_sec"]
            turns.append({
                "speaker": m["speaker"], "start_time": round(t, 4),
                "end_time": round(t + dur, 4),
            })
            t += dur + gap
        return turns, "speaker"
    raise ValueError(f"{fixture}: don't know how to build ref turns from meta keys {list(meta.keys())}")


def main():
    fixture = sys.argv[1]
    wav_path = FIXTURES[fixture]
    meta_path = wav_path.with_name(wav_path.stem + "_meta.json")
    meta = json.loads(meta_path.read_text())
    ref_turns, ref_key = build_ref_turns(fixture, meta)

    pcm, sr = audio_ingest.decode_to_pcm_16k(wav_path.read_bytes(), wav_path.name)
    duration = pcm.size / sr
    print(f"{fixture}: duration={duration:.2f}s sr={sr}")

    # --- Stage 1: batched sliding-window embed pass + centroid-margin detect
    t0 = time.time()
    sw_result = dsw.detect_turns_from_audio(pcm, sr, threshold=0.65)
    sw_elapsed = time.time() - t0
    turns = sw_result["turns"]
    print(f"{fixture}: sliding-window stage: {sw_elapsed:.1f}s -> {len(turns)} candidate turns")
    print(f"  boundaries: {[round(b,2) for b in sw_result['boundaries']]}")

    # --- Stage 2: diarize_local.diarize_turns (REAL, unbatched embed_fn --
    # existing production code path, untouched).
    t0 = time.time()
    result = diarize_local.diarize_turns(pcm, sr, turns)
    dt_elapsed = time.time() - t0
    print(f"{fixture}: diarize_turns stage: {dt_elapsed:.1f}s")

    out = {
        "fixture": fixture,
        "duration": duration,
        "sliding_window_elapsed_s": sw_elapsed,
        "diarize_turns_elapsed_s": dt_elapsed,
        "total_elapsed_s": sw_elapsed + dt_elapsed,
        "n_candidate_turns": len(turns),
        "boundaries": sw_result["boundaries"],
    }

    if result is None:
        print(f"{fixture}: diarize_turns returned None (nothing trustworthy)")
        out["result"] = None
    else:
        print(f"{fixture}: num_speakers={result['num_speakers']} "
              f"pooled_cosine={result['pooled_cosine']} "
              f"segments_total={result['segments_total']}")
        scored = scoring.score_against_reference(result["turns"], ref_turns, ref_key)
        print(f"{fixture}: per_turn_acc={scored['per_turn_accuracy']:.3f} "
              f"({scored['per_turn_correct']}/{scored['per_turn_total']}) "
              f"dur_weighted_acc={scored['duration_weighted_accuracy']:.3f}")
        for t, tr, pr in zip(result["turns"], scored["truth"], scored["pred"]):
            ok = scored["mapping"].get(pr) == tr
            print(f"    [{t['start_time']:6.2f}-{t['end_time']:6.2f}] "
                  f"truth={tr:<12} pred={pr:<12} ok={ok}")
        out["result"] = result
        out["scored"] = scored

    out_path = OUT_DIR / f"v3_result_{fixture}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
