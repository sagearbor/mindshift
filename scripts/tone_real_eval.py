"""Score server/tone_id.py's audio tone on the SAME five recordings every other
signal was judged on — the first time it has been measured outside its own
corpus.

Round 2 measured it on acted corpora and left it dark with a stated flip rule:
the audio signal must clear ~60% AND add lift over text tone. This asks the
only question that matters for shipping: on the owner's real recording and the
scene fixtures, does the escalation flag land on the turns a human called
heated, and does it stay off everything else?

Ground truth is the replay report's per-turn `expected` (mild/strong = heated,
null = not) — the same labels the nudge gate uses, so no new hand-labelling.

Usage: python scripts/tone_real_eval.py [backend]
Reads the newest tmp/nudge-report-*.json for ground truth (regenerate it with
`npx jest __tests__/replay.nudgeReport.test.ts` in apps/mobile) and writes
tmp/tone-real/results.json. Needs requirements-voice.txt.
"""
import json, os, sys, wave
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "server"))
os.environ.setdefault("MINDSHIFT_TONE_AUDIO", "dark")
if len(sys.argv) > 1:
    os.environ["MINDSHIFT_TONE_BACKEND"] = sys.argv[1]

import tone_id  # noqa: E402

AUDIO = os.path.join(REPO, "server/tests/fixtures/audio")
def _newest_report():
    """The most recent replay report — ground truth for which turns are heated.
    Named by date, so the newest file IS the current one (no LATEST pointer)."""
    import glob
    hits = sorted(glob.glob(os.path.join(REPO, "tmp/nudge-report-*.json")))
    if not hits:
        raise SystemExit("no tmp/nudge-report-*.json — run the replay report test first")
    return hits[-1]


REPORT = _newest_report()
# The one scene whose audio is not committed (CC BY-NC-SA); skipped when absent.
RAVDESS_WAV = os.path.join(REPO, "tmp/ravdess-scene/scene_ravdess_pair.wav")


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, path
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    return pcm, sr


def wav_for(scene):
    p = os.path.join(AUDIO, f"test_recording_{scene}.wav")
    if os.path.exists(p):
        return p
    return RAVDESS_WAV if os.path.exists(RAVDESS_WAV) else None


def main():
    report = json.load(open(REPORT))
    rows, missing = [], []
    for s in report["scenes"]:
        path = wav_for(s["scene"])
        if path is None:
            missing.append(s["scene"])
            continue
        pcm, sr = read_wav(path)
        assert sr == tone_id.TARGET_SR, f"{path}: {sr} Hz"
        tracker = tone_id.EscalationTracker()
        for t in s["turns"]:
            a, b = int(t["start"] * sr), int(t["end"] * sr)
            slice_ = pcm[a:b]
            secs = len(slice_) / sr
            if secs < tone_id.MIN_TURN_SECONDS:
                # Honest: too short to classify. Counted as "no flag", which is
                # what the live pipeline does with it too.
                rows.append({"scene": s["scene"], "turn": t["index"], "speaker": t["speaker"],
                             "isSelf": t["isSelf"], "heated": t["expected"] is not None,
                             "expected": t["expected"], "flag": False, "delta": None,
                             "arousal": None, "skipped": "too short", "seconds": round(secs, 2)})
                continue
            res = tone_id.classify_pcm(slice_, sr)
            tone_id.annotate_escalation(res, t["speaker"], tracker)
            esc = res["escalation"]
            rows.append({"scene": s["scene"], "turn": t["index"], "speaker": t["speaker"],
                         "isSelf": t["isSelf"], "heated": t["expected"] is not None,
                         "expected": t["expected"], "flag": bool(esc["flag"]),
                         "delta": esc["delta"], "arousal": round(float(res["arousal"]), 4),
                         "baseline": esc["baseline"], "history": esc["history"],
                         "skipped": None, "seconds": round(secs, 2)})
            print(f'  {s["scene"]:24} t{t["index"]:<2} {t["speaker"]:<10} '
                  f'heated={str(t["expected"]):6} arousal={res["arousal"]:.3f} '
                  f'delta={esc["delta"]} flag={esc["flag"]}', flush=True)

    scored = [r for r in rows if r["skipped"] is None]
    heated = [r for r in scored if r["heated"]]
    calm = [r for r in scored if not r["heated"]]
    # A turn with no baseline yet cannot flag; report it separately so the
    # recall number is not quietly punished for the causal rule.
    scorable_heated = [r for r in heated if r["delta"] is not None]
    out = {
        "backend": tone_id.backend(),
        "threshold": tone_id.escalation_threshold(),
        "missing_audio": missing,
        "turns_total": len(rows),
        "turns_scored": len(scored),
        "heated_total": len(heated),
        "heated_with_baseline": len(scorable_heated),
        "hits": sum(1 for r in scorable_heated if r["flag"]),
        "false_flags": sum(1 for r in calm if r["flag"]),
        "calm_total": len(calm),
        "rows": rows,
    }
    out["recall"] = round(out["hits"] / out["heated_with_baseline"], 3) if out["heated_with_baseline"] else None
    out["false_flag_rate"] = round(out["false_flags"] / out["calm_total"], 3) if out["calm_total"] else None
    out_dir = os.path.join(REPO, "tmp/tone-real")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
