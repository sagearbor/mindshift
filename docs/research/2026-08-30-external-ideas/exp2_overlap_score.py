"""Experiment 2, step 2 — mask overlapped windows out of B's grid (and out of
production's window pass), re-cluster, re-score. Runs under tmp/venv-voice
after exp2_overlap_pyannote.py has written cache/overlap_<fixture>.npz.

A 1.5 s window is MASKED when >= MASK_FRAC of its 10 ms frames have
p_overlap > P_OV (spec: 0.5 / 30 %; 10 % and 50 % swept for sensitivity).
B: masked windows are dropped before spectral clustering (eigengap k,
p=0.80, mode filter, run absorption — the bake-off's headline variant); the
masked span then inherits the nearest kept window's label, as any VAD gap
does. Production: speaker_id.speech_mask is wrapped (monkeypatched in this
process only — server/ is untouched) so the diarizer's window pass treats
overlapped 30 ms frames as non-speech; the transcript-utterance path is
otherwise unchanged (baseline/run.py's inputs: GT boundaries + the two real
Deepgram transcripts for maggiano3).

Also reports how many seconds pyannote flagged and how they line up with the
rubric's overlap segments (27-28 s, 29.8-30.4 s).
"""
from __future__ import annotations

import json
import sys

import numpy as np

import common as C
from common import score, speaker_id

P_OV = 0.5
MASK_FRACS = [0.1, 0.3, 0.5]
FIXTURES = ["maggiano3", "scene_family3", "scene_meeting4", "family_real", "poker6", "scene_couple"]


def overlap_of(name: str):
    z = np.load(C.CACHE / f"overlap_{name}.npz")
    return z["t"], z["p_overlap"], z["p_speech"]


def runs_of(mask: np.ndarray, t: np.ndarray) -> list[list[float]]:
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        if not m and s is not None:
            out.append([round(float(t[s]), 2), round(float(t[i]), 2)]); s = None
    if s is not None:
        out.append([round(float(t[s]), 2), round(float(t[-1]), 2)])
    return out


def window_overlap_frac(starts: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    hot = p > P_OV
    out = np.zeros(len(starts))
    for i, s in enumerate(starts):
        a, b = int(round(s / 0.01)), int(round((s + C.WINDOW) / 0.01))
        seg = hot[a:b]
        out[i] = seg.mean() if seg.size else 0.0
    return out


def gt_overlap_alignment(name: str, t: np.ndarray, p: np.ndarray) -> dict:
    fx = score.load_fixture(name)
    ov = [(s, e) for s, e, l in fx["gt"] if not isinstance(l, str)]
    hot = p > P_OV
    inside = np.zeros(len(t), bool)
    for s, e in ov:
        inside[(t >= s) & (t < e)] = True
    return {"rubric_overlap_segments": ov, "rubric_overlap_s": round(sum(e - s for s, e in ov), 2),
            "flagged_s": round(float(hot.sum() * 0.01), 2),
            "flagged_inside_rubric_overlap_s": round(float((hot & inside).sum() * 0.01), 2),
            "flagged_outside_s": round(float((hot & ~inside).sum() * 0.01), 2),
            "rubric_overlap_recall": round(float((hot & inside).sum() / max(1, inside.sum())), 3) if ov else None,
            "flagged_runs": runs_of(hot, t)}


def main(names: list[str]) -> None:
    C.torch_threads()
    speaker_id._load_model()
    import diarize_local  # noqa: F401  (production diarizer)
    res = {"p_overlap_threshold": P_OV, "mask_fracs": MASK_FRACS, "fixtures": {}}
    for name in names:
        pcm, sr = C.load_audio(name)
        dur = pcm.size / sr
        t, p_ov, p_sp = overlap_of(name)
        we = C.b_windows(name, pcm, sr)
        starts, embs = we["starts"], we["embs"]
        frac = window_overlap_frac(starts, t, p_ov)
        entry = {"k_true": score.load_fixture(name)["k_true"], "alignment": gt_overlap_alignment(name, t, p_ov),
                 "n_windows": int(len(starts)), "B": {}, "production": {}}
        # ---- B -------------------------------------------------------------
        segs, k_hat = C.b_cluster(starts, embs, dur)
        entry["B"]["unmasked"] = {"masked_windows": 0, "k_eigengap": k_hat, **C.score_segments(name, segs)}
        for mf in MASK_FRACS:
            keep = frac < mf
            segs, k_hat = C.b_cluster(starts, embs, dur, keep=keep)
            entry["B"][f"mask_frac>={mf}"] = {"masked_windows": int((~keep).sum()), "k_eigengap": k_hat,
                                              **C.score_segments(name, segs)}
        # ---- production (window pass gated by the overlap mask) -------------
        orig = speaker_id.speech_mask
        entry["production"]["unmasked"] = C.run_production(name, pcm, sr)
        hot = p_ov > P_OV

        def masked_speech_mask(pcm_, sr_, **kw):
            mask, thr, frame_s = orig(pcm_, sr_, **kw)
            m = mask.copy()
            for i in range(len(m)):
                a, b = int(round(i * frame_s / 0.01)), int(round((i + 1) * frame_s / 0.01))
                seg = hot[a:b]
                if seg.size and seg.mean() >= 0.5:
                    m[i] = False
            return m, thr, frame_s
        speaker_id.speech_mask = masked_speech_mask
        try:
            entry["production"]["masked"] = C.run_production(name, pcm, sr)
        finally:
            speaker_id.speech_mask = orig
        res["fixtures"][name] = entry
        (C.CACHE / f"exp2_{name}.json").write_text(json.dumps(entry, indent=1))   # partial, per fixture
        a = entry["alignment"]
        print(f"{name:14s} flagged {a['flagged_s']}s (inside rubric overlap {a['flagged_inside_rubric_overlap_s']}s, "
              f"recall {a['rubric_overlap_recall']}) runs {a['flagged_runs']}")
        for v, r in entry["B"].items():
            print(f"   B {v:16s} masked {r['masked_windows']:3d}/{len(starts)} k={r['k_pred']} acc={r['frame_accuracy']} purity={r['owner_purity']}")
        for v, r in entry["production"].items():
            for vv, rr in r.items():
                print(f"   prod {v:9s} {vv:18s} k={rr['k_pred']} acc={rr['frame_accuracy']} purity={rr['owner_purity']}")
        sys.stdout.flush()
    for f in C.CACHE.glob("exp2_*.json"):                 # merge every fixture run so far
        res["fixtures"].setdefault(f.stem[5:], json.loads(f.read_text()))
    C.merge_results("exp2", res)
    print("wrote results.json[exp2]")


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")] or FIXTURES)
