"""Calibrate UNCLAIMED_COSINE on maggiano3 (private; nothing here is committed
beyond the script — see .gitignore).

For each Deepgram transcript variant, reproduce the diarizer's ROUND-1
instrument exactly as the per-word pass sees it: the whole-clip window pass,
the round-1 k-selection (its chosen pooled centroids = what
``split_long_utterances`` scores words against) and the window pass's pooled
SPECTRAL centroids at the eigengap k. Then, for every word of every utterance
long enough to be scanned, embed the same WORD_WINDOW_SECONDS window the
per-word pass embeds and record its best cosine to (a) the round-1 centroids,
(b) the spectral centroids, (c) both. Each word is tagged by the rubric
(truth at its midpoint) and by whether PRODUCTION's final label for that
instant was right (the diarizer's own output, mapped one-to-one onto the
rubric with the shared scorer). The distribution of (c) for right vs wrong
words is what UNCLAIMED_COSINE is chosen from.

Also prints, for the FINAL output turns, each embeddable turn's best cosine to
the final pooled centroids (the whole-turn rule's quantity).
"""
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(HERE.parent / "2026-08-29-voice-separation"))
logging.basicConfig(level=logging.WARNING)
import audio_ingest  # noqa: E402
import diarize_local as dl  # noqa: E402
import score  # noqa: E402
import speaker_id  # noqa: E402

fx = score.load_fixture("maggiano3")
pcm, sr = audio_ingest.decode_to_pcm_16k(open(fx["audio_path"], "rb").read(), "audio.m4a")
gt = fx["gt"]


def truth_at(t: float):
    for s, e, lab in gt:
        if s <= t < e:
            return set(lab) if isinstance(lab, tuple) else {lab}
    return None


embed = dl._default_embed
rows = []
for tpath in fx["transcripts"]:
    variant = Path(tpath).stem
    turns = json.loads(Path(tpath).read_text())
    # --- production's final answer for this variant (mapped onto the rubric)
    out = dl.diarize_turns(pcm, sr, [dict(t) for t in turns])
    pred = [(t["start_time"], t["end_time"], t["speaker"]) for t in out["turns"]]
    sc = score.score_fixture("maggiano3", pred)
    mapping = sc["mapping"]

    def pred_at(t: float):
        for s, e, lab in pred:
            if s <= t < e:
                return mapping.get(lab)
        return None

    # --- round-1 instrument, exactly as diarize_turns builds it
    embedded = dl._embed_turns(pcm, sr, turns, embed, dl.MIN_SECONDS)
    order, embs = embedded
    wp = dl._WindowPass(pcm, sr, speaker_id.embed_pcm_batch, embed)
    wp.run_global()
    pass1 = dl._refine_k(pcm, sr, turns, embed, order, embs, 2)
    k_eval, chosen = dl._select_k(
        pcm, sr, turns, embed, order, embs, dl.MAX_POOLED_COSINE, pass2=pass1, window_pass=wp,
    )
    lin = [c for _, c in sorted(chosen[1].items())] if chosen else [pass1[1][0], pass1[1][1]]
    spec_d = wp.pooled_centroids(wp.k_eigengap) if wp.k_eigengap else None
    spec = [spec_d[c] for c in sorted(spec_d)] if spec_d else []
    print(f"=== {variant}: round-1 k={len(lin)} (route {chosen[2].get('route') if chosen else None}), "
          f"eigengap k={wp.k_eigengap}, production acc={sc['frame_accuracy']} purity={sc['owner_purity']}")
    for t in turns:
        start, end = float(t["start_time"]), float(t["end_time"])
        words = t.get("words") or []
        if end - start < dl.SCAN_MIN_UTTERANCE_SECONDS or len(words) < 2:
            continue
        for w in words:
            mid = (float(w["start_time"]) + float(w["end_time"])) / 2
            lo = max(start, mid - dl.WORD_WINDOW_SECONDS / 2)
            hi = min(end, mid + dl.WORD_WINDOW_SECONDS / 2)
            e = speaker_id.l2_normalize(embed(np.ascontiguousarray(pcm[int(lo * sr):int(hi * sr)]), sr))
            cl = [float(np.dot(e, c)) for c in lin]
            cs = [float(np.dot(e, c)) for c in spec]
            best_lin = max(cl)
            best_spec = max(cs) if cs else float("nan")
            best_all = max(cl + cs)
            srt = sorted(cl)
            margin = srt[-1] - srt[-2]
            tr = truth_at(mid)
            pr = pred_at(mid)
            ok = None if tr is None else (pr in tr)
            rows.append(dict(variant=variant, word=w["word"], mid=round(mid, 2),
                             truth="/".join(sorted(tr)) if tr else "-", pred=pr, ok=ok,
                             best_lin=round(best_lin, 3), best_spec=round(best_spec, 3),
                             best_all=round(best_all, 3), margin=round(margin, 3)))
    # --- whole-turn quantity on the FINAL output: pooled centroid per final label
    labels = sorted({t["speaker"] for t in out["turns"]})
    cents = {}
    for lab in labels:
        idxs = [i for i, t in enumerate(out["turns"]) if t["speaker"] == lab]
        cents[lab] = speaker_id.l2_normalize(embed(dl._pooled(pcm, sr, out["turns"], idxs), sr))
    print("  final turns (best cosine to any final pooled centroid; * = wrong per rubric):")
    for t in out["turns"]:
        d = t["end_time"] - t["start_time"]
        if d < dl.MIN_SECONDS:
            continue
        e = speaker_id.l2_normalize(embed(dl._slice(pcm, sr, t), sr))
        best = max(float(np.dot(e, c)) for c in cents.values())
        mid = (t["start_time"] + t["end_time"]) / 2
        tr = truth_at(mid)
        flag = "*" if tr and mapping.get(t["speaker"]) not in tr else " "
        print(f"   {flag} {t['speaker']:9s}->{mapping.get(t['speaker'])!s:6s} {t['start_time']:5.2f}-{t['end_time']:5.2f} "
              f"best={best:.3f} truth={'/'.join(sorted(tr)) if tr else '-'} {t['text'][:40]!r}")

print()
print("per-word best cosine to ANY round-1/spectral centroid, by production correctness")
for variant in sorted({r["variant"] for r in rows}):
    for ok in (True, False):
        v = np.array([r["best_all"] for r in rows if r["variant"] == variant and r["ok"] is ok])
        if v.size:
            q = np.percentile(v, [0, 5, 10, 25, 50, 75, 90])
            print(f"  {variant} {'RIGHT' if ok else 'WRONG'} n={v.size:3d} "
                  f"p0={q[0]:.3f} p5={q[1]:.3f} p10={q[2]:.3f} p25={q[3]:.3f} p50={q[4]:.3f} p75={q[5]:.3f} p90={q[6]:.3f}")
for floor in (0.08, 0.10, 0.12, 0.15, 0.18):
    below = [r for r in rows if r["best_all"] < floor]
    wrong = sum(1 for r in below if r["ok"] is False)
    print(f"  floor {floor:.2f}: {len(below)} words below, {wrong} of them wrong in production")
print()
print("words below 0.18 (variant mid word truth pred ok best_lin best_spec best_all margin):")
for r in sorted(rows, key=lambda r: r["best_all"]):
    if r["best_all"] < 0.18:
        print("  ", r["variant"][-4:], r["mid"], repr(r["word"]), r["truth"], r["pred"], r["ok"],
              r["best_lin"], r["best_spec"], r["best_all"], r["margin"])
json.dump(rows, open(HERE / "out" / "calibration_maggiano3_words.json", "w"), indent=0)
