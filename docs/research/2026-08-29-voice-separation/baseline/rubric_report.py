"""Per-rubric-segment view of what production predicted on maggiano3.

For each segment of the owner's rubric: the predicted speaker(s) covering it
(under the best mapping), the fraction of the segment that is correct, and
the transcript the prediction came from."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import score
fx = score.load_fixture("maggiano3")
for variant in ("transcript_7utt", "transcript_8utt", "gt_boundaries"):
    pred = [tuple(x) for x in json.loads((HERE / f"pred_maggiano3_{variant}.json").read_text())]
    r = score.score_fixture("maggiano3", pred)
    m = r["mapping"]
    print(f"=== {variant}: k={r['k_pred']}/{r['k_true']} acc={r['frame_accuracy']} owner_purity={r['owner_purity']} recall={r['per_gt_recall']} map={m}")
    for s, e, lab in fx["gt"]:
        allowed = set(lab) if isinstance(lab, tuple) else {lab}
        n = int((e - s) / 0.01); ok = 0; got = {}
        for i in range(n):
            t = s + i * 0.01
            pl = next((l for ps, pe, l in pred if ps <= t < pe), None)
            mapped = m.get(pl) if pl else None
            got[mapped or "—"] = got.get(mapped or "—", 0) + 1
            ok += mapped in allowed
        who = "/".join(sorted(allowed))
        flag = "" if ok / n >= 0.8 else "  <-- WRONG"
        print(f"  {s:5.1f}-{e:5.1f} {who:9s} got {', '.join(f'{k} {v/n:.0%}' for k, v in sorted(got.items(), key=lambda kv: -kv[1]))}{flag}")
