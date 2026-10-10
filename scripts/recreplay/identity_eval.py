#!/usr/bin/env python3
"""identity_eval.py — phone-only ($0) wearer-identity + alert-buzz sweep
over the overnight-tuning splits.

  python scripts/recreplay/identity_eval.py --tag base --split dev
  python scripts/recreplay/identity_eval.py --tag thr55 --split dev \
      --speaker-opts '{"matchThreshold":0.55}'

Re-runs ONLY the phone side (apps/mobile/src/live/replay/recordingReplay.ts:
Silero VAD + segmenter + ECAPA labeler + instant tier) over each item's
cached ``work/<name>/audio16k.wav`` + ``phone_meta.json`` — no Deepgram, no
LLM, no server — then scores the result with the SAME scorer the replay
report uses (``score.identity`` on the item's frozen ``run.json`` bundle with
its ``phone`` swapped for the new run).

Identity numbers are reported on the items whose wearer print was enrolled
from a held-out part of the session (``<name>.voiceprint.json``) — the
honest setting. Items without one replay with ``--enroll same`` (a print
pooled from the first seconds of this very recording: optimistic) and are
reported apart, never pooled into the identity numbers.

Metrics (all reported together — never accuracy alone):
  wearer_recall     wearer turns the phone called self / wearer turns
  false_self_rate   other-speaker turns the phone called self / other-speaker turns
  coverage          turns with a decided is_self / all turns
  time_to_confirm_s median over items of the first correct "that's you"
                    (an item that never confirms counts as its duration)
  confirm_delay_s   the same, minus when the wearer's first turn was sent
                    (time_to_confirm is bounded below by when the wearer
                    first speaks: 64 s into ami_TS3004a)
  never_confirmed   items with no correct "that's you" at all
  alert_buzz_hits   wearer raised/shouted moments (annotation coach_moments,
                    rule raised-voice) with an ALERT haptic in [-1.5, +6] s
  alert_buzz_false_h  alert haptics per hour in calm-group items
  alert_buzz_nomoment_h alert haptics per hour in heated items with no
                    raised-voice wearer moment within the window
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.recreplay import score as score_mod  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MAIN = Path("/Users/sagearbor/projects/githubs/mindshift")
RECORDINGS = Path(os.getenv("MINDSHIFT_RECORDINGS_DIR") or (MAIN / "tmp" / "recordings"))
CLI_TS = REPO / "apps" / "mobile" / "src" / "live" / "replay" / "recordingReplay.ts"
ALERT_CODES = {None, "H", "C", "A"}
WIN_BEFORE, WIN_AFTER = score_mod.MOMENT_PRE_S, score_mod.MOMENT_WINDOW_S


def _tsx() -> Path:
    for d in [REPO, *REPO.parents]:
        p = d / "node_modules" / ".bin" / "tsx"
        if p.exists():
            return p
    raise SystemExit("tsx not found")


def run_item(name: str, out: Path, speaker_opts: str | None) -> dict:
    work = RECORDINGS / "work" / name
    vp = RECORDINGS / "inbox" / name / f"{name}.voiceprint.json"
    cmd = [str(_tsx()), str(CLI_TS), "--wav", str(work / "audio16k.wav"), "--meta", str(work / "phone_meta.json"),
           "--out", str(out), "--mode", "earpiece"]
    cmd += ["--enroll", "profile", "--profile", str(vp)] if vp.exists() else ["--enroll", "same"]
    if speaker_opts:
        cmd += ["--speaker-opts", speaker_opts]
    env = dict(os.environ)
    env["PATH"] = "/opt/homebrew/bin:" + env.get("PATH", "")
    if not out.exists():
        p = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if p.returncode != 0:
            raise RuntimeError(f"{name}: {p.stderr[-1500:]}")
    return json.loads(out.read_text())


def _moments(name: str, segs: list[dict], seg_basis: str) -> list[float]:
    """Raised/shouted wearer moments, anchored the way score.moments anchors
    them (agent A's fix: at the END of the wearer turn they are about — the
    instant tier can only buzz once that turn closes)."""
    p = RECORDINGS / "inbox" / name / f"{name}.annotation.groundtruth.json"
    if not p.exists():
        return []
    a = json.loads(p.read_text())
    basis = "segment" if seg_basis.startswith("annotation") else "deepgram-turn"
    return [score_mod.anchor_moment(m, segs, basis)[0]
            for m in a.get("coach_moments") or [] if m.get("rule") == "raised-voice"]


def score_item(name: str, phone: dict) -> dict:
    bundle = json.loads((RECORDINGS / "work" / name / "run.json").read_text())
    bundle["phone"] = {k: v for k, v in phone.items() if not k.startswith("_")}
    segs, src = score_mod.truth_segments(bundle)
    ph = score_mod.identity(bundle, [], segs)["phone"]
    rows = ph["rows"]
    other = [r for r in rows if r["truth"] is False]
    corpus = json.loads((RECORDINGS / "inbox" / name / f"{name}.corpus.json").read_text())
    dur = float(bundle["audio"]["duration_s"])
    alerts = [h["atSec"] for h in phone.get("haptics") or [] if h.get("code") in ALERT_CODES]
    moms = _moments(name, segs, src)
    hits = sum(1 for m in moms if any(m - WIN_BEFORE <= a <= m + WIN_AFTER for a in alerts))
    nomoment = sum(1 for a in alerts if not any(m - WIN_BEFORE <= a <= m + WIN_AFTER for m in moms))
    return {
        "name": name, "group": corpus.get("group"), "dur_s": dur,
        "honest": (RECORDINGS / "inbox" / name / f"{name}.voiceprint.json").exists(),
        "turns": len(rows), "decided": ph["decided"], "correct": ph["correct"],
        "wearer_turns": ph["wearer_turns"],
        "wearer_hits": sum(1 for r in rows if r["truth"] is True and r["pred"] is True),
        "other_turns": len(other), "false_self": ph["false_self"],
        "first_confirmed_s": ph["first_confirmed_s"],
        # The earliest a confirmation was possible: the first wearer turn's send.
        "first_wearer_sent_s": min((r["sent_at"] for r in rows if r["truth"] is True), default=None),
        "moments": len(moms), "alert_hits": hits, "alerts": len(alerts), "alerts_nomoment": nomoment,
    }


def aggregate(items: list[dict]) -> dict:
    honest = [i for i in items if i["honest"]]
    out: dict = {}
    if honest:
        wt = sum(i["wearer_turns"] for i in honest)
        ot = sum(i["other_turns"] for i in honest)
        tt = sum(i["turns"] for i in honest)
        ttc = [i["first_confirmed_s"] if i["first_confirmed_s"] is not None else i["dur_s"] for i in honest]
        out.update({
            "wearer_recall": round(sum(i["wearer_hits"] for i in honest) / wt, 3) if wt else None,
            "false_self_rate": round(sum(i["false_self"] for i in honest) / ot, 3) if ot else None,
            "coverage": round(sum(i["decided"] for i in honest) / tt, 3) if tt else None,
            "accuracy": round(sum(i["correct"] for i in honest) / max(1, sum(i["decided"] for i in honest)), 3),
            "time_to_confirm_s": round(statistics.median(ttc), 1),
            # time_to_confirm minus the first moment it was possible
            "confirm_delay_s": round(statistics.median(
                [(i["first_confirmed_s"] if i["first_confirmed_s"] is not None else i["dur_s"]) - i["first_wearer_sent_s"]
                 for i in honest if i["first_wearer_sent_s"] is not None]), 1),
            "never_confirmed": sum(1 for i in honest if i["first_confirmed_s"] is None),
            "id_items": len(honest),
        })
    calm = [i for i in items if i["group"] == "calm"]
    heated = [i for i in items if i["group"] != "calm"]
    out["alert_buzz_hits"] = sum(i["alert_hits"] for i in items)
    out["raised_moments"] = sum(i["moments"] for i in items)
    calm_h = sum(i["dur_s"] for i in calm) / 3600
    out["alert_buzz_false_h"] = round(sum(i["alerts"] for i in calm) / calm_h, 1) if calm_h else None
    heat_h = sum(i["dur_s"] for i in heated) / 3600
    out["alert_buzz_nomoment_h"] = round(sum(i["alerts_nomoment"] for i in heated) / heat_h, 1) if heat_h else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "held_out"])
    ap.add_argument("--items", default="", help="comma list overriding the split")
    ap.add_argument("--speaker-opts", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out-root", default=str(RECORDINGS / "landscape" / "agentC"))
    a = ap.parse_args()
    if a.split == "held_out" and not os.getenv("AGENT_C_FINAL_HELDOUT"):
        raise SystemExit("held_out is for the final check only (set AGENT_C_FINAL_HELDOUT=1)")
    splits = json.loads((RECORDINGS / "landscape" / "splits.json").read_text())
    names = [n for n in (a.items.split(",") if a.items else splits[a.split]) if n]
    out_dir = Path(a.out_root) / a.split / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "speaker_opts.json").write_text(a.speaker_opts or "{}")
    with ThreadPoolExecutor(a.jobs) as ex:
        phones = list(ex.map(lambda n: run_item(n, out_dir / f"{n}.json", a.speaker_opts or None), names))
    items = [score_item(n, p) for n, p in zip(names, phones)]
    agg = aggregate(items)
    (out_dir / "summary.json").write_text(json.dumps({"items": items, "aggregate": agg}, indent=1))
    for i in items:
        print(f'{i["name"]:24} {i["group"]:6} {"H" if i["honest"] else "s"} wearer {i["wearer_hits"]}/{i["wearer_turns"]} '
              f'false {i["false_self"]}/{i["other_turns"]} decided {i["decided"]}/{i["turns"]} '
              f'confirm {i["first_confirmed_s"]} alerts {i["alerts"]} hits {i["alert_hits"]}/{i["moments"]}')
    print(json.dumps(agg))


if __name__ == "__main__":
    main()
