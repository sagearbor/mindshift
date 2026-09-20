"""CALIBRATE THE HEAT JUDGE — pick its two arousal thresholds from labelled data.

``server/watch/heat_judge.py`` vetoes a loudness nudge on two grounds:

    valence > VALENCE_VETO_MAX     the speech sounds PLEASANT (laughing)
    arousal < CALM_AROUSAL_FLOOR   the speech sounds FLAT (projected, not angry)

and CONFIRMS at ``arousal >= CONFIRM_AROUSAL``. The valence bar is inherited
from PR #186 (leave-one-speaker-out over 85 held-out CREMA-D speakers,
docs/decisions/2026-09-17-valence-veto.md). This script chooses the two
arousal bars, against two datasets pulling in opposite directions:

  CONFER   ten human raters' continuous conflict rating at 1 Hz over TV
           debates. A window is HEATED at >= 400/1000 (confer_corpus.HEATED_AT).
           These are real arguments: vetoing one is the judge silencing the
           product. GATE: >= 90% of heated windows survive as confirm-or-unknown.
  CREMA-D  acted HAPPY clips — high arousal, high valence, exactly the
           laughing-vs-shouting confusion the dB ladder cannot resolve.
           GATE: >= 50% of them vetoed.

Both read CACHED features, never audio:
  tmp/heat-map/reference_series/*.json  5 s windows of arousal/valence/dB-over
                                        (scripts/heat_reference.py)
  tmp/feature-bank/tone.parquet         per-clip WavLM dims (scripts/feature_bank.py)

Caveat kept in the open: the cached reference windows are 5 s and the judge
runs on 2 s. Both are inside the model's trained span and the smoother gives
the judge ~4 s of context, so the 5 s cache is the closest thing on disk to
what it will see — but the thresholds should be re-checked against real
sessions once ``MINDSHIFT_TONE_AUDIO=dark`` has accumulated
``arousal_series``/``valence_series`` from live lanes (that is exactly why
those series are persisted).

    python scripts/heat_judge_calibrate.py
    python scripts/heat_judge_calibrate.py --dose     # also replay AMI/SBCSAE
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "server"))
sys.path.insert(0, str(REPO / "scripts"))

import conversation_audit as ca  # noqa: E402
from watch import heat_judge as hj  # noqa: E402


def eval_tmp() -> Path:
    """Where the gitignored eval data actually lives.

    Normally ``<repo>/tmp``. A git WORKTREE gets its own empty ``tmp/``, so
    fall back to the main checkout's: the corpora are multi-GB and are never
    copied per worktree. Returns ``<repo>/tmp`` when nothing is built, so
    every caller's ``.exists()`` check still reads false and skips cleanly.
    """
    here = REPO / "tmp"
    if (here / "heat-map").exists():
        return here
    for parent in REPO.parents:
        candidate = parent / "tmp"
        if (candidate / "heat-map").exists() and (parent / ".git").exists():
            return candidate
    return here


TMP = eval_tmp()
REFS = TMP / "heat-map/reference_series"
CONFER_MANIFEST = TMP / "corpora/confer/heatmap_manifest.json"
SBCSAE_MANIFEST = TMP / "corpora/sbcsae/heatmap_manifest.json"
AMI_DIR = TMP / "ami-corpus"
AMI_MANIFEST = AMI_DIR / "manifest.json"
BANK = TMP / "feature-bank"
OUT = TMP / "heat-judge"

#: confer_corpus.HEATED_AT — a window is "heated" at this mean rating of 1000.
HEATED_AT = 400.0


# ----------------------------------------------------------------- inputs --

def smooth_runs(idx: list[int], values: list[float]) -> list[float]:
    """Apply the judge's smoother along CONSECUTIVE windows only.

    The reference series are subsampled for long recordings, so
    ``window_idx`` has gaps; two windows either side of a gap were never two
    consecutive judge scores and must not be averaged together. Within a run
    the smoother is exactly ``heat_judge.smooth`` over the trailing
    ``SMOOTHING_N``, which is what the live judge computes.
    """
    out: list[float] = []
    run: list[float] = []
    prev: int | None = None
    for k, x in zip(idx, values):
        if prev is not None and k != prev + 1:
            run = []
        run.append(float(x))
        out.append(hj.smooth(run))
        prev = k
    return out


def confer_windows(*, smoothed: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(arousal, valence, human rating) per scored CONFER window.

    ``smoothed`` applies the judge's own 3-window exponential smoother along
    each contiguous run — the judge never sees a bare window, and the
    smoother's whole job is to stop one freak score flipping a verdict, so
    calibrating on bare windows would tune the wrong distribution. CREMA-D
    clips get NO smoothing (:func:`cremad_dims`): a 2.5 s clip is one steady
    state, so the judge's three windows would all be that clip and the
    smoother would return the same number. The asymmetry is real and it
    biases this calibration toward vetoing MORE acted-happy than heated
    conversation, which is the safe direction.
    """
    if not CONFER_MANIFEST.exists():
        return np.array([]), np.array([]), np.array([]), np.array([])
    by_id = {e["id"]: e for e in json.loads(CONFER_MANIFEST.read_text())}
    a: list[float] = []
    v: list[float] = []
    h: list[float] = []
    d: list[float] = []
    for path in sorted(REFS.glob("confer_*.json")):
        entry = by_id.get(path.stem)
        if entry is None or not entry.get("conflict_per_second"):
            continue
        ref = json.loads(path.read_text())
        w = int(ref["win_s"])
        rating = np.asarray(entry["conflict_per_second"], dtype=float)
        idx = list(ref["window_idx"])
        ar = smooth_runs(idx, ref["arousal"]) if smoothed else list(ref["arousal"])
        va = smooth_runs(idx, ref["valence"]) if smoothed else list(ref["valence"])
        for k, x, y, over in zip(idx, ar, va, ref["ours_db_over"]):
            seg = rating[k * w:(k + 1) * w]
            if seg.size == 0:
                continue
            a.append(float(x))
            v.append(float(y))
            h.append(float(seg.mean()))
            d.append(float(over))
    return np.asarray(a), np.asarray(v), np.asarray(h), np.asarray(d)


def cremad_dims() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """emotion -> (arousal, valence, dB over that speaker's own neutral).

    The dB column is what makes the population comparable with CONFER's: the
    veto is a SECOND gate behind loudness, so the only clips it can act on
    are the ones already over the +6 dB first rung (PR #186's framing, and
    the reason its tables are quoted as "of what the rung already caught").

    Reads the SAME two artefacts PR #186 calibrated the valence bar on —
    ``tmp/corpora/valence_probe.json`` (tone_id's own dimensional output) and
    ``tmp/corpora/heat_rubric_cremad.csv`` (dB over each speaker's own
    neutral) — so every number here is directly comparable with
    docs/decisions/2026-09-17-valence-veto.md rather than nearly so. The
    feature bank carries the same dims for a subset of these clips but its
    own dB estimator, which puts 99 happy clips over the rung where the
    rubric puts 116; using the rubric keeps one number meaning one thing.
    """
    probe_p = TMP / "corpora/valence_probe.json"
    rubric_p = TMP / "corpora/heat_rubric_cremad.csv"
    if not probe_p.exists() or not rubric_p.exists():
        return {}
    import csv

    rubric: dict[str, tuple[str, float]] = {}
    with rubric_p.open() as fh:
        for row in csv.DictReader(fh):
            try:
                rubric[row["path"]] = (row["emotion"], float(row["db_over_own_neutral"]))
            except (KeyError, TypeError, ValueError):
                continue
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    for clip in json.loads(probe_p.read_text()):
        hit = rubric.get(clip.get("path", ""))
        if hit is None or "arousal" not in clip or "valence" not in clip:
            continue
        grouped.setdefault(hit[0], []).append((float(clip["arousal"]), float(clip["valence"]), hit[1]))
    return {
        emotion: (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
                  np.array([r[2] for r in rows]))
        for emotion, rows in grouped.items()
    }


# ------------------------------------------------------------------ sweep --

def verdicts(arousal: np.ndarray, valence: np.ndarray, floor: float, confirm: float,
             *, veto_first: bool = False) -> np.ndarray:
    """The heat_judge rules, vectorised, on already-smoothed dims.

    ``MIN_SCORES`` is a liveness rule (the first second of a session), not a
    threshold — a calibration corpus has every window, so it does not apply.

    ``veto_first`` is the rule order this module was FIRST specified with:
    valence vetoes before confirm is even considered. It is kept as a
    measurable alternative because the shipped order (confirm first) was
    chosen from the number it produces — see the CONFER rows in the docs
    decision record.
    """
    out = np.full(len(arousal), hj.VERDICT_UNKNOWN, dtype=object)
    if veto_first:
        out[arousal >= confirm] = hj.VERDICT_CONFIRM
        out[arousal < floor] = hj.VERDICT_VETO
        out[valence > hj.VALENCE_VETO_MAX] = hj.VERDICT_VETO
        return out
    out[valence > hj.VALENCE_VETO_MAX] = hj.VERDICT_VETO
    out[arousal < floor] = hj.VERDICT_VETO
    out[arousal >= confirm] = hj.VERDICT_CONFIRM
    return out


#: The first rung of the shipped ladder, in dB over baseline. Both gate
#: populations are restricted to moments at or above it, because a veto on a
#: window the ladder was never going to nudge on costs nothing and proves
#: nothing (docs/decisions/2026-09-17-valence-veto.md, "The veto can only act
#: on clips that already clear the +6 dB rung").
FIRST_RUNG_DB = 6.0
#: Gates. All three at once, or the pair does not ship.
MAX_HEATED_VETOED = 0.10   # CONFER human-rated heated windows the judge may veto
MIN_HAPPY_VETOED = 0.50    # acted HAPPY over the first rung the judge must veto
#: PR #186's own gate, inherited: the valence veto was chosen as the highest
#: threshold that still keeps >= 90% of the anger the rung already caught
#: (docs/decisions/2026-09-17-valence-veto.md). A calm floor that fixes the
#: happy false alarms by throwing away real anger is not an improvement, and
#: the first pair this sweep produced without this constraint did exactly
#: that: floor 0.70 vetoed 85.9% of happy AND 58.7% of angry.
MAX_ANGRY_VETOED = 0.10


def sweep(ca_: np.ndarray, cv: np.ndarray, heated: np.ndarray,
          ha: np.ndarray, hv: np.ndarray, aa: np.ndarray, av: np.ndarray) -> list[dict]:
    rows = []
    for floor in np.round(np.arange(0.20, 0.71, 0.01), 3):
        for confirm in np.round(np.arange(0.30, 0.91, 0.02), 3):
            if confirm <= floor:
                continue
            cw = verdicts(ca_[heated], cv[heated], floor, confirm)
            cc = verdicts(ca_[~heated], cv[~heated], floor, confirm)
            vf = verdicts(ca_[heated], cv[heated], floor, confirm, veto_first=True)
            hh = verdicts(ha, hv, floor, confirm)
            aaa = verdicts(aa, av, floor, confirm)
            rows.append({
                "calm_floor": float(floor),
                "confirm_arousal": float(confirm),
                "confer_heated_vetoed": float((cw == hj.VERDICT_VETO).mean()),
                "confer_heated_confirmed": float((cw == hj.VERDICT_CONFIRM).mean()),
                "confer_calm_vetoed": float((cc == hj.VERDICT_VETO).mean()),
                "confer_calm_confirmed": float((cc == hj.VERDICT_CONFIRM).mean()),
                "confer_heated_vetoed_veto_first": float((vf == hj.VERDICT_VETO).mean()),
                "cremad_happy_vetoed": float((hh == hj.VERDICT_VETO).mean()),
                "cremad_angry_vetoed": float((aaa == hj.VERDICT_VETO).mean()),
            })
    return rows


def clearing(rows: list[dict]) -> list[dict]:
    """Every threshold pair that clears all three gates."""
    return [r for r in rows
            if r["confer_heated_vetoed"] <= MAX_HEATED_VETOED
            and r["cremad_happy_vetoed"] >= MIN_HAPPY_VETOED
            and r["cremad_angry_vetoed"] <= MAX_ANGRY_VETOED]


def pick(rows: list[dict]) -> dict | None:
    """Best row that clears every gate.

    Ranked by (1) how much acted HAPPY it vetoes — the defect this module
    exists to fix — then (2) how well CONFIRM separates human-rated heated
    from calm windows (Youden's J on the confirm rate), then (3) the LOWER
    calm floor, because a floor is a veto and the cheaper veto is the safer
    one.
    """
    ok = clearing(rows)
    if not ok:
        return None
    return max(ok, key=lambda r: (
        round(r["cremad_happy_vetoed"], 3),
        round(r["confer_heated_confirmed"] - r["confer_calm_confirmed"], 3),
        -r["calm_floor"],
    ))


# ------------------------------------------------------------------- dose --

def _series_gate(ref: dict, n: int, floor: float, confirm: float) -> np.ndarray:
    """Per-SECOND gate for one recording from its cached 5 s reference series.

    Windows the reference never scored stay OPEN — same fail-open rule as the
    shipped valence veto: an absent signal is not evidence a moment was calm.
    """
    w = int(ref["win_s"])
    gate = np.ones(n, dtype=bool)
    ar = np.asarray(ref["arousal"], dtype=float)
    va = np.asarray(ref["valence"], dtype=float)
    vs = verdicts(ar, va, floor, confirm)
    for k, verdict in zip(ref["window_idx"], vs):
        if verdict == hj.VERDICT_VETO:
            gate[k * w:(k + 1) * w] = False
    return gate


def dose_rows(floor: float, confirm: float) -> list[dict]:
    """Replay AMI + SBCSAE with and without the judge gate. Needs the audio."""
    entries: list[dict] = []
    if AMI_MANIFEST.exists():
        m = json.loads(AMI_MANIFEST.read_text())
        entries += [{"id": x["meeting"], "corpus": "AMI",
                     "audio": str(AMI_DIR / x["mix"]),
                     "duration_s": x["duration_s"]} for x in m["meetings"]]
    if SBCSAE_MANIFEST.exists():
        base = SBCSAE_MANIFEST.parent
        for e in json.loads(SBCSAE_MANIFEST.read_text()):
            p = Path(e["audio"])
            entries.append({"id": e["id"], "corpus": "SBCSAE",
                            "audio": str(p if p.is_absolute() else base / p),
                            "duration_s": e.get("duration_s")})
    rows = []
    for e in entries:
        ref_path = REFS / f"{e['id']}.json"
        if not Path(e["audio"]).exists() or not ref_path.exists():
            continue
        pcm, sr = ca_load(e["audio"])
        over = ca.db_over_baseline(ca.windows_dbfs(pcm, sr))
        hours = (e.get("duration_s") or len(pcm) / sr) / 3600.0
        if not hours:
            continue
        ref = json.loads(ref_path.read_text())
        gate = _series_gate(ref, len(over), floor, confirm)
        rows.append({
            "id": e["id"], "corpus": e["corpus"],
            "dose_before": round(len(ca.replay(over, False, True)) / hours, 1),
            "dose_after": round(len(ca.replay(over, False, True, gate=gate)) / hours, 1),
            "windows_vetoed": round(float(1.0 - gate.mean()), 3),
            "loud_and_vetoed": round(float(((over >= 6.0) & ~gate).sum()), 0),
        })
    return rows


def ca_load(path: str) -> tuple[np.ndarray, int]:
    import wave
    with wave.open(path) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm, sr


# ------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", action="store_true", help="also replay AMI/SBCSAE dose (needs the audio)")
    ap.add_argument("--json", default=str(OUT / "calibration.json"))
    args = ap.parse_args()

    ca_, cv, ch, cd = confer_windows()
    if not len(ca_):
        print("no CONFER reference series — run scripts/heat_reference.py first")
        return 1
    heated = ch >= HEATED_AT
    loud = cd >= FIRST_RUNG_DB
    cremad = cremad_dims()
    if "happy" not in cremad:
        print("no CREMA-D tone/prosody features — run scripts/feature_bank.py --group tone")
        return 1
    ha_all, hv_all, hd_all = cremad["happy"]
    hloud = hd_all >= FIRST_RUNG_DB
    ha, hv = ha_all[hloud], hv_all[hloud]
    aa_all, av_all, ad_all = cremad["angry"]
    aloud = ad_all >= FIRST_RUNG_DB
    aa, av = aa_all[aloud], av_all[aloud]

    ra, rv, _, _ = confer_windows(smoothed=False)
    hot_loud = heated & loud
    print(f"CONFER  {len(ca_)} scored windows over "
          f"{len(list(REFS.glob('confer_*.json')))} recordings, "
          f"{int(heated.sum())} heated (>= {HEATED_AT:.0f}/1000), "
          f"{int(hot_loud.sum())} of those also >= +{FIRST_RUNG_DB:.0f} dB "
          f"(i.e. the dB ladder would nudge on {int(hot_loud.sum())} of them)")
    print(f"        smoothing cuts the heated-window valence sd "
          f"{rv[heated].std():.3f} -> {cv[heated].std():.3f}, arousal "
          f"{ra[heated].std():.3f} -> {ca_[heated].std():.3f}")
    print(f"        heated  arousal {ca_[heated].mean():.3f} +/- {ca_[heated].std():.3f}  "
          f"valence {cv[heated].mean():.3f} +/- {cv[heated].std():.3f}")
    print(f"        calm    arousal {ca_[~heated].mean():.3f} +/- {ca_[~heated].std():.3f}  "
          f"valence {cv[~heated].mean():.3f} +/- {cv[~heated].std():.3f}")
    print(f"        valence alone (> {hj.VALENCE_VETO_MAX}, PR #186's inherited bar) vetoes "
          f"{(cv[heated] > hj.VALENCE_VETO_MAX).mean():.3f} of heated windows — "
          f"which is why CONFIRM must outrank it")
    for name, (a, v, d) in sorted(cremad.items()):
        k = d >= FIRST_RUNG_DB
        print(f"CREMA-D {name:<8} n={len(a):<4} ({int(k.sum())} over +{FIRST_RUNG_DB:.0f} dB)  "
              f"arousal {a.mean():.3f}  valence {v.mean():.3f}")

    rows = sweep(ca_, cv, heated, ha, hv, aa, av)
    chosen = pick(rows)
    if chosen is None:
        print("\nNO threshold pair clears all three gates — the judge cannot ship as specified.")
        best = max(rows, key=lambda r: r["cremad_happy_vetoed"]
                   - 5 * max(0.0, r["confer_heated_vetoed"] - MAX_HEATED_VETOED))
        print("closest:", json.dumps(best, indent=1))
        return 2

    print("\n--- sweep (rows clearing all three gates, best first) ---")
    ok = sorted(clearing(rows),
                key=lambda r: (-r["cremad_happy_vetoed"],
                               -(r["confer_heated_confirmed"] - r["confer_calm_confirmed"]), r["calm_floor"]))
    print(f"{'floor':>6} {'confirm':>8} {'heated vetoed':>14} {'heated conf':>12} "
          f"{'calm conf':>10} {'happy vetoed':>13} {'angry vetoed':>13} {'veto-first':>11}")
    for r in ok[:12]:
        print(f"{r['calm_floor']:6.2f} {r['confirm_arousal']:8.2f} {r['confer_heated_vetoed']:14.3f} "
              f"{r['confer_heated_confirmed']:12.3f} {r['confer_calm_confirmed']:10.3f} "
              f"{r['cremad_happy_vetoed']:13.3f} {r['cremad_angry_vetoed']:13.3f} "
              f"{r['confer_heated_vetoed_veto_first']:11.3f}")
    print(f"\n{len(ok)} of {len(rows)} threshold pairs clear all three gates.")
    print(f"CHOSEN  CALM_AROUSAL_FLOOR = {chosen['calm_floor']:.2f}   "
          f"CONFIRM_AROUSAL = {chosen['confirm_arousal']:.2f}")
    print(f"        shipped values      = {hj.CALM_AROUSAL_FLOOR:.2f}   {hj.CONFIRM_AROUSAL:.2f}")

    # Contrast: what the same pair does to the emotions the judge must NOT veto,
    # over the clips the rung already catches.
    at = {}
    for name, (a, v, d) in cremad.items():
        k = d >= FIRST_RUNG_DB
        at[name] = (float((verdicts(a[k], v[k], chosen["calm_floor"],
                                    chosen["confirm_arousal"]) == hj.VERDICT_VETO).mean())
                    if k.sum() else None)
    print("        CREMA-D veto rate over +6 dB clips: "
          + "  ".join(f"{k}={'n/a' if v is None else f'{v:.3f}'}" for k, v in sorted(at.items())))

    out = {
        "generated": "scripts/heat_judge_calibrate.py",
        "valence_veto_max": hj.VALENCE_VETO_MAX,
        "first_rung_db": FIRST_RUNG_DB,
        "gates": {"max_heated_vetoed": MAX_HEATED_VETOED, "min_happy_vetoed": MIN_HAPPY_VETOED,
                  "max_angry_vetoed": MAX_ANGRY_VETOED},
        "chosen": chosen,
        "pairs_clearing_all_gates": len(ok),
        "confer": {
            "windows": int(len(ca_)), "heated": int(heated.sum()),
            "heated_and_loud": int(hot_loud.sum()), "heated_at": HEATED_AT,
            "recordings": len(list(REFS.glob("confer_*.json"))),
        },
        "cremad": {"happy_over_first_rung": int(hloud.sum()), "happy_total": int(len(ha_all))},
        "cremad_veto_rate": at,
        "sweep": rows,
    }
    if args.dose:
        doses = dose_rows(chosen["calm_floor"], chosen["confirm_arousal"])
        out["dose"] = doses
        print("\n--- dose per hour, before -> after the judge gate ---")
        for corpus in ("AMI", "SBCSAE"):
            rs = [r for r in doses if r["corpus"] == corpus]
            if not rs:
                continue
            b = np.median([r["dose_before"] for r in rs])
            a = np.median([r["dose_after"] for r in rs])
            fell = sum(1 for r in rs if r["dose_after"] < r["dose_before"])
            rose = sum(1 for r in rs if r["dose_after"] > r["dose_before"])
            print(f"{corpus:<7} n={len(rs):<3} median {b:6.1f} -> {a:6.1f}   "
                  f"({fell}/{len(rs)} recordings fell, {rose} rose)")
            out.setdefault("dose_summary", {})[corpus] = {
                "n": len(rs), "median_before": round(float(b), 1), "median_after": round(float(a), 1),
                "fell": fell, "rose": rose,
            }

    OUT.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=1))
    print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
