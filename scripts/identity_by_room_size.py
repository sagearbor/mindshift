"""Is the voiceprint threshold wrong for a REAL ROOM, or only for a CROWDED one?

The audit (docs/plans/2026-09-18-real-conversation-audit.md) measured identity
on four-way AMI meetings and found the shipped MATCH_THRESHOLD of 0.65 finds
only ~35% of the wearer's own turns. That is grounds to re-tune it — but the
app's primary case is one conversation with one other person, not a meeting,
and a threshold re-tuned on the hardest condition could be wrong for the common
one.

This isolates room size as the only variable. AMI gives each participant their
own headset, so the SAME audio can be mixed as a two-person conversation or a
four-person one: same speakers, same room, same microphones, same enrolment.
Anything that changes between the two conditions is caused by the extra voices
and nothing else.

    python scripts/ami_corpus.py --keep-headsets
    python scripts/identity_by_room_size.py
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "server"))
sys.path.insert(0, str(REPO / "scripts"))

TARGET_SR = 16000
ENROL_TAKES, ENROL_MIN_S, MIN_TURN_S = 4, 2.0, 1.0
MAX_TURNS = 90


def read16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if sr != TARGET_SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(sr, TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)
    return x


def mix_of(chans: list[np.ndarray]) -> np.ndarray:
    """Sum then peak-normalise — the same mixdown ami_corpus.py writes, so the
    two-person and four-person conditions differ only in who is in the room."""
    n = min(len(c) for c in chans)
    m = np.sum([c[:n] for c in chans], axis=0)
    peak = np.max(np.abs(m))
    return (m / peak * 10 ** (-3.0 / 20.0)).astype(np.float32) if peak > 0 else m.astype(np.float32)


def score(meta, heads: dict[int, np.ndarray], present: list[int], embed, rng) -> list[dict]:
    import speaker_id as sid

    mixed = mix_of([heads[i] for i in present])
    rows = []
    for wearer in present:
        spans = meta["speakers"][f"S{wearer}"]
        usable = sorted((s for s in spans if s[1] - s[0] >= ENROL_MIN_S),
                        key=lambda s: s[1] - s[0], reverse=True)[:ENROL_TAKES]
        vecs = []
        for a, b in usable:
            seg = heads[wearer][int(a * TARGET_SR):int(b * TARGET_SR)]
            if len(seg) >= int(ENROL_MIN_S * TARGET_SR):
                vecs.append(embed(seg))
        if not vecs:
            continue
        print_vec = sid.l2_normalize(np.mean(vecs, axis=0))
        cands = [(s, a, b) for s in present for a, b in meta["speakers"][f"S{s}"] if b - a >= MIN_TURN_S]
        if len(cands) > MAX_TURNS:
            cands = [cands[i] for i in sorted(rng.choice(len(cands), MAX_TURNS, replace=False))]
        for spk, a, b in cands:
            seg = mixed[int(a * TARGET_SR):int(b * TARGET_SR)]
            if len(seg) < int(MIN_TURN_S * TARGET_SR):
                continue
            rows.append({
                "room": len(present), "wearer": wearer, "is_self": spk == wearer,
                "score": float(np.dot(sid.l2_normalize(embed(seg)), print_vec)),
            })
    return rows


def summarise(rows: list[dict], label: str, threshold: float) -> None:
    own = np.array([r["score"] for r in rows if r["is_self"]])
    oth = np.array([r["score"] for r in rows if not r["is_self"]])
    auc = (own[:, None] > oth[None, :]).mean()
    print(f"\n{label}: {len(own)} own turns, {len(oth)} others · AUC {auc:.3f} · "
          f"median own {np.median(own):.3f}")
    print(f"  {'threshold':>10} {'finds you':>11} {'false accepts':>14}")
    for t in (0.50, 0.55, 0.60, threshold, 0.70):
        print(f"  {t:>10.2f} {(own >= t).mean()*100:>10.1f}% {(oth >= t).mean()*100:>13.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO / "tmp/ami-corpus"))
    ap.add_argument("--meetings", type=int, default=4)
    args = ap.parse_args()
    import speaker_id as sid
    if not sid.is_available():
        raise SystemExit("ECAPA unavailable — pip install -r requirements-voice.txt")

    corpus = Path(args.corpus)
    manifest = json.loads((corpus / "manifest.json").read_text())
    rng = np.random.default_rng(3)

    def embed(seg):
        return sid.embed_pcm(seg.astype(np.float32), TARGET_SR)

    dyad, quad = [], []
    for meta in manifest["meetings"][:args.meetings]:
        hp = corpus / "headsets"
        paths = {i: hp / f"{meta['meeting']}.Headset-{i}.wav" for i in range(meta["n_speakers"])}
        if not all(p.exists() for p in paths.values()):
            print(f"{meta['meeting']}: headsets missing — rerun ami_corpus.py --keep-headsets")
            continue
        heads = {i: read16k(p) for i, p in paths.items()}
        print(f"{meta['meeting']} ...", flush=True)
        dyad += score(meta, heads, [0, 1], embed, rng)
        quad += score(meta, heads, list(paths), embed, rng)

    if not dyad or not quad:
        raise SystemExit("nothing scored")
    print(f"\n{'='*70}\nSAME SPEAKERS, SAME ROOM, SAME ENROLMENT — ONLY THE CROWD CHANGES\n{'='*70}")
    summarise(dyad, "TWO people in the room", sid.MATCH_THRESHOLD)
    summarise(quad, "FOUR people in the room", sid.MATCH_THRESHOLD)
    d = np.array([r["score"] for r in dyad if r["is_self"]])
    q = np.array([r["score"] for r in quad if r["is_self"]])
    t = sid.MATCH_THRESHOLD
    print(f"\n-> at the shipped {t}, the wearer is found on {(d >= t).mean()*100:.1f}% of their own "
          f"turns in a dyad and {(q >= t).mean()*100:.1f}% in a four-way.")
    print("   " + ("The dyad is fine — the threshold is only wrong for crowded rooms, so a GLOBAL "
                   "change would be the wrong fix."
                   if (d >= t).mean() > 0.7 else
                   "The dyad is ALSO poor — the threshold is miscalibrated for real rooms in "
                   "general, not just crowded ones, so a global change is justified."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
