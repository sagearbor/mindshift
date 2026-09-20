"""DOES IT KNOW IT'S YOU — in a real room, on one microphone?

The loop only ever nudges the coached user about THEIR OWN behaviour, so every
nudge rests on one claim: that the voice it just heard is yours. That claim has
only ever been tested on two-speaker synthetic scenes and a 30-second family
recording. This tests it where it actually has to work.

Method, which mirrors the product exactly:

  ENROL from clean solo audio. AMI gives each participant their own headset, so
  a voiceprint is built from that person speaking alone — which is what voice
  training on the phone produces, and it is the only honest way to enrol (a
  print built from the mix would already contain the people we then ask it to
  reject).

  MATCH on the single-microphone mix. The summed headsets are what one phone or
  one watch in the room hears: four people, crosstalk, and real overlap. For
  every ground-truth turn we embed that span of the MIX and ask whether it
  matches the enrolled print.

The two failures have very different costs, so they are reported separately:

  a MISS  — your own raised turn is filed under a stranger, and the coach says
            nothing. This is the defect that made shouting invisible until the
            raised-print fix.
  a FALSE ACCEPT — someone else's turn is filed as yours, so the wrist tells
            you off for what a colleague just did. Worse than silence.

    python scripts/ami_corpus.py --keep-headsets
    python scripts/identity_audit.py
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
#: Enrolment: clean solo spans from the wearer's own headset. Four takes is
#: what the phone's voice training asks for (before the raised prompts).
ENROL_TAKES = 4
ENROL_MIN_S = 2.0
#: Turns shorter than this are not scored — ECAPA needs material, and the live
#: loop does not embed sub-second fragments either.
MIN_TURN_S = 1.0
#: Cap per meeting so a full audit stays minutes, not hours, on CPU.
MAX_TURNS_PER_MEETING = 120


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
    return x, sr


def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return x
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(sr, TARGET_SR)
    return resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)


def enrol(headset: Path, spans: list[list[float]], embed) -> np.ndarray | None:
    """A voiceprint from the longest clean solo takes, exactly as the phone builds one."""
    import speaker_id as sid

    x, sr = read_wav(headset)
    x = resample_to_16k(x, sr)
    usable = [s for s in spans if s[1] - s[0] >= ENROL_MIN_S]
    usable.sort(key=lambda s: s[1] - s[0], reverse=True)
    vecs = []
    for a, b in usable[:ENROL_TAKES]:
        seg = x[int(a * TARGET_SR):int(b * TARGET_SR)]
        if len(seg) < int(ENROL_MIN_S * TARGET_SR):
            continue
        vecs.append(embed(seg))
    if not vecs:
        return None
    return sid.l2_normalize(np.mean(vecs, axis=0))


def audit_meeting(meta: dict, corpus: Path, embed, rng: np.random.Generator) -> list[dict]:
    import speaker_id as sid

    mix, sr = read_wav(corpus / meta["mix"])
    mix = resample_to_16k(mix, sr)
    heads = corpus / "headsets"
    rows = []
    # ONE candidate set for the whole meeting, chosen before the wearer loop.
    # It used to be re-sampled inside the loop, which meant each wearer scored a
    # DIFFERENT subset of turns — fine for per-wearer rates, but it made the
    # rows impossible to join back into "this turn, scored against every print",
    # which is what a margin rule (top vs runner-up) has to see.
    all_turns = [(spk, a, b) for spk, sp in meta["speakers"].items() for a, b in sp if b - a >= MIN_TURN_S]
    if len(all_turns) > MAX_TURNS_PER_MEETING:
        pick = rng.choice(len(all_turns), MAX_TURNS_PER_MEETING, replace=False)
        all_turns = [all_turns[i] for i in sorted(pick)]
    for wearer, spans in meta["speakers"].items():
        idx = int(wearer[1:])
        hs = heads / f"{meta['meeting']}.Headset-{idx}.wav"
        if not hs.exists():
            continue
        print_vec = enrol(hs, spans, embed)
        if print_vec is None:
            continue
        # The same turns for every wearer — see all_turns above.
        for spk, a, b in all_turns:
            seg = mix[int(a * TARGET_SR):int(b * TARGET_SR)]
            if len(seg) < int(MIN_TURN_S * TARGET_SR):
                continue
            score = float(np.dot(sid.l2_normalize(embed(seg)), print_vec))
            # How much of this turn had somebody ELSE talking over it. A single
            # mic cannot separate them, so an overlapped span hands ECAPA a
            # blend of two voices — which is a different failure from "the
            # threshold is too strict", and the fix for each is different.
            others = [sp for sp in meta["speakers"] if sp != spk]
            clash = 0.0
            for o in others:
                for c, d in meta["speakers"][o]:
                    clash += max(0.0, min(b, d) - max(a, c))
            rows.append({
                # Stable id so the same turn's scores against every print can be
                # joined: the margin rule needs top-vs-runner-up, not one score.
                "turn_id": f"{meta['meeting']}:{spk}:{a:.2f}",
                "meeting": meta["meeting"], "wearer": wearer, "turn_speaker": spk,
                "is_self": spk == wearer, "score": round(score, 4),
                "seconds": round(b - a, 2),
                "overlap_fraction": round(min(1.0, clash / max(b - a, 1e-6)), 3),
            })
        print(f"  {meta['meeting']} wearer={wearer}: {len([r for r in rows if r['wearer']==wearer])} turns scored",
              flush=True)
    return rows


def report(rows: list[dict], threshold: float) -> None:
    self_scores = np.array([r["score"] for r in rows if r["is_self"]])
    other_scores = np.array([r["score"] for r in rows if not r["is_self"]])
    if not len(self_scores) or not len(other_scores):
        print("not enough data")
        return
    print(f"\n{'='*74}\nIDENTITY ON ONE MICROPHONE, IN A REAL ROOM\n{'='*74}")
    print(f"{len(self_scores)} of the wearer's own turns · {len(other_scores)} other people's turns")
    print(f"median cosine — own {np.median(self_scores):.3f} · others {np.median(other_scores):.3f}")
    pairs = (self_scores[:, None] > other_scores[None, :]).mean()
    print(f"AUC (can it rank you above them at all): {pairs:.3f}")
    print(f"\n{'threshold':>10} {'finds you':>12} {'FALSE ACCEPTS':>15}   (someone else read as you)")
    for t in (0.45, 0.5, 0.55, 0.6, threshold, 0.7, 0.75):
        print(f"{t:>10.2f} {(self_scores >= t).mean()*100:>11.1f}% {(other_scores >= t).mean()*100:>14.1f}%")
    print(f"\nAt the SHIPPED threshold {threshold}: finds you on "
          f"{(self_scores >= threshold).mean()*100:.1f}% of your own turns, and reads "
          f"{(other_scores >= threshold).mean()*100:.1f}% of other people's turns as you.")

    # Is the miss caused by OVERLAP, or by the threshold? Different fixes.
    clean = np.array([r["score"] for r in rows if r["is_self"] and r["overlap_fraction"] < 0.1])
    messy = np.array([r["score"] for r in rows if r["is_self"] and r["overlap_fraction"] >= 0.5])
    print(f"\n{'='*74}\nIS IT THE THRESHOLD, OR IS IT THE CROSSTALK?\n{'='*74}")
    if len(clean):
        print(f"your turns with almost NO overlap (<10%):  n={len(clean):<4} "
              f"median {np.median(clean):.3f} · found at {threshold} = {(clean >= threshold).mean()*100:.1f}%")
    if len(messy):
        print(f"your turns MOSTLY overlapped (>=50%):      n={len(messy):<4} "
              f"median {np.median(messy):.3f} · found at {threshold} = {(messy >= threshold).mean()*100:.1f}%")
    if len(clean) and len(messy):
        # Weigh by VOLUME, not by the gap between medians. An earlier version of
        # this compared medians alone and concluded "crosstalk is the dominant
        # cause" — which was wrong: overlapped turns score far worse, but there
        # are very few of them, so they account for a small share of the actual
        # misses. Both problems are real and they need different fixes, so say
        # how much of the damage each one does.
        miss_clean = int((clean < threshold).sum())
        miss_messy = int((messy < threshold).sum())
        all_self_missed = int((self_scores < threshold).sum())
        print(f"\nof {all_self_missed} missed turns of your own: {miss_clean} were CLEAN, "
              f"{miss_messy} were heavily overlapped")
        print("-> two separate problems:")
        print(f"   crosstalk makes a turn unmatchable (median {np.median(messy):.3f}, "
              f"essentially noise) — those spans should be EXCLUDED, not thresholded; "
              f"they are {miss_messy/max(all_self_missed,1)*100:.0f}% of the misses")
        print(f"   and the bar sits at {threshold} while clean turns have a median of "
              f"{np.median(clean):.3f} — so it misses about half of your CLEAN turns by "
              f"construction, which is {miss_clean/max(all_self_missed,1)*100:.0f}% of them")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO / "tmp/ami-corpus"))
    ap.add_argument("--out", default=str(REPO / "tmp/ami-corpus/identity.json"))
    ap.add_argument("--meetings", type=int, default=4, help="how many meetings to score")
    args = ap.parse_args()

    import speaker_id as sid
    if not sid.is_available():
        raise SystemExit("ECAPA unavailable — pip install -r requirements-voice.txt")

    corpus = Path(args.corpus)
    manifest = json.loads((corpus / "manifest.json").read_text())
    rng = np.random.default_rng(7)

    def embed(seg: np.ndarray) -> np.ndarray:
        return sid.embed_pcm(seg.astype(np.float32), TARGET_SR)

    rows: list[dict] = []
    for meta in manifest["meetings"][:args.meetings]:
        if not (corpus / "headsets" / f"{meta['meeting']}.Headset-0.wav").exists():
            print(f"{meta['meeting']}: no headsets kept — rerun ami_corpus.py --keep-headsets")
            continue
        rows.extend(audit_meeting(meta, corpus, embed, rng))
    if not rows:
        raise SystemExit("nothing scored")
    Path(args.out).write_text(json.dumps(rows, indent=1))
    report(rows, sid.MATCH_THRESHOLD)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
