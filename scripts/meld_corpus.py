"""MELD -> heat-map recordings: multi-party dialogue with per-utterance emotion AND speaker.

Why MELD: it is the one conversational corpus with anger/joy/sadness labelled
per utterance, named speakers, and no access gate (10 GB, downloadable today).
It is the first data we hold where "how heated is this conversation" has a
human answer per turn rather than a loudness proxy.

Why it is not the last word: it is sitcom acting (Friends) with a laugh track
and music, not spontaneous conflict. It answers "does our heat measure agree
with human anger labels in dialogue" — it does not answer "does it generalise
to a real argument at home". Carry that caveat into every number.

Each MELD dialogue becomes ONE recording: its utterance clips concatenated in
order with a short gap, so the loop sees turn-taking; the per-utterance emotion
gives heated_spans (anger, disgust) and the named speaker gives speaker spans.

    python scripts/meld_corpus.py --max-dialogues 300
    -> tmp/corpora/meld/dialogues/*.wav + tmp/corpora/meld/heatmap_manifest.json

Licence: repo GPL-3.0; underlying footage is copyrighted TV. Research use in a
gitignored tmp/ only — nothing here is committed or redistributed.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import tarfile
import wave
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "tmp/corpora/meld"
SR = 16000
GAP_S = 0.3
HEATED = {"anger", "disgust"}


def extract_all(root: Path) -> None:
    """MELD.Raw.tar.gz holds nested train/dev/test tarballs — unpack whatever is
    present and has not been unpacked yet."""
    for tgz in sorted(root.rglob("*.tar.gz")):
        marker = tgz.with_suffix(".extracted")
        if marker.exists():
            continue
        print(f"  extracting {tgz.name} ...", flush=True)
        with tarfile.open(tgz) as t:
            t.extractall(tgz.parent, filter="data")
        marker.write_text("ok")


def clip_index(root: Path) -> dict[str, Path]:
    return {p.name: p for p in root.rglob("dia*_utt*.mp4")}


def load_dialogues(root: Path) -> dict[tuple[str, int], list[dict]]:
    """split+dialogue_id -> utterances in order."""
    dialogues: dict[tuple[str, int], list[dict]] = {}
    for csv_path in sorted(root.rglob("*_sent_emo*.csv")):
        split = csv_path.stem.split("_")[0]          # train / dev / test
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (split, int(row["Dialogue_ID"]))
                dialogues.setdefault(key, []).append({
                    "utt": int(row["Utterance_ID"]), "speaker": row["Speaker"].strip(),
                    "emotion": row["Emotion"].strip().lower(), "text": row["Utterance"],
                })
    for v in dialogues.values():
        v.sort(key=lambda u: u["utt"])
    return dialogues


def mp4_to_pcm(path: Path) -> np.ndarray | None:
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
        capture_output=True,
    )
    if r.returncode != 0 or not r.stdout:
        return None
    return np.frombuffer(r.stdout, dtype="<i2")


def build(root: Path, max_dialogues: int, min_utts: int) -> list[dict]:
    clips = clip_index(root)
    dialogues = load_dialogues(root)
    out_dir = root / "dialogues"
    out_dir.mkdir(exist_ok=True)
    entries = []
    # Prefer dialogues with the most heated content first, then longest — the
    # point is to cover the heated end of the axis our other corpora lack.
    def rank(item):
        _, utts = item
        heated = sum(u["emotion"] in HEATED for u in utts)
        return (-heated, -len(utts))
    for (split, did), utts in sorted(dialogues.items(), key=rank):
        if len(utts) < min_utts or len({u["speaker"] for u in utts}) < 2:
            continue
        if len(entries) >= max_dialogues:
            break
        pcm_parts, t, speakers, heated_spans, per_utt = [], 0.0, {}, [], []
        gap = np.zeros(int(GAP_S * SR), dtype="<i2")
        for u in utts:
            p = clips.get(f"dia{did}_utt{u['utt']}.mp4")
            if p is None:
                continue
            pcm = mp4_to_pcm(p)
            if pcm is None or len(pcm) < SR // 4:
                continue
            a, b = t, t + len(pcm) / SR
            speakers.setdefault(u["speaker"], []).append([round(a, 3), round(b, 3)])
            if u["emotion"] in HEATED:
                heated_spans.append([round(a, 3), round(b, 3)])
            per_utt.append({"start": round(a, 3), "end": round(b, 3), "speaker": u["speaker"], "emotion": u["emotion"]})
            pcm_parts += [pcm, gap]
            t = b + GAP_S
        if len(per_utt) < min_utts:
            continue
        wav = out_dir / f"{split}_dia{did}.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(np.concatenate(pcm_parts).tobytes())
        emo = Counter(u["emotion"] for u in per_utt)
        dominant = emo.most_common(1)[0][0]
        # The coached "self" is the speaker with the most utterances — arbitrary
        # but deterministic, and every speaker's spans are kept regardless.
        self_spk = Counter(u["speaker"] for u in per_utt).most_common(1)[0][0]
        entries.append({
            "id": f"meld_{split}_dia{did}", "corpus": "MELD",
            "setting": f"sitcom dialogue — mostly {dominant}" + (" (heated)" if heated_spans else ""),
            "audio": f"dialogues/{wav.name}", "speakers": speakers, "self": self_spk,
            "heated_spans": heated_spans or None, "duration_s": round(t, 1),
            "utterances": per_utt, "emotion_counts": dict(emo),
        })
        if len(entries) % 25 == 0:
            print(f"  {len(entries)} dialogues built", flush=True)
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--max-dialogues", type=int, default=300)
    ap.add_argument("--min-utts", type=int, default=5)
    args = ap.parse_args()
    root = Path(args.root)
    if not glob.glob(str(root / "**/*.tar.gz"), recursive=True) and not clip_index(root):
        raise SystemExit(f"no MELD archive under {root} — download MELD.Raw.tar.gz first")
    extract_all(root)
    entries = build(root, args.max_dialogues, args.min_utts)
    (root / "heatmap_manifest.json").write_text(json.dumps(entries, indent=1))
    heated = sum(1 for e in entries if e["heated_spans"])
    hours = sum(e["duration_s"] for e in entries) / 3600
    print(f"\n{len(entries)} dialogues ({heated} with heated utterances), {hours:.1f} h -> {root/'heatmap_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
