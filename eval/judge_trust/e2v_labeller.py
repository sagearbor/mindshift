"""emotion2vec+ large as an audio-only, $0, local labeller/baseline (EVAL ONLY).

Licence: FunASR MODEL_LICENSE grants use/copy/modify/share with attribution but also says
"provided for reference and learning purposes only" -> commercial labelling status is
ambiguous, so this is used as an evaluation baseline only, never to make training labels.
Caveat: emotion2vec+ was trained on pseudo-labelled data seeded from public emotion corpora
that may include CREMA-D, so its CREMA-D score may be optimistic (possible train/test overlap).

Run with tmp/judge-trust-20261105/venv-e2v/bin/python e2v_labeller.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from env import MAIN, OUT  # noqa: E402

ROOT = MAIN / "tmp/corpora/cremad"
EMOS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
# emotion2vec+ 9-way head order
E2V = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]
MAP = {"angry": "anger", "disgusted": "disgust", "fearful": "fear", "happy": "happy", "neutral": "neutral", "sad": "sad"}


def main() -> None:
    from funasr import AutoModel
    model = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", disable_update=True)
    sample = json.loads((OUT / "cremad_sample.json").read_text())["items"]
    jobs = [(s["clip"], ROOT / "AudioWAV" / f"{s['clip']}.wav") for s in sample]
    tts = sorted((OUT / "tts").glob("tts_*.wav"))
    jobs += [(p.stem, p) for p in tts]
    out = open(OUT / "emotion_results_e2v.jsonl", "w")
    for i, (cid, wav) in enumerate(jobs):
        r = model.generate(str(wav), granularity="utterance", extract_embedding=False)[0]
        labs = [l.split("/")[-1] for l in r["labels"]]
        sc = dict(zip(labs, r["scores"]))
        vec = [float(sc.get(k, 0.0)) for k in ("angry", "disgusted", "fearful", "happy", "neutral", "sad")]
        s = sum(vec) or 1.0
        vec = [x / s for x in vec]
        out.write(json.dumps({"id": cid, "model": "emotion2vec-plus-large", "cond": "audio_only", "probs": vec,
                              "label": EMOS[max(range(6), key=lambda j: vec[j])],
                              "mass_outside_6": round(1 - sum(float(sc.get(k, 0)) for k in MAP), 4)}) + "\n")
        if i % 50 == 0:
            print(i, len(jobs), flush=True)
    out.close()
    print("done", len(jobs))


if __name__ == "__main__":
    main()
