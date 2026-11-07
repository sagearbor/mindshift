"""Audio emotion labellers (API) under four conditions, with soft (probability) outputs.

Conditions
  audio_only  : audio, no words given (PRIMARY)
  text_only   : transcript only, no audio (CREMA-D sentences carry no emotion words)
  both        : audio + the true transcript
  contradict  : audio + a lexically CONTRADICTORY transcript (strong words of another emotion)
TTS leakage set: flat macOS `say` voice reading emotional sentences, audio_only.

Every call goes through ledger (cost log, cache, hard cap).  GPT outputs are EVALUATION
only, never training labels.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ledger
from cremad import EMOS, ROOT
from env import OUT, load_env

load_env()

INSTR = (
    "You label the emotional tone of a speaker's VOICE in a short audio clip. Judge HOW it is said "
    "(pitch, loudness, tempo, voice quality), not WHAT is said: the words may be unrelated to the "
    "emotion. Give a probability distribution over exactly these six labels: anger, disgust, fear, "
    "happy, neutral, sad (numbers sum to 1). Also rate arousal 1 (very calm) to 5 (very activated). "
    'Reply with JSON only: {"probs": {"anger": 0.0, "disgust": 0.0, "fear": 0.0, "happy": 0.0, '
    '"neutral": 0.0, "sad": 0.0}, "arousal": 3}'
)
INSTR_TEXT = (
    "No audio is available, only a transcript of one short utterance. Estimate the speaker's emotion "
    "from the transcript alone. Give a probability distribution over exactly these six labels: anger, "
    "disgust, fear, happy, neutral, sad (numbers sum to 1), and arousal 1-5. "
    'Reply with JSON only: {"probs": {"anger": 0.0, "disgust": 0.0, "fear": 0.0, "happy": 0.0, '
    '"neutral": 0.0, "sad": 0.0}, "arousal": 3}'
)
CONTRA = {
    "anger": "I am absolutely furious with you, this is outrageous and I won't stand for it!",
    "disgust": "Ugh, that is revolting, honestly it makes me sick to look at it.",
    "fear": "I'm terrified, please help me, I think someone is following me.",
    "happy": "This is the best day ever, I'm so thrilled and grateful!",
    "sad": "I miss her so much, I just feel empty and heartbroken.",
}
TEXT_EMOS = ["anger", "disgust", "fear", "happy", "sad"]


def contra_emotion(clip: str, crowd: str) -> str:
    opts = [e for e in TEXT_EMOS if e != crowd]
    return opts[int(ledger.cache_key(clip)[:8], 16) % len(opts)]


def prompt_for(cond: str, sentence: str | None) -> str:
    if cond == "text_only":
        return INSTR_TEXT + f'\nTranscript: "{sentence}"'
    if cond in ("both", "contradict"):
        return INSTR + f'\nTranscript of the clip (for reference only): "{sentence}"'
    return INSTR


def parse(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    p = d.get("probs") if isinstance(d.get("probs"), dict) else d
    vec = [max(0.0, float(p.get(e, 0) or 0)) for e in EMOS]
    s = sum(vec)
    if s <= 0:
        return None
    vec = [x / s for x in vec]
    return {"probs": vec, "label": EMOS[max(range(6), key=lambda i: vec[i])], "arousal": d.get("arousal")}


# ---------------- model backends ----------------
_gclient = None


def gemini(model: str, prompt: str, wav: bytes | None, task: str):
    global _gclient
    from google import genai
    from google.genai import types
    if _gclient is None:
        _gclient = genai.Client(vertexai=True, project="arborfam-hub", location="global")
    parts = [types.Part.from_text(text=prompt)]
    if wav is not None:
        parts.insert(0, types.Part.from_bytes(data=wav, mime_type="audio/wav"))
    cfg = types.GenerateContentConfig(response_mime_type="application/json", max_output_tokens=2048,
                                      thinking_config=types.ThinkingConfig(thinking_level="low"))
    r = _gclient.models.generate_content(model=model, contents=[types.Content(role="user", parts=parts)], config=cfg)
    u = r.usage_metadata
    audio_in = sum(d.token_count or 0 for d in (u.prompt_tokens_details or []) if str(d.modality).endswith("AUDIO"))
    text_in = (u.prompt_token_count or 0) - audio_in
    out = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
    ledger.record(task, model, text_in, audio_in, out)
    return r.text


_oclient = None


def openai_audio(model: str, prompt: str, wav: bytes | None, task: str):
    global _oclient
    import openai
    if _oclient is None:
        _oclient = openai.OpenAI()
    content = [{"type": "text", "text": prompt}]
    if wav is not None:
        # audio BEFORE the instruction: with text first gpt-audio-1.5 often answered as if no audio was attached
        content.insert(0, {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}})
    r = _oclient.chat.completions.create(model=model, modalities=["text"], messages=[{"role": "user", "content": content}],
                                         max_completion_tokens=300)
    u = r.usage
    audio_in = (u.prompt_tokens_details.audio_tokens or 0) if u.prompt_tokens_details else 0
    ledger.record(task, model, u.prompt_tokens - audio_in, audio_in, u.completion_tokens)
    return r.choices[0].message.content


BACKENDS = {"gemini-3.8-flash": gemini, "gemini-3.5-flash-lite": gemini, "gpt-audio-1.5": openai_audio}
EST = {"gemini-3.8-flash": 0.0015, "gemini-3.5-flash-lite": 0.0008, "gpt-audio-1.5": 0.004}


def label(model: str, cond: str, item_id: str, wav_path: Path | None, sentence: str | None, task: str) -> dict:
    wav = wav_path.read_bytes() if wav_path is not None else None
    prompt = prompt_for(cond, sentence)
    key = ledger.cache_key("emo-v2" if model.startswith("gpt") else "emo-v1", model, cond, prompt, wav or b"")
    hit = ledger.cache_get(key)
    if hit is not None:
        return hit
    ledger.check_budget(EST[model])
    raw, res = None, None
    for _ in range(2):
        try:
            raw = BACKENDS[model](model, prompt, wav, task)
        except ledger.BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001
            raw = f"ERROR {type(e).__name__}: {str(e)[:200]}"
        res = parse(raw)
        if res is not None:
            break
    out = {"id": item_id, "model": model, "cond": cond, "raw": (raw or "")[:400], **(res or {"probs": None, "label": None})}
    if res is not None:
        ledger.cache_put(key, out)
    return out


# ---------------- TTS leakage set ----------------
TTS_SENTENCES = {
    "anger": ["I am so angry at you right now, this is unacceptable!", "Get out! I hate this, you ruined everything!"],
    "disgust": ["That smell is disgusting, I think I'm going to throw up.", "Gross, there are maggots all over the food."],
    "fear": ["Help! Somebody is breaking into the house, I'm so scared!", "Please don't hurt me, I'm terrified."],
    "happy": ["Yes! We won! I'm so happy I could dance!", "I love you all, this is the most wonderful surprise!"],
    "sad": ["My dog died this morning and I can't stop crying.", "Everyone left, and I feel so lonely and hopeless."],
    "neutral": ["The train leaves at nine fifteen from platform two.", "Please put the forms in the grey tray by the door."],
}
TTS_VOICES = ["Samantha", "Fred"]


def build_tts() -> list[dict]:
    d = OUT / "tts"
    d.mkdir(parents=True, exist_ok=True)
    items = []
    for emo, sents in TTS_SENTENCES.items():
        for i, s in enumerate(sents):
            for v in TTS_VOICES:
                stem = f"tts_{emo}_{i}_{v}"
                wav = d / f"{stem}.wav"
                if not wav.exists():
                    aiff = d / f"{stem}.aiff"
                    subprocess.run(["say", "-v", v, "-o", str(aiff), s], check=True, timeout=60)
                    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
                                   check=True, timeout=60)
                    aiff.unlink()
                items.append({"id": stem, "word_emotion": emo, "voice": v, "sentence": s, "wav": str(wav)})
    return items


# ---------------- runner ----------------
def run(models: list[str], conds: list[str], per_class: int | None, workers: int) -> None:
    sample = json.loads((OUT / "cremad_sample.json").read_text())["items"]
    if per_class:
        keep, seen = [], {}
        for s in sample:
            seen[s["crowd_majority"]] = seen.get(s["crowd_majority"], 0) + 1
            if seen[s["crowd_majority"]] <= per_class:
                keep.append(s)
        sample = keep
    jobs = []
    for m in models:
        for c in conds:
            if c == "tts":
                for t in build_tts():
                    jobs.append((m, "audio_only", t["id"], Path(t["wav"]), None, "emo-tts"))
            elif c == "text_only":
                for sid, sent in {s["clip"].split("_")[1]: s["sentence"] for s in sample}.items():
                    jobs.append((m, c, f"text_{sid}", None, sent, "emo-text"))
            else:
                for s in sample:
                    sent = CONTRA[contra_emotion(s["clip"], s["crowd_majority"])] if c == "contradict" else s["sentence"]
                    jobs.append((m, c, s["clip"], ROOT / "AudioWAV" / f"{s['clip']}.wav", sent, f"emo-{c}"))
    print(f"{len(jobs)} jobs; spent so far ${ledger.spent():.3f}", flush=True)
    res_path = OUT / "emotion_results.jsonl"
    done = 0
    with ThreadPoolExecutor(workers) as ex, open(res_path, "a") as f:
        futs = [ex.submit(label, *j) for j in jobs]
        for fu in as_completed(futs):
            try:
                r = fu.result()
            except ledger.BudgetExceeded as e:
                print("BUDGET STOP:", e, flush=True)
                ex.shutdown(cancel_futures=True)
                break
            f.write(json.dumps(r) + "\n")
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} ${ledger.spent():.3f}", flush=True)
    print(f"finished {done}; total spend ${ledger.spent():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--conds", nargs="+", default=["audio_only"])
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run(a.models, a.conds, a.per_class, a.workers)
    sys.exit(0)
