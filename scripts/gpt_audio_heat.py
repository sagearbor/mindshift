#!/usr/bin/env python3
"""Benchmark OpenAI's gpt-audio family as a "how heated is this moment"
labeller against HUMAN conflict ratings — CONFER's ten raters, continuous
0..1000 conflict intensity on real televised Greek debates — and compare it
side by side with the WavLM tone model (`tone_id`, already in the codebase)
and raw loudness, using the exact protocol `scripts/heat_map.py`'s
`human_conflict_agreement()` established (so the numbers are comparable):

  within-clip  = MEDIAN of the per-clip Spearman correlation (gpt heat vs the
                 human per-window mean), one number per clip with >=10 scored
                 5 s windows.
  across-clip  = Spearman correlation of the CLIP-LEVEL MEANS (mean gpt heat
                 vs mean human conflict), one point per clip, all clips.

CONFER windows are the SAME 5 s windows already scored by the WavLM
reference (tmp/heat-map/reference_series/confer_*.json's `window_idx`), so
gpt-audio, WavLM and the human raters line up exactly on the same audio.

ENDPOINT NOTE: the task asked for the Responses API (POST /v1/responses,
input_audio content parts). Verified live 2026-09-20: this org's key gets
HTTP 400 "Audio input is not available" from /v1/responses for gpt-audio,
gpt-audio-mini AND gpt-audio-1.5 — audio input is gated to a different
endpoint for this account. POST /v1/chat/completions with
`modalities: ["text"]` (the same endpoint scripts/audio_tone_probe.py already
uses successfully in this repo) works and returns full usage token detail
(prompt_tokens_details.audio_tokens), so that is what this script uses. See
docs/research/2026-09-20-gpt-audio-labeller.md for the verbatim error.

USAGE
    python3 scripts/gpt_audio_heat.py --stage confer --models mini
    python3 scripts/gpt_audio_heat.py --stage confer --models mini,full
    python3 scripts/gpt_audio_heat.py --stage cremad --models mini,full
    python3 scripts/gpt_audio_heat.py --stage report            # no API calls

Every response is cached to disk under MAIN_REPO/tmp/heat-map/gpt_audio/ —
re-running never re-pays for a window already scored. A running dollar total
is tracked in a small ledger file (spend_ledger.json) so cumulative spend
across separate invocations tonight is respected against the $20 cap.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np

# ---------------------------------------------------------------------------
# Normally this resolves relative to the script (like every other tool in
# scripts/). Tonight it runs from an isolated git worktree whose gitignored
# tmp/ is NOT populated — corpora, caches and .env live only in the main repo
# checkout — so fall back to that absolute path when the relative one has no
# tmp/corpora. Once this file lives in the main checkout (post-merge) or runs
# there directly, the relative branch is the one that fires.
# ---------------------------------------------------------------------------
_REPO_RELATIVE = Path(__file__).resolve().parent.parent
_MAIN_REPO_FALLBACK = Path("/Users/sagearbor/projects/githubs/mindshift")
MAIN_REPO = _REPO_RELATIVE if (_REPO_RELATIVE / "tmp/corpora").exists() else _MAIN_REPO_FALLBACK
ENV_FILE = MAIN_REPO / ".env"
CONFER_ROOT = MAIN_REPO / "tmp/corpora/confer"
REF_SERIES_DIR = MAIN_REPO / "tmp/heat-map/reference_series"
HEAT_MAP_JSON = MAIN_REPO / "tmp/heat-map/heat_map.json"
CREMAD_DIR = MAIN_REPO / "tmp/corpora/cremad/AudioWAV"
CACHE_DIR = MAIN_REPO / "tmp/heat-map/gpt_audio"
LEDGER_PATH = CACHE_DIR / "spend_ledger.json"
BUDGET_CAP_USD = 20.0
BUDGET_WARN_USD = 18.0  # stop issuing NEW calls once cumulative spend crosses this

WIN_S = 5.0
CHAT_URL = "https://api.openai.com/v1/chat/completions"
TIMEOUT_SEC = 60.0

MODEL_IDS = {"mini": "gpt-audio-mini", "full": "gpt-audio"}
# $ / 1,000,000 tokens, verified 2026-09-20 (developers.openai.com/api/docs/pricing)
PRICING = {
    "gpt-audio-mini": {"audio_in": 10.00, "text_in": 0.60, "text_out": 2.40},
    "gpt-audio": {"audio_in": 32.00, "text_in": 2.50, "text_out": 10.00},
}

SYSTEM_PROMPT = (
    "You are an acoustic tone analyzer. The speech in this clip may be in a "
    "language you do not understand (Greek). You MUST ignore word meaning "
    "entirely — judge ONLY from tone of voice: pitch, loudness, pace, "
    "tremor, harshness, overlapping voices. Do not try to translate or "
    "guess the words.\n\n"
    "Return ONLY strict JSON, no markdown fences, no other text:\n"
    '{"heat": 0-100, "arousal": 0-100, "valence": -100..100, '
    '"angry_vs_excited": -100..100, "one_line_reason": "short phrase"}\n\n'
    "heat = how heated/conflictual this moment sounds (0 calm, 100 an "
    "extremely heated argument). arousal = vocal energy/intensity (0 low, "
    "100 high). valence = how negative vs positive the tone sounds (-100 "
    "very hostile/negative, +100 very warm/positive). angry_vs_excited "
    "disambiguates high-arousal moments that loudness alone confuses: -100 "
    "clearly angry/hostile, +100 clearly excited/joyful/enthusiastic, 0 "
    "ambiguous or neutral."
)
USER_TEXT = (
    "Rate this audio clip's vocal tone only, ignoring words/language. "
    "Judge purely from how it sounds."
)

# CREMA-D clips are ~2.5 s (much shorter than CONFER's 5 s windows) and are
# English, not Greek — the CONFER prompt's "language you do not understand
# (Greek)" framing is nonsensical for them. Live-tested 2026-09-20: on these
# short clips, `gpt-audio` (the FULL model only — mini was fine) reliably
# responded "please provide audio" instead of scoring it — i.e. it silently
# ignored the audio it was sent. Putting the audio content part FIRST and
# stating explicitly that it's attached fixed 100% of a small sample. mini
# never showed this failure mode either way; used for both models on
# CREMA-D for one consistent prompt/ordering.
CREMAD_SYSTEM_PROMPT = (
    "You are an acoustic tone analyzer receiving an attached audio clip in "
    "this same message. Judge ONLY from tone of voice: pitch, loudness, "
    "pace, tremor, harshness. Ignore any words.\n\n"
    "Return ONLY strict JSON, no markdown fences, no other text:\n"
    '{"heat": 0-100, "arousal": 0-100, "valence": -100..100, '
    '"angry_vs_excited": -100..100, "one_line_reason": "short phrase"}\n\n'
    "heat = how heated/angry this voice sounds (0 calm, 100 furious). "
    "arousal = vocal energy/intensity (0 low, 100 high). valence = how "
    "negative vs positive the tone sounds (-100 very hostile/negative, "
    "+100 very warm/positive). angry_vs_excited disambiguates high-arousal "
    "moments that loudness alone confuses: -100 clearly angry/hostile, "
    "+100 clearly excited/joyful/enthusiastic, 0 ambiguous or neutral."
)
CREMAD_USER_TEXT = (
    "The audio clip is attached to this message right now. Listen to it "
    "and rate its vocal tone only, ignoring words/language. Respond "
    "immediately with ONLY the JSON object."
)


# ---------------------------------------------------------------------------
# .env reader — mirrors scripts/audio_tone_probe.py::read_env exactly, just
# pointed at the main repo's .env (this worktree carries none).
# ---------------------------------------------------------------------------
def read_env(key: str) -> str:
    if not ENV_FILE.is_file():
        return ""
    pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(key)}\s*=(.*)$")
    value = ""
    for line in ENV_FILE.read_text().splitlines():
        m = pattern.match(line)
        if m:
            value = m.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


# ---------------------------------------------------------------------------
# Spend ledger — persisted across invocations so the $20 cap is honoured
# even if this script is run several times tonight.
# ---------------------------------------------------------------------------
def load_ledger() -> dict:
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text())
    return {"total_usd": 0.0, "calls": 0, "by_model": {}}


def save_ledger(ledger: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=1))


def record_spend(ledger: dict, model: str, cost: float) -> None:
    ledger["total_usd"] = round(ledger.get("total_usd", 0.0) + cost, 6)
    ledger["calls"] = ledger.get("calls", 0) + 1
    by = ledger.setdefault("by_model", {})
    by[model] = round(by.get(model, 0.0) + cost, 6)
    save_ledger(ledger)


# ---------------------------------------------------------------------------
# Audio slicing
# ---------------------------------------------------------------------------
def read_wav_mono16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise SystemExit(f"ERROR: {path} is {sampwidth * 8}-bit; expected 16-bit PCM.")
    samples = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    return samples, rate


def slice_to_wav_b64(samples: np.ndarray, rate: int, start_s: float, end_s: float) -> str | None:
    start_sample = max(0, int(round(start_s * rate)))
    end_sample = min(len(samples), int(round(end_s * rate)))
    if end_sample <= start_sample:
        return None
    clip = samples[start_sample:end_sample]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(clip.astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# OpenAI call (chat/completions — see module docstring for why, not Responses)
# ---------------------------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _extract_json_obj(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    return text[start:end + 1]


def parse_gpt_json(content: str) -> dict:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty response content")
    stripped = _strip_code_fences(content)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        data = json.loads(_extract_json_obj(stripped))
    if not isinstance(data, dict):
        raise ValueError("response JSON is not an object")
    out: dict[str, Any] = {}
    out["heat"] = max(0.0, min(100.0, float(data["heat"])))
    out["arousal"] = max(0.0, min(100.0, float(data["arousal"])))
    out["valence"] = max(-100.0, min(100.0, float(data["valence"])))
    out["angry_vs_excited"] = max(-100.0, min(100.0, float(data["angry_vs_excited"])))
    out["one_line_reason"] = str(data.get("one_line_reason", ""))[:200]
    return out


def call_gpt_audio(client: httpx.Client, api_key: str, model: str, b64_wav: str,
                    system_prompt: str = SYSTEM_PROMPT, user_text: str = USER_TEXT,
                    audio_first: bool = False) -> dict:
    """Returns {"parsed": {...} | None, "error": str | None, "usage": {...} | None,
    "cost_usd": float}. audio_first=True puts the input_audio content part before
    the text part — see CREMAD_SYSTEM_PROMPT's note on why that matters for
    short clips on the full gpt-audio model."""
    audio_part = {"type": "input_audio", "input_audio": {"data": b64_wav, "format": "wav"}}
    text_part = {"type": "text", "text": user_text}
    content = [audio_part, text_part] if audio_first else [text_part, audio_part]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "modalities": ["text"],
        "max_completion_tokens": 300,
    }
    try:
        resp = client.post(CHAT_URL, headers={"Authorization": f"Bearer {api_key}"},
                            json=payload, timeout=TIMEOUT_SEC)
    except httpx.HTTPError as e:
        return {"parsed": None, "error": f"request failed: {e}", "usage": None, "cost_usd": 0.0}

    if resp.status_code != 200:
        return {"parsed": None, "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
                "usage": None, "cost_usd": 0.0}

    body = resp.json()
    usage = body.get("usage") or {}
    prices = PRICING[model]
    audio_in = usage.get("prompt_tokens_details", {}).get("audio_tokens", 0)
    text_in = usage.get("prompt_tokens_details", {}).get("text_tokens", 0)
    text_out = usage.get("completion_tokens_details", {}).get("text_tokens",
                                                                usage.get("completion_tokens", 0))
    cost = (audio_in * prices["audio_in"] + text_in * prices["text_in"]
            + text_out * prices["text_out"]) / 1_000_000.0

    try:
        content = body["choices"][0]["message"]["content"]
        parsed = parse_gpt_json(content)
    except Exception as e:
        return {"parsed": None, "error": f"parse failure: {e}", "usage": usage, "cost_usd": cost}

    return {"parsed": parsed, "error": None, "usage": usage, "cost_usd": cost}


# ---------------------------------------------------------------------------
# CONFER stage
# ---------------------------------------------------------------------------
def confer_manifest() -> dict[str, dict]:
    entries = json.loads((CONFER_ROOT / "heatmap_manifest.json").read_text())
    return {e["id"]: e for e in entries}


def confer_windows() -> dict[str, dict]:
    """clip id ('confer_...') -> {"win_s", "window_idx"} from the cached
    WavLM reference series — the SAME windows, so all three signals line up."""
    out = {}
    for p in sorted(REF_SERIES_DIR.glob("confer_*.json")):
        ref = json.loads(p.read_text())
        out[p.stem] = {"win_s": float(ref["win_s"]), "window_idx": list(ref["window_idx"])}
    return out


def run_confer(models: list[str], api_key: str, ledger: dict) -> None:
    manifest = confer_manifest()
    windows = confer_windows()
    if not windows:
        print("  no cached reference_series/confer_*.json found — nothing to score")
        return

    with httpx.Client() as client:
        for model_key in models:
            model = MODEL_IDS[model_key]
            out_dir = CACHE_DIR / model
            total_new_cost = 0.0
            n_new, n_cached, n_err = 0, 0, 0
            for clip_id, w in windows.items():
                entry = manifest.get(clip_id)
                if entry is None:
                    continue
                wav_path = CONFER_ROOT / entry["audio"]
                samples = None  # loaded lazily, only if a window is actually missing
                clip_dir = out_dir / clip_id
                for idx in w["window_idx"]:
                    cache_file = clip_dir / f"{idx}.json"
                    if cache_file.exists():
                        n_cached += 1
                        continue
                    if ledger.get("total_usd", 0.0) >= BUDGET_WARN_USD:
                        print(f"  STOPPING: cumulative spend ${ledger['total_usd']:.2f} "
                              f"has reached the ${BUDGET_WARN_USD} guard (cap ${BUDGET_CAP_USD}).")
                        return
                    if samples is None:
                        samples, rate = read_wav_mono16(wav_path)
                    start_s, end_s = idx * w["win_s"], (idx + 1) * w["win_s"]
                    b64 = slice_to_wav_b64(samples, rate, start_s, end_s)
                    if b64 is None:
                        continue
                    result = call_gpt_audio(client, api_key, model, b64)
                    if result["error"]:
                        n_err += 1
                        print(f"  [{model_key}] {clip_id}#{idx}: ERROR {result['error'][:120]}")
                    else:
                        n_new += 1
                    total_new_cost += result["cost_usd"]
                    record_spend(ledger, model, result["cost_usd"])
                    clip_dir.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps({
                        "clip_id": clip_id, "window_idx": idx, "model": model,
                        "start_s": start_s, "end_s": end_s,
                        "gpt": result["parsed"], "error": result["error"],
                        "usage": result["usage"], "cost_usd": round(result["cost_usd"], 6),
                    }, indent=1))
                    print(f"  [{model_key}] {clip_id}#{idx}: "
                          f"{'ok' if not result['error'] else 'ERR'} "
                          f"cost=${result['cost_usd']:.5f} running_total=${ledger['total_usd']:.3f}",
                          flush=True)
            print(f"\n[{model_key}] CONFER done: {n_new} new calls (${total_new_cost:.3f}), "
                  f"{n_cached} already cached, {n_err} errors. "
                  f"Ledger total so far: ${ledger['total_usd']:.3f}\n")


# ---------------------------------------------------------------------------
# CREMA-D stage
# ---------------------------------------------------------------------------
def cremad_pick(n_per_class: int = 20, seed: int = 0) -> list[tuple[str, Path]]:
    rng = np.random.RandomState(seed)
    picks = []
    for label, code in (("angry", "_ANG_"), ("happy", "_HAP_")):
        files = sorted(p for p in CREMAD_DIR.glob(f"*{code}*.wav"))
        idx = rng.choice(len(files), size=min(n_per_class, len(files)), replace=False)
        idx.sort()
        picks.extend((label, files[i]) for i in idx)
    return picks


def run_cremad(models: list[str], api_key: str, ledger: dict, n_per_class: int = 20) -> None:
    picks = cremad_pick(n_per_class)
    with httpx.Client() as client:
        for model_key in models:
            model = MODEL_IDS[model_key]
            out_dir = CACHE_DIR / model / "cremad"
            n_new, n_err = 0, 0
            for label, wav_path in picks:
                cache_file = out_dir / f"{wav_path.stem}.json"
                if cache_file.exists():
                    continue
                if ledger.get("total_usd", 0.0) >= BUDGET_WARN_USD:
                    print(f"  STOPPING: cumulative spend ${ledger['total_usd']:.2f} "
                          f"has reached the ${BUDGET_WARN_USD} guard.")
                    return
                samples, rate = read_wav_mono16(wav_path)
                b64 = slice_to_wav_b64(samples, rate, 0.0, len(samples) / rate)
                if b64 is None:
                    continue
                result = call_gpt_audio(client, api_key, model, b64,
                                         system_prompt=CREMAD_SYSTEM_PROMPT,
                                         user_text=CREMAD_USER_TEXT, audio_first=True)
                if result["error"]:
                    n_err += 1
                    print(f"  [{model_key}] {wav_path.name}: ERROR {result['error'][:120]}")
                else:
                    n_new += 1
                record_spend(ledger, model, result["cost_usd"])
                out_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps({
                    "file": wav_path.name, "label": label, "model": model,
                    "gpt": result["parsed"], "error": result["error"],
                    "usage": result["usage"], "cost_usd": round(result["cost_usd"], 6),
                }, indent=1))
                print(f"  [{model_key}] {wav_path.name} ({label}): "
                      f"{'ok' if not result['error'] else 'ERR'} "
                      f"running_total=${ledger['total_usd']:.3f}", flush=True)
            print(f"\n[{model_key}] CREMA-D done: {n_new} new calls, {n_err} errors.\n")


# ---------------------------------------------------------------------------
# Analysis — replicates scripts/heat_map.py::human_conflict_agreement exactly
# ---------------------------------------------------------------------------
def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def auc_from_scores(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """AUC via the Mann-Whitney U statistic (rank-based, no sklearn needed):
    P(a random positive score > a random negative score), ties counted half."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


def analyze_confer(model_key: str) -> dict:
    model = MODEL_IDS[model_key]
    manifest = confer_manifest()
    out_dir = CACHE_DIR / model
    per_clip_spearman_heat = []
    per_clip_spearman_arousal = []
    clip_mean_heat, clip_mean_human = [], []
    clip_mean_arousal = []
    all_hit, all_miss_pos, all_false_pos, all_neg = 0, 0, 0, 0
    n_windows_total = 0

    for clip_dir in sorted(out_dir.glob("confer_*")):
        if not clip_dir.is_dir():
            continue
        clip_id = clip_dir.name
        entry = manifest.get(clip_id)
        if entry is None:
            continue
        human = np.asarray(entry["conflict_per_second"], dtype=float)
        heats, arousals, humans_w = [], [], []
        for f in sorted(clip_dir.glob("*.json"), key=lambda p: int(p.stem)):
            rec = json.loads(f.read_text())
            if rec.get("gpt") is None:
                continue
            idx = rec["window_idx"]
            w = 5  # WIN_S
            if (idx + 1) * w > len(human):
                continue
            hp = float(np.mean(human[idx * w:(idx + 1) * w]))
            heats.append(rec["gpt"]["heat"])
            arousals.append(rec["gpt"]["arousal"])
            humans_w.append(hp)
        if len(heats) < 3:
            continue
        heats_a, arousals_a, humans_a = np.array(heats), np.array(arousals), np.array(humans_w)
        n_windows_total += len(heats_a)

        for h, hp in zip(heats_a, humans_a):
            if hp >= 400:
                all_hit += 1 if h >= 50 else 0
                all_miss_pos += 1
            elif hp < 200:
                all_false_pos += 1 if h >= 50 else 0
                all_neg += 1

        if len(heats_a) >= 10:
            sp_h = spearman(heats_a, humans_a)
            sp_a = spearman(arousals_a, humans_a)
            if sp_h is not None:
                per_clip_spearman_heat.append(sp_h)
            if sp_a is not None:
                per_clip_spearman_arousal.append(sp_a)

        clip_mean_heat.append(float(heats_a.mean()))
        clip_mean_arousal.append(float(arousals_a.mean()))
        clip_mean_human.append(float(humans_a.mean()))

    across_heat = spearman(np.array(clip_mean_heat), np.array(clip_mean_human)) \
        if len(clip_mean_heat) >= 3 else None
    across_arousal = spearman(np.array(clip_mean_arousal), np.array(clip_mean_human)) \
        if len(clip_mean_arousal) >= 3 else None

    return {
        "model": model,
        "n_clips_scored": len(clip_mean_heat),
        "n_windows_scored": n_windows_total,
        "within_clip_heat_median": (round(float(np.median(per_clip_spearman_heat)), 3)
                                     if per_clip_spearman_heat else None),
        "within_clip_heat_n": len(per_clip_spearman_heat),
        "within_clip_arousal_median": (round(float(np.median(per_clip_spearman_arousal)), 3)
                                        if per_clip_spearman_arousal else None),
        "across_clip_heat": round(across_heat, 3) if across_heat is not None else None,
        "across_clip_arousal": round(across_arousal, 3) if across_arousal is not None else None,
        "heated_hit_rate": round(all_hit / all_miss_pos, 3) if all_miss_pos else None,
        "heated_hit_n": all_miss_pos,
        "false_positive_rate": round(all_false_pos / all_neg, 3) if all_neg else None,
        "false_positive_n": all_neg,
    }


def analyze_cremad(model_key: str) -> dict:
    model = MODEL_IDS[model_key]
    out_dir = CACHE_DIR / model / "cremad"
    heat, angry_vs_excited, labels = [], [], []
    for f in sorted(out_dir.glob("*.json")):
        rec = json.loads(f.read_text())
        if rec.get("gpt") is None:
            continue
        heat.append(rec["gpt"]["heat"])
        angry_vs_excited.append(rec["gpt"]["angry_vs_excited"])
        labels.append(1 if rec["label"] == "angry" else 0)
    if not labels:
        return {"model": model, "n": 0}
    heat, angry_vs_excited, labels = np.array(heat), np.array(angry_vs_excited), np.array(labels)
    pos_h, neg_h = heat[labels == 1], heat[labels == 0]
    # angry_vs_excited: MORE NEGATIVE = more angry, so flip sign for the "angry=positive" AUC convention
    pos_a, neg_a = -angry_vs_excited[labels == 1], -angry_vs_excited[labels == 0]
    return {
        "model": model, "n": len(labels), "n_angry": int(labels.sum()), "n_happy": int((1 - labels).sum()),
        "auc_heat": round(auc_from_scores(pos_h, neg_h), 3) if auc_from_scores(pos_h, neg_h) is not None else None,
        "auc_angry_vs_excited": round(auc_from_scores(pos_a, neg_a), 3) if auc_from_scores(pos_a, neg_a) is not None else None,
    }


def wavlm_and_loudness_confer() -> dict:
    """The existing numbers from heat_map.json, computed by the SAME
    within/across protocol, for the side-by-side table."""
    if not HEAT_MAP_JSON.exists():
        return {}
    rows = [r for r in json.loads(HEAT_MAP_JSON.read_text()) if r["corpus"] == "CONFER"]
    wavlm_within = [r["reference_vs_human_conflict"] for r in rows if r.get("reference_vs_human_conflict") is not None]
    loud_within = [r["agreement_with_human_conflict"] for r in rows if r.get("agreement_with_human_conflict") is not None]
    wavlm_pairs = [(r["ref_arousal_mean"], r["human_conflict_mean"]) for r in rows
                   if r.get("ref_arousal_mean") is not None and r.get("human_conflict_mean") is not None]
    loud_pairs = [(r["db_sd"], r["human_conflict_mean"]) for r in rows
                  if r.get("db_sd") is not None and r.get("human_conflict_mean") is not None]
    wa = np.array([p[0] for p in wavlm_pairs]); wh = np.array([p[1] for p in wavlm_pairs])
    la = np.array([p[0] for p in loud_pairs]); lh = np.array([p[1] for p in loud_pairs])
    return {
        "wavlm_within_clip_median": round(float(np.median(wavlm_within)), 3) if wavlm_within else None,
        "wavlm_across_clip": round(spearman(wa, wh), 3) if len(wa) >= 3 else None,
        "loudness_within_clip_median": round(float(np.median(loud_within)), 3) if loud_within else None,
        "loudness_across_clip": round(spearman(la, lh), 3) if len(la) >= 3 else None,
    }


def print_report(models: list[str]) -> None:
    print("\n=== CONFER: gpt-audio heat vs human conflict ratings ===")
    ref = wavlm_and_loudness_confer()
    print(f"  WavLM arousal    within-clip median {ref.get('wavlm_within_clip_median')}   "
          f"across-clip {ref.get('wavlm_across_clip')}")
    print(f"  loudness (db_sd) within-clip median {ref.get('loudness_within_clip_median')}   "
          f"across-clip {ref.get('loudness_across_clip')}")
    for mk in models:
        r = analyze_confer(mk)
        print(f"\n  [{r['model']}] {r['n_clips_scored']} clips, {r['n_windows_scored']} windows scored")
        print(f"    heat     within-clip median {r['within_clip_heat_median']} (n={r['within_clip_heat_n']})   "
              f"across-clip {r['across_clip_heat']}")
        print(f"    arousal  within-clip median {r['within_clip_arousal_median']}")
        print(f"    heated-hit-rate (human>=400 -> gpt heat>=50): {r['heated_hit_rate']} (n={r['heated_hit_n']})")
        print(f"    false-positive-rate (human<200 -> gpt heat>=50): {r['false_positive_rate']} (n={r['false_positive_n']})")

    print("\n=== CREMA-D sanity check: angry vs happy ===")
    for mk in models:
        r = analyze_cremad(mk)
        print(f"  [{r['model']}] n={r.get('n')} (angry {r.get('n_angry')}, happy {r.get('n_happy')})   "
              f"AUC(heat)={r.get('auc_heat')}   AUC(angry_vs_excited)={r.get('auc_angry_vs_excited')}")

    ledger = load_ledger()
    print(f"\nTotal spend so far: ${ledger.get('total_usd', 0.0):.3f}  ({ledger.get('calls', 0)} calls)")
    print(f"By model: {ledger.get('by_model', {})}")

    # Dump a machine-readable summary alongside the raw cache.
    summary = {
        "reference": ref,
        "confer": {mk: analyze_confer(mk) for mk in models},
        "cremad": {mk: analyze_cremad(mk) for mk in models},
        "spend": ledger,
    }
    (CACHE_DIR / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n-> {CACHE_DIR / 'summary.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["confer", "cremad", "report", "all"], default="report")
    ap.add_argument("--models", default="mini", help="comma-separated: mini,full")
    ap.add_argument("--cremad-n", type=int, default=20, help="clips per class (angry/happy)")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in MODEL_IDS:
            print(f"unknown model key '{m}' (use mini or full)", file=sys.stderr)
            return 1

    ledger = load_ledger()
    if ledger.get("total_usd", 0.0) >= BUDGET_WARN_USD and args.stage != "report":
        print(f"Cumulative spend ${ledger['total_usd']:.2f} already at/above the "
              f"${BUDGET_WARN_USD} guard (cap ${BUDGET_CAP_USD}) — refusing new API calls. "
              f"Use --stage report to see numbers from what's cached.", file=sys.stderr)
        return 1

    api_key = None
    if args.stage in ("confer", "cremad", "all"):
        api_key = read_env("OPENAI_API_KEY")
        if not api_key:
            print(f"OPENAI_API_KEY not set in {ENV_FILE}", file=sys.stderr)
            return 1

    if args.stage in ("confer", "all"):
        print(f"→ CONFER stage, models={models}")
        run_confer(models, api_key, ledger)
    if args.stage in ("cremad", "all"):
        print(f"→ CREMA-D stage, models={models}, n_per_class={args.cremad_n}")
        run_cremad(models, api_key, ledger, args.cremad_n)

    print_report(models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
