#!/usr/bin/env python3
"""Pilot the phase-2 "audio-native tone" engine.

For each turn in a prerecorded test conversation, send the ACTUAL AUDIO clip
to OpenAI's audio-input model (gpt-audio-1.5, falling back to gpt-audio) and
ask it to rate the speaker's *perceived* vocal affect — valence, arousal,
dominance, and an emotion-probability distribution — from the audio alone.
That rating is then compared against:

  * the `scripted_emotion` the fixture asked for (always present), and
  * the `expected` physics ground truth (present only for fixtures generated
    by scripts/make_test_recording.py — the DSP-modulated Aura fixture; the
    OpenAI-acted fixtures from scripts/make_test_recording_openai.py
    deliberately carry no physical ground truth, only `scripted_emotion` +
    `instruction`).

This is a pilot/comparison tool, not a pipeline component: it never
fabricates a rating. A turn whose model response can't be parsed into the
expected shape is recorded as {"error": "..."} — honestly — rather than a
made-up number.

USAGE
    python3 scripts/audio_tone_probe.py [--wav PATH] [--meta PATH] [--force]

Defaults: --wav tmp/test_recording.wav, --meta tmp/test_recording_meta.json
(the DSP-modulated Aura fixture). Point --wav/--meta at one of the
openai/gptaudio fixture pairs (e.g. tmp/test_recording_gptaudio.wav +
tmp/test_recording_gptaudio_meta.json) to probe the acted fixture instead.

REQUIREMENTS
    OPENAI_API_KEY in the repo-root .env (or exported in the shell). No key,
    no mock output: exits 1 with a clear message instead of faking ratings.

OUTPUT
    tmp/<wavname>_tone_probe.json — per turn {index, speaker,
    scripted_emotion, expected (if present in meta), gpt: <parsed rating or
    {"error": ...}>, model_used}. Also prints a compact comparison table and
    a plain-language agreement summary for the emotionally-scripted turns
    (shout / cold-contempt / sad / scared / calm) whose acoustic signature we
    can sanity-check against the rating.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import wave
from pathlib import Path
from typing import Any

import httpx
import numpy as np

# ---------------------------------------------------------------------------
# Repo-root resolution + .env reader — mirrors scripts/make_test_recording_
# openai.py's read_env() (itself mirroring scripts/deploy_cloudrun.sh) so
# secrets are handled identically everywhere in this repo: only the one key
# we need is parsed (no arbitrary .env execution), a real exported env var
# always wins over the file, and the value is never printed.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ENV_FILE = REPO_ROOT / ".env"


def read_env(key: str) -> str:
    """Last matching `KEY=VALUE` (or `export KEY=VALUE`) line in .env, or ""."""
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
# Constants
# ---------------------------------------------------------------------------
DEFAULT_WAV = REPO_ROOT / "tmp" / "test_recording.wav"
DEFAULT_META = REPO_ROOT / "tmp" / "test_recording_meta.json"

CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
MODELS = ("gpt-audio-1.5", "gpt-audio")  # pinned, then graceful fallback
TIMEOUT_SEC = 60.0
MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_GAP_SEC = 0.4  # matches make_test_recording*.py's silence gap

EMOTION_KEYS = ("neutral", "joy", "sadness", "anger", "fear", "frustration")

USER_TEXT_PROMPT = "Analyze only the audible properties of this speaker's voice."

SYSTEM_PROMPT = (
    "You are an acoustic vocal-affect analyzer. Estimate the speaker's "
    "PERCEIVED vocal affect from this clip — how the voice sounds — not "
    "their internal emotional state. Consider only audible vocal "
    "properties: pitch, intensity, rhythm, rate, pauses, and voice quality. "
    "Do not infer personality, diagnosis, or intent.\n\n"
    "Return ONLY JSON, no other text, no markdown fences:\n"
    "{\n"
    '  "valence": 0-1,\n'
    '  "arousal": 0-1,\n'
    '  "dominance": 0-1,\n'
    '  "emotions": {"neutral": p, "joy": p, "sadness": p, "anger": p, '
    '"fear": p, "frustration": p},\n'
    "  // emotion probabilities should sum to ~1\n"
    '  "confidence": 0-1,\n'
    '  "evidence": ["up to 3 short acoustic observations"],\n'
    '  "insufficient_audio": true|false\n'
    "}\n"
)


# ---------------------------------------------------------------------------
# Meta / WAV loading
# ---------------------------------------------------------------------------

def load_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text())


def build_turn_windows(meta: dict) -> list[dict]:
    """meta["turns"] -> list of {speaker, text, scripted_emotion, expected,
    start_time, end_time}.

    Two meta shapes exist in this repo:
      * make_test_recording.py:        explicit start_time/end_time (+ "expected")
      * make_test_recording_openai.py: only duration_sec (+ silence_gap_sec at
        the top level) — boundaries are reconstructed in generation order.
    """
    turns = meta.get("turns", [])
    gap = float(meta.get("silence_gap_sec", DEFAULT_GAP_SEC))
    has_explicit_bounds = bool(turns) and "start_time" in turns[0] and "end_time" in turns[0]

    windows = []
    cursor = 0.0
    for t in turns:
        if has_explicit_bounds:
            start = float(t["start_time"])
            end = float(t["end_time"])
        else:
            dur = float(t.get("duration_sec", 0.0))
            start = cursor
            end = start + dur
            cursor = end + gap
        windows.append({
            "speaker": t.get("speaker"),
            "text": t.get("text"),
            "scripted_emotion": t.get("scripted_emotion"),
            "expected": t.get("expected"),  # None if absent (acted fixtures)
            "start_time": start,
            "end_time": end,
        })
    return windows


def read_wav_mono16(path: Path) -> tuple[np.ndarray, int]:
    """WAV file -> (mono int16 samples, sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise SystemExit(
            f"ERROR: {path} is {sampwidth * 8}-bit audio; expected 16-bit PCM.")
    samples = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    return samples, rate


def slice_to_wav_b64(samples: np.ndarray, rate: int,
                      start_time: float, end_time: float) -> str | None:
    """Slice [start_time, end_time) out of `samples`, re-encode as a small
    mono 16-bit WAV in memory, base64 it. None if the slice is empty."""
    start_sample = max(0, int(round(start_time * rate)))
    end_sample = min(len(samples), int(round(end_time * rate)))
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
# OpenAI call + lenient JSON parsing
# ---------------------------------------------------------------------------

def _is_model_not_found(resp: httpx.Response) -> bool:
    """True when the API says the requested model doesn't exist — the one
    error we recover from (by switching to the unpinned alias)."""
    if resp.status_code not in (400, 404):
        return False
    try:
        err = resp.json().get("error") or {}
    except Exception:  # non-JSON error body -> not the error we handle
        return False
    return (err.get("code") == "model_not_found"
            or "does not exist" in (err.get("message") or ""))


def call_gpt_audio(client: httpx.Client, api_key: str, model: str,
                    b64_wav: str) -> httpx.Response:
    return client.post(
        CHAT_COMPLETIONS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": USER_TEXT_PROMPT},
                    {"type": "input_audio",
                     "input_audio": {"data": b64_wav, "format": "wav"}},
                ]},
            ],
        },
        timeout=TIMEOUT_SEC,
    )


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _extract_json_obj(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    return text[start:end + 1]


def _clamp01(x: Any) -> float:
    return max(0.0, min(1.0, float(x)))


def parse_gpt_json(content: str) -> dict:
    """Lenient parse of the model's reply into the rating contract. Raises
    ValueError (caller records {"error": ...}) if the required shape isn't
    there; clamps out-of-range values rather than failing on those."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty response content")

    stripped = _strip_code_fences(content)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        data = json.loads(_extract_json_obj(stripped))
    if not isinstance(data, dict):
        raise ValueError("response JSON is not an object")

    required_scalar = ("valence", "arousal", "dominance", "confidence")
    for key in required_scalar:
        if key not in data:
            raise ValueError(f"missing required field '{key}'")
    out: dict[str, Any] = {k: _clamp01(data[k]) for k in required_scalar}

    emotions_raw = data.get("emotions")
    if not isinstance(emotions_raw, dict):
        raise ValueError("missing/invalid 'emotions' object")
    out["emotions"] = {k: _clamp01(emotions_raw.get(k, 0.0)) for k in EMOTION_KEYS}

    evidence_raw = data.get("evidence", [])
    if not isinstance(evidence_raw, list):
        evidence_raw = [evidence_raw]
    out["evidence"] = [str(e) for e in evidence_raw[:3]]

    insufficient = data.get("insufficient_audio", False)
    if isinstance(insufficient, str):
        insufficient = insufficient.strip().lower() in ("true", "1", "yes")
    out["insufficient_audio"] = bool(insufficient)

    return out


def process_turn(client: httpx.Client, api_key: str, resolved: dict,
                  b64_wav: str | None) -> tuple[dict, str | None, bool]:
    """Returns (gpt_result, model_used, is_api_failure). is_api_failure is
    True only for HTTP/network-level failures (drives the >=3-consecutive
    abort) — a 200 response that fails to parse is recorded as an honest
    per-turn error but does NOT count toward that abort."""
    if b64_wav is None:
        return ({"error": "empty audio slice — could not extract turn "
                           "boundaries from meta"}, None, False)

    model = resolved["model"] or MODELS[0]
    try:
        resp = call_gpt_audio(client, api_key, model, b64_wav)
    except httpx.HTTPError as e:
        return ({"error": f"request failed: {e}"}, model, True)

    if resolved["model"] is None and model != MODELS[1] and _is_model_not_found(resp):
        print(f"  note: model '{model}' not found — falling back to '{MODELS[1]}'")
        model = MODELS[1]
        try:
            resp = call_gpt_audio(client, api_key, model, b64_wav)
        except httpx.HTTPError as e:
            return ({"error": f"request failed: {e}"}, model, True)

    if resp.status_code != 200:
        return ({"error": f"HTTP {resp.status_code}: {resp.text[:300]}"}, model, True)

    resolved["model"] = model
    try:
        message_content = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        return ({"error": f"malformed response shape: {e}"}, model, False)

    try:
        parsed = parse_gpt_json(message_content)
    except Exception as e:
        return ({"error": f"parse failure: {e}"}, model, False)

    return parsed, model, False


# ---------------------------------------------------------------------------
# Comparison table + agreement summary
# ---------------------------------------------------------------------------

def _top_emotion(emotions: dict) -> tuple[str, float]:
    return max(emotions.items(), key=lambda kv: kv[1])


def _agreement_check(scripted: str, gpt: dict) -> str | None:
    """Plain match/mismatch line for scripted emotions with a known expected
    acoustic signature. None if `scripted` isn't one of the rule-covered
    categories (shout / cold-contempt / sad / scared / calm)."""
    if "error" in gpt:
        return None
    s = (scripted or "").lower()
    top, top_p = _top_emotion(gpt["emotions"])
    arousal, valence = gpt["arousal"], gpt["valence"]

    if "shout" in s:
        ok = arousal > 0.6 and top in ("anger", "frustration")
        label = "high arousal + anger/frustration mass"
    elif "cold" in s or "contempt" in s:
        ok = arousal < 0.55 and (top in ("anger", "frustration") or valence < 0.45)
        label = "low-mid arousal + anger/frustration mass or low valence"
    elif "sad" in s or "hurt" in s:
        ok = top == "sadness" and valence < 0.45
        label = "sadness mass + low valence"
    elif "scare" in s or "fear" in s:
        ok = (top == "fear" or gpt["emotions"]["fear"] > 0.3) and arousal > 0.55
        label = "fear mass + high arousal"
    elif "calm" in s:
        ok = top == "neutral" and 0.3 <= valence <= 0.75
        label = "neutral mass + mid valence"
    else:
        return None

    verdict = "consistent" if ok else "MISMATCH"
    return (f"top emotion {top} ({top_p:.2f}), arousal {arousal:.2f}, "
            f"valence {valence:.2f} — {verdict} (expected {label})")


def print_report(results: list[dict]) -> None:
    print(f"\n{'turn':>4} | {'scripted':<18} | {'gpt top-emotion (p)':<22} | "
          f"{'arousal':>7} | {'valence':>7} | {'confidence':>10}")
    print("-" * 84)
    for r in results:
        gpt = r["gpt"]
        scripted = r["scripted_emotion"] or ""
        if "error" in gpt:
            print(f"{r['index']:>4} | {scripted:<18} | ERROR: {gpt['error'][:55]}")
            continue
        top, top_p = _top_emotion(gpt["emotions"])
        top_str = f"{top} ({top_p:.2f})"
        print(f"{r['index']:>4} | {scripted:<18} | {top_str:<22} | "
              f"{gpt['arousal']:>7.2f} | {gpt['valence']:>7.2f} | "
              f"{gpt['confidence']:>10.2f}")

    print("\nAgreement on emotionally-scripted turns:")
    any_line = False
    for r in results:
        line = _agreement_check(r["scripted_emotion"], r["gpt"])
        if line:
            any_line = True
            print(f"  turn {r['index']} ({r['scripted_emotion']}): {line}")
    if not any_line:
        print("  (no turns produced a usable rating for a rule-covered category)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Probe OpenAI's audio-input model for perceived vocal "
                     "affect on each turn of a test recording, compared "
                     "against the scripted emotion and (where present) the "
                     "physics ground truth.")
    p.add_argument("--wav", type=Path, default=DEFAULT_WAV,
                    help=f"Path to the input WAV (default: {DEFAULT_WAV})")
    p.add_argument("--meta", type=Path, default=DEFAULT_META,
                    help=f"Path to the turn metadata JSON (default: {DEFAULT_META})")
    p.add_argument("--force", action="store_true",
                    help="Overwrite an existing probe JSON (default: skip if it exists)")
    args = p.parse_args()

    wav_path: Path = args.wav
    meta_path: Path = args.meta
    out_path = REPO_ROOT / "tmp" / f"{wav_path.stem}_tone_probe.json"

    # Skip-if-exists is checked first (no key or input files required), like
    # scripts/make_test_recording_openai.py.
    if out_path.exists() and not args.force:
        print(f"✓ {out_path} already exists — skipping (use --force to regenerate).")
        return 0

    # A real exported env var takes precedence over .env; the value is never
    # printed or logged.
    api_key = os.environ.get("OPENAI_API_KEY") or read_env("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set (add to repo-root .env)", file=sys.stderr)
        return 1

    if not wav_path.is_file():
        print(f"ERROR: WAV file not found: {wav_path}", file=sys.stderr)
        return 1
    if not meta_path.is_file():
        print(f"ERROR: meta file not found: {meta_path}", file=sys.stderr)
        return 1

    meta = load_meta(meta_path)
    turn_windows = build_turn_windows(meta)
    if not turn_windows:
        print(f"ERROR: no turns found in {meta_path}", file=sys.stderr)
        return 1

    samples, rate = read_wav_mono16(wav_path)

    print(f"→ probing {len(turn_windows)} turns from {wav_path.name} "
          f"against {meta_path.name} (model {MODELS[0]}, fallback {MODELS[1]})")

    results: list[dict] = []
    resolved: dict = {"model": None}
    consecutive_failures = 0
    aborted = False

    with httpx.Client() as client:
        for i, turn in enumerate(turn_windows, start=1):
            b64_wav = slice_to_wav_b64(samples, rate, turn["start_time"], turn["end_time"])
            print(f"  [{i}/{len(turn_windows)}] {turn['speaker']} "
                  f"{turn['scripted_emotion']!s:<16} "
                  f"({turn['start_time']:.2f}s-{turn['end_time']:.2f}s) ...",
                  end="", flush=True)
            gpt_result, model_used, api_failure = process_turn(
                client, api_key, resolved, b64_wav)
            print(" error" if "error" in gpt_result else " ok")

            consecutive_failures = consecutive_failures + 1 if api_failure else 0

            turn_out: dict[str, Any] = {
                "index": i,
                "speaker": turn["speaker"],
                "scripted_emotion": turn["scripted_emotion"],
            }
            if turn["expected"] is not None:
                turn_out["expected"] = turn["expected"]
            turn_out["gpt"] = gpt_result
            turn_out["model_used"] = model_used
            results.append(turn_out)

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\nERROR: {consecutive_failures} consecutive API "
                      f"failures — aborting after turn {i}/{len(turn_windows)}. "
                      f"No output written.", file=sys.stderr)
                aborted = True
                break

    if aborted:
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "wav": str(wav_path),
        "meta": str(meta_path),
        "model_used": resolved["model"],
        "turns": results,
    }, indent=2))
    print(f"\n→ {out_path}")

    print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
