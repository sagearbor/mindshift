#!/usr/bin/env python3
"""Generate an ACTED two-speaker argument recording via OpenAI's audio models.

This is the realism counterpart to the physics-modulated Aura fixture: instead
of synthesizing prosody from formant/pitch DSP, we ask OpenAI's audio models to
*perform* each line with an emotion direction and record what they produce.
That makes this a test of "does the pipeline read naturally acted emotion
correctly" — it is NOT a source of physical ground truth. An actor being told
to sound "furious" does not guarantee any specific pitch/energy signature, so
the `*_meta.json` files intentionally carry no expected prosody labels (no
energy/pitch/rate predictions) — only what we asked the model to do. See each
meta file's top-level "_note" for the same caveat in situ.

Two variants (--variant tts | gpt-audio | both, default both):

  tts        POST /v1/audio/speech, model gpt-4o-mini-tts-2025-12-15 (pinned
             snapshot; steerable via its `instructions` param), automatically
             falling back to plain gpt-4o-mini-tts if the API reports the
             snapshot as model-not-found.
             -> tmp/test_recording_openai.wav (+ _meta.json)

  gpt-audio  POST /v1/chat/completions, model gpt-audio-1.5 (fallback
             gpt-audio on model-not-found), modalities ["text","audio"]; the
             delivery direction rides in a system prompt pinning the model to
             verbatim acting, and the spoken line comes back base64-encoded
             in choices[0].message.audio.data.
             -> tmp/test_recording_gptaudio.wav (+ _meta.json)

Each meta.json records `model_used` — the exact model that actually responded
(pinned snapshot or fallback), so fixture provenance is never ambiguous.

Household-chores argument, 10 turns, alternating speakers, arc:
  calm open -> rising tension -> SHOUTED spike -> cold flat contempt
  -> sad/hurt -> scared/shaky -> repair attempt -> calm close

USAGE
  python3 scripts/make_test_recording_openai.py [--variant both] [--out PATH] [--force]

REQUIREMENTS
  OPENAI_API_KEY in the repo-root .env (or exported in the shell). No key,
  no mock output: the script exits 1 with a clear message instead of faking
  a recording. Any API failure other than the one gracefully handled
  model-not-found fallback also exits 1 honestly.
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

import httpx
import numpy as np

# ---------------------------------------------------------------------------
# Repo-root resolution + .env reader — mirrors scripts/deploy_cloudrun.sh's
# read_env() so secrets are handled the same way everywhere in this repo:
# only the one key we need is parsed (no arbitrary .env execution), a real
# exported env var always wins over the file, and we never print the value.
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
# The script. Two speakers, voices "coral" (A) / "onyx" (B) in both variants.
# `instruction` is the steerable emotion direction — sent as the TTS API's
# `instructions` param, or folded into the chat variant's system prompt;
# `scripted_emotion` is just our short label for it.
# ---------------------------------------------------------------------------
TURNS = [
    dict(
        speaker="Speaker A", voice="coral",
        scripted_emotion="calm_open",
        instruction="Calm, warm, and a little tentative — opening a sensitive conversation gently.",
        text="Hey, can we talk about the dishes? They've been piling up all week "
             "and I wanted to check in before it becomes a thing.",
    ),
    dict(
        speaker="Speaker B", voice="onyx",
        scripted_emotion="calm_guarded",
        instruction="Calm but guarded, a hint of defensiveness under the surface.",
        text="Sure, but I did them Tuesday night. I don't think it's fair to act "
             "like I never touch them.",
    ),
    dict(
        speaker="Speaker A", voice="coral",
        scripted_emotion="tense_rising",
        instruction="Tense and clipped, patience visibly running out.",
        text="Tuesday was five days ago. I've done every load since then and I'm "
             "getting tired of keeping score.",
    ),
    dict(
        speaker="Speaker B", voice="onyx",
        scripted_emotion="defensive_rising",
        instruction="Defensive and sharp, volume and pace rising.",
        text="So now I'm the villain because I didn't get to it fast enough? I "
             "have a life outside this kitchen too.",
    ),
    dict(
        speaker="Speaker A", voice="coral",
        scripted_emotion="shout_angry",
        instruction="Shout this, furious and losing control.",
        text="I AM SO SICK OF BEING THE ONLY ONE WHO CARES ABOUT THIS HOUSE! I "
             "CANNOT DO THIS BY MYSELF ANYMORE!",
    ),
    dict(
        speaker="Speaker B", voice="onyx",
        scripted_emotion="cold_contempt",
        instruction="Deliver flat, cold, quietly contemptuous — no warmth at all.",
        text="Wow. Impressive speech. You really do love playing the martyr, "
             "don't you.",
    ),
    dict(
        speaker="Speaker A", voice="coral",
        scripted_emotion="hurt_sad",
        instruction="On the edge of tears, hurt and vulnerable.",
        text="I don't want to be the martyr. I just... I wanted us to be a team, "
             "and I don't feel like one right now.",
    ),
    dict(
        speaker="Speaker B", voice="onyx",
        scripted_emotion="scared_shaky",
        instruction="Frightened, voice shaking, close to panic.",
        text="I don't want that either. I don't want us to be like this. I'm "
             "actually scared we're breaking something here.",
    ),
    dict(
        speaker="Speaker A", voice="coral",
        scripted_emotion="repair_hopeful",
        instruction="Soft, hopeful, still raw but reaching out to repair things.",
        text="I don't want that either. Can we just stop and figure out a system "
             "together? Something that actually works for both of us?",
    ),
    dict(
        speaker="Speaker B", voice="onyx",
        scripted_emotion="calm_close",
        instruction="Calm and warm, relieved, sincere.",
        text="Yeah. Let's start over. I'm sorry I got so defensive — I do want "
             "to fix this with you.",
    ),
]

SILENCE_GAP_SEC = 0.4
TIMEOUT_SEC = 120.0

# Per-variant config: pinned model snapshot first, then the unpinned alias we
# gracefully fall back to if the API reports the snapshot as model-not-found.
VARIANTS = {
    "tts": dict(
        models=("gpt-4o-mini-tts-2025-12-15", "gpt-4o-mini-tts"),
        default_out=REPO_ROOT / "tmp" / "test_recording_openai.wav",
    ),
    "gpt-audio": dict(
        models=("gpt-audio-1.5", "gpt-audio"),
        default_out=REPO_ROOT / "tmp" / "test_recording_gptaudio.wav",
    ),
}

# The chat variant has no dedicated `instructions` param, so the delivery
# direction rides in a system prompt that pins the model to verbatim acting.
GPT_AUDIO_SYSTEM = (
    "You are a voice actor performing one line of a couple's argument. "
    "Perform the user's line exactly, word for word — do not add, change, "
    "or comment on anything. Delivery direction: {instruction}"
)


# ---------------------------------------------------------------------------
# OpenAI calls. One request builder per variant; both yield raw WAV bytes or
# exit 1. The single gracefully-handled failure is model-not-found on the
# pinned snapshot, which retries once with the fallback alias.
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


def _fail(variant: str, turn_no: int, resp: httpx.Response) -> None:
    """Honest failure: status + body snippet, exit 1. No mock fallback."""
    print(f"ERROR: OpenAI {variant} failed on turn {turn_no} "
          f"(HTTP {resp.status_code}):\n{resp.text[:500]}", file=sys.stderr)
    sys.exit(1)


def _post_tts(client: httpx.Client, api_key: str, model: str,
              turn: dict) -> httpx.Response:
    return client.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "voice": turn["voice"],
            "input": turn["text"],
            "instructions": turn["instruction"],
            "response_format": "wav",
        },
        timeout=TIMEOUT_SEC,
    )


def _post_gpt_audio(client: httpx.Client, api_key: str, model: str,
                    turn: dict) -> httpx.Response:
    return client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "modalities": ["text", "audio"],
            "audio": {"voice": turn["voice"], "format": "wav"},
            "messages": [
                {"role": "system",
                 "content": GPT_AUDIO_SYSTEM.format(instruction=turn["instruction"])},
                {"role": "user", "content": turn["text"]},
            ],
        },
        timeout=TIMEOUT_SEC,
    )


def _extract_wav(variant: str, turn_no: int, resp: httpx.Response) -> bytes:
    """Variant-specific 200 response -> WAV bytes (exit 1 if malformed)."""
    if variant == "tts":
        return resp.content
    # gpt-audio: base64 WAV inside the chat completion message.
    try:
        return base64.b64decode(
            resp.json()["choices"][0]["message"]["audio"]["data"])
    except (KeyError, IndexError, TypeError, ValueError) as e:
        print(f"ERROR: gpt-audio turn {turn_no}: response had no "
              f"choices[0].message.audio.data ({e}). Body snippet:\n"
              f"{resp.text[:500]}", file=sys.stderr)
        sys.exit(1)


def synth_turn(client: httpx.Client, api_key: str, variant: str,
               models: tuple[str, str], resolved: dict, turn_no: int,
               turn: dict) -> bytes:
    """Synthesize one turn, resolving the model on first use.

    `resolved` is a shared {"model": str | None} cell: once a model has
    answered successfully we stick with it for every later turn, so the
    fallback probe happens at most once per variant, not per turn.
    """
    post = _post_tts if variant == "tts" else _post_gpt_audio
    model = resolved["model"] or models[0]
    resp = post(client, api_key, model, turn)
    if resolved["model"] is None and model != models[1] and _is_model_not_found(resp):
        print(f"\n  note: model '{model}' not found — falling back to '{models[1]}'")
        model = models[1]
        resp = post(client, api_key, model, turn)
    if resp.status_code != 200:
        _fail(variant, turn_no, resp)
    resolved["model"] = model
    return _extract_wav(variant, turn_no, resp)


# ---------------------------------------------------------------------------
# WAV decode / resample / write helpers (stdlib wave + numpy only)
# ---------------------------------------------------------------------------

def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """WAV bytes -> (mono int16 samples, sample_rate)."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        sys.exit(f"ERROR: API returned {sampwidth * 8}-bit audio; expected 16-bit PCM.")
    samples = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    return samples, rate


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resample to dst_rate (no-op if rates already match).
    Good enough for a test fixture — not a broadcast-quality resampler."""
    if src_rate == dst_rate or len(samples) == 0:
        return samples
    n_dst = int(round(len(samples) * dst_rate / src_rate))
    src_idx = np.arange(len(samples))
    dst_idx = np.linspace(0, len(samples) - 1, num=n_dst)
    return np.interp(dst_idx, src_idx, samples.astype(np.float64)).astype(np.int16)


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.astype(np.int16).tobytes())


# ---------------------------------------------------------------------------
# Per-variant driver: synthesize all turns, concat, write wav + meta, report.
# ---------------------------------------------------------------------------

def generate_variant(client: httpx.Client, api_key: str, variant: str,
                     out_path: Path, force: bool) -> None:
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    if out_path.exists() and not force:
        print(f"✓ [{variant}] {out_path} already exists — skipping "
              f"(use --force to regenerate).")
        return

    models: tuple[str, str] = VARIANTS[variant]["models"]
    resolved: dict = {"model": None}  # filled in by the first successful call
    print(f"→ [{variant}] generating {len(TURNS)}-turn acted argument "
          f"(model {models[0]}, fallback {models[1]})")

    target_rate: int | None = None
    segments: list[np.ndarray] = []
    durations: list[float] = []
    for i, turn in enumerate(TURNS, start=1):
        print(f"  [{i}/{len(TURNS)}] {turn['speaker']:10s} "
              f"{turn['scripted_emotion']:16s} ...", end="", flush=True)
        wav_bytes = synth_turn(client, api_key, variant, models, resolved, i, turn)
        samples, rate = decode_wav(wav_bytes)
        if target_rate is None:
            target_rate = rate           # first turn's rate is the fixture rate
        elif rate != target_rate:
            samples = resample(samples, rate, target_rate)
        segments.append(samples)
        dur = len(samples) / target_rate
        durations.append(dur)
        print(f" {dur:5.2f}s")
    assert target_rate is not None

    # Concatenate with fixed silence gaps between turns.
    gap = np.zeros(int(round(SILENCE_GAP_SEC * target_rate)), dtype=np.int16)
    pieces: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        if i > 0:
            pieces.append(gap)
        pieces.append(seg)
    write_wav(out_path, np.concatenate(pieces), target_rate)

    meta = {
        "_note": ("Acted emotion, not physical ground truth: each turn is the "
                  "model performing the given `instruction`, not a DSP-derived "
                  "prosody measurement. No expected pitch/energy/rate labels "
                  "are included here on purpose — see file docstring."),
        "variant": variant,
        "model_used": resolved["model"],
        "sample_rate": target_rate,
        "silence_gap_sec": SILENCE_GAP_SEC,
        "turns": [
            {
                "speaker": t["speaker"],
                "text": t["text"],
                "scripted_emotion": t["scripted_emotion"],
                "instruction": t["instruction"],
                "duration_sec": round(d, 3),
            }
            for t, d in zip(TURNS, durations)
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # Summary table for this variant.
    total = sum(durations) + SILENCE_GAP_SEC * (len(TURNS) - 1)
    print()
    print(f"[{variant}] model responded: {resolved['model']}")
    print(f"{'#':>2}  {'Speaker':10s} {'Emotion':16s} {'Duration':>8s}")
    for i, (t, d) in enumerate(zip(TURNS, durations), start=1):
        print(f"{i:>2}  {t['speaker']:10s} {t['scripted_emotion']:16s} {d:7.2f}s")
    print(f"\nTotal: {len(TURNS)} turns, {total:.2f}s "
          f"(incl. {SILENCE_GAP_SEC}s gaps) @ {target_rate} Hz")
    print(f"→ {out_path}")
    print(f"→ {meta_path}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variant", choices=["tts", "gpt-audio", "both"],
                   default="both", help="Which model pipeline(s) to run")
    p.add_argument("--out", type=Path, default=None,
                   help="Output WAV path override (single-variant runs only; "
                        "defaults: tmp/test_recording_openai.wav for tts, "
                        "tmp/test_recording_gptaudio.wav for gpt-audio)")
    p.add_argument("--force", action="store_true",
                   help="Regenerate even if the output WAV already exists")
    args = p.parse_args()

    variants = ["tts", "gpt-audio"] if args.variant == "both" else [args.variant]
    if args.out is not None and len(variants) > 1:
        print("ERROR: --out only applies to a single variant; pass "
              "--variant tts or --variant gpt-audio with it.", file=sys.stderr)
        return 2

    pending = [
        (v, args.out if args.out is not None else VARIANTS[v]["default_out"])
        for v in variants
    ]
    # Skip-if-exists is checked before we require a key, so a re-run with all
    # outputs already present is free (no network, no key needed).
    if not args.force and all(out.exists() for _, out in pending):
        for v, out in pending:
            print(f"✓ [{v}] {out} already exists — skipping (use --force to regenerate).")
        return 0

    # A real exported env var takes precedence over .env, mirroring
    # read_env() in deploy_cloudrun.sh. The key value is never printed.
    api_key = os.environ.get("OPENAI_API_KEY") or read_env("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set (add to repo-root .env)", file=sys.stderr)
        return 1

    with httpx.Client() as client:
        for v, out in pending:
            generate_variant(client, api_key, v, out, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
