"""Our own reference transcript: Deepgram pre-recorded (the server's
``audio_ingest`` endpoint + params: nova-2, diarize, utterances,
smart_format) on the normalised 16 kHz WAV, with per-word timings AND
per-word speaker. The raw response is cached next to the recording, so a
re-run (and the regression fixture) never calls Deepgram again."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

import audio_ingest  # server/audio_ingest.py — the server's own Deepgram client constants

from .annotation import dg_label


class SttUnavailable(RuntimeError):
    pass


def _find_env_file(start: Path) -> Path | None:
    """The repo's ``.env``; a git worktree has none of its own, so walk up
    to the main checkout's."""
    for d in [start, *start.parents]:
        p = d / ".env"
        if p.is_file():
            return p
    return None


def env_value(key: str) -> str | None:
    v = os.getenv(key, "").strip()
    if v:
        return v
    env = _find_env_file(Path(__file__).resolve().parent)
    if env is None:
        return None
    for line in env.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith(f"{key}=") or line.startswith(f"export {key}="):
            val = line.split("=", 1)[1].strip().strip('"').strip("'")
            return val or None
    return None


def load_dotenv_into_environ() -> Path | None:
    """Load the repo's ``.env`` (the main checkout's when run from a
    worktree) without overriding real environment variables — what
    server/main.py does for its own checkout. Returns the file used."""
    env = _find_env_file(Path(__file__).resolve().parent)
    if env is None:
        return None
    for line in env.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.removeprefix("export ").split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val
    return env


def deepgram_key() -> str | None:
    return env_value("DEEPGRAM_API_KEY")


def _post_deepgram(wav: bytes, key: str) -> dict:
    resp = httpx.post(
        audio_ingest.DEEPGRAM_PRERECORDED_URL,
        params=audio_ingest.DEEPGRAM_PRERECORDED_PARAMS,
        headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
        content=wav,
        timeout=audio_ingest.DEEPGRAM_PRERECORDED_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


def transcribe(wav: bytes, cache: Path, *, offline: bool = False, refresh: bool = False) -> tuple[dict, str]:
    """(raw Deepgram response, "cache" | "deepgram"). ``offline`` never calls
    out: a missing cache is an honest :class:`SttUnavailable`."""
    cache = Path(cache)
    if cache.exists() and not refresh:
        return json.loads(cache.read_text()), "cache"
    if offline:
        raise SttUnavailable(f"offline and no Deepgram cache at {cache}")
    key = deepgram_key()
    if not key:
        raise SttUnavailable("DEEPGRAM_API_KEY is not set (.env) and no cached transcript exists")
    try:
        raw = _post_deepgram(wav, key)
    except (httpx.HTTPError, ValueError) as exc:
        raise SttUnavailable(f"Deepgram request failed: {exc}") from exc
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(raw))
    return raw, "deepgram"


def words_from_raw(raw: dict) -> list[dict]:
    """``[{word, start, end, speaker, confidence}]`` from channel 0."""
    try:
        alt = raw["results"]["channels"][0]["alternatives"][0]
    except (KeyError, IndexError, TypeError):
        return []
    out = []
    for w in alt.get("words") or []:
        text = w.get("punctuated_word") or w.get("word")
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            start, end = float(w["start"]), float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({
            "word": text.strip(), "start": start, "end": end,
            "speaker": int(w.get("speaker", 0) or 0), "confidence": w.get("confidence"),
        })
    return out


def turns_from_words(words: list[dict], *, max_gap_s: float = 1.0) -> list[dict]:
    """Group words into turns: a new turn on a speaker change or a pause
    longer than ``max_gap_s`` (the annotation prompt's own segment rule)."""
    turns: list[dict] = []
    for w in words:
        label = dg_label(w["speaker"])
        cur = turns[-1] if turns else None
        if cur is None or cur["speaker"] != label or w["start"] - cur["end_time"] > max_gap_s:
            cur = {"speaker": label, "start_time": w["start"], "end_time": w["end"], "words": []}
            turns.append(cur)
        cur["words"].append({"word": w["word"], "start_time": w["start"], "end_time": w["end"]})
        cur["end_time"] = w["end"]
    for t in turns:
        t["text"] = " ".join(x["word"] for x in t["words"])
        t["start_time"] = round(t["start_time"], 3)
        t["end_time"] = round(t["end_time"], 3)
    return turns


def talk_seconds(turns: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for t in turns:
        out[t["speaker"]] = out.get(t["speaker"], 0.0) + (t["end_time"] - t["start_time"])
    return out
