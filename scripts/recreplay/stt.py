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

import spend_ledger  # server/spend_ledger.py — shared paid-call budget
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


def _post_deepgram(wav: bytes, key: str, params: dict | None = None) -> dict:
    resp = httpx.post(
        audio_ingest.DEEPGRAM_PRERECORDED_URL,
        params=params or audio_ingest.DEEPGRAM_PRERECORDED_PARAMS,
        headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
        content=wav,
        timeout=audio_ingest.DEEPGRAM_PRERECORDED_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


def transcribe(wav: bytes, cache: Path, *, offline: bool = False, refresh: bool = False,
               language: str | None = None) -> tuple[dict, str]:
    """(raw Deepgram response, "cache" | "deepgram"). ``offline`` never calls
    out: a missing cache is an honest :class:`SttUnavailable`. ``language``
    (e.g. "el" for a CONFER Greek debate) overrides Deepgram's English default."""
    cache = Path(cache)
    if cache.exists() and not refresh:
        return json.loads(cache.read_text()), "cache"
    if offline:
        raise SttUnavailable(f"offline and no Deepgram cache at {cache}")
    key = deepgram_key()
    if not key:
        raise SttUnavailable("DEEPGRAM_API_KEY is not set (.env) and no cached transcript exists")
    try:
        params = dict(audio_ingest.DEEPGRAM_PRERECORDED_PARAMS)
        if language:
            params["language"] = language
        spend_ledger.reserve("deepgram", spend_ledger.deepgram_cost(len(wav)), str(cache))
        raw = _post_deepgram(wav, key, params)
        spend_ledger.record("deepgram", spend_ledger.deepgram_cost(len(wav)), str(cache))
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


# ---------------------------------------------------------------------------
# FREE reference transcript: local faster-whisper (``--stt whisper``)
# ---------------------------------------------------------------------------
# faster-whisper lives in its own venv (tmp/venv-whisper) apart from this
# torch/speechbrain one, so the decode runs as a subprocess
# (recreplay/whisper_worker.py). Its cache sits next to the Deepgram cache:
# work/<name>/whisper.raw.json (the decode, no speakers) and
# work/<name>/whisper.diar.json (the local diarizer's speaker per word).
# Whisper does no diarization: words get speakers from the annotation's
# aligned segments when there is one (pipeline), else from the server's own
# local ECAPA diarizer (server/diarize_local.py), else ONE label (honest).

WHISPER_WORKER = Path(__file__).resolve().parent / "whisper_worker.py"
DEFAULT_WHISPER_MODEL = os.getenv("MINDSHIFT_WHISPER_MODEL", "small")


def whisper_python() -> Path | None:
    env = os.getenv("MINDSHIFT_WHISPER_PYTHON")
    if env and Path(env).exists():
        return Path(env)
    for d in Path(__file__).resolve().parents:
        p = d / "tmp" / "venv-whisper" / "bin" / "python"
        if p.exists():
            return p
    return None


def _run_worker(wav: Path, out: Path, model: str, language: str | None) -> None:
    import subprocess

    py = whisper_python()
    if py is None:
        raise SttUnavailable("faster-whisper venv not found (tmp/venv-whisper; or set MINDSHIFT_WHISPER_PYTHON)")
    cmd = [str(py), "-I", str(WHISPER_WORKER), "--wav", str(wav), "--out", str(out), "--model", model]
    if language:
        cmd += ["--language", language]
    proc = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=4 * 3600)
    if proc.returncode != 0 or not out.exists():
        raise SttUnavailable(f"whisper decode failed ({proc.returncode}): {(proc.stderr or proc.stdout)[-600:]}")


def transcribe_whisper(wav: Path, cache: Path, *, offline: bool = False, refresh: bool = False,
                       language: str | None = None, model: str | None = None, runner=None) -> tuple[dict, str]:
    """(whisper decode document, "whisper-cache" | "whisper"). $0, local.
    ``offline`` never decodes: a missing cache is :class:`SttUnavailable`."""
    cache = Path(cache)
    if cache.exists() and not refresh:
        return json.loads(cache.read_text()), "whisper-cache"
    if offline:
        raise SttUnavailable(f"offline and no whisper cache at {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.json")
    (runner or _run_worker)(Path(wav), tmp, model or DEFAULT_WHISPER_MODEL, language)
    doc = json.loads(tmp.read_text())
    tmp.replace(cache)
    return doc, "whisper"


def _whisper_words(doc: dict) -> list[dict]:
    out = []
    for s in doc.get("segments") or []:
        for w in s.get("words") or []:
            text = (w.get("word") or "").strip()
            if text:
                out.append({"word": text, "start": float(w["start"]), "end": float(w["end"]),
                            "probability": w.get("probability")})
    return out


def raw_from_whisper(doc: dict, speakers: list[int] | None = None, *, diarization: dict | None = None) -> dict:
    """A Deepgram-SHAPED raw response from a whisper decode, so
    :func:`words_from_raw` and everything downstream work unchanged."""
    ws = _whisper_words(doc)
    spk = speakers if speakers is not None else [0] * len(ws)
    words = [{"word": w["word"], "punctuated_word": w["word"], "start": w["start"], "end": w["end"],
              "speaker": int(spk[i]) if i < len(spk) else 0, "confidence": w["probability"]}
             for i, w in enumerate(ws)]
    return {"metadata": {"engine": "faster-whisper", "model": doc.get("model"), "language": doc.get("language"),
                         "diarization": diarization or {"source": "none"}},
            "results": {"channels": [{"alternatives": [{"words": words}]}]}}


def segment_label_ids(segments: list[dict]) -> dict[str, int]:
    """Segment speaker label -> integer speaker, in order of first appearance."""
    ids: dict[str, int] = {}
    for s in sorted(segments, key=lambda s: float(s["start"])):
        ids.setdefault(str(s["speaker"]), len(ids))
    return ids


def speakers_from_segments(words: list[dict], segments: list[dict]) -> list[int]:
    """Each word's speaker index from labelled time segments
    (``[{start, end, speaker}]``): the segment it overlaps most, else the
    nearest one. Labels are numbered in order of first appearance."""
    if not segments:
        return [0] * len(words)
    segs = sorted(segments, key=lambda s: float(s["start"]))
    ids = segment_label_ids(segs)
    out = []
    for w in words:
        a, b = float(w["start"]), float(w["end"])
        best, best_ov = None, 0.0
        for s in segs:
            ov = min(b, float(s["end"])) - max(a, float(s["start"]))
            if ov > best_ov:
                best, best_ov = s, ov
        if best is None:
            mid = (a + b) / 2
            best = min(segs, key=lambda s: min(abs(mid - float(s["start"])), abs(mid - float(s["end"]))))
        out.append(ids[str(best["speaker"])])
    return out


def _default_diarize(pcm, sr, turns):
    import diarize_local  # server/diarize_local.py (torch + speechbrain, this venv)

    return diarize_local.diarize_windows_first(pcm, sr, turns) or diarize_local.diarize_turns(pcm, sr, turns)


def diarize_whisper(pcm, doc: dict, *, sr: int = 16000, diarize_fn=None) -> tuple[list[int], dict]:
    """(speaker index per whisper word, info). Whisper segments become the
    diarizer's input turns (with word timings for its split pass); its
    relabelled turns are mapped back onto the words by time. ``None`` from
    the diarizer (one voice / nothing trustworthy) = one label for all."""
    turns = []
    for s in doc.get("segments") or []:
        ws = [w for w in s.get("words") or [] if (w.get("word") or "").strip()]
        if not ws:
            continue
        turns.append({"speaker": "Speaker A", "start_time": float(s["start"]), "end_time": float(s["end"]),
                      "text": s.get("text", ""),
                      "words": [{"word": w["word"].strip(), "start_time": float(w["start"]),
                                 "end_time": float(w["end"])} for w in ws]})
    words = _whisper_words(doc)
    res = (diarize_fn or _default_diarize)(pcm, sr, turns) if turns else None
    if not res:
        return [0] * len(words), {"source": "none", "num_speakers": 1,
                                  "note": "local diarizer found no trustworthy second voice"}
    segs = [{"start": t["start_time"], "end": t["end_time"], "speaker": t["speaker"]} for t in res["turns"]]
    return speakers_from_segments(words, segs), {"source": res.get("source", "local"),
                                                 "num_speakers": res.get("num_speakers")}
