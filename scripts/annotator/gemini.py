"""Gemini on Vertex AI: one audio file (or window) in, raw annotation text out.

* Auth: Application Default Credentials (gcloud), project ``arborfam-hub``.
  No API key.
* Region: ``us-central1`` first; the Gemini 3.x models are only served from
  the ``global`` endpoint for this project, so a 404 there retries ``global``.
* Audio up to ``INLINE_MAX_BYTES`` goes inline; bigger files are uploaded to
  ``gs://$MINDSHIFT_RECORDINGS_BUCKET/annotation-tmp/<uuid>/`` and deleted
  after the call.
* JSON mode (``response_mime_type=application/json``, plus the response
  schema when the model accepts it). A reply cut off at the output limit is
  continued with the documented follow-up prompt (JSON mode off for the
  continuation so the parts concatenate).
* Every call is priced from its token usage and recorded in the
  :class:`annotator.core.Ledger`; a call that could cross the cap is never made.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from annotator import core

PROJECT = os.getenv("MINDSHIFT_VERTEX_PROJECT", "arborfam-hub")
DEFAULT_BUCKET = "arborfam-hub-mindshift-recordings"
INLINE_MAX_BYTES = int(2.5 * 1024 * 1024)  # ~7 min of 48 kbps MP3; longer windows go via GCS
MAX_CONTINUATIONS = 3
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "server/tests/fixtures/annotation/mindshift-annotation-v1.schema.json"


def ffmpeg() -> str:
    for c in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg"):
        if c == "ffmpeg" or Path(c).exists():
            return c
    return "ffmpeg"


def duration_s(path: Path) -> float:
    probe = ffmpeg().replace("ffmpeg", "ffprobe")
    out = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, timeout=60)
    return float(out.stdout.strip())


def encode_window(src: Path, dst: Path, start: float | None = None, length: float | None = None) -> Path:
    """16 kHz mono MP3 (Gemini bills audio by duration, not bytes; small files go inline)."""
    cmd = [ffmpeg(), "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if length is not None:
        cmd += ["-t", f"{length:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(dst)]
    subprocess.run(cmd, check=True, timeout=600)
    return dst


def response_schema() -> dict:
    s = json.loads(SCHEMA_PATH.read_text())
    for k in ("$schema", "$id", "title", "description"):
        s.pop(k, None)
    return s


def degenerate(text: str) -> bool:
    """A reply that hit the output limit because it looped: one JSON string
    longer than 1500 chars, or a last 3000 chars that compress 20:1 (the
    same few phrases over and over; real annotations compress ~8:1 at most)."""
    if re.search(r'"[^"]{1500,}', text):
        return True
    tail = text[-3000:].encode()
    if len(tail) < 3000:
        return False
    return len(zlib.compress(tail)) / len(tail) < 0.05


@dataclass
class CallResult:
    text: str
    parts: list[str]
    usd: float
    usage: dict
    finish: list[str] = field(default_factory=list)
    location: str = ""
    schema_used: bool = False


class Gemini:
    def __init__(self, ledger: core.Ledger, *, bucket: str | None = None, max_out: int = 65535, log=print,
                 use_schema: bool = False):
        from google import genai  # lazy: tests never need it
        self._genai = genai
        self.ledger = ledger
        self.bucket = bucket or os.getenv("MINDSHIFT_RECORDINGS_BUCKET") or DEFAULT_BUCKET
        self.max_out = max_out
        # Constrained decoding against the full schema made gemini-2.5-flash
        # run away inside a string (64k tokens for 60 s of audio), so the
        # default is JSON mode + the documented prompt; the sanitiser and the
        # strict validation after it do the rest.
        self.use_schema = use_schema
        self.log = log
        self._clients: dict[str, object] = {}
        self._loc_for: dict[str, str] = {}
        self._schema_ok: dict[str, bool] = {}

    def _client(self, loc: str):
        if loc not in self._clients:
            self._clients[loc] = self._genai.Client(vertexai=True, project=PROJECT, location=loc)
        return self._clients[loc]

    # -- audio --------------------------------------------------------------
    def audio_part(self, path: Path):
        """(Part, cleanup) — inline bytes, or a gs:// URI for big files."""
        from google.genai import types
        mime = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg",
                ".m4a": "audio/mp4", ".aac": "audio/aac", ".webm": "audio/webm"}.get(path.suffix.lower(), "audio/mpeg")
        size = path.stat().st_size
        if size <= INLINE_MAX_BYTES:
            return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime), (lambda: None)
        from google.cloud import storage
        client = storage.Client(project=PROJECT)
        blob = client.bucket(self.bucket).blob(f"annotation-tmp/{uuid.uuid4().hex}/{path.name}")
        blob.upload_from_filename(str(path), content_type=mime)
        uri = f"gs://{self.bucket}/{blob.name}"
        self.log(f"    uploaded {size / 1e6:.1f} MB -> {uri}")

        def cleanup():
            try:
                blob.delete()
                self.log(f"    deleted {uri}")
            except Exception as exc:  # noqa: BLE001 — report, don't hide
                self.log(f"    WARNING: could not delete {uri}: {exc}")
        return types.Part.from_uri(file_uri=uri, mime_type=mime), cleanup

    # -- generation ---------------------------------------------------------
    def _usage(self, resp) -> dict:
        u = resp.usage_metadata
        audio_in = text_in = 0
        for d in (getattr(u, "prompt_tokens_details", None) or []):
            mod = str(getattr(d, "modality", "")).upper()
            if "AUDIO" in mod:
                audio_in += d.token_count or 0
            else:
                text_in += d.token_count or 0
        if not audio_in and not text_in:
            text_in = u.prompt_token_count or 0
        return {"audio_in": audio_in, "text_in": text_in, "out": u.candidates_token_count or 0,
                "thoughts": getattr(u, "thoughts_token_count", 0) or 0}

    def _generate(self, model: str, contents, config):
        locs = [self._loc_for[model]] if model in self._loc_for else ["us-central1", "global"]
        last = None
        for loc in locs:
            try:
                r = self._client(loc).models.generate_content(model=model, contents=contents, config=config)
                self._loc_for[model] = loc
                return r, loc
            except Exception as exc:  # noqa: BLE001
                if "404" in str(exc) or "NOT_FOUND" in str(exc):
                    last = exc
                    continue
                raise
        raise last

    def max_out_for(self, audio_s: float) -> int:
        """Output budget scaled to the audio: ~120 tokens/s of audio (about 4x
        what a dense annotation needs) + thinking headroom. A runaway reply
        therefore stops early instead of burning 65k tokens."""
        return int(min(self.max_out, 8000 + 120 * audio_s))

    def annotate(self, model: str, audio: Path, prompt: str, audio_s: float, item: str) -> CallResult:
        from google.genai import types
        part, cleanup = self.audio_part(audio)
        try:
            texts, finishes, usd_total, usage_total = [], [], 0.0, {}
            contents = [types.Content(role="user", parts=[part, types.Part.from_text(text=prompt)])]
            schema_used = False
            loc = ""
            budget = self.max_out_for(audio_s)
            fresh_retries = 1
            attempt = 0
            while attempt <= MAX_CONTINUATIONS:
                self.ledger.check(core.projected_cost(model, audio_s, budget))
                cfg = {"max_output_tokens": budget}
                if attempt == 0:
                    cfg["response_mime_type"] = "application/json"
                    if self.use_schema and self._schema_ok.get(model, True):
                        cfg["response_json_schema"] = response_schema()
                try:
                    resp, loc = self._generate(model, contents, types.GenerateContentConfig(**cfg))
                except Exception as exc:  # noqa: BLE001
                    if attempt == 0 and "response_json_schema" in cfg and "400" in str(exc):
                        self.log(f"    {model} rejected the response schema ({str(exc)[:120]}); JSON mode only")
                        self._schema_ok[model] = False
                        cfg.pop("response_json_schema")
                        resp, loc = self._generate(model, contents, types.GenerateContentConfig(**cfg))
                    else:
                        raise
                schema_used = schema_used or ("response_json_schema" in cfg)
                usage = self._usage(resp)
                usd = core.estimate_cost(model, usage)
                self.ledger.record(model, item, usd, usage)
                usd_total += usd
                for k, v in usage.items():
                    usage_total[k] = usage_total.get(k, 0) + v
                cand = (resp.candidates or [None])[0]
                finish = str(getattr(cand, "finish_reason", "")) if cand else "NO_CANDIDATE"
                finishes.append(finish)
                text = resp.text or ""
                self.log(f"    call {attempt + 1}: {usage} finish={finish} ${usd:.4f} (running total ${self.ledger.total:.4f})")
                if "MAX_TOKENS" in finish and degenerate(text):
                    if fresh_retries > 0:
                        fresh_retries -= 1
                        self.log("    reply is a runaway repetition; one fresh retry")
                        texts, attempt = [], 0
                        contents = contents[:1]
                        continue
                    raise RuntimeError("model produced a runaway repetition twice")
                texts.append(text)
                if "MAX_TOKENS" not in finish:
                    break
                attempt += 1
                contents = contents + [types.Content(role="model", parts=[types.Part.from_text(text=text)]),
                                       types.Content(role="user", parts=[types.Part.from_text(text=core.CONTINUE_PROMPT)])]
            return CallResult(text="".join(texts), parts=texts, usd=usd_total, usage=usage_total, finish=finishes,
                              location=loc, schema_used=schema_used)
        finally:
            cleanup()
