#!/usr/bin/env python3
"""live_e2e.py — act as the phone and exercise the whole realtime + analysis
path of the MindShift server, end to end, against a REAL server (a deployed
URL or a local uvicorn).

What one run does, in order (each step is a ✅/⚠️/❌ line in the report):

  1. AUTH      mint Firebase ID tokens for a "patient" and a "therapist"
               account (``--email/--password``, ``--id-token``, or
               ``--signup`` throwaway accounts via the Identity Toolkit
               REST API using the app's public web API key).
  2. ENROLL    (optional, default on) upload the scene's SELF turns to
               ``POST /voice/enroll-direct`` so the server has a voiceprint
               to confirm identity against during the live session.
  3. LIVE WS   open ``/ws/session/{id}``, send the config frame
               (``tts: "on-device"``, ``report_latency: true``, the
               id_token), stream the scene WAV as PCM16 16 kHz 100 ms
               binary frames on a real-time clock (``--speed`` scales it),
               and per scene turn send a ``turn_local`` exactly like
               apps/mobile/src/live/fastLoop.ts does (text/speaker from the
               scene meta, ``is_self`` from ``self_speaker``, prosody
               measured with server/prosody.py, ``text_tone`` from the
               EMOTION -> TEXT-TONE table below, ``suggestion: null``).
               Collect every server event (cloud ``suggestion`` incl.
               ``partial`` previews, ``tone_flag``, ``speaker_identity``,
               ``suggestion_error``, ``limit_reached``), send ``stop`` and
               read ``session_complete`` (+ ``latency_summary``).
  4. EPISODE   ``POST /sessions/live`` with the same turns + the collected
               identity/tone events; poll ``GET /recordings/{id}`` until
               the background batch analysis is ``full`` (bounded);
               ``POST /episodes/{id}/reflect``; ``GET /recordings/{id}``;
               ``GET /growth``.
  5. SHARE     ``POST /recordings/{id}/shares`` to the therapist's email,
               then, AS THE THERAPIST, ``GET /sessions`` must list the
               episode with its reflections + tone summary, and
               ``GET /recordings/{id}`` must be readable.

Exit status: 0 when no line is ❌, 1 otherwise (⚠️ lines are informational:
an optional server capability that isn't deployed, never a broken path).

Usage
-----
  # throwaway accounts (Firebase email/password sign-up; +addressed emails)
  python scripts/live_e2e.py --base-url https://<cloud-run-url> --signup \
      --signup-email-base sagearbor@gmail.com --scene scene_couple_escalation

  # existing accounts
  python scripts/live_e2e.py --base-url https://... \
      --email you@example.com --password ... \
      --therapist-email doc@example.com --therapist-password ... \
      --scene scene_family3 --speed 2

  # tokens you already hold (no Firebase call at all)
  python scripts/live_e2e.py --base-url http://localhost:8000 \
      --id-token <patient id token> --therapist-id-token <therapist id token>

Options: ``--speed N`` streams N× faster than real time (the on-device
timeline still advances by audio, so turn timestamps stay exact);
``--mode earpiece|speaker|therapist``; ``--no-enroll`` skips voiceprint
enrollment; ``--analysis-timeout`` bounds the wait for the batch analysis;
``--cleanup`` deletes the episode, the voiceprint and (for ``--signup``)
the throwaway Firebase accounts at the end; ``--json`` prints the raw
report dict after the text report.

The same functions are driven in-process by
server/tests/test_live_e2e_inprocess.py against a real uvicorn on
127.0.0.1 with fake providers, so this script cannot rot silently.

EMOTION -> TEXT-TONE table
--------------------------
The phone's on-device text-tone model scores each turn 0–100 on warmth /
defensiveness / sarcasm / sadness / frustration plus a free label. This
client has no model — it derives those scores from the scene meta's
``scripted_emotion`` (fine) and ``emotion_coarse`` (fallback), so the tone
the server stores and the therapist sees is the tone the scene was ACTED
with. The mapping is deliberately aligned with server/live_sessions.py's
rules: ``frustration >= 60`` or label ``angry`` reads as an escalation, a
dominant ``sadness`` reads "sad", ``warmth >= 60`` reads "warm", and a
``neutral`` label stays neutral. See ``TEXT_TONE_BY_EMOTION`` /
``TEXT_TONE_BY_COARSE``.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import secrets
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = REPO_ROOT / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import prosody  # noqa: E402 — server/prosody.py, the phone's port measures the same thing

FIXTURE_DIR = SERVER_DIR / "tests" / "fixtures" / "audio"

# The app's PUBLIC Firebase web API key (apps/mobile/src/auth/firebaseConfig.ts)
# — a client identifier, not a secret; overridable like the app does.
FIREBASE_WEB_API_KEY = os.getenv(
    "EXPO_PUBLIC_FIREBASE_API_KEY", "AIzaSyAJA-C1dpMqpjmM9A7GIGb-IfsOJSl7XS4",
)
IDENTITY_TOOLKIT = "https://identitytoolkit.googleapis.com/v1"

SAMPLE_RATE = 16000
FRAME_MS = 100
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000       # 1600 samples
FRAME_BYTES = FRAME_SAMPLES * 2                       # 3200 bytes, the wire contract

# Simulated on-device STT lag: the phone finalizes a turn (VAD end + local
# STT) a beat after the last sample of it was captured, so its turn_local
# lands AFTER the audio for that span was streamed — the server's ring
# buffer contract (audio_pipeline.PcmRingBuffer).
STT_LAG_S = 0.3

# ---------------------------------------------------------------------------
# EMOTION -> TEXT-TONE (see the module docstring)
# ---------------------------------------------------------------------------

TEXT_TONE_BY_EMOTION: dict[str, dict[str, Any]] = {
    # angry family — every one is an escalation (frustration >= 60 / "angry")
    "tense_rising":     {"warmth": 25, "defensiveness": 55, "sarcasm": 10, "sadness": 5,  "frustration": 65, "label": "angry"},
    "defensive_rising": {"warmth": 20, "defensiveness": 80, "sarcasm": 15, "sadness": 5,  "frustration": 70, "label": "angry"},
    "shout_angry":      {"warmth": 5,  "defensiveness": 70, "sarcasm": 20, "sadness": 5,  "frustration": 95, "label": "angry"},
    "cold_contempt":    {"warmth": 0,  "defensiveness": 60, "sarcasm": 75, "sadness": 10, "frustration": 85, "label": "angry"},
    # sad family
    "hurt_sad":         {"warmth": 45, "defensiveness": 20, "sarcasm": 0,  "sadness": 85, "frustration": 20, "label": "sad"},
    "sullen_sad":       {"warmth": 30, "defensiveness": 35, "sarcasm": 5,  "sadness": 75, "frustration": 30, "label": "sad"},
    "scared_shaky":     {"warmth": 40, "defensiveness": 25, "sarcasm": 0,  "sadness": 70, "frustration": 15, "label": "sad"},
    # happy
    "warm_happy":       {"warmth": 90, "defensiveness": 0,  "sarcasm": 0,  "sadness": 0,  "frustration": 0,  "label": "warm"},
    # neutral family (calm_* / repair_*)
    "calm_open":        {"warmth": 60, "defensiveness": 10, "sarcasm": 0,  "sadness": 5,  "frustration": 5,  "label": "neutral"},
    "calm_guarded":     {"warmth": 40, "defensiveness": 35, "sarcasm": 5,  "sadness": 10, "frustration": 20, "label": "neutral"},
    "calm_neutral":     {"warmth": 50, "defensiveness": 15, "sarcasm": 5,  "sadness": 5,  "frustration": 10, "label": "neutral"},
    "calm_close":       {"warmth": 65, "defensiveness": 5,  "sarcasm": 0,  "sadness": 5,  "frustration": 5,  "label": "neutral"},
    "calm_deescalate":  {"warmth": 60, "defensiveness": 10, "sarcasm": 0,  "sadness": 5,  "frustration": 15, "label": "neutral"},
    "repair_apology":   {"warmth": 70, "defensiveness": 5,  "sarcasm": 0,  "sadness": 25, "frustration": 5,  "label": "neutral"},
}

TEXT_TONE_BY_COARSE: dict[str, dict[str, Any]] = {
    "angry":   {"warmth": 15, "defensiveness": 65, "sarcasm": 15, "sadness": 5,  "frustration": 80, "label": "angry"},
    "sad":     {"warmth": 40, "defensiveness": 25, "sarcasm": 0,  "sadness": 80, "frustration": 20, "label": "sad"},
    "happy":   {"warmth": 90, "defensiveness": 0,  "sarcasm": 0,  "sadness": 0,  "frustration": 0,  "label": "warm"},
    "neutral": {"warmth": 50, "defensiveness": 15, "sarcasm": 5,  "sadness": 5,  "frustration": 10, "label": "neutral"},
}


def text_tone_for(turn_meta: dict) -> dict:
    """The phone's text_tone dict for one scene turn (fine table first, the
    coarse table as fallback — an unknown scripted_emotion never crashes)."""
    fine = TEXT_TONE_BY_EMOTION.get(str(turn_meta.get("scripted_emotion")))
    if fine is not None:
        return dict(fine)
    coarse = TEXT_TONE_BY_COARSE.get(str(turn_meta.get("emotion_coarse")), TEXT_TONE_BY_COARSE["neutral"])
    return dict(coarse)


# ---------------------------------------------------------------------------
# Report plumbing
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    ok: bool | None          # True ✅, False ❌, None ⚠️ (informational / skipped)
    detail: str

    @property
    def mark(self) -> str:
        return "✅" if self.ok is True else ("❌" if self.ok is False else "⚠️")


@dataclass
class Report:
    scene: str
    base_url: str
    speed: float
    mode: str
    checks: list[Check] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)   # raw numbers for --json / tests

    def add(self, name: str, ok: bool | None, detail: str) -> Check:
        c = Check(name, ok, detail)
        self.checks.append(c)
        return c

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.ok is False]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.ok is None]

    def to_dict(self) -> dict:
        return {
            "scene": self.scene, "base_url": self.base_url, "speed": self.speed, "mode": self.mode,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks],
            "data": self.data,
        }


def format_report(report: Report) -> str:
    lines = [
        f"=== live_e2e — scene {report.scene} @ {report.base_url} "
        f"(speed {report.speed:g}x, mode {report.mode}) ===",
    ]
    for c in report.checks:
        lines.append(f"{c.mark} {c.name}: {c.detail}")
    lat = report.data.get("latency_summary")
    if lat:
        lines.append("latency_summary (ms, server-measured per stage):")
        lines.append(f"    {'stage':<18}{'p50':>9}{'p95':>9}{'n':>5}")
        for stage, v in lat.items():
            lines.append(f"    {stage:<18}{v.get('p50', 0):>9.1f}{v.get('p95', 0):>9.1f}{v.get('n', 0):>5}")
    verdict = "PASS" if not report.failures else "FAIL"
    lines.append(f"RESULT: {verdict} ({len(report.failures)} ❌, {len(report.warnings)} ⚠️)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Accounts + Firebase Identity Toolkit
# ---------------------------------------------------------------------------

@dataclass
class Account:
    """One signed-in user as the client sees it.

    ``ws_token`` rides in the WS config frame's ``id_token``; ``headers`` is
    what every REST call sends. Against a real server both come from the
    same Firebase ID token; the in-process test builds an Account whose
    ``ws_token`` is the suite's fake token and whose headers carry its
    ``X-Test-Uid`` override — the client code does not know the difference.
    """
    email: str
    ws_token: str
    headers: dict[str, str]
    uid: str | None = None
    signed_up: bool = False
    id_token: str | None = None

    @classmethod
    def from_id_token(cls, email: str, id_token: str, *, uid: str | None = None, signed_up: bool = False) -> "Account":
        return cls(
            email=email, ws_token=id_token,
            headers={"Authorization": f"Bearer {id_token}"},
            uid=uid, signed_up=signed_up, id_token=id_token,
        )


async def firebase_password_auth(
    http: httpx.AsyncClient, *, email: str, password: str, signup: bool,
    api_key: str = FIREBASE_WEB_API_KEY,
) -> Account:
    """Mint a Firebase ID token by email/password (``accounts:signUp`` when
    ``signup``, else ``accounts:signInWithPassword``)."""
    verb = "signUp" if signup else "signInWithPassword"
    res = await http.post(
        f"{IDENTITY_TOOLKIT}/accounts:{verb}",
        params={"key": api_key},
        json={"email": email, "password": password, "returnSecureToken": True},
        timeout=30.0,
    )
    if res.status_code != 200:
        msg = (res.json().get("error") or {}).get("message", res.text) if res.headers.get("content-type", "").startswith("application/json") else res.text
        raise RuntimeError(f"Firebase {verb} failed for {email}: {res.status_code} {msg}")
    body = res.json()
    return Account.from_id_token(
        email, body["idToken"], uid=body.get("localId"), signed_up=signup,
    )


async def firebase_delete_account(http: httpx.AsyncClient, account: Account, api_key: str = FIREBASE_WEB_API_KEY) -> bool:
    if not account.id_token:
        return False
    res = await http.post(
        f"{IDENTITY_TOOLKIT}/accounts:delete", params={"key": api_key},
        json={"idToken": account.id_token}, timeout=30.0,
    )
    return res.status_code == 200


def throwaway_email(base: str, tag: str) -> str:
    """``user+e2e-<tag>-<random>@domain`` — plus-addressing so every run is a
    fresh Firebase account that still lands in one real inbox."""
    local, _, domain = base.partition("@")
    return f"{local}+e2e-{tag}-{secrets.token_hex(3)}@{domain}"


# ---------------------------------------------------------------------------
# Scene loading — the "phone's" ground truth
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    name: str
    meta: dict
    pcm: np.ndarray            # int16 mono 16 kHz
    sr: int
    turns: list[dict]          # speaker/text/start_time/end_time + meta fields

    @property
    def self_speaker(self) -> str:
        return self.meta["self_speaker"]

    @property
    def duration_s(self) -> float:
        return self.pcm.shape[0] / self.sr

    def is_self(self, speaker: str) -> bool:
        info = (self.meta.get("speakers") or {}).get(speaker) or {}
        return bool(info.get("is_self", speaker == self.self_speaker))

    @property
    def self_turn_indexes(self) -> list[int]:
        return [i for i, t in enumerate(self.turns) if self.is_self(t["speaker"])]

    @property
    def expected_self_escalations(self) -> list[int]:
        """Self turns the scene scripted as angry — what the stored tone
        summary must count as escalations (live_sessions.is_escalated on
        our TEXT_TONE tables)."""
        return [
            i for i, t in enumerate(self.turns)
            if self.is_self(t["speaker"]) and t.get("emotion_coarse") == "angry"
        ]


def _build_turns(meta: dict) -> list[dict]:
    """Same reconstruction as test_diarize_regression_ladder._build_turns:
    turns are concatenated back-to-back with ``silence_gap_sec`` between."""
    gap = float(meta["silence_gap_sec"])
    out = []
    t = 0.0
    for m in meta["turns"]:
        dur = float(m["duration_sec"])
        out.append({
            **m,
            "start_time": round(t, 4),
            "end_time": round(t + dur, 4),
        })
        t += dur + gap
    return out


def load_scene(name: str, fixture_dir: Path = FIXTURE_DIR) -> Scene:
    """``name`` is e.g. ``scene_couple_escalation`` (the ``test_recording_``
    prefix and ``.wav`` are optional)."""
    stem = name.removeprefix("test_recording_").removesuffix(".wav")
    wav_path = fixture_dir / f"test_recording_{stem}.wav"
    meta_path = fixture_dir / f"test_recording_{stem}_meta.json"
    if not wav_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"scene {name!r}: expected {wav_path.name} + {meta_path.name} in {fixture_dir}")
    meta = json.loads(meta_path.read_text())
    with wave.open(str(wav_path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sr != SAMPLE_RATE or ch != 1 or width != 2:
        raise ValueError(f"{wav_path.name}: need 16 kHz mono int16, got {sr} Hz / {ch} ch / {width * 8}-bit")
    pcm = np.frombuffer(raw, dtype="<i2")
    return Scene(name=stem, meta=meta, pcm=pcm, sr=sr, turns=_build_turns(meta))


def list_scenes(fixture_dir: Path = FIXTURE_DIR) -> list[str]:
    return sorted(
        p.name.removeprefix("test_recording_").removesuffix("_meta.json")
        for p in fixture_dir.glob("test_recording_scene_*_meta.json")
    )


def turn_prosody(scene: Scene, turn: dict, text: str) -> dict:
    """What apps/mobile/src/live/prosody.ts::turnProsody reports, measured
    with the server's own reference implementation on the same samples."""
    pcm_f = scene.pcm.astype(np.float32) / 32768.0
    feats = prosody.turn_features(pcm_f, scene.sr, turn["start_time"], turn["end_time"])
    rms = feats["rms"]
    dbfs = 20.0 * math.log10(rms) if rms > 0 else None
    duration = turn["end_time"] - turn["start_time"]
    words = len(text.split())
    rate = words / duration if words > 0 and duration > 0 else None
    return {
        "rms_dbfs": None if dbfs is None else round(dbfs, 2),
        "pitch_hz": None if feats["f0_median"] is None else round(feats["f0_median"], 2),
        "speech_rate": None if rate is None else round(rate, 3),
    }


def build_turn_locals(scene: Scene, session_id: str) -> list[dict]:
    """One ``turn_local`` per scene turn, shaped exactly like fastLoop.ts
    sends it. The "phone" is an oracle here: its speaker verdict is the
    scene's ground truth (``is_self`` + the reserved ``self`` person id for
    the owner, unknown for everyone else)."""
    out = []
    for t in scene.turns:
        is_self = scene.is_self(t["speaker"])
        out.append({
            "type": "turn_local",
            "session_id": session_id,
            "speaker": t["speaker"],
            "speaker_person_id": "self" if is_self else None,
            "speaker_match_score": 0.9 if is_self else None,
            "is_self": is_self,
            "text": t["text"],
            "start_time": t["start_time"],
            "end_time": t["end_time"],
            "transcript_source": "on-device",
            "prosody": turn_prosody(scene, t, t["text"]),
            "text_tone": text_tone_for(t),
            "suggestion": None,
            "suggestion_source": None,
            "tts_source": "on-device",
        })
    return out


def self_voice_wav(scene: Scene, max_seconds: float = 20.0) -> bytes:
    """A WAV of only the SELF speaker's turns (what a guided enrollment clip
    promises), capped so it stays well under the upload limit."""
    parts = []
    total = 0
    for t in scene.turns:
        if not scene.is_self(t["speaker"]):
            continue
        a, b = int(t["start_time"] * scene.sr), int(t["end_time"] * scene.sr)
        chunk = scene.pcm[a:b]
        if total + chunk.shape[0] > int(max_seconds * scene.sr):
            chunk = chunk[: int(max_seconds * scene.sr) - total]
        parts.append(chunk)
        total += chunk.shape[0]
        if total >= int(max_seconds * scene.sr):
            break
    samples = np.concatenate(parts) if parts else np.zeros(0, dtype="<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(scene.sr)
        w.writeframes(samples.astype("<i2").tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# The live WebSocket session
# ---------------------------------------------------------------------------

def ws_url(base_url: str, session_id: str) -> str:
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    return f"{base}/ws/session/{session_id}"


@dataclass
class WsRun:
    session_id: str
    events: list[tuple[float, dict]] = field(default_factory=list)   # (wall time, event)
    sent_turns: list[tuple[float, dict]] = field(default_factory=list)  # (wall time, turn_local)
    frames_sent: int = 0
    audio_seconds: float = 0.0
    wall_seconds: float = 0.0
    config_ack: bool = False
    session_complete: dict | None = None
    close_code: int | None = None
    error: str | None = None

    def of_type(self, kind: str) -> list[dict]:
        return [e for _, e in self.events if e.get("type") == kind]


async def stream_live_session(
    base_url: str, account: Account, scene: Scene, *, session_id: str,
    speed: float = 1.0, config: dict | None = None, stop_timeout_s: float = 60.0,
) -> WsRun:
    """Stream ``scene`` to the server the way the phone does; return every
    event. Never raises for a protocol-level failure — ``run.error`` says
    what went wrong so the report can say ❌ with the reason."""
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed

    run = WsRun(session_id=session_id)
    turn_locals = build_turn_locals(scene, session_id)
    cfg = {
        "type": "config",
        "id_token": account.ws_token,
        "empathy_slider": 60,
        "interject_level": 0,
        "self_speaker": scene.self_speaker,
        "tts": "on-device",
        "report_latency": True,
        **(config or {}),
    }
    url = ws_url(base_url, session_id)
    t_start = time.monotonic()
    try:
        async with connect(url, max_size=None, open_timeout=30.0) as ws:
            await ws.send(json.dumps(cfg))
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=30.0))
            run.events.append((time.monotonic(), first))
            if first.get("type") != "config_ack":
                run.error = f"expected config_ack, got {first}"
                return run
            run.config_ack = True

            done = asyncio.Event()

            async def reader() -> None:
                try:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            continue
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            msg = {"type": "unparseable", "raw": raw[:200]}
                        run.events.append((time.monotonic(), msg))
                        if msg.get("type") == "session_complete":
                            run.session_complete = msg
                            done.set()
                except ConnectionClosed:
                    pass
                finally:
                    done.set()

            reader_task = asyncio.create_task(reader())

            # Sender: frames on the (scaled) real-time clock; a turn_local as
            # soon as its audio (+ STT lag) has been streamed.
            pcm_bytes = scene.pcm.astype("<i2").tobytes()
            n_frames = math.ceil(len(pcm_bytes) / FRAME_BYTES)
            pending = list(turn_locals)
            t0 = time.monotonic()
            for i in range(n_frames):
                target = t0 + (i * FRAME_MS / 1000.0) / speed
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                frame = pcm_bytes[i * FRAME_BYTES:(i + 1) * FRAME_BYTES]
                await ws.send(frame)
                run.frames_sent += 1
                run.audio_seconds = run.frames_sent * FRAME_MS / 1000.0
                while pending and pending[0]["end_time"] + STT_LAG_S <= run.audio_seconds:
                    ev = pending.pop(0)
                    await ws.send(json.dumps(ev))
                    run.sent_turns.append((time.monotonic(), ev))
                if done.is_set():
                    break
            # Anything the tail of the audio didn't release (last turn).
            for ev in pending:
                await asyncio.sleep(STT_LAG_S / speed)
                await ws.send(json.dumps(ev))
                run.sent_turns.append((time.monotonic(), ev))
            if not done.is_set():
                await ws.send(json.dumps({"type": "stop"}))
            try:
                await asyncio.wait_for(done.wait(), timeout=stop_timeout_s)
            except asyncio.TimeoutError:
                run.error = f"no session_complete within {stop_timeout_s:g}s of stop"
            if done.is_set():
                # The server closes with 1000 right after session_complete;
                # let the reader see that close so close_code is real.
                try:
                    await asyncio.wait_for(reader_task, timeout=10.0)
                except asyncio.TimeoutError:
                    reader_task.cancel()
            else:
                reader_task.cancel()
            run.close_code = ws.close_code
    except Exception as exc:  # noqa: BLE001 — reported, never raised past the report
        run.error = run.error or f"{type(exc).__name__}: {exc}"
    run.wall_seconds = time.monotonic() - t_start
    return run


def first_response_ms(run: WsRun) -> dict[str, Any]:
    """Per sent turn: ms from the turn_local going out to the FIRST cloud
    suggestion event for that utterance (partial preview when the LLM
    streams, else the final). Plus the p50 over turns that got one."""
    per_turn: list[dict] = []
    for sent_at, ev in run.sent_turns:
        text = ev["text"]
        first_partial = None
        first_final = None
        for at, msg in run.events:
            if msg.get("type") != "suggestion" or msg.get("utterance_text") != text or at < sent_at:
                continue
            if msg.get("partial") and first_partial is None:
                first_partial = (at - sent_at) * 1000.0
            elif not msg.get("partial") and first_final is None:
                first_final = (at - sent_at) * 1000.0
        per_turn.append({
            "text": text[:40], "is_self": ev.get("is_self"),
            "first_partial_ms": None if first_partial is None else round(first_partial, 1),
            "first_final_ms": None if first_final is None else round(first_final, 1),
        })
    partials = [p["first_partial_ms"] for p in per_turn if p["first_partial_ms"] is not None]
    finals = [p["first_final_ms"] for p in per_turn if p["first_final_ms"] is not None]
    firsts = [
        min(x for x in (p["first_partial_ms"], p["first_final_ms"]) if x is not None)
        for p in per_turn if p["first_partial_ms"] is not None or p["first_final_ms"] is not None
    ]
    return {
        "per_turn": per_turn,
        "partial_p50_ms": round(statistics.median(partials), 1) if partials else None,
        "final_p50_ms": round(statistics.median(finals), 1) if finals else None,
        "first_p50_ms": round(statistics.median(firsts), 1) if firsts else None,
        "first_min_ms": round(min(firsts), 1) if firsts else None,
        "first_max_ms": round(max(firsts), 1) if firsts else None,
        "turns_with_response": len(firsts),
    }


# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------

async def _req(http: httpx.AsyncClient, method: str, base_url: str, path: str, account: Account, **kw) -> tuple[int, Any]:
    headers = {**account.headers, **kw.pop("headers", {})}
    res = await http.request(method, base_url.rstrip("/") + path, headers=headers, timeout=kw.pop("timeout", 60.0), **kw)
    try:
        body = res.json()
    except ValueError:
        body = res.text
    return res.status_code, body


async def enroll_self_voice(http: httpx.AsyncClient, base_url: str, account: Account, scene: Scene) -> tuple[int, Any, float]:
    wav = self_voice_wav(scene)
    seconds = (len(wav) - 44) / (2 * scene.sr)
    status, body = await _req(
        http, "POST", base_url, "/voice/enroll-direct", account,
        files={"file": ("self.wav", wav, "audio/wav")},
        data={"person_id": "self"}, timeout=180.0,
    )
    return status, body, seconds


async def post_live_session(
    http: httpx.AsyncClient, base_url: str, account: Account, *, session_id: str, mode: str,
    turns: list[dict], started_at: str, ended_at: str, tone_flags: list[dict], identities: list[dict],
    title: str | None = None,
) -> tuple[int, Any]:
    body = {
        "session_id": session_id, "started_at": started_at, "ended_at": ended_at, "mode": mode,
        "turns": turns, "tone_flags": tone_flags, "speaker_identities": identities,
        "title": title, "context": "", "analyze": True, "reflect": True,
    }
    return await _req(http, "POST", base_url, "/sessions/live", account, json=body)


async def wait_for_analysis(
    http: httpx.AsyncClient, base_url: str, account: Account, episode_id: str, *,
    timeout_s: float, poll_s: float = 2.0,
) -> tuple[str, dict | None, float]:
    """Poll the episode until ``analysis.live.analysis_status`` leaves
    ``lite``. Returns ``(status, detail, seconds_waited)``; status is
    ``"timeout"`` when the bound hit, ``"http-<code>"`` on a read error."""
    t0 = time.monotonic()
    last: dict | None = None
    while True:
        code, body = await _req(http, "GET", base_url, f"/recordings/{episode_id}", account)
        if code != 200:
            return f"http-{code}", None, time.monotonic() - t0
        last = body
        live = ((body.get("analysis") or {}).get("live") or {})
        status = live.get("analysis_status")
        if status and status != "lite":
            return str(status), body, time.monotonic() - t0
        if time.monotonic() - t0 > timeout_s:
            return "timeout", last, time.monotonic() - t0
        await asyncio.sleep(poll_s)


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------

def _iso_now(offset_s: float = 0.0) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_s)).isoformat()


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


async def run_e2e(
    *, base_url: str, patient: Account, therapist: Account, scene: Scene,
    speed: float = 1.0, mode: str = "earpiece", enroll: bool = True,
    analysis_timeout_s: float = 180.0, cleanup: bool = False,
    http: httpx.AsyncClient | None = None, session_id: str | None = None,
) -> Report:
    report = Report(scene=scene.name, base_url=base_url, speed=speed, mode=mode)
    own_http = http is None
    http = http or httpx.AsyncClient()
    session_id = session_id or f"e2e-{scene.name}-{int(time.time() * 1000)}"
    report.data["session_id"] = session_id
    report.data["accounts"] = {"patient": patient.email, "therapist": therapist.email}
    report.add("accounts", True, f"patient {patient.email}" + (f" (uid {patient.uid})" if patient.uid else "")
               + f" | therapist {therapist.email}" + (f" (uid {therapist.uid})" if therapist.uid else ""))
    try:
        # --- 2. voiceprint enrollment -----------------------------------
        enrolled = False
        if enroll:
            code, body, secs = await enroll_self_voice(http, base_url, patient, scene)
            if code == 200 and isinstance(body, dict) and body.get("enrolled"):
                enrolled = True
                report.add("voice enrollment", True,
                           f"POST /voice/enroll-direct 200 — {secs:.1f}s of {scene.self_speaker} audio, "
                           f"enroll_count={body.get('enroll_count')} dim={body.get('dim')}")
            elif code == 503:
                detail = body.get("detail") if isinstance(body, dict) else body
                report.add("voice enrollment", None, f"skipped — server answered 503 ({detail}); identity verdicts will not flow")
            else:
                report.add("voice enrollment", False, f"POST /voice/enroll-direct -> {code} {str(body)[:200]}")
        else:
            report.add("voice enrollment", None, "skipped (--no-enroll)")
        report.data["enrolled"] = enrolled

        # --- 3. the live WebSocket session --------------------------------
        started_at = _iso_now()
        run = await stream_live_session(base_url, patient, scene, session_id=session_id, speed=speed)
        ended_at = _iso_now()
        report.data["ws"] = {
            "frames_sent": run.frames_sent, "audio_seconds": run.audio_seconds,
            "wall_seconds": round(run.wall_seconds, 2), "turn_locals": len(run.sent_turns),
            "close_code": run.close_code, "error": run.error,
            "event_counts": _count_types(run),
        }
        if run.error or not run.config_ack or run.session_complete is None:
            report.add("live ws", False,
                       f"config_ack={run.config_ack} frames={run.frames_sent} turn_local={len(run.sent_turns)} "
                       f"session_complete={run.session_complete is not None} close={run.close_code} error={run.error} "
                       f"events={_count_types(run)}")
        else:
            report.add("live ws", True,
                       f"config_ack, {run.frames_sent} frames ({run.audio_seconds:.1f}s audio) in {run.wall_seconds:.1f}s wall, "
                       f"{len(run.sent_turns)} turn_local, session_complete (close {run.close_code}); "
                       f"events={_count_types(run)}")
        errors = [e for _, e in run.events if "error" in e and e.get("type") is None]
        if errors:
            report.add("ws protocol errors", False, f"{len(errors)} {{error}} frames: {errors[:3]}")

        # latency summary
        lat = (run.session_complete or {}).get("latency_summary")
        if lat:
            report.data["latency_summary"] = lat
            n_total = (lat.get("total") or {}).get("n", 0)
            report.add("latency_summary", True,
                       f"{len(lat)} stages; total p50 {lat.get('total', {}).get('p50', '-')} ms / p95 "
                       f"{lat.get('total', {}).get('p95', '-')} ms over n={n_total}; llm p50 {lat.get('llm', {}).get('p50', '-')} ms"
                       + (f"; llm_first_partial p50 {lat['llm_first_partial'].get('p50')} ms" if 'llm_first_partial' in lat else "; no llm_first_partial stage (LLM did not stream)"))
        elif run.session_complete is not None:
            report.add("latency_summary", False, "session_complete carried no latency_summary (report_latency/turn_local not honoured?)")

        # cloud suggestions
        suggestions = run.of_type("suggestion")
        finals = [s for s in suggestions if not s.get("partial")]
        partials = [s for s in suggestions if s.get("partial")]
        kinds = {}
        for s in finals:
            kinds[s.get("kind", "response")] = kinds.get(s.get("kind", "response"), 0) + 1
        sources = sorted({s.get("suggestion_source") for s in suggestions})
        sug_errors = run.of_type("suggestion_error")
        limit_hit = run.of_type("limit_reached")
        timing = first_response_ms(run)
        report.data["suggestions"] = {
            "final": len(finals), "partial": len(partials), "kinds": kinds, "sources": sources,
            "errors": len(sug_errors), "limit_reached": len(limit_hit), "timing": timing,
        }
        ttf = timing["first_p50_ms"]
        ttf_txt = (
            f"time-to-first-{'partial' if timing['partial_p50_ms'] is not None else 'suggestion'} p50 "
            f"{ttf} ms (min {timing['first_min_ms']}, max {timing['first_max_ms']}) over {timing['turns_with_response']} turns"
            if ttf is not None else "no turn received a suggestion"
        )
        ok_sugg = bool(finals) and not sug_errors and sources and all(s == "cloud" for s in sources)
        report.add("cloud suggestions", ok_sugg,
                   f"{len(finals)} final ({', '.join(f'{v} {k}' for k, v in kinds.items()) or 'none'}), "
                   f"{len(partials)} partial previews, source={sources or ['-']}, {len(sug_errors)} suggestion_error, "
                   f"{len(limit_hit)} limit_reached; {ttf_txt}"
                   + (f"; speed {speed:g}x means later turns supersede pending ones (latest-wins)" if speed > 1 else ""))

        # identity verdicts
        idents = run.of_type("speaker_identity")
        agree = [i for i in idents if bool(i.get("is_self")) == scene.is_self(i.get("speaker", ""))]
        self_scores = [float(i.get("score", 0)) for i in idents if scene.is_self(i.get("speaker", ""))]
        other_scores = [float(i.get("score", 0)) for i in idents if not scene.is_self(i.get("speaker", ""))]
        report.data["identity"] = {
            "verdicts": len(idents), "agree": len(agree), "self_score_median": _median(self_scores),
            "other_score_median": _median(other_scores),
            "events": [{k: i.get(k) for k in ("speaker", "person_id", "is_self", "score")} for i in idents],
        }
        if idents:
            ok_id = True if len(agree) == len(idents) else None
            report.add("identity verdicts", ok_id,
                       f"{len(idents)}/{len(run.sent_turns)} turns got a speaker_identity; {len(agree)}/{len(idents)} agree with the scene "
                       f"(self_speaker={scene.self_speaker}); score median self={_median(self_scores)} other={_median(other_scores)}")
        elif enrolled:
            report.add("identity verdicts", False, "voiceprint enrolled but no speaker_identity event arrived (speaker_id unavailable on the server?)")
        else:
            report.add("identity verdicts", None, "none — no voiceprint enrolled for this account (expected)")

        # tone flags
        flags = run.of_type("tone_flag")
        labels = {}
        for f in flags:
            labels[f.get("label")] = labels.get(f.get("label"), 0) + 1
        report.data["tone_flags"] = {"count": len(flags), "labels": labels}
        report.add("tone flags", True if flags else None,
                   f"{len(flags)} audio tone_flag events" + (f" {labels}" if labels else
                   " (none — MINDSHIFT_TONE_AUDIO defaults to 'dark': computed server-side, never surfaced)"))

        # --- 4. the episode ------------------------------------------------
        turns = [dict(ev) for _, ev in run.sent_turns] or build_turn_locals(scene, session_id)
        code, body = await post_live_session(
            http, base_url, patient, session_id=session_id, mode=mode, turns=turns,
            started_at=started_at, ended_at=ended_at, tone_flags=flags, identities=idents,
            title=f"e2e {scene.name}",
        )
        if code != 201 or not isinstance(body, dict):
            report.add("episode ingest", False, f"POST /sessions/live -> {code} {str(body)[:300]}")
            return report
        episode_id = body["episode_id"]
        report.data["episode_id"] = episode_id
        report.add("episode ingest", True,
                   f"POST /sessions/live 201 episode {episode_id} created={body.get('created')} turns={body.get('turn_count')} "
                   f"self_speaker={body.get('self_speaker')} analysis_scheduled={body.get('analysis_scheduled')} "
                   f"reflect_scheduled={body.get('reflect_scheduled')}")
        if body.get("self_speaker") != scene.self_speaker:
            report.add("episode self speaker", False, f"server resolved self as {body.get('self_speaker')!r}, scene says {scene.self_speaker!r}")

        if body.get("analysis_scheduled"):
            status, detail, waited = await wait_for_analysis(http, base_url, patient, episode_id, timeout_s=analysis_timeout_s)
            live = ((detail or {}).get("analysis") or {}).get("live") or {}
            if status == "full":
                heats = [p.get("heat") for p in ((detail or {}).get("analysis") or {}).get("per_turn") or []]
                report.add("batch analysis", True, f"full after {waited:.1f}s — {len(heats)} heats, peak {max(h for h in heats if h is not None) if any(h is not None for h in heats) else '-'}")
            elif status == "failed":
                report.add("batch analysis", False, f"analysis_status failed after {waited:.1f}s: {live.get('analysis_error')}")
            else:
                report.add("batch analysis", False, f"{status} after {waited:.1f}s (analysis_status={live.get('analysis_status')})")
        else:
            report.add("batch analysis", None, "not scheduled by the server (too few turns)")

        # reflection
        code, body = await _req(http, "POST", base_url, f"/episodes/{episode_id}/reflect", patient, timeout=180.0)
        if code == 200 and isinstance(body, dict):
            refl = body.get("could_have_said") or []
            report.data["reflections"] = refl
            report.add("reflection", bool(refl),
                       f"POST /episodes/{{id}}/reflect 200 — {len(refl)} reflections for {len(scene.self_turn_indexes)} self turns "
                       f"(cached={body.get('cached')}, indexes {[r.get('turn_index') for r in refl]})")
        else:
            report.add("reflection", False, f"POST /episodes/{{id}}/reflect -> {code} {str(body)[:300]}")

        # detail + growth
        code, detail = await _req(http, "GET", base_url, f"/recordings/{episode_id}", patient)
        if code == 200 and isinstance(detail, dict):
            labels = detail.get("speaker_labels") or {}
            self_label = (labels.get(scene.self_speaker) or {}).get("display_label")
            live = ((detail.get("analysis") or {}).get("live") or {})
            esc = ((live.get("tone_summary") or {}).get("self") or {}).get("escalation_turns")
            ok_detail = self_label == "You" and esc == scene.expected_self_escalations
            report.data["detail"] = {"self_label": self_label, "escalation_turns": esc, "episodes": len(detail.get("episodes") or [])}
            report.add("episode detail", ok_detail,
                       f"GET /recordings/{{id}} 200 — {scene.self_speaker} labelled {self_label!r}, episodes={len(detail.get('episodes') or [])}, "
                       f"self escalation turns {esc} (scene expects {scene.expected_self_escalations}), "
                       f"analysis_status={live.get('analysis_status')}")
        else:
            report.add("episode detail", False, f"GET /recordings/{{id}} -> {code} {str(detail)[:200]}")

        code, growth = await _req(http, "GET", base_url, "/growth", patient)
        if code == 200 and isinstance(growth, dict):
            pts = [p for p in growth.get("points") or [] if p.get("recording_id") == episode_id]
            pt = pts[0] if pts else None
            report.data["growth_point"] = pt
            report.add("growth", pt is not None,
                       (f"GET /growth 200 — point present: my_score={pt.get('my_score')} source={pt.get('source')} mode={pt.get('mode')} "
                        f"partners={pt.get('partner_names')} self_tone escalations={(pt.get('self_tone') or {}).get('escalation_count')}; "
                        f"people={[p.get('display_name') for p in growth.get('people') or []]}")
                       if pt else f"GET /growth 200 but no point for the episode (identified_recordings={growth.get('identified_recordings')})")
        else:
            report.add("growth", False, f"GET /growth -> {code} {str(growth)[:200]}")

        # --- 5. therapist visibility --------------------------------------
        code, body = await _req(http, "POST", base_url, f"/recordings/{episode_id}/shares", patient, json={"email": therapist.email})
        if code != 200:
            report.add("share", False, f"POST /recordings/{{id}}/shares -> {code} {str(body)[:200]}")
        else:
            report.add("share", True, f"shared with {therapist.email} — shares={[s.get('email') for s in (body.get('shares') or [])]}")
            code, sessions = await _req(http, "GET", base_url, "/sessions", therapist)
            rows = (sessions.get("sessions") if isinstance(sessions, dict) else None) or []
            row = next((s for s in rows if s.get("id") == episode_id), None)
            if code != 200:
                report.add("therapist visibility", False, f"GET /sessions as therapist -> {code} {str(sessions)[:200]}")
            elif row is None:
                report.add("therapist visibility", False, f"GET /sessions as therapist lists {len(rows)} sessions, none is the episode")
            else:
                chs = row.get("couldHaveSaid") or []
                ts = row.get("toneSummary") or {}
                esc = ((ts.get("self") or {}).get("escalation_turns"))
                escalated_turns = [t.get("index", i) for i, t in enumerate(row.get("turns") or []) if t.get("escalated")]
                ok_row = bool(row.get("shared")) and bool(chs) and esc == scene.expected_self_escalations
                report.data["therapist_row"] = {
                    "shared": row.get("shared"), "role": row.get("role"), "couldHaveSaid": len(chs),
                    "escalation_turns": esc, "escalated_turns": escalated_turns, "avgPleasantness": row.get("avgPleasantness"),
                }
                report.add("therapist visibility", ok_row,
                           f"GET /sessions as therapist lists the episode: shared={row.get('shared')} role={row.get('role')!r} "
                           f"mode={row.get('mode')} {len(chs)} couldHaveSaid, toneSummary self escalations {esc}, "
                           f"escalated turn indexes {escalated_turns}, avgPleasantness={row.get('avgPleasantness')}")
            code, shared_detail = await _req(http, "GET", base_url, f"/recordings/{episode_id}", therapist)
            shared_flag = shared_detail.get("shared") if isinstance(shared_detail, dict) else None
            report.add("therapist detail read", code == 200 and bool(shared_flag),
                       f"GET /recordings/{{id}} as therapist -> {code} shared={shared_flag}")

        # --- cleanup --------------------------------------------------------
        if cleanup:
            notes = []
            code, _ = await _req(http, "DELETE", base_url, f"/recordings/{episode_id}", patient)
            notes.append(f"episode delete {code}")
            if enrolled:
                code, _ = await _req(http, "DELETE", base_url, "/voice/voiceprint", patient)
                notes.append(f"voiceprint delete {code}")
            for acct in (patient, therapist):
                if acct.signed_up:
                    notes.append(f"firebase delete {acct.email}: {await firebase_delete_account(http, acct)}")
            report.add("cleanup", None, "; ".join(notes))
    finally:
        if own_http:
            await http.aclose()
    return report


def _count_types(run: WsRun) -> dict[str, int]:
    out: dict[str, int] = {}
    for _, e in run.events:
        k = e.get("type") or ("error" if "error" in e else "?")
        out[k] = out.get(k, 0) + 1
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True, help="http(s)://host[:port] of the MindShift server")
    p.add_argument("--scene", default="scene_couple_escalation", help=f"one of {list_scenes()}")
    p.add_argument("--speed", type=float, default=1.0, help="stream N× faster than real time (default 1)")
    p.add_argument("--mode", default="earpiece", choices=["earpiece", "speaker", "therapist"])
    p.add_argument("--no-enroll", action="store_true", help="skip POST /voice/enroll-direct")
    p.add_argument("--analysis-timeout", type=float, default=180.0, help="seconds to wait for the batch analysis")
    p.add_argument("--cleanup", action="store_true", help="delete the episode/voiceprint (+ --signup accounts) afterwards")
    p.add_argument("--json", action="store_true", help="also print the raw report dict")
    # patient auth
    p.add_argument("--id-token", help="patient Firebase ID token")
    p.add_argument("--email")
    p.add_argument("--password")
    # therapist auth
    p.add_argument("--therapist-id-token")
    p.add_argument("--therapist-email")
    p.add_argument("--therapist-password")
    # throwaway accounts
    p.add_argument("--signup", action="store_true", help="create throwaway email/password accounts for both roles")
    p.add_argument("--signup-email-base", default="sagearbor@gmail.com", help="base address for +e2e plus-addressing")
    p.add_argument("--firebase-api-key", default=FIREBASE_WEB_API_KEY)
    return p


async def resolve_accounts(args: argparse.Namespace, http: httpx.AsyncClient) -> tuple[Account, Account]:
    key = args.firebase_api_key
    if args.signup:
        pw_p, pw_t = secrets.token_urlsafe(12), secrets.token_urlsafe(12)
        patient = await firebase_password_auth(http, email=args.email or throwaway_email(args.signup_email_base, "patient"), password=args.password or pw_p, signup=True, api_key=key)
        therapist = await firebase_password_auth(http, email=args.therapist_email or throwaway_email(args.signup_email_base, "therapist"), password=args.therapist_password or pw_t, signup=True, api_key=key)
        return patient, therapist
    if args.id_token:
        patient = Account.from_id_token(args.email or "<id-token>", args.id_token)
    elif args.email and args.password:
        patient = await firebase_password_auth(http, email=args.email, password=args.password, signup=False, api_key=key)
    else:
        raise SystemExit("need --signup, --id-token, or --email + --password for the patient")
    if args.therapist_id_token:
        therapist = Account.from_id_token(args.therapist_email or "<therapist id-token>", args.therapist_id_token)
    elif args.therapist_email and args.therapist_password:
        therapist = await firebase_password_auth(http, email=args.therapist_email, password=args.therapist_password, signup=False, api_key=key)
    else:
        raise SystemExit("need --signup, --therapist-id-token, or --therapist-email + --therapist-password for the therapist")
    return patient, therapist


async def amain(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene = load_scene(args.scene)
    async with httpx.AsyncClient() as http:
        patient, therapist = await resolve_accounts(args, http)
        report = await run_e2e(
            base_url=args.base_url, patient=patient, therapist=therapist, scene=scene,
            speed=args.speed, mode=args.mode, enroll=not args.no_enroll,
            analysis_timeout_s=args.analysis_timeout, cleanup=args.cleanup, http=http,
        )
    print(format_report(report))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    return 1 if report.failures else 0


def main() -> None:
    sys.exit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
