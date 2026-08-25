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

  6. WATCH     (``--with-watch``) before step 2, pair a FAKE WATCH to the
               patient the way a real Wear OS watch does — ``POST
               /me/pair/start`` (no auth) -> the phone's ``POST
               /me/pair/claim`` -> the watch's ``GET /me/pair/status`` hands
               back the device token — and hold its ``/ws/live-session/{id}``
               socket open (silent: no PCM, no HR) for the whole live
               session. Then assert that the phone's SELF escalations reached
               the wrist as ``vector_event``/``nudge`` frames exactly as the
               shared relay + ``NudgePolicy`` predicts (same code, same
               inputs), nothing for the other speaker, the scene's
               ``expected_nudges`` spec is met, report turn_local -> wrist
               timing, and check the watch's persisted live session carries
               the relayed events. See "The paired watch" below.

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

``--call`` runs the in-app call walk instead (two phones on one call —
see run_call_e2e and docs/plans/2026-08-25-in-app-calls.md);
``--call --participants 3`` adds a second coached participant (a third
account, ``--peer-*`` or a third ``--signup`` throwaway) and makes the
therapist the read-only observer on her own socket (run_call_e2e_three_way:
full-mesh signaling, per-participant coaching with ``for_uid`` copies for
her, hang-up order host → peer → therapist, two episodes granted to her).

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
import contextlib
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
            if stage == "hedge":
                # Whole-session counts, not a stage (perf/llm-hedging).
                lines.append(f"    hedge: {v.get('hedged', 0)}/{v.get('n', 0)} calls hedged, "
                             f"{v.get('hedge_won', 0)} won, {v.get('slow_llm', 0)} abandoned (slow_llm)")
                continue
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
    pcm: np.ndarray | None = None, turn_locals: list[dict] | None = None,
    pre_stream=None, post_stream=None,
) -> WsRun:
    """Stream ``scene`` to the server the way the phone does; return every
    event. Never raises for a protocol-level failure — ``run.error`` says
    what went wrong so the report can say ❌ with the reason.

    ``pcm`` / ``turn_locals`` override what this "phone" hears and reports
    (an in-app call participant hears only itself — see ``--call``);
    ``pre_stream(ws, run)`` is awaited after ``config_ack`` with the reader
    running, so a call participant can bind (``call_join``) and exchange
    signaling before the first frame; ``post_stream(ws, run)`` after the
    last turn_local and before ``stop`` (a call participant waits for the
    others to finish talking rather than hanging up on them)."""
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed

    run = WsRun(session_id=session_id)
    if turn_locals is None:
        turn_locals = build_turn_locals(scene, session_id)
    if pcm is None:
        pcm = scene.pcm
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

            if pre_stream is not None:
                try:
                    await pre_stream(ws, run)
                except Exception as exc:  # noqa: BLE001 — reported, never raised past the report
                    run.error = f"pre_stream: {type(exc).__name__}: {exc}"
                    reader_task.cancel()
                    return run

            # Sender: frames on the (scaled) real-time clock; a turn_local as
            # soon as its audio (+ STT lag) has been streamed.
            pcm_bytes = pcm.astype("<i2").tobytes()
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
            if post_stream is not None and not done.is_set():
                try:
                    await post_stream(ws, run)
                except Exception as exc:  # noqa: BLE001 — reported, never raised past the report
                    run.error = run.error or f"post_stream: {type(exc).__name__}: {exc}"
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
# The paired watch (--with-watch): phone -> server -> wrist
# ---------------------------------------------------------------------------
#
# The one production path nothing else exercises end to end: the phone's
# turn_local -> audio_pipeline._enrich_turn_local -> watch.relay.push_turn_local
# -> the uid's OPEN watch socket (watch/routers/ws.py) -> a `nudge` frame.
# This section plays the WATCH: it pairs the way a real Wear OS watch does
# (apps/watch/wearApp/.../auth/DevicePairingClient.kt + PairingPoller.kt,
# with the phone's claim from apps/mobile/src/api/watchPairing.ts), opens
# `/ws/live-session/{id}` with the device token it was handed, and only
# LISTENS — no PCM windows, no HR — so every frame that comes down the socket
# is the relay's doing and the watch's stream clock stays at 0 (no cooldown
# ticks, so the shared NudgePolicy's answer is deterministic).
#
# Auth: the server's WS handshake (watch/auth.resolve_ws_principal) takes
# `?token=<device token>` (what a paired watch holds) or the legacy
# `?account=<uid>` (what the SHIPPED Wear app's EpisodeWsClient still sends;
# on by default via MINDSHIFT_ALLOW_LEGACY_ACCOUNT). `--watch-auth` picks;
# `token` is the default because it proves the whole pairing flow minted a
# credential the server honours.

WATCH_VECTOR_ORDER = {"yelling": 0, "aggressive_tone": 1}   # relay emission order
WATCH_SPEC_LEVELS = {"mild": 1, "strong": 3}                 # scene meta `expected_nudges.level` -> min channel level


class WatchPairingError(RuntimeError):
    """The pairing handshake did not produce a device token — the message
    says which step and what the server answered."""


@dataclass
class WatchRun:
    live_session_id: str
    auth_mode: str = "token"
    pairing_id: str | None = None
    code: str | None = None
    account_id: str | None = None
    device_token: str | None = None
    frames: list[tuple[float, dict]] = field(default_factory=list)   # (wall time, frame)
    opened_at: float | None = None
    saved: dict | None = None
    close_code: int | None = None
    error: str | None = None

    def of_type(self, kind: str) -> list[dict]:
        return [f for _, f in self.frames if f.get("type") == kind]

    @property
    def headers(self) -> dict[str, str]:
        """What the watch sends on REST (WatchApiClient.kt: Bearer device token)."""
        return {"Authorization": f"Bearer {self.device_token}"} if self.device_token else {}

    @property
    def query(self) -> dict[str, str]:
        return {} if self.auth_mode == "token" else {"account": self.account_id or ""}


def watch_ws_url(base_url: str, live_session_id: str, *, token: str | None = None, account: str | None = None) -> str:
    base = ws_url(base_url, "").rsplit("/ws/session/", 1)[0]
    if token:
        return f"{base}/ws/live-session/{live_session_id}?token={token}"
    return f"{base}/ws/live-session/{live_session_id}?account={account}"


async def pair_fake_watch(http: httpx.AsyncClient, base_url: str, patient: Account, watch: WatchRun) -> None:
    """The real three-step pairing, headless. Fills ``watch`` in place;
    raises WatchPairingError with the failing step on any deviation."""
    base = base_url.rstrip("/")
    # 1. the watch asks for a code (no auth — it has no identity yet):
    #    DevicePairingClient.start()
    res = await http.post(f"{base}/me/pair/start", timeout=30.0)
    if res.status_code != 200:
        raise WatchPairingError(f"POST /me/pair/start -> {res.status_code} {res.text[:200]}")
    started = res.json()
    watch.pairing_id, watch.code = started["pairing_id"], started["code"]
    # 2. the signed-in phone types the code: watchPairing.ts claimWatchPairing()
    res = await http.post(f"{base}/me/pair/claim", headers=patient.headers, json={"code": watch.code}, timeout=30.0)
    if res.status_code != 200:
        raise WatchPairingError(f"POST /me/pair/claim (as patient) -> {res.status_code} {res.text[:200]}")
    claimed = res.json()
    if claimed.get("status") != "claimed" or claimed.get("pairing_id") != watch.pairing_id:
        raise WatchPairingError(f"POST /me/pair/claim answered {claimed}")
    # 3. the watch polls for its credential: PairingPoller.poll()
    res = await http.get(f"{base}/me/pair/status", params={"pairing_id": watch.pairing_id}, timeout=30.0)
    if res.status_code != 200:
        raise WatchPairingError(f"GET /me/pair/status -> {res.status_code} {res.text[:200]}")
    status = res.json()
    if status.get("status") != "claimed" or not status.get("device_token"):
        raise WatchPairingError(f"GET /me/pair/status answered {status} (expected claimed + device_token)")
    watch.account_id = status.get("account_id") or claimed.get("account_id")
    watch.device_token = status["device_token"]
    if patient.uid and watch.account_id != patient.uid:
        raise WatchPairingError(f"device token bound to {watch.account_id!r}, patient uid is {patient.uid!r}")


class FakeWatch:
    """The wrist end of the relay: an open watch live-session socket that
    records every frame with its arrival time, then ends the session the
    way the watch does (``{"type": "end"}`` -> ``live_session_saved``)."""

    def __init__(self, url: str, run: WatchRun) -> None:
        self.url = url
        self.run = run
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._saved = asyncio.Event()

    async def open(self) -> None:
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed

        self._ws = await connect(self.url, max_size=None, open_timeout=30.0)
        self.run.opened_at = time.monotonic()

        async def reader() -> None:
            try:
                async for raw in self._ws:
                    if isinstance(raw, bytes):
                        continue
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        msg = {"type": "unparseable", "raw": raw[:200]}
                    self.run.frames.append((time.monotonic(), msg))
                    if msg.get("type") == "live_session_saved":
                        self.run.saved = msg
                        self._saved.set()
            except ConnectionClosed:
                pass
            finally:
                self._saved.set()

        self._reader = asyncio.create_task(reader())

    async def end(self, timeout_s: float = 30.0) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.send(json.dumps({"type": "end"}))
            try:
                await asyncio.wait_for(self._saved.wait(), timeout=timeout_s)
            except asyncio.TimeoutError:
                self.run.error = self.run.error or f"no live_session_saved within {timeout_s:g}s of end"
        finally:
            await self.close()

    async def close(self) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self.run.close_code = self._ws.close_code
        if self._reader is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._reader, timeout=5.0)
            if not self._reader.done():
                self._reader.cancel()


def expected_watch_relay(turn_locals: list[dict], *, baseline_rms_db: float | None = None) -> list[dict]:
    """What the server's own relay + shared NudgePolicy must send a SILENT
    paired watch for these phone turns, computed with the same code
    (``watch.relay.turn_local_to_vector_events`` on the relay's phone-side
    running-median loudness baseline, ``nudge_policy.NudgePolicy`` on the
    account's default subscriptions, stream clock 0). One entry per self
    turn that produced anything: ``{turn_index, events: [(vector, level)],
    nudges: [(channel, level)], level_after}``. ``baseline_rms_db`` is the
    account's watch ENROLLMENT baseline when it has one (a throwaway
    account never does -> None -> running median of prior phone turns)."""
    from models.audio import TurnLocalEvent
    from nudge_policy import NudgePolicy
    from watch import relay as watch_relay
    from watch.models import EnrollmentBaseline, VectorSubscription
    from watch.store import DEFAULT_VECTOR_NAMES
    from watch.vectors import VectorEngine

    baseline = None
    if baseline_rms_db is not None:
        baseline = EnrollmentBaseline(account_id="e2e", rms_db=baseline_rms_db, f0_median=120.0, updated_at="e2e")
    session = watch_relay.LiveWatchSession(
        account_id="e2e", live_session_id="e2e", engine=VectorEngine(baseline), emit=None, loop=None,  # type: ignore[arg-type]
    )
    policy = NudgePolicy([VectorSubscription(vector=v) for v in DEFAULT_VECTOR_NAMES])
    out: list[dict] = []
    for i, tl in enumerate(turn_locals):
        ev = TurnLocalEvent.model_validate(tl)
        if ev.is_self is not True:
            continue
        rms = ev.prosody.rms_dbfs if ev.prosody is not None else None
        events = watch_relay.turn_local_to_vector_events(ev, t=0.0, baseline_rms_db=session.phone_baseline_rms_db())
        session.observe_phone_rms(rms)
        if not events:
            continue
        nudges = policy.on_events(events, 0.0)
        out.append({
            "turn_index": i,
            "events": [(e.vector, e.level) for e in events],
            "nudges": [(n.channel, n.level) for n in nudges],
            "level_after": policy.current()["A"],
        })
    return out


def group_watch_frames(frames: list[tuple[float, dict]]) -> list[dict]:
    """Split the watch socket's frames into one group per relayed turn.

    ws.py's ``emit`` sends a turn's vector_events (relay order: yelling,
    then aggressive_tone) followed by that call's nudges, all from one
    coroutine on one loop — so a turn's frames are contiguous. A new group
    starts at a vector_event that follows a nudge, or whose vector does not
    advance the relay order (a second ``aggressive_tone`` in a row is the
    next turn). A nudge right after a group that already nudged (a cooldown
    de-escalation from the watch's own clock — impossible for the silent
    fake watch, whose clock never ticks) gets a group of its own."""
    groups: list[dict] = []
    cur: dict | None = None
    for at, f in frames:
        kind = f.get("type")
        if kind == "vector_event":
            order = WATCH_VECTOR_ORDER.get(f.get("vector"), 99)
            last = cur["events"][-1][0] if cur and cur["events"] else None
            if cur is None or cur["nudges"] or (last is not None and order <= WATCH_VECTOR_ORDER.get(last, 99)):
                cur = {"at": at, "events": [], "nudges": [], "t": f.get("t")}
                groups.append(cur)
            cur["events"].append((f.get("vector"), f.get("level")))
        elif kind == "nudge":
            # NudgePolicy emits at most one nudge per channel per call and
            # the relay only ever feeds channel A, so a second nudge in a
            # group is the next policy call, not this one.
            if cur is None or cur["nudges"]:
                cur = {"at": at, "events": [], "nudges": [], "t": f.get("t")}
                groups.append(cur)
            cur["nudges"].append((f.get("channel"), f.get("level")))
    return groups


def _channel_a_after(groups: list[dict], upto: int) -> int:
    level = 0
    for g in groups[: upto + 1]:
        for ch, lvl in g["nudges"]:
            if ch == "A":
                level = lvl
    return level


def _spec_level_ok(spec_level: str, level: int) -> bool:
    need = WATCH_SPEC_LEVELS.get(str(spec_level))
    return need is not None and level >= need


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
    with_watch: bool = False, watch_auth: str = "token", watch_settle_s: float = 1.5,
) -> Report:
    report = Report(scene=scene.name, base_url=base_url, speed=speed, mode=mode)
    own_http = http is None
    http = http or httpx.AsyncClient()
    session_id = session_id or f"e2e-{scene.name}-{int(time.time() * 1000)}"
    report.data["session_id"] = session_id
    report.data["accounts"] = {"patient": patient.email, "therapist": therapist.email}
    report.add("accounts", True, f"patient {patient.email}" + (f" (uid {patient.uid})" if patient.uid else "")
               + f" | therapist {therapist.email}" + (f" (uid {therapist.uid})" if therapist.uid else ""))
    watch: WatchRun | None = None
    fake_watch: FakeWatch | None = None
    try:
        # --- 1b. pair a watch to the patient's account --------------------
        if with_watch:
            watch = WatchRun(live_session_id=f"{session_id}-watch", auth_mode=watch_auth)
            if not patient.uid:
                # The watch domain's /me tells a signed-in caller its own uid.
                code, me = await _req(http, "GET", base_url, "/me", patient)
                if code == 200 and isinstance(me, dict) and me.get("account_id"):
                    patient.uid = me["account_id"]
            try:
                await pair_fake_watch(http, base_url, patient, watch)
                code, me_as_watch = await _req(http, "GET", base_url, "/me", Account(email="watch", ws_token="", headers=watch.headers))
                code_p, me_as_patient = await _req(http, "GET", base_url, "/me", patient)
                watch_sees_self = code == 200 and isinstance(me_as_watch, dict) and me_as_watch.get("account_id") == watch.account_id
                phone_sees_watch = code_p == 200 and isinstance(me_as_patient, dict) and bool(me_as_patient.get("has_paired_watch"))
                report.add("watch pairing", watch_sees_self and phone_sees_watch,
                           f"POST /me/pair/start -> code {watch.code}; POST /me/pair/claim as patient -> claimed; "
                           f"GET /me/pair/status -> device_token for account {watch.account_id}; "
                           f"GET /me as the watch (Bearer device token) -> {code} account_id match={watch_sees_self}; "
                           f"GET /me as patient has_paired_watch={me_as_patient.get('has_paired_watch') if isinstance(me_as_patient, dict) else me_as_patient}")
            except WatchPairingError as exc:
                report.add("watch pairing", False, str(exc))
                watch = None
            if watch is not None:
                url = watch_ws_url(base_url, watch.live_session_id,
                                   token=watch.device_token if watch_auth == "token" else None,
                                   account=watch.account_id)
                fake_watch = FakeWatch(url, watch)
                try:
                    await fake_watch.open()
                    # ws.py registers the socket with the relay only after
                    # its baseline/subscription reads; give it a beat.
                    await asyncio.sleep(watch_settle_s)
                except Exception as exc:  # noqa: BLE001 — reported below
                    watch.error = f"{type(exc).__name__}: {exc}"
                    report.add("watch ws", False, f"open {url.split('?')[0]}?{watch_auth}=… failed: {watch.error}")
                    fake_watch = None
            report.data["watch"] = {"auth_mode": watch_auth, "live_session_id": watch.live_session_id if watch else None,
                                    "paired": watch is not None and watch.device_token is not None}
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
        error_reasons: dict[str, int] = {}
        for e in sug_errors:
            error_reasons[str(e.get("reason"))] = error_reasons.get(str(e.get("reason")), 0) + 1
        limit_hit = run.of_type("limit_reached")
        timing = first_response_ms(run)
        report.data["suggestions"] = {
            "final": len(finals), "partial": len(partials), "kinds": kinds, "sources": sources,
            "errors": len(sug_errors), "error_reasons": error_reasons,
            "limit_reached": len(limit_hit), "timing": timing,
        }
        ttf = timing["first_p50_ms"]
        ttf_txt = (
            f"time-to-first-{'partial' if timing['partial_p50_ms'] is not None else 'suggestion'} p50 "
            f"{ttf} ms (min {timing['first_min_ms']}, max {timing['first_max_ms']}) over {timing['turns_with_response']} turns"
            if ttf is not None else "no turn received a suggestion"
        )
        # A suggestion_error is the server honestly reporting an utterance
        # that yielded nothing. The model occasionally answering with
        # unparseable JSON ("llm_parse_error") is a soft wart of the path
        # working (⚠️, counted); any OTHER reason (provider auth, timeout,
        # a crash class name) is a broken path (❌).
        hard_errors = {r: n for r, n in error_reasons.items() if r != "llm_parse_error"}
        path_ok = bool(finals) and bool(sources) and all(s == "cloud" for s in sources)
        ok_sugg: bool | None = False if (not path_ok or hard_errors) else (True if not sug_errors else None)
        report.add("cloud suggestions", ok_sugg,
                   f"{len(finals)} final ({', '.join(f'{v} {k}' for k, v in kinds.items()) or 'none'}), "
                   f"{len(partials)} partial previews, source={sources or ['-']}, "
                   f"{len(sug_errors)} suggestion_error{(' ' + str(error_reasons)) if error_reasons else ''}, "
                   f"{len(limit_hit)} limit_reached; {ttf_txt}"
                   + (f"; speed {speed:g}x means later turns supersede pending ones (latest-wins)" if speed > 1 else ""))

        # Server-side transcripts: a local-first client gets NO transcript
        # echo for its turn_local, so every `transcript` event is a span
        # the server's own transcriber (Deepgram) finalized and did NOT
        # suppress — i.e. it landed BEFORE the phone's turn_local for that
        # span (overlap suppression only drops segments that arrive after).
        transcripts = run.of_type("transcript")
        report.data["server_transcripts"] = len(transcripts)
        report.add("server transcripts", None if transcripts else True,
                   f"{len(transcripts)} un-suppressed transcript events from the server's transcriber"
                   + (" — Deepgram is live and finalized these spans before the phone's turn_local arrived, "
                      "so they were coached twice (once per source); the latency n above counts both"
                      if transcripts else " (none: nothing coached twice)"))

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

        # --- 3b. what reached the wrist -----------------------------------
        if fake_watch is not None and watch is not None:
            # Every enrichment task for the last turn_local lands before
            # session_complete (audio_pipeline's graceful stop); one more
            # beat covers the relay's hop onto the watch socket's loop.
            await asyncio.sleep(1.0)
            await fake_watch.end()
            groups = group_watch_frames(watch.frames)
            expected = expected_watch_relay([dict(ev) for _, ev in run.sent_turns])
            observed_sig = [(g["events"], g["nudges"]) for g in groups]
            expected_sig = [(e["events"], e["nudges"]) for e in expected]
            match = observed_sig == expected_sig
            # Timing: group k is the k-th relayed turn; ms from that
            # turn_local leaving the phone to the first frame on the wrist.
            timings = []
            for k, g in enumerate(groups[: len(expected)]):
                ti = expected[k]["turn_index"]
                if ti < len(run.sent_turns):
                    timings.append({"turn_index": ti, "ms": round((g["at"] - run.sent_turns[ti][0]) * 1000.0, 1),
                                    "events": g["events"], "nudges": g["nudges"]})
            other_frames = [f for _, f in watch.frames if f.get("type") not in ("vector_event", "nudge", "live_session_saved")]
            self_turns = set(scene.self_turn_indexes)
            spec = scene.meta.get("expected_nudges") or []
            spec_rows = []
            for entry in spec:
                ti = int(entry.get("after_turn_index", -1))
                k = next((k for k, e in enumerate(expected) if e["turn_index"] == ti), None)
                got = k is not None and k < len(groups)
                level = _channel_a_after(groups, k) if got else 0
                spec_rows.append({"turn_index": ti, "spec": entry.get("level"), "relayed": got,
                                  "level_after": level, "ok": got and _spec_level_ok(entry.get("level"), level)})
            n_events = sum(len(e["events"]) for e in expected)
            n_nudges = sum(len(e["nudges"]) for e in expected)
            report.data["watch"].update({
                "frames": len(watch.frames), "groups": observed_sig, "expected": expected,
                "timing": timings, "timing_p50_ms": _median([t["ms"] for t in timings]),
                "spec": spec_rows, "saved": watch.saved, "close_code": watch.close_code, "error": watch.error,
            })
            report.add("watch ws", watch.saved is not None and not watch.error,
                       f"/ws/live-session/{watch.live_session_id}?{watch_auth}=… open {round(run.wall_seconds + watch_settle_s + 1.0, 1)}s "
                       f"(silent: no PCM/HR), {len(watch.frames)} frames, end -> live_session_saved={watch.saved is not None} "
                       f"status={(watch.saved or {}).get('status')} close={watch.close_code}"
                       + (f" error={watch.error}" if watch.error else "")
                       + (f"; unexpected frames {other_frames[:3]}" if other_frames else ""))
            report.add("watch nudges", match and bool(expected),
                       f"{len(groups)} relayed turns on the wrist vs {len(expected)} the shared relay+NudgePolicy predicts: "
                       + ("MATCH" if match else "MISMATCH")
                       + f" — per self turn {[(e['turn_index'], e['events'], e['nudges']) for e in expected]}; "
                       f"observed {observed_sig}; {n_events} vector_event + {n_nudges} nudge frames expected; "
                       f"other-speaker turns {sorted(set(range(len(scene.turns))) - self_turns)} produced nothing"
                       + ("" if match else f" (identity verdicts disagreeing with the scene: {len(idents) - len(agree)})"))
            report.add("watch nudge timing", True if timings else None,
                       f"turn_local -> wrist frame: " + ", ".join(f"turn {t['turn_index']} {t['ms']:.0f} ms" for t in timings)
                       + (f"; p50 {report.data['watch']['timing_p50_ms']} ms" if timings else "no relayed turn to time"))
            if spec:
                report.add("watch scene spec", all(r["ok"] for r in spec_rows),
                           "expected_nudges: " + "; ".join(
                               f"turn {r['turn_index']} {r['spec']} -> {'relayed' if r['relayed'] else 'NOT relayed'}, "
                               f"channel A at {r['level_after']} (needs >= {WATCH_SPEC_LEVELS.get(str(r['spec']), '?')})"
                               for r in spec_rows))
            # Persisted: the watch's own live session must carry the relayed
            # events with phone provenance (ws.py's emit appends before it sends).
            code, saved_ls = await _req(http, "GET", base_url, f"/live-sessions/{watch.live_session_id}",
                                        Account(email="watch", ws_token="", headers=watch.headers), params=watch.query)
            if code == 200 and isinstance(saved_ls, dict):
                phone_events = [e for e in saved_ls.get("vector_events") or [] if str(e.get("detail", "")).startswith("phone turn")]
                saved_nudges = saved_ls.get("nudge_events") or []
                report.add("watch session persisted", len(phone_events) == n_events and len(saved_nudges) == n_nudges,
                           f"GET /live-sessions/{{id}} as the watch 200 status={saved_ls.get('status')} owner={saved_ls.get('owner_account')}: "
                           f"{len(phone_events)} phone-provenance vector_events (expected {n_events}), "
                           f"{len(saved_nudges)} nudge_events (expected {n_nudges})")
            else:
                report.add("watch session persisted", False, f"GET /live-sessions/{{id}} as the watch -> {code} {str(saved_ls)[:200]}")

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
            if watch is not None:
                code, _ = await _req(http, "DELETE", base_url, f"/live-sessions/{watch.live_session_id}", patient)
                notes.append(f"watch live session delete {code}")
                code, body = await _req(http, "DELETE", base_url, "/me/watch-pairing", patient)
                notes.append(f"unpair watch {code} count={body.get('count') if isinstance(body, dict) else body}")
            for acct in (patient, therapist):
                if acct.signed_up:
                    notes.append(f"firebase delete {acct.email}: {await firebase_delete_account(http, acct)}")
            report.add("cleanup", None, "; ".join(notes))
    finally:
        if fake_watch is not None:
            await fake_watch.close()
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
# --call: two phones, one in-app call (server/calls.py)
# ---------------------------------------------------------------------------
#
# MindShift places the call itself so both sides can be coached: the audio
# is peer-to-peer (WebRTC — signaling relayed by the server, no media through
# it), each phone transcribes ONLY ITS OWNER on-device and reports
# `turn_local` as in a solo session, and the server merges both into one
# shared transcript, pushes every turn to the other phone as a `transcript`
# event (slot labels: host = "Speaker A", joiner = "Speaker B"), coaches
# each participant on the merged context, and at the end persists one
# episode per participant (mode "call") with auto-share through the
# therapist link.
#
# This section plays BOTH phones at once: the patient is the host and
# speaks the scene's self turns; the therapist joins and speaks every other
# turn. Each side streams the scene WAV with the OTHER side's turns silenced
# (a phone's mic hears only its owner) and sends turn_local for its own
# turns only — on the same audio clock, so arrival order is scene order.
# Signaling is exercised with a fake SDP offer/answer (there is no real
# WebRTC stack in a script; the RELAY is what the server owns).

CALL_JOIN_TIMEOUT_S = 30.0


async def _wait_event(run: WsRun, predicate, *, timeout_s: float, after: int = 0) -> dict | None:
    """Poll ``run.events`` (the reader task appends) for the first event at
    index >= ``after`` matching ``predicate``; None on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for _, ev in run.events[after:]:
            if predicate(ev):
                return ev
        await asyncio.sleep(0.02)
    return None


def call_side_pcm(scene: Scene, own_turn_indexes: list[int]) -> np.ndarray:
    """The scene audio as THIS participant's phone hears it: everything
    but its own turns silenced."""
    out = np.zeros_like(scene.pcm)
    for i in own_turn_indexes:
        t = scene.turns[i]
        a, b = int(t["start_time"] * scene.sr), int(t["end_time"] * scene.sr)
        out[a:b] = scene.pcm[a:b]
    return out


def call_side_turn_locals(scene: Scene, session_id: str, own_turn_indexes: list[int]) -> list[dict]:
    """This participant's turn_local reports — its own turns only, labelled
    the way its phone would (its single voice, "Speaker A" locally; the
    server relabels by slot) and self by its own verdict."""
    all_events = build_turn_locals(scene, session_id)
    out = []
    for i in own_turn_indexes:
        ev = dict(all_events[i])
        ev.update({"speaker": "Speaker A", "is_self": True, "speaker_person_id": "self", "speaker_match_score": 0.9})
        out.append(ev)
    return out


@dataclass
class CallSide:
    role: str                  # "host" | "joiner"
    account: Account
    session_id: str
    turn_indexes: list[int]
    display_name: str
    run: WsRun | None = None
    uid: str | None = None     # from call_state.self_uid (the server's view)
    bound_state: dict | None = None
    signal_in: dict | None = None
    signal_out: dict | None = None
    signal_error: str | None = None


def _pre_stream_for(side: CallSide, call_id: str, peer_ready: asyncio.Event, self_ready: asyncio.Event):
    """The call handshake this side runs after config_ack: bind with
    call_join, wait until call_state shows BOTH connected, then the host
    offers and the joiner answers (fake SDP) — proving the relay both ways
    before a single frame is streamed."""
    async def pre(ws, run: WsRun) -> None:
        await ws.send(json.dumps({"type": "call_join", "call_id": call_id, "display_name": side.display_name}))
        state = await _wait_event(run, lambda e: e.get("type") == "call_state" or "error" in e, timeout_s=CALL_JOIN_TIMEOUT_S)
        if state is None or "error" in state:
            raise RuntimeError(f"call_join answered {state}")
        side.uid = state.get("self_uid")
        self_ready.set()
        both = await _wait_event(
            run, lambda e: e.get("type") == "call_state" and len(e.get("participants") or []) == 2
            and all(p.get("connected") for p in e["participants"]),
            timeout_s=CALL_JOIN_TIMEOUT_S,
        )
        if both is None:
            raise RuntimeError("call_state never showed both participants connected")
        side.bound_state = both
        await peer_ready.wait()
        n_before = len(run.events)
        if side.role == "host":
            side.signal_out = {"type": "offer", "sdp": f"v=0 e2e-offer {side.session_id}"}
            await ws.send(json.dumps({"type": "rtc_signal", "call_id": call_id, "payload": side.signal_out}))
            got = await _wait_event(run, lambda e: e.get("type") == "rtc_signal" or "error" in e, timeout_s=CALL_JOIN_TIMEOUT_S)
        else:
            got = await _wait_event(run, lambda e: e.get("type") == "rtc_signal" or "error" in e, timeout_s=CALL_JOIN_TIMEOUT_S, after=n_before)
            if got is not None and got.get("type") == "rtc_signal":
                side.signal_out = {"type": "answer", "sdp": f"v=0 e2e-answer {side.session_id}"}
                await ws.send(json.dumps({"type": "rtc_signal", "call_id": call_id, "to": got.get("from"), "payload": side.signal_out}))
        if got is None:
            side.signal_error = "no rtc_signal arrived"
        elif "error" in got:
            side.signal_error = str(got["error"])
        else:
            side.signal_in = got
    return pre


def _post_stream_for(side: CallSide, call_id: str, self_done: asyncio.Event, peer_done: asyncio.Event, n_peer_turns: int):
    """Before hanging up: say we are done, wait for the peer to be done,
    and give the peer's last turn a moment to land as a transcript — a
    person does not hang up mid-sentence."""
    async def post(ws, run: WsRun) -> None:
        self_done.set()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(peer_done.wait(), timeout=CALL_JOIN_TIMEOUT_S)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            got = [e for _, e in run.events if e.get("type") == "transcript" and e.get("call_id") == call_id]
            if len(got) >= n_peer_turns:
                break
            await asyncio.sleep(0.05)
    return post


async def run_call_e2e(
    *, base_url: str, patient: Account, therapist: Account, scene: Scene,
    speed: float = 1.0, analysis_timeout_s: float = 180.0, cleanup: bool = False,
    http: httpx.AsyncClient | None = None, session_id: str | None = None,
    link_therapist: bool = True,
) -> Report:
    """The in-app call walk: link → POST /calls → join by code → both
    sockets bind + signal → both stream their own turns concurrently → the
    merged transcript on each phone → per-participant coaching → both stop
    → GET /calls/{id} (ended, one episode each) → the patient's episode
    (analysis, reflection, detail, growth) → the therapist sees the
    patient's episode through the auto-share AND has its own."""
    report = Report(scene=scene.name, base_url=base_url, speed=speed, mode="call")
    own_http = http is None
    http = http or httpx.AsyncClient()
    session_id = session_id or f"e2e-call-{scene.name}-{int(time.time() * 1000)}"
    report.data["session_id"] = session_id
    report.data["accounts"] = {"patient": patient.email, "therapist": therapist.email}
    report.add("accounts", True, f"patient (host) {patient.email} | therapist (joiner) {therapist.email}")
    self_idx = scene.self_turn_indexes
    other_idx = [i for i in range(len(scene.turns)) if i not in self_idx]
    host = CallSide("host", patient, f"{session_id}-a", self_idx, "Patient")
    joiner = CallSide("joiner", therapist, f"{session_id}-b", other_idx, "Therapist")
    call_id: str | None = None
    linked = False
    try:
        # --- 0. the therapist link (so the call episode auto-shares) ---------
        if link_therapist:
            code, body = await _req(http, "PUT", base_url, "/therapist/link", patient, json={"email": therapist.email})
            linked = code == 200 and isinstance(body, dict) and body.get("linked") is True
            report.add("therapist link", True if linked else None,
                       f"PUT /therapist/link -> {code} {body if isinstance(body, dict) else str(body)[:120]}"
                       + ("" if linked else " — the call episode will need a manual share"))

        # --- 1. create + join ------------------------------------------------
        code, created = await _req(http, "POST", base_url, "/calls", patient,
                                   json={"invitee_email": therapist.email, "display_name": host.display_name})
        if code != 201 or not isinstance(created, dict):
            report.add("call create", False, f"POST /calls -> {code} {str(created)[:300]}")
            return report
        call_id = created["call_id"]
        report.data["call"] = {"call_id": call_id, "join_code": created.get("join_code"), "join_url": created.get("join_url")}
        report.add("call create", True,
                   f"POST /calls 201 call {call_id} code {created.get('join_code')} url {created.get('join_url')} "
                   f"invitee={created.get('invitee_email')} ice={[s.get('urls') for s in created.get('ice_servers') or []]}")
        code, joined = await _req(http, "POST", base_url, "/calls/join", therapist,
                                  json={"join_code": created["join_code"], "display_name": joiner.display_name})
        if code != 200 or not isinstance(joined, dict) or joined.get("status") != "active":
            report.add("call join", False, f"POST /calls/join (as therapist, by code) -> {code} {str(joined)[:300]}")
            return report
        report.add("call join", True,
                   f"POST /calls/join 200 as the therapist: status {joined.get('status')}, self_label {joined.get('self_label')}, "
                   f"participants {[(p.get('display_name'), p.get('label')) for p in joined.get('participants') or []]}")

        # --- 2. both phones on the call, concurrently -------------------------
        host_ready, joiner_ready = asyncio.Event(), asyncio.Event()
        host_done, joiner_done = asyncio.Event(), asyncio.Event()

        async def side_run(side: CallSide, self_ready, peer_ready, self_done, peer_done, n_peer_turns: int) -> WsRun:
            return await stream_live_session(
                base_url, side.account, scene, session_id=side.session_id, speed=speed,
                pcm=call_side_pcm(scene, side.turn_indexes),
                turn_locals=call_side_turn_locals(scene, side.session_id, side.turn_indexes),
                pre_stream=_pre_stream_for(side, call_id, peer_ready, self_ready),
                post_stream=_post_stream_for(side, call_id, self_done, peer_done, n_peer_turns),
            )

        host.run, joiner.run = await asyncio.gather(
            side_run(host, host_ready, joiner_ready, host_done, joiner_done, len(other_idx)),
            side_run(joiner, joiner_ready, host_ready, joiner_done, host_done, len(self_idx)),
        )
        for side in (host, joiner):
            run = side.run
            ok = run.error is None and run.config_ack and run.session_complete is not None and side.bound_state is not None
            report.data[f"ws_{side.role}"] = {
                "frames_sent": run.frames_sent, "turn_locals": len(run.sent_turns), "close_code": run.close_code,
                "error": run.error, "event_counts": _count_types(run), "uid": side.uid,
            }
            report.add(f"call ws ({side.role})", ok,
                       f"{side.account.email}: config_ack={run.config_ack} bound={side.bound_state is not None} "
                       f"{run.frames_sent} frames, {len(run.sent_turns)} turn_local (own turns {side.turn_indexes}), "
                       f"session_complete={run.session_complete is not None} close={run.close_code} error={run.error} "
                       f"events={_count_types(run)}")
        if host.run.error or joiner.run.error or host.bound_state is None or joiner.bound_state is None:
            return report

        # signaling: the offer reached the joiner from the host, the answer came back
        offer_ok = (host.signal_error is None and joiner.signal_in is not None
                    and joiner.signal_in.get("payload") == host.signal_out and joiner.signal_in.get("from") == host.uid)
        answer_ok = (joiner.signal_error is None and host.signal_in is not None
                     and host.signal_in.get("payload") == joiner.signal_out and host.signal_in.get("from") == joiner.uid)
        report.data["signaling"] = {"offer_ok": offer_ok, "answer_ok": answer_ok,
                                    "host_error": host.signal_error, "joiner_error": joiner.signal_error}
        report.add("rtc signaling", offer_ok and answer_ok,
                   f"host offer -> joiner: {'delivered' if offer_ok else 'MISSING'} (from={joiner.signal_in.get('from') if joiner.signal_in else None}); "
                   f"joiner answer -> host: {'delivered' if answer_ok else 'MISSING'}"
                   + (f"; errors host={host.signal_error} joiner={joiner.signal_error}" if host.signal_error or joiner.signal_error else ""))

        # the merged transcript on each phone: every PEER turn, with the peer's slot label
        def _remote(side: CallSide) -> list[dict]:
            return [e for e in side.run.of_type("transcript") if e.get("call_id") == call_id]
        expect = [(host, "Speaker B", other_idx), (joiner, "Speaker A", self_idx)]
        merged_ok = True
        merged_txt = []
        for side, peer_label, peer_idx in expect:
            got = _remote(side)
            texts_ok = sorted(e.get("text") for e in got) == sorted(scene.turns[i]["text"] for i in peer_idx)
            labels_ok = all(e.get("speaker") == peer_label and e.get("is_self") is False for e in got)
            in_order = [e.get("text") for e in got] == [scene.turns[i]["text"] for i in peer_idx]
            names = sorted({e.get("display_name") for e in got})
            merged_ok = merged_ok and texts_ok and labels_ok
            merged_txt.append(
                f"{side.role} saw {len(got)}/{len(peer_idx)} peer turns as {peer_label} named {names}"
                f"{'' if in_order else ' (order differs from the scene)'}"
            )
            report.data[f"merged_{side.role}"] = {"count": len(got), "expected": len(peer_idx), "in_order": in_order, "names": names}
        report.add("merged transcript", merged_ok, "; ".join(merged_txt))

        # per-participant coaching: nudges on own turns, suggestions on the peer's
        coach_ok = True
        coach_txt = []
        for side, peer_label, _peer_idx in expect:
            finals = [s for s in side.run.of_type("suggestion") if not s.get("partial")]
            nudges = [s for s in finals if s.get("kind") == "nudge"]
            responses = [s for s in finals if s.get("kind", "response") == "response"]
            wrong = [s for s in responses if s.get("speaker") != peer_label] + [s for s in nudges if s.get("speaker") == peer_label]
            errors = side.run.of_type("suggestion_error")
            hard = [e for e in errors if e.get("reason") != "llm_parse_error"]
            side_ok = bool(responses) and not wrong and not hard
            coach_ok = coach_ok and side_ok
            coach_txt.append(
                f"{side.role}: {len(responses)} suggestions about {peer_label}, {len(nudges)} nudges on own turns"
                + (f", {len(wrong)} MISLABELLED" if wrong else "") + (f", {len(errors)} suggestion_error" if errors else "")
            )
            report.data[f"coaching_{side.role}"] = {"responses": len(responses), "nudges": len(nudges), "errors": len(errors)}
        report.add("per-participant coaching", coach_ok,
                   "; ".join(coach_txt) + (f" (speed {speed:g}x: latest-wins supersedes some turns)" if speed > 1 else ""))

        # --- 3. the call ended, one episode each -----------------------------
        code, h_view = await _req(http, "GET", base_url, f"/calls/{call_id}", patient)
        code_j, j_view = await _req(http, "GET", base_url, f"/calls/{call_id}", therapist)
        if code != 200 or code_j != 200 or not isinstance(h_view, dict) or not isinstance(j_view, dict):
            report.add("call ended", False, f"GET /calls/{{id}} -> host {code} / joiner {code_j}")
            return report
        h_ep, j_ep = h_view.get("episode_id"), j_view.get("episode_id")
        ended_frames = [side.run.of_type("call_ended") for side in (host, joiner)]
        ended_ok = h_view.get("status") == "ended" and bool(h_ep) and bool(j_ep) and h_ep != j_ep
        report.data["episodes"] = {"host": h_ep, "joiner": j_ep, "shared_with": h_view.get("shared_with")}
        report.add("call ended", ended_ok,
                   f"GET /calls/{{id}}: status {h_view.get('status')} ({h_view.get('end_reason')}), {h_view.get('turn_count')} merged turns, "
                   f"host episode {h_ep} shared_with={h_view.get('shared_with')}, joiner episode {j_ep}; "
                   f"call_ended frames on the sockets: {[len(f) for f in ended_frames]}")
        if linked and h_view.get("shared_with") != [therapist.email]:
            report.add("call auto-share", False, f"host episode shared_with={h_view.get('shared_with')}, expected [{therapist.email}]")
        if not ended_ok:
            return report
        episode_id = h_ep
        report.data["episode_id"] = episode_id

        # --- 4. the patient's episode ----------------------------------------
        status, detail, waited = await wait_for_analysis(http, base_url, patient, episode_id, timeout_s=analysis_timeout_s)
        live = ((detail or {}).get("analysis") or {}).get("live") or {}
        if status == "full":
            report.add("batch analysis", True, f"full after {waited:.1f}s — {len((detail or {}).get('analysis', {}).get('per_turn') or [])} heats")
        elif status == "failed":
            report.add("batch analysis", False, f"analysis_status failed after {waited:.1f}s: {live.get('analysis_error')}")
        else:
            report.add("batch analysis", False, f"{status} after {waited:.1f}s (analysis_status={live.get('analysis_status')})")

        code, body = await _req(http, "POST", base_url, f"/episodes/{episode_id}/reflect", patient, timeout=180.0)
        refl = (body.get("could_have_said") or []) if code == 200 and isinstance(body, dict) else []
        report.add("reflection", code == 200 and bool(refl),
                   f"POST /episodes/{{id}}/reflect -> {code}: {len(refl)} reflections for {len(self_idx)} own turns (cached={body.get('cached') if isinstance(body, dict) else None})")

        code, detail = await _req(http, "GET", base_url, f"/recordings/{episode_id}", patient)
        if code == 200 and isinstance(detail, dict):
            turns = detail.get("turns") or []
            labels = detail.get("speaker_labels") or {}
            live = ((detail.get("analysis") or {}).get("live") or {})
            esc = ((live.get("tone_summary") or {}).get("self") or {}).get("escalation_turns")
            exp_esc_texts = sorted(scene.turns[i]["text"] for i in scene.expected_self_escalations)
            esc_texts = sorted(turns[i].get("text") for i in (esc or []) if i < len(turns))
            in_order = [t.get("text") for t in turns] == [t["text"] for t in scene.turns]
            ok_detail = (
                detail.get("mode") == "call" and live.get("mode") == "call"
                and sorted(t.get("text") for t in turns) == sorted(t["text"] for t in scene.turns)
                and (labels.get("Speaker A") or {}).get("display_label") == "You"
                and live.get("self_speaker") == "Speaker A"
                and esc_texts == exp_esc_texts
                and all(t.get("is_self") is (t.get("speaker") == "Speaker A") for t in turns)
                and all("call_seq" in t and "local_start_time" in t for t in turns)
            )
            report.data["detail"] = {"mode": detail.get("mode"), "turns": len(turns), "in_order": in_order,
                                     "escalation_turns": esc, "self_speaker": live.get("self_speaker"),
                                     "peer_label": (labels.get("Speaker B") or {}).get("display_label")}
            report.add("episode detail (patient)", ok_detail,
                       f"GET /recordings/{{id}} 200 — mode {detail.get('mode')}, {len(turns)}/{len(scene.turns)} merged turns"
                       f"{'' if in_order else ' (order differs from the scene)'}, Speaker A labelled "
                       f"{(labels.get('Speaker A') or {}).get('display_label')!r}, Speaker B labelled "
                       f"{(labels.get('Speaker B') or {}).get('display_label')!r}, self escalation turns {esc} "
                       f"(scene expects {scene.expected_self_escalations}), analysis_status={live.get('analysis_status')}")
        else:
            report.add("episode detail (patient)", False, f"GET /recordings/{{id}} -> {code} {str(detail)[:200]}")

        code, growth = await _req(http, "GET", base_url, "/growth", patient)
        pt = next((p for p in (growth.get("points") or []) if p.get("recording_id") == episode_id), None) if code == 200 and isinstance(growth, dict) else None
        report.data["growth_point"] = pt
        report.add("growth", pt is not None and pt.get("mode") == "call",
                   f"GET /growth -> {code}: " + (f"point present, my_score={pt.get('my_score')} mode={pt.get('mode')} source={pt.get('source')}" if pt else "no point for the episode"))

        # --- 5. the therapist: the patient's episode (shared) + its own -------
        code, sessions = await _req(http, "GET", base_url, "/sessions", therapist)
        rows = (sessions.get("sessions") if isinstance(sessions, dict) else None) or []
        shared_row = next((s for s in rows if s.get("id") == episode_id), None)
        own_row = next((s for s in rows if s.get("id") == j_ep), None)
        if shared_row is None and not linked:
            code_s, body = await _req(http, "POST", base_url, f"/recordings/{episode_id}/shares", patient, json={"email": therapist.email})
            report.add("share (manual)", code_s == 200, f"POST /recordings/{{id}}/shares -> {code_s}")
            code, sessions = await _req(http, "GET", base_url, "/sessions", therapist)
            rows = (sessions.get("sessions") if isinstance(sessions, dict) else None) or []
            shared_row = next((s for s in rows if s.get("id") == episode_id), None)
            own_row = next((s for s in rows if s.get("id") == j_ep), None)
        ok_rows = (
            code == 200 and shared_row is not None and bool(shared_row.get("shared")) and shared_row.get("mode") == "call"
            and own_row is not None and not own_row.get("shared") and own_row.get("mode") == "call"
        )
        report.data["therapist_rows"] = {
            "shared": {k: shared_row.get(k) for k in ("shared", "role", "mode", "analysisStatus")} if shared_row else None,
            "own": {k: own_row.get(k) for k in ("shared", "role", "mode", "analysisStatus")} if own_row else None,
        }
        report.add("therapist visibility", ok_rows,
                   f"GET /sessions as therapist -> {code}: patient's call episode "
                   + (f"listed shared={shared_row.get('shared')} role={shared_row.get('role')!r} mode={shared_row.get('mode')}" if shared_row else "MISSING")
                   + "; therapist's own call episode "
                   + (f"listed role={own_row.get('role')!r} mode={own_row.get('mode')} couldHaveSaid={len(own_row.get('couldHaveSaid') or [])}" if own_row else "MISSING"))

        # --- cleanup --------------------------------------------------------
        if cleanup:
            notes = []
            for acct, ep in ((patient, h_ep), (therapist, j_ep)):
                code, _ = await _req(http, "DELETE", base_url, f"/recordings/{ep}", acct)
                notes.append(f"episode {ep} delete {code}")
            if linked:
                code, _ = await _req(http, "DELETE", base_url, "/therapist/link", patient)
                notes.append(f"unlink {code}")
            for acct in (patient, therapist):
                if acct.signed_up:
                    notes.append(f"firebase delete {acct.email}: {await firebase_delete_account(http, acct)}")
            report.add("cleanup", None, "; ".join(notes))
    finally:
        if own_http:
            await http.aclose()
    return report


# ---------------------------------------------------------------------------
# --call --participants 3: two coached participants + an observing therapist
# ---------------------------------------------------------------------------
#
# Three phones on one call. The host (Sage, slot A) and a second participant
# (Dad, slot B — the named invitee, joins over REST without a code) are the
# coached ones; the therapist (Mom, slot C) joins as an OBSERVER over her own
# socket (`call_join` with the join code and role "therapist"). Her turns are
# transcribed and merged like anyone's, nobody coaches her, and she receives
# every participant's coaching read-only, tagged `for_uid`. Signaling is a
# full mesh (`to` is required with three members; an unaddressed frame is an
# error). Hang-up order is host → Dad → Mom, so the participants' sockets see
# the call still active when they leave (their episode comes from GET
# /calls/{id}) and HER socket — the last one — ends the call and receives
# `call_ended` with both participants' episodes. Both are granted to her
# directly (no therapist link is created), and she gets no episode of her own.

THERAPIST_LINES: list[tuple[int, str]] = [
    # (scene turn index she speaks after, what she says) — short, in the gaps
    (1, "Take a breath, both of you."),
    (6, "Let's pause there for a second."),
    (9, "Can you each say what you need right now?"),
]
THERAPIST_TURN_S = 1.2
# Her line is finalized this long after the turn it follows ended: on the
# streaming clock that is two 100 ms frames after that turn's turn_local went
# out, so it lands between that turn and the next one.
THERAPIST_GAP_S = 0.2
MISSING_TO_ERROR = "rtc_signal: 'to' is required in a call with more than two members"
CALL_HANGUP_TIMEOUT_S = 90.0


def therapist_turn_locals(scene: Scene, session_id: str, lines: list[tuple[int, str]] = THERAPIST_LINES) -> list[dict]:
    """The observer's own turn_locals (her phone hears only her): a calm
    line after each of the scene turns in ``lines``, on her capture clock."""
    out = []
    for after_idx, text in lines:
        end = round(scene.turns[after_idx]["end_time"] + THERAPIST_GAP_S, 4)
        start = round(max(0.0, end - THERAPIST_TURN_S), 4)
        words = len(text.split())
        out.append({
            "type": "turn_local",
            "session_id": session_id,
            "speaker": "Speaker A",
            "speaker_person_id": "self",
            "speaker_match_score": 0.9,
            "is_self": True,
            "text": text,
            "start_time": start,
            "end_time": end,
            "transcript_source": "on-device",
            "prosody": {"rms_dbfs": None, "pitch_hz": None, "speech_rate": round(words / (end - start), 3)},
            "text_tone": dict(TEXT_TONE_BY_EMOTION["calm_deescalate"]),
            "suggestion": None,
            "suggestion_source": None,
            "tts_source": "on-device",
        })
    return out


@dataclass
class MeshMember:
    role: str                      # "host" | "peer" | "therapist" (this walk's names)
    call_role: str                 # "participant" | "therapist" (the server's)
    account: Account
    session_id: str
    display_name: str
    turn_locals: list[dict]
    pcm: np.ndarray
    join: dict = field(default_factory=dict)      # extra call_join fields (join_code / role)
    run: WsRun | None = None
    uid: str | None = None
    label: str | None = None
    first_state: dict | None = None               # the call_state that answered call_join
    bound_state: dict | None = None               # the call_state showing every member connected
    signals_out: dict[str, dict] = field(default_factory=dict)   # to uid -> payload sent
    missing_to_error: str | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def events(self, kind: str, call_id: str) -> list[dict]:
        if self.run is None:
            return []
        return [e for e in self.run.of_type(kind) if e.get("call_id") == call_id]

    @property
    def own_texts(self) -> list[str]:
        return [t["text"] for t in self.turn_locals]


def _pctl(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 1)


def transcript_delivery_ms(members: list[MeshMember], call_id: str) -> dict:
    """Per (sender turn, other viewer): ms from the turn_local going out to
    that viewer's ``transcript`` event for it — the relay's own latency."""
    deltas: list[float] = []
    rows: list[dict] = []
    for m in members:
        if m.run is None:
            continue
        for sent_at, ev in m.run.sent_turns:
            for o in members:
                if o is m or o.run is None:
                    continue
                hit = next((
                    at for at, e in o.run.events
                    if e.get("type") == "transcript" and e.get("call_id") == call_id
                    and e.get("participant_uid") == m.uid and e.get("text") == ev["text"] and at >= sent_at
                ), None)
                ms = None if hit is None else round((hit - sent_at) * 1000.0, 1)
                if ms is not None:
                    deltas.append(ms)
                rows.append({"from": m.role, "to": o.role, "text": ev["text"][:30], "ms": ms})
    return {
        "n": len(deltas), "expected": len(rows),
        "p50_ms": _pctl(deltas, 0.5), "p95_ms": _pctl(deltas, 0.95),
        "max_ms": round(max(deltas), 1) if deltas else None,
        "per_delivery": rows,
    }


def _mesh_pre_stream(m: MeshMember, members: list[MeshMember], call_id: str, go: asyncio.Event, arrived: list[str]):
    """Bind (the therapist joins here, with the code), wait until call_state
    shows every member connected, then the full-mesh signaling: the host
    first proves an unaddressed frame is refused, then everyone sends one
    addressed offer to each other member and waits for the two it is owed.
    Streaming starts together (a barrier) so arrival order is scene order."""
    others = [o for o in members if o is not m]

    async def pre(ws, run: WsRun) -> None:
        m.run = run
        await ws.send(json.dumps({"type": "call_join", "call_id": call_id, "display_name": m.display_name, **m.join}))
        state = await _wait_event(run, lambda e: e.get("type") == "call_state" or "error" in e, timeout_s=CALL_JOIN_TIMEOUT_S)
        if state is None or "error" in state:
            raise RuntimeError(f"call_join answered {state}")
        m.first_state, m.uid, m.label = state, state.get("self_uid"), state.get("self_label")
        n = len(members)
        full = await _wait_event(
            run, lambda e: e.get("type") == "call_state" and len(e.get("participants") or []) == n
            and all(p.get("connected") for p in e["participants"]),
            timeout_s=CALL_JOIN_TIMEOUT_S,
        )
        if full is None:
            raise RuntimeError(f"call_state never showed all {n} members connected")
        m.bound_state = full
        m.ready.set()
        for o in others:
            await asyncio.wait_for(o.ready.wait(), timeout=CALL_JOIN_TIMEOUT_S)
        other_uids = [p["uid"] for p in full["participants"] if p["uid"] != m.uid]
        if m.role == "host":
            n_before = len(run.events)
            await ws.send(json.dumps({"type": "rtc_signal", "call_id": call_id, "payload": {"type": "offer", "sdp": "v=0 unaddressed"}}))
            err = await _wait_event(run, lambda e: e.get("type") is None and "error" in e, timeout_s=CALL_JOIN_TIMEOUT_S, after=n_before)
            m.missing_to_error = None if err is None else str(err.get("error"))
        for to_uid in other_uids:
            payload = {"type": "offer", "sdp": f"v=0 e2e-offer {m.session_id} -> {to_uid}"}
            m.signals_out[to_uid] = payload
            await ws.send(json.dumps({"type": "rtc_signal", "call_id": call_id, "to": to_uid, "payload": payload}))
        deadline = time.monotonic() + CALL_JOIN_TIMEOUT_S
        while len(m.events("rtc_signal", call_id)) < len(others) and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        arrived.append(m.role)
        if len(arrived) == n:
            go.set()
        await asyncio.wait_for(go.wait(), timeout=CALL_JOIN_TIMEOUT_S)
    return pre


def _mesh_post_stream(m: MeshMember, members: list[MeshMember], call_id: str, hang_up_after: MeshMember | None):
    """Before hanging up: wait for everyone to finish talking and for their
    last turns to land, then for ``hang_up_after`` to have left (its
    session_complete) and for the call_state that shows it disconnected."""
    others = [o for o in members if o is not m]
    n_other_turns = sum(len(o.turn_locals) for o in others)

    async def post(ws, run: WsRun) -> None:
        m.done.set()
        for o in others:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(o.done.wait(), timeout=CALL_JOIN_TIMEOUT_S)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(m.events("transcript", call_id)) < n_other_turns:
            await asyncio.sleep(0.05)
        if hang_up_after is not None:
            deadline = time.monotonic() + CALL_HANGUP_TIMEOUT_S
            while time.monotonic() < deadline:
                prev = hang_up_after.run
                if prev is not None and (prev.session_complete is not None or prev.error is not None):
                    break
                await asyncio.sleep(0.05)
            await _wait_event(
                run, lambda e: e.get("type") == "call_state" and any(
                    p.get("uid") == hang_up_after.uid and not p.get("connected") for p in e.get("participants") or []
                ), timeout_s=5.0,
            )
    return post


async def run_call_e2e_three_way(
    *, base_url: str, patient: Account, peer: Account, therapist: Account, scene: Scene,
    speed: float = 1.0, analysis_timeout_s: float = 180.0, cleanup: bool = False,
    http: httpx.AsyncClient | None = None, session_id: str | None = None,
) -> Report:
    """The three-way call walk (see the section comment above): POST /calls
    (Dad invited) → Dad joins over REST → three sockets bind (Mom joins on
    hers with the code as the therapist) → full-mesh signaling → all three
    speak concurrently (host = Speaker A turns, Dad = Speaker B turns, Mom
    three short lines) → merged transcript per viewer, coaching per
    participant, read-only copies for Mom → host, Dad, then Mom hang up →
    two participant episodes (Mom's dashboard lists both, none of her own)."""
    report = Report(scene=scene.name, base_url=base_url, speed=speed, mode="call-3way")
    own_http = http is None
    http = http or httpx.AsyncClient()
    session_id = session_id or f"e2e-call3-{scene.name}-{int(time.time() * 1000)}"
    report.data["session_id"] = session_id
    report.data["accounts"] = {"patient": patient.email, "peer": peer.email, "therapist": therapist.email}
    report.add("accounts", True, f"host {patient.email} | second participant {peer.email} | therapist {therapist.email}")
    a_idx = scene.self_turn_indexes
    b_idx = [i for i in range(len(scene.turns)) if i not in a_idx]
    host = MeshMember("host", "participant", patient, f"{session_id}-a", "Sage",
                      call_side_turn_locals(scene, f"{session_id}-a", a_idx), call_side_pcm(scene, a_idx))
    dad = MeshMember("peer", "participant", peer, f"{session_id}-b", "Dad",
                     call_side_turn_locals(scene, f"{session_id}-b", b_idx), call_side_pcm(scene, b_idx))
    mom = MeshMember("therapist", "therapist", therapist, f"{session_id}-c", "Mom",
                     therapist_turn_locals(scene, f"{session_id}-c"), np.zeros_like(scene.pcm))
    members = [host, dad, mom]
    by_role = {m.role: m for m in members}
    call_id: str | None = None
    h_ep = d_ep = None
    try:
        # --- 1. create + Dad joins over REST (Mom joins on her socket) -------
        code, created = await _req(http, "POST", base_url, "/calls", patient,
                                   json={"invitee_email": peer.email, "display_name": host.display_name, "max_participants": 3})
        if code != 201 or not isinstance(created, dict):
            report.add("call create", False, f"POST /calls -> {code} {str(created)[:300]}")
            return report
        call_id = created["call_id"]
        join_code = created.get("join_code")
        mom.join = {"join_code": join_code, "role": "therapist"}
        report.data["call"] = {"call_id": call_id, "join_code": join_code, "join_url": created.get("join_url")}
        report.add("call create", created.get("max_participants") == 3 and created.get("therapist_label") == "Speaker C",
                   f"POST /calls 201 call {call_id} code {join_code} max_participants={created.get('max_participants')} "
                   f"invitee={created.get('invitee_email')} therapist_label={created.get('therapist_label')} "
                   f"ice={[s.get('urls') for s in created.get('ice_servers') or []]}")
        code, joined = await _req(http, "POST", base_url, f"/calls/{call_id}/join", peer, json={"display_name": dad.display_name})
        rows = {p.get("display_name"): p for p in (joined.get("participants") or [])} if isinstance(joined, dict) else {}
        join_ok = (
            code == 200 and isinstance(joined, dict) and joined.get("status") == "active"
            and joined.get("self_role") == "participant" and joined.get("self_label") == "Speaker B"
            and joined.get("peer_label") == "Speaker A" and joined.get("therapist_uid") is None
            and (rows.get("Sage") or {}).get("label") == "Speaker A" and (rows.get("You") or {}).get("label") == "Speaker B"
        )
        report.add("call join (Dad, invitee, no code)", join_ok,
                   f"POST /calls/{{id}}/join -> {code}: status {joined.get('status') if isinstance(joined, dict) else joined}, "
                   f"self_label {joined.get('self_label') if isinstance(joined, dict) else None}, "
                   f"participants {[(p.get('display_name'), p.get('label'), p.get('role')) for p in (joined.get('participants') or [])] if isinstance(joined, dict) else None}")
        if not join_ok:
            return report

        # --- 2. three phones on the call, concurrently -----------------------
        go, arrived = asyncio.Event(), []
        hang_up_after = {"host": None, "peer": host, "therapist": dad}

        async def member_run(m: MeshMember) -> WsRun:
            return await stream_live_session(
                base_url, m.account, scene, session_id=m.session_id, speed=speed,
                pcm=m.pcm, turn_locals=m.turn_locals,
                pre_stream=_mesh_pre_stream(m, members, call_id, go, arrived),
                post_stream=_mesh_post_stream(m, members, call_id, hang_up_after[m.role]),
                stop_timeout_s=CALL_HANGUP_TIMEOUT_S,
            )

        runs = await asyncio.gather(*(member_run(m) for m in members))
        for m, run in zip(members, runs):
            m.run = run
            ok = run.error is None and run.config_ack and run.session_complete is not None and m.bound_state is not None and run.close_code == 1000
            report.data[f"ws_{m.role}"] = {
                "frames_sent": run.frames_sent, "turn_locals": len(run.sent_turns), "close_code": run.close_code,
                "error": run.error, "event_counts": _count_types(run), "uid": m.uid, "label": m.label,
            }
            report.add(f"call ws ({m.role})", ok,
                       f"{m.account.email} as {m.label}: config_ack={run.config_ack} bound={m.bound_state is not None} "
                       f"{run.frames_sent} frames, {len(run.sent_turns)}/{len(m.turn_locals)} turn_local, "
                       f"session_complete={run.session_complete is not None} close={run.close_code} error={run.error} "
                       f"events={_count_types(run)}")
        if any(m.run.error or m.bound_state is None for m in members):
            return report
        uids = {m.role: m.uid for m in members}

        # --- call_state: roles, labels, relative names, connected transitions --
        expect_names = {
            "host": {"peer": "Dad", "therapist": "Mom (therapist)", "host": "You"},
            "peer": {"host": "Sage", "therapist": "Mom (therapist)", "peer": "You"},
            "therapist": {"host": "Sage", "peer": "Dad", "therapist": "You"},
        }
        expect_labels = {"host": ("participant", "Speaker A", "Speaker B"), "peer": ("participant", "Speaker B", "Speaker A"),
                         "therapist": ("therapist", "Speaker C", "Speaker A")}
        state_problems: list[str] = []
        for m in members:
            role, label, peer_label = expect_labels[m.role]
            st = m.bound_state
            if (m.first_state.get("self_role"), m.first_state.get("self_label"), st.get("peer_label")) != (role, label, peer_label):
                state_problems.append(f"{m.role}: self_role/self_label/peer_label = {m.first_state.get('self_role')}/{m.first_state.get('self_label')}/{st.get('peer_label')}")
            if st.get("status") != "active" or st.get("therapist_uid") != uids["therapist"] or st.get("therapist_label") != "Speaker C":
                state_problems.append(f"{m.role}: status={st.get('status')} therapist_uid ok={st.get('therapist_uid') == uids['therapist']}")
            rows = {p["uid"]: p for p in st.get("participants") or []}
            for other_role, name in expect_names[m.role].items():
                row = rows.get(uids[other_role]) or {}
                exp_role, exp_label, _ = expect_labels[other_role]
                if (row.get("display_name"), row.get("role"), row.get("label"), row.get("is_self")) != (name, exp_role, exp_label, other_role == m.role):
                    state_problems.append(f"{m.role} sees {other_role} as {row.get('display_name')!r}/{row.get('role')}/{row.get('label')}/is_self={row.get('is_self')} (want {name!r})")
            first_rows = {p["uid"]: p for p in m.first_state.get("participants") or []}
            if not (first_rows.get(m.uid) or {}).get("connected"):
                state_problems.append(f"{m.role}: the call_state answering call_join does not show it connected")

        def _saw_disconnect(viewer: MeshMember, left: MeshMember) -> bool:
            return any(
                any(p.get("uid") == left.uid and p.get("connected") is False for p in e.get("participants") or [])
                and any(p.get("uid") == viewer.uid and p.get("connected") for p in e.get("participants") or [])
                for e in viewer.events("call_state", call_id)
            )
        transitions = {
            "peer saw host leave": _saw_disconnect(dad, host),
            "therapist saw host leave": _saw_disconnect(mom, host),
            "therapist saw peer leave": _saw_disconnect(mom, dad),
        }
        for name, ok in transitions.items():
            if not ok:
                state_problems.append(f"no call_state where {name}")
        report.data["call_state"] = {"problems": state_problems, "transitions": transitions,
                                     "n_frames": {m.role: len(m.events("call_state", call_id)) for m in members}}
        report.add("call_state (roles, labels, names, transitions)", not state_problems,
                   f"host A/participant, Dad B/participant, Mom C/therapist; relative names as expected on all three; "
                   f"transitions {transitions}; call_state frames {report.data['call_state']['n_frames']}"
                   + (f"; PROBLEMS: {state_problems}" if state_problems else ""))

        # --- full-mesh signaling --------------------------------------------
        mesh_problems: list[str] = []
        for m in members:
            got = m.events("rtc_signal", call_id)
            for o in members:
                if o is m:
                    continue
                want = {"type": "rtc_signal", "call_id": call_id, "from": o.uid, "payload": o.signals_out.get(m.uid)}
                if want not in got:
                    mesh_problems.append(f"{o.role} -> {m.role} MISSING")
            if len(got) != len(members) - 1:
                mesh_problems.append(f"{m.role} received {len(got)} rtc_signal frames (want {len(members) - 1})")
            errs = [e for _, e in m.run.events if e.get("type") is None and "error" in e]
            expected_errs = 1 if m.role == "host" else 0
            if len(errs) != expected_errs:
                mesh_problems.append(f"{m.role}: {len(errs)} error frames {errs[:3]}")
        if host.missing_to_error != MISSING_TO_ERROR:
            mesh_problems.append(f"unaddressed rtc_signal answered {host.missing_to_error!r}")
        report.data["signaling"] = {"problems": mesh_problems, "missing_to_error": host.missing_to_error,
                                    "delivered": {m.role: len(m.events("rtc_signal", call_id)) for m in members}}
        report.add("rtc signaling (full mesh)", not mesh_problems,
                   f"6 addressed offers delivered verbatim with the right `from` ({report.data['signaling']['delivered']}); "
                   f"unaddressed frame with 3 members -> {host.missing_to_error!r}"
                   + (f"; PROBLEMS: {mesh_problems}" if mesh_problems else ""))

        # --- the merged transcript per viewer -------------------------------
        merged_problems: list[str] = []
        merged_txt: list[str] = []
        for m in members:
            got = m.events("transcript", call_id)
            by_sender: dict[str, list[dict]] = {}
            for e in got:
                by_sender.setdefault(e.get("participant_uid"), []).append(e)
            for o in members:
                if o is m:
                    continue
                mine = by_sender.pop(o.uid, [])
                texts = [e.get("text") for e in mine]
                if sorted(texts) != sorted(o.own_texts):
                    merged_problems.append(f"{m.role} saw {len(mine)}/{len(o.own_texts)} of {o.role}'s turns")
                bad = [e for e in mine if e.get("speaker") != o.label or e.get("is_self") is not False
                       or e.get("role") != o.call_role or e.get("display_name") != expect_names[m.role][o.role]]
                if bad:
                    merged_problems.append(f"{m.role} saw {o.role}'s turns as {sorted({(e.get('speaker'), e.get('display_name'), e.get('role'), e.get('is_self')) for e in bad})}")
                untagged = [e for e in mine if e.get("text_tone") is None or "local_start_time" not in e or e.get("seq") is None]
                if untagged:
                    # A phone's turn_local always carries text_tone; a row
                    # without one is the server's transcriber (Deepgram)
                    # hearing that member's own audio — a second copy of a
                    # turn the phone also reported (see calls.py push_turn).
                    merged_problems.append(
                        f"{m.role}: {len(untagged)} of {o.role}'s rows have no text_tone/local clock "
                        f"(server-STT copies of the phone's own turns): {[e.get('text', '')[:40] for e in untagged]}"
                    )
            if by_sender:
                merged_problems.append(f"{m.role} saw turns from unknown senders {list(by_sender)} (own echo?)")
            participant_texts = [e.get("text") for e in got if e.get("participant_uid") in (uids["host"], uids["peer"])]
            scene_order = [scene.turns[i]["text"] for i in range(len(scene.turns)) if scene.turns[i]["text"] in participant_texts]
            in_order = participant_texts == scene_order
            names = sorted({e.get("display_name") for e in got})
            report.data[f"merged_{m.role}"] = {"count": len(got), "expected": sum(len(o.turn_locals) for o in members if o is not m),
                                               "in_order": in_order, "names": names}
            merged_txt.append(f"{m.role} saw {len(got)} turns named {names}{'' if in_order else ' (participant order differs from the scene)'}")
        report.add("merged transcript (per viewer)", not merged_problems,
                   "; ".join(merged_txt) + (f"; PROBLEMS: {merged_problems}" if merged_problems else ""))

        # --- coaching: per participant only; read-only copies for the therapist --
        coach_problems: list[str] = []
        coach_txt: list[str] = []
        for m in (host, dad):
            other_labels = {o.label for o in members if o is not m}
            finals = [s for s in m.run.of_type("suggestion") if not s.get("partial")]
            tagged = [e for e in m.run.events if isinstance(e[1], dict) and "for_uid" in e[1]]
            nudges = [s for s in finals if s.get("kind") == "nudge"]
            responses = [s for s in finals if s.get("kind", "response") == "response"]
            about = {lbl: len([s for s in responses if s.get("speaker") == lbl]) for lbl in sorted(other_labels)}
            peer_label = dad.label if m is host else host.label
            wrong = [s for s in responses if s.get("speaker") not in other_labels] + [s for s in nudges if s.get("speaker") != m.label]
            errors = m.run.of_type("suggestion_error")
            hard = [e for e in errors if e.get("reason") != "llm_parse_error"]
            if about.get(peer_label, 0) < 1:
                coach_problems.append(f"{m.role}: no suggestion about {peer_label}")
            if wrong:
                coach_problems.append(f"{m.role}: {len(wrong)} mislabelled events")
            if hard:
                coach_problems.append(f"{m.role}: {len(hard)} suggestion_error {sorted({str(e.get('reason')) for e in hard})}")
            if tagged:
                coach_problems.append(f"{m.role}: {len(tagged)} events carry for_uid (a participant's wire must not)")
            report.data[f"coaching_{m.role}"] = {"responses": len(responses), "about": about, "nudges": len(nudges), "errors": len(errors), "tagged": len(tagged)}
            coach_txt.append(f"{m.role}: {len(responses)} suggestions about {about}, {len(nudges)} nudges on own turns"
                             + (f", {len(errors)} suggestion_error" if errors else ""))
        own = [e for _, e in mom.run.events if e.get("type") in ("suggestion", "tone_flag", "speaker_identity") and "for_uid" not in e]
        own_sugg = [e for e in own if e.get("type") == "suggestion"]
        copies = [e for _, e in mom.run.events if "for_uid" in e]
        copies_for = {r: [c for c in copies if c.get("for_uid") == uids[r]] for r in ("host", "peer")}
        foreign = [c for c in copies if c.get("for_uid") not in (uids["host"], uids["peer"])]
        copy_kinds = {r: sorted({(c.get("type"), c.get("kind")) for c in cs}) for r, cs in copies_for.items()}
        if own_sugg:
            coach_problems.append(f"therapist was coached: {len(own_sugg)} suggestion(s) without for_uid")
        for r, cs in copies_for.items():
            if not [c for c in cs if c.get("type") == "suggestion" and not c.get("partial")]:
                coach_problems.append(f"therapist got no suggestion copy for {r}")
        if foreign:
            coach_problems.append(f"therapist got {len(foreign)} copies tagged for a non-participant")
        report.data["coaching_therapist"] = {
            "own_suggestions": len(own_sugg), "own_other_events": len(own) - len(own_sugg),
            "copies": {r: len(cs) for r, cs in copies_for.items()}, "copy_kinds": copy_kinds, "foreign": len(foreign),
        }
        coach_txt.append(
            f"therapist: {len(own_sugg)} suggestions of her own (never coached), "
            f"{len(own) - len(own_sugg)} own-audio tone/identity events, read-only copies for host={len(copies_for['host'])} "
            f"peer={len(copies_for['peer'])} kinds={copy_kinds}"
        )
        report.add("per-participant coaching + read-only therapist copies", not coach_problems,
                   "; ".join(coach_txt) + (f" (speed {speed:g}x: latest-wins supersedes some turns)" if speed > 1 else "")
                   + (f"; PROBLEMS: {coach_problems}" if coach_problems else ""))

        # --- relay latency: turn_local -> the other viewers' transcript -------
        timing = transcript_delivery_ms(members, call_id)
        report.data["delivery"] = {k: v for k, v in timing.items() if k != "per_delivery"}
        report.data["delivery"]["slowest"] = sorted((r for r in timing["per_delivery"] if r["ms"] is not None), key=lambda r: -r["ms"])[:3]
        report.add("turn_local -> other-viewer transcript timing", timing["n"] == timing["expected"] and timing["n"] > 0,
                   f"{timing['n']}/{timing['expected']} deliveries timed: p50 {timing['p50_ms']} ms, p95 {timing['p95_ms']} ms, "
                   f"max {timing['max_ms']} ms; slowest {[(r['from'], r['to'], r['ms']) for r in report.data['delivery']['slowest']]}")

        # --- hang-up: host, Dad, then Mom (the last socket ends the call) -----
        end_problems: list[str] = []
        for m in (host, dad):
            ended = m.events("call_ended", call_id)
            done_call = (m.run.session_complete or {}).get("call") or {}
            if ended:
                end_problems.append(f"{m.role} received call_ended (it left while others were still on)")
            if done_call.get("status") != "active" or done_call.get("episode_id") is not None:
                end_problems.append(f"{m.role} session_complete.call={done_call}")
        ended = mom.events("call_ended", call_id)
        last = ended[-1] if ended else {}
        mom_done = (mom.run.session_complete or {}).get("call") or {}
        if len(ended) != 1:
            end_problems.append(f"therapist received {len(ended)} call_ended frames (want 1)")
        elif (last.get("reason"), last.get("ended_by"), last.get("episode_id")) != ("all participants left", uids["therapist"], None):
            end_problems.append(f"therapist call_ended reason/ended_by/episode_id = {last.get('reason')}/{last.get('ended_by')}/{last.get('episode_id')}")
        episodes = last.get("episodes") or {}
        if sorted(episodes) != sorted([uids["host"], uids["peer"]]) or not all(episodes.values()) or len(set(episodes.values())) != 2:
            end_problems.append(f"call_ended.episodes={episodes} (want one per participant, none for the therapist)")
        if last.get("turn_count") != sum(len(m.turn_locals) for m in members):
            end_problems.append(f"call_ended.turn_count={last.get('turn_count')} (want {sum(len(m.turn_locals) for m in members)})")
        if mom_done.get("status") != "ended" or mom_done.get("episode_id") is not None:
            end_problems.append(f"therapist session_complete.call={mom_done}")
        report.data["call_ended"] = {k: last.get(k) for k in ("reason", "ended_by", "episode_id", "episodes", "shared_with", "turn_count")}
        report.add("hang-up order + call_ended on the last socket", not end_problems,
                   f"host and Dad left with the call active (no call_ended, episode later via REST); Mom's socket ended it: "
                   f"reason={last.get('reason')!r} ended_by=therapist episodes={ {k[:8]: (v or '')[:8] for k, v in episodes.items()} } "
                   f"episode_id={last.get('episode_id')} turn_count={last.get('turn_count')}"
                   + (f"; PROBLEMS: {end_problems}" if end_problems else ""))

        # --- 3. GET /calls/{id} as everyone -----------------------------------
        views = {}
        for m in members:
            code, v = await _req(http, "GET", base_url, f"/calls/{call_id}", m.account)
            views[m.role] = (code, v if isinstance(v, dict) else {})
        h_view, d_view, t_view = views["host"][1], views["peer"][1], views["therapist"][1]
        h_ep, d_ep = h_view.get("episode_id"), d_view.get("episode_id")
        rest_problems: list[str] = []
        codes = {r: c for r, (c, _) in views.items()}
        if any(c != 200 for c in codes.values()):
            rest_problems.append(f"status codes {codes}")
        if h_view.get("status") != "ended" or h_view.get("end_reason") != "all participants left":
            rest_problems.append(f"status={h_view.get('status')} end_reason={h_view.get('end_reason')}")
        if not h_ep or not d_ep or h_ep == d_ep or episodes.get(uids["host"]) != h_ep or episodes.get(uids["peer"]) != d_ep:
            rest_problems.append(f"episode ids host={h_ep} dad={d_ep} vs call_ended.episodes")
        if t_view.get("episode_id") is not None or t_view.get("self_role") != "therapist":
            rest_problems.append(f"therapist view episode_id={t_view.get('episode_id')} self_role={t_view.get('self_role')}")
        for r, v in (("host", h_view), ("peer", d_view)):
            if v.get("shared_with") != [therapist.email]:
                rest_problems.append(f"{r} shared_with={v.get('shared_with')} (want [{therapist.email}])")
        if any(p.get("connected") for p in h_view.get("participants") or []):
            rest_problems.append("a participant still reads connected after the end")
        report.data["episodes"] = {"host": h_ep, "peer": d_ep, "shared_with": h_view.get("shared_with"), "turn_count": h_view.get("turn_count")}
        report.add("call ended (REST)", not rest_problems,
                   f"GET /calls/{{id}}: status {h_view.get('status')} ({h_view.get('end_reason')}), {h_view.get('turn_count')} merged turns; "
                   f"host episode {h_ep}, Dad episode {d_ep}, therapist episode {t_view.get('episode_id')}; "
                   f"shared_with host={h_view.get('shared_with')} dad={d_view.get('shared_with')}"
                   + (f"; PROBLEMS: {rest_problems}" if rest_problems else ""))
        if not h_ep or not d_ep:
            return report
        report.data["episode_id"] = h_ep

        # --- 4. the participants' episodes -------------------------------------
        status, detail, waited = await wait_for_analysis(http, base_url, patient, h_ep, timeout_s=analysis_timeout_s)
        live = ((detail or {}).get("analysis") or {}).get("live") or {}
        if status == "full":
            report.add("batch analysis (host)", True, f"full after {waited:.1f}s — {len((detail or {}).get('analysis', {}).get('per_turn') or [])} heats")
        elif status == "failed":
            report.add("batch analysis (host)", False, f"analysis_status failed after {waited:.1f}s: {live.get('analysis_error')}")
        else:
            report.add("batch analysis (host)", False, f"{status} after {waited:.1f}s (analysis_status={live.get('analysis_status')})")

        code, body = await _req(http, "POST", base_url, f"/episodes/{h_ep}/reflect", patient, timeout=180.0)
        refl = (body.get("could_have_said") or []) if code == 200 and isinstance(body, dict) else []
        report.add("reflection (host)", code == 200 and bool(refl),
                   f"POST /episodes/{{id}}/reflect -> {code}: {len(refl)} reflections for {len(a_idx)} own turns (cached={body.get('cached') if isinstance(body, dict) else None})")

        all_texts = sorted(t["text"] for m in members for t in m.turn_locals)
        expect_labels_by = {
            "host": ("Speaker A", {"Speaker A": "You", "Speaker B": "Dad", "Speaker C": "Mom (therapist)"},
                     sorted(scene.turns[i]["text"] for i in scene.expected_self_escalations)),
            "peer": ("Speaker B", {"Speaker B": "You", "Speaker A": "Sage", "Speaker C": "Mom (therapist)"},
                     sorted(t["text"] for t in scene.turns if t["speaker"] == "Speaker B" and t.get("emotion_coarse") == "angry")),
        }
        for m, ep in ((host, h_ep), (dad, d_ep)):
            self_label, want_labels, want_esc = expect_labels_by[m.role]
            code, detail = await _req(http, "GET", base_url, f"/recordings/{ep}", m.account)
            if code != 200 or not isinstance(detail, dict):
                report.add(f"episode detail ({m.role})", False, f"GET /recordings/{{id}} -> {code} {str(detail)[:200]}")
                continue
            turns = detail.get("turns") or []
            labels = {k: (v or {}).get("display_label") for k, v in (detail.get("speaker_labels") or {}).items()}
            live = ((detail.get("analysis") or {}).get("live") or {})
            esc = ((live.get("tone_summary") or {}).get("self") or {}).get("escalation_turns")
            esc_texts = sorted(turns[i].get("text") for i in (esc or []) if i < len(turns))
            got_labels = {k: labels.get(k) for k in want_labels}
            ok_detail = (
                detail.get("mode") == "call" and live.get("mode") == "call"
                and sorted(t.get("text") for t in turns) == all_texts
                and got_labels == want_labels and live.get("self_speaker") == self_label
                and esc_texts == want_esc
                and all(t.get("is_self") is (t.get("speaker") == self_label) for t in turns)
                and all("call_seq" in t and "local_start_time" in t and "participant_uid" in t for t in turns)
                and [t.get("speaker") for t in turns if t.get("speaker") == "Speaker C"] == ["Speaker C"] * len(mom.turn_locals)
            )
            cloud_rows = [t for t in turns if t.get("transcript_source") == "cloud"]
            report.data[f"detail_{m.role}"] = {"mode": detail.get("mode"), "turns": len(turns), "labels": got_labels,
                                               "escalation_turns": esc, "self_speaker": live.get("self_speaker"), "title": detail.get("title"),
                                               "cloud_rows": len(cloud_rows)}
            report.add(f"episode detail ({m.role})", ok_detail,
                       f"GET /recordings/{{id}} 200 — mode {detail.get('mode')}, {len(turns)}/{len(all_texts)} merged turns "
                       f"({len(mom.turn_locals)} from the therapist"
                       + (f", {len(cloud_rows)} server-STT duplicates of phone turns: {[t.get('text', '')[:40] for t in cloud_rows]}" if cloud_rows else "")
                       + f"), labels {got_labels}, self_speaker {live.get('self_speaker')}, "
                       f"self escalation turns {esc} (texts match scene: {esc_texts == want_esc}), title {detail.get('title')!r}, "
                       f"analysis_status={live.get('analysis_status')}")

        code, growth = await _req(http, "GET", base_url, "/growth", patient)
        pt = next((p for p in (growth.get("points") or []) if p.get("recording_id") == h_ep), None) if code == 200 and isinstance(growth, dict) else None
        report.data["growth_point"] = pt
        report.add("growth (host)", pt is not None and pt.get("mode") == "call",
                   f"GET /growth -> {code}: " + (f"point present, my_score={pt.get('my_score')} mode={pt.get('mode')} partners={pt.get('partner_names')}" if pt else "no point for the episode"))

        # --- 5. the therapist's dashboard: both episodes, none of her own ------
        code, sessions = await _req(http, "GET", base_url, "/sessions", therapist)
        rows = (sessions.get("sessions") if isinstance(sessions, dict) else None) or []
        h_row = next((s for s in rows if s.get("id") == h_ep), None)
        d_row = next((s for s in rows if s.get("id") == d_ep), None)
        own_call_rows = [s for s in rows if not s.get("shared") and s.get("mode") == "call"]
        dash_problems: list[str] = []
        if code != 200:
            dash_problems.append(f"GET /sessions -> {code}")
        for name, row, owner in (("host", h_row, patient.email), ("peer", d_row, peer.email)):
            if row is None:
                dash_problems.append(f"{name}'s episode not listed")
            elif not row.get("shared") or row.get("mode") != "call" or row.get("role") != owner:
                dash_problems.append(f"{name}'s row shared={row.get('shared')} mode={row.get('mode')} role={row.get('role')!r}")
        if own_call_rows:
            dash_problems.append(f"{len(own_call_rows)} call episode(s) of her OWN listed: {[r.get('title') for r in own_call_rows]}")
        report.data["therapist_rows"] = {
            "host": {k: h_row.get(k) for k in ("shared", "role", "mode", "title")} if h_row else None,
            "peer": {k: d_row.get(k) for k in ("shared", "role", "mode", "title")} if d_row else None,
            "own_call_rows": len(own_call_rows), "total": len(rows),
        }
        report.add("therapist dashboard (direct grants, no own episode)", not dash_problems,
                   f"GET /sessions as Mom -> {code}: {len(rows)} rows; host's call episode "
                   + (f"shared={h_row.get('shared')} role={h_row.get('role')!r} title={h_row.get('title')!r}" if h_row else "MISSING")
                   + "; Dad's call episode " + (f"shared={d_row.get('shared')} role={d_row.get('role')!r}" if d_row else "MISSING")
                   + f"; own call episodes {len(own_call_rows)}" + (f"; PROBLEMS: {dash_problems}" if dash_problems else ""))
        code, shared_detail = await _req(http, "GET", base_url, f"/recordings/{h_ep}", therapist)
        shared_flag = shared_detail.get("shared") if isinstance(shared_detail, dict) else None
        report.add("therapist detail read", code == 200 and bool(shared_flag),
                   f"GET /recordings/{{host episode}} as Mom -> {code} shared={shared_flag}")

        # --- cleanup ----------------------------------------------------------
        if cleanup:
            notes = []
            for acct, ep in ((patient, h_ep), (peer, d_ep)):
                code, _ = await _req(http, "DELETE", base_url, f"/recordings/{ep}", acct)
                notes.append(f"episode {ep} delete {code}")
            code, sessions = await _req(http, "GET", base_url, "/sessions", therapist)
            left = [s for s in ((sessions.get("sessions") if isinstance(sessions, dict) else None) or []) if s.get("id") in (h_ep, d_ep)]
            notes.append(f"therapist dashboard rows left for the call: {len(left)}")
            for acct in (patient, peer, therapist):
                if acct.signed_up:
                    notes.append(f"firebase delete {acct.email}: {await firebase_delete_account(http, acct)}")
            report.add("cleanup", None, "; ".join(notes))
    finally:
        if own_http:
            await http.aclose()
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True, help="http(s)://host[:port] of the MindShift server")
    p.add_argument("--scene", default="scene_couple_escalation", help=f"one of {list_scenes()}")
    p.add_argument("--speed", type=float, default=1.0, help="stream N× faster than real time (default 1)")
    p.add_argument("--mode", default="earpiece", choices=["earpiece", "speaker", "therapist"])
    p.add_argument("--call", action="store_true",
                   help="in-app call: the patient hosts (POST /calls), the therapist joins by code, both phones "
                        "bind their sockets (call_join), exchange fake SDP over rtc_signal, speak their own scene "
                        "turns concurrently, and each gets a mode=call episode of the merged transcript")
    p.add_argument("--participants", type=int, default=2, choices=[2, 3],
                   help="with --call: 2 (patient + therapist as the two participants) or 3 (patient hosts, a "
                        "second participant account joins as the invitee, the therapist joins her socket as the "
                        "read-only observer; full-mesh signaling, per-participant coaching, two episodes both "
                        "granted to her)")
    p.add_argument("--no-enroll", action="store_true", help="skip POST /voice/enroll-direct")
    p.add_argument("--analysis-timeout", type=float, default=180.0, help="seconds to wait for the batch analysis")
    p.add_argument("--cleanup", action="store_true", help="delete the episode/voiceprint (+ --signup accounts) afterwards")
    p.add_argument("--json", action="store_true", help="also print the raw report dict")
    # the watch
    p.add_argument("--with-watch", action="store_true",
                   help="pair a fake watch to the patient (POST /me/pair/start -> claim -> status), open its "
                        "/ws/live-session socket, and assert the phone's self escalations reach it as nudges")
    p.add_argument("--watch-auth", default="token", choices=["token", "account"],
                   help="how the watch socket authenticates: ?token=<device token> (default) or the shipped Wear "
                        "app's legacy ?account=<uid>")
    # patient auth
    p.add_argument("--id-token", help="patient Firebase ID token")
    p.add_argument("--email")
    p.add_argument("--password")
    # therapist auth
    p.add_argument("--therapist-id-token")
    p.add_argument("--therapist-email")
    p.add_argument("--therapist-password")
    # the second participant (--call --participants 3)
    p.add_argument("--peer-id-token")
    p.add_argument("--peer-email")
    p.add_argument("--peer-password")
    # throwaway accounts
    p.add_argument("--signup", action="store_true", help="create throwaway email/password accounts for both roles")
    p.add_argument("--signup-email-base", default="sagearbor@gmail.com", help="base address for +e2e plus-addressing")
    p.add_argument("--firebase-api-key", default=FIREBASE_WEB_API_KEY)
    return p


async def resolve_accounts(args: argparse.Namespace, http: httpx.AsyncClient) -> tuple[Account, Account, Account | None]:
    """The patient and therapist accounts, plus the second participant when
    ``--call --participants 3`` (else None)."""
    key = args.firebase_api_key
    want_peer = bool(args.call) and args.participants == 3
    if args.signup:
        pw_p, pw_t, pw_d = secrets.token_urlsafe(12), secrets.token_urlsafe(12), secrets.token_urlsafe(12)
        patient = await firebase_password_auth(http, email=args.email or throwaway_email(args.signup_email_base, "patient"), password=args.password or pw_p, signup=True, api_key=key)
        therapist = await firebase_password_auth(http, email=args.therapist_email or throwaway_email(args.signup_email_base, "therapist"), password=args.therapist_password or pw_t, signup=True, api_key=key)
        peer = None
        if want_peer:
            peer = await firebase_password_auth(http, email=args.peer_email or throwaway_email(args.signup_email_base, "peer"), password=args.peer_password or pw_d, signup=True, api_key=key)
        return patient, therapist, peer
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
    peer = None
    if want_peer:
        if args.peer_id_token:
            peer = Account.from_id_token(args.peer_email or "<peer id-token>", args.peer_id_token)
        elif args.peer_email and args.peer_password:
            peer = await firebase_password_auth(http, email=args.peer_email, password=args.peer_password, signup=False, api_key=key)
        else:
            raise SystemExit("--participants 3 needs --signup, --peer-id-token, or --peer-email + --peer-password for the second participant")
    return patient, therapist, peer


async def amain(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene = load_scene(args.scene)
    async with httpx.AsyncClient() as http:
        patient, therapist, peer = await resolve_accounts(args, http)
        if args.call:
            if peer is not None:
                report = await run_call_e2e_three_way(
                    base_url=args.base_url, patient=patient, peer=peer, therapist=therapist, scene=scene,
                    speed=args.speed, analysis_timeout_s=args.analysis_timeout, cleanup=args.cleanup, http=http,
                )
            else:
                report = await run_call_e2e(
                    base_url=args.base_url, patient=patient, therapist=therapist, scene=scene,
                    speed=args.speed, analysis_timeout_s=args.analysis_timeout, cleanup=args.cleanup, http=http,
                )
            print(format_report(report))
            if args.json:
                print(json.dumps(report.to_dict(), indent=2, default=str))
            return 1 if report.failures else 0
        report = await run_e2e(
            base_url=args.base_url, patient=patient, therapist=therapist, scene=scene,
            speed=args.speed, mode=args.mode, enroll=not args.no_enroll,
            analysis_timeout_s=args.analysis_timeout, cleanup=args.cleanup, http=http,
            with_watch=args.with_watch, watch_auth=args.watch_auth,
        )
    print(format_report(report))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    return 1 if report.failures else 0


def main() -> None:
    sys.exit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
