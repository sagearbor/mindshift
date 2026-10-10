"""The SERVER side, in real time: ``scripts/live_e2e.py``'s phone client
streams the recording's PCM at real-time pace plus the phone loop's
turn_local stream (each at the audio second the loop emitted it) to a REAL
server, and every event is recorded with its arrival time.

Default target: a LOCAL uvicorn in this process serving ``main.app`` (so a
run never touches prod), with

* the LLM through ``server/llm_cache.py`` — recorded per fixture, replayed
  offline with the recorded latency (``offline=True`` never calls out);
* the server's REAL voiceprint matcher (server/speaker_id.py) against the
  owner's enrolled print (``owner_profile.json``) in an in-memory store;
* no server-side Deepgram: the session is local-first (the phone's
  turn_locals carry the words), so nothing is spent and it is deterministic;
* auth stubbed at ``auth.verify_id_token*`` for one local uid.

``url=...`` targets a deployed server instead (true network + LLM latency),
signing in with ``id_token`` or ``email``/``password`` like live_e2e.
"""

from __future__ import annotations

import asyncio
import os
import socket
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LOCAL_UID = "recording-replay-owner"
LOCAL_EMAIL = "recording-replay@example.test"
LOCAL_TOKEN = "recording-replay-token"


@dataclass
class ServerConfig:
    url: str | None = None                       # None = local in-process uvicorn
    mode: str = "earpiece"
    session_context: str | None = None
    relationship: str | None = None
    library_item_ids: list[str] = field(default_factory=list)
    empathy: int = 50                            # the app's defaults (useAudioStream.ts)
    interject: int = 0
    speed: float = 1.0
    llm_cache_dir: Path | None = None
    offline: bool = False                        # LLM: cache only, a miss is an honest error
    replay_latency: bool = True                  # LLM cache hit sleeps the recorded latency
    profile: dict | None = None                  # the owner's voiceprint document (local server)
    id_token: str | None = None
    email: str | None = None
    password: str | None = None
    stop_timeout_s: float = 90.0
    llm_client: Any = None                       # tests inject a double here
    llm_model: str | None = None                 # None = main.MINDSHIFT_MODEL; a fixture pins its recorded model


@dataclass
class ServerRun:
    target: str
    session_id: str
    events: list[dict]          # each: {"at_s": seconds since first frame, "event": {...}}
    sent: list[dict]            # each: {"at_s": ..., "turn": turn_local}
    audio_seconds: float
    wall_seconds: float
    error: str | None
    session_complete: dict | None
    config_ack: dict | None
    llm: dict
    notes: list[str]

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Local server
# ---------------------------------------------------------------------------

class _NullTranscriber:
    async def connect(self) -> None:
        pass

    async def stream(self, audio_bytes: bytes):
        return []

    async def finish(self):
        return []

    async def close(self) -> None:
        pass


class _NoTTS:
    async def synthesize(self, text: str):
        return None


class _ReplayStore:
    """The slice of the recordings store a live socket reads: the owner's
    voiceprint(s). Anything else answers "nothing stored" and is logged."""

    def __init__(self, voiceprints: list[dict]) -> None:
        self.voiceprints = voiceprints
        self.calls: list[str] = []

    async def list_voiceprints(self, uid: str) -> list[dict]:
        return list(self.voiceprints) if uid == LOCAL_UID else []

    def __getattr__(self, name: str):
        async def _nothing(*a, **k):
            self.calls.append(name)
            return None
        return _nothing


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LocalServer:
    """``main.app`` on 127.0.0.1 in a daemon thread (lifespan off; providers
    injected on app.state like every in-process suite)."""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self.notes: list[str] = []
        self._server = None
        self._thread = None
        self.llm = None

    def __enter__(self) -> "LocalServer":
        self._saved_env = {k: os.environ.get(k) for k in ("MINDSHIFT_ECAPA_CACHE",)}
        os.environ.setdefault("MINDSHIFT_DB_PATH", tempfile.NamedTemporaryFile(suffix=".db", delete=False).name)
        main_ecapa = _main_checkout_ecapa_cache()
        if main_ecapa and not os.getenv("MINDSHIFT_ECAPA_CACHE"):
            # A worktree has no server/.ecapa_cache of its own.
            os.environ["MINDSHIFT_ECAPA_CACHE"] = str(main_ecapa)
        import uvicorn

        import auth
        import main
        from llm_cache import LLMResponseCache

        asyncio.run(main.init_db())

        def _verify(token: str) -> str:
            if token != LOCAL_TOKEN:
                raise ValueError("invalid token")
            return LOCAL_UID

        def _verify_identity(token: str):
            return auth.Identity(uid=_verify(token), email=LOCAL_EMAIL, sign_in_provider="password")

        # Saved and restored on exit: this also runs inside the pytest
        # process, whose other suites own these seams.
        self._saved_auth = (auth.verify_id_token, auth.verify_id_token_identity)
        auth.verify_id_token = _verify
        auth.verify_id_token_identity = _verify_identity
        app = main.app
        self._state_keys = ("llm_client", "transcriber_factory", "tts_client", "recordings_store")
        self._saved_state = {k: getattr(app.state, k) for k in self._state_keys if hasattr(app.state, k)}
        self._app = app
        if self.cfg.llm_client is not None:
            llm = self.cfg.llm_client
            self.notes.append("LLM: injected test double")
        else:
            real = None
            model = self.cfg.llm_model or main.MINDSHIFT_MODEL
            if not self.cfg.offline:
                from llm_client import LLMClient
                real = LLMClient(model=model)
            # MINDSHIFT_REPLAY_LLM_CACHE: one cache dir shared by parallel tuning runs (a prompt
            # any run already paid for is a hit everywhere); else the item's own work/<name>/llm_cache
            cache_dir = (os.getenv("MINDSHIFT_REPLAY_LLM_CACHE") or "").strip() or self.cfg.llm_cache_dir \
                or Path(tempfile.mkdtemp(prefix="recreplay-llm-"))
            llm = LLMResponseCache(real, cache_dir=Path(cache_dir), model=model,
                                   replay_latency=self.cfg.replay_latency)
            self.notes.append(
                f"LLM: {model} via llm_cache ({'offline: cache only' if self.cfg.offline else 'record: real call on a miss'}"
                f"{', recorded latency replayed on hits' if self.cfg.replay_latency else ''})"
            )
        self.llm = llm
        app.state.llm_client = llm
        app.state.transcriber_factory = lambda: _NullTranscriber()
        app.state.tts_client = _NoTTS()
        voiceprints = [self.cfg.profile] if self.cfg.profile else []
        self.store = _ReplayStore(voiceprints)
        app.state.recordings_store = self.store
        self.notes.append("server voiceprint: owner_profile.json" if voiceprints else "server voiceprint: none enrolled")
        if hasattr(main, "_rate_limiter"):
            main._rate_limiter.reset()
        port = _free_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 20.0
        while not self._server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        if not self._server.started:
            raise RuntimeError("local uvicorn did not start")
        self.url = f"http://127.0.0.1:{port}"
        return self

    def __exit__(self, *exc) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        import auth
        auth.verify_id_token, auth.verify_id_token_identity = self._saved_auth
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for k in self._state_keys:
            if k in self._saved_state:
                setattr(self._app.state, k, self._saved_state[k])
            elif hasattr(self._app.state, k):
                delattr(self._app.state, k)


def ensure_ecapa_cache() -> None:
    """Point server/speaker_id.py at an existing checkpoint cache (a worktree
    has none of its own; without this it would download one)."""
    p = _main_checkout_ecapa_cache()
    if p and not os.getenv("MINDSHIFT_ECAPA_CACHE"):
        os.environ["MINDSHIFT_ECAPA_CACHE"] = str(p)


def _main_checkout_ecapa_cache() -> Path | None:
    here = Path(__file__).resolve()
    for d in here.parents:
        p = d / "server" / ".ecapa_cache"
        if p.is_dir() and any(p.glob("*.ckpt")):
            return p
    return None


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def _config_frame(cfg: ServerConfig) -> dict:
    """What the app's useAudioStream sends (self_speaker null, wearer not
    confirmed — identity has to be EARNED by the voiceprints)."""
    frame: dict = {
        "empathy_slider": cfg.empathy, "interject_level": cfg.interject,
        "self_speaker": None, "wearer_known": False,
        "tts": "on-device", "report_latency": True,
    }
    if cfg.relationship:
        frame["relationship"] = cfg.relationship
    if cfg.session_context:
        frame["session_context"] = cfg.session_context[:4000]
    if cfg.library_item_ids:
        frame["library_item_ids"] = cfg.library_item_ids
    if cfg.mode == "room":
        frame["mode"] = "room"
    return frame


async def _account(cfg: ServerConfig):
    import httpx

    import live_e2e
    if cfg.url is None:
        return live_e2e.Account(email=LOCAL_EMAIL, ws_token=LOCAL_TOKEN, headers={}, uid=LOCAL_UID)
    if cfg.id_token:
        return live_e2e.Account.from_id_token(cfg.email or "<id-token>", cfg.id_token)
    if cfg.email and cfg.password:
        async with httpx.AsyncClient() as http:
            return await live_e2e.firebase_password_auth(http, email=cfg.email, password=cfg.password, signup=False)
    raise RuntimeError("--url needs --id-token or --email/--password (a real Firebase account)")


def run_server(pcm16: np.ndarray, turn_locals: list[dict], release_times: list[float], cfg: ServerConfig,
               *, session_id: str | None = None) -> ServerRun:
    import live_e2e

    session_id = session_id or str(uuid.uuid4())
    for ev in turn_locals:
        ev["session_id"] = session_id
    scene = live_e2e.Scene(name="recording", meta={"self_speaker": "Speaker A"}, pcm=pcm16.astype("<i2"),
                           sr=16000, turns=[])
    notes: list[str] = []
    local = None
    llm_stats: dict = {}

    async def go(base_url: str):
        account = await _account(cfg)
        return await live_e2e.stream_live_session(
            base_url, account, scene, session_id=session_id, speed=cfg.speed,
            config=_config_frame(cfg), stop_timeout_s=cfg.stop_timeout_s,
            pcm=scene.pcm, turn_locals=turn_locals, release_times=release_times,
        )

    t0 = time.monotonic()
    if cfg.url is None:
        local = LocalServer(cfg)
        with local:
            notes += local.notes
            run = asyncio.run(go(local.url))
            llm = local.llm
            if hasattr(llm, "stats"):
                llm_stats = {**llm.stats, "model": getattr(llm, "model", None), "log": list(getattr(llm, "log", []))}
            if local.store.calls:
                notes.append(f"store calls answered empty: {sorted(set(local.store.calls))}")
        target = "local uvicorn (main.app, in-process)"
    else:
        run = asyncio.run(go(cfg.url))
        target = cfg.url
        notes.append("deployed server: LLM not cached; latency is the real network + provider latency")
    origin = run.first_frame_at or t0
    events = [{"at_s": round(at - origin, 3), "event": ev} for at, ev in run.events]
    sent = [{"at_s": round(at - origin, 3), "turn": ev} for at, ev in run.sent_turns]
    ack = next((e["event"] for e in events if e["event"].get("type") == "config_ack"), None)
    return ServerRun(
        target=target, session_id=session_id, events=events, sent=sent,
        audio_seconds=run.audio_seconds, wall_seconds=round(time.monotonic() - t0, 2),
        error=run.error, session_complete=run.session_complete, config_ack=ack, llm=llm_stats, notes=notes,
    )

