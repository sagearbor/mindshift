# Ported from gauge@2157433 server/main.py's `_build_transcriber`/`_build_llm`; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B11):
# * `_build_transcriber` ports verbatim in behavior: `MINDSHIFT_WATCH_STT`
#   (`watch/config.py`'s `Settings.stt`, replacing `GAUGE_STT`) drives the
#   choice — "whisper" -> `WhisperTranscriptionService`, "none" or anything
#   else -> `NullTranscriptionService` (honest degradation, never a guess at
#   an unconfigured provider).
# * `_build_llm` DROPS gauge's `GAUGE_MODEL` / `Settings.model` entirely per
#   the plan's global constraints ("GAUGE_MODEL -> drop — use this repo's
#   existing LLM config via llm_client"): `watch/config.py`'s `Settings`
#   deliberately has NO `model` field (a second, watch-scoped model knob
#   would just be `GAUGE_MODEL` renamed, contradicting "drop"). Instead
#   `build_llm` reads the SAME `MINDSHIFT_MODEL` env var `server/main.py`'s
#   `lifespan()` already builds its own top-level `LLMClient` from — one
#   shared LLM config for the whole app, not a parallel watch-only one.
#   `server/main.py` calls `LLMClient(model=MINDSHIFT_MODEL)` UNGUARDED
#   (a bad/unsupported model crashes startup there); this module instead
#   keeps gauge's own honest-degradation shape for `_build_llm` — missing
#   key or unsupported provider must never crash the watch WS handler or the
#   analyze endpoint, so construction failures are caught and logged, and
#   `None` is returned (matches `analyze_live_session`'s already-honest
#   "llm=None -> skip the summary" contract).
"""Settings-driven backend selection for the watch's post-session analysis
pipeline — used by both `POST /live-sessions/{id}/analyze`
(`watch/routers/live_sessions.py`) and the WS `end` wiring
(`watch/routers/ws.py`) via whatever app-assembly code (Task B12) builds
these once at startup and threads them through."""

from __future__ import annotations

import logging
import os

from llm_client import LLMClient
from watch.config import Settings
from watch.post_session import NullTranscriptionService, TranscriptionService, WhisperTranscriptionService

logger = logging.getLogger(__name__)

# Matches server/main.py's own MINDSHIFT_MODEL default — see module
# docstring's ADAPTED note for why this reuses that same env var/default
# rather than inventing a watch-scoped one. This literal is duplicated (not
# imported) from server/main.py's own MINDSHIFT_MODEL default — keep it in
# sync with server/main.py's MINDSHIFT_MODEL if that default ever changes.
DEFAULT_MODEL = "claude-3-haiku-20240307"


def build_transcriber(settings: Settings) -> TranscriptionService:
    """Resolve the configured STT backend, honestly, from Settings.

    Only "whisper" is wired to a real backend today; anything else
    (including "none") degrades to reporting transcription unavailable
    rather than guessing at an unconfigured provider.
    """
    if settings.stt == "whisper":
        return WhisperTranscriptionService()
    return NullTranscriptionService()


def build_llm(model: str | None = None):
    """Build the configured LLM client, or None if it can't be (missing key,
    unsupported provider, ...) — analyze_live_session treats None as "skip
    the summary honestly" rather than crashing analysis.

    ``model`` defaults to the ``MINDSHIFT_MODEL`` env var (read here, not via
    ``watch.config.Settings`` — see module docstring's ADAPTED note) so
    callers can override it directly in tests without touching the process
    environment.
    """
    resolved_model = model if model is not None else os.environ.get("MINDSHIFT_MODEL", DEFAULT_MODEL)
    try:
        return LLMClient(resolved_model)
    except Exception:  # noqa: BLE001 — missing/invalid key must never crash startup
        logger.warning(
            "LLM client unavailable (missing key or unsupported model %r) — "
            "post-session summaries will be skipped",
            resolved_model,
            exc_info=True,
        )
        return None
