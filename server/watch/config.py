# Ported from gauge@2157433 server/config.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
"""Runtime configuration.

Deliberately pydantic-settings-free: plain ``os.environ`` reads, evaluated at
``Settings()`` construction time (not at import time) so tests can set env
vars before building the app.
"""

import os


class Settings:
    def __init__(self) -> None:
        self.firestore_project = os.environ.get("MINDSHIFT_FIRESTORE_PROJECT") or None
        self.stt = os.environ.get("MINDSHIFT_WATCH_STT", "whisper")
        self.allow_legacy_account = (
            os.environ.get("MINDSHIFT_ALLOW_LEGACY_ACCOUNT", "true").strip().lower()
            not in {"0", "false", "no", "off"}
        )
