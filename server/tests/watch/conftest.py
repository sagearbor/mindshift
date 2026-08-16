import os

os.environ.setdefault("MINDSHIFT_WATCH_STT", "none")  # WS tests must never invoke real Whisper
