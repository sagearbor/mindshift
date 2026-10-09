"""Recording replay: turn a real phone recording (+ the owner's notes + an
optional audio-LLM annotation) into a timeline report of what the live coach
would have done, and a re-runnable regression fixture.

Entry point: ``scripts/recording_replay.py``. Modules:

* :mod:`recreplay.notes`       the owner's ``<name>.notes.txt``
* :mod:`recreplay.annotation`  ``mindshift-annotation/v1`` (lenient) + re-timing
* :mod:`recreplay.audio`       ffmpeg normalisation to 16 kHz mono PCM16
* :mod:`recreplay.stt`         Deepgram reference transcript (cached)
* :mod:`recreplay.identity`    which voice is the owner (who: / annotation / voiceprint)
* :mod:`recreplay.phone`       the phone's REAL on-device loop (tsx -> sceneReplay.ts)
* :mod:`recreplay.server_run`  the server in real time (live_e2e streaming, local uvicorn or --url)
* :mod:`recreplay.score`       moments, identity, latency, violations
* :mod:`recreplay.report`      the HTML report
* :mod:`recreplay.fixture`     freeze inputs + baseline, regression comparison
* :mod:`recreplay.app_pull`    "use my latest app recording" (GCS)
"""
