#!/usr/bin/env python3
"""Probe: how does Deepgram diarize each test fixture? (speaker-count check —
the pitch-shifting physics modulation may fragment one voice into several)."""
import asyncio, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "mindshift-dynamics" / "server"))
# key: same runtime .env pattern the other scripts use
import os, re
env = pathlib.Path(__file__).resolve().parent.parent / ".env"
if "DEEPGRAM_API_KEY" not in os.environ and env.exists():
    for line in env.read_text().splitlines():
        m = re.match(r"\s*(?:export\s+)?DEEPGRAM_API_KEY=(.*)", line)
        if m:
            os.environ["DEEPGRAM_API_KEY"] = m.group(1).strip().strip('"').strip("'")
from audio_ingest import transcribe_prerecorded

async def main(path):
    data = pathlib.Path(path).read_bytes()
    turns = await transcribe_prerecorded(data, "audio/wav")
    speakers = {}
    for t in turns:
        speakers.setdefault(t["speaker"], 0)
        speakers[t["speaker"]] += 1
    print(f"{path}: {len(turns)} turns, {len(speakers)} distinct speakers -> {speakers}")

for p in ["tmp/test_recording.wav", "tmp/test_recording_openai.wav", "tmp/test_recording_gptaudio.wav"]:
    asyncio.run(main(p))
