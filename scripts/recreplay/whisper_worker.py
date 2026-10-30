"""Local faster-whisper decode, run as a SUBPROCESS under tmp/venv-whisper
(faster-whisper/ctranslate2 live in their own venv, apart from the torch +
speechbrain replay venv). Called by :func:`recreplay.stt.transcribe_whisper`;
never imported by it.

    tmp/venv-whisper/bin/python -I scripts/recreplay/whisper_worker.py \
        --wav work/x/audio16k.wav --out work/x/whisper.raw.json --model small [--language en]

Writes the whisper segments + per-word timings as plain JSON (no speakers:
Whisper does no diarization). $0: the model runs on this machine; its
weights come from the Hugging Face hub once and are cached there.
"""

from __future__ import annotations

import argparse
import json
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="small")
    ap.add_argument("--language", default=None)
    ap.add_argument("--vad-threshold", type=float, default=0.3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    from faster_whisper import WhisperModel

    t0 = time.monotonic()
    model = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=a.threads)
    segs, info = model.transcribe(
        a.wav, language=a.language, word_timestamps=True, beam_size=5,
        # Silero gate INSIDE whisper only trims silence (hallucination guard);
        # permissive threshold so quiet talk still reaches the decoder.
        vad_filter=True, vad_parameters={"threshold": a.vad_threshold, "min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )
    out_segs = []
    for s in segs:
        out_segs.append({
            "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
            "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
            "words": [{"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3),
                       "probability": round(w.probability, 4)} for w in (s.words or []) if w.word.strip()],
        })
    doc = {"engine": "faster-whisper", "model": a.model, "language": info.language,
           "language_probability": round(info.language_probability, 4), "duration_s": round(info.duration, 3),
           "decode_s": round(time.monotonic() - t0, 1), "segments": out_segs}
    with open(a.out, "w") as fh:
        json.dump(doc, fh)
    print(f"whisper {a.model}: {len(out_segs)} segments, {doc['decode_s']} s for {doc['duration_s']} s audio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
