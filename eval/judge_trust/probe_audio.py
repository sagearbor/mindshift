"""Check which OpenAI models accept audio input (one 2-3 s CREMA-D clip each, ~$0.01 total)."""
import base64
import sys

import openai

from env import MAIN, load_env

load_env()
o = openai.OpenAI()
wav = MAIN / "tmp/corpora/cremad/AudioWAV/1001_DFA_ANG_XX.wav"
b64 = base64.b64encode(wav.read_bytes()).decode()
prompt = "Which emotion does the speaker's voice convey? Answer with one word."
for model in sys.argv[1:] or ["gpt-audio-1.5", "gpt-6.1-terra", "gpt-audio-mini"]:
    try:
        r = o.chat.completions.create(
            model=model,
            **({"modalities": ["text"]} if "audio" in model else {}),
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
            ]}],
            max_completion_tokens=400,
        )
        print("OK chat", model, repr(r.choices[0].message.content)[:60], r.usage.model_dump() if r.usage else None)
    except Exception as e:  # noqa: BLE001
        print("NO chat", model, str(e)[:160])
    try:
        r = o.responses.create(
            model=model,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
            ]}],
            max_output_tokens=400,
        )
        print("OK responses", model, repr(r.output_text)[:60], r.usage.model_dump() if r.usage else None)
    except Exception as e:  # noqa: BLE001
        print("NO responses", model, str(e)[:160])
