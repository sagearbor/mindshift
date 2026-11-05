"""List OpenAI / Anthropic models reachable with the existing keys (no generation, $0)."""
import os
import re

import anthropic
import openai

from env import load_env

load_env()
o = openai.OpenAI()
ids = sorted(m.id for m in o.models.list())
print("OPENAI audio/realtime/newest:", [i for i in ids if re.search(r"audio|realtime|gpt-6|gpt-5\.[5-9]|terra|transcribe", i)])
print("OPENAI gpt-5/6 family:", [i for i in ids if i.startswith(("gpt-5", "gpt-6"))][:40])
a = anthropic.Anthropic()
print("ANTHROPIC:", [m.id for m in a.models.list(limit=50)])
