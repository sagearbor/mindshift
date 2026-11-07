"""Probe which judge models the existing credentials can reach (tiny calls, ~$0)."""
import os
import sys

from google import genai

PROJECT = os.environ.get("JT_GCP_PROJECT", "arborfam-hub")
c = genai.Client(vertexai=True, project=PROJECT, location="global")
names = []
try:
    for m in c.models.list(config={"page_size": 300}):
        if "gemini" in m.name:
            names.append(m.name.split("/")[-1])
except Exception as e:  # noqa: BLE001
    print("list failed", str(e)[:200])
print("LISTED:", " ".join(sorted(names)))
cands = sys.argv[1:] or [
    "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.5-pro", "gemini-3.1-flash",
    "gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-2.5-flash",
]
for mid in cands:
    try:
        r = c.models.generate_content(model=mid, contents="Say OK")
        print("OK", mid, (r.text or "").strip()[:10], r.usage_metadata.total_token_count)
    except Exception as e:  # noqa: BLE001
        print("NO", mid, str(e)[:90])
