"""Print the declared HF licence of candidate open-weight labellers (metadata only)."""
import json
import sys
import urllib.request

MODELS = sys.argv[1:] or [
    "Qwen/Qwen2.5-Omni-3B", "Qwen/Qwen2.5-Omni-7B", "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "mlx-community/Qwen2.5-Omni-3B-Thinker-4bit", "emotion2vec/emotion2vec_plus_large",
    "emotion2vec/emotion2vec_plus_base", "mlx-community/Qwen2.5-Omni-7B-4bit",
]
for m in MODELS:
    try:
        d = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{m}", timeout=20))
        cd = d.get("cardData") or {}
        cfg = d.get("config") or {}
        print(m, "|", cd.get("license"), cd.get("license_name"), cd.get("license_link"),
              [t for t in d.get("tags", []) if t.startswith("license")], cfg.get("model_type"))
    except Exception as e:  # noqa: BLE001
        print(m, "ERR", str(e)[:80])
