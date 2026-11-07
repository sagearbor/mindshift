"""Load API keys from the main checkout's .env without printing them."""
import os
from pathlib import Path

MAIN = Path(os.environ.get("MINDSHIFT_MAIN", "/Users/sagearbor/projects/githubs/mindshift"))
OUT = MAIN / "tmp" / "judge-trust-20261105"


def load_env() -> None:
    p = MAIN / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
