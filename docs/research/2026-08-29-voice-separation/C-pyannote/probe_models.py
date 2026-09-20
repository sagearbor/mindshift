"""Probe which pyannote pipelines this venv + HF token can actually load."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import torch
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
from pyannote.audio import Pipeline
tok = os.environ.get("HF_TOKEN")
for name in ["pyannote/speaker-diarization-3.1", "pyannote/speaker-diarization-3.0", "pyannote/speaker-diarization-community-1"]:
    t0 = time.time()
    try:
        p = Pipeline.from_pretrained(name, use_auth_token=tok)
        if p is None:
            print(f"{name}: returned None (gated / license not accepted)")
        else:
            print(f"{name}: OK in {time.time()-t0:.1f}s; params={p.parameters(instantiated=True)}")
    except Exception as e:
        print(f"{name}: FAIL {type(e).__name__}: {str(e)[:300]}")
