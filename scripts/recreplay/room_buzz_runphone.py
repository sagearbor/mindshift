"""Re-run the phone replay (worktree TS, heat log dumped) over items. $0: no STT/LLM."""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WT = Path("/Users/sagearbor/projects/githubs/mindshift/.claude/worktrees/agent-aadf1d331d9b62aa5")
R = Path("/Users/sagearbor/projects/githubs/mindshift/tmp/recordings")
TSX = "/Users/sagearbor/projects/githubs/mindshift/node_modules/.bin/tsx"
OUT = Path(sys.argv[1])
names = sys.argv[2].split(",")
jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
OUT.mkdir(parents=True, exist_ok=True)


def src(n):
    w = R / "work" / n
    if (w / "audio16k.wav").exists() and (w / "phone_meta.json").exists():
        return w
    w = R / "ot/X/X-Y0" / n
    return w


def run(n):
    out = OUT / f"{n}.json"
    if out.exists():
        return n, 0.0, "cached"
    w = src(n)
    vp = R / "inbox" / n / f"{n}.voiceprint.json"
    cmd = [TSX, str(WT / "apps/mobile/src/live/replay/recordingReplay.ts"), "--wav", str(w / "audio16k.wav"),
           "--meta", str(w / "phone_meta.json"), "--out", str(out), "--mode", "earpiece"]
    cmd += ["--enroll", "profile", "--profile", str(vp)] if vp.exists() else ["--enroll", "same"]
    env = dict(os.environ)
    env["PATH"] = "/opt/homebrew/bin:" + env.get("PATH", "")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(WT), env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=1200)
    return n, time.time() - t0, ("ok" if p.returncode == 0 else "FAIL " + p.stderr[-400:])


with ThreadPoolExecutor(jobs) as ex:
    for n, dt, st in ex.map(run, names):
        print(f"{n:28} {dt:6.1f}s {st}", flush=True)
