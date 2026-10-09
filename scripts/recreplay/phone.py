"""The PHONE side: the real on-device loop (Silero VAD + StreamingSegmenter
+ ECAPA speaker-ID + the nudge policy) run over the recording by
``apps/mobile/src/live/replay/recordingReplay.ts`` under tsx — the app's own
TypeScript, not a re-implementation.

The loop needs a "script" (the replay meta) for the parts a laptop cannot
run: on-device STT text and the on-device LLM's tone read. We write it from
our reference transcript (Deepgram words) and, as the tone stand-in, the
annotation's vocal emotion (``--phone-tone annotation``, the default when an
annotation exists) or nothing (``neutral``).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .annotation import Alignment, Annotation, Segment
from .audio import _EXTRA_BIN

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_TS = REPO_ROOT / "apps" / "mobile" / "src" / "live" / "replay" / "recordingReplay.ts"
SELF_LABEL = "You"
# Annotation alignment good enough to use its segmentation as the script.
MIN_ALIGN_PCT = 50.0


class PhoneReplayUnavailable(RuntimeError):
    pass


def find_tsx() -> Path | None:
    for d in [REPO_ROOT, *REPO_ROOT.parents]:
        p = d / "node_modules" / ".bin" / "tsx"
        if p.exists():
            return p
    return None


def _words_in(words: list[dict], start: float, end: float) -> list[dict]:
    return [w for w in words if w["start"] >= start - 0.08 and w["end"] <= end + 0.08]


def _majority_coarse(segs: list[Segment], start: float, end: float) -> str | None:
    best, best_ov = None, 0.0
    for s in segs:
        ov = min(end, s.end) - max(start, s.start)
        if ov > best_ov and s.coarse:
            best, best_ov = s.coarse, ov
    return best


def build_meta(
    name: str,
    dg_turns: list[dict],
    words: list[dict],
    *,
    wearer_label: str | None,
    wearer_ann_id: str | None,
    alignment: Alignment | None,
    annotation: Annotation | None,
    phone_tone: str = "annotation",
) -> tuple[dict, str]:
    """(replay meta, which segmentation it used: "annotation" | "deepgram").

    Speaker labels in the meta are the GROUND TRUTH the loop is scored
    against: the owner is ``You``; everyone else keeps their annotation id
    (S2…) or Deepgram label."""
    use_ann = (
        alignment is not None and annotation is not None
        and alignment.quality.get("words_matched_pct", 0.0) >= MIN_ALIGN_PCT
    )
    turns: list[dict] = []
    if use_ann:
        for s in alignment.segments:
            ws = _words_in(words, s.start, s.end)
            text = " ".join(w["word"] for w in ws) or s.text
            turns.append({
                "speaker": SELF_LABEL if s.speaker == wearer_ann_id else s.speaker,
                "text": text, "start_time": round(s.start, 3), "end_time": round(max(s.end, s.start + 0.05), 3),
                **({"emotion_coarse": s.coarse} if phone_tone == "annotation" and s.coarse else {}),
            })
        source = "annotation"
    else:
        segs = alignment.segments if alignment else []
        for t in dg_turns:
            coarse = _majority_coarse(segs, t["start_time"], t["end_time"]) if phone_tone == "annotation" else None
            turns.append({
                "speaker": SELF_LABEL if t["speaker"] == wearer_label else t["speaker"],
                "text": t["text"], "start_time": t["start_time"], "end_time": t["end_time"],
                **({"emotion_coarse": coarse} if coarse else {}),
            })
        source = "deepgram"
    turns.sort(key=lambda t: t["start_time"])
    meta = {
        "_note": ("Written by scripts/recording_replay.py: speaker = ground truth (owner = 'You'); text = Deepgram "
                  f"words; segmentation = {source}; emotion_coarse = "
                  + ("annotation vocal emotion (stand-in for the phone's on-device tone read)" if phone_tone == "annotation" else "none")),
        "scene": name,
        "sample_rate": 16000,
        "self_speaker": SELF_LABEL if any(t["speaker"] == SELF_LABEL for t in turns) else None,
        "turns": turns,
    }
    if meta["self_speaker"] is None:
        meta.pop("self_speaker")
    return meta, source


def run_phone(
    wav: Path, meta_path: Path, out: Path, *, mode: str = "earpiece", enroll: str = "profile",
    profile: Path | None = None, timeout_s: float = 900.0,
) -> dict:
    tsx = find_tsx()
    if tsx is None:
        raise PhoneReplayUnavailable("tsx not found (npm install at the repo root)")
    if enroll == "profile" and (profile is None or not Path(profile).exists()):
        raise PhoneReplayUnavailable(f"--enroll profile needs the owner's profile JSON (missing: {profile})")
    cmd = [str(tsx), str(CLI_TS), "--wav", str(wav), "--meta", str(meta_path), "--out", str(out),
           "--mode", "earpiece" if mode == "room" else mode, "--enroll", enroll]
    if enroll == "profile":
        cmd += ["--profile", str(profile)]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([env.get("PATH", ""), *_EXTRA_BIN])
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
                          timeout=timeout_s, stdin=subprocess.DEVNULL)
    if proc.returncode != 0 or not Path(out).exists():
        raise PhoneReplayUnavailable(f"phone replay failed ({proc.returncode}): {(proc.stderr or proc.stdout)[-800:]}")
    data = json.loads(Path(out).read_text())
    data["_stdout"] = proc.stdout.strip()[-400:]
    return data


def turn_locals_for_server(phone: dict, session_id: str) -> tuple[list[dict], list[float]]:
    """The phone's turn_local stream, ready for the server, + each one's send
    time on the audio clock. The scripted on-device suggestion is STRIPPED:
    it is the replay's placeholder text (replay/fakes.ts), not anything a
    real model said, and the server must not see it as the phone's coaching."""
    events, releases = [], []
    for e in phone.get("sent", []):
        ev = {k: v for k, v in e.items() if k != "sent_at_audio_s"}
        ev["session_id"] = session_id
        ev["suggestion"] = None
        ev["suggestion_source"] = None
        events.append(ev)
        releases.append(float(e.get("sent_at_audio_s", e["end_time"])))
    return events, releases
