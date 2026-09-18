"""WOULD THE WRIST SHUT UP? — the whole nudge chain, replayed over real meetings.

The owner cannot be the manual tester (his own rule: verify from files, never
the owner). This is the file-driven answer for the one question device testing
was being used for: during a real conversation, how often does the watch buzz,
and is it buzzing about the right person?

Everything here is a faithful port of the shipped chain, not an approximation:

  SentinelDetector  seeding rule, silence floor, and the "loud windows never
                    feed the baseline" rule (apps/watch/shared/.../SentinelDetector.kt)
  NudgeStateMachine the +6/+10/+14 dB ladder and one-level-per-cooldown decay
                    (apps/watch/shared/.../NudgeStateMachine.kt, mirrored by
                    server/nudge_policy.py and the golden vectors)
  PulseEngine       the proportional pulse train (apps/watch/.../PulseEngine.kt)
  NudgeHapticSchedule  the PRD §6 reminder, including the 2026-09-10 back-off

Two things this measures that nothing else in the repo does:

1. DOSE on real conversation — buzzes per hour. The defect that made the owner
   put the watch down (36 buzzes/min on a poker game) was invisible to ~1,250
   passing tests, because every one of them asked "is this the right cue" and
   none asked "how often".

2. WHOSE turn it landed on. AMI's per-speaker headsets give ground truth for
   who was actually talking, so we can ask the question a single microphone
   cannot answer for itself: how often does the wrist buzz its wearer because
   SOMEONE ELSE in the room got loud? On one mic, loudness has no identity —
   the detector cannot tell. This is the first measurement of what that costs.

    python scripts/ami_corpus.py            # build the corpus first
    python scripts/conversation_audit.py
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# --- mirrored constants -----------------------------------------------------
SILENCE_FLOOR_DBFS = -45.0          # SentinelDetector
BASELINE_WINDOW_COUNT = 30
SEED_WINDOWS = 3
WINDOW_S = 1.0                      # SentinelController MAX_SLICE_BYTES 32000 / 2 / 16000
YELLING_LEVELS = ((14.0, 3), (10.0, 2), (6.0, 1))     # NudgeStateMachine.levelFor
COOLDOWN_S = 20.0                   # NudgePolicy: drop ONE level after this much quiet
PULSE_THRESHOLD_DB = 6.0            # pulseThresholdDbFor(SESSION)
REMINDER_BASE_MS = {1: 120_000, 2: 60_000, 3: 10_000}
REMINDER_CAP_MS = 120_000           # NudgeHapticSchedule.REMINDER_BACKOFF_CAP_MS


def level_for(db_over: float) -> int:
    for bar, lvl in YELLING_LEVELS:
        if db_over >= bar:
            return lvl
    return 0


def reminder_interval_ms(level: int, repeat_index: int) -> int | None:
    """NudgeHapticSchedule.reminderIntervalMs — doubling, capped."""
    base = REMINDER_BASE_MS.get(level)
    if base is None:
        return None
    interval = base
    for _ in range(max(0, min(repeat_index, 16))):
        if interval >= REMINDER_CAP_MS:
            return REMINDER_CAP_MS
        interval *= 2
    return min(interval, REMINDER_CAP_MS)


def windows_dbfs(pcm: np.ndarray, sr: int) -> np.ndarray:
    n = int(sr * WINDOW_S)
    count = len(pcm) // n
    w = pcm[: count * n].reshape(count, n).astype(np.float64)
    rms = np.sqrt(np.mean(w * w, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-12) / 32768.0)


def db_over_baseline(dbfs: np.ndarray) -> np.ndarray:
    """SentinelDetector.observe(), window by window. Unvoiced windows report
    0.0 — the detector's own convention, and why silence can never pulse."""
    voiced_history: list[float] = []
    baseline = None
    out = np.zeros(len(dbfs))
    for i, d in enumerate(dbfs):
        if d <= SILENCE_FLOOR_DBFS:
            continue
        if len(voiced_history) < SEED_WINDOWS:
            if not voiced_history:
                baseline = d
            voiced_history.append(d)
            if len(voiced_history) >= SEED_WINDOWS:
                baseline = float(np.median(voiced_history))
            out[i] = d - (baseline if baseline is not None else d)
            continue
        over = d - baseline
        out[i] = over
        if over < YELLING_LEVELS[-1][0]:
            voiced_history.append(d)
            del voiced_history[:-BASELINE_WINDOW_COUNT]
            baseline = float(np.median(voiced_history))
    return out


def replay(over: np.ndarray, pulse_on: bool, reminder_backoff: bool) -> list[tuple[float, str, int]]:
    """The full chain, one call per 1 s window. Returns every moment the wrist
    would buzz as (seconds, kind, level)."""
    felt: list[tuple[float, str, int]] = []
    level, last_qualifying_t, last_reminder_ms, repeats = 0, None, 0.0, 0
    pulse_last_ms = None
    for i, o in enumerate(over):
        t = i * WINDOW_S
        now_ms = t * 1000.0
        # --- pulse train (independent of the policy) ---
        if pulse_on:
            if o >= PULSE_THRESHOLD_DB:
                if pulse_last_ms is None or now_ms - pulse_last_ms >= 250:
                    felt.append((t, "pulse", 0))
                    pulse_last_ms = now_ms
            else:
                pulse_last_ms = None
        # --- escalation ladder ---
        e = level_for(o)
        if e > level:
            level, last_qualifying_t = e, t
            felt.append((t, "escalation", level))
            last_reminder_ms, repeats = now_ms, 0
        elif e == level and level > 0:
            last_qualifying_t = t
        elif level > 0 and last_qualifying_t is not None and t - last_qualifying_t > COOLDOWN_S:
            level -= 1
            last_qualifying_t = t
            if level > 0:
                last_reminder_ms, repeats = now_ms, 0
        # --- PRD §6 reminder ---
        if level > 0:
            iv = reminder_interval_ms(level, repeats if reminder_backoff else 0)
            if iv is not None and now_ms - last_reminder_ms >= iv:
                felt.append((t, "reminder", level))
                last_reminder_ms = now_ms
                repeats += 1
    return felt


def speaking_at(speakers: dict[str, list[list[float]]], t: float) -> set[str]:
    """Ground truth: who was talking at this instant (AMI headsets)."""
    return {s for s, spans in speakers.items() if any(a <= t <= b for a, b in spans)}


def audit(meta: dict, corpus: Path, pulse_on: bool, reminder_backoff: bool) -> dict:
    with wave.open(str(corpus / meta["mix"])) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    over = db_over_baseline(windows_dbfs(pcm, sr))
    felt = replay(over, pulse_on, reminder_backoff)

    # The coached user is, by convention, headset 0 — the choice is arbitrary
    # and the aggregate is reported across every speaker as the wearer below.
    hours = meta["duration_s"] / 3600.0
    per_wearer = {}
    for wearer in meta["speakers"]:
        on_self = sum(1 for t, _, _ in felt if wearer in speaking_at(meta["speakers"], t))
        # The base rate is what makes the previous number mean anything. In a
        # four-person meeting each person is quiet most of the time, so "most
        # buzzes landed while someone else spoke" is unremarkable ON ITS OWN —
        # chance alone would produce it. The question is whether the detector
        # does BETTER than chance, i.e. whether a buzz carries any information
        # about who was talking. Base rate = the share of the meeting this
        # person was speaking.
        talk_s = meta["speaking_seconds"].get(wearer, 0.0)
        base = talk_s / meta["duration_s"] if meta["duration_s"] else 0.0
        per_wearer[wearer] = {
            "buzzes": len(felt),
            "on_own_turn": on_self,
            "on_someone_else": len(felt) - on_self,
            "base_rate": round(base, 4),
            "expected_on_own_turn": round(base * len(felt), 1),
        }
    return {
        "meeting": meta["meeting"],
        "duration_min": round(meta["duration_s"] / 60, 1),
        "buzzes": len(felt),
        "per_hour": round(len(felt) / hours, 1) if hours else 0.0,
        "by_kind": {k: sum(1 for _, kk, _ in felt if kk == k) for k in ("pulse", "escalation", "reminder")},
        "per_wearer": per_wearer,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO / "tmp/ami-corpus"))
    ap.add_argument("--json-out", default=str(REPO / "tmp/ami-corpus/audit.json"))
    args = ap.parse_args()
    corpus = Path(args.corpus)
    manifest = json.loads((corpus / "manifest.json").read_text())

    scenarios = {
        "SHIPPED TODAY (pulse off, reminder backs off)": dict(pulse_on=False, reminder_backoff=True),
        "before 2026-09-10 (pulse on, flat reminder)":   dict(pulse_on=True, reminder_backoff=False),
    }
    results = {}
    for name, kw in scenarios.items():
        rows = [audit(m, corpus, **kw) for m in manifest["meetings"]]
        results[name] = rows
        total_min = sum(r["duration_min"] for r in rows)
        total_buzz = sum(r["buzzes"] for r in rows)
        print(f"\n{'='*74}\n{name}\n{'='*74}")
        print(f"{'meeting':10} {'min':>6} {'buzzes':>8} {'per hour':>10}   {'pulse/esc/rem':>16}")
        for r in rows:
            k = r["by_kind"]
            print(f"{r['meeting']:10} {r['duration_min']:>6.1f} {r['buzzes']:>8} {r['per_hour']:>10.1f}"
                  f"   {k['pulse']:>5}/{k['escalation']:>4}/{k['reminder']:>4}")
        print(f"{'TOTAL':10} {total_min:>6.1f} {total_buzz:>8} "
              f"{total_buzz/(total_min/60):>10.1f}" if total_min else "")
        # Whose turn did it land on? Averaged over every speaker taking a turn
        # as the wearer, so the answer is not an artefact of picking one person.
        on_self = sum(w["on_own_turn"] for r in rows for w in r["per_wearer"].values())
        total = sum(w["buzzes"] for r in rows for w in r["per_wearer"].values())
        expected = sum(w["expected_on_own_turn"] for r in rows for w in r["per_wearer"].values())
        if total:
            got_pct, exp_pct = on_self / total * 100, expected / total * 100
            print(f"\n  landed while the WEARER was speaking: {got_pct:.1f}%")
            print(f"  chance alone would give:              {exp_pct:.1f}%   (their share of the talking)")
            lift = got_pct - exp_pct
            verdict = ("NO BETTER THAN CHANCE — a buzz carries no information about who spoke"
                       if abs(lift) < 5 else
                       f"{'better' if lift > 0 else 'WORSE'} than chance by {abs(lift):.1f} points")
            print(f"  -> {verdict}")
    Path(args.json_out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
