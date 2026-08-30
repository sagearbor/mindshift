#!/usr/bin/env python3
"""Offline grader for Live Coach nudge/suggestion quality (MVP item #2).

Runs the REAL cloud coaching functions (server/audio_pipeline.py
``_generate_nudge`` / ``_generate_suggestions`` — the same prompt bytes the
WebSocket worker sends, minus the transport) over:

  * the three scene fixtures (server/tests/fixtures/audio/test_recording_
    scene_*), pushed through the pipeline turn by turn exactly as
    ``handle_turn_local`` -> ``process_segment`` would see them (turn_local
    events from scripts/live_e2e.py: is_self oracle, measured prosody, the
    fixed text_tone table), and graded against the hand-authored
    ``expected_nudges`` timeline;
  * the owner's real stored live sessions under tmp/nudge-eval/<id>/
    (PRIVATE, never committed): the stored on-device suggestions graded
    as-is, plus a cloud REPLAY of each transcript through the same coach.

Metrics (deterministic first, then a cached LLM judge):

  TIMING      scenes only — hit / late / miss per expected nudge, false
              positives (a non-empty nudge on a calm self turn), and the
              structural check that no nudge ever lands on a non-self turn.
  RELEVANCE   claude-haiku-4-5 judge, fixed rubric, temperature 0, cached on
              disk (tmp/nudge-eval/llm_cache) so re-runs are free: five
              1-5 scores — addressed_to_self, specific, actionable,
              not_preachy, not_repeating.
  REPETITION  word-bigram Jaccard against the previous coaching line in the
              same session; a line with overlap >= 0.5 is a repeat.
  LENGTH      words; "speakable" = <= 12 words (~4 s at conversational pace).
  LATENCY     from the owner's diagnostics records (tmp/nudge-eval/
              diagnostics) + the suggestion_source split of stored turns.

Usage (from the repo root, after ``set -a; source .env; set +a``):

  tmp/venv-voice/bin/python docs/research/2026-08-30-nudge-quality/grade.py \
      --tag before --sources scenes,real,replay,ondevice

Results land in tmp/nudge-eval/results/<tag>.json and a Markdown summary
is printed. ``MINDSHIFT_COACH_CONTEXT=0`` in the environment turns the
conversation-context prompt block OFF in the pipeline (the pre-fix prompt),
so before/after runs are the same code with one switch.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
SERVER = REPO / "server"
SCRIPTS = REPO / "scripts"
EVAL_DIR = REPO / "tmp" / "nudge-eval"
CACHE_DIR = EVAL_DIR / "llm_cache"
RESULTS_DIR = EVAL_DIR / "results"

for p in (str(SERVER), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); never prints values."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(REPO / ".env")

import audio_pipeline  # noqa: E402
import live_e2e  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from models.audio import Utterance  # noqa: E402

COACH_MODEL = os.environ.get("MINDSHIFT_MODEL", "claude-haiku-4-5-20251001")
JUDGE_MODEL = "claude-haiku-4-5-20251001"
# Anthropic list price for Haiku 4.5 ($ per 1M tokens) — for the spend log.
PRICE_IN, PRICE_OUT = 1.00, 5.00

EMPATHY = 50            # the app's default slider (balanced)
INTERJECT_LEVEL = 0     # the app's default: every non-empty line is voiced
STRONG_IMPORTANCE = 60  # a "strong" expected nudge should clear this
SPEAKABLE_WORDS = 12    # ~4 s of speech
REPEAT_JACCARD = 0.5
ROLE_BY_SCENE = {
    "scene_couple_escalation": "Husband",
    "scene_family3": "Parent",
    "scene_meeting4": "Team lead",
}
REAL_ROLE = "Husband"   # the app's default role


# ---------------------------------------------------------------------------
# A disk-cached LLM client with a spend log
# ---------------------------------------------------------------------------

class CachedLLM:
    """LLMClient wrapper: SHA-256(model+system+user+temperature+max_tokens)
    -> tmp/nudge-eval/llm_cache/<key>.json. Real calls record the provider's
    usage so the run can report tokens and dollars actually spent."""

    def __init__(self, model: str, *, refresh: bool = False) -> None:
        self.model = model
        self._client: LLMClient | None = None
        self._refresh = refresh
        self.spent = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_calls": 0}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _real(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient(self.model, os.environ.get("ANTHROPIC_API_KEY"))
        return self._client

    def complete(self, system: str, user: str, temperature: float = 0.7,
                 max_tokens: int = 512, response_schema: dict | None = None) -> str:
        key = hashlib.sha256(
            f"{self.model}\n---\n{system}\n---\n{user}\n---\n{temperature}\n{max_tokens}".encode()
        ).hexdigest()
        path = CACHE_DIR / f"{key}.json"
        if path.exists() and not self._refresh:
            self.spent["cached_calls"] += 1
            return json.loads(path.read_text())["response"]
        client = self._real()
        response = client.complete(system=system, user=user, temperature=temperature,
                                   max_tokens=max_tokens)
        usage = dict(client.last_usage or {})
        self.spent["calls"] += 1
        self.spent["input_tokens"] += int(usage.get("input_tokens", 0)) + int(
            usage.get("cache_read_input_tokens", 0)) + int(usage.get("cache_creation_input_tokens", 0))
        self.spent["output_tokens"] += int(usage.get("output_tokens", 0))
        path.write_text(json.dumps({
            "model": self.model, "system": system, "user": user, "temperature": temperature,
            "max_tokens": max_tokens, "response": response, "usage": usage,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=1))
        return response

    def dollars(self) -> float:
        return (self.spent["input_tokens"] * PRICE_IN + self.spent["output_tokens"] * PRICE_OUT) / 1e6


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------

def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _grams(text: str) -> set[tuple[str, ...]]:
    w = _words(text)
    if len(w) < 2:
        return {(x,) for x in w}
    return {(w[i], w[i + 1]) for i in range(len(w) - 1)}


def repeat_score(prev: str | None, cur: str) -> float:
    """Bigram Jaccard with the previous coaching line (0 when none)."""
    if not prev or not cur:
        return 0.0
    a, b = _grams(prev), _grams(cur)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# The LLM judge (fixed rubric — do not edit between before/after runs)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You grade ONE line from a real-time conversation coach that whispers to one "
    "person (the coached person, marked YOU) during a live conversation. Two "
    "kinds of line exist: a NUDGE about how YOU just came across (e.g. \"ease up\"), "
    "given right after YOUR own turn; and a RESPONSE, words YOU could say next to "
    "the other person, given right after THEIR turn.\n"
    "Score 1-5 (5 = best) on exactly these five dimensions:\n"
    "- addressed_to_self: the line is clearly for YOU (a nudge about YOUR delivery, "
    "or first-person words YOU could say) — not advice aimed at the other person, "
    "not a description of the other person, not a meta-instruction like \"acknowledge "
    "their concern\" that YOU would have to translate before speaking.\n"
    "- specific: grounded in what was JUST said in this exchange, not generic "
    "advice that would fit any conversation.\n"
    "- actionable: one clear move YOU can make immediately, in 12 words or fewer; "
    "a nudge should be 6 words or fewer.\n"
    "- not_preachy: no lecturing, moralizing, therapy jargon or explanation.\n"
    "- not_repeating: differs meaningfully from the previous coaching line "
    "(5 when there is no previous line or it is clearly different; 1 when it "
    "is the same advice reworded).\n"
    "Return ONLY a JSON object: {\"addressed_to_self\": n, \"specific\": n, "
    "\"actionable\": n, \"not_preachy\": n, \"not_repeating\": n, \"note\": "
    "\"<= 15 words\"}."
)
JUDGE_DIMS = ("addressed_to_self", "specific", "actionable", "not_preachy", "not_repeating")


def judge(llm: CachedLLM, *, kind: str, history: list[dict], turn: dict,
          line: str, previous: str | None) -> dict:
    lines = []
    for h in history[-4:]:
        who = "YOU" if h.get("is_self") else (h.get("speaker") or "Other")
        lines.append(f"  {who}: \"{h['text']}\"")
    who = "YOU" if turn.get("is_self") else (turn.get("speaker") or "Other")
    user = (
        "Recent transcript (oldest first):\n" + ("\n".join(lines) if lines else "  (none)") + "\n"
        f"Turn the coach reacted to ({kind.upper()}): {who}: \"{turn['text']}\"\n"
        f"Previous coaching line: {json.dumps(previous) if previous else 'none'}\n"
        f"Coaching line to grade ({kind.upper()}): {json.dumps(line)}"
    )
    raw = llm.complete(system=JUDGE_SYSTEM, user=user, temperature=0.0, max_tokens=200)
    try:
        from main import parse_llm_json
        data = parse_llm_json(raw)
    except Exception:
        data = {}
    out = {}
    for d in JUDGE_DIMS:
        v = data.get(d) if isinstance(data, dict) else None
        out[d] = max(1, min(5, int(v))) if isinstance(v, (int, float)) else None
    out["note"] = (data.get("note") if isinstance(data, dict) else None) or ""
    return out


# ---------------------------------------------------------------------------
# Running the cloud coach over a transcript (the pipeline's own functions)
# ---------------------------------------------------------------------------

def _pipeline_context_kwargs(ctx, utterance: Utterance, is_self: bool | None) -> dict:
    """Whatever extra prompt context the checked-out pipeline offers (the
    context fix adds ``history``); nothing on a pipeline without it."""
    fn = getattr(audio_pipeline, "_history_for_prompt", None)
    if fn is None:
        return {}
    return {"history": fn(ctx, utterance, is_self=is_self)}


def coach_transcript(llm: CachedLLM, turns: list[dict], *, role: str, session_id: str,
                     empathy: int = EMPATHY) -> list[dict]:
    """Push turns (dicts with speaker/text/is_self/text_tone/prosody) through
    ``_generate_nudge`` (self) / ``_generate_suggestions`` (other) exactly as
    ``process_segment`` decides, with the same session bookkeeping helpers.
    Returns one record per turn with the coach's line (or ``""``)."""
    ctx = audio_pipeline.SessionContext(session_id=session_id, empathy_slider=empathy, role=role)
    out: list[dict] = []
    remember_coaching = getattr(audio_pipeline, "_remember_coaching", None)
    for i, t in enumerate(turns):
        text = (t.get("text") or "").strip()
        utt = Utterance(session_id=session_id, speaker=t.get("speaker") or "Unknown",
                        text=text, start_time=float(t.get("start_time") or 0.0),
                        end_time=float(t.get("end_time") or 0.0))
        rec = {"turn": i, "speaker": utt.speaker, "is_self": t.get("is_self"), "text": text,
               "kind": None, "line": "", "importance": None, "spoken": False, "ms": None}
        if not text:
            audio_pipeline._remember_utterance(ctx, utt)
            out.append(rec)
            continue
        is_self = t.get("is_self")
        self_turn = bool(is_self) if is_self is not None else False  # server: self_speaker None
        tone_context = audio_pipeline._tone_context_from_parts(t.get("text_tone"), t.get("prosody"))
        extra = _pipeline_context_kwargs(ctx, utt, is_self)
        audio_pipeline._remember_utterance(ctx, utt)
        t0 = time.monotonic()
        if self_turn:
            nudge, importance = asyncio.run(audio_pipeline._generate_nudge(
                llm, utt, empathy, role, None, tone_context, **extra))
            gate = getattr(audio_pipeline, "_gate_nudge", None)
            if gate is not None and nudge:
                nudge, importance = gate(ctx, utt, nudge, importance, tone_context)
            rec.update(kind="nudge", line=nudge, importance=importance,
                       spoken=bool(nudge) and importance >= INTERJECT_LEVEL)
        else:
            suggestions, importance = asyncio.run(audio_pipeline._generate_suggestions(
                llm, utt, empathy, role, None, tone_context, **extra))
            gate = getattr(audio_pipeline, "_gate_suggestions", None)
            if gate is not None:
                suggestions = gate(ctx, utt, suggestions)
            line = suggestions[0] if suggestions else ""
            rec.update(kind="response", line=line, importance=importance,
                       spoken=bool(line) and importance >= INTERJECT_LEVEL)
        rec["ms"] = round((time.monotonic() - t0) * 1000)
        if rec["line"] and remember_coaching is not None:
            remember_coaching(ctx, utt, rec["line"], rec["kind"])
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# The on-device prompt (apps/mobile/src/live/localLlm.ts buildPrompt), ported
# so the SAME cloud model can stand in for Gemini Nano and grade the PROMPT.
# ---------------------------------------------------------------------------

def _stance(empathy: int) -> str:
    if empathy <= 20:
        return "assertive and direct"
    if empathy <= 50:
        return "balanced"
    if empathy <= 80:
        return "warm and empathetic"
    return "validating and gentle"


ONDEVICE_PROMPTS = {
    # Verbatim port of localLlm.ts before this work (2026-08-30).
    "v1": {
        "system": (
            "You are a discreet real-time conversation coach whispering to one person "
            "during a conversation. Reply with ONLY a JSON object, no prose, no markdown: "
            '{"suggestion": string, "tone": {"warmth": 0-100, "defensiveness": 0-100, '
            '"sarcasm": 0-100, "sadness": 0-100, "frustration": 0-100, "label": string}}. '
            '"tone" scores the turn you were given. Keep "suggestion" under 18 words.'
        ),
        "self_task": (
            "The coached person just said this. Give a single delivery nudge for them "
            "(6 words or fewer, e.g. \"ease up\", \"let them finish\")."
        ),
        "other_task": (
            "Suggest what the coached person should say next to {speaker}, in a {stance} stance."
        ),
    },
    # The prompt after this work — byte-identical to localLlm.ts v2.
    "v2": {
        "system": (
            "You are a discreet real-time conversation coach whispering to one person "
            "(the coached person) during a live conversation. Reply with ONLY a JSON object, "
            'no prose, no markdown: {"suggestion": string, "tone": {"warmth": 0-100, '
            '"defensiveness": 0-100, "sarcasm": 0-100, "sadness": 0-100, "frustration": 0-100, '
            '"label": string}}. "tone" scores the turn you were given. "suggestion" is ONE '
            "line, 10 words or fewer, in the coached person's own voice — never advice "
            "about the other person, never an instruction to be translated first. "
            "Do not repeat or reword a coaching line already given in the transcript."
        ),
        "self_task": (
            "The coached person just said this. Reply with ONE delivery nudge about HOW they "
            "came across (6 words or fewer, imperative, e.g. \"ease up\", \"let them finish\"). "
            "If their delivery was fine — calm, sincere, apologizing, agreeing — reply with an "
            "empty \"suggestion\"; never praise."
        ),
        "other_task": (
            "Reply with ONE sentence the coached person could say next to {speaker}, verbatim, "
            "first person, 10 words or fewer, in a {stance} stance."
        ),
    },
}


def ondevice_prompt(version: str, *, text: str, speaker: str, is_self: bool | None,
                    empathy: int, context: list[dict], prosody_hint: str | None,
                    previous: str | None = None) -> tuple[str, str]:
    """Port of localLlm.ts buildPrompt for the given prompt version. The
    phone's SuggestInput carries no "previous coaching line" field, so
    ``previous`` is unused — kept so a later wiring can be graded here."""
    p = ONDEVICE_PROMPTS[version]
    history = "\n".join(f"{t['speaker']}: {t['text']}" for t in context if t.get("text"))
    who = "the coached person (YOU)" if is_self else speaker
    task = p["self_task"] if is_self else p["other_task"].format(speaker=speaker, stance=_stance(empathy))
    cue = f"\nDelivery cue: {prosody_hint}." if prosody_hint else ""
    user = (f"Earlier:\n{history}\n\n" if history else "") + f"Latest turn from {who}: \"{text}\"{cue}\n\n{task}"
    return p["system"], user


def parse_suggestion_json(raw: str) -> dict | None:
    """Port of localLlm.ts parseSuggestionJson: tolerate fences/prose; the
    suggestion string (or suggestions[0]); None when it isn't one."""
    if not isinstance(raw, str):
        return None
    text = re.sub(r"```(?:json)?", "", raw, flags=re.I).strip()
    first, last = text.find("{"), text.rfind("}")
    if first < 0 or last <= first:
        return None
    try:
        obj = json.loads(text[first:last + 1])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    s = obj.get("suggestion")
    if not isinstance(s, str):
        alt = obj.get("suggestions")
        s = alt[0] if isinstance(alt, list) and alt and isinstance(alt[0], str) else ""
    return {"suggestion": s.strip()}


def _prosody_hint(p: dict | None) -> str | None:
    if not p:
        return None
    parts = []
    rms, rate = p.get("rms_dbfs"), p.get("speech_rate")
    if rms is not None:
        if rms > -15:
            parts.append("loud")
        elif rms < -35:
            parts.append("quiet")
    if rate is not None:
        if rate > 3.5:
            parts.append("fast")
        elif rate < 1.5:
            parts.append("slow")
    return ", ".join(parts) or None


def coach_ondevice(llm: CachedLLM, turns: list[dict], *, version: str, empathy: int = EMPATHY,
                   context_turns: int = 6) -> list[dict]:
    """The phone's prompt path (fastLoop.finalizeTurn -> localLlm.buildPrompt)
    with the cloud model standing in for Gemini Nano. Mirrors fastLoop's
    coachedAsSelf rule: is_self true, or unknown + the "Speaker A speaks
    first" fallback."""
    out: list[dict] = []
    seen: list[dict] = []
    previous: str | None = None
    for i, t in enumerate(turns):
        text = (t.get("text") or "").strip()
        is_self = t.get("is_self")
        coached_as_self = is_self is True or (is_self is None and t.get("speaker") == "Speaker A")
        rec = {"turn": i, "speaker": t.get("speaker"), "is_self": is_self, "text": text,
               "kind": "nudge" if coached_as_self else "response", "line": "", "importance": None,
               "spoken": False, "ms": None}
        if text:
            system, user = ondevice_prompt(
                version, text=text, speaker=t.get("speaker") or "Unknown", is_self=coached_as_self,
                empathy=empathy, context=[{"speaker": s["speaker"], "text": s["text"]} for s in seen[-context_turns:]],
                prosody_hint=_prosody_hint(t.get("prosody")), previous=previous)
            raw = llm.complete(system=system, user=user, temperature=0.7, max_tokens=200)
            parsed = parse_suggestion_json(raw)
            line = parsed["suggestion"] if parsed else ""
            rec.update(line=line, spoken=bool(line))
            if line:
                previous = line
        seen.append({"speaker": t.get("speaker"), "text": text})
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Grading a coached transcript
# ---------------------------------------------------------------------------

def grade_records(llm: CachedLLM, turns: list[dict], records: list[dict],
                  expected: list[dict] | None) -> dict:
    """Attach deterministic + judge metrics to each coached record; compute
    the session summary (timing only when ``expected`` is given)."""
    previous: str | None = None
    for i, rec in enumerate(records):
        line = rec.get("line") or ""
        if not line:
            continue
        rec["words"] = len(_words(line))
        rec["speakable"] = rec["words"] <= SPEAKABLE_WORDS
        rec["repeat_jaccard"] = round(repeat_score(previous, line), 3)
        rec["is_repeat"] = rec["repeat_jaccard"] >= REPEAT_JACCARD
        rec["judge"] = judge(llm, kind=rec["kind"], history=turns[max(0, i - 4):i], turn=turns[i],
                             line=line, previous=previous)
        previous = line

    coached = [r for r in records if r.get("line")]
    by_kind: dict[str, dict] = {}
    for kind in ("nudge", "response"):
        rs = [r for r in coached if r["kind"] == kind]
        if not rs:
            continue
        dims = {}
        for d in JUDGE_DIMS:
            vals = [r["judge"][d] for r in rs if r.get("judge", {}).get(d) is not None]
            dims[d] = round(statistics.mean(vals), 2) if vals else None
        by_kind[kind] = {
            "n": len(rs),
            "mean_words": round(statistics.mean(r["words"] for r in rs), 1),
            "speakable_rate": round(sum(r["speakable"] for r in rs) / len(rs), 2),
            "repeat_rate": round(sum(r["is_repeat"] for r in rs) / len(rs), 2),
            "judge": dims,
            "judge_overall": round(statistics.mean(v for v in dims.values() if v is not None), 2)
            if any(v is not None for v in dims.values()) else None,
        }

    summary: dict = {"turns": len(turns), "coached": len(coached), "by_kind": by_kind}
    if expected is not None:
        self_idx = [i for i, t in enumerate(turns) if t.get("is_self")]
        exp_by_turn = {e["after_turn_index"]: e for e in expected}
        hits, late, miss, fps = [], [], [], []
        for e in expected:
            i = e["after_turn_index"]
            rec = records[i]
            if rec.get("line"):
                strong_ok = e["level"] != "strong" or (rec.get("importance") or 0) >= STRONG_IMPORTANCE
                hits.append({"turn": i, "level": e["level"], "line": rec["line"],
                             "importance": rec.get("importance"), "level_ok": strong_ok})
                continue
            # late: the next self turn within two turns
            nxt = [j for j in self_idx if i < j <= i + 2 and records[j].get("line")]
            if nxt:
                late.append({"turn": i, "fired_at": nxt[0], "line": records[nxt[0]]["line"]})
            else:
                miss.append({"turn": i, "level": e["level"]})
        for i in self_idx:
            if i not in exp_by_turn and records[i].get("line"):
                fps.append({"turn": i, "line": records[i]["line"], "importance": records[i].get("importance"),
                            "emotion": turns[i].get("scripted_emotion")})
        non_self_nudges = [r["turn"] for r in records if r.get("kind") == "nudge" and not turns[r["turn"]].get("is_self")]
        summary["timing"] = {
            "expected": len(expected), "hit": len(hits), "late": len(late), "miss": len(miss),
            "level_ok": sum(h["level_ok"] for h in hits),
            "false_positive": len(fps), "self_turns": len(self_idx),
            "silent_self_turns": sum(1 for i in self_idx if not records[i].get("line")),
            "non_self_nudges": len(non_self_nudges),
            "hits": hits, "late_hits": late, "misses": miss, "false_positives": fps,
        }
    return summary


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def scene_turns(scene) -> list[dict]:
    tls = live_e2e.build_turn_locals(scene, "grade")
    for tl, meta in zip(tls, scene.turns):
        tl["scripted_emotion"] = meta.get("scripted_emotion")
    return tls


def run_scenes(coach: CachedLLM, judge_llm: CachedLLM, only: list[str] | None) -> dict:
    out = {}
    for name in live_e2e.list_scenes():
        if only and name not in only:
            continue
        scene = live_e2e.load_scene(name)
        turns = scene_turns(scene)
        records = coach_transcript(coach, turns, role=ROLE_BY_SCENE.get(name, "Husband"),
                                   session_id=f"grade-{name}")
        summary = grade_records(judge_llm, turns, records, scene.meta["expected_nudges"])
        out[name] = {"summary": summary, "records": records}
        print(f"  scene {name}: {summary['timing']['hit']}/{summary['timing']['expected']} hit, "
              f"{summary['timing']['false_positive']} FP, {summary['coached']} coached", file=sys.stderr)
    return out


def load_real_sessions() -> list[tuple[str, dict, list[dict]]]:
    out = []
    for d in sorted(glob.glob(str(EVAL_DIR / "*-*-*-*-*"))):
        tp = Path(d) / "turns.json"
        mp = Path(d) / "meta.json"
        if not tp.exists():
            continue
        turns = json.loads(tp.read_text())
        meta = json.loads(mp.read_text()) if mp.exists() else {}
        out.append((Path(d).name, meta, turns))
    return out


def stored_kind(turn: dict) -> str:
    """fastLoop's coachedAsSelf: is_self true, or unknown + "Speaker A" fallback."""
    is_self = turn.get("is_self")
    if is_self is True or (is_self is None and turn.get("speaker") == "Speaker A"):
        return "nudge"
    return "response"


def run_real_stored(judge_llm: CachedLLM) -> dict:
    out = {}
    for sid, meta, turns in load_real_sessions():
        records = []
        for i, t in enumerate(turns):
            records.append({"turn": i, "speaker": t.get("speaker"), "is_self": t.get("is_self"),
                            "text": t.get("text") or "", "kind": stored_kind(t) if t.get("suggestion") else None,
                            "line": t.get("suggestion") or "", "importance": None,
                            "source": t.get("suggestion_source"), "spoken": bool(t.get("suggestion"))})
        summary = grade_records(judge_llm, turns, records, None)
        summary["sources"] = {}
        for r in records:
            if r["line"]:
                summary["sources"][r["source"] or "?"] = summary["sources"].get(r["source"] or "?", 0) + 1
        summary["self_known"] = sum(1 for t in turns if t.get("is_self") is not None)
        summary["with_text_tone"] = sum(1 for t in turns if t.get("text_tone"))
        summary["mode"] = (meta.get("title") or "").split("·")[-1].strip()
        summary["created_at"] = meta.get("created_at")
        out[sid] = {"summary": summary, "records": records}
    return out


def run_real_replay(coach: CachedLLM, judge_llm: CachedLLM) -> dict:
    out = {}
    for sid, meta, turns in load_real_sessions():
        records = coach_transcript(coach, turns, role=REAL_ROLE, session_id=f"replay-{sid[:8]}")
        summary = grade_records(judge_llm, turns, records, None)
        summary["created_at"] = meta.get("created_at")
        out[sid] = {"summary": summary, "records": records}
        print(f"  replay {sid[:8]}: {summary['coached']} coached", file=sys.stderr)
    return out


def run_ondevice(coach: CachedLLM, judge_llm: CachedLLM, version: str) -> dict:
    """The on-device PROMPT on the real transcripts + the scenes, with the
    cloud model standing in for Gemini Nano."""
    out = {}
    for sid, meta, turns in load_real_sessions():
        records = coach_ondevice(coach, turns, version=version)
        out[sid] = {"summary": grade_records(judge_llm, turns, records, None), "records": records}
    for name in live_e2e.list_scenes():
        scene = live_e2e.load_scene(name)
        turns = scene_turns(scene)
        records = coach_ondevice(coach, turns, version=version)
        out[name] = {"summary": grade_records(judge_llm, turns, records, scene.meta["expected_nudges"]),
                     "records": records}
    return out


# ---------------------------------------------------------------------------
# Latency (recorded)
# ---------------------------------------------------------------------------

def latency_from_diagnostics() -> list[dict]:
    path = EVAL_DIR / "diagnostics" / "owner-latest.json"
    if not path.exists():
        return []
    raw = path.read_text()
    dec = json.JSONDecoder()
    i, docs = 0, []
    while i < len(raw):
        j = raw.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(raw, j)
            docs.append(obj)
            i = end
        except Exception:
            i = j + 1
    out = []
    for d in docs:
        data = d.get("data", d)
        s = data.get("last_session") or {}
        lat = s.get("latency") or {}
        if not lat.get("turns"):
            continue
        out.append({"diagnostics_id": data.get("diagnostics_id"), "started_at": s.get("startedAt"),
                    "mode": s.get("mode"), "turns": lat.get("turns"), "spoken": lat.get("spoken"),
                    "held": lat.get("held"), "median_llm_ms": lat.get("medianLlmMs"),
                    "median_to_speak_ms": lat.get("medianToSpeakMs"), "p90_to_speak_ms": lat.get("p90ToSpeakMs"),
                    "by_outcome": lat.get("byOutcome"), "by_provider": lat.get("byProvider")})
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def markdown_report(results: dict) -> str:
    lines = [f"## Run `{results['tag']}` — coach `{results['coach_model']}`, judge `{results['judge_model']}`", ""]
    scenes = results.get("scenes") or {}
    if scenes:
        lines += ["### Scenes (cloud coach, real pipeline functions)", "",
                  "| scene | expected | hit | late | miss | strong ok | FP (calm self turn) | non-self nudges | nudge words | nudge repeat | nudge judge | response words | response repeat | response judge |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in scenes.items():
            s, t = r["summary"], r["summary"]["timing"]
            n, rp = s["by_kind"].get("nudge", {}), s["by_kind"].get("response", {})
            lines.append(f"| {name} | {t['expected']} | {t['hit']} | {t['late']} | {t['miss']} | {t['level_ok']} | {t['false_positive']}/{t['self_turns'] - t['expected']} | {t['non_self_nudges']} | "
                         f"{_fmt(n.get('mean_words'))} | {_fmt(n.get('repeat_rate'))} | {_fmt(n.get('judge_overall'))} | "
                         f"{_fmt(rp.get('mean_words'))} | {_fmt(rp.get('repeat_rate'))} | {_fmt(rp.get('judge_overall'))} |")
        lines.append("")
    for key, title in (("real_stored", "Real sessions — stored on-device suggestions, as-is"),
                       ("real_replay", "Real sessions — transcript replayed through the cloud coach"),
                       ("ondevice_v1", "On-device prompt v1 (cloud model standing in for Nano)"),
                       ("ondevice_v2", "On-device prompt v2 (cloud model standing in for Nano)")):
        block = results.get(key) or {}
        if not block:
            continue
        lines += [f"### {title}", "",
                  "| session | turns | coached | kind | mean words | speakable | repeat rate | self | specific | actionable | not preachy | not repeating | overall |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for sid, r in block.items():
            s = r["summary"]
            for kind, k in s["by_kind"].items():
                j = k["judge"]
                lines.append(f"| {sid[:8]} | {s['turns']} | {s['coached']} | {kind} | {k['mean_words']} | {k['speakable_rate']} | {k['repeat_rate']} | "
                             f"{_fmt(j['addressed_to_self'])} | {_fmt(j['specific'])} | {_fmt(j['actionable'])} | {_fmt(j['not_preachy'])} | {_fmt(j['not_repeating'])} | {_fmt(k['judge_overall'])} |")
            if not s["by_kind"]:
                lines.append(f"| {sid[:8]} | {s['turns']} | 0 | - | - | - | - | - | - | - | - | - | - |")
        lines.append("")
    agg = results.get("aggregate") or {}
    if agg:
        lines += ["### Aggregate (all coached lines per source × kind)", "",
                  "| source | kind | n | mean words | speakable | repeat rate | self | specific | actionable | not preachy | not repeating | overall |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for (src, kind), a in agg.items():
            j = a["judge"]
            lines.append(f"| {src} | {kind} | {a['n']} | {a['mean_words']} | {a['speakable_rate']} | {a['repeat_rate']} | "
                         f"{_fmt(j['addressed_to_self'])} | {_fmt(j['specific'])} | {_fmt(j['actionable'])} | {_fmt(j['not_preachy'])} | {_fmt(j['not_repeating'])} | {_fmt(a['judge_overall'])} |")
        lines.append("")
    lat = results.get("latency") or []
    if lat:
        lines += ["### Latency (owner's diagnostics records)", "",
                  "| diagnostics | started | mode | turns | spoken | held | median LLM ms | median to-speak ms | outcomes |", "|---|---|---|---|---|---|---|---|---|"]
        for r in lat:
            lines.append(f"| {r['diagnostics_id']} | {r['started_at']} | {r['mode']} | {r['turns']} | {r['spoken']} | {r['held']} | {_fmt(r['median_llm_ms'])} | {_fmt(r['median_to_speak_ms'])} | {json.dumps(r['by_outcome'])} |")
        lines.append("")
    sp = results["spend"]
    lines.append(f"Spend this run: coach {sp['coach']['calls']} real calls ({sp['coach']['cached_calls']} cached), "
                 f"judge {sp['judge']['calls']} real calls ({sp['judge']['cached_calls']} cached); "
                 f"{sp['input_tokens']} in / {sp['output_tokens']} out tokens ≈ ${sp['dollars']:.3f}.")
    return "\n".join(lines)


def aggregate(results: dict) -> dict:
    agg: dict = {}
    for src in ("scenes", "real_stored", "real_replay", "ondevice_v1", "ondevice_v2"):
        block = results.get(src) or {}
        recs = [r for sess in block.values() for r in sess["records"] if r.get("line")]
        for kind in ("nudge", "response"):
            rs = [r for r in recs if r["kind"] == kind]
            if not rs:
                continue
            dims = {}
            for d in JUDGE_DIMS:
                vals = [r["judge"][d] for r in rs if r.get("judge", {}).get(d) is not None]
                dims[d] = round(statistics.mean(vals), 2) if vals else None
            agg[(src, kind)] = {
                "n": len(rs), "mean_words": round(statistics.mean(r["words"] for r in rs), 1),
                "speakable_rate": round(sum(r["speakable"] for r in rs) / len(rs), 2),
                "repeat_rate": round(sum(r["is_repeat"] for r in rs) / len(rs), 2),
                "judge": dims,
                "judge_overall": round(statistics.mean(v for v in dims.values() if v is not None), 2),
            }
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="results label, e.g. before / after")
    ap.add_argument("--sources", default="scenes,real,replay,ondevice",
                    help="comma list of: scenes, real, replay, ondevice, ondevice_v1, ondevice_v2")
    ap.add_argument("--scene", action="append", help="limit --sources scenes to these scene names")
    ap.add_argument("--refresh", action="store_true", help="ignore the LLM cache (spends money)")
    args = ap.parse_args()
    sources = {s.strip() for s in args.sources.split(",") if s.strip()}
    if "ondevice" in sources:
        sources |= {"ondevice_v1", "ondevice_v2"}

    coach = CachedLLM(COACH_MODEL, refresh=args.refresh)
    judge_llm = CachedLLM(JUDGE_MODEL, refresh=args.refresh)
    results: dict = {"tag": args.tag, "coach_model": COACH_MODEL, "judge_model": JUDGE_MODEL,
                     "coach_context": os.environ.get("MINDSHIFT_COACH_CONTEXT", "1"),
                     "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if "scenes" in sources:
        print("scenes…", file=sys.stderr)
        results["scenes"] = run_scenes(coach, judge_llm, args.scene)
    if "real" in sources:
        print("real (stored)…", file=sys.stderr)
        results["real_stored"] = run_real_stored(judge_llm)
    if "replay" in sources:
        print("real (replay)…", file=sys.stderr)
        results["real_replay"] = run_real_replay(coach, judge_llm)
    for v in ("v1", "v2"):
        if f"ondevice_{v}" in sources:
            print(f"on-device prompt {v}…", file=sys.stderr)
            results[f"ondevice_{v}"] = run_ondevice(coach, judge_llm, v)
    results["latency"] = latency_from_diagnostics()
    results["aggregate"] = aggregate(results)
    results["spend"] = {
        "coach": coach.spent, "judge": judge_llm.spent,
        "input_tokens": coach.spent["input_tokens"] + judge_llm.spent["input_tokens"],
        "output_tokens": coach.spent["output_tokens"] + judge_llm.spent["output_tokens"],
        "dollars": round(coach.dollars() + judge_llm.dollars(), 4),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.tag}.json"
    serial = dict(results)
    serial["aggregate"] = {f"{k[0]}|{k[1]}": v for k, v in results["aggregate"].items()}
    out_path.write_text(json.dumps(serial, indent=1))
    print(markdown_report(results))
    print(f"\n(results: {out_path})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
