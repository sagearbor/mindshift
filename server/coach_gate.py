"""Coach output policy that runs AROUND the LLM — pure functions, no I/O.

The live coach used to speak ~400 lines/hour even in calm talk (overnight
corpus batch, 2026-10-10): every line was voiced because the app's interject
slider defaults to 0 while the model's importance sits near 72 whatever the
moment; 31% of lines answered a turn of two words or fewer; dozens landed
right after laughter; and 63 said internal diarization labels aloud ("Let
Speaker E finish"). Three independent policies, each behind its own env flag
(read at CALL time so a test or a replay can flip them per run):

1. **Label scrub** — ``MINDSHIFT_LABEL_SCRUB`` (default ON). A raw label
   ("Speaker E", "SPEAKER_01", "S2", "spk 3") never reaches the screen or the
   earpiece: it is rewritten to "them" / "their" / "the other person" (or
   "you" when the label is the confirmed wearer's), and a line that still
   carries one after the rewrite is dropped.

2. **Turn skip** — ``MINDSHIFT_TURN_SKIP`` (default ON; "0" is the kill
   switch). Before any LLM call: a turn of <= ``MINDSHIFT_TURN_SKIP_MAX_WORDS``
   (2) words, a NaturalTurn backchannel ("yeah okay", "mhm right"), laughter
   itself, or a turn that starts within ``MINDSHIFT_LAUGH_SKIP_S`` (4 s) of
   laughter is not coached at all. Laughter is only detectable from the
   transcript (bracketed tags like "[laughter]", "haha"/"hehe") or a
   ``laughter`` flag in the phone's tone context — the phone's on-device STT
   and the replay's Deepgram words usually carry neither, so in practice the
   laughter rule fires rarely; the speak gate below is what keeps the coach
   quiet in happy, loud talk. A strongly negative text tone (frustration or
   defensiveness >= 0.6) exempts a short turn ("Shut up!").

3. **Speak gate** — ``MINDSHIFT_SPEAK_GATE`` (default ON). A line is VOICED
   only when its importance >= ``MINDSHIFT_SPEAK_MIN_IMPORTANCE``, at least
   ``MINDSHIFT_SPEAK_MIN_GAP_S`` session-seconds have passed since the last
   voiced line, and the turn it answers is substantive (>=
   ``MINDSHIFT_SPEAK_MIN_WORDS`` words, not a backchannel). When the wearer is
   UNKNOWN the coach can only give speaker-neutral cues, so importance is
   capped at ``MINDSHIFT_SPEAK_UNKNOWN_CAP`` before the threshold — with the
   cap below the threshold, an unknown-wearer cue is never voiced. Lines that
   fail the gate are STILL SENT with ``speak: false``: the phone shows them
   dimmed in the feed (useAudioStream ``muted``) and never voices them. The
   session's interject slider still applies on top (both must pass).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import natural_turn as _nt


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip() not in ("0", "false", "off", "")


def _num(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# 1. Label scrub
# ---------------------------------------------------------------------------

# "Speaker E", "speaker e", "Speaker AA", "Speaker 2", "Speaker_2",
# "SPEAKER_01", "spk 3", "S2", "Speakers A and B". Bare "S" + digits only as a
# standalone token so "S3 bucket"-style words are rare; this is coach advice.
_LABEL = (
    r"(?:\b(?i:speaker)[ _-](?:[A-Z]{1,2}|[a-z])\b"
    r"|\b(?i:speaker)[ _-]?\d{1,2}\b"
    r"|\b(?i:spk)[ _-]?\d{1,2}\b"
    r"|\bS\d{1,2}\b)"
)
_LABEL_RE = re.compile(_LABEL)
_PAIR_RE = re.compile(rf"{_LABEL}(?:'s)?\s*(?:and|&|or)\s*{_LABEL}(?:'s)?")
_SPEAKERS_PAIR_RE = re.compile(r"\b[Ss]peakers\s+[A-Z]{1,2}\s*(?:and|&|or)\s*[A-Z]{1,2}\b")
_POSS_RE = re.compile(rf"{_LABEL}(?:'s|’s)")
# Words after which a label is the SUBJECT of a clause ("…because Speaker E
# is upset") rather than an object ("Let Speaker E finish").
_SUBJECT_AFTER = frozenset({
    "and", "but", "because", "while", "when", "if", "that", "so", "since",
    "until", "unless", "although", "though", "as", "before", "after", "or",
})


def has_label(text: str) -> bool:
    return bool(_LABEL_RE.search(text or "") or _SPEAKERS_PAIR_RE.search(text or ""))


def _norm_label(raw: str) -> str:
    """"speaker e" / "Speaker_E" -> "Speaker E" (the session's label shape)."""
    m = re.match(r"(?i)speaker[ _-]?([a-z]{1,2}|\d{1,2})$", raw.strip())
    return f"Speaker {m.group(1).upper()}" if m else raw.strip()


def scrub_labels(text: str, self_labels: frozenset[str] | set[str] = frozenset()) -> str | None:
    """Rewrite raw diarization labels out of one coach line.

    Returns the rewritten line, the line unchanged when it has no label, or
    ``None`` when a label survives the rewrite (drop the line). A label in
    ``self_labels`` (the confirmed wearer's) becomes "you"/"your".
    """
    if not text or not has_label(text):
        return text
    out = _SPEAKERS_PAIR_RE.sub("both of them", text)
    out = _PAIR_RE.sub("both of them", out)

    def poss(m: re.Match) -> str:
        label = _norm_label(m.group(0)[:-2])
        return "your" if label in self_labels else "their"

    out = _POSS_RE.sub(poss, out)

    def other(m: re.Match) -> str:
        label = _norm_label(m.group(0))
        before = out[: m.start()].rstrip()
        sentence_start = not before or before[-1] in ".!?:;—–-\"“("
        prev_word = re.findall(r"[A-Za-z']+$", before)
        subject = sentence_start or (prev_word and prev_word[0].lower() in _SUBJECT_AFTER)
        if label in self_labels:
            return "You" if sentence_start else "you"
        if subject:
            return "The other person" if sentence_start else "the other person"
        return "them"

    out = _LABEL_RE.sub(other, out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    if out and out[0].islower() and not text.lstrip()[:1].islower():
        out = out[0].upper() + out[1:]
    if has_label(out):
        return None
    return out


def scrub_lines(lines: list[str], self_labels: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """Scrub every line; drop the ones that cannot be cleaned."""
    if not _flag("MINDSHIFT_LABEL_SCRUB", "1"):
        return list(lines)
    out = []
    for ln in lines:
        s = scrub_labels(ln, self_labels) if isinstance(ln, str) else ln
        if s:
            out.append(s)
    return out


def scrub_line(line: str, self_labels: frozenset[str] | set[str] = frozenset()) -> str:
    """Single-line form: "" when the line had to be dropped."""
    if not _flag("MINDSHIFT_LABEL_SCRUB", "1"):
        return line
    return scrub_labels(line, self_labels) or ""


# ---------------------------------------------------------------------------
# 2. Turn skip (before the LLM)
# ---------------------------------------------------------------------------

_LAUGH_RE = re.compile(
    r"[\[(<*]\s*(?:laugh\w*|chuckl\w*|giggl\w*)[^\])>*]*[\])>*]"
    r"|\b(?:ha){2,}h?\b|\bhah(?:a)+\b|\b(?:he){2,}\b|\blol\b|\blmao\b",
    re.IGNORECASE,
)
_WORDS_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")


def words(text: str) -> list[str]:
    """Unicode-aware word tokens (Greek, accented Latin, ...)."""
    return _WORDS_RE.findall(text or "")


def has_laughter(text: str) -> bool:
    return bool(_LAUGH_RE.search(text or ""))


def _strong_negative(tone_context: dict | None) -> bool:
    tt = (tone_context or {}).get("text_tone") or tone_context or {}
    for k in ("frustration", "defensiveness"):
        v = tt.get(k) if isinstance(tt, dict) else None
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0.6:
            return True
    return False


def _tone_laughter(tone_context: dict | None) -> bool:
    if not isinstance(tone_context, dict):
        return False
    for d in (tone_context, tone_context.get("prosody") or {}, tone_context.get("text_tone") or {}):
        if isinstance(d, dict) and d.get("laughter"):
            return True
    return False


def skip_reason(
    text: str,
    start_time: float,
    end_time: float,
    recent: list[tuple[float, float, str]] = (),
    tone_context: dict | None = None,
) -> str | None:
    """Why this turn should NOT be coached (no LLM call), or ``None``.

    ``recent`` = (start, end, text) of earlier turns, any speaker, for the
    after-laughter rule. Reasons: "fragment", "backchannel", "laughter",
    "after_laughter".
    """
    if not _flag("MINDSHIFT_TURN_SKIP", "1"):
        return None
    if _tone_laughter(tone_context):
        return "laughter"
    stripped = _LAUGH_RE.sub(" ", text or "")
    w = words(stripped)
    if has_laughter(text) and len(w) <= 3:
        return "laughter"
    window = _num("MINDSHIFT_LAUGH_SKIP_S", 4.0)
    if window > 0:
        for s, e, t in recent:
            if has_laughter(t) and e <= start_time + 0.05 and start_time - e <= window:
                return "after_laughter"
        if re.match(r"\s*(?:" + _LAUGH_RE.pattern + r")", text or "", re.IGNORECASE):
            return "after_laughter"
    if _strong_negative(tone_context):
        return None
    if len(w) <= int(_num("MINDSHIFT_TURN_SKIP_MAX_WORDS", 2)):
        return "fragment"
    if _nt.live_turn_kind(stripped, max(0.0, end_time - start_time)) == "backchannel":
        return "backchannel"
    return None


# ---------------------------------------------------------------------------
# 3. Speak gate (after the LLM)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeakGate:
    enabled: bool
    min_importance: int
    min_gap_s: float
    min_words: int
    unknown_cap: int

    @classmethod
    def from_env(cls) -> "SpeakGate":
        return cls(
            enabled=_flag("MINDSHIFT_SPEAK_GATE", "1"),
            min_importance=int(_num("MINDSHIFT_SPEAK_MIN_IMPORTANCE", 78)),
            min_gap_s=_num("MINDSHIFT_SPEAK_MIN_GAP_S", 30.0),
            min_words=int(_num("MINDSHIFT_SPEAK_MIN_WORDS", 3)),
            unknown_cap=int(_num("MINDSHIFT_SPEAK_UNKNOWN_CAP", 72)),
        )

    def substantive(self, text: str, duration_s: float) -> bool:
        w = words(_LAUGH_RE.sub(" ", text or ""))
        if len(w) < self.min_words:
            return False
        return _nt.live_turn_kind(text, duration_s) != "backchannel"

    def effective_importance(self, importance: int, wearer_unknown: bool) -> int:
        if self.enabled and wearer_unknown:
            return min(int(importance), self.unknown_cap)
        return int(importance)

    def passes(
        self,
        importance: int,
        *,
        wearer_unknown: bool,
        turn_text: str,
        turn_duration_s: float,
        now_s: float,
        last_spoken_s: float | None,
    ) -> tuple[bool, str]:
        """(voice it?, reason). ``now_s``/``last_spoken_s`` are session
        seconds (the answered turn's end time)."""
        if not self.enabled:
            return True, "gate_off"
        if self.effective_importance(importance, wearer_unknown) < self.min_importance:
            return False, "unknown_cap" if wearer_unknown and importance >= self.min_importance else "importance"
        if last_spoken_s is not None and now_s - last_spoken_s < self.min_gap_s:
            return False, "gap"
        if not self.substantive(turn_text, turn_duration_s):
            return False, "not_substantive"
        return True, "pass"
