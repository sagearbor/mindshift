"""The owner's ``<name>.notes.txt``: parsed leniently, problems reported.

Format (docs/recording-annotation-format.md)::

    who: I'm S? / "the man with the low voice"; other voice is my son (12)
    setting: dinner at home, son talking about his school day and a quiz
    phone: on the table, no earpiece
    moments (optional, one per line, mm:ss — what a good coach would have said):
    03:10 I interrupted him — "let him finish"

Optional flag lines: ``relationship:``, ``mode:``, ``library:``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

KEYS = ("who", "setting", "phone", "relationship", "mode", "library")
_KEY_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?P<key>who|setting|phone|relationship|mode|library)\s*(?:[:=]|\s-\s|-)\s*(?P<val>.*)$",
    re.I,
)
_MOMENT_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,3}):(?P<s>\d{2}(?:\.\d+)?)\s*(?:[—–\-:]+\s*)?(?P<text>.*)$"
)
_MOMENTS_HEADER_RE = re.compile(r"^\s*moments\b", re.I)
_SPEAKER_ID_RE = re.compile(r"\bS(\d{1,2})\b")
RELATIONSHIPS = ("child", "partner", "parent", "coworker", "friend", "other")
MODES = ("earpiece", "speaker", "therapist", "room")


@dataclass
class Moment:
    t: float
    text: str
    source: str = "owner"


@dataclass
class Notes:
    who: str | None = None
    setting: str | None = None
    phone: str | None = None
    relationship: str | None = None
    mode: str | None = None
    library_item_ids: list[str] = field(default_factory=list)
    moments: list[Moment] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    raw: str = ""

    # -- derived -----------------------------------------------------------
    @property
    def self_clause(self) -> str | None:
        """The part of ``who:`` describing the owner: "I'm X; other..." -> X."""
        if not self.who:
            return None
        first = re.split(r"[;\n]|\bother (?:voice|voices|person|people)\b", self.who, maxsplit=1, flags=re.I)[0]
        first = re.sub(r"^\s*(?:i\s*['’]?\s*a?m|i am|me\s*=|me:)\s*", "", first, flags=re.I)
        first = first.strip().strip("/").strip().strip('"“”\'').strip()
        # 'S? / "the man..."' -> drop a placeholder id
        first = re.sub(r"^S\?\s*/\s*", "", first).strip().strip('"“”\'').strip()
        return first or None

    @property
    def others_clause(self) -> str | None:
        if not self.who:
            return None
        parts = re.split(r";|\bother (?=voice|voices|person|people)", self.who, maxsplit=1, flags=re.I)
        return parts[1].strip() if len(parts) > 1 else None

    @property
    def self_speaker_id(self) -> str | None:
        """``I'm S2`` -> "S2" (an annotation speaker id named outright)."""
        clause = self.self_clause or ""
        m = _SPEAKER_ID_RE.search(clause)
        return f"S{int(m.group(1))}" if m else None

    @property
    def earpiece(self) -> bool | None:
        """True when the phone line mentions an earpiece/earbud (not "no ...")."""
        if not self.phone:
            return None
        p = self.phone.lower()
        if re.search(r"\bno (?:ear ?piece|ear ?buds?|headphones?|airpods?)", p):
            return False
        if re.search(r"ear ?piece|ear ?buds?|headphones?|airpods?|pixel buds", p):
            return True
        return False


def _ts_seconds(h: str | None, m: str, s: str) -> float:
    return (int(h) * 3600 if h else 0) + int(m) * 60 + float(s)


def parse_notes(text: str) -> Notes:
    n = Notes(raw=text or "")
    for lineno, line in enumerate((text or "").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        km = _KEY_RE.match(line)
        if km:
            key, val = km.group("key").lower(), km.group("val").strip()
            if key == "library":
                n.library_item_ids = [x.strip() for x in re.split(r"[,\s]+", val) if x.strip()]
            elif key == "relationship":
                v = val.lower().strip()
                if v in RELATIONSHIPS:
                    n.relationship = v
                else:
                    n.problems.append(f"notes line {lineno}: relationship {val!r} is not one of {', '.join(RELATIONSHIPS)} (ignored)")
            elif key == "mode":
                v = val.lower().strip()
                if v in MODES:
                    n.mode = v
                else:
                    n.problems.append(f"notes line {lineno}: mode {val!r} is not one of {', '.join(MODES)} (ignored)")
            else:
                setattr(n, key, val or None)
            continue
        if _MOMENTS_HEADER_RE.match(line):
            continue
        mm = _MOMENT_RE.match(line)
        if mm:
            t = _ts_seconds(mm.group("h"), mm.group("m"), mm.group("s"))
            n.moments.append(Moment(t=t, text=mm.group("text").strip()))
            continue
        n.problems.append(f"notes line {lineno}: not understood, ignored: {line.strip()[:80]!r}")
    for key in ("who", "setting", "phone"):
        if getattr(n, key) is None:
            n.problems.append(f"notes: no `{key}:` line")
    return n


# ---------------------------------------------------------------------------
# Voice descriptors -> comparable keywords (who: vs annotation voice_description)
# ---------------------------------------------------------------------------

_SYNONYMS: dict[str, tuple[str, ...]] = {
    "male": ("man", "male", "guy", "dad", "father", "husband", "boy", "son", "he", "his", "him", "gentleman", "masculine"),
    "female": ("woman", "female", "lady", "mom", "mum", "mother", "wife", "girl", "daughter", "she", "her", "feminine"),
    "low": ("low", "deep", "bass", "baritone", "gravelly", "lower"),
    "high": ("high", "higher", "squeaky", "bright", "light", "pitched"),
    "child": ("child", "kid", "boy", "girl", "son", "daughter", "young", "little", "toddler", "teen", "teenager", "tween"),
    "adult": ("man", "woman", "adult", "dad", "mom", "mother", "father", "husband", "wife", "grown", "older", "middle-aged"),
    "fast": ("fast", "rapid", "quick"),
    "slow": ("slow", "measured", "calm"),
    "loud": ("loud", "booming", "shouting"),
    "soft": ("soft", "quiet", "gentle", "whisper"),
}


def descriptor_keywords(text: str | None) -> set[str]:
    """Canonical keywords in a free-text voice description."""
    if not text:
        return set()
    tokens = set(re.findall(r"[a-z\-]+", text.lower()))
    out = {canon for canon, syns in _SYNONYMS.items() if tokens & set(syns)}
    # an explicit age under 18 means child
    for num in re.findall(r"\((\d{1,2})\)|\b(\d{1,2})\s*(?:yo|y/o|years?)", text.lower()):
        age = int(next(x for x in num if x))
        if age < 18:
            out.add("child")
            out.discard("adult")
        else:
            out.add("adult")
    if "child" in out and "adult" in out and tokens & {"son", "daughter", "boy", "girl", "kid", "child"}:
        out.discard("adult")
    return out
