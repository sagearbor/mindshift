"""Live-session analysis — the PURE half of Track 2 ("how you're doing over
time" + "what you could have said").

A live coaching session (earpiece / speaker / therapist mode) ends on the
phone with a list of ``TurnLocalEvent`` reports (words + the phone's own tone
and identity verdicts) and no audio on the server. This module turns that
into the SAME episode record an uploaded recording gets — meta + turns +
analysis.json in the recordings store — so YourDay / Growth / Replay render a
live session exactly like an upload, and adds the two things a live session
carries that an upload never had:

* per-turn TEXT TONE (``text_tone`` from the phone's on-device classifier)
  and per-turn PERSON IDENTITY (``speaker_person_id`` / the server's
  ``SpeakerIdentityEvent`` verdicts), aggregated into a ``tone_summary`` —
  "how do I sound, and how do I sound *with Mom* vs *with Asher*";
* a post-session LLM reflection over the user's OWN turns ("what you could
  have said"), whose prompt + parsing live here so the router stays thin.

House rules enforced throughout:

* PURE — no I/O, no LLM calls, no clock reads. The router
  (``routers/sessions.py``) owns storage, the LLM client and the clock.
* Honest nulls — a turn without ``text_tone`` contributes NOTHING to a tone
  distribution (never "neutral" by default); a session with no ``is_self``
  turn has no self summary (``None``), never a guessed one; audio tone is
  surfaced ONLY when the tone classifier's own ``surface_allowed()`` says so.
* Derived, never fabricated — every label here is a deterministic function
  of the scores the phone sent. The LLM batch analysis (heats, report cards)
  is run by the router AFTER ingest and merged in; until then the stored
  analysis is honestly "lite" (``per_turn: []`` → the client shows no heat
  chart and gray "heat unknown" ribbons, exactly like a degraded upload).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Iterable

import episodes as episodes_mod
import speaker_id  # pure constants/report readers only; no torch at import
import word_metrics as word_metrics_mod

# --- Audio tone (Foundation C) — guarded import. tone_id.py lands on main
# while this track is in flight; until it exists, audio tone is simply never
# surfaced (never faked). After it merges, `surface_allowed()` is the single
# gate: MINDSHIFT_TONE_AUDIO=off|dark|on decides whether a classifier verdict
# may reach a user-facing aggregate. Import failure is the ONLY thing caught
# here — a bug inside tone_id must surface, not hide behind this guard.
try:  # pragma: no cover — exercised for real once Foundation C merges
    import tone_id as _tone_id
except ImportError:  # pragma: no cover
    _tone_id = None


def audio_tone_allowed() -> bool:
    """True when an audio-tone verdict may be SURFACED to the user.

    False when tone_id isn't on this server yet (guarded import) or when its
    ``surface_allowed()`` gate says the classifier is off / dark-launched.
    Read defensively: a tone_id without the gate function is treated as
    "not allowed" (the safe default), never as "allowed"."""
    if _tone_id is None:
        return False
    gate = getattr(_tone_id, "surface_allowed", None)
    if not callable(gate):
        return False
    try:
        return bool(gate())
    except Exception:  # noqa: BLE001 — a gate failure must never sink ingest
        return False


# ---------------------------------------------------------------------------
# Tone vocabulary
# ---------------------------------------------------------------------------

# The five scored dimensions of models.audio.TurnTextTone, in display order.
TONE_DIMS: tuple[str, ...] = (
    "warmth", "defensiveness", "sarcasm", "sadness", "frustration",
)

# Dimensions whose HIGH score means the user's own delivery was escalating.
# Sadness is deliberately NOT here (a sad turn is not an escalation), and
# warmth is the opposite pole. The label each dimension maps to when it is
# the dominant signal is the adjective form the client renders as a chip.
_ESCALATION_DIM_LABELS: dict[str, str] = {
    "defensiveness": "defensive",
    "sarcasm": "sarcastic",
    "frustration": "frustrated",
}

# A dimension at or above this (0–100) counts as the dominant signal for the
# derived label, and — for the escalation dims — flags the turn as an
# escalation. 60 is "clearly present", not "faintly": the same threshold
# glanceSummary.ts's heat bands use for "rough" (>66) rounded to the phone
# classifier's coarser resolution. HUMAN-TUNABLE.
TONE_DOMINANT_THRESHOLD = 60

# Free-text labels (from TurnTextTone.label, an on-device classifier's own
# vocabulary, or a ToneFlagEvent.label) that read as the user escalating.
# Lower-cased, matched after stripping — an unknown label is NOT an
# escalation (never guess from a word we don't know).
ESCALATION_LABELS: frozenset[str] = frozenset({
    "defensive", "sarcastic", "frustrated", "angry", "anger", "hostile",
    "contempt", "contemptuous", "irritated", "annoyed", "critical",
    "aggressive", "dismissive", "frustration", "defensiveness", "sarcasm",
})

# The derived label for a scored turn that is neither escalating nor clearly
# warm/sad: the scores are present and simply low. Distinct from a turn
# with NO scores at all, which gets ``None`` (nothing to say).
NEUTRAL_LABEL = "neutral"

# Label-ladder rungs written into the stored analysis. Kept as plain literals
# (not imports from main) — main imports the router that imports this module;
# the shape matches main.SpeakerLabelOut exactly. With Foundation B's
# multi-person voiceprints, BOTH the user ("You") and a matched partner
# ("Mom") are the "enrolled" rung — a voiceprint match is a voiceprint match;
# "me" is the enrolled entry whose display label is "You" specifically (the
# same rule main._growth_point applies).
LABEL_SOURCE_ENROLLED = "enrolled"
LABEL_SOURCE_GENERIC = "generic"
# The human-assertion rungs (main.LABEL_SOURCE_MANUAL / _MANUAL_PERSON) —
# literals here because main imports this module. See manual_overlay.
LABEL_SOURCE_MANUAL = "manual"
LABEL_SOURCE_MANUAL_PERSON = "manual-person"
ENROLLED_DISPLAY_LABEL = speaker_id.SELF_DISPLAY_NAME  # "You"
SELF_PERSON_ID = speaker_id.SELF_PERSON_ID              # "self"

# Values of the live block's ``analysis_status``: "lite" = derived-only (no
# LLM pass yet / too short for one); "full" = the batch analysis merged in;
# "failed" = the batch pass was attempted and failed (the reason is kept).
ANALYSIS_LITE = "lite"
ANALYSIS_FULL = "full"
ANALYSIS_FAILED = "failed"

# Caps on the reflection strings we persist — a runaway LLM answer is
# trimmed on write, never rejected (the same policy as the report-card caps).
REFLECT_TEXT_MAX = 280
REFLECT_WHY_MAX = 200
REFLECT_TONE_READ_MAX = 60


# ---------------------------------------------------------------------------
# Small defensive readers (stored turns vary by client version)
# ---------------------------------------------------------------------------

def _num(value: object) -> float | None:
    """A real number, or None (bools and non-numerics excluded)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _clean_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    return cleaned or None


def _text_tone(turn: dict) -> dict | None:
    tone = turn.get("text_tone")
    return tone if isinstance(tone, dict) else None


def tone_label(text_tone: dict | None) -> str | None:
    """The single display label for one turn's text tone.

    Precedence: the classifier's own free-text ``label`` (it saw the words;
    we didn't) > the dominant scored dimension at/above
    :data:`TONE_DOMINANT_THRESHOLD` (escalation dims first, then sad, then
    warm) > ``"neutral"`` when scores exist but none dominates > ``None``
    when the turn carries no tone at all (an honest gap, not neutral)."""
    if not text_tone:
        return None
    explicit = _clean_label(text_tone.get("label"))
    if explicit:
        return explicit
    scores = {d: _num(text_tone.get(d)) for d in TONE_DIMS}
    if all(v is None for v in scores.values()):
        return None
    # Escalation dims: pick the strongest one that clears the threshold.
    best_dim, best_val = None, -1.0
    for dim, label in _ESCALATION_DIM_LABELS.items():
        val = scores.get(dim)
        if val is not None and val >= TONE_DOMINANT_THRESHOLD and val > best_val:
            best_dim, best_val = label, val
    if best_dim is not None:
        return best_dim
    sadness = scores.get("sadness")
    if sadness is not None and sadness >= TONE_DOMINANT_THRESHOLD:
        return "sad"
    warmth = scores.get("warmth")
    if warmth is not None and warmth >= TONE_DOMINANT_THRESHOLD:
        return "warm"
    return NEUTRAL_LABEL


def is_escalated(text_tone: dict | None) -> bool:
    """True when a turn's text tone reads as the speaker escalating — by an
    explicit escalation label OR any escalation dimension at/above the
    threshold (the two are OR'd so a "warm" label with frustration 85 is
    still honest to its scores)."""
    if not text_tone:
        return False
    if _clean_label(text_tone.get("label")) in ESCALATION_LABELS:
        return True
    for dim in _ESCALATION_DIM_LABELS:
        val = _num(text_tone.get(dim))
        if val is not None and val >= TONE_DOMINANT_THRESHOLD:
            return True
    return False


def has_tone_scores(text_tone: dict | None) -> bool:
    """A turn "carries tone" when it has a label or at least one score."""
    if not text_tone:
        return False
    if _clean_label(text_tone.get("label")):
        return True
    return any(_num(text_tone.get(d)) is not None for d in TONE_DIMS)


# ---------------------------------------------------------------------------
# Identity: who is "me", who is everyone else
# ---------------------------------------------------------------------------

def self_speaker(turns: list[dict]) -> str | None:
    """The speaker label that is the user, or ``None`` when no turn says so.

    The phone marks each turn ``is_self`` True/False/None; the label with the
    MOST True turns wins (a phone can re-label a speaker mid-session, so one
    stray True on a second label must not create two "me"s). Ties go to the
    first-seen label. No True anywhere → no self, and downstream every
    self-based aggregate is honestly ``None``."""
    counts: dict[str, int] = {}
    order: list[str] = []
    for turn in turns:
        speaker = turn.get("speaker")
        if not isinstance(speaker, str):
            continue
        # is_self True, or the phone matched the reserved "self" voiceprint
        # (Foundation B's person id for the account owner) — both are the
        # phone saying "this is me".
        if turn.get("is_self") is not True and turn.get("speaker_person_id") != SELF_PERSON_ID:
            continue
        if speaker not in counts:
            order.append(speaker)
        counts[speaker] = counts.get(speaker, 0) + 1
    if not counts:
        return None
    return max(order, key=lambda sp: counts[sp])


def person_map(
    turns: list[dict],
    speaker_identities: Iterable[dict] | None,
    self_label: str | None,
    known_people: Iterable[dict] | None = None,
) -> dict[str, dict]:
    """Per NON-self speaker label → ``{person_id, display_name}``.

    The server's ``SpeakerIdentityEvent`` verdicts win (they carry a
    display_name); a per-turn ``speaker_person_id`` from the phone fills in a
    person_id for a label the server never ruled on, and its display name is
    looked up in ``known_people`` — the account's enrolled voiceprint
    documents (Foundation B: ``{person_id, display_name, is_self}``), which
    share the SAME person ids the phone matches against. A speaker nobody
    identified still gets an entry (``person_id``/``display_name`` None) so
    the "with ___" rows can list them under their raw label rather than
    silently dropping the conversation partner."""
    names: dict[str, str] = {}
    for doc in known_people or ():
        if not isinstance(doc, dict):
            continue
        pid, name = doc.get("person_id"), doc.get("display_name")
        if isinstance(pid, str) and pid and isinstance(name, str) and name.strip():
            names[pid] = name.strip()
    people: dict[str, dict] = {}
    for turn in turns:
        speaker = turn.get("speaker")
        if not isinstance(speaker, str) or speaker == self_label:
            continue
        entry = people.setdefault(
            speaker, {"person_id": None, "display_name": None},
        )
        pid = turn.get("speaker_person_id")
        if entry["person_id"] is None and isinstance(pid, str) and pid.strip():
            entry["person_id"] = pid.strip()
            if pid.strip() in names:
                entry["display_name"] = names[pid.strip()]
    for ident in speaker_identities or ():
        if not isinstance(ident, dict):
            continue
        speaker = ident.get("speaker")
        if not isinstance(speaker, str) or speaker == self_label:
            continue
        if ident.get("is_self") is True:
            continue  # the server says this label is me — never a "person"
        entry = people.setdefault(
            speaker, {"person_id": None, "display_name": None},
        )
        pid = ident.get("person_id")
        if isinstance(pid, str) and pid.strip():
            entry["person_id"] = pid.strip()
        name = ident.get("display_name")
        if isinstance(name, str) and name.strip():
            entry["display_name"] = name.strip()
    return people


def build_speaker_labels(
    turns: list[dict],
    self_label: str | None,
    people: dict[str, dict],
) -> dict[str, dict]:
    """The label-ladder map for a live session, in first-appearance order.

    ``enrolled`` for the self speaker ("You") AND for a speaker the identity
    path put a display name on (a voiceprint match to a named person — the
    same rung Foundation B's matcher writes for an enrolled partner, so
    /growth groups by it); ``generic`` (raw label) for everyone else. Same
    two-field shape as main.SpeakerLabelOut."""
    labels: dict[str, dict] = {}
    for speaker in dict.fromkeys(
        t.get("speaker") for t in turns if isinstance(t.get("speaker"), str)
    ):
        if speaker == self_label:
            labels[speaker] = {
                "display_label": ENROLLED_DISPLAY_LABEL,
                "label_source": LABEL_SOURCE_ENROLLED,
            }
            continue
        name = (people.get(speaker) or {}).get("display_name")
        if isinstance(name, str) and name.strip():
            labels[speaker] = {
                "display_label": name.strip(), "label_source": LABEL_SOURCE_ENROLLED,
            }
        else:
            labels[speaker] = {
                "display_label": speaker, "label_source": LABEL_SOURCE_GENERIC,
            }
    return labels


def identity_report(self_label: str | None, people: dict[str, dict]) -> dict | None:
    """A ``speaker_identity`` report in Foundation B's MULTI shape so every
    existing ladder reader (``speaker_id.enrolled_display_labels`` — used by
    main's resolver, episodes' participants and the voice router) labels a
    live session's speakers without a special case: ``matched`` (speaker →
    person_id) + ``people`` (person_id → display_name/is_self), plus the
    legacy ``matched_speaker`` for pre-multi readers. ``source: "live"``
    records that the verdicts came from the phone, not a server match. None
    when nobody (not even the user) was identified."""
    matched: dict[str, str] = {}
    people_meta: dict[str, dict] = {}
    if self_label is not None:
        matched[self_label] = SELF_PERSON_ID
        people_meta[SELF_PERSON_ID] = {
            "display_name": ENROLLED_DISPLAY_LABEL, "is_self": True,
        }
    for speaker, info in people.items():
        pid, name = info.get("person_id"), info.get("display_name")
        if not (isinstance(pid, str) and pid) or pid == SELF_PERSON_ID:
            continue
        matched[speaker] = pid
        people_meta[pid] = {"display_name": name, "is_self": False}
    if not matched:
        return None
    return {
        "matched_speaker": self_label,
        "matched": matched,
        "people": people_meta,
        "speakers": {},
        "source": "live",
    }


def overlay_identity_labels(
    base: dict | None, identity_labels: dict[str, dict],
) -> dict[str, dict]:
    """Merge live identity labels over the batch analysis's LLM ladder.

    The LLM's "name" rung is a guess from the words; the identity path's
    name comes from a matched voiceprint (the enrolled rung) — it wins, as
    does "You". A generic identity label never overwrites an LLM-found name
    (the LLM at least read the transcript)."""
    merged = {
        sp: dict(entry) for sp, entry in (base or {}).items()
        if isinstance(entry, dict)
    }
    for sp, entry in identity_labels.items():
        if entry.get("label_source") == LABEL_SOURCE_ENROLLED:
            merged[sp] = dict(entry)
        elif sp not in merged:
            merged[sp] = dict(entry)
    return merged


# ---------------------------------------------------------------------------
# Per-turn tone rows + the tone summary
# ---------------------------------------------------------------------------

def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start is not None and b_start is not None and a_end is not None \
        and b_end is not None and a_start < b_end and b_start < a_end


def _audio_flag_for_turn(turn: dict, flags: list[dict]) -> dict | None:
    """The audio-sourced ToneFlagEvent overlapping this turn's time span (the
    highest-confidence one when several do), or None."""
    t_start, t_end = _num(turn.get("start_time")), _num(turn.get("end_time"))
    best, best_conf = None, -1.0
    for flag in flags:
        if not isinstance(flag, dict) or flag.get("source") != "audio":
            continue
        f_start, f_end = _num(flag.get("start_time")), _num(flag.get("end_time"))
        if not _overlaps(t_start, t_end, f_start, f_end):
            continue
        conf = _num(flag.get("confidence")) or 0.0
        if conf > best_conf:
            best, best_conf = flag, conf
    return best


def turn_tone_rows(
    turns: list[dict],
    self_label: str | None,
    people: dict[str, dict],
    tone_flags: list[dict] | None = None,
    *,
    audio_allowed: bool = False,
) -> list[dict]:
    """One derived row per turn, index-aligned with ``turns``::

        {index, speaker, is_self, person_id, display_name, with_speaker,
         label, escalated, scored, audio_label, audio_escalated}

    ``with_speaker`` (self turns only) is the OTHER party this turn was said
    TO: the most recent non-self speaker before it, else the next one after
    (a self turn that opens the session was still said to whoever answered).
    That attribution is what makes "how I sound with Mom" possible in a
    multi-party session. ``audio_*`` are filled ONLY when ``audio_allowed``
    — otherwise they stay None even if flags were sent (the classifier's own
    surfacing gate, never overridden here)."""
    flags = [f for f in (tone_flags or []) if isinstance(f, dict)]
    rows: list[dict] = []
    last_other: str | None = None
    for i, turn in enumerate(turns):
        speaker = turn.get("speaker") if isinstance(turn.get("speaker"), str) else None
        is_self = self_label is not None and speaker == self_label
        tone = _text_tone(turn)
        row = {
            "index": i,
            "speaker": speaker,
            "is_self": is_self,
            "person_id": None if is_self else (people.get(speaker or "") or {}).get("person_id"),
            "display_name": None if is_self else (people.get(speaker or "") or {}).get("display_name"),
            "with_speaker": last_other if is_self else None,
            "label": tone_label(tone),
            "escalated": is_escalated(tone) if is_self or tone else False,
            "scored": has_tone_scores(tone),
            "audio_label": None,
            "audio_escalated": None,
        }
        if audio_allowed:
            flag = _audio_flag_for_turn(turn, flags)
            if flag is not None:
                label = _clean_label(flag.get("label"))
                row["audio_label"] = label
                row["audio_escalated"] = label in ESCALATION_LABELS if label else None
        rows.append(row)
        if speaker is not None and not is_self:
            last_other = speaker
    # Forward-fill: a self turn with no prior other party is attributed to
    # the NEXT other party (still honest — that is who it was said to).
    next_other: str | None = None
    for row in reversed(rows):
        if row["speaker"] is not None and not row["is_self"]:
            next_other = row["speaker"]
        elif row["is_self"] and row["with_speaker"] is None:
            row["with_speaker"] = next_other
    return rows


def _distribution(rows: list[dict], key: str = "label") -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = row.get(key)
        if isinstance(label, str):
            counts[label] = counts.get(label, 0) + 1
    return counts


def _mean_scores(turns: list[dict], indexes: Iterable[int]) -> dict[str, int | None]:
    """Per-dimension mean over the turns at ``indexes`` that carry that
    dimension; None for a dimension no turn scored (never 0)."""
    sums: dict[str, float] = {d: 0.0 for d in TONE_DIMS}
    counts: dict[str, int] = {d: 0 for d in TONE_DIMS}
    for i in indexes:
        tone = _text_tone(turns[i])
        if not tone:
            continue
        for d in TONE_DIMS:
            val = _num(tone.get(d))
            if val is not None:
                sums[d] += val
                counts[d] += 1
    return {
        d: (int(round(sums[d] / counts[d])) if counts[d] else None)
        for d in TONE_DIMS
    }


def _bucket(turns: list[dict], rows: list[dict]) -> dict:
    """Aggregate one set of SELF rows into the shared bucket shape."""
    scored = [r for r in rows if r["scored"]]
    return {
        "turns": len(rows),
        "scored_turns": len(scored),
        "labels": _distribution(scored),
        "mean": _mean_scores(turns, (r["index"] for r in rows)),
        "escalation_turns": [r["index"] for r in rows if r["escalated"]],
        "escalation_count": sum(1 for r in rows if r["escalated"]),
    }


def _audio_bucket(rows: list[dict]) -> dict | None:
    heard = [r for r in rows if r["audio_label"] is not None]
    if not heard:
        return None
    return {
        "turns": len(heard),
        "labels": _distribution(heard, "audio_label"),
        "escalation_turns": [r["index"] for r in heard if r["audio_escalated"]],
        "escalation_count": sum(1 for r in heard if r["audio_escalated"]),
    }


def tone_summary(
    turns: list[dict],
    rows: list[dict],
    self_label: str | None,
    people: dict[str, dict],
    *,
    audio_allowed: bool = False,
) -> dict:
    """The stored ``tone_summary`` block::

        {self_speaker, self: bucket|None, audio: bucket|None,
         audio_tone_surfaced, people: [person-bucket…]}

    ``self`` is None when no turn is the user's (nothing honest to
    summarize). Each ``people`` entry is the user's OWN tone in the turns
    said TO that person — "how I sound with Mom" — plus how many turns the
    person themselves spoke. Sorted by most-talked-with first."""
    self_rows = [r for r in rows if r["is_self"]]
    people_out: list[dict] = []
    for speaker, info in people.items():
        with_rows = [r for r in self_rows if r["with_speaker"] == speaker]
        their_turns = sum(1 for r in rows if r["speaker"] == speaker)
        bucket = _bucket(turns, with_rows)
        people_out.append({
            "speaker": speaker,
            "person_id": info.get("person_id"),
            "display_name": info.get("display_name"),
            "their_turns": their_turns,
            "self_turns": bucket["turns"],
            "scored_turns": bucket["scored_turns"],
            "labels": bucket["labels"],
            "mean": bucket["mean"],
            "escalation_turns": bucket["escalation_turns"],
            "escalation_count": bucket["escalation_count"],
        })
    people_out.sort(key=lambda p: (-p["self_turns"], -p["their_turns"]))
    return {
        "self_speaker": self_label,
        "self": _bucket(turns, self_rows) if self_label is not None else None,
        "audio": _audio_bucket(self_rows) if audio_allowed else None,
        "audio_tone_surfaced": bool(audio_allowed),
        "people": people_out,
    }


def annotate_episodes(episodes: list[dict] | None, rows: list[dict]) -> list[dict] | None:
    """ADD per-episode self-tone keys to episodes.py's output (never remove
    or rename one): ``self_tone_labels`` (label → count over the user's
    scored turns inside the episode) and ``self_escalation_count``. Empty /
    zero when the episode has no self turns — the client hides the chip."""
    if episodes is None:
        return None
    out: list[dict] = []
    for ep in episodes:
        first, last = ep.get("first_turn_index"), ep.get("last_turn_index")
        inside = [
            r for r in rows
            if r["is_self"] and isinstance(first, int) and isinstance(last, int)
            and first <= r["index"] <= last
        ]
        out.append({
            **ep,
            "self_tone_labels": _distribution([r for r in inside if r["scored"]]),
            "self_escalation_count": sum(1 for r in inside if r["escalated"]),
        })
    return out


# ---------------------------------------------------------------------------
# Storage shapes
# ---------------------------------------------------------------------------

def storage_turns(turn_events: list[dict]) -> list[dict]:
    """turns.json rows for a live session: the four fields every consumer of
    a stored transcript reads (speaker/text/start/end — HeatChart, episodes,
    word metrics) PLUS the live-only per-turn facts preserved verbatim so a
    re-aggregation never needs the phone again."""
    out: list[dict] = []
    for ev in turn_events:
        out.append({
            "speaker": ev.get("speaker"),
            "text": ev.get("text"),
            "start_time": ev.get("start_time"),
            "end_time": ev.get("end_time"),
            "is_self": ev.get("is_self"),
            "speaker_person_id": ev.get("speaker_person_id"),
            "speaker_match_score": ev.get("speaker_match_score"),
            "transcript_source": ev.get("transcript_source"),
            "text_tone": ev.get("text_tone"),
            "prosody": ev.get("prosody"),
            "suggestion": ev.get("suggestion"),
            "suggestion_source": ev.get("suggestion_source"),
        })
    return out


def turns_hash(turns: list[dict]) -> str:
    """Fingerprint of the transcript (speaker + text per turn). The
    reflection cache key: a re-POST of the same session with the same words
    reuses the stored reflection; changed words invalidate it."""
    payload = json.dumps(
        [[t.get("speaker"), t.get("text")] for t in turns], ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def duration_seconds(turns: list[dict], started_at: str | None, ended_at: str | None) -> float | None:
    """The session's length: the transcript's furthest end time, else the
    wall-clock span of started_at→ended_at, else None (never 0)."""
    ends = [e for e in (_num(t.get("end_time")) for t in turns) if e is not None]
    if ends:
        return max(ends)
    try:
        if started_at and ended_at:
            a = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            b = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
            span = (b - a).total_seconds()
            return span if span > 0 else None
    except ValueError:
        return None
    return None


def _empty_dynamics() -> dict:
    return {
        "coupling": {"strength": None, "leader": None, "description": ""},
        "deescalation": {"who_first": None, "follow_rate": None, "description": ""},
        "triggers": [],
        "requests": [],
    }


def lite_analysis(
    *,
    session_id: str,
    mode: str,
    started_at: str | None,
    ended_at: str | None,
    turns: list[dict],
    tone_flags: list[dict] | None,
    speaker_identities: list[dict] | None,
    title: str | None,
    gap_seconds: float,
    audio_allowed: bool | None = None,
    known_people: list[dict] | None = None,
) -> dict:
    """The analysis.json written at INGEST — everything derivable without an
    LLM, in the same top-level shape as an upload's ``AnalyzeResponse`` dump
    (so every existing reader keeps working) plus a ``live`` block::

        live: {session_id, mode, started_at, ended_at, turn_count,
               self_speaker, tone_summary, turn_tone: [rows…],
               could_have_said: None, reflection: None,
               analysis_status: "lite", analysis_error: None,
               turns_hash}

    ``per_turn`` is EMPTY (not zero-filled): a heat we did not measure is a
    gap. The router's post-ingest batch pass fills it via
    :func:`merge_full_analysis`."""
    if audio_allowed is None:
        audio_allowed = audio_tone_allowed()
    self_label = self_speaker(turns)
    people = person_map(turns, speaker_identities, self_label, known_people)
    rows = turn_tone_rows(turns, self_label, people, tone_flags, audio_allowed=audio_allowed)
    labels = build_speaker_labels(turns, self_label, people)
    identity = identity_report(self_label, people)
    eps = episodes_mod.segment_episodes(
        turns, per_turn=None, speaker_labels=labels, speaker_identity=identity,
        title=title, gap_seconds=gap_seconds,
    )
    return {
        "per_turn": [],
        "per_speaker": {},
        "dynamics": _empty_dynamics(),
        "narrative": "",
        "report_cards": {},
        "speaker_labels": labels,
        "speaker_identity": identity,
        "title": title,
        "word_metrics": word_metrics_mod.compute_word_metrics(
            [{"speaker": t.get("speaker"), "text": t.get("text")} for t in turns]
        ),
        "episodes": annotate_episodes(eps, rows),
        "live": {
            "session_id": session_id,
            "mode": mode,
            "started_at": started_at,
            "ended_at": ended_at,
            "turn_count": len(turns),
            "self_speaker": self_label,
            "tone_summary": tone_summary(
                turns, rows, self_label, people, audio_allowed=audio_allowed,
            ),
            "turn_tone": rows,
            # Raw flags kept verbatim so a later, permitted surfacing can
            # re-aggregate without the phone; never read for display here.
            "tone_flags": [f for f in (tone_flags or []) if isinstance(f, dict)],
            "could_have_said": None,
            "reflection": None,
            "analysis_status": ANALYSIS_LITE,
            "analysis_error": None,
            "turns_hash": turns_hash(turns),
        },
    }


def merge_full_analysis(
    lite: dict, full: dict, turns: list[dict], *, gap_seconds: float,
) -> dict:
    """Fold the batch LLM analysis (an ``AnalyzeResponse`` dump) into a lite
    live analysis: heats/report cards/narrative come from ``full``; the live
    block, the identity-derived labels ("You", matched names) and the
    session title are kept from ``lite``; episodes are re-segmented WITH
    the now-known heats and re-annotated with tone."""
    live = dict(lite.get("live") or {})
    labels = overlay_identity_labels(
        full.get("speaker_labels"), lite.get("speaker_labels") or {},
    )
    identity = lite.get("speaker_identity")
    eps = episodes_mod.segment_episodes(
        turns, per_turn=full.get("per_turn"), speaker_labels=labels,
        speaker_identity=identity, title=lite.get("title"), gap_seconds=gap_seconds,
    )
    live["analysis_status"] = ANALYSIS_FULL
    live["analysis_error"] = None
    return {
        **full,
        "speaker_labels": labels,
        "speaker_identity": identity,
        "title": lite.get("title"),
        "episodes": annotate_episodes(eps, live.get("turn_tone") or []),
        "live": live,
    }


# ---------------------------------------------------------------------------
# "What you could have said" — prompt + parsing (the LLM call is the router's)
# ---------------------------------------------------------------------------

REFLECT_SYSTEM_PROMPT = (
    "You are a warm, precise communication coach reviewing a conversation "
    "AFTER it happened. The transcript is numbered; the turns marked (YOU) "
    "were spoken by the person you are coaching — everyone else is someone "
    "they were talking with. For EACH listed (YOU) turn index, offer what they "
    "could have said instead: a single sentence in their own voice that keeps "
    "their real point but lands more warmly, less defensively, and more "
    "constructively. Never invent facts they didn't say. If a turn was already "
    "good, say so briefly in `why` and keep `could_have_said` close to the "
    "original.\n\n"
    "Return ONLY JSON of the form:\n"
    '{"reflections": [{"turn_index": <int>, "could_have_said": "<one sentence>", '
    '"why": "<one short sentence: what this changes for the listener>", '
    '"tone_read": "<1-3 words naming how the original turn came across>"}]}\n'
    "Include exactly one object per requested index, in order."
)

# Same corrective retry suffix policy as /analyze: one terse nudge when the
# first answer isn't parseable JSON.
REFLECT_RETRY_SUFFIX = (
    "Your previous reply was not valid JSON. Reply with ONLY the JSON object "
    "described above — no prose, no markdown fences."
)


def build_reflect_prompt(
    turns: list[dict],
    self_label: str,
    speaker_labels: dict | None,
    *,
    mode: str | None = None,
    context: str | None = None,
) -> tuple[str, list[int]]:
    """The user prompt for a reflection pass + the self turn indexes it asks
    about. Every turn is numbered so the model's ``turn_index`` aligns with
    the stored transcript; self turns are tagged ``(YOU)``; other speakers
    are shown under their display label (a matched name reads better than
    "Speaker B")."""
    lines: list[str] = []
    self_indexes: list[int] = []
    for i, turn in enumerate(turns):
        speaker = turn.get("speaker")
        text = turn.get("text") if isinstance(turn.get("text"), str) else ""
        if speaker == self_label:
            self_indexes.append(i)
            tag = "YOU"
        else:
            entry = (speaker_labels or {}).get(speaker) if isinstance(speaker, str) else None
            label = entry.get("display_label") if isinstance(entry, dict) else None
            tag = label if isinstance(label, str) and label.strip() else str(speaker)
        lines.append(f"{i}. ({tag}) {text}")
    user = (
        f"Conversation ({len(turns)} turns):\n" + "\n".join(lines)
        + "\n\nReflect on (YOU) turn indexes: "
        + ", ".join(str(i) for i in self_indexes)
    )
    if mode:
        user += f"\n\nSession mode: {mode}"
    if context:
        user += f"\n\nContext: {context}"
    return user, self_indexes


def reflect_max_tokens(n_self_turns: int) -> int:
    """Output budget: ~120 tokens per reflection object + headroom, capped
    like /analyze so a long session can't request an absurd generation."""
    return min(8192, 300 + 120 * max(1, n_self_turns))


def parse_reflections(data: object, self_indexes: list[int]) -> list[dict]:
    """Clean the LLM's ``reflections`` into the stored shape, keeping ONLY
    objects whose ``turn_index`` is one of the requested self turns (an index
    the model invented is dropped, never re-mapped) and whose
    ``could_have_said`` is a non-empty string. Strings are trimmed to the
    REFLECT_* caps. Ordered by turn index; one per index (first wins).
    Raises ``ValueError`` when the payload has no usable list at all so the
    caller can retry once, then report honestly."""
    if not isinstance(data, dict) or not isinstance(data.get("reflections"), list):
        raise ValueError("LLM returned no reflections list")
    wanted = set(self_indexes)
    seen: set[int] = set()
    out: list[dict] = []
    for item in data["reflections"]:
        if not isinstance(item, dict):
            continue
        idx = item.get("turn_index")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx not in wanted or idx in seen:
            continue
        said = item.get("could_have_said")
        if not isinstance(said, str) or not said.strip():
            continue
        why = item.get("why")
        tone_read = item.get("tone_read")
        seen.add(idx)
        out.append({
            "turn_index": idx,
            "could_have_said": said.strip()[:REFLECT_TEXT_MAX],
            "why": (why.strip()[:REFLECT_WHY_MAX] if isinstance(why, str) and why.strip() else ""),
            "tone_read": (
                tone_read.strip()[:REFLECT_TONE_READ_MAX]
                if isinstance(tone_read, str) and tone_read.strip() else ""
            ),
        })
    out.sort(key=lambda r: r["turn_index"])
    return out


def cached_reflection(analysis: dict | None, turns: list[dict]) -> list[dict] | None:
    """The stored ``could_have_said`` list when it is still valid for these
    turns (same transcript hash), else None. A stale cache (words changed by
    a re-POST) is never served."""
    if not isinstance(analysis, dict):
        return None
    live = analysis.get("live")
    if not isinstance(live, dict):
        return None
    cached = live.get("could_have_said")
    reflection = live.get("reflection")
    if not isinstance(cached, list) or not isinstance(reflection, dict):
        return None
    if reflection.get("turns_hash") != turns_hash(turns):
        return None
    return cached


# ---------------------------------------------------------------------------
# Cross-recording aggregates (GET /growth) + the therapist dashboard shape
# ---------------------------------------------------------------------------

def manual_overlay(rec: dict) -> dict[str, dict]:
    """The recording's MANUAL speaker labels as ladder entries —
    ``{speaker: {display_label, label_source, person_id?}}`` — read from
    meta.json's ``manual_speaker_labels`` (names) + ``manual_speaker_people``
    (people labeling: which enrolled person a named speaker is). The same
    overlay main._effective_speaker_labels applies at read time, duplicated
    here (main imports this module) so the growth aggregates and the
    therapist rows honor a human's correction exactly as the detail endpoint
    does. Only speakers present in the turns count."""
    names = rec.get("manual_speaker_labels") or {}
    people = rec.get("manual_speaker_people") or {}
    if not isinstance(names, dict):
        return {}
    speakers = {
        t.get("speaker") for t in (rec.get("turns") or [])
        if isinstance(t, dict) and isinstance(t.get("speaker"), str)
    }
    out: dict[str, dict] = {}
    for speaker, name in names.items():
        if speaker not in speakers or not (isinstance(name, str) and name.strip()):
            continue
        pid = people.get(speaker) if isinstance(people, dict) else None
        if isinstance(pid, str) and pid.strip():
            out[speaker] = {
                "display_label": name.strip(),
                "label_source": LABEL_SOURCE_MANUAL_PERSON,
                "person_id": pid.strip(),
            }
        else:
            out[speaker] = {
                "display_label": name.strip(), "label_source": LABEL_SOURCE_MANUAL,
            }
    return out


def _person_row_with_manual(p: dict, overlay: dict[str, dict]) -> dict:
    """A tone-summary person bucket with the recording's manual label for
    that speaker applied: a manual-person label supplies the cross-session
    ``person_id`` (and its name); a plain manual name supplies the name."""
    entry = overlay.get(p.get("speaker") or "")
    if not entry:
        return p
    out = dict(p)
    out["display_name"] = entry["display_label"]
    if entry.get("person_id"):
        out["person_id"] = entry["person_id"]
    return out


def growth_extras(rec: dict) -> dict:
    """Additive GrowthPoint fields for one stored recording: ``source``
    ("live"/"upload"/"link"), ``mode``, and ``self_tone`` — the session's
    self bucket (labels/mean/escalations) plus per-person buckets — when the
    recording is a live session with a tone summary. Uploads (no per-turn
    tone) get ``self_tone: None``: the chart's "How you sound" section then
    counts them as unscored days, never as neutral ones."""
    source = (rec.get("source") or {}).get("type") if isinstance(rec.get("source"), dict) else None
    analysis = rec.get("analysis")
    live = analysis.get("live") if isinstance(analysis, dict) else None
    summary = live.get("tone_summary") if isinstance(live, dict) else None
    self_bucket = summary.get("self") if isinstance(summary, dict) else None
    people = summary.get("people") if isinstance(summary, dict) else None
    # People labeling: a manual "that's Mom" on this recording names (and
    # identifies, via person_id) the per-person bucket without a re-ingest.
    overlay = manual_overlay(rec)
    if overlay and isinstance(people, list):
        people = [
            _person_row_with_manual(p, overlay) if isinstance(p, dict) else p
            for p in people
        ]
    return {
        "source": source,
        "mode": rec.get("mode") or (live.get("mode") if isinstance(live, dict) else None),
        "self_tone": (
            {
                "scored_turns": self_bucket.get("scored_turns", 0),
                "labels": dict(self_bucket.get("labels") or {}),
                "mean": dict(self_bucket.get("mean") or {}),
                "escalation_count": self_bucket.get("escalation_count", 0),
                "people": [
                    {
                        "person_id": p.get("person_id"),
                        "display_name": p.get("display_name") or p.get("speaker"),
                        "scored_turns": p.get("scored_turns", 0),
                        "labels": dict(p.get("labels") or {}),
                        "escalation_count": p.get("escalation_count", 0),
                    }
                    for p in (people or []) if isinstance(p, dict)
                ],
            }
            if isinstance(self_bucket, dict) else None
        ),
    }


def aggregate_people(recs: Iterable[dict]) -> list[dict]:
    """"How do I sound with Mom vs with Asher" ACROSS sessions: one row per
    identified person (keyed by person_id, else by display name — a raw
    "Speaker B" is a per-session artifact and is NOT merged across sessions,
    mirroring /growth's partner_names rule), summing the user's own scored
    turns / labels / escalations in the turns said to them. Sorted by most
    sessions, then most turns."""
    rows: dict[str, dict] = {}
    for rec in recs:
        extras = growth_extras(rec)
        tone = extras.get("self_tone")
        if not tone:
            continue
        for p in tone.get("people") or []:
            pid = p.get("person_id")
            name = p.get("display_name")
            # A real cross-session identity only: a person_id, or a name that
            # came from the identity path (growth_extras already substituted
            # the raw speaker label when no name existed — skip those).
            if not pid and not (isinstance(name, str) and name and not name.lower().startswith("speaker ")):
                continue
            key = f"id:{pid}" if pid else f"name:{name}"
            row = rows.setdefault(key, {
                "person_id": pid,
                "display_name": name,
                "sessions": 0,
                "scored_turns": 0,
                "labels": {},
                "escalation_count": 0,
            })
            if not row["display_name"] and name:
                row["display_name"] = name
            row["sessions"] += 1
            row["scored_turns"] += int(p.get("scored_turns") or 0)
            row["escalation_count"] += int(p.get("escalation_count") or 0)
            for label, n in (p.get("labels") or {}).items():
                row["labels"][label] = row["labels"].get(label, 0) + int(n)
    out = list(rows.values())
    out.sort(key=lambda r: (-r["sessions"], -r["scored_turns"]))
    return out


def dashboard_session(rec: dict, *, patient: str, shared: bool) -> dict:
    """The therapist-dashboard row (``SavedSession`` on the client) for one
    stored recording — live sessions AND uploads, so a therapist's patient
    view is complete.

    ``role`` is the PATIENT label (the dashboard groups + filters by it —
    that is the existing patient/session navigation). Per-turn
    ``toneScores`` carry only what was measured: ``pleasantness`` =
    100 − the batch analysis's heat (the one derivation the PRD defines —
    heat is the escalation score, pleasantness its inverse), ``warmth`` from
    the phone's text tone. Nothing else is filled in; the client averages
    over the keys present."""
    analysis = rec.get("analysis") if isinstance(rec.get("analysis"), dict) else {}
    live = analysis.get("live") if isinstance(analysis.get("live"), dict) else None
    per_turn = analysis.get("per_turn") if isinstance(analysis.get("per_turn"), list) else []
    labels = analysis.get("speaker_labels") if isinstance(analysis.get("speaker_labels"), dict) else {}
    # People labeling: the human's manual labels (incl. the person picked
    # from the people list) sit on top, exactly as the detail endpoint
    # serves them — a therapist sees the patient's own names for people.
    labels = {**labels, **manual_overlay(rec)}
    turns = rec.get("turns") or []
    rows = (live or {}).get("turn_tone") or []
    by_index = {r.get("index"): r for r in rows if isinstance(r, dict)}
    out_turns: list[dict] = []
    pleasantness: list[float] = []
    speakers_out: list[dict] = []
    seen_speakers: set[str] = set()
    for i, turn in enumerate(turns):
        speaker = turn.get("speaker")
        entry = labels.get(speaker) if isinstance(speaker, str) else None
        display = entry.get("display_label") if isinstance(entry, dict) else None
        if isinstance(speaker, str) and speaker not in seen_speakers:
            seen_speakers.add(speaker)
            speakers_out.append({
                "id": speaker,
                "display": display if isinstance(display, str) and display.strip() else speaker,
                "labelSource": entry.get("label_source") if isinstance(entry, dict) else None,
                "personId": entry.get("person_id") if isinstance(entry, dict) else None,
            })
        scores: dict[str, float] = {}
        heat = _num((per_turn[i] or {}).get("heat")) if i < len(per_turn) and isinstance(per_turn[i], dict) else None
        if heat is not None:
            scores["pleasantness"] = max(0.0, min(100.0, 100.0 - heat))
            pleasantness.append(scores["pleasantness"])
        tone = _text_tone(turn)
        warmth = _num(tone.get("warmth")) if tone else None
        if warmth is not None:
            scores["warmth"] = warmth
        row = by_index.get(i) or {}
        out_turns.append({
            "speaker": display if isinstance(display, str) and display.strip() else speaker,
            # People labeling: the raw diarized id + provenance so the client's
            # "Who is this?" sheet can relabel THIS speaker (the display name
            # above is what the row shows; it isn't a stable key).
            "speakerId": speaker,
            "labelSource": entry.get("label_source") if isinstance(entry, dict) else None,
            "personId": entry.get("person_id") if isinstance(entry, dict) else None,
            "text": turn.get("text"),
            "toneScores": scores,
            "isSelf": bool(row.get("is_self")) if row else bool(turn.get("is_self")),
            "toneLabel": row.get("label"),
            "escalated": bool(row.get("escalated")) if row else False,
            "audioLabel": row.get("audio_label"),
            "withPerson": row.get("with_speaker"),
        })
    avg = (sum(pleasantness) / len(pleasantness)) if pleasantness else None
    summary = (live or {}).get("tone_summary") if live else None
    return {
        "id": rec.get("id"),
        "recordingId": rec.get("id"),
        "date": rec.get("created_at"),
        "role": patient,
        "patient": patient,
        "shared": shared,
        "title": rec.get("title") or rec.get("filename"),
        "source": (rec.get("source") or {}).get("type") if isinstance(rec.get("source"), dict) else None,
        "mode": rec.get("mode") or ((live or {}).get("mode") if live else None),
        "turns": out_turns,
        # People labeling: every distinct speaker with its effective label +
        # provenance (first-appearance order) — the dashboard's people line
        # and the detail's speaker picker read this.
        "speakers": speakers_out,
        # Whether the server kept audio for this recording — the "Remember
        # this voice" affordance is only offered when it did (a live session
        # is media_type "none").
        "hasAudio": rec.get("media_type") not in (None, "none"),
        "avgPleasantness": avg,
        "toneSummary": summary,
        "couldHaveSaid": (live or {}).get("could_have_said") if live else None,
        "analysisStatus": (live or {}).get("analysis_status") if live else ("full" if per_turn else None),
    }
