"""Episode segmentation — split one recording's diarized transcript into
conversation EPISODES on silence gaps ("Your Day" timeline, Companion P1).

A long companion-style recording (a dinner, a work block) is rarely one
continuous conversation: people talk, fall silent, talk again. This module
finds those seams and summarizes each stretch so the client can draw a day
timeline of episodes instead of one undifferentiated blob.

Design constraints (house rules):

* PURE — no I/O, no LLM calls, no clock reads. Everything an episode carries
  is DERIVED from data the analysis pipeline already produced (turn timing,
  per-turn heats, resolved speaker labels, the recording title). The one-line
  ``summary`` is either the existing LLM title (single-episode recordings —
  the title IS that episode's summary, no new LLM cost) or a verbatim excerpt
  of the episode's opening turn. Never a fabricated description.
* Honest nulls — a transcript without timestamps (text /analyze, degraded
  decode) cannot reveal silence gaps, so it is ONE episode with null
  start/end rather than invented boundaries. Missing/misaligned per-turn
  heats yield ``mean_heat``/``peak_heat`` of ``None``, never 0.
* Backward compatible — an existing short recording segments to exactly one
  episode; old stored analyses can be segmented on read (see
  :func:`episodes_from_analysis`) so the detail endpoint serves episodes for
  every recording without a migration.

The boundary rule: a new episode starts when the gap between one turn's end
and the next turn's start exceeds ``gap_seconds`` (default 60s, tunable via
the caller — ``EPISODE_GAP_SECONDS`` env in main.py).
"""

from __future__ import annotations

# Default silence gap (seconds) that splits two turns into separate episodes.
DEFAULT_GAP_SECONDS = 60.0

# Cap for the verbatim opening-turn excerpt used as a summary fallback.
_EXCERPT_MAX_CHARS = 100

# Provenance values for Episode.summary_source (mirrors the label_source
# pattern): "title" = the existing LLM-suggested recording title; "excerpt" =
# a verbatim quote of the episode's first turn.
SUMMARY_SOURCE_TITLE = "title"
SUMMARY_SOURCE_EXCERPT = "excerpt"


def _excerpt(text: object) -> str | None:
    """A verbatim, length-capped quote of ``text`` (the honest no-LLM summary).

    Whitespace is collapsed for display; content is never rewritten. ``None``
    for a blank/non-string turn text (never an empty-string summary).
    """
    if not isinstance(text, str):
        return None
    collapsed = " ".join(text.split())
    if not collapsed:
        return None
    if len(collapsed) <= _EXCERPT_MAX_CHARS:
        return collapsed
    return collapsed[: _EXCERPT_MAX_CHARS - 1].rstrip() + "…"


def _display_label(
    speaker: str,
    speaker_labels: dict | None,
    enrolled_speaker: str | None,
) -> str:
    """The display label for one canonical speaker id.

    Precedence mirrors the server's label ladder: an enrolled voiceprint match
    renders as "You"; else the resolved ``speaker_labels`` display_label; else
    the raw id. Read defensively — stored analyses vary by server version.
    """
    if enrolled_speaker is not None and speaker == enrolled_speaker:
        return "You"
    entry = (speaker_labels or {}).get(speaker)
    if isinstance(entry, dict):
        label = entry.get("display_label")
        if isinstance(label, str) and label.strip():
            return label.strip()
    return speaker


def _matched_speaker(speaker_identity: object) -> str | None:
    """``speaker_identity.matched_speaker`` when present and non-empty, else
    ``None`` — the same defensive read as main._enrolled_speaker."""
    if not isinstance(speaker_identity, dict):
        return None
    matched = speaker_identity.get("matched_speaker")
    if isinstance(matched, str) and matched.strip():
        return matched
    return None


def _turn_time(turn: dict, key: str) -> float | None:
    """A turn's ``start_time``/``end_time`` as float, or ``None`` (absent or
    non-numeric — booleans excluded)."""
    value = turn.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _boundaries(turns: list[dict], gap_seconds: float) -> list[int]:
    """Indexes where a NEW episode starts (always includes 0 for non-empty
    input). A boundary is declared only when BOTH sides of the gap carry real
    timestamps — missing timing reads as contiguous, never as silence."""
    starts = [0]
    prev_end: float | None = None
    for i, turn in enumerate(turns):
        start = _turn_time(turn, "start_time")
        if (
            i > 0
            and start is not None
            and prev_end is not None
            and start - prev_end > gap_seconds
        ):
            starts.append(i)
        end = _turn_time(turn, "end_time")
        if end is not None:
            # Track the furthest spoken moment so an out-of-order or
            # overlapping end never manufactures a phantom gap.
            prev_end = end if prev_end is None else max(prev_end, end)
    return starts


def segment_episodes(
    turns: list[dict],
    *,
    per_turn: list[dict] | None = None,
    speaker_labels: dict | None = None,
    speaker_identity: dict | None = None,
    title: str | None = None,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> list[dict]:
    """Split ``turns`` into episodes on silence gaps > ``gap_seconds``.

    ``turns`` are the transcribed turn dicts ({speaker, text, start_time,
    end_time}); ``per_turn`` is the analysis's index-aligned per-turn list
    (heats) — used only when its length matches, else heats are honest nulls;
    ``speaker_labels``/``speaker_identity`` feed participant display labels
    (incl. "You" for an enrolled match); ``title`` is the recording's existing
    display title, reused as the summary when the whole recording is a single
    episode.

    Returns a list of plain JSON-ready episode dicts::

        {index, start_time, end_time, duration_seconds,
         first_turn_index, last_turn_index, turn_count,
         speakers, participants, mean_heat, peak_heat,
         summary, summary_source}
    """
    if not turns:
        return []

    # Heats only when the analysis aligns with the transcript; else null.
    heats: list[int | None] = [None] * len(turns)
    if isinstance(per_turn, list) and len(per_turn) == len(turns):
        for i, entry in enumerate(per_turn):
            if isinstance(entry, dict):
                heat = entry.get("heat")
                if isinstance(heat, (int, float)) and not isinstance(heat, bool):
                    heats[i] = int(heat)

    enrolled = _matched_speaker(speaker_identity)
    starts = _boundaries(turns, gap_seconds)
    single_episode = len(starts) == 1

    episodes: list[dict] = []
    for ep_index, first in enumerate(starts):
        last = (starts[ep_index + 1] - 1) if ep_index + 1 < len(starts) else len(turns) - 1
        ep_turns = turns[first: last + 1]

        # Timing: first known start / furthest known end inside the episode.
        ep_start = _turn_time(ep_turns[0], "start_time")
        ep_ends = [t for t in (_turn_time(x, "end_time") for x in ep_turns) if t is not None]
        ep_end = max(ep_ends) if ep_ends else None
        duration = (
            ep_end - ep_start
            if ep_start is not None and ep_end is not None and ep_end >= ep_start
            else None
        )

        # Participants: canonical ids in first-appearance order + display labels.
        speakers = list(dict.fromkeys(
            t.get("speaker") for t in ep_turns if isinstance(t.get("speaker"), str)
        ))
        participants = list(dict.fromkeys(
            _display_label(sp, speaker_labels, enrolled) for sp in speakers
        ))

        # Heat stats over the turns that actually carry an aligned heat.
        ep_heats = [h for h in heats[first: last + 1] if h is not None]
        mean_heat = round(sum(ep_heats) / len(ep_heats), 1) if ep_heats else None
        peak_heat = max(ep_heats) if ep_heats else None

        # Summary: the existing LLM title covers a whole-recording episode; a
        # multi-episode recording gets a verbatim opening-turn excerpt each.
        cleaned_title = (title or "").strip() or None
        if single_episode and cleaned_title:
            summary, summary_source = cleaned_title, SUMMARY_SOURCE_TITLE
        else:
            summary = _excerpt(ep_turns[0].get("text"))
            summary_source = SUMMARY_SOURCE_EXCERPT if summary is not None else None

        episodes.append({
            "index": ep_index,
            "start_time": ep_start,
            "end_time": ep_end,
            "duration_seconds": duration,
            "first_turn_index": first,
            "last_turn_index": last,
            "turn_count": len(ep_turns),
            "speakers": speakers,
            "participants": participants,
            "mean_heat": mean_heat,
            "peak_heat": peak_heat,
            "summary": summary,
            "summary_source": summary_source,
        })
    return episodes


def episodes_from_analysis(
    turns: list[dict],
    analysis: dict | None,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> list[dict] | None:
    """Episodes for a STORED recording, derived from its persisted analysis.

    The read-path backfill: recordings analyzed before episode segmentation
    shipped carry no ``episodes`` in analysis.json, so the detail endpoint
    derives them on the fly from what IS stored. Returns ``None`` when there is
    no analysis at all (a stored-but-unanalyzed recording has nothing honest to
    segment — heat, labels, and title all live in the analysis).
    """
    if not isinstance(analysis, dict):
        return None
    return segment_episodes(
        turns or [],
        per_turn=analysis.get("per_turn"),
        speaker_labels=analysis.get("speaker_labels"),
        speaker_identity=analysis.get("speaker_identity"),
        title=analysis.get("title"),
        gap_seconds=gap_seconds,
    )
