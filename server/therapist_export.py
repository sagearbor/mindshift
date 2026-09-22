"""One session, as a document a therapist can put in a patient's file.

``GET /therapist/export/{episode_id}?format=text|pdf`` (routers/therapist.py)
serves what these builders produce. Both formats carry the SAME sections, in
the same order, so a text export and a PDF of the same session say the same
things:

    1. Header      — patient, date, title, mode, duration, who exported it.
    2. Disclosure  — how the therapist came to have this session
                     (auto-share / by hand / on the call) and the consent
                     record in force when it was shared (server/consent.py).
                     A session with no stamp says so plainly.
    3. Tone        — the patient's own tone distribution, escalation count,
                     and how they sounded with each named person.
    4. Escalations — the turn numbers the phone flagged on the patient's own
                     turns (the markers the dashboard draws).
    5. Transcript  — every turn: speaker, tone label, ↑ for an escalation,
                     the pleasantness score when one was measured, and any
                     "could have said" reflection under the turn it belongs
                     to.
    6. Notes       — the exporting therapist's OWN private notes on this
                     episode (``/therapist/notes/{id}``). Never another
                     viewer's: notes are per-viewer by construction.

Input is the dashboard projection (``live_sessions.dashboard_session``), so
the document shows exactly what the therapist saw on screen — the patient's
own names for people included — with no second derivation to drift.

Both builders are PURE (no store, no HTTP, no LLM): a session dict in,
``str`` / ``bytes`` out. reportlab is imported INSIDE the PDF builder, the
same way main.py's ``_build_pdf_export`` does it, so a text export never
pays for the import and a missing reportlab is a PDF-only failure.

Honesty rules kept here: a score that was never measured prints "—", never
0; an empty section prints its "nothing here" line rather than vanishing (a
patient file must not leave the reader guessing whether the section was
empty or the export was truncated).
"""

from __future__ import annotations

from typing import Any

import consent as consent_mod

RULE = "-" * 66

# reportlab's Paragraph parses mini-HTML; the same bound main.py's PDF export
# uses keeps a pathological transcript from producing a 400-page document.
PDF_MAX_TURNS = 400


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _score_text(turn: dict) -> str:
    score = _num((turn.get("toneScores") or {}).get("pleasantness"))
    return "—" if score is None else str(round(score))


def _labels_line(bucket: dict | None) -> str:
    """"mostly warm (5), tense (2)" from a tone bucket's label distribution."""
    if not isinstance(bucket, dict):
        return "not measured"
    labels = bucket.get("labels")
    if not isinstance(labels, dict) or not labels:
        return "not measured"
    ordered = sorted(labels.items(), key=lambda kv: (-int(kv[1] or 0), str(kv[0])))
    return ", ".join(f"{name} ({count})" for name, count in ordered)


def _fmt_duration(seconds: Any) -> str:
    value = _num(seconds)
    if value is None or value <= 0:
        return "unknown"
    minutes, secs = divmod(int(round(value)), 60)
    return f"{minutes}m {secs:02d}s"


def escalation_turn_numbers(session: dict) -> list[int]:
    """1-based turn numbers flagged as escalations on the PATIENT's own
    turns — the same set TherapistSessionPanel draws as red markers."""
    return [
        i + 1
        for i, turn in enumerate(session.get("turns") or [])
        if turn.get("escalated") and turn.get("isSelf")
    ]


def disclosure_lines(session: dict) -> list[str]:
    """How the exporting viewer came to have this session, in plain words."""
    origin = session.get("shareOrigin")
    if not origin:
        return [
            "This session carries no sharing record (it was shared before "
            "MindShift recorded one, or it is your own session)."
        ]
    lines = [consent_mod.ORIGIN_LABELS.get(origin, str(origin)).capitalize() + "."]
    shared_at = session.get("sharedAt")
    if shared_at:
        lines.append(f"Shared: {shared_at}")
    record = session.get("consent")
    if isinstance(record, dict):
        lines.append(
            f"Patient consent: granted {record.get('at')} "
            f"(scope \"{record.get('scope')}\", disclosure version "
            f"{record.get('text_version')})."
        )
        text = consent_mod.disclosure(str(record.get("scope") or ""))
        if text:
            lines.append(f"Wording shown to the patient: “{text}”")
    else:
        lines.append("No consent record is stored against this session.")
    return lines


def _sections(session: dict, *, note: str, patient: str, exported_by: str, exported_at: str) -> list[tuple[str, list[str]]]:
    """Every section as ``(heading, lines)`` — the single source both
    builders render, so the two formats can never drift apart."""
    turns = session.get("turns") or []
    summary = session.get("toneSummary") if isinstance(session.get("toneSummary"), dict) else None
    self_bucket = (summary or {}).get("self") if summary else None
    people = (summary or {}).get("people") or []
    escalations = escalation_turn_numbers(session)
    reflections = {
        r.get("turn_index"): r
        for r in (session.get("couldHaveSaid") or [])
        if isinstance(r, dict)
    }

    header = [
        f"Patient: {patient}",
        f"Session: {session.get('title') or 'Untitled'}",
        f"Date: {session.get('date') or 'unknown'}",
        f"Mode: {session.get('mode') or session.get('source') or 'unknown'}",
        f"Turns: {len(turns)}",
        f"Duration: {_fmt_duration(session.get('durationSeconds'))}",
        f"Exported by: {exported_by} at {exported_at}",
        f"Episode id: {session.get('id') or ''}",
    ]

    tone: list[str] = []
    if isinstance(self_bucket, dict):
        tone.append(f"Patient's own tone: {_labels_line(self_bucket)}")
        tone.append(
            f"Escalations on their own turns: {self_bucket.get('escalation_count', 0)} "
            f"of {self_bucket.get('scored_turns', 0)} scored turns"
        )
    else:
        tone.append("No turn in this session was identified as the patient's own.")
    avg = _num(session.get("avgPleasantness"))
    tone.append(f"Average pleasantness: {'—' if avg is None else round(avg)}")
    for person in people:
        if not isinstance(person, dict):
            continue
        name = person.get("display_name") or person.get("speaker")
        tone.append(
            f"  With {name}: they spoke {person.get('their_turns', 0)} turn(s); "
            f"patient spoke to them {person.get('self_turns', 0)}× — "
            f"{_labels_line(person)}; {person.get('escalation_count', 0)} escalation(s)"
        )

    escalation_lines = (
        ["No escalations were flagged on the patient's turns."]
        if not escalations
        else [f"Flagged on turns: {', '.join(str(n) for n in escalations)}"]
    )

    transcript: list[str] = []
    for i, turn in enumerate(turns):
        speaker = turn.get("speaker") or "Unknown"
        marks = []
        if turn.get("toneLabel"):
            marks.append(str(turn["toneLabel"]))
        if turn.get("escalated"):
            marks.append("ESCALATION")
        suffix = f" [{' · '.join(marks)}]" if marks else ""
        transcript.append(
            f"{i + 1:>3}. {speaker} ({_score_text(turn)}){suffix}: {turn.get('text') or ''}"
        )
        reflection = reflections.get(i)
        if isinstance(reflection, dict):
            transcript.append(f"     → could have said: {reflection.get('could_have_said') or ''}")
            if reflection.get("why"):
                transcript.append(f"       why: {reflection['why']}")
            if reflection.get("tone_read"):
                transcript.append(f"       tone read: {reflection['tone_read']}")
    if not transcript:
        transcript.append("(no turns)")

    could: list[str] = []
    for reflection in session.get("couldHaveSaid") or []:
        if not isinstance(reflection, dict):
            continue
        index = reflection.get("turn_index")
        number = index + 1 if isinstance(index, int) else "?"
        could.append(f"Turn {number}: {reflection.get('could_have_said') or ''}")
        if reflection.get("why"):
            could.append(f"  why: {reflection['why']}")
    if not could:
        could.append("No reflections were generated for this session.")

    return [
        ("Session", header),
        ("How you have this session", disclosure_lines(session)),
        ("Tone", tone),
        ("Escalation markers", escalation_lines),
        ("Transcript", transcript),
        ("What could have been said", could),
        ("Your notes", (note.strip() or "(no notes)").splitlines() or ["(no notes)"]),
    ]


def build_text(
    session: dict,
    *,
    note: str = "",
    patient: str = "",
    exported_by: str = "",
    exported_at: str = "",
) -> str:
    """The plain-text export (the format the phone shares/copies)."""
    out = ["MindShift — session record for a patient file", RULE, ""]
    for heading, lines in _sections(
        session, note=note, patient=patient or "unknown",
        exported_by=exported_by or "unknown", exported_at=exported_at,
    ):
        out.append(heading.upper())
        out.append(RULE)
        out.extend(f"  {line}" for line in lines)
        out.append("")
    out.append(
        "Generated by MindShift. Clinical judgement remains the therapist's; "
        "tone scores are automated estimates, not diagnoses."
    )
    return "\n".join(out)


def build_pdf(
    session: dict,
    *,
    note: str = "",
    patient: str = "",
    exported_by: str = "",
    exported_at: str = "",
) -> bytes:
    """The same document as :func:`build_text`, as PDF bytes.

    Every interpolated string is XML-escaped before it enters a reportlab
    ``Paragraph`` (its mini-HTML markup) — a stray ``<`` or ``&`` in real
    speech must not break parsing or inject styling. Same guard, same
    reason, as main.py's ``_build_pdf_export``."""
    import io
    from xml.sax.saxutils import escape

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    story: list = [
        Paragraph("MindShift — session record", styles["Title"]),
        Spacer(1, 10),
    ]
    for heading, lines in _sections(
        session, note=note, patient=patient or "unknown",
        exported_by=exported_by or "unknown", exported_at=exported_at,
    ):
        story.append(Paragraph(escape(heading), styles["Heading2"]))
        shown = lines[:PDF_MAX_TURNS] if heading == "Transcript" else lines
        for line in shown:
            story.append(Paragraph(escape(line), styles["Normal"]))
        if len(shown) < len(lines):
            story.append(Paragraph(
                escape(f"… {len(lines) - len(shown)} further lines omitted"),
                styles["Normal"],
            ))
        story.append(Spacer(1, 10))
    story.append(Paragraph(
        escape(
            "Generated by MindShift. Clinical judgement remains the "
            "therapist's; tone scores are automated estimates, not diagnoses."
        ),
        styles["Normal"],
    ))
    doc.build(story)
    return buf.getvalue()
