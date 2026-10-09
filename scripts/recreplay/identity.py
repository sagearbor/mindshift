"""Which voice in the recording is the owner (the wearer)?

Three independent votes, all shown in the report:

1. ``notes-explicit``     — ``who: I'm S2`` names an annotation voice outright.
2. ``notes-description``  — ``who: I'm the man with the low voice`` matched
                            against each annotation voice's description/age.
3. ``voiceprint``         — the owner's enrolled print
                            (``tmp/private_fixtures/owner_profile.json``, the
                            server's speechbrain ECAPA space) against each
                            Deepgram speaker's pooled audio, with the SERVER's
                            own matcher (server/speaker_id.py).

The owner's own words outrank the voiceprint (a print from another room
scores 0.08-0.45 for the same man — auto-memory / replay.real.test.ts), and
the voiceprint outranks a blind fallback. Disagreement is a warning, never
silently resolved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import notes as notes_mod
from .annotation import Alignment, Annotation

# A print from the same session scores >= 0.6; across rooms the same man
# can sit at 0.24-0.45 (speaker_id.CROSS_MATCH_*). Below this, or without a
# clear margin over the runner-up, the voiceprint abstains.
VOICEPRINT_MIN = 0.40
VOICEPRINT_MARGIN = 0.10
_CONFLICTS = (("male", "female"), ("child", "adult"), ("low", "high"))


@dataclass
class Resolution:
    wearer_label: str | None          # Deepgram label ("Speaker A")
    wearer_ann_id: str | None          # annotation voice ("S2")
    method: str
    votes: dict = field(default_factory=dict)
    agree: bool | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"wearer_label": self.wearer_label, "wearer_ann_id": self.wearer_ann_id, "method": self.method,
                "votes": self.votes, "agree": self.agree, "warnings": self.warnings}


def description_scores(self_clause: str | None, others_clause: str | None, a: Annotation) -> dict[str, float]:
    want = notes_mod.descriptor_keywords(self_clause)
    others = notes_mod.descriptor_keywords(others_clause)
    out: dict[str, float] = {}
    for sp in a.speakers:
        have = notes_mod.descriptor_keywords(" ".join(x for x in (sp.voice_description, sp.approx_age) if x))
        if sp.approx_age in ("child", "teen"):
            have.add("child")
            have.discard("adult")
        elif sp.approx_age in ("adult", "older_adult"):
            have.add("adult")
        score = float(len(want & have))
        for x, y in _CONFLICTS:
            if (x in want and y in have) or (y in want and x in have):
                score -= 1.5
        # matching the description of "the other voice" counts against
        score -= 0.5 * len((others - want) & have)
        out[sp.id] = score
    return out


def resolve(
    notes: notes_mod.Notes,
    a: Annotation | None,
    al: Alignment | None,
    *,
    voiceprint_scores: dict[str, float] | None,
    dg_speakers: list[str],
    talk_seconds: dict[str, float] | None = None,
    override: str | None = None,
) -> Resolution:
    votes: dict = {}
    warnings: list[str] = []
    speaker_map = al.speaker_map if al else {}

    explicit = notes.self_speaker_id
    desc_pick = None
    if a is not None and a.speakers:
        ds = description_scores(notes.self_clause, notes.others_clause, a)
        votes["description"] = ds
        if ds:
            ranked = sorted(ds.items(), key=lambda kv: kv[1], reverse=True)
            if ranked[0][1] > 0 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
                desc_pick = ranked[0][0]
            elif notes.self_clause:
                warnings.append(f"who: {notes.self_clause!r} does not single out one annotation voice ({ds})")
    vp_label = None
    if voiceprint_scores:
        votes["voiceprint"] = {k: round(v, 3) for k, v in voiceprint_scores.items()}
        ranked = sorted(voiceprint_scores.items(), key=lambda kv: kv[1], reverse=True)
        best, score = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else -1.0
        if score >= VOICEPRINT_MIN and score - runner >= VOICEPRINT_MARGIN:
            vp_label = best
        else:
            warnings.append(f"voiceprint abstains: best {best} {score:.2f}, runner-up {runner:.2f} "
                            f"(needs >= {VOICEPRINT_MIN} and a {VOICEPRINT_MARGIN} margin)")
    votes["voiceprint_label"] = vp_label
    votes["explicit"] = explicit
    votes["description_pick"] = desc_pick

    def label_for(sid: str | None) -> str | None:
        return speaker_map.get(sid) if sid else None

    if override:
        res = Resolution(override, _ann_for(override, speaker_map), "override")
    elif explicit and (a is None or a.speaker(explicit)):
        res = Resolution(label_for(explicit), explicit, "notes-explicit")
    elif desc_pick:
        res = Resolution(label_for(desc_pick), desc_pick, "notes-description")
    elif vp_label:
        res = Resolution(vp_label, _ann_for(vp_label, speaker_map), "voiceprint")
    else:
        talk = talk_seconds or {}
        pick = max(dg_speakers, key=lambda s: talk.get(s, 0.0)) if talk else (dg_speakers[0] if dg_speakers else None)
        res = Resolution(pick, _ann_for(pick, speaker_map), "fallback-most-talk")
        warnings.append(f"could not tell which voice is the owner; assumed the one who talks most ({pick}). "
                        "Add `who: I'm S<n>` to the notes.")
    if explicit and a is not None and not a.speaker(explicit):
        warnings.append(f"who: names {explicit} but the annotation has no such voice")
    if res.wearer_label is None and res.wearer_ann_id:
        warnings.append(f"annotation voice {res.wearer_ann_id} did not overlap any Deepgram speaker")
    if vp_label and res.method not in ("voiceprint", "override"):
        res.agree = vp_label == res.wearer_label
        if not res.agree:
            warnings.append(f"the owner's voiceprint points at {vp_label} but the notes point at {res.wearer_label}: "
                            "votes disagree (notes win; check the report's identity section)")
    res.votes = votes
    res.warnings = warnings
    return res


def _ann_for(label: str | None, speaker_map: dict[str, str]) -> str | None:
    for sid, lab in speaker_map.items():
        if lab == label:
            return sid
    return None


# ---------------------------------------------------------------------------
# The voiceprint vote (server matcher, optional deps)
# ---------------------------------------------------------------------------

def load_profile(path: Path) -> dict | None:
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d.get("embedding"), list) else None


def voiceprint_scores(pcm16: np.ndarray, turns: list[dict], profile: dict, sr: int = 16000) -> dict[str, float] | None:
    """Cosine of the owner's print against each Deepgram speaker's pooled
    turns (>= 1 s each), via server/speaker_id.py. None when the voice deps
    (torch + speechbrain) are unavailable."""
    try:
        import speaker_id
        if not speaker_id.is_available():
            return None
    except Exception:  # noqa: BLE001 — optional dependency
        return None
    ref = np.asarray(profile["embedding"], dtype=np.float32)
    pcm = pcm16.astype(np.float32) / 32768.0
    by: dict[str, list[np.ndarray]] = {}
    for t in turns:
        a, b = int(t["start_time"] * sr), int(t["end_time"] * sr)
        if b - a >= sr:
            by.setdefault(t["speaker"], []).append(pcm[a:b])
    out: dict[str, float] = {}
    for label, chunks in by.items():
        emb = speaker_id.embed_pcm(np.concatenate(chunks), sr)
        out[label] = float(speaker_id.cosine(emb, ref))
    return out or None
