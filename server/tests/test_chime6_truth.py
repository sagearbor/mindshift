"""Is CHiME-6's speaker ground truth actually trustworthy now?

The retraction (docs/plans/2026-09-19-heat-map-rounds.md, "CHiME-6
retraction"): without the CHiME-6 transcript JSONs, scripts/chime6_corpus.py
fell back to deriving speaker turns from headset ENERGY — the same method
AMI uses. AMI's binaural headsets are close enough in gain that this works
(passed an 8.2%-overlap sanity check); CHiME-6's are not gain-matched, so the
"loudest headset wins" bleed-rejection test handed nearly every frame to
whichever microphone happened to run hottest — one participant credited with
~85% of all speech in the first segment alone. The identity and overlap
columns built on that were struck from the graph.

This test loads the REGENERATED manifest — built from the official CHiME-6
transcript JSONs (CHiME6_transcriptions.tar.gz, OpenSLR resource 150,
https://openslr.org/150/) — and asserts the two things the retraction hinged
on: the truth source really is the transcript, not the unreliable headset
fallback, and no single participant dominates the way the broken labels did.

SKIPPED without the manifest, which lives in the MAIN repo's gitignored tmp/
(never this worktree's), since CHiME-6 audio/transcripts are not committed.
Build it with:

    python scripts/chime6_corpus.py --root tmp/corpora/chime6
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
MANIFEST = REPO / "tmp/corpora/chime6/heatmap_manifest.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="CHiME-6 manifest not built — see this module's docstring "
           "(tmp/corpora/chime6/heatmap_manifest.json not found)",
)


@pytest.fixture(scope="module")
def entries():
    data = json.loads(MANIFEST.read_text())
    assert len(data) > 0, f"{MANIFEST} is empty"
    return data


def test_every_segment_is_built_from_the_transcript_not_the_headset_fallback(entries):
    """The retraction's root cause was the headset-energy fallback silently
    standing in for real ground truth. Every regenerated segment must say so
    explicitly, so a session missing its transcript JSON fails loudly here
    instead of quietly re-polluting the identity/overlap columns."""
    sources = Counter(e.get("speaker_truth") for e in entries)
    assert sources.get("transcript", 0) == len(entries), (
        f"not every CHiME-6 segment used transcript truth: {dict(sources)} — "
        f"a session is missing its transcriptions/dev/S0*.json under "
        f"tmp/corpora/chime6/raw/ and fell back to the unreliable headset-energy "
        f"derivation (see scripts/chime6_corpus.py)"
    )


def test_no_participant_dominates_any_segment(entries):
    """The concrete symptom of the broken labels: one headset credited with
    ~85% of speech in a segment. Real conversation at a four-person dinner is
    never that lopsided — assert no one clears half the talk time in any
    single segment."""
    worst = 0.0
    for e in entries:
        talk = e.get("speaking_seconds") or {}
        total = sum(talk.values())
        if total <= 0:
            continue
        share = max(talk.values()) / total
        worst = max(worst, share)
        assert share <= 0.5, (
            f"{e['id']}: one participant has {share:.1%} of speech "
            f"({talk}) — looks like the bad headset-energy derivation, not "
            f"transcript truth"
        )
    assert worst > 0.0, "no segment had any speech at all — nothing was actually checked"


def test_no_participant_dominates_the_whole_corpus(entries):
    """Same check aggregated across every segment (i.e. per session), so a
    skew that is invisible segment-by-segment but real over the whole
    recording would still be caught."""
    totals = Counter()
    for e in entries:
        for spk, secs in (e.get("speaking_seconds") or {}).items():
            totals[spk] += secs
    grand = sum(totals.values())
    assert grand > 0
    share = max(totals.values()) / grand
    assert share <= 0.5, f"one participant has {share:.1%} of all speech across the corpus: {dict(totals)}"


def test_speaker_ids_match_the_headset_filenames(entries):
    """Speaker keys should be the transcript's own participant ids (P05, P06,
    ...), which are exactly the headset filename stems (S02_P05.wav) — the
    whole point of ground truth this cheap is that no relabelling is needed."""
    import re

    for e in entries:
        for spk in (e.get("speakers") or {}):
            assert re.fullmatch(r"P\d+", spk), f"{e['id']}: unexpected speaker id {spk!r}"
