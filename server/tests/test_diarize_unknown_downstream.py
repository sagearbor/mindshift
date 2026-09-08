"""The diarizer's "Unknown" label (speaker_id.UNKNOWN_SPEAKER, 2026-08-30) is
speech no found voice claimed. Downstream it must never become a person:
no talk share, no heat stats / per-speaker block (and so no report card
demanded of the LLM), never one of the two coupled speakers, never matched
to an enrolled voiceprint ("You"). Pure functions — no torch, no LLM."""

from __future__ import annotations

import numpy as np

import dynamics
import speaker_id

U = speaker_id.UNKNOWN_SPEAKER


class TestDynamicsExcludeUnknown:
    def test_talk_share_ignores_unknown_and_still_sums_to_one(self):
        shares = dynamics.talk_share(["A", U, "B", U], [30, 500, 10, 500])
        assert shares == {"A": 0.75, "B": 0.25}
        assert U not in shares

    def test_talk_share_all_unknown_is_empty_not_a_crash(self):
        assert dynamics.talk_share([U, U], [10, 20]) == {}

    def test_speaker_heat_stats_has_no_unknown_entry(self):
        stats = dynamics.speaker_heat_stats(["A", U, "B"], [40, 90, 20])
        assert set(stats) == {"A", "B"}
        assert stats["A"]["turns"] == 1 and stats["B"]["peak_heat"] == 20

    def test_coupling_pairs_the_two_most_active_real_speakers(self):
        # Unknown has MORE turns than anyone; it is still never ranked.
        speakers = ["A", "B"] * 6 + [U] * 14
        heats = [10, 15, 40, 30, 55, 20, 25, 10, 15, 12, 18, 14] + [50] * 14
        result = dynamics.compute_coupling(speakers, heats)
        assert U not in result["description"]
        assert "A" in result["description"] and "B" in result["description"]

    def test_coupling_with_one_real_speaker_is_an_honest_monologue(self):
        result = dynamics.compute_coupling(["A", U, "A", U], [10, 20, 30, 40])
        assert result["strength"] is None and result["leader"] is None
        assert "only one speaker" in result["description"].lower()


class TestEnrollmentMatchingSkipsUnknown:
    def test_identify_speakers_multi_never_embeds_or_matches_unknown(self, monkeypatch):
        seen: list[str] = []
        vec = np.zeros(4, dtype=np.float32)
        vec[0] = 1.0

        def fake_embed_speaker(pcm, sr, turns, speaker, *, min_seconds=None):
            seen.append(speaker)
            return vec

        monkeypatch.setattr(speaker_id, "embed_speaker", fake_embed_speaker)
        turns = [
            {"speaker": "Speaker A", "text": "hi", "start_time": 0.0, "end_time": 2.0},
            {"speaker": U, "text": "??", "start_time": 2.0, "end_time": 4.0},
            {"speaker": "Speaker B", "text": "yo", "start_time": 4.0, "end_time": 6.0},
        ]
        report = speaker_id.identify_speakers_multi(
            np.zeros(6 * 16000, dtype=np.float32), 16000, turns,
            {speaker_id.SELF_PERSON_ID: vec},
        )
        assert seen == ["Speaker A", "Speaker B"]
        assert U not in report.get("speakers", {})
        assert U not in (report.get("matched") or {})
        assert report.get("matched_speaker") != U
