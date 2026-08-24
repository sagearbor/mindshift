"""Unit tests for live_sessions.py — the PURE half of Track 2.

No app, no LLM, no I/O: every function here is a deterministic derivation
over the phone's TurnLocalEvent-shaped turns, so each case is hand-computed.
Endpoint coverage (ingest → store → background analysis/reflection → the
reads that feed YourDay/Growth/Replay/the dashboard) lives in
test_sessions_live.py.
"""

import live_sessions as ls


def _turn(speaker, text, start, end, *, is_self=None, tone=None, pid=None):
    return {
        "speaker": speaker, "text": text, "start_time": start, "end_time": end,
        "is_self": is_self, "text_tone": tone, "speaker_person_id": pid,
    }


# A 2-party live session: A is the user (is_self), B is "Mom" (person p-mom).
# A's turns: warm → frustrated (by score) → defensive (by explicit label).
SESSION_TURNS = [
    _turn("Speaker A", "Hey Mom, I got your message.", 0.0, 2.0,
          is_self=True, tone={"warmth": 80, "frustration": 10}),
    _turn("Speaker B", "You never call back.", 2.5, 4.0, pid="p-mom"),
    _turn("Speaker A", "I was working, I told you that.", 4.5, 6.0,
          is_self=True, tone={"warmth": 20, "frustration": 75}),
    _turn("Speaker B", "That's what you always say.", 6.5, 8.0, pid="p-mom"),
    _turn("Speaker A", "Fine. Whatever you want.", 8.5, 10.0,
          is_self=True, tone={"label": "Defensive", "defensiveness": 40}),
    _turn("Speaker B", "Don't be like that.", 10.5, 12.0, pid="p-mom"),
]
IDENTITIES = [
    {"speaker": "Speaker B", "person_id": "p-mom", "display_name": "Mom",
     "is_self": False, "score": 0.81},
]


class TestToneLabel:
    def test_none_without_any_tone(self):
        assert ls.tone_label(None) is None
        assert ls.tone_label({}) is None
        assert ls.tone_label({"warmth": None, "label": ""}) is None

    def test_explicit_label_wins_and_is_normalized(self):
        assert ls.tone_label({"label": "  Warm ", "frustration": 95}) == "warm"

    def test_dominant_escalation_dim(self):
        assert ls.tone_label({"frustration": 75, "warmth": 20}) == "frustrated"
        assert ls.tone_label({"sarcasm": 61, "frustration": 60}) == "sarcastic"
        assert ls.tone_label({"defensiveness": 90}) == "defensive"

    def test_sad_then_warm_then_neutral(self):
        assert ls.tone_label({"sadness": 70, "warmth": 65}) == "sad"
        assert ls.tone_label({"warmth": 65}) == "warm"
        assert ls.tone_label({"warmth": 30, "frustration": 20}) == "neutral"


class TestEscalation:
    def test_by_label_or_score_never_by_sadness(self):
        assert ls.is_escalated({"label": "angry"}) is True
        assert ls.is_escalated({"frustration": 60}) is True
        assert ls.is_escalated({"label": "warm", "frustration": 85}) is True
        assert ls.is_escalated({"sadness": 95}) is False
        assert ls.is_escalated({"label": "melancholy"}) is False
        assert ls.is_escalated(None) is False


class TestIdentity:
    def test_self_speaker_majority_and_none(self):
        assert ls.self_speaker(SESSION_TURNS) == "Speaker A"
        mixed = SESSION_TURNS + [_turn("Speaker C", "x", 20, 21, is_self=True)]
        assert ls.self_speaker(mixed) == "Speaker A"
        assert ls.self_speaker([_turn("Speaker A", "x", 0, 1)]) is None

    def test_person_map_merges_turn_ids_and_server_verdicts(self):
        people = ls.person_map(SESSION_TURNS, IDENTITIES, "Speaker A")
        assert people == {"Speaker B": {"person_id": "p-mom", "display_name": "Mom"}}
        # Without a server verdict the phone's person id still lands; no
        # name is invented.
        people = ls.person_map(SESSION_TURNS, None, "Speaker A")
        assert people == {"Speaker B": {"person_id": "p-mom", "display_name": None}}
        # …unless the account's enrolled voiceprints know that person id.
        people = ls.person_map(
            SESSION_TURNS, None, "Speaker A",
            known_people=[{"person_id": "p-mom", "display_name": "Mum", "is_self": False}],
        )
        assert people == {"Speaker B": {"person_id": "p-mom", "display_name": "Mum"}}

    def test_self_via_reserved_person_id(self):
        turns = [dict(t, is_self=None) for t in SESSION_TURNS]
        turns[0]["speaker_person_id"] = "self"
        assert ls.self_speaker(turns) == "Speaker A"

    def test_speaker_labels_ladder(self):
        people = ls.person_map(SESSION_TURNS, IDENTITIES, "Speaker A")
        labels = ls.build_speaker_labels(SESSION_TURNS, "Speaker A", people)
        assert labels["Speaker A"] == {"display_label": "You", "label_source": "enrolled"}
        # A matched partner is the ENROLLED rung too (a voiceprint match).
        assert labels["Speaker B"] == {"display_label": "Mom", "label_source": "enrolled"}
        unnamed = ls.build_speaker_labels(SESSION_TURNS, None, {})
        assert unnamed["Speaker A"]["label_source"] == "generic"

    def test_overlay_identity_over_llm_names(self):
        base = {
            "Speaker A": {"display_label": "Sage", "label_source": "name"},
            "Speaker B": {"display_label": "Linda", "label_source": "name"},
            "Speaker C": {"display_label": "Speaker C", "label_source": "generic"},
        }
        identity = {
            "Speaker A": {"display_label": "You", "label_source": "enrolled"},
            "Speaker B": {"display_label": "Mom", "label_source": "enrolled"},
            "Speaker C": {"display_label": "Speaker C", "label_source": "generic"},
        }
        merged = ls.overlay_identity_labels(base, identity)
        assert merged["Speaker A"]["display_label"] == "You"
        assert merged["Speaker B"]["display_label"] == "Mom"   # voiceprint beats guess
        assert merged["Speaker C"]["display_label"] == "Speaker C"
        # A generic identity label never erases an LLM-found name.
        merged = ls.overlay_identity_labels(
            {"Speaker B": base["Speaker B"]}, {"Speaker B": identity["Speaker C"]},
        )
        assert merged["Speaker B"]["display_label"] == "Linda"


class TestToneRowsAndSummary:
    def _rows(self):
        people = ls.person_map(SESSION_TURNS, IDENTITIES, "Speaker A")
        return ls.turn_tone_rows(SESSION_TURNS, "Speaker A", people), people

    def test_rows_are_index_aligned_with_attribution(self):
        rows, _ = self._rows()
        assert [r["index"] for r in rows] == list(range(6))
        assert [r["label"] for r in rows] == [
            "warm", None, "frustrated", None, "defensive", None,
        ]
        assert [r["escalated"] for r in rows] == [False, False, True, False, True, False]
        # The opening self turn (no prior other party) is attributed to the
        # NEXT other party; later ones to the most recent.
        assert [r["with_speaker"] for r in rows if r["is_self"]] == ["Speaker B"] * 3
        assert rows[1]["display_name"] == "Mom" and rows[1]["person_id"] == "p-mom"

    def test_summary_self_and_people(self):
        rows, people = self._rows()
        summary = ls.tone_summary(SESSION_TURNS, rows, "Speaker A", people)
        assert summary["self_speaker"] == "Speaker A"
        me = summary["self"]
        assert me["turns"] == 3 and me["scored_turns"] == 3
        assert me["labels"] == {"warm": 1, "frustrated": 1, "defensive": 1}
        assert me["escalation_turns"] == [2, 4] and me["escalation_count"] == 2
        # Means only over turns that scored the dimension; None otherwise.
        assert me["mean"]["warmth"] == 50          # (80 + 20) / 2
        assert me["mean"]["frustration"] == round((10 + 75) / 2)
        assert me["mean"]["sarcasm"] is None
        assert summary["audio"] is None and summary["audio_tone_surfaced"] is False
        [mom] = summary["people"]
        assert mom["display_name"] == "Mom" and mom["person_id"] == "p-mom"
        assert mom["self_turns"] == 3 and mom["their_turns"] == 3
        assert mom["escalation_turns"] == [2, 4]

    def test_no_self_means_no_self_summary(self):
        turns = [dict(t, is_self=None) for t in SESSION_TURNS]
        rows = ls.turn_tone_rows(turns, None, {})
        summary = ls.tone_summary(turns, rows, None, {})
        assert summary["self"] is None and summary["people"] == []

    def test_audio_flags_only_when_allowed(self):
        flags = [
            {"source": "audio", "speaker": "Speaker A", "start_time": 4.6,
             "end_time": 5.9, "label": "angry", "confidence": 0.7},
            {"source": "text", "speaker": "Speaker A", "start_time": 0.0,
             "end_time": 2.0, "label": "warm", "confidence": 0.9},
        ]
        people = ls.person_map(SESSION_TURNS, IDENTITIES, "Speaker A")
        gated = ls.turn_tone_rows(SESSION_TURNS, "Speaker A", people, flags)
        assert all(r["audio_label"] is None for r in gated)
        allowed = ls.turn_tone_rows(
            SESSION_TURNS, "Speaker A", people, flags, audio_allowed=True,
        )
        assert allowed[2]["audio_label"] == "angry" and allowed[2]["audio_escalated"] is True
        assert allowed[0]["audio_label"] is None   # text-source flags never count as audio
        summary = ls.tone_summary(
            SESSION_TURNS, allowed, "Speaker A", people, audio_allowed=True,
        )
        assert summary["audio"] == {
            "turns": 1, "labels": {"angry": 1}, "escalation_turns": [2],
            "escalation_count": 1,
        }

    def test_audio_tone_gate_is_closed_until_foundation_c(self):
        # tone_id.py isn't on this branch (guarded import) → never surfaced.
        # Once Foundation C merges this assertion becomes env-driven; the
        # guard itself is what's under test here.
        if ls._tone_id is None:
            assert ls.audio_tone_allowed() is False


class TestLiteAnalysisAndMerge:
    def _lite(self):
        return ls.lite_analysis(
            session_id="s1", mode="earpiece", started_at="2026-08-24T10:00:00+00:00",
            ended_at="2026-08-24T10:00:12+00:00", turns=SESSION_TURNS,
            tone_flags=None, speaker_identities=IDENTITIES, title="Call with Mom",
            gap_seconds=60.0, audio_allowed=False,
        )

    def test_lite_shape_is_honest(self):
        lite = self._lite()
        assert lite["per_turn"] == [] and lite["report_cards"] == {}
        # Foundation B's multi-shape report: every ladder reader labels a
        # live session the way it labels a server voiceprint match.
        assert lite["speaker_identity"] == {
            "matched_speaker": "Speaker A",
            "matched": {"Speaker A": "self", "Speaker B": "p-mom"},
            "people": {"self": {"display_name": "You", "is_self": True},
                       "p-mom": {"display_name": "Mom", "is_self": False}},
            "speakers": {}, "source": "live",
        }
        import speaker_id
        assert speaker_id.enrolled_display_labels(lite["speaker_identity"]) == {
            "Speaker A": "You", "Speaker B": "Mom",
        }
        assert lite["speaker_labels"]["Speaker B"]["display_label"] == "Mom"
        [ep] = lite["episodes"]
        assert ep["participants"] == ["You", "Mom"]
        assert ep["mean_heat"] is None                       # no heats measured
        assert ep["self_tone_labels"] == {"warm": 1, "frustrated": 1, "defensive": 1}
        assert ep["self_escalation_count"] == 2
        live = lite["live"]
        assert live["analysis_status"] == "lite" and live["could_have_said"] is None
        assert live["turns_hash"] == ls.turns_hash(SESSION_TURNS)
        assert lite["word_metrics"]["speakers"]["Speaker A"]["word_count"] > 0

    def test_merge_keeps_live_block_and_identity_labels(self):
        lite = self._lite()
        full = {
            "per_turn": [
                {"index": i, "speaker": t["speaker"], "heat": 10 * i, "markers": [],
                 "is_spike": False, "trigger_phrase": None, "voice": None}
                for i, t in enumerate(SESSION_TURNS)
            ],
            "per_speaker": {}, "dynamics": {}, "narrative": "n",
            "report_cards": {"Speaker A": {"score": 70}},
            "speaker_labels": {
                "Speaker A": {"display_label": "You", "label_source": "enrolled"},
                "Speaker B": {"display_label": "Linda", "label_source": "name"},
            },
            "title": None, "word_metrics": None,
        }
        merged = ls.merge_full_analysis(lite, full, SESSION_TURNS, gap_seconds=60.0)
        assert merged["report_cards"] == {"Speaker A": {"score": 70}}
        assert merged["speaker_labels"]["Speaker B"]["display_label"] == "Mom"
        assert merged["title"] == "Call with Mom"
        assert merged["live"]["analysis_status"] == "full"
        assert merged["live"]["tone_summary"]["self"]["escalation_turns"] == [2, 4]
        [ep] = merged["episodes"]
        assert ep["peak_heat"] == 50 and ep["self_escalation_count"] == 2

    def test_duration_prefers_transcript_then_wall_clock(self):
        assert ls.duration_seconds(SESSION_TURNS, None, None) == 12.0
        assert ls.duration_seconds(
            [], "2026-08-24T10:00:00Z", "2026-08-24T10:05:00Z",
        ) == 300.0
        assert ls.duration_seconds([], "bad", "worse") is None


class TestReflectionPromptAndParsing:
    def test_prompt_numbers_turns_and_tags_you(self):
        labels = {"Speaker B": {"display_label": "Mom", "label_source": "name"}}
        user, idx = ls.build_reflect_prompt(SESSION_TURNS, "Speaker A", labels, mode="earpiece")
        assert idx == [0, 2, 4]
        assert "0. (YOU) Hey Mom, I got your message." in user
        assert "1. (Mom) You never call back." in user
        assert user.rstrip().endswith("Session mode: earpiece")
        assert "Reflect on (YOU) turn indexes: 0, 2, 4" in user

    def test_parse_keeps_only_requested_indexes(self):
        data = {"reflections": [
            {"turn_index": 2, "could_have_said": " I hear you — I was slammed at work. ",
             "why": "Owns it.", "tone_read": "defensive"},
            {"turn_index": 1, "could_have_said": "not mine", "why": "", "tone_read": ""},
            {"turn_index": 0, "could_have_said": "", "why": "x", "tone_read": "x"},
            {"turn_index": 4, "could_have_said": "y" * 500, "why": None, "tone_read": 3},
            {"turn_index": 2, "could_have_said": "duplicate", "why": "", "tone_read": ""},
            "junk",
        ]}
        out = ls.parse_reflections(data, [0, 2, 4])
        assert [r["turn_index"] for r in out] == [2, 4]
        assert out[0]["could_have_said"] == "I hear you — I was slammed at work."
        assert out[0]["why"] == "Owns it." and out[0]["tone_read"] == "defensive"
        assert len(out[1]["could_have_said"]) == ls.REFLECT_TEXT_MAX
        assert out[1]["why"] == "" and out[1]["tone_read"] == ""

    def test_parse_rejects_shapeless_payload(self):
        for bad in ({}, {"reflections": "no"}, [], None):
            try:
                ls.parse_reflections(bad, [0])
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for {bad!r}")

    def test_cached_reflection_requires_matching_hash(self):
        analysis = {"live": {
            "could_have_said": [{"turn_index": 0, "could_have_said": "x", "why": "", "tone_read": ""}],
            "reflection": {"turns_hash": ls.turns_hash(SESSION_TURNS)},
        }}
        assert ls.cached_reflection(analysis, SESSION_TURNS) is not None
        changed = [dict(SESSION_TURNS[0], text="different words")] + SESSION_TURNS[1:]
        assert ls.cached_reflection(analysis, changed) is None
        assert ls.cached_reflection(None, SESSION_TURNS) is None
        assert ls.cached_reflection({"live": {"could_have_said": None}}, SESSION_TURNS) is None


class TestAggregates:
    def _rec(self, *, with_tone=True, source="live", created="2026-08-24T10:00:00+00:00"):
        analysis = ls.lite_analysis(
            session_id="s", mode="earpiece", started_at=created, ended_at=created,
            turns=SESSION_TURNS if with_tone else [dict(t, text_tone=None) for t in SESSION_TURNS],
            tone_flags=None, speaker_identities=IDENTITIES, title=None,
            gap_seconds=60.0, audio_allowed=False,
        )
        return {
            "id": "r", "created_at": created, "source": {"type": source},
            "mode": "earpiece", "turns": SESSION_TURNS, "analysis": analysis,
        }

    def test_growth_extras(self):
        extras = ls.growth_extras(self._rec())
        assert extras["source"] == "live" and extras["mode"] == "earpiece"
        assert extras["self_tone"]["labels"] == {"warm": 1, "frustrated": 1, "defensive": 1}
        assert extras["self_tone"]["escalation_count"] == 2
        [mom] = extras["self_tone"]["people"]
        assert mom["display_name"] == "Mom" and mom["escalation_count"] == 2
        # An upload (no live block) contributes nothing — not a neutral bucket.
        upload = {"id": "u", "source": {"type": "upload"}, "analysis": {"per_turn": []}}
        assert ls.growth_extras(upload) == {"source": "upload", "mode": None, "self_tone": None}

    def test_aggregate_people_merges_by_identity_only(self):
        recs = [self._rec(), self._rec(created="2026-08-25T10:00:00+00:00")]
        # A third session with an UNIDENTIFIED partner must not become a row.
        anon_turns = [dict(t, speaker_person_id=None) for t in SESSION_TURNS]
        anon = ls.lite_analysis(
            session_id="a", mode="speaker", started_at="2026-08-26T10:00:00+00:00",
            ended_at="2026-08-26T10:00:00+00:00", turns=anon_turns, tone_flags=None,
            speaker_identities=None, title=None, gap_seconds=60.0, audio_allowed=False,
        )
        recs.append({"id": "a", "source": {"type": "live"}, "turns": anon_turns, "analysis": anon})
        people = ls.aggregate_people(recs)
        assert len(people) == 1
        [mom] = people
        assert mom["display_name"] == "Mom" and mom["sessions"] == 2
        assert mom["scored_turns"] == 6 and mom["escalation_count"] == 4
        assert mom["labels"] == {"warm": 2, "frustrated": 2, "defensive": 2}

    def test_dashboard_session_projection(self):
        rec = self._rec()
        rec["analysis"]["per_turn"] = [
            {"index": i, "speaker": t["speaker"], "heat": 30} for i, t in enumerate(SESSION_TURNS)
        ]
        rec["analysis"]["live"]["could_have_said"] = [
            {"turn_index": 2, "could_have_said": "x", "why": "y", "tone_read": "z"},
        ]
        row = ls.dashboard_session(rec, patient="You", shared=False)
        assert row["role"] == "You" and row["source"] == "live" and row["mode"] == "earpiece"
        assert row["avgPleasantness"] == 70
        first = row["turns"][0]
        assert first["speaker"] == "You" and first["isSelf"] is True
        assert first["toneScores"] == {"pleasantness": 70, "warmth": 80}
        assert first["toneLabel"] == "warm" and first["withPerson"] == "Speaker B"
        assert row["turns"][1]["speaker"] == "Mom" and row["turns"][1]["toneScores"] == {"pleasantness": 70}
        assert row["turns"][2]["escalated"] is True
        assert row["couldHaveSaid"][0]["turn_index"] == 2
        assert row["toneSummary"]["self"]["escalation_count"] == 2
        # No heats at all → no pleasantness, honest None average.
        rec["analysis"]["per_turn"] = []
        assert ls.dashboard_session(rec, patient="You", shared=False)["avgPleasantness"] is None
