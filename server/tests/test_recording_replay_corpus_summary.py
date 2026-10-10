"""The corpus batch summary (recreplay.corpus_summary): per-item metrics from
a run bundle, per-group/per-corpus aggregation with heated and calm kept
apart, the defect detectors, and the HTML page. Synthetic bundles only."""

from __future__ import annotations

import json

import pytest

from recreplay import corpus_summary as cs


def _bundle(name="sbcsae_X", dur=600.0):
    segs = [
        {"start": 0.0, "end": 10.0, "speaker": "S1", "text": "a", "is_backchannel": False},
        {"start": 8.0, "end": 20.0, "speaker": "S2", "text": "b", "is_backchannel": False},
        {"start": 30.0, "end": 40.0, "speaker": "S1", "text": "c", "is_backchannel": False},
    ]
    lines = [
        {"source": "server", "kind": "nudge", "at_s": 12.0, "fires": True, "text": "Let them finish.", "latency_ms": 1000.0},
        {"source": "phone", "kind": "haptic", "at_s": 31.0, "fires": True, "text": "buzz L2", "latency_ms": 500.0},
        {"source": "server", "kind": "response", "at_s": 300.0, "fires": True, "text": "Ask Bob.", "latency_ms": 3000.0},
        {"source": "server", "kind": "response", "at_s": 301.0, "fires": False, "text": "quiet", "latency_ms": 2000.0},
    ]
    return {
        "name": name, "audio": {"duration_s": dur},
        "identity": {"wearer_ann_id": "S1"},
        "primary_annotation": "groundtruth",
        "annotations": [{"label": "groundtruth", "aligned": {"segments": segs, "events": [{"t": 30.0, "type": "laughter", "speakers": ["S2"]}],
                                                              "coach_moments": []}}],
        "phone": {"sent": [{"start_time": 0.0, "end_time": 19.0, "text": "a b", "is_self": True, "sent_at_audio_s": 19.5},
                           {"start_time": 30.0, "end_time": 40.0, "text": "c", "is_self": False, "sent_at_audio_s": 40.5}]},
        "score": {
            "lines": lines,
            "moments": {"hits": 1, "total": 2, "fires": 3,
                        "items": [{"t": 10.0, "hit": True, "text": "Let them finish first.", "priority": 2, "what_happened": "x"},
                                  {"t": 200.0, "hit": False, "text": "Lower your voice, slow down.", "priority": 1, "what_happened": "y"}],
                        "unmatched_lines": [lines[1], lines[2]]},
            "identity": {"phone": {"accuracy": 0.5, "wearer_recall": 1.0, "false_self": 0, "first_confirmed_s": 19.5,
                                   "decided": 2, "correct": 1, "wearer_turns": 1, "turns": 2},
                         "server": {"first_confirmed_s": 25.0}},
            "latency": {"p50_ms": 2000.0, "p90_ms": 2800.0},
            "violations": [{"kind": "invented-fact", "at_s": 300.0, "text": "Ask Bob.", "evidence": "Bob"}],
            "errors": [],
        },
    }


def test_item_metrics():
    m = cs.item_metrics(_bundle(), {"group": "heated", "corpus": "SBCSAE", "heat_basis": "argument"})
    assert m["moments_hits"] == 1 and m["moments_total"] == 2
    assert m["fires"] == 3 and m["false_fires"] == 2
    assert m["false_fires_per_h"] == pytest.approx(12.0)
    assert m["identity_accuracy"] == 0.5 and m["time_to_confirm_s"] == 19.5
    assert m["latencies_ms"] == [1000.0, 3000.0, 2000.0]
    assert m["violations"] == {"invented-fact": 1}
    # the phone's first turn spans S1 and S2 (both >= 0.5 s): a merged turn
    assert m["merged_turns"] == 1 and m["phone_turns"] == 2
    # the buzz at 31 s lands right after laughter at 30 s, with no moment nearby
    assert m["fires_near_laughter"] == 1
    assert m["missed_by_rule"] == {"raised-voice": 1} and m["hit_by_rule"] == {"talks-over": 1}


def test_raised_interject_slider_gates_server_lines_on_importance():
    b = _bundle()
    b["score"]["lines"][0]["importance"] = 80      # nudge at 12 s, near the 10 s moment
    b["score"]["lines"][2]["importance"] = 40      # the 300 s line, no moment
    m = cs.item_metrics(b, {"group": "heated", "corpus": "SBCSAE"})
    g50 = m["gated"][50]
    assert g50["fires"] == 2 and g50["false_fires"] == 1 and g50["hits"] == 1   # nudge + the phone buzz
    assert m["importances"] == [80.0, 40.0]
    agg = cs.aggregate([m])[("heated", "ALL")]
    assert agg["gated"][50]["moment_hit_rate"] == 0.5


def test_identity_is_not_scored_without_speaker_truth():
    m = cs.item_metrics(_bundle("confer_Y"), {"group": "heated", "corpus": "CONFER"})
    assert m["identity_accuracy"] is None and m["time_to_confirm_s"] is None


def test_aggregate_keeps_groups_apart():
    a = cs.item_metrics(_bundle("a"), {"group": "heated", "corpus": "SBCSAE"})
    b = cs.item_metrics(_bundle("b", dur=1200.0), {"group": "calm", "corpus": "SBCSAE"})
    agg = cs.aggregate([a, b])
    assert set(agg) >= {("heated", "SBCSAE"), ("heated", "ALL"), ("calm", "SBCSAE"), ("calm", "ALL")}
    h = agg[("heated", "ALL")]
    assert h["items"] == 1 and h["hours"] == pytest.approx(600 / 3600)
    assert h["moment_hit_rate"] == 0.5
    assert agg[("calm", "ALL")]["false_fires_per_h"] == pytest.approx(6.0)
    assert h["latency_p50_ms"] == 2000.0


def test_render_html_has_both_themes_and_links(tmp_path):
    items = [cs.item_metrics(_bundle("a"), {"group": "heated", "corpus": "SBCSAE"}),
             cs.item_metrics(_bundle("b"), {"group": "calm", "corpus": "AMI"})]
    page = cs.render(items, cs.aggregate(items), cs.defects(items))
    assert "prefers-color-scheme: dark" in page and 'name="viewport"' in page
    assert 'href="a.html"' in page and "Heated" in page and "Calm control" in page
    assert "Ask Bob." in page            # the worst examples are shown
