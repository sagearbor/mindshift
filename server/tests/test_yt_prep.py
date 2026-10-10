"""yt_* open-web intake prep: wearer choice, the held-out enrollment TAIL
(the clip's end is enrolled from and not replayed), and truncating the
whisper reference to the replayed window. Pure functions; no model."""

from __future__ import annotations

from recreplay import yt_prep


def _w(word, a, b, spk):
    return {"word": word, "start": a, "end": b, "speaker": spk}


def test_pick_wearer_named_annotation_id_maps_to_label():
    label, how = yt_prep.pick_wearer([_w("a", 0, 1, 0), _w("b", 1, 9, 1)], explicit_id="S2",
                                     labels={"S1": 0, "S2": 1})
    assert label == "Speaker B" and how.startswith("notes")


def test_pick_wearer_most_talk_when_no_annotation():
    words = [_w("a", 0, 1, 0), _w("b", 2, 9, 1), _w("c", 10, 11, 0)]
    label, how = yt_prep.pick_wearer(words, explicit_id=None, labels=None)
    assert label == "Speaker B" and "most" in how


def test_enroll_tail_takes_wearer_solo_speech_from_the_end():
    turns = []
    t = 0.0
    while t < 300:
        turns.append({"speaker": "Speaker A", "start_time": t, "end_time": t + 8})
        turns.append({"speaker": "Speaker B", "start_time": t + 8, "end_time": t + 12})
        t += 12
    plan = yt_prep.choose_enroll_tail(turns, "Speaker A", 300.0, target_s=30, guard_s=10)
    assert plan is not None
    assert sum(b - a for a, b in plan["enroll"]) >= 30
    assert all(a >= plan["enroll_from_s"] for a, _ in plan["enroll"])
    assert plan["end_s"] == plan["enroll_from_s"] - 10
    assert plan["end_s"] > 0.6 * 300


def test_enroll_tail_excludes_overlap_with_others():
    turns = [{"speaker": "Speaker A", "start_time": 200, "end_time": 300},
             {"speaker": "Speaker B", "start_time": 250, "end_time": 260}]
    plan = yt_prep.choose_enroll_tail(turns, "Speaker A", 300.0, target_s=30, guard_s=10)
    for a, b in plan["enroll"]:
        assert b <= 250 or a >= 260


def test_enroll_tail_refuses_when_too_little_wearer_speech_late():
    turns = [{"speaker": "Speaker A", "start_time": 0, "end_time": 100},
             {"speaker": "Speaker B", "start_time": 100, "end_time": 300}]
    assert yt_prep.choose_enroll_tail(turns, "Speaker A", 300.0, target_s=30, guard_s=10) is None


def _openweb_item(tmp_path):
    import json
    import wave

    import numpy as np
    d = tmp_path / "inbox" / "yt_q"
    d.mkdir(parents=True)
    with wave.open(str(d / "yt_q.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.ones(16000 * 10, dtype="<i2").tobytes())
    (d / "yt_q.notes.txt").write_text("who: unknown\nsetting: x\nphone: none\nsource: open web, licence CC BY\n")
    (d / "yt_q.replay_window.json").write_text(json.dumps({"end_s": 6.0}))
    (d / "yt_q.wearer.json").write_text(json.dumps({"label": "Speaker B"}))
    return d


def test_open_web_item_defaults_to_whisper_never_owner_print(tmp_path, monkeypatch):
    from recreplay import pipeline
    prof = tmp_path / "owner_profile.json"
    prof.write_text("{}")
    monkeypatch.setattr(pipeline, "DEFAULT_PROFILE", prof)
    inp = pipeline.inputs_from_inbox(_openweb_item(tmp_path), work_root=tmp_path / "work")
    assert inp.stt_engine == "whisper" and inp.profile_path is None
    assert inp.replay_end_s == 6.0 and inp.wearer_label == "Speaker B"


def test_trim_wav_and_clip_annotation(tmp_path):
    import shutil

    from recreplay import annotation as ann
    from recreplay import audio as audio_mod
    from recreplay import pipeline
    d = _openweb_item(tmp_path)
    wav = tmp_path / "a.wav"
    shutil.copyfile(d / "yt_q.wav", wav)
    info = pipeline.trim_wav(wav, 6.0, {"duration_s": 10.0})
    assert info["duration_s"] == 6.0 and len(audio_mod.read_wav16(wav)) == 6 * 16000
    a = ann.Annotation(ok=True, segments=[ann.Segment(1, 2, "S1", "a"), ann.Segment(7, 8, "S2", "b")],
                       coach_moments=[ann.CoachMoment(t=7.5, for_speaker="S1")])
    pipeline.clip_annotation(a, 6.0)
    assert [s.speaker for s in a.segments] == ["S1"] and a.coach_moments == []


def test_truncate_raw_and_doc():
    raw = {"metadata": {}, "results": {"channels": [{"alternatives": [{"words": [
        {"word": "a", "start": 1, "end": 2}, {"word": "b", "start": 99, "end": 101}]}]}]}}
    out = yt_prep.truncate_raw(raw, 100.0)
    assert [w["word"] for w in out["results"]["channels"][0]["alternatives"][0]["words"]] == ["a"]
    assert out["metadata"]["replay_end_s"] == 100.0
    doc = {"segments": [{"start": 1, "end": 2, "words": [{"word": "a", "start": 1, "end": 2}]},
                        {"start": 98, "end": 102, "words": [{"word": "x", "start": 98, "end": 99},
                                                           {"word": "b", "start": 99, "end": 101}]}]}
    d2 = yt_prep.truncate_doc(doc, 100.0)
    assert [w["word"] for s in d2["segments"] for w in s["words"]] == ["a", "x"]
