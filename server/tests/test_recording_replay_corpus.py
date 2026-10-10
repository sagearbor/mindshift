"""Recording-replay pipeline hooks for corpus items (scripts/corpus_to_inbox.py):
human ground-truth timing is kept (``retime: false``), a heat-only
annotation (CONFER: moments, no speaker turns) still counts, a per-item
held-out voiceprint (``<name>.voiceprint.json``) is picked up and frozen
into the fixture, and the summary aggregates per group. Pure, always runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recreplay import annotation as ann
from recreplay import fixture, pipeline

WORDS = [
    {"word": "you", "start": 1.3, "end": 1.5, "speaker": 0},
    {"word": "never", "start": 1.5, "end": 1.9, "speaker": 0},
    {"word": "listen", "start": 1.9, "end": 2.4, "speaker": 0},
]
GT = {
    "format": "mindshift-annotation/v1", "retime": False,
    "speakers": [{"id": "S1"}],
    "segments": [{"start": 1.0, "end": 2.6, "speaker": "S1", "text": "You never listen."}],
    "coach_moments": [{"t": 1.0, "for_speaker": "S1", "kind": "warning", "ideal_nudge": "x"}],
}


def test_ground_truth_timing_is_not_retimed():
    a = ann.parse_annotation_text(json.dumps(GT))
    al = ann.align(a, WORDS)
    s = al.segments[0]
    assert (s.start, s.end) == (1.0, 2.6) and s.aligned is True
    assert al.quality["words_matched_pct"] == 100.0
    assert al.quality.get("retimed") is False
    assert al.speaker_map == {"S1": "Speaker A"}
    assert al.coach_moments[0].t == 1.0


def test_llm_annotation_is_still_retimed():
    obj = dict(GT)
    obj.pop("retime")
    al = ann.align(ann.parse_annotation_text(json.dumps(obj)), WORDS)
    assert al.segments[0].start == 1.3


def test_heat_only_annotation_is_usable():
    obj = {"format": "mindshift-annotation/v1", "retime": False, "speakers": [], "segments": [],
           "coach_moments": [{"t": 10.0, "for_speaker": None, "kind": "warning", "ideal_nudge": "slow down"}]}
    a = ann.parse_annotation_text(json.dumps(obj))
    assert a.ok
    al = ann.align(a, WORDS)
    assert [m.t for m in al.coach_moments] == [10.0]


def _item(folder: Path, name: str, with_vp: bool = True) -> Path:
    d = folder / name
    d.mkdir(parents=True)
    (d / f"{name}.wav").write_bytes(b"RIFF")
    (d / f"{name}.notes.txt").write_text("# source: corpus ground truth\nwho: I'm S1\nsetting: x\nphone: y\n")
    (d / f"{name}.annotation.groundtruth.json").write_text(json.dumps(GT))
    if with_vp:
        (d / f"{name}.voiceprint.json").write_text(json.dumps({"embedding": [0.1] * 192}))
    return d


def test_inbox_item_voiceprint_overrides_the_owner_profile(tmp_path):
    d = _item(tmp_path, "sbcsae_SBC033")
    inp = pipeline.inputs_from_inbox(d, work_root=tmp_path / "work", profile=tmp_path / "owner.json")
    assert inp.profile_path == d / "sbcsae_SBC033.voiceprint.json"
    assert [f.label for f in inp.annotation_files] == ["groundtruth"]
    d2 = _item(tmp_path, "confer_x", with_vp=False)
    inp2 = pipeline.inputs_from_inbox(d2, work_root=tmp_path / "work", profile=None)
    assert inp2.profile_path is None          # never the owner's print on a corpus item


def test_freeze_carries_the_item_voiceprint(tmp_path, monkeypatch):
    d = _item(tmp_path, "ami_ES2002a")
    inp = pipeline.inputs_from_inbox(d, work_root=tmp_path / "work")
    inp.work.mkdir(parents=True)
    (inp.work / "audio16k.wav").write_bytes(b"RIFF")
    bundle = {"name": inp.name, "settings": {"mode": "earpiece", "enroll": "profile"},
              "voiceprint_profile": str(inp.profile_path), "identity": {"wearer_label": "Speaker A"},
              "annotations": [], "score": {}}
    fx = fixture.freeze(bundle, inp, dest=tmp_path / "fixtures" / inp.name)
    assert (fx / "voiceprint.json").exists()
    inp2, base = fixture.inputs_from_fixture(fx, tmp_path / "rerun")
    assert inp2.profile_path is not None and inp2.profile_path.exists()
    assert json.loads(inp2.profile_path.read_text())["embedding"][0] == 0.1
    assert fixture.is_corpus_fixture(fx)


def test_stt_language_comes_from_the_groundtruth_annotation(tmp_path, monkeypatch):
    from recreplay import stt
    d = _item(tmp_path, "confer_y", with_vp=False)
    obj = {**GT, "audio": {"language": "el"}}
    (d / "confer_y.annotation.groundtruth.json").write_text(json.dumps(obj))
    inp = pipeline.inputs_from_inbox(d, work_root=tmp_path / "work")
    assert inp.stt_language == "el"
    seen = {}

    def fake_post(wav, key, params):
        seen.update(params)
        return {"results": {}}

    monkeypatch.setattr(stt, "_post_deepgram", fake_post)
    monkeypatch.setattr(stt, "deepgram_key", lambda: "k")
    stt.transcribe(b"RIFF", tmp_path / "dg.json", language="el")
    assert seen["language"] == "el" and seen["model"] == "nova-2"
    assert pipeline.inputs_from_inbox(_item(tmp_path, "plain"), work_root=tmp_path / "w").stt_language is None
