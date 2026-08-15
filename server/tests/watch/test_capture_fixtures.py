# Ported from gauge@2157433 server/tests/test_capture_fixtures.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B7): server.capture_fixtures -> watch.capture_fixtures. No
# other change -- write_fixture/load_fixture/list_fixtures are pure,
# network-free filesystem helpers with no episode/capture naming to rename
# (Capture keeps its name entirely per the locked rename map).
import json

from watch.capture_fixtures import list_fixtures, load_fixture, write_fixture

CAPTURE = {
    "id": "cap-1", "account_id": "alice", "captured_at": "2026-08-02T09:00:00Z",
    "received_at": "2026-08-02T09:01:00Z", "duration_s": 240.0, "trigger": "volume",
    "device": "pixel-watch-1", "sample_rate": 16000, "status": "stored",
    "audio_uri": "gs://b/captures/alice/cap-1.pcm", "audio_bytes": 6,
    "upload_encoding": "gzip",
    "labels": {"speakers": ["self", "other-1"],
               "events": [{"vector": "interrupting", "level": 2, "t": 41.5}]},
    "labels_updated_at": "2026-08-02T10:00:00Z", "consents": [],
}


def test_write_fixture_lays_out_the_four_files(tmp_path):
    d = write_fixture(tmp_path, CAPTURE, b"\x01\x02\x03\x04\x05\x06",
                      source_url="https://example/captures/cap-1", exported_at="2026-08-02T11:00:00Z")
    assert d == tmp_path / "cap-1"
    assert (d / "audio.pcm").read_bytes() == b"\x01\x02\x03\x04\x05\x06"
    meta = json.loads((d / "meta.json").read_text())
    assert meta["duration_s"] == 240.0 and meta["sample_rate"] == 16000
    assert "labels" not in meta                    # labels live in their own file
    assert json.loads((d / "labels.json").read_text()) == CAPTURE["labels"]
    readme = (d / "README.md").read_text()
    assert "https://example/captures/cap-1" in readme and "2026-08-02T11:00:00Z" in readme
    assert "Exporter version: 1" in readme


def test_write_fixture_without_audio_still_writes_metadata(tmp_path):
    d = write_fixture(tmp_path, CAPTURE, None, source_url="u", exported_at="t")
    assert not (d / "audio.pcm").exists()
    assert (d / "meta.json").exists() and (d / "labels.json").exists()


def test_load_fixture_roundtrip(tmp_path):
    write_fixture(tmp_path, CAPTURE, b"\x01\x02", source_url="u", exported_at="t")
    f = load_fixture(tmp_path, "cap-1")
    assert f.capture_id == "cap-1"
    assert f.meta["trigger"] == "volume"
    assert f.labels["events"][0]["vector"] == "interrupting"
    assert f.audio_available is True and f.audio() == b"\x01\x02"


def test_load_fixture_reports_missing_audio_honestly(tmp_path):
    import pytest
    write_fixture(tmp_path, CAPTURE, None, source_url="u", exported_at="t")
    f = load_fixture(tmp_path, "cap-1")
    assert f.audio_available is False
    with pytest.raises(FileNotFoundError):
        f.audio()


def test_write_fixture_is_idempotent_and_overwrites(tmp_path):
    write_fixture(tmp_path, CAPTURE, b"\x01", source_url="u", exported_at="t1")
    updated = {**CAPTURE, "labels": {"speakers": ["self"]}}
    write_fixture(tmp_path, updated, b"\x02\x03", source_url="u", exported_at="t2")
    f = load_fixture(tmp_path, "cap-1")
    assert f.audio() == b"\x02\x03" and f.labels == {"speakers": ["self"]}


def test_list_fixtures_is_sorted_and_ignores_stray_dirs(tmp_path):
    for cid in ("cap-2", "cap-1"):
        write_fixture(tmp_path, {**CAPTURE, "id": cid}, None, source_url="u", exported_at="t")
    (tmp_path / "not-a-fixture").mkdir()
    assert list_fixtures(tmp_path) == ["cap-1", "cap-2"]
    assert list_fixtures(tmp_path / "missing") == []


def test_unlabeled_capture_exports_an_empty_labels_object(tmp_path):
    write_fixture(tmp_path, {**CAPTURE, "labels": {}}, None, source_url="u", exported_at="t")
    assert load_fixture(tmp_path, "cap-1").labels == {}
