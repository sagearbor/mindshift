"""Recording-replay: baseline comparison rules, and pulling the owner's
latest app recording from GCS (against an in-memory bucket double)."""

from __future__ import annotations

import json

import pytest

from recreplay import app_pull, fixture

BASE = {"metrics": {
    "identity_phone_accuracy": 0.6, "identity_wearer_recall": 0.5, "phone_self_correct": 3, "identity_false_self": 1,
    "moments_hits": 5, "violations": 0, "errors": 0, "phone_first_confirmed_s": 18.6,
    "latency_p50_ms": 1900.0, "latency_p90_ms": 2400.0, "server_lines": 7, "llm_offline_misses": 0,
}}


def test_identical_run_has_no_regressions():
    reg, notes = fixture.compare(BASE, dict(BASE["metrics"]))
    assert reg == [] and notes == []


@pytest.mark.parametrize("key,value", [
    ("identity_phone_accuracy", 0.4), ("identity_wearer_recall", 0.25), ("phone_self_correct", 2),
    ("identity_false_self", 2), ("moments_hits", 4), ("violations", 1), ("errors", 1),
    ("phone_first_confirmed_s", 25.0), ("latency_p90_ms", 4500.0),
])
def test_each_regression_is_caught(key, value):
    cur = dict(BASE["metrics"], **{key: value})
    reg, _ = fixture.compare(BASE, cur)
    assert any(r.startswith(key) for r in reg), reg


def test_latency_has_headroom_and_misses_are_explained():
    cur = dict(BASE["metrics"], latency_p90_ms=3500.0, llm_offline_misses=2, server_lines=5)
    reg, notes = fixture.compare(BASE, cur)
    assert reg == []
    assert any("offline cache miss" in n for n in notes) and any("server_lines" in n for n in notes)


def test_metrics_from_a_bundle():
    b = {"score": {"identity": {"phone": {"turns": 7, "accuracy": 0.6, "wearer_recall": 0.5, "false_self": 1,
                                          "first_confirmed_s": 18.6}},
                   "latency": {"p50_ms": 1900, "p90_ms": 2400, "n": 7, "llm_cache": {"offline_misses": 0}},
                   "moments": {"hits": 5, "total": 7, "fires": 9}, "violations": [], "errors": []},
         "phone": {"attribution": {"selfCorrect": 3}}}
    m = fixture.metrics(b)
    assert m["phone_self_correct"] == 3 and m["moments_total"] == 7 and m["fires"] == 9 and m["errors"] == 0


# ---------------------------------------------------------------------------
# app pull
# ---------------------------------------------------------------------------

class _Blob:
    def __init__(self, name, data):
        self.name, self.data = name, data

    def download_as_bytes(self):
        return self.data

    def download_to_filename(self, path):
        with open(path, "wb") as f:
            f.write(self.data)


class _Bucket:
    def __init__(self, blobs):
        self.blobs = blobs

    def list_blobs(self, prefix):
        return [b for b in self.blobs if b.name.startswith(prefix)]


class _Client:
    def __init__(self, bucket):
        self._b = bucket
        self.asked = None

    def bucket(self, name):
        self.asked = name
        return self._b


def _meta(rid, created, title):
    return json.dumps({"id": rid, "created_at": created, "title": title}).encode()


def test_pull_latest_with_audio(tmp_path):
    uid = "u1"
    blobs = [
        _Blob(f"recordings/{uid}/new/meta.json", _meta("new", "2026-10-08T20:00:00+00:00", "no audio kept")),
        _Blob(f"recordings/{uid}/new/turns.json", b"[]"),
        _Blob(f"recordings/{uid}/old/meta.json", _meta("old", "2026-10-07T19:30:00+00:00", "Dinner")),
        _Blob(f"recordings/{uid}/old/audio.m4a", b"AUDIO"),
        _Blob(f"recordings/{uid}/old/turns.json", b"[{\"speaker\": \"Speaker A\"}]"),
        _Blob("recordings/someone-else/x/meta.json", _meta("x", "2026-10-09T00:00:00+00:00", "not mine")),
        _Blob("recordings/someone-else/x/audio.m4a", b"NOPE"),
    ]
    client = _Client(_Bucket(blobs))
    dest = app_pull.pull_latest(tmp_path, uid=uid, client=client)
    assert dest.name == "app-20261007-1930-old"
    assert (dest / f"{dest.name}.m4a").read_bytes() == b"AUDIO"
    meta = json.loads((dest / f"{dest.name}.app_meta.json").read_text())
    assert meta["recording_id"] == "old" and meta["uid"] == uid
    notes = (dest / f"{dest.name}.notes.txt").read_text()
    assert notes.startswith("who:") and "setting: Dinner" in notes
    # an owner-edited notes file is never overwritten
    (dest / f"{dest.name}.notes.txt").write_text("who: I'm S1\n")
    app_pull.pull_latest(tmp_path, uid=uid, client=client)
    assert (dest / f"{dest.name}.notes.txt").read_text() == "who: I'm S1\n"


def test_pull_without_audio_or_identity_is_an_honest_error(tmp_path):
    client = _Client(_Bucket([_Blob("recordings/u1/a/meta.json", _meta("a", "2026-10-08T20:00:00Z", "t"))]))
    with pytest.raises(app_pull.AppPullError, match="stored audio"):
        app_pull.pull_latest(tmp_path, uid="u1", client=client)
    with pytest.raises(app_pull.AppPullError, match="--email or --uid"):
        app_pull.pull_latest(tmp_path, client=client)
