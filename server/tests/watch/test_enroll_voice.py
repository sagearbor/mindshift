# Ported from gauge@2157433 server/tests/test_enroll_voice.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B5, speaker_id v2): server.main.create_app ->
# watch.testing.create_watch_test_app; GAUGE_ALLOW_LEGACY_ACCOUNT env var ->
# the explicit allow_legacy=True kwarg.
#
# CRITICAL adaptation (see task-B5-report.md for the full disposition table):
# gauge's SpeakerProfile.sources field name tracked its v1-shaped
# server/engine/speaker_id.new_profile() dict 1:1. THIS repo's speaker_id
# module (server/speaker_id.py) is v2 -- new_profile() keeps each enrollment
# as an individual, independently-deletable `samples` entry instead of a
# v1-only running-mean blend (see server/watch/models.py's SpeakerProfile
# docstring for the rename). ONE test below --
# test_second_enroll_refines_the_same_profile -- asserted `len(p.sources)`;
# it is REWRITTEN to `len(p.samples)`, same intent (two enrollments produced
# two provenance entries), same behavior, only the v2 field name changes.
# Every other test in this file ports verbatim (the enroll_count/dim/
# available/enrolled surface gauge's tests actually assert on is identical
# between v1 and v2).
import asyncio

import numpy as np
import speaker_id
from fastapi.testclient import TestClient

from server.tests.watch.test_vectors import pcm
from watch.store import MemoryLiveSessionStore
from watch.testing import create_watch_test_app

ACC = {"account": "alice"}


def _fixed_embedder(seed: int):
    """Deterministic stand-in for speaker_id.embed_pcm — no torch, no model."""
    def embed(audio: np.ndarray, sr: int) -> np.ndarray:
        assert audio.dtype == np.float32 and sr == 16000
        rng = np.random.default_rng(seed)
        v = rng.normal(size=192).astype(np.float32)
        return v / np.linalg.norm(v)
    return embed


def _client(embedder=None):
    store = MemoryLiveSessionStore()
    return store, TestClient(create_watch_test_app(
        store=store, embedder=embedder, allow_legacy=True,
    ))


def test_enroll_response_shape_is_unchanged(_capsys=None):
    # Regression guard for the six existing /enroll tests: the voiceprint is a
    # side effect, never a change to this response body.
    _, client = _client(_fixed_embedder(1))
    body = client.post("/enroll", params=ACC, content=pcm(0.2, seconds=3.0)).json()
    assert set(body) == {"account_id", "rms_db", "f0_median", "updated_at"}


def test_enroll_stores_speaker_profile_when_embedder_available():
    store, client = _client(_fixed_embedder(1))
    assert client.post("/enroll", params=ACC, content=pcm(0.2, seconds=3.0)).status_code == 200
    p = asyncio.run(store.get_speaker_profile("alice"))
    assert p is not None and p.enroll_count == 1 and p.dim == 192


def test_second_enroll_refines_the_same_profile():
    store, client = _client(_fixed_embedder(1))
    client.post("/enroll", params=ACC, content=pcm(0.2, seconds=3.0))
    client.post("/enroll", params=ACC, content=pcm(0.25, seconds=4.0))
    p = asyncio.run(store.get_speaker_profile("alice"))
    # REWRITTEN (v1 -> v2): gauge asserted `len(p.sources) == 2`. v2's
    # per-enrollment provenance list is `samples` (see module docstring) --
    # same intent (two enrollments -> two provenance entries), same
    # behavior, only the field name moved.
    assert p.enroll_count == 2 and len(p.samples) == 2


def test_enroll_without_embedder_still_200_and_no_profile(monkeypatch):
    # `_client(None)` means "nothing injected" — the router then falls back to
    # the REAL speaker_id.embed_pcm whenever torch/speechbrain import, so this
    # test only exercised the no-embedder path on machines WITHOUT the voice
    # deps (it failed in venv-voice). Pin the unavailable branch explicitly.
    monkeypatch.setattr(speaker_id, "is_available", lambda: False)
    store, client = _client(None)          # honest degradation, not a 500
    assert client.post("/enroll", params=ACC, content=pcm(0.2, seconds=3.0)).status_code == 200
    assert asyncio.run(store.get_baseline("alice")) is not None
    assert asyncio.run(store.get_speaker_profile("alice")) is None


def test_throwing_embedder_never_fails_enrollment():
    def boom(audio, sr):
        raise RuntimeError("model exploded")
    store, client = _client(boom)
    assert client.post("/enroll", params=ACC, content=pcm(0.2, seconds=3.0)).status_code == 200
    assert asyncio.run(store.get_baseline("alice")) is not None
    assert asyncio.run(store.get_speaker_profile("alice")) is None


def test_too_short_clip_still_422_and_no_profile():
    store, client = _client(_fixed_embedder(1))
    assert client.post("/enroll", params=ACC, content=pcm(0.2, seconds=1.0)).status_code == 422
    assert asyncio.run(store.get_speaker_profile("alice")) is None


def test_voice_status_unavailable_without_embedder(monkeypatch):
    monkeypatch.setattr(speaker_id, "is_available", lambda: False)  # see above
    _, client = _client(None)
    assert client.get("/enroll/voice", params=ACC).json() == {
        "available": False, "enrolled": False, "enroll_count": 0,
        "dim": None, "model": None, "updated_at": None}


def test_voice_status_available_but_not_enrolled():
    _, client = _client(_fixed_embedder(1))
    body = client.get("/enroll/voice", params=ACC).json()
    assert body["available"] is True and body["enrolled"] is False and body["dim"] is None


def test_voice_status_after_enroll():
    _, client = _client(_fixed_embedder(1))
    client.post("/enroll", params=ACC, content=pcm(0.2, seconds=3.0))
    body = client.get("/enroll/voice", params=ACC).json()
    assert body["available"] is True and body["enrolled"] is True
    assert body["enroll_count"] == 1 and body["dim"] == 192
    assert body["model"] and body["updated_at"]
