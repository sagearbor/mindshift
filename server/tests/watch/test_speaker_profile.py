# Ported from gauge@2157433 server/tests/test_speaker_profile.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B5, speaker_id v2): server.engine.speaker_id -> THIS repo's
# flat speaker_id module (server/speaker_id.py, v2 -- see
# server/watch/models.py's SpeakerProfile docstring); server.models ->
# watch.models; server.store.MemoryEpisodeStore -> watch.store.
# MemoryLiveSessionStore. None of this file's assertions reference the v1
# `sources`/v2 `samples` field split (they check dim/enroll_count/embedding/
# created_at/updated_at only), so every test here ports VERBATIM in intent --
# only the imports move.
import asyncio

import numpy as np

from speaker_id import new_profile
from watch.models import SpeakerProfile
from watch.store import MemoryLiveSessionStore


def _vec(seed: int, dim: int = 192) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_engine_new_profile_constructs_the_model_directly():
    # The model's field names must line up with the vendored engine's output
    # dict, so enrollment needs no translation layer between them.
    doc = new_profile(_vec(1), None, recording_id="enroll-1", speaker="self",
                      now_iso="2026-08-02T00:00:00+00:00")
    p = SpeakerProfile(account_id="uid-1", **doc)
    assert p.dim == 192 and p.enroll_count == 1 and len(p.embedding) == 192
    assert p.created_at == p.updated_at == "2026-08-02T00:00:00+00:00"


def test_profile_roundtrip():
    s = MemoryLiveSessionStore()
    doc = new_profile(_vec(2), None, recording_id="r", speaker="self", now_iso="t")
    p = SpeakerProfile(account_id="uid-1", **doc)
    asyncio.run(s.put_speaker_profile(p))
    assert asyncio.run(s.get_speaker_profile("uid-1")) == p
    assert asyncio.run(s.get_speaker_profile("nope")) is None


def test_second_enrollment_refines_via_running_mean():
    s = MemoryLiveSessionStore()
    first = SpeakerProfile(account_id="uid-1",
                           **new_profile(_vec(3), None, recording_id="r1", speaker="self", now_iso="t1"))
    asyncio.run(s.put_speaker_profile(first))
    existing = asyncio.run(s.get_speaker_profile("uid-1")).model_dump()
    second = SpeakerProfile(account_id="uid-1",
                            **new_profile(_vec(4), existing, recording_id="r2", speaker="self", now_iso="t2"))
    asyncio.run(s.put_speaker_profile(second))
    got = asyncio.run(s.get_speaker_profile("uid-1"))
    assert got.enroll_count == 2
    assert got.created_at == "t1" and got.updated_at == "t2"
    assert abs(float(np.linalg.norm(np.asarray(got.embedding))) - 1.0) < 1e-5   # stays L2-normalized


def test_profile_store_copies_out():
    s = MemoryLiveSessionStore()
    p = SpeakerProfile(account_id="uid-1",
                       **new_profile(_vec(5), None, recording_id="r", speaker="self", now_iso="t"))
    asyncio.run(s.put_speaker_profile(p))
    got = asyncio.run(s.get_speaker_profile("uid-1"))
    got.enroll_count = 99
    assert asyncio.run(s.get_speaker_profile("uid-1")).enroll_count == 1
