# Ported from gauge@2157433 server/tests/test_diarize.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
import numpy as np

from watch.diarize import (
    EmbeddingDiarizationService,
    NullDiarizationService,
    assign_speakers,
    diarize,
    speech_segments,
)


def tone(amp: float, seconds: float, sr: int = 16000, freq: float = 150.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (np.sin(2 * np.pi * freq * t) * amp * 32767).astype(np.int16)


def silence(seconds: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(sr * seconds), dtype=np.int16)


def unit(seed: int, dim: int = 192) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_speech_segments_finds_two_bursts():
    pcm = np.concatenate([tone(0.2, 2.0), silence(1.0), tone(0.2, 2.0)])
    segs = speech_segments(pcm, 16000)
    assert len(segs) == 2
    assert abs(segs[0][0] - 0.0) < 0.3 and abs(segs[0][1] - 2.0) < 0.3
    assert abs(segs[1][0] - 3.0) < 0.3 and abs(segs[1][1] - 5.0) < 0.3


def test_speech_segments_merges_a_short_gap():
    pcm = np.concatenate([tone(0.2, 1.0), silence(0.25), tone(0.2, 1.0)])
    assert len(speech_segments(pcm, 16000)) == 1


def test_speech_segments_drops_too_short_bursts():
    pcm = np.concatenate([tone(0.2, 0.25), silence(1.0), tone(0.2, 2.0)])
    segs = speech_segments(pcm, 16000)
    assert len(segs) == 1 and segs[0][0] > 1.0


def test_speech_segments_pure_silence_is_empty():
    assert speech_segments(silence(3.0), 16000) == []


def test_speech_segments_includes_trailing_sub_frame_speech():
    """Task-7-deferred minor: a trailing partial frame (< frame_seconds, e.g.
    a clip whose length isn't an exact multiple of 0.25s) used to be dropped
    outright by ``n_frames = pcm.size // frame_samples`` — never even
    evaluated for speech — silently truncating the last <250ms of every
    real (non-frame-aligned) recording."""
    sr = 16000
    pcm = np.concatenate([tone(0.2, 1.0, sr=sr), tone(0.2, 0.125, sr=sr)])
    segs = speech_segments(pcm, sr)
    assert len(segs) == 1
    total_seconds = pcm.size / sr
    assert abs(segs[0][1] - total_seconds) < 0.05


def test_assign_speakers_labels_the_matching_voice_self():
    me = unit(1)
    labels = assign_speakers([me, unit(2), me], self_print=me)
    assert labels == ["self", "other-1", "self"]


def test_assign_speakers_clusters_two_distinct_others():
    labels = assign_speakers([unit(2), unit(3), unit(2)], self_print=unit(1))
    assert labels == ["other-1", "other-2", "other-1"]


def test_assign_speakers_never_says_self_without_a_voiceprint():
    labels = assign_speakers([unit(1), unit(2)], self_print=None)
    assert "self" not in labels
    assert labels == ["other-1", "other-2"]


def test_assign_speakers_passes_through_unembeddable_segments():
    labels = assign_speakers([unit(1), None, unit(2)], self_print=unit(1))
    assert labels == ["self", None, "other-1"]


def test_diarize_returns_vector_engine_turn_tuples():
    me = unit(1)
    them = unit(2)
    pcm = np.concatenate([tone(0.2, 2.0), silence(1.0), tone(0.05, 2.0)])
    # loud burst -> me, quiet burst -> them (a stand-in for a real embedder)
    def embed(audio, sr):
        return me if float(np.abs(audio).mean()) > 0.1 else them
    turns = diarize(pcm, 16000, self_print=me, embed_fn=embed)
    assert [t[0] for t in turns] == ["self", "other-1"]
    assert all(isinstance(t[1], float) and isinstance(t[2], float) and t[2] > t[1] for t in turns)
    assert turns == sorted(turns, key=lambda t: t[1])


def test_diarize_drops_a_segment_whose_embedder_raises():
    me = unit(1)
    pcm = np.concatenate([tone(0.2, 2.0), silence(1.0), tone(0.05, 2.0)])
    calls = {"n": 0}
    def embed(audio, sr):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("embed failed")
        return me
    turns = diarize(pcm, 16000, self_print=me, embed_fn=embed)
    assert [t[0] for t in turns] == ["self"]


def test_diarize_without_voiceprint_yields_no_self_turns():
    pcm = np.concatenate([tone(0.2, 2.0), silence(1.0), tone(0.05, 2.0)])
    turns = diarize(pcm, 16000, self_print=None, embed_fn=lambda a, sr: unit(int(np.abs(a).mean() * 100)))
    assert turns and all(t[0] != "self" for t in turns)


def test_embedding_diarization_service_finds_turns_in_real_pcm16_bytes():
    """Service-level seam (C1): PCM16 bytes -> EmbeddingDiarizationService.diarize
    must reach the same speech-segment detection as the pure diarize() does on
    int16-scale floats. A loud tone at full PCM16 amplitude must clear the
    silence floor and come back as a non-empty, "self"-labeled turn — it must
    NOT be silently scaled down to below the silence floor (C1's double-scaling
    bug: dividing by PCM16_FULL_SCALE here, on top of rms_dbfs's own /32768,
    made every frame read near -104 dBFS, always under the -45 dBFS floor)."""
    me = unit(1)
    loud = tone(0.9, 2.0)  # near full-scale int16 amplitude
    pcm_bytes = loud.tobytes()

    def embed(audio, sr):
        return me

    service = EmbeddingDiarizationService(embed)
    turns = service.diarize(pcm_bytes, 16000, self_print=me)

    assert turns, "a loud, full-amplitude tone must be detected as speech, not silently dropped"
    assert turns[0][0] == "self"


def test_null_diarization_service_returns_no_turns():
    # Coverage added in review round 1: NullDiarizationService had zero
    # direct test coverage in this port (it has no importer yet either —
    # wiring a diarizer selector into the pipeline is Task B11's job). A
    # loud, obviously-speech-shaped clip and a real voiceprint are passed
    # deliberately, so the empty result is provably the class's own honest
    # "no diarization configured" contract, not an artifact of empty input.
    me = unit(1)
    loud = tone(0.9, 2.0).tobytes()
    assert NullDiarizationService().diarize(loud, 16000, self_print=me) == []
