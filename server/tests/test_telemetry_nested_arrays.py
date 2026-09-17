"""Firestore refuses arrays nested directly in arrays — the phone's
device_diarization event (segments: [[start, end, label], ...]) 500'd on
2026-08-30. clamp_event now rewrites such payloads at the persistence boundary."""
from watch.telemetry_store import firestore_safe


def test_segment_triples_become_objects():
    data = {"device_diarization": {"segments": [[0.0, 1.5, "Speaker A"], [1.5, 3.0, 1]], "k": 2}}
    safe = firestore_safe(data)
    assert safe["device_diarization"]["segments"] == [
        {"start": 0.0, "end": 1.5, "label": "Speaker A"},
        {"start": 1.5, "end": 3.0, "label": 1},
    ]
    assert safe["device_diarization"]["k"] == 2


def test_other_nested_lists_become_items_and_scalars_pass_through():
    safe = firestore_safe({"m": [[1, 2], [3, 4, 5, 6]], "eig": [0.1, 0.2], "n": None, "s": "x"})
    assert safe["m"] == [{"items": [1, 2]}, {"items": [3, 4, 5, 6]}]
    assert safe["eig"] == [0.1, 0.2] and safe["n"] is None and safe["s"] == "x"


def test_no_list_directly_inside_list_survives_anywhere():
    def has_nested(v):
        if isinstance(v, list):
            return any(isinstance(x, list) or has_nested(x) for x in v)
        if isinstance(v, dict):
            return any(has_nested(x) for x in v.values())
        return False
    deep = {"a": [[[1, 2]], {"b": [[3, "x", "y"]]}]}
    assert not has_nested(firestore_safe(deep))
