"""listen_sweep metrics: merged phone turns and fragmentation of
single-voice ground-truth segments."""

from recreplay import listen_sweep as L


def seg(a, b, lab, bc=False):
    return {"start": a, "end": b, "label": lab, "is_backchannel": bc}


def turn(a, b):
    return {"start_time": a, "end_time": b}


def test_merged_turn_pct_needs_two_voices_half_a_second_each():
    segs = [seg(0, 3, "S1"), seg(3, 6, "S2"), seg(6, 6.3, "S1"), seg(10, 12, "S3", bc=True)]
    sent = [turn(0, 6), turn(5.8, 6.4), turn(9, 12)]
    # turn 1 holds S1 3s + S2 3s -> merged; turn 2: S2 0.2 + S1 0.3 -> not; turn 3: only a backchannel
    assert L.merged_turn_pct(sent, segs) == 100.0 / 3


def test_frag_pct_counts_long_segments_split_across_turns():
    segs = [seg(0, 4, "S1"), seg(5, 6, "S2"), seg(10, 13, "S1")]
    sent = [turn(0, 2), turn(2.1, 4), turn(10, 13)]
    assert L.frag_pct(sent, segs) == 50.0      # seg 0-4 split; 10-13 whole; 5-6 too short to count


def test_time_identity_weighs_by_speech_time():
    segs = [dict(seg(0, 4, "S1"), wearer=True), dict(seg(4, 6, "S2"), wearer=False), dict(seg(8, 10, "S1"), wearer=True)]
    sent = [dict(turn(0, 6), is_self=True), dict(turn(8, 10), is_self=None)]
    m = L.time_identity(sent, segs)
    assert m["purity"] == (4 + 2) / 8          # turn 1: S1 4 of 6; turn 2 pure
    assert m["self_time_recall"] == 4 / 6      # wearer 6 s, 4 s inside a self-called turn
    assert m["self_time_precision"] == 4 / 6   # self-called turn: 4 of 6 s are the wearer


def test_none_without_truth():
    assert L.merged_turn_pct([turn(0, 1)], []) is None
    assert L.frag_pct([], []) is None
