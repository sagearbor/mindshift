"""Shared scoring helpers for the sliding-window experiment (pure math)."""
import itertools


def best_permutation_accuracy(truth, pred):
    """Copied from server/tests/test_diarize_regression_ladder.py's
    _best_permutation_accuracy (per-turn exact accuracy under the best
    label bijection). Not imported directly since that module lives under
    pytest and this is a standalone script."""
    truth_labels = sorted(set(truth))
    pred_labels = sorted(set(pred))
    best_acc, best_correct, best_map = -1.0, 0, {}
    width = min(len(truth_labels), len(pred_labels))
    for perm in itertools.permutations(truth_labels, width):
        mapping = dict(zip(pred_labels, perm))
        correct = sum(1 for t, p in zip(truth, pred) if mapping.get(p) == t)
        acc = correct / len(truth)
        if acc > best_acc:
            best_acc, best_correct, best_map = acc, correct, mapping
    return best_acc, best_correct, best_map


def majority_overlap_label(seg_start, seg_end, ref_turns, speaker_key="speaker"):
    """The reference speaker with the most time-overlap with [seg_start,
    seg_end]. ``ref_turns`` items need start/end keys (approx_start/
    approx_end or start_time/end_time, tried in that order)."""
    best_speaker, best_overlap = None, 0.0
    for t in ref_turns:
        s = t.get("approx_start", t.get("start_time"))
        e = t.get("approx_end", t.get("end_time"))
        overlap = max(0.0, min(seg_end, e) - max(seg_start, s))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = t[speaker_key]
    return best_speaker


def score_against_reference(pred_turns, ref_turns, speaker_key="speaker"):
    """Build (truth, pred) arrays — ONE ENTRY PER PREDICTED TURN — by
    labeling each predicted turn with its majority-time-overlap reference
    speaker, then score with best_permutation_accuracy. This is the honest
    adaptation needed when the predicted turn COUNT/boundaries don't match
    the reference's turn count (our sliding-window turns don't align 1:1
    with the fixture's approx_turns / transcript turns)."""
    truth = [
        majority_overlap_label(t["start_time"], t["end_time"], ref_turns, speaker_key)
        for t in pred_turns
    ]
    pred = [t["speaker"] for t in pred_turns]
    # Duration-weighted accuracy alongside plain per-turn accuracy: a 0.3s
    # sliver turn getting the wrong label matters less than a 5s one.
    acc, correct, mapping = best_permutation_accuracy(truth, pred)
    total_dur = sum(t["end_time"] - t["start_time"] for t in pred_turns)
    correct_dur = sum(
        (t["end_time"] - t["start_time"])
        for t, tr, pr in zip(pred_turns, truth, pred)
        if mapping.get(pr) == tr
    )
    weighted_acc = correct_dur / total_dur if total_dur else 0.0
    return {
        "per_turn_accuracy": acc,
        "per_turn_correct": correct,
        "per_turn_total": len(pred_turns),
        "duration_weighted_accuracy": weighted_acc,
        "mapping": mapping,
        "truth": truth,
        "pred": pred,
    }
