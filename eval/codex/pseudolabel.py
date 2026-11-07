"""Guarded machine-label admission; unqualified teachers fail closed."""
import math
import numpy as np
from .benchmark import LABELS, metrics, validate_training_pool


def require_teacher_quality(score):
    categories=score.get('per_emotion',{})
    value=score.get('macro_f1',float('nan'))
    if (set(categories) != set(LABELS) or not np.isfinite(value) or not 0<=value<=1
        or any(not np.isfinite(r.get('recall',float('nan'))) or not 0<=r['recall']<=1
               or r.get('support',0)<=0 for r in categories.values())):
        raise ValueError('Teacher quality scores incomplete or invalid')
    if value < .75 or min(r['recall'] for r in categories.values()) < .60:
        raise ValueError('Teacher quality gate failed')


def wilson_lower(correct, n, z=1.96):
    if not n:
        return 0.0
    p = correct / n
    return (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)


def class_gates(votes, probabilities, min_support=30, min_precision_lower=.8):
    v, p = np.asarray(votes), np.asarray(probabilities)
    metrics(v, p)
    unique = (v == v.max(1, keepdims=True)).sum(1) == 1
    truth, pred, confidence = v.argmax(1), p.argmax(1), p.max(1)
    gates = {}
    for i, name in enumerate(LABELS):
        gate = dict(threshold=None, support=0, reason='insufficient validated precision/support')
        for threshold in [.5, .6, .7, .8, .9, .95]:
            keep = unique & (pred == i) & (confidence >= threshold)
            n = int(keep.sum())
            correct = int((truth[keep] == i).sum())
            lower = wilson_lower(correct, n)
            if n >= min_support and lower >= min_precision_lower:
                gate = dict(threshold=threshold, support=n, correct=correct,
                            precision=correct/n, wilson_lower=lower)
                break
        gates[name] = gate
    return gates


def admit_rows(pool, protected, dev_votes, dev_probabilities, teacher_revision):
    """Admission only, not label generation. Gates are provisional, not new-domain validation."""
    require_teacher_quality(metrics(dev_votes, dev_probabilities))
    if not teacher_revision:
        raise ValueError('Missing teacher revision')
    validate_training_pool(pool, protected)
    gates=class_gates(dev_votes, dev_probabilities)
    admitted, rejected=[], []
    seen=set()
    for row in pool:
        if row['id'] in seen:
            raise ValueError('Duplicate pool ID')
        seen.add(row['id'])
        if row.get('model_revision') != teacher_revision:
            raise ValueError('Pool model revision differs from evaluated teacher')
        if not row.get('license_url') or not row.get('source_url'):
            raise ValueError('Missing reviewed licence/source provenance')
        prob=np.asarray(row['probabilities'],dtype=float)
        if prob.shape!=(6,) or not np.isfinite(prob).all() or (prob<0).any() or not np.isclose(prob.sum(),1):
            raise ValueError('Invalid machine probability distribution')
        i=int(prob.argmax()); threshold=gates[LABELS[i]]['threshold']
        accepted=threshold is not None and prob[i]>=threshold
        record=dict(row,training_admitted=bool(accepted),human_ground_truth=False)
        (admitted if accepted else rejected).append(record)
    return dict(admitted=admitted,rejected=rejected,gates=gates,
        limitation='Thresholds are exploratory, selected on development clips; not a guarantee on new speakers/corpora.')
