import numpy as np
import pytest
from eval.codex import pseudolabel as p


def test_confidence_gate_requires_per_class_evidence():
    votes=np.eye(6)[[0]*40]
    probs=np.eye(6)[[0]*40]*.99+.01/6
    gates=p.class_gates(votes,probs)
    assert gates['anger']['threshold'] is not None
    assert gates['sadness']['threshold'] is None


def test_high_confidence_wrong_labels_do_not_pass():
    votes=np.eye(6)[[0]*40];probs=np.eye(6)[[1]*40]*.99+.01/6
    assert p.class_gates(votes,probs)['disgust']['threshold'] is None


def test_rejects_teacher_not_qualified():
    with pytest.raises(ValueError,match='quality'):
        p.require_teacher_quality({'macro_f1':.5,'per_emotion':{str(i):{'recall':.9} for i in range(6)}})


@pytest.mark.parametrize('score',[
    {'macro_f1':float('nan'),'per_emotion':{'anger':{'recall':.99}}},
    {'macro_f1':.9,'per_emotion':{'anger':{'recall':.99}}},
])
def test_incomplete_or_nonfinite_teacher_scores_fail_closed(score):
    with pytest.raises(ValueError):
        p.require_teacher_quality(score)


def test_admission_composes_quality_provenance_and_overlap_guards():
    from eval.codex.benchmark import LABELS
    v=np.repeat(np.eye(6),40,axis=0)
    probs=v*.99+.01/6
    gold=[dict(id='gold',speaker='speaker1',sha256='goldhash',split='test')]
    row=dict(id='new',speaker='speaker1',sha256='newhash',split='train',
             label_source='machine',commercial_use=True,model_revision='r',
             license_url='https://example.org/license',source_url='https://example.org/source',
             probabilities=probs[0].tolist())
    with pytest.raises(ValueError,match='overlap'):
        p.admit_rows([row],gold,v,probs,'r')


def test_admission_rejects_different_teacher_revision():
    v=np.repeat(np.eye(6),40,axis=0);probs=v*.99+.01/6
    row=dict(id='new',speaker='new',sha256='newhash',split='train',
             label_source='machine',commercial_use=True,model_revision='wrong',
             license_url='https://example.org/license',source_url='https://example.org/source',
             probabilities=probs[0].tolist())
    with pytest.raises(ValueError,match='revision'):
        p.admit_rows([row],[],v,probs,'expected')
