import numpy as np
import pytest
from eval.codex import train as t


def test_fit_rejects_test_rows():
    with pytest.raises(ValueError,match='train'):
        t.fit_head(np.zeros((1,2)),np.ones((1,6)),['test'],1)


def test_selection_never_uses_test_metrics():
    candidates=[{'name':'a','dev_macro_f1':.4,'test_macro_f1':1},
                {'name':'b','dev_macro_f1':.6,'test_macro_f1':0}]
    assert t.select_candidate(candidates)['name']=='b'


def test_feature_alignment_rejects_shuffled_ids():
    with pytest.raises(ValueError,match='IDs'):
        t.check_alignment(['a','b'],['b','a'])
