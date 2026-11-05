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


def test_actor_agreement_filter_only_changes_training():
    rows=[dict(split='train',majority=5,intended_emotion='SAD'),
          dict(split='train',majority=4,intended_emotion='SAD'),
          dict(split='test',majority=5,intended_emotion='SAD'),
          dict(split='train',majority=None,intended_emotion='SAD')]
    assert list(t.training_mask(rows,'actor_agrees')) == [True,False,False,False]
    assert list(t.training_mask(rows,'all_votes')) == [True,True,False,True]


def test_features_reject_wrong_manifest_and_changed_archive(tmp_path):
    import json
    path=tmp_path/'features.npz'
    np.savez(path,x=np.ones((1,2)),ids=['a'])
    path.with_suffix('.json').write_text(json.dumps({'manifest_sha256':'wrong','archive_sha256':'wrong'}))
    with pytest.raises(ValueError,match='manifest'):
        t.load_features(path,['a'],'expected')
