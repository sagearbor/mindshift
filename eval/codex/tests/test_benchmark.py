import csv
from pathlib import Path
import numpy as np
import pytest
from eval.codex import benchmark as b


def test_audio_votes_ignore_video_and_intended_emotion(tmp_path):
    path = tmp_path / 'votes.csv'
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=['queryType','clipName','respEmo'])
        w.writeheader()
        w.writerows([dict(queryType=q, clipName='1001_IEO_ANG_LO', respEmo=e)
                     for q,e in [(1,'H'),(1,'N'),(1,'H'),(2,'A'),(3,'A')]])
    votes = b.read_audio_votes(path)
    assert votes['1001_IEO_ANG_LO'] == [0,0,0,2,1,0]


def test_tie_is_not_arbitrarily_broken():
    assert b.majority([1,0,0,1,0,0]) is None
    assert b.majority([1,0,0,2,0,0]) == 3
    with pytest.raises(ValueError):
        b.majority([0]*6)


def test_speaker_split_is_order_independent_and_disjoint():
    speakers = [str(x) for x in range(1001,1092)]
    a = b.speaker_splits(speakers)
    assert a == b.speaker_splits(speakers[::-1])
    assert set(a.values()) == {'train','dev','test'}
    assert len(a) == 91


def test_duplicate_audio_across_splits_is_rejected():
    rows = [dict(id='a',speaker='1',split='train',sha256='x'),
            dict(id='b',speaker='2',split='test',sha256='x')]
    with pytest.raises(ValueError, match='audio'):
        b.validate_partition(rows)


def test_same_speaker_across_splits_is_rejected():
    rows = [dict(id='a',speaker='1',split='train',sha256='x'),
            dict(id='b',speaker='1',split='test',sha256='y')]
    with pytest.raises(ValueError, match='speaker'):
        b.validate_partition(rows)


def test_metrics_perfect_and_ties_counted():
    votes = np.eye(6)*10
    votes = np.vstack([votes,[5,5,0,0,0,0]])
    probs = votes / votes.sum(axis=1,keepdims=True)
    result = b.metrics(votes, probs)
    assert result['macro_f1'] == 1
    assert result['kappa'] == 1
    assert result['n_single_label'] == 6
    assert result['n_tied'] == 1
    assert result['brier_to_votes'] == 0
    assert result['per_emotion']['fear']['recall'] == 1


@pytest.mark.parametrize('bad', [np.full((6,6),np.nan), np.ones((6,6)), np.ones((5,6))/6])
def test_bad_predictions_fail_loudly(bad):
    with pytest.raises(ValueError):
        b.metrics(np.eye(6), bad)


def test_machine_labels_never_enter_gold_or_overlap(tmp_path):
    gold = [dict(id='gold',speaker='s1',sha256='a',split='test')]
    with pytest.raises(ValueError):
        b.validate_training_pool([dict(id='new',speaker='s1',sha256='b',split='train',label_source='machine',commercial_use=True)], gold)
    with pytest.raises(ValueError):
        b.validate_training_pool([dict(id='new',speaker='s2',sha256='b',split='test',label_source='machine',commercial_use=True)], gold)
    with pytest.raises(ValueError):
        b.validate_training_pool([dict(id='new',speaker='s2',sha256='b',split='train',label_source='machine',commercial_use=False)], gold)


def test_lfs_pointer_is_not_audio(tmp_path):
    p = tmp_path / 'bad.wav'
    p.write_text('version https://git-lfs.github.com/spec/v1\n')
    with pytest.raises(ValueError, match='audio'):
        b.audio_info(p)


def test_original_response_time_filter_is_reproduced(tmp_path):
    path=tmp_path/'votes.csv'
    path.write_text('queryType,clipName,respEmo,sessionNums,questNum\n1,clip,A,1,1\n1,clip,N,2,1\n')
    timing=tmp_path/'timing.csv'
    timing.write_text('queryType,sessionNums,questNum,ttr\n1,1,1,10001\n1,2,1,10000\n')
    assert b.read_audio_votes(path,timing)['clip'] == [0,0,0,0,1,0]


def test_missing_response_time_matches_original_r_filter(tmp_path):
    path=tmp_path/'votes.csv'
    path.write_text('queryType,clipName,respEmo,sessionNums,questNum\n1,clip,A,1,1\n')
    timing=tmp_path/'timing.csv'
    timing.write_text('queryType,sessionNums,questNum,ttr\n1,1,1,NULL\n')
    assert b.read_audio_votes(path,timing)['clip'] == [1,0,0,0,0,0]
