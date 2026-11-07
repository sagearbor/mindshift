import pytest
from eval.codex.mess_data import shared_label, read_ratings
from eval.codex.mess_evaluate import score_shared
import numpy as np


def test_calm_is_not_relabelled_neutral_or_dropped_before_plurality():
    assert shared_label([.1,.7,.1,.1]) is None
    assert shared_label([.4,.4,.1,.1]) is None
    assert shared_label([.1,.1,.7,.1]) == 3
    assert shared_label([.1,.1,.1,.7]) == 5
    assert shared_label([.7,.1,.1,.1]) == 0


def test_invalid_rating_distribution_rejected():
    for q in [[.1,.1,.1,.1],[-.1,.9,.1,.1],[float('nan'),0,0,1]]:
        with pytest.raises(ValueError): shared_label(q)


def test_predictions_outside_shared_emotions_remain_errors():
    result=score_shared([0,3,5],np.eye(6)[[0,4,5]])
    assert result['accuracy']==pytest.approx(2/3)
    assert result['macro_f1']==pytest.approx(2/3)
    assert result['balanced_accuracy']==pytest.approx(2/3)
    assert result['confusion_matrix'][3][4]==1
