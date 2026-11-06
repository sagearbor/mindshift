import numpy as np
import pytest
from eval.codex.specialist_probe import screen_metrics, ordered_labels


def test_surprise_prediction_counts_as_wrong_not_removed():
    scores=screen_metrics(list(range(6)),np.eye(7)[[6,1,2,3,4,5]])
    assert scores['accuracy']==5/6
    assert scores['macro_f1']==5/6
    assert scores['surprise_predictions']==1
    assert scores['confusion_matrix'][0][6]==1


def test_publisher_config_can_keep_json_string_keys():
    assert ordered_labels({'1':'happy','0':'neutral'})==['neutral','happy']
    assert ordered_labels({1:'happy',0:'neutral'})==['neutral','happy']


def test_specialist_probabilities_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        screen_metrics([0],np.array([[-1.,2.,0.,0.,0.,0.,0.]]))
