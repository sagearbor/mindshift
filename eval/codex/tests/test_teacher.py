import numpy as np
import pytest
from eval.codex import teacher as t


def test_forced_choice_distribution_is_finite_and_tracks_mass():
    p,m=t.choice_probabilities(np.log([.1,.2,.3,.4]),[0,2])
    np.testing.assert_allclose(p,[.25,.75]);assert m==pytest.approx(.4)


def test_nonfinite_teacher_output_fails():
    with pytest.raises(ValueError):
        t.choice_probabilities(np.array([np.nan,0]),[0,1])


def test_prompt_never_includes_filename_or_gold_label():
    row=dict(id='1001_IEO_SAD_HI',transcript="It's eleven o'clock",majority=5,intended_emotion='SAD')
    prompt=t.messages(row,'audio')
    assert row['id'] not in str(prompt)
    assert 'SAD_HI' not in str(prompt)
    assert row['transcript'] not in str(prompt)
