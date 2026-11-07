import numpy as np
from eval.codex import evaluate as e


def test_actor_and_consensus_slices_keep_full_benchmark():
    rows=[dict(majority=5,intended_emotion='SAD',agreement=.8),
          dict(majority=4,intended_emotion='SAD',agreement=.5),
          dict(majority=None,intended_emotion='ANG',agreement=.5)]
    masks=e.slices(rows)
    assert masks['all'].tolist()==[True,True,True]
    assert masks['actor_agrees'].tolist()==[True,False,False]
    assert masks['actor_disagrees'].tolist()==[False,True,False]
    assert masks['strong_consensus'].tolist()==[True,False,False]
