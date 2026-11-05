import numpy as np
import torch
from eval.codex import model as m


def test_pool_excludes_padded_frames():
    hidden = torch.tensor([[[1.,2.],[3.,4.],[100.,100.]]])
    pooled = m.pool_hidden(hidden,torch.tensor([2]))
    np.testing.assert_allclose(pooled.numpy(),[[2.,3.,1.,1.]],atol=1e-6)


def test_soft_vote_training_uses_distribution():
    x=np.array([[-1.],[1.]],dtype=np.float32)
    votes=np.array([[9,1,0,0,0,0],[1,9,0,0,0,0]],dtype=float)
    xx,yy,weights=m.expand_votes(x,votes)
    assert xx.shape == (4,1)
    assert list(yy)==[0,1,0,1]
    np.testing.assert_allclose(weights,[.9,.1,.1,.9])


def test_feature_cache_fingerprint_changes_with_audio_or_preprocessing():
    row=dict(id='a',sha256='audio1')
    a=m.feature_key(row,'revision',6)
    assert a != m.feature_key(dict(id='a',sha256='audio2'),'revision',6)
    assert a != m.feature_key(row,'revision',8)


def test_silent_audio_is_rejected():
    import pytest
    with pytest.raises(ValueError,match='silence'):
        m.validate_waveform(np.zeros(16000),16000)
