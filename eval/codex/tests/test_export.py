import numpy as np
import torch
from eval.codex import export as e
from eval.codex.train import predict_head


def test_portable_head_matches_numpy_probabilities():
    head=dict(mean=np.array([1.,2.]),scale=np.array([2.,3.]),
              coef=np.arange(12).reshape(6,2)*.1,intercept=np.arange(6)*.05)
    x=np.array([[3.,5.],[-1.,1.]],dtype=np.float32)
    out=e.PortableHead(head)(torch.tensor(x)).detach().numpy()
    np.testing.assert_allclose(out,predict_head(x,head),rtol=1e-5,atol=1e-6)
