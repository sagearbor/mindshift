import torch
import pytest
from eval.codex import finetune as f


def test_vote_loss_uses_minority_votes_and_has_gradients():
    p=torch.tensor([[.8,.2,0,0,0,0]],requires_grad=True)
    q=torch.tensor([[.6,.4,0,0,0,0]])
    loss=f.vote_loss(p,q)
    assert loss.item()==pytest.approx(-.6*torch.log(torch.tensor(.8)).item()-.4*torch.log(torch.tensor(.2)).item())
    loss.backward()
    assert p.grad[0,1] != 0


def test_vote_loss_rejects_nonfinite():
    with pytest.raises(ValueError):
        f.vote_loss(torch.full((1,6),float('nan')),torch.ones((1,6))/6)


def test_run_source_path_accepts_relative_cli_paths():
    from pathlib import Path
    assert f.run_source_path(Path('eval/codex/results/base-all-v1'))=='results/base-all-v1'
