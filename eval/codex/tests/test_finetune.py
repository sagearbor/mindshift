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


def test_balance_weights_use_training_votes_only_and_keep_average_weight_one():
    q=torch.tensor([[.9,.1],[.8,.2],[0.,1.]])
    weights=f.balance_weights(q,torch.tensor([0,1]),.5)
    assert weights[1]>weights[0]
    assert float((q[:2]*weights).sum(1).mean())==pytest.approx(1.)
    changed=q.clone();changed[2]=torch.tensor([1.,0.])
    assert torch.equal(weights,f.balance_weights(changed,torch.tensor([0,1]),.5))
    assert torch.equal(f.balance_weights(q,torch.tensor([0,1]),0),torch.ones(2))


def test_weighted_vote_loss_keeps_minority_votes():
    p=torch.tensor([[.8,.2]],requires_grad=True);q=torch.tensor([[.6,.4]])
    weights=torch.tensor([.5,2.])
    assert f.vote_loss(p,q,weights).item()==pytest.approx(float(-(.3*torch.log(p[0,0].detach())+.8*torch.log(p[0,1].detach()))))
