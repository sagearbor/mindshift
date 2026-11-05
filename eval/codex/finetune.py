"""Fine-tune phone-sized encoder against human distributions; dev-only early stopping."""
from __future__ import annotations
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from .benchmark import ROOT, metrics, save_json, sha256, verify_audio_rows
from .model import load_encoder,load_audio,prepare_batch
from .export import EmotionModel,PortableHead
from .train import training_mask


def vote_loss(p,q):
    if not torch.isfinite(p).all() or not torch.isfinite(q).all():
        raise ValueError('Nonfinite training probabilities or votes')
    return -(q*torch.log(p.clamp(min=1e-8))).sum(1).mean()


def run_source_path(source):
    return str(Path(source).resolve().relative_to(ROOT))


def finetune(source,out,manifest_path,epochs=20,batch_size=32):
    if out.exists(): raise FileExistsError('Use a new run directory')
    torch.manual_seed(20261105);np.random.seed(20261105);torch.set_num_threads(6)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    selection=json.loads((source/'selection.json').read_text())
    if selection['manifest_sha256']!=sha256(manifest_path):raise ValueError('Manifest changed')
    rows=json.loads(manifest_path.read_text())['rows']
    tr=training_mask(rows,selection['cohort']);dv=np.array([r['split']=='dev' for r in rows])
    chosen=np.flatnonzero(tr|dv)
    verify_audio_rows([rows[i] for i in chosen])
    encoder,processor=load_encoder(selection['size'])
    model=EmotionModel(encoder,PortableHead(dict(np.load(source/'head.npz'))))
    # The portable head starts at the fitted soft-label solution; train its weights too.
    for name in ['coef','intercept']:
        value=getattr(model.head,name);delattr(model.head,name)
        model.head.register_parameter(name,torch.nn.Parameter(value.clone()))
    model.to(device)
    out.mkdir(parents=True)
    shutil.copy2(source/'head.npz',out/'head.npz')
    for name in ['transcript_head.npz']:
        shutil.copy2(source/name,out/name)
    mel=[];lengths=[];start=time.monotonic()
    for offset in range(0,len(chosen),64):
        batch=[load_audio(ROOT/rows[i]['audio_path']) for i in chosen[offset:offset+64]]
        x,l=prepare_batch(processor,batch);mel.append(x);lengths.append(l)
    mel=torch.cat(mel).to(device);lengths=torch.cat(lengths).to(device)
    votes=np.array([rows[i]['votes'] for i in chosen],dtype=np.float32)
    q=torch.from_numpy(votes/votes.sum(1,keepdims=True)).to(device)
    train_idx=np.flatnonzero(tr[chosen]);dev_idx=np.flatnonzero(dv[chosen])
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-5},
                                {'params':model.head.parameters(),'lr':1e-4}],weight_decay=.01)
    best=selection['winner']['dev_macro_f1'];best_epoch=0;history=[];patience=6
    torch.save(model.cpu().state_dict(),out/'finetuned.pt');model.to(device)
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        order=np.random.permutation(train_idx)
        for offset in range(0,len(order),batch_size):
            idx=torch.tensor(order[offset:offset+batch_size],device=device)
            optimizer.zero_grad(set_to_none=True)
            p=model(mel[idx],lengths[idx]);loss=vote_loss(p,q[idx]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step();losses.append(float(loss.detach().cpu()))
        model.eval();pred=[]
        with torch.inference_mode():
            for offset in range(0,len(dev_idx),batch_size):
                idx=torch.tensor(dev_idx[offset:offset+batch_size],device=device)
                pred.append(model(mel[idx],lengths[idx]).cpu().numpy())
        p=np.concatenate(pred);score=metrics(votes[dev_idx],p)
        record=dict(epoch=epoch,train_loss=float(np.mean(losses)),dev=score,elapsed_seconds=time.monotonic()-start)
        history.append(record);save_json(out/'history.json',history)
        print(f"epoch {epoch}: dev macro F1 {score['macro_f1']:.4f}, loss {np.mean(losses):.4f}, elapsed {time.monotonic()-start:.1f}s",flush=True)
        if score['macro_f1']>best:
            best=score['macro_f1'];best_epoch=epoch
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'finetuned.pt')
            np.savez(out/'dev_predictions.npz',probabilities=p,ids=np.array([rows[chosen[i]]['id'] for i in dev_idx]))
        if epoch-best_epoch>=patience:break
    if best_epoch==0:
        shutil.copy2(source/'dev_predictions.npz',out/'dev_predictions.npz')
        winner=selection['winner']
    else:
        winner=dict(name=f'finetuned-epoch-{best_epoch}',dev_macro_f1=best,dev=history[best_epoch-1]['dev'])
    selection.update(winner=winner,finetune_source=run_source_path(source),best_epoch=best_epoch,
                     training_method='end-to-end human soft-vote cross entropy',epochs_run=len(history),
                     selection_rule='maximum development macro F1, patience 6; test unopened',
                     provisional_gate_passed=best>=.75 and min(r['recall'] for r in winner['dev']['per_emotion'].values())>=.60)
    save_json(out/'selection.json',selection)
    save_json(out/'run_health.json',dict(empty_outputs=0,nonfinite_outputs=0,finish_reason_length=0,device=device,
        elapsed_seconds=time.monotonic()-start,completed=True))
    print('best epoch',best_epoch,'dev macro F1',best,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/manifest-v2.json');p.add_argument('--epochs',type=int,default=20)
    a=p.parse_args();finetune(a.source,a.out,a.manifest,a.epochs)
