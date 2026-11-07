"""Train on train speakers, choose using development only. Test is a separate command."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .benchmark import ROOT, LABELS, metrics, save_json, sha256, validate_partition, verify_audio_rows
from .model import expand_votes


def check_alignment(expected, actual):
    if list(expected) != list(actual):
        raise ValueError('Feature IDs do not match manifest order')


def load_features(path, ids, manifest_hash):
    path=Path(path)
    meta=json.loads(path.with_suffix('.json').read_text())
    if meta['manifest_sha256'] != manifest_hash:
        raise ValueError('Feature cache belongs to different manifest')
    if meta.get('archive_sha256') != sha256(path):
        raise ValueError('Feature archive changed or lacks a recorded hash')
    features=np.load(path)
    check_alignment(ids,features['ids'])
    if not np.isfinite(features['x']).all():
        raise ValueError('Nonfinite cached features')
    return features


def fit_head(x,votes,splits,c,balanced=False):
    if not all(s=='train' for s in splits):
        raise ValueError('Fitting accepts train split only')
    scaler=StandardScaler().fit(x)
    xx,yy,weights=expand_votes(scaler.transform(x),votes)
    clf=LogisticRegression(C=c,max_iter=2000,class_weight=None)
    if balanced:
        total=np.bincount(yy,weights=weights,minlength=6)
        weights=weights * (total.sum()/(6*total))[yy]
    clf.fit(xx,yy,sample_weight=weights)
    if list(clf.classes_) != list(range(6)):
        raise ValueError('Missing emotion in training data')
    return scaler,clf


def select_candidate(candidates):
    return max(candidates,key=lambda x:x['dev_macro_f1'])


def predict_head(x,head):
    z=(x-head['mean'])/head['scale']
    logits=z @ head['coef'].T + head['intercept']
    logits-=logits.max(1,keepdims=True)
    p=np.exp(logits)
    return p/p.sum(1,keepdims=True)


def training_mask(rows,cohort):
    if cohort not in {'all_votes','actor_agrees'}:
        raise ValueError('Unknown training cohort')
    intended={'ANG':0,'DIS':1,'FEA':2,'HAP':3,'NEU':4,'SAD':5}
    return np.array([r['split']=='train' and (cohort=='all_votes' or r['majority']==intended[r['intended_emotion']]) for r in rows])


def train(size,manifest_path,out,cohort='all_votes'):
    if out.exists():
        raise FileExistsError('Use a new run directory; never overwrite experiment evidence')
    out.mkdir(parents=True)
    manifest=json.loads(manifest_path.read_text())
    rows=manifest['rows']; validate_partition(rows)
    verify_audio_rows(rows)
    features_path=ROOT/'artifacts'/f'features-{size}.npz'
    features=load_features(features_path,[r['id'] for r in rows],sha256(manifest_path))
    check_alignment([r['id'] for r in rows],features['ids'])
    meta=json.loads(features_path.with_suffix('.json').read_text())
    if meta['manifest_sha256'] != sha256(manifest_path):
        raise ValueError('Feature cache belongs to different manifest')
    x=features['x']; votes=np.array([r['votes'] for r in rows])
    tr=training_mask(rows,cohort); dev=np.array([r['split']=='dev' for r in rows])
    candidates=[]; start=time.monotonic()
    for balanced in [False,True]:
        for c in [.01,.1,1,10]:
            name=f'C{c}-balanced{balanced}'
            scaler,clf=fit_head(x[tr],votes[tr],['train']*tr.sum(),c,balanced)
            p=clf.predict_proba(scaler.transform(x[dev]))
            score=metrics(votes[dev],p)
            head=dict(mean=scaler.mean_,scale=scaler.scale_,coef=clf.coef_,intercept=clf.intercept_)
            np.savez(out/f'{name}.npz',**head)
            candidate=dict(name=name,c=c,balanced=balanced,dev_macro_f1=score['macro_f1'],dev=score)
            candidates.append(candidate)
            print(size,name,'dev macro F1',round(score['macro_f1'],4),flush=True)
    winner=select_candidate(candidates)
    head=dict(np.load(out/f"{winner['name']}.npz"))
    np.savez(out/'head.npz',**head)
    dev_p=predict_head(x[dev],head)
    np.savez(out/'dev_predictions.npz',probabilities=dev_p,ids=np.array([r['id'] for r in rows if r['split']=='dev']))
    # Two controls, fit exclusively on training speakers.
    sentence_keys=sorted({r['transcript'] for r in rows})
    text_x=np.eye(len(sentence_keys))[np.array([sentence_keys.index(r['transcript']) for r in rows])]
    sc,cl=fit_head(text_x[tr],votes[tr],['train']*tr.sum(),1)
    text_head=dict(mean=sc.mean_,scale=sc.scale_,coef=cl.coef_,intercept=cl.intercept_)
    np.savez(out/'transcript_head.npz',**text_head)
    prior=votes[tr]/votes[tr].sum(1,keepdims=True)
    prior=prior.mean(0)
    save_json(out/'selection.json',dict(size=size,labels=LABELS,manifest_sha256=sha256(manifest_path),
        feature_metadata=meta,selection_rule='maximum development macro F1; final test unopened',
        winner=winner,candidates=candidates,cohort=cohort,n_train=int(tr.sum()),seconds=time.monotonic()-start,
        transcript_dev=metrics(votes[dev],predict_head(text_x[dev],text_head)),
        prior_dev=metrics(votes[dev],np.tile(prior,(dev.sum(),1))),
        sentence_keys=sentence_keys,train_prior=prior.tolist(),
        provisional_gate_passed=winner['dev']['macro_f1']>=.75 and min(r['recall'] for r in winner['dev']['per_emotion'].values())>=.60))
    print('selected',winner['name'],'saved',out,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--size',choices=['tiny','base','small'],required=True)
    ap.add_argument('--manifest',type=Path,default=ROOT/'data/manifest-v2.json')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cohort',choices=['all_votes','actor_agrees'],default='all_votes')
    a=ap.parse_args(); train(a.size,a.manifest,a.out,a.cohort)
