"""Final held-out evaluation, never called by training/model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .benchmark import ROOT, metrics, bootstrap, save_json, sha256, verify_audio_rows
from .model import load_encoder, load_audio, prepare_batch
from .train import predict_head, check_alignment, load_features
from .export import EmotionModel, PortableHead, ort_session
from .freeze import validate_run


def slices(rows):
    intended={'ANG':0,'DIS':1,'FEA':2,'HAP':3,'NEU':4,'SAD':5}
    return {'all':np.ones(len(rows),dtype=bool),
        'actor_agrees':np.array([r['majority']==intended[r['intended_emotion']] for r in rows]),
        'actor_disagrees':np.array([r['majority'] is not None and r['majority']!=intended[r['intended_emotion']] for r in rows]),
        'strong_consensus':np.array([r['agreement']>=2/3 for r in rows]),
        'weak_consensus':np.array([r['agreement']<2/3 for r in rows])}


def score_slices(rows, p):
    votes=np.array([r['votes'] for r in rows])
    result={}
    for name, mask in slices(rows).items():
        if mask.any() and any(r['majority'] is not None for r,m in zip(rows,mask) if m):
            result[name]=metrics(votes[mask],p[mask])
    result['confidence_intervals']=bootstrap(votes,p,[r['speaker'] for r in rows])
    intended={'ANG':0,'DIS':1,'FEA':2,'HAP':3,'NEU':4,'SAD':5}
    result['actor_intent_secondary']=metrics(np.eye(6)[[intended[r['intended_emotion']] for r in rows]],p)
    return result


def evaluate(run, manifest_path, freeze_path=ROOT/'reports/pretest-freeze.json'):
    out=run/'final-test.json'
    if out.exists(): raise FileExistsError('Final test already evaluated; inspect saved evidence instead of retuning')
    frozen=json.loads(freeze_path.read_text())
    record=next((r for r in frozen['candidates'] if r['run']==run.name),None)
    if record is None:
        raise ValueError('Candidate not present in pre-test freeze; create a new documented evaluation protocol')
    if frozen['manifest_sha256']!=sha256(manifest_path):
        raise ValueError('Frozen manifest changed')
    validate_run(run,record)
    selection=json.loads((run/'selection.json').read_text())
    if selection['manifest_sha256']!=sha256(manifest_path):raise ValueError('Manifest changed')
    all_rows=json.loads(manifest_path.read_text())['rows']
    indices=[i for i,r in enumerate(all_rows) if r['split']=='test']
    rows=[all_rows[i] for i in indices]
    verify_audio_rows(rows)
    predictions={}
    if (run/'finetuned.pt').exists():
        device='mps' if torch.backends.mps.is_available() else 'cpu'
        encoder,processor=load_encoder(selection['size'])
        model=EmotionModel(encoder,PortableHead(dict(np.load(run/'head.npz')))).eval()
        model.load_state_dict(torch.load(run/'finetuned.pt',map_location='cpu',weights_only=True))
        model.to(device);chunks=[]
        with torch.inference_mode():
            for offset in range(0,len(rows),32):
                x,l=prepare_batch(processor,[load_audio(ROOT/r['audio_path']) for r in rows[offset:offset+32]])
                chunks.append(model(x.to(device),l.to(device)).cpu().numpy())
        predictions['float']=np.concatenate(chunks)
    else:
        features=load_features(ROOT/'artifacts'/f"features-{selection['size']}.npz",[r['id'] for r in all_rows],sha256(manifest_path))
        check_alignment([r['id'] for r in all_rows],features['ids'])
        predictions['float']=predict_head(features['x'][indices],dict(np.load(run/'head.npz')))
    if (run/'export/emotion-int8.onnx').exists():
        session=ort_session(run/'export/emotion-int8.onnx')
        _,processor=load_encoder(selection['size']);chunks=[]
        for offset in range(0,len(rows),16):
            x,l=prepare_batch(processor,[load_audio(ROOT/r['audio_path']) for r in rows[offset:offset+16]])
            chunks.append(session.run(None,{'log_mel':x.numpy(),'valid_frames':l.numpy()})[0])
        predictions['int8']=np.concatenate(chunks)
    sentence_keys=selection['sentence_keys']
    text_x=np.eye(len(sentence_keys))[[sentence_keys.index(r['transcript']) for r in rows]]
    predictions['transcript']=predict_head(text_x,dict(np.load(run/'transcript_head.npz')))
    predictions['prior']=np.tile(selection['train_prior'],(len(rows),1))
    np.savez(run/'test-predictions.npz',ids=np.array([r['id'] for r in rows]),**predictions)
    results={name:score_slices(rows,p) for name,p in predictions.items()}
    save_json(out,dict(manifest_sha256=sha256(manifest_path),run=str(run.resolve().relative_to(ROOT)),
        pretest_freeze_sha256=sha256(freeze_path),
        selected_on='development only',selection_sha256=sha256(run/'selection.json'),
        test_prediction_sha256=sha256(run/'test-predictions.npz'),results=results,
        empty_outputs=0,nonfinite_outputs=0,finish_reason_length=0,
        n_test=len(rows),n_test_speakers=len({r['speaker'] for r in rows}),
        note='Repeated future regression checks do not create a new independent final test. No natural-speech or phone-device accuracy claim.'))
    print(json.dumps({k:{m:v['all'][m] for m in ['macro_f1','accuracy','kappa']} for k,v in results.items()},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/manifest-v2.json')
    p.add_argument('--freeze',type=Path,default=ROOT/'reports/pretest-freeze.json')
    a=p.parse_args();evaluate(a.run,a.manifest,a.freeze)
