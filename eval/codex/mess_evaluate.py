"""Frozen MESS diagnostic: retain all six model outputs, score three target labels."""
import json
import time
import numpy as np
from .benchmark import ROOT, LABELS, sha256, save_json, metrics
from .model import load_audio, prepare_batch, REVISIONS
from .export import ort_session


def score_shared(labels,probabilities):
    if not set(labels)<= {0,3,5}:raise ValueError('Unexpected MESS target')
    p=np.asarray(probabilities)
    if p.shape!=(len(labels),6) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('Invalid model probabilities')
    full=metrics(np.eye(6)[labels],p)
    shared=[full['per_emotion'][LABELS[i]] for i in [0,3,5]]
    return dict(n_total=len(labels),accuracy=full['accuracy'],macro_f1=float(np.mean([r['f1'] for r in shared])),
        balanced_accuracy=float(np.mean([r['recall'] for r in shared])),confusion_matrix=full['confusion_matrix'],
        per_emotion=full['per_emotion'],macro_average_labels=[LABELS[i] for i in [0,3,5]])


def run():
    import torch
    from transformers import WhisperFeatureExtractor
    protocol_path=ROOT/'reports/mess-protocol.json';protocol=json.loads(protocol_path.read_text())
    out=ROOT/'results/mess-small-int8-v1'
    if out.exists():raise FileExistsError('Preserve existing MESS evaluation')
    out.mkdir();health=dict(completed=False,errors=0);start=time.monotonic()
    try:
        torch.set_num_threads(1)
        model=ROOT/protocol['selected_model']
        if sha256(model)!=protocol['model_sha256']:raise ValueError('Frozen model differs')
        frontend=WhisperFeatureExtractor.from_pretrained(ROOT/'artifacts/hf/models--openai--whisper-tiny/snapshots'/REVISIONS['tiny'],local_files_only=True)
        session=ort_session(model);predictions=[]
        for i,row in enumerate(protocol['rows']):
            path=ROOT/row['audio_path']
            if sha256(path)!=row['sha256']:raise ValueError('Frozen audio differs')
            x,lengths=prepare_batch(frontend,[load_audio(path)])
            p=session.run(None,{'log_mel':x.numpy(),'valid_frames':lengths.numpy()})[0][0]
            predictions.append(p)
            if i%100==0:print(f'MESS diagnostic {i+1}/{len(protocol["rows"])}',flush=True)
        probabilities=np.array(predictions)
        measured=score_shared([r['perceived_label'] for r in protocol['rows']],probabilities)
        np.savez(out/'predictions.npz',ids=np.array([r['id'] for r in protocol['rows']]),probabilities=probabilities)
        save_json(out/'summary.json',dict(protocol_sha256=sha256(protocol_path),prediction_sha256=sha256(out/'predictions.npz'),
            model_sha256=sha256(model),scores=measured,elapsed_seconds=time.monotonic()-start,caveat=protocol['caveat']))
        health['completed']=True
    except BaseException as exc:
        health.update(errors=1,error_type=type(exc).__name__,error=str(exc));raise
    finally:
        health['elapsed_seconds']=time.monotonic()-start;save_json(out/'run_health.json',health)


if __name__=='__main__':run()
