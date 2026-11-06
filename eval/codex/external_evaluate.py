"""Evaluate the frozen Italian diagnostic using the existing compact ONNX export."""
import json
import time
import numpy as np
from .benchmark import ROOT, sha256, save_json, metrics
from .model import load_audio, prepare_batch, REVISIONS
from .export import ort_session


def run():
    import torch
    from transformers import WhisperFeatureExtractor
    protocol_path=ROOT/'reports/emozionalmente-protocol.json'
    protocol=json.loads(protocol_path.read_text())
    out=ROOT/'results/emozionalmente-small-int8-v1'
    if out.exists():raise FileExistsError('Preserve existing external evaluation')
    out.mkdir()
    health=dict(completed=False,errors=0)
    start=time.monotonic()
    try:
        torch.set_num_threads(1)
        model=ROOT/protocol['selected_model']
        if sha256(model)!=protocol['model_sha256']:raise ValueError('Frozen model differs')
        frontend=WhisperFeatureExtractor.from_pretrained(ROOT/'artifacts/hf/models--openai--whisper-tiny/snapshots'/REVISIONS['tiny'],local_files_only=True)
        session=ort_session(model);predictions=[]
        for i,row in enumerate(protocol['rows']):
            path=ROOT/row['audio_path']
            if sha256(path)!=row['sha256']:raise ValueError('External audio differs')
            x,lengths=prepare_batch(frontend,[load_audio(path)])
            p=session.run(None,{'log_mel':x.numpy(),'valid_frames':lengths.numpy()})[0][0]
            if not np.isfinite(p).all() or not np.isclose(p.sum(),1,atol=1e-5):raise ValueError('Invalid probabilities')
            predictions.append(p)
            if i%50==0:print(f'Italian diagnostic {i+1}/{len(protocol["rows"])}',flush=True)
        probabilities=np.array(predictions)
        np.savez(out/'predictions.npz',ids=np.array([r['id'] for r in protocol['rows']]),probabilities=probabilities)
        hard_targets=np.eye(6)[[r['perceived_label'] for r in protocol['rows']]]
        calculated=metrics(hard_targets,probabilities)
        scores={k:calculated[k] for k in ['n_total','accuracy','macro_f1','balanced_accuracy','kappa','confusion_matrix','per_emotion']}
        save_json(out/'summary.json',dict(protocol_sha256=sha256(protocol_path),prediction_sha256=sha256(out/'predictions.npz'),
            model_sha256=sha256(model),scores=scores,elapsed_seconds=time.monotonic()-start,
            caveat=protocol['caveat'],excluded=protocol['excluded'],original_test_clips=protocol['original_test_clips'],
            note='No Italian training. Hard perceived-label scoring, not renormalized six-class listener votes.'))
        health['completed']=True
    except Exception as exc:
        health.update(errors=1,error_type=type(exc).__name__,error=str(exc));raise
    finally:
        health['elapsed_seconds']=time.monotonic()-start
        save_json(out/'run_health.json',health)


if __name__=='__main__':run()
