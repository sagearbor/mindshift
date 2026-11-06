"""Unchanged seven-class specialist, preserving out-of-benchmark surprise predictions."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from .benchmark import ROOT, LABELS, save_json, sha256, verify_audio_rows
from .candidate_probe import select_rows


def ordered_labels(mapping):
    normalized={int(k):v for k,v in mapping.items()}
    return [normalized[i] for i in range(len(normalized))]


def screen_metrics(targets, probabilities):
    if probabilities.shape!=(len(targets),7) or not np.isfinite(probabilities).all() or (probabilities<0).any() or (probabilities>1).any() or not np.allclose(probabilities.sum(1),1):
        raise ValueError('Invalid seven-class predictions')
    predicted=probabilities.argmax(1)
    precision,recall,f1,support=precision_recall_fscore_support(targets,predicted,labels=list(range(6)),zero_division=0)
    return dict(n=len(targets),accuracy=accuracy_score(targets,predicted),macro_f1=float(f1.mean()),
        balanced_accuracy=float(recall.mean()),surprise_predictions=int((predicted==6).sum()),
        confusion_matrix=confusion_matrix(targets,predicted,labels=list(range(7))).tolist(),
        per_emotion={name:dict(precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),support=int(support[i])) for i,name in enumerate(LABELS)})


def run(args):
    import torch
    from transformers import AutoModelForAudioClassification, AutoProcessor
    from .model import load_audio
    if args.out.exists():raise FileExistsError('Fresh output directory required')
    args.out.mkdir(parents=True)
    health=dict(completed=False,runtime_errors=0);start=time.monotonic()
    try:
        torch.set_num_threads(6)
        manifest=ROOT/'data/manifest-v2.json'
        rows=select_rows(json.loads(manifest.read_text())['rows'],args.limit);verify_audio_rows(rows)
        device=args.device if args.device!='auto' else ('mps' if torch.backends.mps.is_available() else 'cpu')
        processor=AutoProcessor.from_pretrained(args.model,local_files_only=True)
        model,loading=AutoModelForAudioClassification.from_pretrained(args.model,local_files_only=True,
            trust_remote_code=True,output_loading_info=True,torch_dtype=torch.float32)
        save_json(args.out/'loading.json',loading)
        if loading.get('mismatched_keys') or loading.get('error_msgs') or any('decoder' not in k for k in loading.get('missing_keys',[])):
            raise ValueError('Unexpected missing or mismatched specialist weights')
        expected=['neutral','happy','sad','angry','fearful','disgusted','surprised']
        if ordered_labels(model.config.id2label)!=expected:raise ValueError('Unexpected publisher label order')
        model=model.eval().to(device)
        config=dict(revision=args.revision,manifest_sha256=sha256(manifest),model_path=str(args.model),
            model_config_sha256=sha256(args.model/'config.json'),ids=[r['id'] for r in rows],
            split='development',precision='float32',device=device,labels=LABELS+['surprise'],
            training_overlap='Publisher paper lists CREMA-D in training; this screen is not held-out evidence for this model',
            training_source='https://arxiv.org/html/2511.04914',
            note='Unchanged specialist. Extra surprise predictions count as incorrect on six-label benchmark; no renormalization.')
        save_json(args.out/'config.json',config)
        probabilities=[]
        for i,row in enumerate(rows):
            begin=time.monotonic()
            inputs=processor(load_audio(ROOT/row['audio_path']),sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
            with torch.inference_mode():
                outputs=model(**{k:v.to(device) for k,v in inputs.items() if k in ['input_features','attention_mask']})
                p=outputs['logits'].float().softmax(-1)[0].cpu().numpy()[[3,5,4,1,0,2,6]]
            probabilities.append(p)
            save_json(args.out/f'{i:04d}.json',dict(id=row['id'],audio_sha256=row['sha256'],probabilities7=p.tolist(),
                dimensions=outputs['dims'][0].float().cpu().tolist(),seconds=time.monotonic()-begin,approved_for_training=False))
            print(f'MERaLiON {i+1}/{len(rows)}',flush=True)
        probabilities=np.array(probabilities)
        scores=screen_metrics([r['majority'] for r in rows],probabilities)
        np.savez(args.out/'predictions.npz',ids=np.array([r['id'] for r in rows]),probabilities7=probabilities)
        save_json(args.out/'summary.json',dict(config=config,metrics=scores,qualified_teacher=False,
            note='Balanced development screen only; no automatic training-label admission'))
        health['completed']=True
    except Exception as exc:
        health.update(runtime_errors=1,error_type=type(exc).__name__,error=str(exc));raise
    finally:
        health['elapsed_seconds']=time.monotonic()-start;save_json(args.out/'run_health.json',health)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--revision',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,default=12)
    p.add_argument('--device',choices=['auto','cpu','mps'],default='auto');run(p.parse_args())
