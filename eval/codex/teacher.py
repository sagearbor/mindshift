"""Local Qwen audio/text forced-choice diagnostic, not an approved label source."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .benchmark import ROOT,LABELS,save_json,sha256,metrics,verify_audio_rows
from .model import load_audio

REVISION='ae9e1690543ffd5c0221dc27f79834d0294cba00'
INSTRUCTION=('Classify the emotion perceived in the speaker\'s voice, not the emotion suggested by the words. '
             'Choose exactly one label: A = anger; B = disgust; C = fear; D = happiness; E = neutral; F = sadness. '
             'Reply with only the single letter A, B, C, D, E, or F.')


def choice_probabilities(logits,choices):
    z=np.asarray(logits,dtype=float)
    if not np.isfinite(z).all():raise ValueError('Nonfinite teacher output')
    chosen=z[choices]
    return np.exp(chosen-logsumexp(chosen)),float(np.exp(logsumexp(chosen)-logsumexp(z)))


def messages(row,mode):
    if mode=='audio':
        content=[{'type':'audio','audio':'local-audio'}, {'type':'text','text':INSTRUCTION}]
    elif mode=='transcript':
        content=[{'type':'text','text':INSTRUCTION+'\nOnly the transcript is available: '+row['transcript']}]
    else:raise ValueError('Unknown teacher mode')
    return [{'role':'system','content':[{'type':'text','text':'You are a careful speech emotion classifier.'}]},
            {'role':'user','content':content}]


def run(manifest_path,out,limit=None):
    import torch
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniProcessor
    torch.set_num_threads(6)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    dtype=torch.bfloat16 if device=='mps' else torch.float32
    location=ROOT/'artifacts/hf/models--Qwen--Qwen2.5-Omni-7B/snapshots'/REVISION
    processor=Qwen2_5OmniProcessor.from_pretrained(location,local_files_only=True)
    model=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(location,local_files_only=True,
        torch_dtype=dtype,attn_implementation='eager').eval().to(device)
    choices=[]
    for letter in 'ABCDEF':
        tokens=processor.tokenizer.encode(letter,add_special_tokens=False)
        if len(tokens)!=1:raise ValueError('Choice must be one token')
        choices.append(tokens[0])
    rows=[r for r in json.loads(manifest_path.read_text())['rows'] if r['split']=='dev']
    # Stable interleaving across human categories for a diagnostic smoke subset only.
    if limit:
        groups=[[r for r in rows if r['majority']==i] for i in range(6)]
        rows=[g[j] for j in range(max(map(len,groups))) for g in groups if j<len(g)][:limit]
    out.mkdir(parents=True,exist_ok=True)
    verify_audio_rows(rows)
    config=dict(revision=REVISION,prompt=INSTRUCTION,manifest_sha256=sha256(manifest_path),
        method='single forward pass, six forced-choice next-token scores; no generation/token cap',
        contamination='CREMA-D exposure during publisher pretraining unknown',n=len(rows),device=device)
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    start=time.monotonic();results={}
    for mode in ['audio','transcript']:
        records=[]
        for i,row in enumerate(rows):
            key=hashlib.sha256(f'{config_hash}:{mode}:{row["sha256"]}'.encode()).hexdigest()
            path=out/f'{key}.json'
            if path.exists():record=json.loads(path.read_text())
            else:
                text=processor.apply_chat_template(messages(row,mode),tokenize=False,add_generation_prompt=True)
                args=dict(text=text,return_tensors='pt',padding=True)
                if mode=='audio':args['audio']=[load_audio(ROOT/row['audio_path'])]
                inputs=processor(**args).to(device)
                for k,v in inputs.items():
                    if v.is_floating_point():inputs[k]=v.to(dtype)
                with torch.inference_mode():
                    logits=model(**inputs,use_cache=False).logits[0,-1].float().cpu().numpy()
                probabilities,mass=choice_probabilities(logits,choices)
                record=dict(id=row['id'],audio_sha256=row['sha256'],config_hash=config_hash,mode=mode,
                    probabilities=probabilities.tolist(),allowed_choice_mass=mass,label_source='machine',
                    approved_for_training=False)
                save_json(path,record)
            records.append(record)
            if (i+1)%20==0 or i==0:print(mode,i+1,'/',len(rows),'elapsed',round(time.monotonic()-start,1),flush=True)
        p=np.array([r['probabilities'] for r in records]);v=np.array([r['votes'] for r in rows])
        results[mode]=dict(metrics=metrics(v,p),low_choice_mass_count=sum(r['allowed_choice_mass']<.1 for r in records),
                           mean_choice_mass=float(np.mean([r['allowed_choice_mass'] for r in records])))
        np.savez(out/f'{mode}-predictions.npz',ids=np.array([r['id'] for r in rows]),probabilities=p)
    save_json(out/'summary.json',dict(config=config,results=results,elapsed_seconds=time.monotonic()-start,
         empty_outputs=0,nonfinite_outputs=0,finish_reason_length=0,
         note='Choice probabilities are conditional scores, not calibrated truth. Development diagnostic only.'))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ROOT/'data/manifest-v2.json')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int)
    a=p.parse_args();run(a.manifest,a.out,a.limit)
