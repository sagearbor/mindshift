"""Whisper encoder features and portable emotion head; all inference is local."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from .benchmark import ROOT, sha256, save_json

REVISIONS = {'tiny':'169d4a4341b33bc18d8881c4b69c2e104e1cc0af',
             'base':'e37978b90ca9030d5170a5c07aadb050351a65bb'}
WINDOW_SECONDS = 6


def validate_waveform(x, sr):
    if sr <= 0 or x.ndim != 1 or len(x) < sr//10 or not np.isfinite(x).all():
        raise ValueError('Invalid waveform')
    if np.sqrt(np.mean(x.astype(float)**2)) < 1e-5:
        raise ValueError('Audio is silence or below -100 dBFS')


def load_audio(path, seconds=WINDOW_SECONDS):
    x,sr = sf.read(path,dtype='float32',always_2d=True)
    x=x.mean(axis=1)
    validate_waveform(x,sr)
    if sr != 16000:
        g=math.gcd(sr,16000)
        x=resample_poly(x,16000//g,sr//g).astype(np.float32)
    if len(x) > seconds*16000:
        raise ValueError(f'Clip exceeds {seconds}s; segment explicitly, never truncate silently: {path}')
    return x


def pool_hidden(hidden, lengths):
    mask=(torch.arange(hidden.shape[1],device=hidden.device)[None,:] < lengths[:,None]).to(hidden.dtype)[:,:,None]
    denom=mask.sum(1).clamp(min=1)
    mean=(hidden*mask).sum(1)/denom
    var=((hidden-mean[:,None,:])**2*mask).sum(1)/denom
    return torch.cat([mean,torch.sqrt(var.clamp(min=1e-12))],dim=1)


def expand_votes(x,votes):
    q=votes/votes.sum(axis=1,keepdims=True)
    row,cls=np.nonzero(q)
    return x[row],cls,q[row,cls]


def feature_key(row,revision,seconds):
    return hashlib.sha256(f"{row['sha256']}:{revision}:{seconds}:valid-mean-std-v1".encode()).hexdigest()


def load_encoder(size,device='cpu',seconds=WINDOW_SECONDS):
    from transformers import WhisperModel, WhisperFeatureExtractor
    location=ROOT/'artifacts/hf'/f'models--openai--whisper-{size}'/'snapshots'/REVISIONS[size]
    feature_extractor=WhisperFeatureExtractor.from_pretrained(location,local_files_only=True)
    encoder=WhisperModel.from_pretrained(location,local_files_only=True,attn_implementation='eager').encoder
    # Preserve learned positions for the shorter prefix; same rule for extraction/export.
    positions=seconds*50
    encoder.embed_positions=torch.nn.Embedding.from_pretrained(encoder.embed_positions.weight[:positions].detach().clone(),freeze=True)
    encoder.config.max_source_positions=positions
    encoder.eval().to(device)
    return encoder,feature_extractor


def prepare_batch(extractor,audios,seconds=WINDOW_SECONDS):
    mel=extractor(audios,sampling_rate=16000,padding='max_length',max_length=seconds*16000,
                  truncation=False,return_tensors='pt').input_features
    lengths=torch.tensor([min(seconds*50,(len(x)+319)//320) for x in audios],dtype=torch.long)
    return mel,lengths


def extract(size,manifest_path,batch_size=32,limit=None):
    torch.set_num_threads(6)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    manifest=json.loads(Path(manifest_path).read_text())
    rows=manifest['rows'][:limit] if limit else manifest['rows']
    encoder,processor=load_encoder(size,device)
    cache=ROOT/'artifacts'/f'features-{size}'
    cache.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    missing=[r for r in rows if not (cache/(feature_key(r,REVISIONS[size],WINDOW_SECONDS)+'.npy')).exists()]
    for offset in range(0,len(missing),batch_size):
        batch=missing[offset:offset+batch_size]
        for r in batch:
            if sha256(ROOT/r['audio_path']) != r['sha256']:
                raise ValueError(f"Audio changed: {r['id']}")
        audios=[load_audio(ROOT/r['audio_path']) for r in batch]
        mel,lengths=prepare_batch(processor,audios)
        with torch.inference_mode():
            h=encoder(mel.to(device)).last_hidden_state
            features=pool_hidden(h,lengths.to(device)).cpu().numpy()
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite encoder features')
        for r,feat in zip(batch,features):
            path=cache/(feature_key(r,REVISIONS[size],WINDOW_SECONDS)+'.npy')
            with path.with_suffix('.partial').open('wb') as f:
                np.save(f,feat)
            path.with_suffix('.partial').replace(path)
        print(f'{size}: {offset+len(batch)}/{len(missing)} new clips, {time.monotonic()-start:.1f}s',flush=True)
    x=np.stack([np.load(cache/(feature_key(r,REVISIONS[size],WINDOW_SECONDS)+'.npy')) for r in rows])
    out=ROOT/'artifacts'/f'features-{size}{"-smoke" if limit else ""}.npz'
    np.savez(out,x=x,ids=np.array([r['id'] for r in rows]))
    save_json(out.with_suffix('.json'),dict(model=f'openai/whisper-{size}',revision=REVISIONS[size],
        manifest_sha256=sha256(manifest_path),seconds=WINDOW_SECONDS,pooling='valid-frame mean+std',
        device=device,n_clips=len(rows),new_clips=len(missing),elapsed_seconds=time.monotonic()-start,
        empty_outputs=0,nonfinite_outputs=0,finish_reason_length=0,
        note='Encoder forward pass, no text generation or token budget. First six seconds of positional table; no clip truncation.'))
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--size',choices=REVISIONS,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/manifest-v1.json')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--limit',type=int)
    a=p.parse_args()
    extract(a.size,a.manifest,a.batch_size,a.limit)
