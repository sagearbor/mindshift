"""Audit publisher human ratings and freeze a separate Italian diagnostic cohort."""
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from .benchmark import ROOT, save_json, sha256

LABEL_MAP={'anger':0,'disgust':1,'fear':2,'joy':3,'neutrality':4,'sadness':5}


def perceived_label(labels):
    counts=Counter(labels)
    if not counts:return None
    winners=[k for k,v in counts.items() if v==max(counts.values())]
    return LABEL_MAP.get(winners[0]) if len(winners)==1 else None


def freeze():
    import numpy as np
    import soundfile as sf
    folder=ROOT/'data/emozionalmente'
    archive=folder/'emozionalmente.zip'
    if hashlib.file_digest(archive.open('rb'),'md5').hexdigest()!='398c6502dd2b0e1e810f927c12897547':
        raise ValueError('Publisher archive checksum differs')
    target=ROOT/'reports/emozionalmente-protocol.json'
    if target.exists():raise FileExistsError('Do not overwrite frozen external protocol')
    with zipfile.ZipFile(archive) as z:
        def read(name):return list(csv.DictReader(io.StringIO(z.read('emozionalmente/metadata/'+name).decode())))
        samples=read('samples.csv');ratings=read('evaluations.csv')
        splits={name:read('split/'+name+'.csv') for name in ['train','dev','test']}
        speakers={name:{r['actor'] for r in rows} for name,rows in splits.items()}
        for a,b in [('train','dev'),('train','test'),('dev','test')]:
            if speakers[a]&speakers[b]:raise ValueError('External speakers overlap')
        all_ids=[r['file_name'] for rows in splits.values() for r in rows]
        if len(all_ids)!=len(set(all_ids)) or set(all_ids)!={r['file_name'] for r in samples}:raise ValueError('External split coverage differs')
        votes=defaultdict(list)
        for r in ratings:votes[r['file_name']].append(r['emotion_recognized'])
        if set(votes)!={r['file_name'] for r in samples}:raise ValueError('Missing external human votes')
        if any(len(v)!=5 for v in votes.values()):raise ValueError('Expected five original ratings')
        chosen=[];excluded=Counter()
        audio_folder=folder/'test-audio';audio_folder.mkdir(exist_ok=True)
        for sample in splits['test']:
            name=sample['file_name'];labels=votes[name];label=perceived_label(labels)
            if label is None:
                excluded['tied_or_surprise_plurality']+=1;continue
            blob=z.read('emozionalmente/audio/'+name)
            audio,rate=sf.read(io.BytesIO(blob),dtype='float32')
            if rate!=16000 or audio.ndim!=1:raise ValueError('Unexpected external audio format')
            duration=len(audio)/rate
            if duration>6:
                excluded['longer_than_six_seconds']+=1;continue
            if not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2))<1e-5:
                excluded['silent_or_invalid']+=1;continue
            path=audio_folder/name
            if not path.exists():path.write_bytes(blob)
            digest=hashlib.sha256(blob).hexdigest()
            if sha256(path)!=digest:raise ValueError('Existing extracted audio differs')
            chosen.append(dict(id=name,speaker=sample['actor'],intended_emotion=sample['emotion_expressed'],
                original_human_votes=labels,perceived_label=label,audio_path=str(path.relative_to(ROOT)),
                sha256=digest,duration_s=duration))
        model=ROOT/'results/small-finetuned-v1/export/emotion-int8.onnx'
        protocol=dict(dataset='Emozionalmente',source='https://zenodo.org/records/12616095',
            licence='CC BY 4.0',licence_source='https://openslr.org/161/',
            attribution='Fabio Catania, Jan W. Wilke and Franca Garzotto; DOI 10.1109/TASLPRO.2025.3540662',
            archive_sha256=sha256(archive),samples=len(samples),ratings=len(ratings),actors=len({r['actor'] for r in samples}),
            splits={s:dict(clips=len(r),speakers=len(speakers[s])) for s,r in splits.items()},
            original_test_clips=len(splits['test']),excluded=dict(excluded),evaluation_clips=len(chosen),
            selected_model=str(model.relative_to(ROOT)),model_sha256=sha256(model),
            label_map=LABEL_MAP,rows=chosen,
            selection='Existing English development-selected small int8 checkpoint; no Italian tuning',
            caveat='Italian acted cross-language diagnostic; not natural English or proof of unknown pretraining independence',
            scoring='Unique plurality over all seven original labels; exclude ties and surprise winners. Hard-label accuracy and macro F1 on six shared emotions only. Preserve all five ratings and actor intent.')
        save_json(target,protocol)
        print(json.dumps({k:v for k,v in protocol.items() if k!='rows'},indent=2))


if __name__=='__main__':freeze()
