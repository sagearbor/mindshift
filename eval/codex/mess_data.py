"""Audit publisher MESS ratings; freeze a three-shared-emotion transfer diagnostic."""
import collections
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import soundfile as sf
from .benchmark import ROOT, save_json, sha256


def shared_label(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or (q<0).any() or not np.isclose(q.sum(),1):
        raise ValueError('Invalid four-category human rating distribution')
    winners=np.flatnonzero(np.isclose(q,q.max(),rtol=0,atol=1e-9))
    return {0:0,2:3,3:5}.get(int(winners[0])) if len(winners)==1 else None


def read_ratings(path):
    n={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        strings=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        records=[]
        for r in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//m:row',n):
            values={}
            for c in r:
                v=c.find('m:v',n)
                if v is not None: values[c.get('r').rstrip('0123456789')]=strings[int(v.text)] if c.get('t')=='s' else v.text
            records.append(values)
    header=records.pop(0)
    if [header.get(c) for c in ['F','N','O','P','Q']]!=['code','Percent_A','Percent_C','Percent_H','Percent_S']:
        raise ValueError('Unexpected publisher spreadsheet columns')
    result=[]
    for r in records:
        q=[float(r[c]) for c in ['N','O','P','Q']]
        result.append(dict(id=r['F'],speaker=r['B']+r['C'],actor_emotion=r['A'],
            human_distribution=q,perceived_label=shared_label(q),sentence=r['G'],
            arousal=float(r['J']),valence=float(r['K']),category_accuracy=float(r['L'])))
    if len({r['id'] for r in result})!=len(result):raise ValueError('Duplicate recording codes')
    return result


def run():
    data=ROOT/'data/mess'; destination=ROOT/'reports/mess-protocol.json'
    if destination.exists():raise FileExistsError('Preserve frozen MESS diagnostic')
    for name,expected in [('ratings.xlsx','3fb67e2d99659bd36176e8c67fcc4ff6'),('audio.zip','48bf6823b9ed2eaa680fd0a58af181f0')]:
        if hashlib.md5((data/name).read_bytes()).hexdigest()!=expected:raise ValueError('Publisher checksum mismatch')
    metadata=json.loads((data/'source-metadata.json').read_text())
    if metadata['metadata']['license']['id']!='cc-by-4.0':raise ValueError('Unexpected publisher license')
    ratings=read_ratings(data/'ratings.xlsx'); selected=[]; excluded=collections.Counter(); durations=[]
    audio=data/'diagnostic-audio'; audio.mkdir(exist_ok=True)
    with zipfile.ZipFile(data/'audio.zip') as z:
        members=[f for f in z.namelist() if f.lower().endswith('.wav') and not f.startswith('__MACOSX/')]
        by_code={Path(f).stem.removesuffix('_SCR'):f for f in members}
        if len(by_code)!=len(members):raise ValueError('Ambiguous audio recording codes')
        for row in ratings:
            if row['id'] not in by_code:raise ValueError('Missing audio for rated recording')
            content=z.read(by_code[row['id']]); info=sf.info(io.BytesIO(content)); duration=info.frames/info.samplerate
            durations.append(duration)
            if row['perceived_label'] is None:excluded['calm_plurality_or_tied_plurality']+=1;continue
            if duration>6:excluded['over_six_seconds']+=1;continue
            if duration<=0:raise ValueError('Empty recording')
            path=audio/(row['id']+'.wav')
            if path.exists() and path.read_bytes()!=content:raise ValueError('Existing audio differs')
            path.write_bytes(content)
            selected.append(dict(row,audio_path=str(path.relative_to(ROOT)),sha256=sha256(path),duration=duration))
    model=ROOT/'results/small-finetuned-v1/export/emotion-int8.onnx'
    save_json(destination,dict(dataset='Morgan Emotional Speech Set, Zenodo v5 7378320',
        source='https://zenodo.org/records/7378320',license='CC BY 4.0 (publisher API metadata)',
        metadata_sha256=sha256(data/'source-metadata.json'),ratings_sha256=sha256(data/'ratings.xlsx'),archive_sha256=sha256(data/'audio.zip'),
        original_clips=len(ratings),original_speakers=len({r['speaker'] for r in ratings}),audio_files=len(members),
        selected_clips=len(selected),selected_speakers=len({r['speaker'] for r in selected}),
        selected_label_counts=dict(collections.Counter(r['perceived_label'] for r in selected)),excluded=dict(excluded),
        max_duration=max(durations),selected_model=str(model.relative_to(ROOT)),model_sha256=sha256(model),
        human_category_order=['angry','calm','happy','sad'],scored_labels=[0,3,5],
        selection_rule='Unique plurality among all four human categories; retain angry/happy/sad winners of at most six seconds. Never map calm to neutral or renormalize before selection.',
        scoring_rule='Unchanged six-output model; predictions outside angry/happy/sad count as errors. Macro F1 and recall averaged over those three target emotions only.',
        caveat='Acted English, six actors, four-option human ratings, selected three-emotion diagnostic; not natural speech or a full six-emotion benchmark. Unknown foundation pretraining overlap. No model selection or training using this corpus.',
        rows=selected))
    print(json.dumps({k:v for k,v in json.loads(destination.read_text()).items() if k!='rows'},indent=2))


if __name__=='__main__':run()
