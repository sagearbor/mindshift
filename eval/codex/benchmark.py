"""Human audio-vote benchmark. No inference services, credentials or product imports."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
LABELS = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
CODES = ['A', 'D', 'F', 'H', 'N', 'S']
SENTENCES = {'IEO': "It's eleven o'clock", 'TIE': 'That is exactly what happened',
 'IOM': "I'm on my way to the meeting", 'IWW': 'I wonder what this is about',
 'TAI': 'The airplane is almost full', 'MTI': 'Maybe tomorrow it will be cold',
 'IWL': 'I would like a new alarm clock', 'ITH': "I think I have a doctor's appointment",
 'DFA': "Don't forget a jacket", 'ITS': "I think I've seen this before",
 'TSI': 'The surface is slick', 'WSI': "We'll stop in a couple of minutes"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def verify_audio_rows(rows):
    """Check consumed audio even when inference/features are cached."""
    for row in rows:
        if sha256(ROOT / row['audio_path']) != row['sha256']:
            raise ValueError(f"Benchmark audio changed: {row['id']}")


def read_audio_votes(path, timing_path=None):
    # Reproduce processFinishedResponses.R: remove first clicks > 10 seconds.
    excluded = set()
    if timing_path is not None:
        with Path(timing_path).open(newline='') as f:
            for row in csv.DictReader(f):
                if row['ttr'] not in {'NULL', 'NA', ''} and float(row['ttr']) > 10000:
                    excluded.add(int(row['sessionNums'])*1000 + int(row['queryType'])*100 + int(row['questNum']))
    votes = defaultdict(lambda: [0]*6)
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            if int(row['queryType']) != 1:
                continue
            if timing_path is not None and int(row['sessionNums'])*1000 + int(row['queryType'])*100 + int(row['questNum']) in excluded:
                continue
            code = row['respEmo']
            if code not in CODES:
                raise ValueError(f'Unknown audio vote: {code}')
            votes[row['clipName']][CODES.index(code)] += 1
    return dict(votes)


def majority(votes):
    v = np.asarray(votes)
    if v.shape != (6,) or not np.isfinite(v).all() or (v < 0).any() or v.sum() <= 0:
        raise ValueError('Invalid vote counts')
    winners = np.flatnonzero(v == v.max())
    return int(winners[0]) if len(winners) == 1 else None


def speaker_splits(speakers, seed='mindshift-emotion-v1'):
    ordered = sorted(set(speakers), key=lambda s: hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest())
    n = len(ordered)
    n_test = max(1, round(n*.2))
    n_dev = max(1, round(n*.2))
    return {s: ('test' if i < n_test else 'dev' if i < n_test+n_dev else 'train')
            for i,s in enumerate(ordered)}


def validate_partition(rows):
    seen = {'id': {}, 'speaker': {}, 'sha256': {}}
    for row in rows:
        if row['split'] not in {'train','dev','test'}:
            raise ValueError('Unknown split')
        for field in seen:
            value = row[field]
            if not value:
                raise ValueError(f'Missing {field}')
            old = seen[field].get(value)
            if old is not None and (old != row['split'] or field == 'id'):
                name = 'audio' if field == 'sha256' else field
                raise ValueError(f'{name} overlap: {value}')
            seen[field][value] = row['split']


def validate_training_pool(pool, gold):
    protected = {k: {r[k] for r in gold} for k in ('id','speaker','sha256')}
    for row in pool:
        if row.get('commercial_use') is not True or row.get('split') != 'train':
            raise ValueError('Training pool must have verified commercial permission and train split')
        if row.get('label_source') != 'machine':
            raise ValueError('Pseudo-label pool must preserve machine provenance')
        if any(not row.get(k) or row[k] in protected[k] for k in protected):
            raise ValueError('Training pool overlaps protected human benchmark or lacks provenance')


def audio_info(path):
    try:
        info = sf.info(path)
    except Exception as exc:
        raise ValueError(f'Invalid or missing audio: {path}') from exc
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f'Empty audio: {path}')
    waveform, _ = sf.read(path, dtype='float32', always_2d=True)
    rms = float(np.sqrt(np.mean(waveform.mean(axis=1).astype(float)**2)))
    if not np.isfinite(rms) or rms < 1e-5:
        raise ValueError(f'silent or nonfinite source audio: {path}')
    return {'rms': rms, 'duration_s': info.duration, 'sample_rate': info.samplerate, 'channels': info.channels}


def metrics(votes, probabilities):
    v, p = np.asarray(votes, dtype=float), np.asarray(probabilities, dtype=float)
    if (v.ndim != 2 or v.shape[1] != 6 or len(v) == 0 or p.shape != v.shape
        or not np.isfinite(v).all() or (v < 0).any() or (v.sum(1) <= 0).any()
        or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
        or not np.allclose(p.sum(1),1,atol=1e-5)):
        raise ValueError('Invalid or incomplete votes/probabilities')
    q = v / v.sum(1,keepdims=True)
    unique = (v == v.max(1,keepdims=True)).sum(1) == 1
    y, pred = v[unique].argmax(1), p[unique].argmax(1)
    if len(y) == 0:
        raise ValueError('No unique human majority labels')
    precision, recall, f1, support = precision_recall_fscore_support(y,pred,labels=range(6),zero_division=0)
    confidence = p[unique].max(1)
    correct = (y == pred).astype(float)
    ece = 0.0
    for lo in np.arange(0,1,.1):
        mask = (confidence >= lo) & (confidence < lo+.1 if lo < .9 else confidence <= 1)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    kappa = (float('nan') if len(set(y) | set(pred)) == 1 else
             float(cohen_kappa_score(y,pred,labels=list(range(6)))))
    return {'n_total': len(v), 'n_single_label': len(y), 'n_tied': int((~unique).sum()),
        'accuracy': float(accuracy_score(y,pred)), 'macro_f1': float(f1.mean()),
        'balanced_accuracy': float(recall.mean()), 'kappa': kappa if np.isfinite(kappa) else None,
        'brier_to_votes': float(((p-q)**2).sum(1).mean()),
        'cross_entropy_to_votes': float(-(q*np.log(np.clip(p,1e-12,1))).sum(1).mean()),
        'expected_individual_vote_agreement': float(q[np.arange(len(q)),p.argmax(1)].mean()),
        'ece_10_bins': float(ece), 'confusion_matrix': confusion_matrix(y,pred,labels=range(6)).tolist(),
        'per_emotion': {label: {'precision': float(precision[i]), 'recall': float(recall[i]),
             'f1': float(f1[i]), 'support': int(support[i])} for i,label in enumerate(LABELS)}}


def bootstrap(votes, probabilities, speakers, repeats=1000, seed=20261105):
    rng = np.random.default_rng(seed)
    groups = {s: np.flatnonzero(np.asarray(speakers)==s) for s in sorted(set(speakers))}
    ids = list(groups)
    values = defaultdict(list)
    for _ in range(repeats):
        idx = np.concatenate([groups[s] for s in rng.choice(ids,len(ids),replace=True)])
        m = metrics(votes[idx],probabilities[idx])
        for key in ('accuracy','macro_f1','balanced_accuracy','kappa','brier_to_votes'):
            if m[key] is not None:
                values[key].append(m[key])
        for emotion, row in m['per_emotion'].items():
            for key in ('precision','recall','f1'):
                values[f'{emotion}.{key}'].append(row[key])
    return {'method': 'speaker-cluster percentile bootstrap', 'repeats': repeats,
            'seed': seed, 'intervals': {k: np.quantile(v,[.025,.975]).tolist() for k,v in values.items()}}


def build_manifest(source, out):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(f'Frozen manifest already exists: {out}')
    votes = read_audio_votes(source/'finishedResponses.csv', source/'finishedEmoResponses.csv')
    splits = speaker_splits([name.split('_')[0] for name in votes])
    with (source/'processedResults/summaryTable.csv').open() as f:
        summaries = {r['FileName']: set(r['VoiceVote'].split(':')) for r in csv.DictReader(f)}
    rows = []
    quarantined = []
    for name in sorted(votes):
        actor, sentence, intended, intensity = name.split('_')
        count = votes[name]
        winners = {CODES[i] for i,x in enumerate(count) if x == max(count)}
        if winners != summaries[name]:
            raise ValueError(f'Raw audio ratings disagree with official VoiceVote: {name}')
        path = source/'AudioWAV'/f'{name}.wav'
        try:
            info = audio_info(path)
        except ValueError as exc:
            if 'silent or nonfinite' not in str(exc):
                raise
            quarantined.append(dict(id=name, sha256=sha256(path), reason=str(exc)))
            continue
        rows.append(dict(id=name, speaker=actor, split=splits[actor],
            audio_path=str(path.relative_to(ROOT)), sha256=sha256(path), **info,
            votes=count, label_source='human_audio_votes', majority=majority(count),
            agreement=max(count)/sum(count), intended_emotion=intended, intensity=intensity,
            transcript=SENTENCES[sentence], corpus='CREMA-D', commercial_use=True,
            license='ODbL-1.0 database; DbCL-1.0 contents'))
    validate_partition(rows)
    manifest = {'schema_version': 1, 'labels': LABELS,
        'source_url': 'https://github.com/CheyneyComputerScience/CREMA-D',
        'source_commit': subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        'ratings_sha256': sha256(source/'finishedResponses.csv'),
        'timing_sha256': sha256(source/'finishedEmoResponses.csv'),
        'rating_filter': 'Original first-response time <= 10000 ms, audio-only queryType=1',
        'split_algorithm': 'sha256(mindshift-emotion-v1:actor), first 20% test, next 20% dev, remainder train',
        'rows': rows, 'quarantined': quarantined}
    save_json(out,manifest)
    summary = {s: {'clips': sum(r['split']==s for r in rows),
        'speakers': len({r['speaker'] for r in rows if r['split']==s}),
        'unique_majority_counts': dict(Counter(LABELS[r['majority']] for r in rows if r['split']==s and r['majority'] is not None)),
        'tied': sum(r['split']==s and r['majority'] is None for r in rows)} for s in ['train','dev','test']}
    summary.update(quarantined=quarantined, manifest_sha256=sha256(out), total_clips=len(rows),
                   max_duration_s=max(r['duration_s'] for r in rows),
                   intended_differs_from_unique_majority=sum(r['majority'] is not None and CODES[r['majority']] != r['intended_emotion'][0] for r in rows))
    save_json(out.with_suffix('.summary.json'),summary)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=ROOT/'data/crema-d')
    ap.add_argument('--out',type=Path,default=ROOT/'data/manifest-v2.json')
    args = ap.parse_args()
    build_manifest(args.source,args.out)
