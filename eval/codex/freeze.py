"""Record all candidate choices before opening the held-out test."""
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from .benchmark import ROOT, sha256, save_json

OPTIONAL=['finetuned.pt','export/emotion-int8.onnx','export/emotion-fp32.onnx','export/validation.json']


def validate_run(run,record):
    for name,present in record['presence'].items():
        if (run/name).exists()!=present:raise ValueError(f'Frozen candidate presence changed: {name}')
    for name,digest in record['hashes'].items():
        if sha256(run/name)!=digest:raise ValueError(f'Frozen candidate changed: {name}')
    if record['presence']['export/emotion-int8.onnx']:
        validation=json.loads((run/'export/validation.json').read_text())
        if not validation['float_parity_passed'] or not validation['quantization_gate_passed']:
            raise ValueError('Export validation gate failed')
        for variant,name in [('int8','emotion-int8.onnx'),('float','emotion-fp32.onnx')]:
            if validation['hashes'][variant]!=sha256(run/'export'/name):raise ValueError('Export validation hash mismatch')

def freeze(names=None,out=ROOT/'reports/pretest-freeze.json'):
    if out.exists():
        raise FileExistsError('Selection already frozen')
    if names is None:
        names=[f'{size}-{cohort}-v1' for size in ['tiny','base','small'] for cohort in ['all','agrees']]
        names += [f'{size}-finetuned-v1' for size in ['tiny','base','small']]
    records=[]
    for name in names:
        run=ROOT/'results'/name
        if (run/'final-test.json').exists():
            raise ValueError('Test already opened')
        selection=json.loads((run/'selection.json').read_text())
        files=['selection.json','head.npz','transcript_head.npz']
        presence={f:(run/f).exists() for f in OPTIONAL}
        files += [f for f,present in presence.items() if present]
        record=dict(run=name,dev_macro_f1=selection['winner']['dev_macro_f1'],presence=presence,
            hashes={f:sha256(run/f) for f in files})
        validate_run(run,record)
        records.append(record)
    winner=max(records,key=lambda r:r['dev_macro_f1'])['run']
    save_json(out,dict(created_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha256(ROOT/'data/manifest-v2.json'),
        benchmark_index_sha256=sha256(ROOT/'benchmarks/cremad-v2.jsonl'),
        candidates=records,selected_accuracy_candidate=winner,
        compact_comparators=[n for n in names if n.startswith(('tiny-','base-')) and 'finetuned' in n],
        selection_rule='Highest development macro F1; final test cannot change training or candidate choice',
        export_rule='Validate float numerical parity and int8 dev macro F1 drop <= 0.02; no test-driven export tuning',
        teacher_gate=dict(macro_f1_min=.75,every_class_recall_min=.60),
        scope='Experimental candidates only; no deployment, no new machine-label pool, no watch qualification'))
    print(winner,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+');p.add_argument('--out',type=Path,default=ROOT/'reports/pretest-freeze.json')
    a=p.parse_args();freeze(a.runs,a.out)
