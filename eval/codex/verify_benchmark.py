"""Verify a rebuilt manifest matches the portable, frozen label/split index."""
import argparse
import json
from pathlib import Path
from .benchmark import ROOT, sha256, validate_partition


def verify_index(manifest_path,index_path):
    metadata=json.loads(index_path.with_suffix('.meta.json').read_text())
    if sha256(index_path)!=metadata['index_sha256']:raise ValueError('Frozen index hash mismatch')
    rows=json.loads(manifest_path.read_text())['rows']
    index=[json.loads(line) for line in index_path.read_text().splitlines()]
    if len(rows)!=len(index):raise ValueError('Manifest/index clip count differs')
    for current,expected in zip(rows,index):
        if {key:current.get(key) for key in expected}!=expected:
            raise ValueError(f'Manifest differs from frozen labels/splits: {expected["id"]}')
    return dict(clips=len(rows),index_sha256=sha256(index_path),verified=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,default=ROOT/'data/manifest-v2.json')
    p.add_argument('--index',type=Path,default=ROOT/'benchmarks/cremad-v2.jsonl')
    a=p.parse_args();validate_partition(json.loads(a.manifest.read_text())['rows'])
    print(json.dumps(verify_index(a.manifest,a.index),indent=2))
