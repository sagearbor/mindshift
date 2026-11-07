import json
import pytest
from eval.codex.benchmark import sha256
from eval.codex.verify_benchmark import verify_index


def test_index_checks_stable_labels_not_machine_local_metadata(tmp_path):
    index=tmp_path/'index.jsonl';index.write_text(json.dumps({'id':'a','votes':[1,0,0,0,0,0]})+'\n')
    index.with_suffix('.meta.json').write_text(json.dumps({'index_sha256':sha256(index)}))
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'rows':[{'id':'a','votes':[1,0,0,0,0,0],'rms':.1}],
        'quarantined':[{'reason':'machine-local path'}]}))
    assert verify_index(manifest,index)['clips']==1
    manifest.write_text(json.dumps({'rows':[{'id':'a','votes':[0,1,0,0,0,0]}]}))
    with pytest.raises(ValueError,match='differs'):verify_index(manifest,index)
