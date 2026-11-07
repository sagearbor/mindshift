import json
import pytest
from eval.codex import freeze, evaluate
from eval.codex.benchmark import sha256


def fixture_run(tmp_path,monkeypatch):
    monkeypatch.setattr(freeze,'ROOT',tmp_path)
    run=tmp_path/'results/candidate';run.mkdir(parents=True)
    (tmp_path/'data').mkdir();(tmp_path/'benchmarks').mkdir()
    manifest=tmp_path/'data/manifest-v2.json';manifest.write_text('{}')
    (tmp_path/'benchmarks/cremad-v2.jsonl').write_text('{}\n')
    (run/'selection.json').write_text(json.dumps({'winner':{'dev_macro_f1':.5}}))
    (run/'head.npz').write_bytes(b'not loaded in these rejection tests')
    (run/'transcript_head.npz').write_bytes(b'control')
    return run,manifest,tmp_path/'protocol.json'


def test_freeze_is_immutable_and_hashes_candidates(tmp_path,monkeypatch):
    run,manifest,out=fixture_run(tmp_path,monkeypatch)
    freeze.freeze(['candidate'],out)
    result=json.loads(out.read_text())
    assert result['manifest_sha256']==sha256(manifest)
    assert result['candidates'][0]['hashes']['head.npz']==sha256(run/'head.npz')
    with pytest.raises(FileExistsError):freeze.freeze(['candidate'],out)


def test_freeze_rejects_already_opened_test(tmp_path,monkeypatch):
    run,_,out=fixture_run(tmp_path,monkeypatch)
    (run/'final-test.json').write_text('{}')
    with pytest.raises(ValueError,match='already opened'):freeze.freeze(['candidate'],out)


def test_evaluation_rejects_postfreeze_model_change(tmp_path,monkeypatch):
    run,manifest,out=fixture_run(tmp_path,monkeypatch)
    freeze.freeze(['candidate'],out)
    (run/'head.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen candidate changed'):
        evaluate.evaluate(run,manifest,out)


def test_evaluation_rejects_added_checkpoint(tmp_path,monkeypatch):
    run,manifest,out=fixture_run(tmp_path,monkeypatch)
    freeze.freeze(['candidate'],out)
    (run/'finetuned.pt').write_bytes(b'new checkpoint')
    with pytest.raises(ValueError,match='presence changed'):
        evaluate.evaluate(run,manifest,out)


def test_evaluation_rejects_changed_onnx(tmp_path,monkeypatch):
    run,manifest,out=fixture_run(tmp_path,monkeypatch)
    export=run/'export';export.mkdir()
    for name in ['emotion-int8.onnx','emotion-fp32.onnx']:(export/name).write_bytes(b'graph')
    (export/'validation.json').write_text(json.dumps(dict(float_parity_passed=True,quantization_gate_passed=True,
        hashes={'float':sha256(export/'emotion-fp32.onnx'),'int8':sha256(export/'emotion-int8.onnx')})))
    freeze.freeze(['candidate'],out)
    (export/'emotion-int8.onnx').write_bytes(b'changed graph')
    with pytest.raises(ValueError,match='Frozen candidate changed'):
        evaluate.evaluate(run,manifest,out)
