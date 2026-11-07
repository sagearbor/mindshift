"""Run an exported experimental model on one WAV of at most six seconds."""
import argparse
import json
from pathlib import Path
import numpy as np
from .benchmark import LABELS, sha256
from .model import load_audio, prepare_batch, REVISIONS, ROOT
from .export import ort_session


def predict(audio_path, model_path):
    try:
        audio=load_audio(audio_path)
    except ValueError as exc:
        if 'silence' in str(exc):
            return dict(status='no_speech',probabilities=None,emotion=None)
        raise
    contract_path=model_path.parent/'input_contract.json'
    contract=json.loads(contract_path.read_text())
    if contract['labels']!=LABELS:
        raise ValueError('Model label order mismatch')
    validation=json.loads((model_path.parent/'validation.json').read_text())
    variant='int8' if model_path.name=='emotion-int8.onnx' else 'float'
    if sha256(model_path)!=validation['hashes'][variant]:
        raise ValueError('Exported model changed')
    from transformers import WhisperFeatureExtractor
    # The same 80-bin frontend is shared across tiny/base/small.
    location=ROOT/'artifacts/hf/models--openai--whisper-tiny/snapshots'/REVISIONS['tiny']
    processor=WhisperFeatureExtractor.from_pretrained(location,local_files_only=True)
    x,l=prepare_batch(processor,[audio])
    p=ort_session(model_path).run(None,{'log_mel':x.numpy(),'valid_frames':l.numpy()})[0][0]
    if not np.isfinite(p).all() or not np.isclose(p.sum(),1,atol=1e-5):
        raise ValueError('Invalid exported model probabilities')
    return dict(status='experimental_prediction',emotion=LABELS[int(p.argmax())],
        probabilities={name:float(value) for name,value in zip(LABELS,p)},
        source_sha256=sha256(audio_path),model_sha256=sha256(model_path),
        note='Perceived-voice emotion estimate; not a verified inner emotional state or a nudge recommendation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(predict(args.audio,args.model),indent=2,allow_nan=False))
