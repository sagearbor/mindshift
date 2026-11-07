import numpy as np
import soundfile as sf
from eval.codex import infer


def test_silence_returns_no_emotion_without_loading_model(tmp_path):
    audio=tmp_path/'silence.wav'
    sf.write(audio,np.zeros(16000),16000)
    result=infer.predict(audio,tmp_path/'absent.onnx')
    assert result['status']=='no_speech'
    assert result['probabilities'] is None
