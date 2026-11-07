"""Export encoder+head; validate ONNX probabilities and measure host CPU latency."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .benchmark import ROOT, LABELS, metrics, sha256, save_json, verify_audio_rows
from .model import load_encoder, load_audio, prepare_batch, pool_hidden, WINDOW_SECONDS


class PortableHead(torch.nn.Module):
    def __init__(self,head):
        super().__init__()
        for key in ['mean','scale','coef','intercept']:
            self.register_buffer(key,torch.as_tensor(head[key],dtype=torch.float32))

    def forward(self,x):
        logits=((x-self.mean)/self.scale) @ self.coef.T + self.intercept
        return torch.softmax(logits,dim=-1)


class EmotionModel(torch.nn.Module):
    def __init__(self,encoder,head):
        super().__init__(); self.encoder=encoder; self.head=head

    def forward(self,log_mel,valid_frames):
        hidden=self.encoder(log_mel).last_hidden_state
        return self.head(pool_hidden(hidden,valid_frames))


def ort_session(path):
    import onnxruntime as ort
    options=ort.SessionOptions()
    options.intra_op_num_threads=1
    options.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])


def export(run,manifest_path):
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    torch.set_num_threads(1)
    selection=json.loads((run/'selection.json').read_text())
    if selection['manifest_sha256'] != sha256(manifest_path):
        raise ValueError('Manifest differs from training')
    rows=json.loads(manifest_path.read_text())['rows']
    dev=[r for r in rows if r['split']=='dev']
    verify_audio_rows(dev)
    encoder,processor=load_encoder(selection['size'])
    model=EmotionModel(encoder,PortableHead(dict(np.load(run/'head.npz')))).eval()
    if (run/'finetuned.pt').exists():
        model.load_state_dict(torch.load(run/'finetuned.pt',map_location='cpu',weights_only=True))
    audio=load_audio(ROOT/dev[0]['audio_path'])
    mel,lengths=prepare_batch(processor,[audio])
    output=run/'export';output.mkdir(exist_ok=False)
    fp=output/'emotion-fp32.onnx';quant=output/'emotion-int8.onnx'
    torch.onnx.export(model,(mel,lengths),str(fp),opset_version=17,dynamo=False,
                      input_names=['log_mel','valid_frames'],output_names=['probabilities'],
                      dynamic_axes={'log_mel':{0:'batch'},'valid_frames':{0:'batch'},'probabilities':{0:'batch'}})
    onnx.checker.check_model(str(fp))
    quantize_dynamic(str(fp),str(quant),weight_type=QuantType.QInt8,per_channel=True,
                     op_types_to_quantize=['MatMul','Gemm'],extra_options={'MatMulConstBOnly':True})
    onnx.checker.check_model(str(quant))
    sessions={'float':ort_session(fp),'int8':ort_session(quant)}
    # Development only: quantization validation is model selection, not final test.
    predictions={k:[] for k in ['pytorch','float','int8']}
    for offset in range(0,len(dev),16):
        audios=[load_audio(ROOT/r['audio_path']) for r in dev[offset:offset+16]]
        x,l=prepare_batch(processor,audios)
        with torch.inference_mode(): predictions['pytorch'].append(model(x,l).numpy())
        for key,session in sessions.items():
            predictions[key].append(session.run(None,{'log_mel':x.numpy(),'valid_frames':l.numpy()})[0])
    predictions={k:np.concatenate(v) for k,v in predictions.items()}
    votes=np.array([r['votes'] for r in dev])
    timings={}
    feed={'log_mel':mel.numpy(),'valid_frames':lengths.numpy()}
    for key,session in sessions.items():
        for _ in range(5): session.run(None,feed)
        inference=[];end_to_end=[]
        for _ in range(30):
            start=time.perf_counter();session.run(None,feed);inference.append((time.perf_counter()-start)*1000)
            start=time.perf_counter();a=load_audio(ROOT/dev[0]['audio_path']);x,l=prepare_batch(processor,[a]);session.run(None,{'log_mel':x.numpy(),'valid_frames':l.numpy()});end_to_end.append((time.perf_counter()-start)*1000)
        timings[key]={'inference_p50_ms':float(np.median(inference)),'inference_p95_ms':float(np.quantile(inference,.95)),
            'wav_read_preprocess_inference_p50_ms':float(np.median(end_to_end)),
            'wav_read_preprocess_inference_p95_ms':float(np.quantile(end_to_end,.95))}
    float_error=float(np.max(np.abs(predictions['pytorch']-predictions['float'])))
    quant_error=float(np.max(np.abs(predictions['float']-predictions['int8'])))
    scores={k:metrics(votes,v) for k,v in predictions.items()}
    report=dict(device='Mac17,6 Apple Silicon CPU, one ORT thread; NOT phone measurement',
        sizes_bytes={'float':fp.stat().st_size,'int8':quant.stat().st_size},
        hashes={'float':sha256(fp),'int8':sha256(quant)},parameters=sum(p.numel() for p in model.parameters()),
        float_max_probability_error=float_error,int8_max_probability_error=quant_error,
        float_parity_passed=float_error<1e-4,
        quantization_gate_passed=scores['float']['macro_f1']-scores['int8']['macro_f1']<=.02,
        scores=scores,timings=timings,n_dev=len(dev),failures=0,
        real_phone_latency_verified=False,real_phone_battery_verified=False,
        observation_window_seconds=WINDOW_SECONDS)
    np.savez(output/'dev_predictions.npz',ids=np.array([r['id'] for r in dev]),**predictions)
    # Explicit frontend contract; the ONNX file is not a WAV decoder or microphone loop.
    np.savez(output/'frontend-constants.npz',mel_filters=processor.mel_filters,
             hann=np.hanning(401)[:-1].astype(np.float32))
    save_json(output/'input_contract.json',dict(labels=LABELS,sample_rate=16000,channels=1,
        waveform='float32 mono, no gain normalization; reject empty/nonfinite/silent input',
        pad_to_samples=WINDOW_SECONDS*16000,fft_size=400,hop_samples=160,
        window='periodic Hann',center_padding='reflect, 200 samples each side',
        power_spectrum=True,mel_bins=80,drop_last_stft_frame=True,
        log_mel='log10(max(mel_power,1e-10)); max(log_mel, per-example max - 8); (log_mel + 4) / 4',
        valid_frames='min(300, ceil(original_resampled_samples / 320))',
        input_shape=['batch',80,600],output_shape=['batch',6],
        note='Frontend must match model.py/WhisperFeatureExtractor. Encoder uses 6-second positional prefix. No silent truncation. Microphone collection delay is additional to inference.'))
    save_json(output/'validation.json',report)
    print(json.dumps(report,indent=2),flush=True)
    if not report['float_parity_passed'] or not report['quantization_gate_passed']:
        raise RuntimeError('Export validation failed; inspect saved evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/manifest-v2.json')
    a=p.parse_args();export(a.run,a.manifest)
