"""Native MLX audio generation with development-only scoring and fail-closed output."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .benchmark import ROOT, sha256, save_json, metrics, verify_audio_rows
from .candidate_probe import select_rows, parse_answer


def run(args):
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config
    from mlx_vlm.prompt_utils import apply_chat_template
    if args.out.exists():raise FileExistsError('Fresh output directory required')
    args.out.mkdir(parents=True)
    health=dict(completed=False,empty_outputs=0,finish_reason_length=0,invalid_outputs=0,runtime_errors=0)
    start=time.monotonic()
    try:
        manifest=ROOT/'data/manifest-v2.json'
        rows=select_rows(json.loads(manifest.read_text())['rows'],args.limit)
        verify_audio_rows(rows)
        model,processor=load(str(args.model))
        config=load_config(str(args.model))
        instruction="Classify the emotion perceived in the speaker's voice, not the emotion suggested by the words. Choose exactly one label: A = anger; B = disgust; C = fear; D = happiness; E = neutral; F = sadness. Put exactly one letter A, B, C, D, E, or F inside <answer>...</answer>."
        prompt=apply_chat_template(processor,config,instruction,num_images=0,num_audios=1)
        metadata=dict(revision=args.revision,model_path=str(args.model),manifest_sha256=sha256(manifest),
            model_config_sha256=sha256(args.model/'config.json'),ids=[r['id'] for r in rows],
            prompt=prompt,split='development',max_tokens=40000,temperature=0,precision='mlx-community 4-bit conversion',
            subset_strategy='round-robin emotion, then speakers within emotion',
            note='Machine predictions, never approved automatically as training labels')
        save_json(args.out/'config.json',metadata)
        predictions=[]
        for i,row in enumerate(rows):
            begin=time.monotonic()
            result=generate(model,processor,prompt,audio=[str(ROOT/row['audio_path'])],max_tokens=40000,temperature=0,verbose=False)
            raw=dict(id=row['id'],audio_sha256=row['sha256'],text=result.text,
                generated_tokens=result.generation_tokens,finish_reason=result.finish_reason,
                seconds=time.monotonic()-begin,peak_memory_gb=result.peak_memory,approved_for_training=False)
            save_json(args.out/f'{i:04d}-raw.json',raw)
            if not result.text.strip():
                health['empty_outputs']+=1;raise ValueError('Empty response')
            if result.finish_reason=='length' or result.generation_tokens>=40000:
                health['finish_reason_length']+=1;raise ValueError('Truncated response')
            try:label=parse_answer(result.text)
            except ValueError:
                health['invalid_outputs']+=1;raise
            predictions.append(np.eye(6)[label])
            print(f'Qwen3 MLX {i+1}/{len(rows)}: {raw["seconds"]:.2f}s',flush=True)
        probabilities=np.array(predictions)
        np.savez(args.out/'predictions.npz',ids=np.array([r['id'] for r in rows]),probabilities=probabilities)
        save_json(args.out/'summary.json',dict(config=metadata,n=len(rows),
            metrics=metrics(np.array([r['votes'] for r in rows]),probabilities),qualified_teacher=False,
            note='Development screening only. One-hot predictions are not calibrated probabilities.'))
        health['completed']=True
    except Exception as exc:
        if not any(health[k] for k in ['empty_outputs','finish_reason_length','invalid_outputs']):health['runtime_errors']+=1
        health.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        health['elapsed_seconds']=time.monotonic()-start
        save_json(args.out/'run_health.json',health)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True);p.add_argument('--revision',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int,default=12)
    run(p.parse_args())
