"""Local audio candidate probes; never admit their predictions as training truth."""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path


def select_rows(rows, limit):
    if limit < 1:
        raise ValueError('Positive development subset size required')
    groups=[]
    for label in range(6):
        speakers=defaultdict(list)
        for row in rows:
            if row['split']=='dev' and row['majority']==label:
                speakers[row.get('speaker',row['id'])].append(row)
        buckets=[speakers[key] for key in sorted(speakers)]
        groups.append([bucket[j] for j in range(max(map(len,buckets),default=0))
                       for bucket in buckets if j<len(bucket)])
    return [g[j] for j in range(max(map(len,groups),default=0)) for g in groups if j<len(g)][:limit]


def parse_answer(text):
    text=re.sub(r'<think>.*?</think>','',text,flags=re.S)
    if '<think>' in text or '</think>' in text:
        raise ValueError('Unfinished reasoning is not a final answer')
    if re.fullmatch(r'\s*[A-F]\s*',text):
        return ord(text.strip())-ord('A')
    answers=re.findall(r'<answer>\s*([A-F])\s*</answer>',text)
    if len(answers)!=1 or len(re.findall(r'<answer>',text))!=1:
        raise ValueError('Missing or ambiguous final answer; not a scored result')
    return ord(answers[0])-ord('A')


def run(args):
    import numpy as np
    import torch
    from .benchmark import ROOT, save_json, sha256, metrics, verify_audio_rows
    from .model import load_audio
    from .teacher import INSTRUCTION, choice_probabilities
    out=args.out
    if out.exists():raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True)
    started=time.monotonic()
    health=dict(completed=False,empty_outputs=0,finish_reason_length=0,invalid_outputs=0,runtime_errors=0)
    try:
        torch.set_num_threads(6)
        device='mps' if torch.backends.mps.is_available() else 'cpu'
        dtype=torch.bfloat16 if device=='mps' else torch.float32
        if args.float32:dtype=torch.float32
        rows=select_rows(json.loads(args.manifest.read_text())['rows'],args.limit)
        if not rows:raise ValueError('Empty development selection')
        verify_audio_rows(rows)
        if args.kind=='liquid':
            from transformers import AutoProcessor, AutoModelForMultimodalLM
            processor=AutoProcessor.from_pretrained(args.model,local_files_only=True,trust_remote_code=True,fix_mistral_regex=True)
            model=AutoModelForMultimodalLM.from_pretrained(args.model,local_files_only=True,
                trust_remote_code=True,dtype=dtype,attn_implementation='eager').eval().to(device)
            processor.audio_tokenizer.to(device)
        else:
            from transformers import Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniProcessor
            processor=Qwen2_5OmniProcessor.from_pretrained(args.model,local_files_only=True,fix_mistral_regex=True)
            model=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(args.model,local_files_only=True,
                torch_dtype=dtype,attn_implementation='eager').eval().to(device)
        choices=[]
        for letter in 'ABCDEF':
            tokens=processor.tokenizer.encode(letter,add_special_tokens=False)
            if len(tokens)!=1:raise ValueError('Letter is not a single token')
            choices.append(tokens[0])
        budget=40000
        prompt=INSTRUCTION if args.mode=='forward' else INSTRUCTION.replace('Reply with only the single letter A, B, C, D, E, or F.',
            'Put exactly one letter A, B, C, D, E, or F inside <answer>...</answer>.')
        if args.kind=='liquid':prompt=INSTRUCTION
        system_prompt=('A conversation between User and Assistant. The user asks a question, and the Assistant solves it. '
            'The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. '
            'The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, '
            'i.e., <think> reasoning process here </think><answer> answer here </answer>.') if args.kind=='emotionthinker' else None
        config=dict(kind=args.kind,revision=args.revision,model_path=str(args.model),mode=args.mode,
                    manifest_sha256=sha256(args.manifest),prompt=prompt,device=device,dtype=str(dtype),
                    model_config_sha256=sha256(args.model/'config.json'),
                    ids=[r['id'] for r in rows],split='development',max_new_tokens=budget if args.mode=='generate' else None,
                    generation_configuration='Reasoning generation budget 40000',system_prompt=system_prompt,
                    subset_strategy='round-robin emotion, then speakers within emotion',
                    repetition_penalty=args.repetition_penalty,
                    note='Forward mode is constrained next-token scoring, not native reasoning. No automatic training-label admission.')
        save_json(out/'config.json',config)
        records=[]
        for i,row in enumerate(rows):
            begin=time.monotonic()
            content=[{'type':'audio','audio':str((ROOT/row['audio_path']).resolve())},{'type':'text','text':prompt}]
            messages=[{'role':'user','content':content}]
            if system_prompt:messages.insert(0,{'role':'system','content':[{'type':'text','text':system_prompt}]})
            if args.kind=='liquid':
                inputs=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,
                        return_dict=True,return_tensors='pt').to(device)
            else:
                text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                inputs=processor(text=text,audio=[load_audio(ROOT/row['audio_path'])],return_tensors='pt',padding=True).to(device)
            for k,v in inputs.items():
                if torch.is_tensor(v) and v.is_floating_point():inputs[k]=v.to(dtype)
            if not any('feature' in k or 'audio' in k for k in inputs):
                raise ValueError('No audio features reached model')
            with torch.inference_mode():
                if args.mode=='forward':
                    logits=model(**inputs,use_cache=False).logits[0,-1].float().cpu().numpy()
                    p,mass=choice_probabilities(logits,choices)
                    response=dict(probabilities=p.tolist(),allowed_choice_mass=mass)
                else:
                    generation=dict(max_new_tokens=budget,do_sample=False,repetition_penalty=args.repetition_penalty)
                    if args.kind=='emotionthinker':
                        from transformers import TextStreamer
                        generation['streamer']=TextStreamer(processor.tokenizer,skip_prompt=True)
                        generation['eos_token_id']=[processor.tokenizer.eos_token_id,processor.tokenizer.pad_token_id]
                    if args.kind=='liquid':generation.update(suppress_mixed=True,suppress_audio_out=True)
                    output=model.generate(**inputs,**generation)
                    sequences=output.sequences if hasattr(output,'sequences') else output
                    tokens=sequences[0,inputs['input_ids'].shape[1]:]
                    text=processor.tokenizer.decode(tokens,skip_special_tokens=True)
                    save_json(out/f'{i:04d}-raw.json',dict(id=row['id'],text=text,n_generated_tokens=len(tokens)))
                    if len(tokens)>=budget:
                        health['finish_reason_length']+=1;raise ValueError('Truncated generation')
                    if not text.strip():
                        health['empty_outputs']+=1;raise ValueError('Empty generation')
                    try:
                        label=parse_answer(text)
                    except ValueError:
                        health['invalid_outputs']+=1
                        raise
                    response=dict(probabilities=np.eye(6)[label].tolist(),generated_text=text,
                                  note='One-hot decision, not calibrated probability')
            response.update(id=row['id'],audio_sha256=row['sha256'],seconds=time.monotonic()-begin,
                            label_source='machine',approved_for_training=False)
            save_json(out/f'{i:04d}.json',response);records.append(response)
            print(f'{args.kind} {args.mode}: {i+1}/{len(rows)} {response["seconds"]:.2f}s',flush=True)
        p=np.array([r['probabilities'] for r in records])
        np.savez(out/'predictions.npz',ids=np.array([r['id'] for r in rows]),probabilities=p)
        save_json(out/'summary.json',dict(config=config,metrics=metrics(np.array([r['votes'] for r in rows]),p),
                  n=len(rows),elapsed_seconds=time.monotonic()-started,
                  low_choice_mass_count=sum(r.get('allowed_choice_mass',1)<.1 for r in records),
                  qualified_teacher=False,note='Development probe only; no independent generalization or label-quality claim'))
        health['completed']=True
    except BaseException as exc:
        if not any(health[k] for k in ('empty_outputs','finish_reason_length','invalid_outputs')):
            health['runtime_errors']+=1
        health.update(error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        health['elapsed_seconds']=time.monotonic()-started
        save_json(out/'run_health.json',health)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['liquid','emotionthinker'],required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--revision',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int,default=12)
    p.add_argument('--mode',choices=['forward','generate'],default='forward')
    p.add_argument('--float32',action='store_true')
    p.add_argument('--repetition-penalty',type=float,default=1.0)
    p.add_argument('--manifest',type=Path,default=Path(__file__).parent/'data/manifest-v2.json')
    run(p.parse_args())
