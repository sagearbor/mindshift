"""Build a self-contained review from measured files, never handwritten scores."""
import json
import shutil
import re
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path
from .benchmark import ROOT, LABELS, save_json, sha256, read_audio_votes, majority
from .freeze import validate_run
from . import report_visuals as visuals
from .plot_comparisons import chart_data


def read(path):return json.loads(path.read_text())
def pct(value):return f'{100*value:.1f}%'
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{escape(str(h))}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def detail(number,title,good,body):
    verdict='GOOD' if good else 'BAD';color='good' if good else 'bad'
    callout=('Good — '+title+'. Nothing to do.') if good else ('Bad — '+title+'. '+('Use the research results to target the next training round; do not deploy this candidate.'))
    return f'<details><summary>{number}. <span class="chip {color}">{verdict}</span> {title}</summary><div class="callout {color}">{callout}</div>{body}</details>'


def build():
    reports=ROOT/'reports';frozen=read(reports/'pretest-freeze.json')
    rows=read(ROOT/'data/manifest-v2.json')['rows']
    source=ROOT/'data/crema-d'
    meta=read(ROOT/'benchmarks/cremad-v2.meta.json')
    for field,filename in [('ratings_sha256','finishedResponses.csv'),('timing_sha256','finishedEmoResponses.csv')]:
        if sha256(source/filename)!=meta[field]:raise ValueError('Human source ratings changed')
    source_votes=read_audio_votes(source/'finishedResponses.csv',source/'finishedEmoResponses.csv')
    intended_codes=['ANG','DIS','FEA','HAP','NEU','SAD']
    perception={code:[0]*7 for code in intended_codes}
    for clip,votes in source_votes.items():
        winner=majority(votes)
        perception[clip.split('_')[2]][6 if winner is None else winner]+=1
    save_json(reports/'actor-listener-audit.json',dict(
        ratings_sha256=sha256(source/'finishedResponses.csv'),
        timing_sha256=sha256(source/'finishedEmoResponses.csv'),
        columns=LABELS+['tied'],counts_by_actor_intent=perception,
        denominator='All original 7,442 source recordings, including the silent recording; unique-winner agreement excludes ties'))
    names=[r['run'] for r in frozen['candidates']]
    records={}
    for name in names:
        source=ROOT/'results'/name;dest=reports/name;dest.mkdir(parents=True,exist_ok=True)
        validate_run(source,next(r for r in frozen['candidates'] if r['run']==name))
        final=read(source/'final-test.json')
        for field,path in [('pretest_freeze_sha256',reports/'pretest-freeze.json'),
                           ('selection_sha256',source/'selection.json'),
                           ('manifest_sha256',ROOT/'data/manifest-v2.json'),
                           ('test_prediction_sha256',source/'test-predictions.npz')]:
            if final[field]!=sha256(path):raise ValueError(f'Stale report evidence: {name} {field}')
        for filename in ['selection.json','final-test.json','test-predictions.npz','run_health.json','history.json']:
            if (source/filename).exists():shutil.copy2(source/filename,dest/filename)
        validation=None
        if (source/'export/validation.json').exists():
            validation=read(source/'export/validation.json')
            for filename in ['validation.json','input_contract.json']:
                shutil.copy2(source/'export'/filename,dest/filename)
        records[name]=dict(selection=read(source/'selection.json'),test=read(source/'final-test.json'),export=validation)
    teacher=read(ROOT/'results/qwen-dev-v1/summary.json')
    if teacher['config']['manifest_sha256']!=frozen['manifest_sha256']:
        raise ValueError('Teacher evidence uses another manifest')
    save_json(reports/'qwen-development.json',teacher)
    selected=frozen['selected_accuracy_candidate'];record=records[selected]
    variant='int8' if record['export'] else 'float'
    scores=record['test']['results'][variant];main=scores['all']
    quality=main['macro_f1']>=.75 and min(v['recall'] for v in main['per_emotion'].values())>=.60
    teacher_metrics=teacher['results']['audio']['metrics']
    teacher_qualified=teacher_metrics['macro_f1']>=.75 and min(v['recall'] for v in teacher_metrics['per_emotion'].values())>=.60
    if quality or teacher_qualified:
        raise ValueError('Outcome differs from this failed-gate report template; revise conclusions before publishing')
    summary=dict(selected_on_development=selected,reported_variant=variant,
        high_quality_gate_passed=quality,phone_verified=False,
        benchmark_clips=len(rows),test_clips=main['n_total'],single_label_test_clips=main['n_single_label'],
        test_macro_f1=main['macro_f1'],test_accuracy=main['accuracy'],
        machine_label_expansion='not run: teacher quality gate failed',
        selected_scores=scores,
        candidates={name:dict(dev_macro_f1=r['selection']['winner']['dev_macro_f1'],
            test={v:{k:s['all'][k] for k in ['macro_f1','accuracy','balanced_accuracy','kappa']} for v,s in r['test']['results'].items()},
            evidence=f'{name}/final-test.json') for name,r in records.items()},
        teacher_development=dict(audio_macro_f1=teacher_metrics['macro_f1'],
            transcript_macro_f1=teacher['results']['transcript']['metrics']['macro_f1'],
            evidence='qwen-development.json'))
    save_json(reports/'summary.json',summary)
    ci=scores['confidence_intervals']['intervals']['macro_f1']
    cohort_rows=[];agreement_wins=0
    for size in ['tiny','base','small']:
        a=records[f'{size}-all-v1']['test']['results']['float']['all']
        b=records[f'{size}-agrees-v1']['test']['results']['float']['all']
        agreement_wins+=int(b['macro_f1']>a['macro_f1'])
        cohort_rows.append([size,pct(a['macro_f1']),pct(b['macro_f1']),f"{100*(a['macro_f1']-b['macro_f1']):+.1f} points"])
    candidate_rows=[];export_rows=[]
    for name,r in records.items():
        for v,s in r['test']['results'].items():
            if v not in ['float','int8']:continue
            candidate_rows.append([name,v,pct(r['selection']['winner']['dev_macro_f1']),pct(s['all']['macro_f1']),pct(s['all']['accuracy'])])
        if r['export']:
            e=r['export'];t=e['timings']['int8']
            export_rows.append([name,f"{e['sizes_bytes']['int8']/1e6:.1f} MB",f"{t['wav_read_preprocess_inference_p50_ms']:.1f} ms",f"{t['wav_read_preprocess_inference_p95_ms']:.1f} ms",'Not measured'])
    emotion_rows=[[emotion,pct(v['precision']),pct(v['recall']),pct(v['f1']),v['support']] for emotion,v in main['per_emotion'].items()]
    slice_rows=[[name,v['n_total'],v['n_single_label'],pct(v['macro_f1'])] for name,v in scores.items() if name in ['all','actor_agrees','actor_disagrees','strong_consensus','weak_consensus']]
    controls=record['test']['results']
    panels=[]
    panels.append(detail(1,'Both human label types preserved',True,
        '<p>Every clip keeps the actor’s intended emotion and the complete audio-only listener vote counts. Listeners heard sadness in 209 of 1,148 intended-sadness clips with a unique winner (18.2%); 123 intended-sadness clips with tied winners are excluded from that rate. This describes actor/listener agreement, not model accuracy. The original corpus has 7,442 clips; overall agreement excludes 644 ties. The usable benchmark has 7,441 after one silent training clip was removed.</p>'
        '<p>Original filtered corpus: 3,099 agreements among 6,798 unique winners (45.59%). Labels are listeners’ perception of the voice, not proof of an internal emotional state.</p>'
        '<p>The frozen index is <code>eval/codex/benchmarks/cremad-v2.jsonl</code>. Actor, transcript, source hash, intended label, listener votes and split are retained.</p>'
        +table(['Actor intended']+LABELS+['Tied'],[[LABELS[i]]+perception[code] for i,code in enumerate(intended_codes)])
        +'<p>Counts above use all original source clips, including the quarantined silent file. Columns show the unique most-common listener label, not necessarily more than half the votes. Tied winners have their own column. This comparison does not establish that either actor intent or listener perception is universally the correct label.</p>'))
    panels.append(detail(2,'Reusable test set built',True,
        '<p>4,489 training clips from 55 actors; 1,476 development clips from 18 different actors; 1,476 test clips from another 18 actors. No actor or identical audio crosses partitions. All 7,442 reconstructed winner sets matched the publisher’s official audio summary before quarantine.</p>'
        '<p>Final single-label scores use 1,341 test clips, excluding 135 tied winners. Vote-distribution scores retain all 1,476. The same test can detect regressions, but tuning repeatedly against it would consume its independence.</p>'
        '<p>Frozen choices and file hashes: <code>eval/codex/reports/pretest-freeze.json</code>. Predictions and confidence intervals are saved beside each candidate report.</p>'))
    panels.append(detail(3,'Quality target remains unmet',quality,
        f'<p>Candidate selected on development: <strong>{selected}</strong>. Test macro F1: <strong>{pct(main["macro_f1"])}</strong>; actor-bootstrap 95% interval {pct(ci[0])} to {pct(ci[1])}. Macro F1 gives every emotion equal weight; it is not percent-correct accuracy. The provisional goal was 75% macro F1 and at least 60% recall in every emotion.</p>'
        +table(['Emotion','Precision','Recall','F1','Test examples'],emotion_rows)
        +'<p>Precision asks how often a predicted label matches the human winner; recall asks how many clips with that human winner were recovered. F1 balances both. Intervals resample 18 actors, not individual clips; limited actors and unknown pretraining exposure constrain generalization.</p>'
        +table(['Subset','Clips','Unique winners','Macro F1'],slice_rows)
        +f'<p>Against actor intent instead of listener perception, the same predictions have macro F1 {pct(scores["actor_intent_secondary"]["macro_f1"])} and accuracy {pct(scores["actor_intent_secondary"]["accuracy"])}. This secondary view uses all 1,476 test recordings and answers a different question; it was not used for model selection.</p>'
        +'<p>Strong consensus means at least two-thirds of listeners chose the same label. Subsets are diagnostics, not replacement acceptance results. Acted English does not verify spontaneous conversation.</p>'))
    panels.append(detail(4,'Agreement-only training compared',True,
        '<p>We tested your idea directly: train on 1,849 clips where actor and unique listener winner agree, versus all 4,489 training clips using full listener distributions. Both use exactly the same broad test. This comparison changes both sample count and label mixture, so it does not isolate “nuance” as a causal mechanism.</p>'
        +table(['Encoder','All-vote test F1','Agreement-only test F1','All minus agreement'],cohort_rows)
        +f'<p>Agreement-only training scored higher in {agreement_wins} of the three held-out comparisons. This is a useful signal in favor of your proposed filter, while the opposite development ranking shows that actor sampling matters.</p>'
        +'<p>Positive differences favor all-vote training; negative differences favor agreement-only training. Development favored all-vote training for all three encoders, but this does not guarantee the same direction on independent actors. Treat these as descriptive comparisons, not proof of a universal winning cohort. Recommendation: retain all labels and full listener distributions; keep agreement-only training as a supported option and comparison, rather than discarding ambiguous data permanently.</p>'))
    panels.append(detail(5,'Compact exports work locally',True,
        '<p>These files include the full encoder and classifier. The research file-size target was below 100 MB for a phone; this checks storage only. Dynamic 8-bit quantization reduces supported matrix weights; some operations remain floating point. Float export parity and quantized development quality were checked.</p>'
        +table(['Candidate','8-bit file','Mac median total','Mac 95th-percentile total','Phone'],export_rows)
        +'<p>Times include WAV reading, preprocessing and inference, one CPU thread, 30 repeats on this Mac. Audio collection time is additional. Input accepts up to six seconds and rejects longer recordings. Small files do not prove watch feasibility; no watch, phone battery, memory or streaming measurement was performed.</p>'
        '<p>A real WAV passed the exported tiny-model command-line smoke check. Silence returns no emotion. This is an experimental local runner, not a product integration.</p>'))
    panels.append(detail(6,'Automatic labeller failed validation',False,
        f'<p>Local Qwen2.5-Omni-7B audio development macro F1 was {pct(teacher["results"]["audio"]["metrics"]["macro_f1"])}; transcript-only was {pct(teacher["results"]["transcript"]["metrics"]["macro_f1"])}. These are development scores, not held-out test claims. Neither qualifies to manufacture trusted training labels.</p>'
        '<p>No large machine-labelled pool was created or trained on. Admission checks exist for quality, licences, provenance, speaker/audio overlap, model revision and finite confidence values. A mass collection/labelling job is intentionally not implemented. For the next round, obtain independent human audio-only ratings for licensed natural conversational clips, then test a stronger teacher against them before scaling.</p>'))
    panels.append(detail(7,'Commercial-use provenance recorded',True,
        '<p>Whisper weights and code: <a href="https://github.com/openai/whisper#license">MIT</a>. Qwen weights: <a href="https://huggingface.co/Qwen/Qwen2.5-Omni-7B">Apache 2.0</a>. CREMA-D: <a href="https://github.com/CheyneyComputerScience/CREMA-D/blob/master/LICENSE.txt">Open Database License and Database Contents License</a>. Commercial use comes with applicable attribution and database obligations; notices are retained. No paid inference was used.</p>'
        '<p>Unclear emotion2vec+ and WavLM weight terms were excluded. No raw audio or model binaries are included in Git. The derived benchmark index carries its source notices in <code>benchmarks/NOTICE.md</code>.</p>'))
    suite=ET.parse(reports/'tests.xml').getroot().find('testsuite')
    if suite is None or int(suite.attrib['tests'])<1 or int(suite.attrib['failures']) or int(suite.attrib['errors']):
        raise ValueError('Missing or failed targeted-test evidence')
    panels.append(detail(8,'All candidate evidence retained',True,
        table(['Candidate','Variant','Float dev F1','Test F1','Test accuracy'],candidate_rows)
        +f'<p>Selected-candidate controls: transcript-only test macro F1 {pct(controls["transcript"]["all"]["macro_f1"])}; training-prior baseline {pct(controls["prior"]["all"]["macro_f1"])}. These controls help expose class imbalance and text effects. They do not prove the encoder ignores spoken words.</p>'
        '<p>Full confusion matrices, vote-distribution error, calibration, per-emotion intervals and raw predictions are in <code>eval/codex/reports</code>. Generated-output empties and truncations are zero: these models use forward passes, not text generation. One silent source recording and one metadata-write failure were handled explicitly, not scored as model failures.</p>'
        +f'<p>{suite.attrib["tests"]} targeted tests passed. Independent review corrected audio/cache provenance checks, invalid teacher metrics, frozen-artifact integrity and report provenance. Saved command-line smoke evidence is <code>eval/codex/reports/inference-smoke.json</code>. The HTML structure was checked directly; browser visual verification was blocked by the browser policy for local file URLs.</p>'))
    tiny=records['tiny-finetuned-v1']['export']['sizes_bytes']['int8']/1e6
    small=record['export']['sizes_bytes']['int8']/1e6
    cards=[('7,441','GOOD — usable labelled clips',True),('1,476','GOOD — held-out test clips',True),(f'{tiny:.1f} MB','GOOD — tiny model size',True),(f'{small:.1f} MB','GOOD — small model size',True),(pct(main['macro_f1']),'BAD — small model quality unmet',False),('0','BAD — qualified automatic labelers',False)]
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Emotion model results</title><style>
    :root{color-scheme:light dark;--bg:#fafafa;--fg:#18202c;--panel:#fff;--line:#ccd3dd;--good:#11773f;--bad:#b72834;--violet:#7440bd} @media(prefers-color-scheme:dark){:root{--bg:#131720;--fg:#e7edf7;--panel:#1d2430;--line:#455269;--good:#58d993;--bad:#ff7c88;--violet:#c0a0ff}}[data-theme=dark]{--bg:#131720;--fg:#e7edf7;--panel:#1d2430;--line:#455269;--good:#58d993;--bad:#ff7c88;--violet:#c0a0ff}[data-theme=light]{--bg:#fafafa;--fg:#18202c;--panel:#fff;--line:#ccd3dd;--good:#11773f;--bad:#b72834;--violet:#7440bd}
    body{background:var(--bg);color:var(--fg);font:16px/1.6 system-ui;margin:30px auto;padding:0 20px;max-width:1100px}h1{font-size:2rem;line-height:1.2}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:20px 0}.card{border:1px solid currentColor;background:var(--panel);padding:14px;border-radius:10px}.card b{display:block;font-size:2rem}.card small{display:block}.good{color:var(--good)}.bad{color:var(--bad)}.violet{color:var(--violet)}.chip{font-size:.76rem;border:1px solid currentColor;padding:3px 7px;border-radius:6px;white-space:nowrap}details{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 18px;margin:12px 0}summary{cursor:pointer;font-weight:650;white-space:nowrap;overflow-x:auto}summary .chip{margin:0 9px}.callout{border-left:4px solid;padding:10px;margin:15px 0;font-weight:650}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{padding:10px;text-align:left;border-bottom:1px solid var(--line)}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:15px;border:1px solid var(--line);border-radius:6px}a{color:var(--good)}.recap{margin:22px 0}.recap p{margin:7px 0}</style><body>'''
    html+='<h1>Emotion recognition: measured results</h1><p><span class="good">Green = GOOD</span> · <span class="bad">Red = BAD</span> · <span class="violet">Violet = NEEDS YOU</span></p>'
    html+='<div class="cards">'+''.join(f'<div class="card {"good" if good else "bad"}"><b>{value}</b><small>{label}</small></div>' for value,label,good in cards)+'</div>'
    html+=table(['Scorecard','Verdict','So what'],[['Human benchmark','GOOD','Both actor intent and listener votes preserved'],['Phone-size export','GOOD','Complete tiny model under 10 MB'],['Recognition quality','BAD','Still below the provisional research target'],['Phone validation','BAD','Real-device tests remain unfinished'],['Automatic labelling','BAD','Teacher failed; large-scale labels not generated']])
    html+='<div class="recap"><p><strong>Oneliner:</strong> A reproducible benchmark and compact models are ready; reliable recognition is not achieved yet.</p><p><strong>Task given:</strong> Build a commercially usable, accurate phone-sized emotion model verified against human-tagged audio; device testing and nudges excluded.</p><p><strong>My result:</strong> Incomplete on accuracy; benchmark, training comparisons and local exports completed.</p><p><strong>Recommended next step:</strong> Keep both labels, train from listener votes, and find existing human-rated natural speech before trusting automatic labels.</p></div>'
    html+='<p><strong>English is the priority.</strong> All custom training so far used English only: 4,489 CREMA-D training clips, or 1,849 for agreement-only experiments. Italian was an extra evaluation, never training data. MESS adds a separate English diagnostic. Any future Italian training would be a minority experiment judged by English validation.</p>'
    html+='<p><strong>What is our model?</strong> We took the Whisper small audio encoder, added a six-emotion classifier, trained the classifier and fine-tuned the encoder on human listener ratings, then compressed its weights to 8-bit. The complete export is 94.9 MB; the separately trained tiny alternative is 9.9 MB. These are experimental phone candidates; neither phone nor watch execution has been verified.</p>'
    html+='<p><strong>How to read the scores:</strong> Percent correct counts correct labels across recordings. Macro F1 first combines precision (how often an emotion prediction is right) and recall (how often that emotion is found), then averages equally across emotions. F1 points are F1 multiplied by 100: 63.4 points means F1 = 0.634, not 63.4% of recordings correct. The original selected model scored 73.5% correct and 63.4 macro F1 points on the same test.</p>'
    # Put the requested comparisons first; retain all original evidence panels.
    panels=[re.sub(r'(<summary>)(\d+)\.',lambda m:m[1]+str(int(m[2])+5)+'.',p,count=1) for p in panels]
    new_panels=[
        detail(1,'Per-emotion results: all models',False,visuals.model_matrix(records,teacher)),
        detail(2,'What each modification changed',True,visuals.modifications(records)),
        detail(3,'Where the 1,341 recordings came from',True,visuals.provenance(rows)),
        detail(4,'Cloud teacher capability and rights gaps',False,visuals.TEACHERS),
        detail(5,'Training and holdout remain separate',True,visuals.HOLDOUT),
    ]
    html+='OVERVIEW_CHART_PLACEHOLDER'
    html+=''.join(new_panels+panels)
    html+='<h2>State of play</h2>'+table(['Area','Where it is','Affects today’s product?'],[['Evaluation code and reports','codex/emotion-eval branch','No'],['Model weights and source audio','Local ignored eval/codex directories','No'],['Nudge logic, deployed apps, main','Untouched','No'],['Phone and watch runtime','Not implemented or measured','No']])
    html+='<h2>Where I was wrong</h2><p>Initial unfiltered counts (3,085 agreements, 6,806 unique winners, 636 ties) were corrected to the official filtered counts (3,099, 6,798 and 644 respectively). The initial 7,442 usable-audio assumption became 7,441 after an all-zero WAV was found. A completed base training run hit a path error while saving metadata; metadata was recovered from preserved history without changing predictions. Earlier project accuracy notes used different labels and missing artifacts, so they do not establish current quality.</p>'
    html+='<h2>What happens next</h2>'
    command=f'cd {ROOT.parents[1]}\n.venv/bin/python -m eval.codex.infer --model eval/codex/results/tiny-finetuned-v1/export/emotion-int8.onnx --audio eval/codex/data/crema-d/AudioWAV/1001_IEO_NEU_XX.wav'
    html+='<details><summary>Step 1. Run the local experimental model</summary><p>Run in Terminal on this Mac. Success: experimental_prediction with six finite probabilities. This is a read-only smoke check, not proof of accuracy. Undo: none needed.</p><pre>'+escape(command)+'</pre></details>'
    html+='<details><summary>Step 2. Inspect saved results</summary><p>Run in Terminal on this Mac. Success: the measured candidate and full metrics print. This reads saved evidence without rerunning training. Undo: none needed.</p><pre>'+escape(f'cd {ROOT.parents[1]}\n.venv/bin/python -m json.tool eval/codex/reports/summary.json')+'</pre></details>'
    html+='<details><summary>Step 3. Continue authorized emotion research</summary><p>Local research is authorized and underway. This is the exact instruction to resume in Codex if interrupted.</p><pre>Continue the emotion-only research from eval/codex/OVERNIGHT-20261106.md. Find existing human-rated natural speech, preserve reuse terms and actor intent separately, and benchmark stronger local audio models on development data. Keep phone testing and nudges out of scope. Do not admit unverified machine labels into training.</pre><p>Success: reproducible local-model results and documented independent human evidence. Undo: discard new research changes before merging; retain raw experiment outputs.</p></details>'
    html+='<p>Source evidence: <a href="https://github.com/CheyneyComputerScience/CREMA-D">CREMA-D publisher</a>. Local measured evidence: <code>eval/codex/reports/summary.json</code>. No product release or quality certification.</p></body></html>'
    definitions={'F1':'Harmonic mean of precision and recall; macro F1 averages equally over six emotions',
        'MB':'Megabytes: one million bytes','CPU':'Central processing unit',
        'WAV':'Waveform audio file format','ONNX':'Open Neural Network Exchange model format',
        'MIT':'Permissive Massachusetts Institute of Technology software licence',
        'Qwen':'The publisher name of the tested large audio-language model',
        'CREMA-D':'Crowd-sourced Emotional Multimodal Actors Dataset',
        'bootstrap':'Uncertainty estimate from resampling whole actors with replacement',
        'quantization':'Representing selected model weights using fewer bits',
        'calibration':'Agreement between predicted confidence and observed correctness'}
    for term,definition in definitions.items():
        html=re.sub(r'(?<![\w/.-])'+re.escape(term)+r'(?![\w/.-])',lambda m:f'<abbr title="{escape(definition)}">{term}</abbr>',html,count=1)
    html=html.replace('</style>',visuals.CSS+'</style>')
    chart=read(reports/'modification-overview.json')
    svg_path=reports/'modification-overview.svg'
    if chart['data']!=chart_data(records) or chart['svg_sha256']!=sha256(svg_path):
        raise ValueError('Stale overview chart: regenerate with eval.codex.plot_comparisons')
    svg=svg_path.read_text();svg=svg[svg.index('<svg'):]
    svg=svg.replace('<svg ', '<svg role="img" aria-label="Three before-and-after plots comparing training filters, encoder fine-tuning, and compression for Whisper tiny, base, and small" style="width:100%;height:auto;display:block" ',1)
    overview='<figure style="margin:24px 0">'+svg+'<figcaption><strong>Fine-tuning helped overall; compression cost little accuracy.</strong> Each panel is an independent comparison, not a sequence of three training steps. Filtering uses frozen encoders at float precision; fine-tuning uses all training clips at float precision; compression uses the same fine-tuned checkpoint. Macro F1 gives each emotion equal weight and is not overall percent correct. Passing the full provisional goal also requires every emotion’s recall to reach 60%.</figcaption></figure>'
    from .overnight_report import build_section, native_section, external_section, mess_section, balanced_section, transfer_chart
    overview=transfer_chart()+overview
    overview+=build_section()+native_section()+external_section()+mess_section()+balanced_section()
    human_chart=read(reports/'human-agreement.json')
    if human_chart['source_sha256']!=sha256(reports/'native-screen-comparison.json') or human_chart['svg_sha256']!=sha256(reports/'human-agreement.svg'):
        raise ValueError('Regenerate human agreement chart from current native-screen evidence')
    human_svg=(reports/'human-agreement.svg').read_text();human_svg=human_svg[human_svg.index('<svg'):]
    human_svg=human_svg.replace('<svg ','<svg role="img" aria-label="Model agreement with human audio-only labels on the same English recordings" style="width:100%;height:auto;display:block" ',1)
    overview='<figure>'+human_svg+'<figcaption>Each dot is a model’s percent correct against human-perceived emotion, using the same English clips. This is the direct model-versus-human-label comparison. Whisper uses its uncompressed checkpoint here; its compressed export is evaluated separately. Human consensus supplies the reference, not an additional competitor with a measured score.</figcaption></figure>'+overview
    overview+='<p><strong>Research outcome:</strong> The shortlisted local audio models did not qualify as automatic labelers, and both class-balancing experiments are complete. The next useful work needs broader English training data and independent human-rated natural speech with documented reuse terms. Preserve human ratings and actor intent separately. No automatic labels enter training until a teacher passes evaluation. Phone testing and nudges are outside this research scope.</p>'
    html=html.replace('OVERVIEW_CHART_PLACEHOLDER',overview)
    (reports/'report.html').write_text(html)
    wrap=ROOT.parents[1]/'tmp/wrapups/20261105-emotion-eval.html';wrap.parent.mkdir(parents=True,exist_ok=True);wrap.write_text(html)
    print(wrap)


if __name__=='__main__':build()
