"""Append new regression evidence without rewriting the original frozen experiment."""
import json
import shutil
import numpy as np
from html import escape
from .benchmark import ROOT, sha256, save_json, metrics
from .freeze import validate_run


def build_section():
    reports=ROOT/'reports'
    protocol=reports/'agreement-v2-freeze.json'
    freeze=json.loads(protocol.read_text())
    evidence=[]
    for candidate in freeze['candidates']:
        name=candidate['run']; source=ROOT/'results'/name
        validate_run(source,candidate)
        result=json.loads((source/'final-test.json').read_text())
        for key,path in [('pretest_freeze_sha256',protocol),('selection_sha256',source/'selection.json'),
                         ('manifest_sha256',ROOT/'data/manifest-v2.json'),('test_prediction_sha256',source/'test-predictions.npz')]:
            if result[key]!=sha256(path):raise ValueError(f'Stale regression evidence: {name} {key}')
        dest=reports/name;dest.mkdir(exist_ok=True)
        for filename in ['selection.json','final-test.json','test-predictions.npz','run_health.json','history.json']:
            shutil.copy2(source/filename,dest/filename)
        size=name.split('-')[0]
        baseline=ROOT/'results'/f'{size}-finetuned-v1'
        olddev=json.loads((baseline/'selection.json').read_text())['winner']['dev_macro_f1']
        oldtest=json.loads((baseline/'final-test.json').read_text())['results']['float']['all']['macro_f1']
        evidence.append(dict(size=size,development=[olddev,candidate['dev_macro_f1']],
                             regression=[oldtest,result['results']['float']['all']['macro_f1']]))
    save_json(reports/'agreement-v2-comparison.json',dict(rows=evidence,
        caveat='Previously inspected test; regression only, not independent confirmation. Selection remains development based.'))
    svg=['<svg viewBox="0 0 900 340" role="img" aria-label="All-clip versus agreement-only fine-tuning: development worsens for all sizes; old-test results are mixed" style="width:100%;background:#fff;color:#172033;border-radius:12px">']
    colors=['#166534','#7c3aed','#0369a1']
    for panel,key in enumerate(['development','regression']):
        left=65+panel*450
        svg.append(f'<text x="{left}" y="25" fill="#172033" font-size="17">{("Development: all three worse" if panel==0 else "Old test: mixed; not fresh evidence")}</text>')
        for tick in [.50,.55,.60,.65]:
            y=275-(tick-.48)*1100
            svg.append(f'<line x1="{left}" x2="{left+310}" y1="{y}" y2="{y}" stroke="#ddd"/><text x="{left-40}" y="{y+5}" fill="#333" font-size="12">{tick*100:.0f}</text>')
        for i,row in enumerate(evidence):
            a,b=row[key];x1=left+30;x2=left+250;y1=275-(a-.48)*1100;y2=275-(b-.48)*1100
            svg.append(f'<line x1="{x1}" x2="{x2}" y1="{y1}" y2="{y2}" stroke="{colors[i]}" stroke-width="3"/>')
            for x,y,value in [(x1,y1,a),(x2,y2,b)]:
                svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{colors[i]}"><title>{row["size"]}: {value*100:.2f} macro F1 points</title></circle>')
            svg.append(f'<text x="{x2+9}" y="{y2+4}" fill="{colors[i]}" font-size="12">{row["size"]}</text>')
        svg.append(f'<text x="{left}" y="310" fill="#333">All training clips</text><text x="{left+205}" y="310" fill="#333">Agreement only</text>')
    svg.append('</svg>')
    table='<div class="scroll"><table><tr><th>Size</th><th>Development: all → agreement</th><th>Old test: all → agreement</th></tr>'
    for row in evidence:
        table+='<tr><td>'+escape(row['size'])+'</td>'+''.join(f'<td>{row[k][0]*100:.2f} → {row[k][1]*100:.2f}</td>' for k in ['development','regression'])+'</tr>'
    table+='</table></div>'
    return '<section><h2>Overnight update: agreement-only fine-tuning</h2><p><strong>Mixed result — no replacement selected.</strong> Development quality fell for all three sizes. Tiny and base improved on the previously inspected test; small worsened. That old test helped motivate this experiment, so it cannot independently confirm the gains.</p>'+''.join(svg)+table+'<p>Values are macro F1 points, equally weighting six emotions, not percent correct. Development has 1,353 uniquely rated clips; old test has 1,341. Respectively 123 and 135 tied clips are excluded from these single-label scores. All models use float precision here. Existing all-clip selection remains in place.</p></section>'


def native_section():
    runs=[('Qwen3-Omni MLX 4-bit','qwen3-mlx-dev120-speakers-v2'),
          ('Liquid LFM3-Audio-2B','liquid-dev120-speakers-v2'),
          ('EmotionThinker: float32, repetition penalty 1.1','emotionthinker-dev120-fp32-v1'),
          ('MERaLiON-SER: training overlaps this corpus','meralion-dev120-v1')]
    manifest=ROOT/'data/manifest-v2.json'
    rows={r['id']:r for r in json.loads(manifest.read_text())['rows']}
    entries=[];common_ids=None
    for label,name in runs:
        source=ROOT/'results'/name
        if not (source/'summary.json').exists():continue
        health=json.loads((source/'run_health.json').read_text())
        if not health['completed']:continue
        summary=json.loads((source/'summary.json').read_text())
        if summary['config']['manifest_sha256']!=sha256(manifest):raise ValueError('Stale native model manifest')
        predictions=np.load(source/'predictions.npz');ids=predictions['ids'].tolist()
        if any(rows[i]['split']!='dev' for i in ids):raise ValueError('Not development')
        if common_ids is not None and ids!=common_ids:raise ValueError('Different screening cohorts')
        common_ids=ids
        if 'probabilities7' in predictions:
            from .specialist_probe import screen_metrics
            measured=screen_metrics([rows[i]['majority'] for i in ids],predictions['probabilities7'])
        else:
            measured=metrics(np.array([rows[i]['votes'] for i in ids]),predictions['probabilities'])
        if abs(measured['macro_f1']-summary['metrics']['macro_f1'])>1e-10:raise ValueError('Stale native metric')
        dest=ROOT/'reports'/name;dest.mkdir(exist_ok=True)
        for path in source.iterdir():
            if path.suffix in ['.json','.npz']:shutil.copy2(path,dest/path.name)
        entries.append(dict(model=label,metrics=measured))
    if not common_ids:return ''
    baseline=np.load(ROOT/'results/small-finetuned-v1/dev_predictions.npz')
    lookup={str(i):p for i,p in zip(baseline['ids'],baseline['probabilities'])}
    measured=metrics(np.array([rows[i]['votes'] for i in common_ids]),np.array([lookup[i] for i in common_ids]))
    entries.insert(0,dict(model='Whisper small, existing fine-tune (float)',metrics=measured))
    speakers=len({rows[i]['speaker'] for i in common_ids})
    counts=np.bincount([rows[i]['majority'] for i in common_ids],minlength=6).tolist()
    if counts!=[20]*6:raise ValueError('Native report expects twenty clips per emotion')
    save_json(ROOT/'reports/native-screen-comparison.json',dict(ids=common_ids,speakers=speakers,entries=entries,
        caveat='Balanced development screen; not independent test; different generation configurations recorded per run'))
    html=f'<section><h2>Local audio models: shared development screen</h2><p>{len(common_ids)} clips from {speakers} development speakers, 20 per emotion. These are screening results, not independent test confirmation. Machine predictions remain excluded from training.</p><div class="scroll"><table><tr><th>Model</th><th>Macro F1 points</th><th>Percent correct</th><th>Anger</th><th>Disgust</th><th>Fear</th><th>Happiness</th><th>Neutral</th><th>Sadness</th><th>Meaning</th></tr>'
    for entry in entries:
        m=entry['metrics'];score=m['macro_f1']*100
        recall_cells=''.join(f'<td>{r["recall"]*100:.1f}%</td>' for r in m["per_emotion"].values())
        html+=f'<tr><td>{escape(entry["model"])}</td><td><span style="display:inline-block;width:{score*2:.1f}px;height:10px;background:#64748b"></span> {score:.1f}</td><td>{m["accuracy"]*100:.1f}%</td>{recall_cells}<td>Development evidence only</td></tr>'
    return html+'</table></div><p>No new model is qualified to label training audio. MERaLiON trained on CREMA-D according to its <a href="https://arxiv.org/html/2511.04914">publisher paper, section 4.1.2</a>, so its row is an overlapping-data diagnostic, not held-out evidence. Extra surprise predictions count as wrong; none were silently removed. The original 12-clip Qwen/Liquid checks were runtime smoke tests concentrated on two speakers. This screen rotates speakers within emotions. Missing candidates are still pending or failed execution; absence is not a zero score.</p></section>'


def external_section():
    source=ROOT/'results/emozionalmente-small-int8-v1'
    protocol_path=ROOT/'reports/emozionalmente-protocol.json'
    protocol=json.loads(protocol_path.read_text())
    summary=json.loads((source/'summary.json').read_text())
    if summary['protocol_sha256']!=sha256(protocol_path) or summary['prediction_sha256']!=sha256(source/'predictions.npz'):
        raise ValueError('Stale external diagnostic')
    predictions=np.load(source/'predictions.npz')
    if predictions['ids'].tolist()!=[r['id'] for r in protocol['rows']]:raise ValueError('External cohort differs')
    measured=metrics(np.eye(6)[[r['perceived_label'] for r in protocol['rows']]],predictions['probabilities'])
    if abs(measured['macro_f1']-summary['scores']['macro_f1'])>1e-10:raise ValueError('External score differs')
    dest=ROOT/'reports/emozionalmente-small-int8-v1';dest.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.suffix in ['.json','.npz']:shutil.copy2(path,dest/path.name)
    n=len(protocol['rows']);speakers=len({r['speaker'] for r in protocol['rows']})
    return f'<section><h2>New human-rated corpus: Italian diagnostic</h2><p><strong>Bad — cross-language recognition remains weak.</strong> The unchanged small int8 model scored {measured["accuracy"]*100:.1f}% correct and {measured["macro_f1"]*100:.1f} macro F1 points on {n} clips from {speakers} speakers. It was not trained on this Italian corpus.</p><p>The publisher test has 1,334 clips from 88 speakers. This six-emotion diagnostic excludes 358 tied or surprise-winning clips and 23 clips exceeding the existing six-second input limit. Plurality is determined across all seven original labels before exclusion; actor intent and all five human votes remain preserved.</p><p>The full corpus adds 6,902 recordings with 34,510 existing human ratings. It is acted Italian, not natural English. The protocol and model hash were frozen before inference; unknown pretraining overlap remains a limitation. This result does not change the selected model.</p><p>Source: <a href="https://zenodo.org/records/12616095">Emozionalmente, Catania and colleagues</a>; <a href="https://openslr.org/161/">OpenSLR licence listing: CC BY 4.0</a>. No fresh human ratings were purchased.</p></section>'


def mess_section():
    from .mess_evaluate import score_shared
    source=ROOT/'results/mess-small-int8-v1'
    if not (source/'summary.json').exists():return ''
    protocol_path=ROOT/'reports/mess-protocol.json';protocol=json.loads(protocol_path.read_text())
    summary=json.loads((source/'summary.json').read_text())
    if not json.loads((source/'run_health.json').read_text())['completed']:raise ValueError('Incomplete MESS run')
    if summary['protocol_sha256']!=sha256(protocol_path) or summary['prediction_sha256']!=sha256(source/'predictions.npz'):
        raise ValueError('Stale MESS evidence')
    predictions=np.load(source/'predictions.npz')
    if predictions['ids'].tolist()!=[r['id'] for r in protocol['rows']]:raise ValueError('MESS cohort differs')
    scores=score_shared([r['perceived_label'] for r in protocol['rows']],predictions['probabilities'])
    if scores!=summary['scores']:raise ValueError('MESS scores differ')
    dest=ROOT/'reports/mess-small-int8-v1';dest.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.suffix in ['.json','.npz']:shutil.copy2(path,dest/path.name)
    from .mess_data import read_ratings
    save_json(ROOT/'reports/mess-human-ratings.json',dict(source=protocol['source'],license=protocol['license'],
        human_category_order=protocol['human_category_order'],rows=read_ratings(ROOT/'data/mess/ratings.xlsx')))
    shutil.copy2(ROOT/'data/mess/source-metadata.json',ROOT/'reports/mess-source-metadata.json')
    html=f'<section><h2>New English human ratings: three-emotion MESS diagnostic</h2><p><strong>Bad — recognition does not transfer reliably to these new English recordings.</strong> The unchanged small int8 model scored <strong>{scores["accuracy"]*100:.1f}% correct</strong> and {scores["macro_f1"]*100:.1f} macro F1 points on {protocol["selected_clips"]:,} recordings from six actors. This narrower diagnostic does not establish six-emotion or natural-conversation quality.</p><div class="scroll"><table><tr><th>Human-perceived emotion</th><th>Recordings</th><th>Percent correctly recognized</th></tr>'
    for label,row in scores['per_emotion'].items():
        value=f'{row["recall"]*100:.1f}%' if row['support'] else 'Not evaluated'
        html+=f'<tr><td>{escape(label)}</td><td>{row["support"]}</td><td>{value}</td></tr>'
    return html+'</table></div><p>The source contains 1,800 recordings with human rating percentages for angry, calm, happy and sad. We excluded 530 calm-winning or tied recordings before inference. Calm is not relabeled neutral. Predictions of disgust, fear or neutral still count as errors on the selected three emotions. All original human distributions and actor intentions are retained.</p><p>The model and cohort were frozen before inference. No MESS training or model selection occurred; unknown foundation-model pretraining overlap remains possible. These are acted English sentences from six speakers with four-choice human ratings, not spontaneous conversations. <a href="https://zenodo.org/records/7378320">Morgan Emotional Speech Set, Shae Morgan, Zenodo v5</a>; publisher metadata specifies CC BY 4.0.</p></section>'


def balanced_section():
    manifest=json.loads((ROOT/'data/manifest-v2.json').read_text())
    rows={r['id']:r for r in manifest['rows']};entries=[]
    for size in ['tiny','small']:
        source=ROOT/'results'/f'{size}-balanced-finetuned-v3'
        if not (source/'run_health.json').exists():continue
        if not json.loads((source/'run_health.json').read_text()).get('completed'):continue
        selection=json.loads((source/'selection.json').read_text());config=json.loads((source/'training-config.json').read_text())
        new=np.load(source/'dev_predictions.npz');old=np.load(ROOT/'results'/f'{size}-finetuned-v1/dev_predictions.npz')
        if new['ids'].tolist()!=old['ids'].tolist():raise ValueError('Balanced comparison cohort differs')
        if any(rows[str(i)]['split']!='dev' for i in new['ids']):raise ValueError('Balanced score not development')
        votes=np.array([rows[str(i)]['votes'] for i in new['ids']])
        before=metrics(votes,old['probabilities']);after=metrics(votes,new['probabilities'])
        if abs(after['macro_f1']-selection['winner']['dev_macro_f1'])>1e-9:raise ValueError('Balanced selection score differs')
        entries.append(dict(size=size,device=config['device'],best_epoch=selection['best_epoch'],before=before,after=after))
        dest=ROOT/'reports'/source.name;dest.mkdir(exist_ok=True)
        for filename in ['selection.json','history.json','training-config.json','run_health.json','dev_predictions.npz']:
            shutil.copy2(source/filename,dest/filename)
    if not entries:return ''
    save_json(ROOT/'reports/balanced-development-comparison.json',dict(entries=entries,
        note='Development only. CPU/MPS comparison is exploratory; class weighting changes the optimized distribution. Epoch zero means fallback to the original frozen-encoder head.'))
    html='<section><h2>Class balancing: development results only</h2><p>Training-only weights give less common emotions more influence while retaining every human vote. No old English test or external diagnostic was used in this experiment. Higher macro F1 and lower Brier error are better.</p><div class="scroll"><table><tr><th>Model</th><th>Existing fine-tune → weighted candidate: macro F1 points</th><th>Brier error: before → after</th><th>Selected epoch</th><th>Device</th></tr>'
    for e in entries:
        a,b=e['before'],e['after']
        html+=f'<tr><td>{e["size"]}</td><td>{a["macro_f1"]*100:.1f} → {b["macro_f1"]*100:.1f}</td><td>{a["brier_to_votes"]:.3f} → {b["brier_to_votes"]:.3f}</td><td>{e["best_epoch"]}</td><td>{e["device"]}</td></tr>'
    html+='</table></div><p>Epoch zero means none of the weighted training epochs beat the starting frozen-encoder classifier, so the candidate fell back to that starting model. Tiny ran on CPU while the old fine-tune used MPS; this is an exploratory comparison, not an isolated causal estimate of weighting. Missing sizes are still pending. Existing models remain selected.</p><div class="scroll"><table><tr><th>Model</th><th>Emotion</th><th>Development clips</th><th>Recall before → after</th></tr>'
    for e in entries:
        for label,row in e['after']['per_emotion'].items():
            old=e['before']['per_emotion'][label]
            html+=f'<tr><td>{e["size"]}</td><td>{label}</td><td>{row["support"]}</td><td>{old["recall"]*100:.1f}% → {row["recall"]*100:.1f}%</td></tr>'
    return html+'</table></div></section>'


def transfer_chart():
    """Expose the external-validation weakness, with per-emotion denominators."""
    from .benchmark import LABELS
    mess=ROOT/'results/mess-small-int8-v1/summary.json'
    if not mess.exists():return ''
    original=json.loads((ROOT/'results/small-finetuned-v1/final-test.json').read_text())['results']['int8']['all']
    italian=json.loads((ROOT/'results/emozionalmente-small-int8-v1/summary.json').read_text())['scores']
    english=json.loads(mess.read_text())['scores']
    data=[dict(dataset='Original English test',cohort='1,341 clips · 18 actors',scores=original),
          dict(dataset='New Italian diagnostic',cohort='953 clips · 86 actors',scores=italian),
          dict(dataset='New English diagnostic',cohort='1,270 clips · 6 actors',scores=english)]
    save_json(ROOT/'reports/transfer-chart.json',dict(model='Existing small int8, unchanged',rows=data,
        note='Different corpora and rating tasks; descriptive transfer diagnostic, not a controlled before/after experiment.'))
    svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1060 370" role="img" aria-label="Per-emotion recognition falls on new Italian and English recordings" style="width:100%;height:auto;display:block"><rect width="1060" height="370" rx="12" fill="#ffffff"/><g font-family="Arial,sans-serif" fill="#17263b"><text x="24" y="34" font-size="23" font-weight="bold">Does the same compact model work on new recordings?</text><text x="24" y="61" font-size="14">Correctly recognized within each human-perceived emotion · each cell shows its denominator</text>'
    for j,label in enumerate(LABELS):
        svg+=f'<text x="{303+j*121}" y="98" font-size="14" text-anchor="middle">{escape(label.title())}</text>'
    for i,row in enumerate(data):
        y=115+i*75
        svg+=f'<text x="24" y="{y+24}" font-size="16" font-weight="bold">{row["dataset"]}</text><text x="24" y="{y+46}" font-size="13">{row["cohort"]}</text>'
        for j,label in enumerate(LABELS):
            m=row['scores']['per_emotion'][label];n=m['support'];v=m['recall']*100
            color='#e7ebf0' if not n else '#d7efdf' if v>=60 else '#fae0e3'
            verdict='Not evaluated' if not n else f'{v:.1f}% · '+('good' if v>=60 else 'low')
            svg+=f'<rect x="{245+j*121}" y="{y}" width="116" height="63" rx="6" fill="{color}"/><text x="{303+j*121}" y="{y+25}" font-size="14" text-anchor="middle">{verdict}</text><text x="{303+j*121}" y="{y+47}" font-size="12" text-anchor="middle">'+(f'{n} recordings' if n else 'No target clips')+'</text>'
    svg+='<text x="24" y="351" font-size="13">Green: reaches provisional 60% recall target. Red: below target. Gray: emotion absent from this diagnostic.</text></g></svg>'
    (ROOT/'reports/transfer-chart.svg').write_text(svg)
    return '<figure style="margin:24px 0">'+svg+'<figcaption><strong>Bad — the existing model does not transfer reliably to the new recordings.</strong> Same small int8 model throughout; all predictions remain six-way. Corpus, language, acting and human rating choices differ, so these rows are not a controlled estimate of any single cause. Original English excludes 135 tied ratings; Italian excludes 358 tied/surprise-winning and 23 long clips; new English excludes 530 calm-winning/tied clips. New English measures only anger, happiness and sadness. All three corpora are acted speech.</figcaption></figure>'
