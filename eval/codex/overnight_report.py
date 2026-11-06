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
    html=f'<section><h2>Local audio models: shared development screen</h2><p>{len(common_ids)} clips from {speakers} development speakers, 20 per emotion. These are screening results, not independent test confirmation. Machine predictions remain excluded from training.</p><div class="scroll"><table><tr><th>Model</th><th>Macro F1 points</th><th>Percent correct</th><th>Meaning</th></tr>'
    for entry in entries:
        m=entry['metrics'];score=m['macro_f1']*100
        html+=f'<tr><td>{escape(entry["model"])}</td><td><span style="display:inline-block;width:{score*2:.1f}px;height:10px;background:#64748b"></span> {score:.1f}</td><td>{m["accuracy"]*100:.1f}%</td><td>Development evidence only</td></tr>'
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
