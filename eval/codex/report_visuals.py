"""Accessible, self-contained dashboard graphics derived from saved measurements."""
from html import escape

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
SIZES = ['tiny', 'base', 'small']
CSS = '''
.meter{display:block;min-width:90px;height:7px;background:var(--line);margin-top:5px;border-radius:4px;overflow:hidden}
.meter i{display:block;height:100%;background:currentColor}
.measure{white-space:nowrap;font-variant-numeric:tabular-nums}
.delta-track{position:relative;display:block;width:150px;height:10px;background:var(--line);margin-top:5px}
.delta-track:after{content:"";position:absolute;left:50%;height:100%;border-left:1px solid var(--fg)}
.delta-track i{position:absolute;height:100%;background:currentColor}
.flow{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}.flow>div{flex:1;min-width:160px;border:1px solid var(--line);padding:14px;border-radius:8px}.flow strong{display:block;font-size:1.5rem}
.visual-table{min-width:1050px}.visual-table th:first-child{min-width:150px}.note{font-size:.9rem}
'''


def table(headers, rows):
    return '<div class="scroll"><table class="visual-table"><thead><tr>' + ''.join(
        f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def meter(value, label=None, maximum=100, verdict=None):
    label = label or f'{value:.1f}%'
    color = f' {verdict}' if verdict else ''
    return f'<span class="measure{color}">{escape(label)}<span class="meter" aria-hidden="true"><i style="width:{100*value/maximum:.4f}%"></i></span></span>'


def delta(before, after):
    points = 100 * (after - before)
    color = 'good' if points >= 0 else 'bad'
    width = min(abs(points), 25) * 2
    left = 50 if points >= 0 else 50 - width
    label = 'gain' if points > 0 else 'loss' if points < 0 else 'unchanged'
    return (f'<span class="measure {color}">{points:+.2f} points — {label}'
            f'<span class="delta-track" aria-hidden="true"><i style="left:{left:.4f}%;width:{width:.4f}%"></i></span></span>')


def paired_metrics(records, size, change):
    """Isolate one change; never compare a float baseline with a different int8 run."""
    if change == 'filter':
        a, b, av, bv = f'{size}-all-v1', f'{size}-agrees-v1', 'float', 'float'
    elif change == 'finetune':
        a, b, av, bv = f'{size}-all-v1', f'{size}-finetuned-v1', 'float', 'float'
    elif change == 'compress':
        a = b = f'{size}-finetuned-v1'
        av, bv = 'float', 'int8'
    else:
        raise ValueError(change)
    if records[a]['test']['manifest_sha256'] != records[b]['test']['manifest_sha256']:
        raise ValueError('Comparison requires the same benchmark')
    x = records[a]['test']['results'][av]['all']
    y = records[b]['test']['results'][bv]['all']
    if (x['n_total'], x['n_single_label']) != (y['n_total'], y['n_single_label']):
        raise ValueError('Comparison requires identical denominators')
    return x, y


def model_matrix(records, teacher):
    rows = []
    descriptions = {
        'all': 'Added trained classifier; encoder frozen; 4,489 training clips',
        'agrees': 'Added trained classifier; encoder frozen; 1,849 agreement clips',
        'finetuned': 'Retrained encoder + classifier; 4,489 clips; then 8-bit compression',
    }
    for size in SIZES:
        for kind in descriptions:
            variant = 'int8' if kind == 'finetuned' else 'float'
            m = records[f'{size}-{kind}-v1']['test']['results'][variant]['all']
            rows.append([f'Whisper {size}<br>{kind} · {variant}', 'Local Mac',
                         str(m['n_single_label']), descriptions[kind]] + [
                meter(100*m['per_emotion'][e]['recall'], verdict='good' if m['per_emotion'][e]['recall'] >= .60 else 'bad') for e in EMOTIONS])
    body = '<p>Each emotion is <strong>percent correct among clips humans labelled that emotion</strong> (recall). Bars run from 0% to 100%; green meets the provisional 60% per-emotion goal, red falls short. These are the same 1,341 uniquely labelled test clips for every Whisper row; 135 ties are excluded.</p>'
    body += table(['Model / variant', 'Where run', 'Scored clips', 'What changed'] + EMOTIONS, rows)
    body += '<p>All Whisper rows already have a trained emotion classifier. Unmodified Whisper transcribes speech; we did not measure an off-the-shelf six-emotion score. Adding the classifier therefore establishes the baseline, with no measured before/after uplift. No pruning or distillation has been performed. Phone and watch execution remain untested.</p>'
    q = teacher['results']['audio']['metrics']
    body += '<h3>Separate development experiment — not the same test set</h3>'
    body += table(['Model', 'Where run', 'Scored clips', 'What changed'] + EMOTIONS, [[
        'Qwen2.5-Omni-7B', 'Local Mac', str(q['n_single_label']), 'No weight changes; constrained six-label scoring'] + [
        meter(100*q['per_emotion'][e]['recall'], verdict='good' if q['per_emotion'][e]['recall'] >= .60 else 'bad') for e in EMOTIONS]])
    body += '<p>Qwen uses 1,353 unique winners from 1,476 development clips; 123 ties are excluded. These results cannot establish its relative test ranking. Recommendation: test stronger audio teachers on development data before deciding whether to fine-tune them.</p>'
    return body


def modifications(records):
    body = '<p>Each comparison changes one part of the pipeline. Same 1,341 uniquely labelled test clips throughout; 135 ties excluded. Changes are percentage points, not relative percentages. These are descriptive results, not statistically established gains. Bars share a −25 to +25 point scale: left/red is loss, right/green is gain.</p>'
    for change, title, explanation in [
        ('filter', 'Training filter: all clips → agreement-only', 'Frozen encoder and trained classifier in both runs; float precision in both. Only training is filtered. Training shrinks from 4,489 to 1,849 clips, changing the label mixture too. Development favored all-clip training for all three sizes, so this test advantage is not a universal result.'),
        ('finetune', 'Fine-tuning: frozen encoder → retrained encoder', 'Both use all 4,489 training clips and float precision. Each includes a trained classifier; fine-tuning also changes encoder weights. Agreement-only encoder fine-tuning has not been run.'),
        ('compress', 'Compression: float → 8-bit weights', 'Same fine-tuned checkpoint and labels. This measures compression loss separately from training gains. Some operations remain floating point.'),
    ]:
        rows = []
        for size in SIZES:
            a, b = paired_metrics(records, size, change)
            rows.append([size, f'{100*a["macro_f1"]:.2f}% → {100*b["macro_f1"]:.2f}%', delta(a['macro_f1'], b['macro_f1'])] + [delta(a['per_emotion'][e]['recall'], b['per_emotion'][e]['recall']) for e in EMOTIONS])
        body += f'<h3>{title}</h3><p>{explanation}</p>'
        body += table(['Encoder', 'Macro F1 before → after', 'Macro F1 change'] + [e+' recall change' for e in EMOTIONS], rows)
    body += '<h3>Compression reduces storage substantially</h3>'
    rows = []
    for size in SIZES:
        v = records[f'{size}-finetuned-v1']['export']['sizes_bytes']
        a, b = v['float']/1e6, v['int8']/1e6
        rows.append([size, meter(a, f'{a:.1f} MB', 350), meter(b, f'{b:.1f} MB', 350), f'<span class="good">GOOD — {100*(1-b/a):.1f}% smaller</span>'])
    return body + table(['Encoder', 'Float size', '8-bit size', 'Storage benefit'], rows) + '<p>Size bars share a 0–350 MB scale. Lower storage is good; phone latency and battery benefit have not been measured. Macro F1 balances precision and recall equally across emotions; it is not overall percent correct.</p>'


def provenance(rows):
    test = [r for r in rows if r['split'] == 'test']
    unique = [r for r in test if r['majority'] is not None]
    supports = [sum(r['majority'] == i for r in unique) for i in range(6)]
    assert len(test) == 1476 and len(unique) == 1341
    body = '<p>The recordings come from the <a href="https://github.com/CheyneyComputerScience/CREMA-D">CREMA-D publisher</a>: actors reading scripted English sentences with intended emotions. We did not create 1,341 new recordings. That number is one held-out portion of this public dataset.</p>'
    body += '<div class="flow"><div>Original corpus<strong>7,442 clips</strong>91 actors</div><div>Usable audio<strong>7,441 clips</strong>One silent training clip excluded</div><div>Test partition<strong>1,476 clips</strong>18 actors absent from training and development</div><div>Single-label scoring<strong>1,341 clips</strong>135 tied winners excluded</div></div>'
    body += '<p>Humans listened to the audio and chose one of six emotions per rating. We keep the counts for all six choices; these are not six independent yes/no ratings or six intensity scores. The unique most-common choice is our single-label target, even when fewer than half the listeners chose it. Ties still contribute to vote-distribution metrics. Actor intent stays separate.</p>'
    body += table(['Human winning emotion', 'Number of test recordings'], [[e, meter(n, str(n), 700)] for e, n in zip(EMOTIONS, supports)])
    body += '<p>Count bars share a 0–700 clip scale. Neutral dominates, so overall accuracy alone can look misleadingly good. Human audio-only ratings come from <code>finishedResponses.csv</code>, filtered with the publisher’s response-time data in <code>finishedEmoResponses.csv</code>. Frozen per-clip votes and splits are in <code>eval/codex/benchmarks/cremad-v2.jsonl</code>.</p>'
    return body


TEACHERS = '''<p><strong>A stronger audio teacher is worth a controlled comparison, but being multimodal does not automatically mean accepting raw audio.</strong> Official model pages checked November 5, 2026 list audio as unsupported for <a href="https://developers.openai.com/api/docs/models/gpt-6-luna">GPT-6 Luna</a>, <a href="https://developers.openai.com/api/docs/models/gpt-5.6-terra">GPT-5.6 Terra</a>, <a href="https://developers.openai.com/api/docs/models/gpt-6-sol">GPT-6 Sol</a> and <a href="https://developers.openai.com/api/docs/models/gpt-6-astra">GPT-6 Astra</a>. They support text and images. Codex model choice alone does not give us a raw-audio labelling interface.</p>
<p><a href="https://developers.openai.com/api/docs/guides/audio-chat-completions">OpenAI’s audio guide</a> demonstrates native audio input using <code>gpt-audio-1.5</code>. That is a plausible cloud comparison candidate, not a measured winner. A transcript-only model misses vocal delivery; using it would answer a different question.</p>
<p>Recommended experiment: first validate the audio model on existing human-rated development clips, hiding actor labels, human votes and filenames that reveal the answer. Compare per-emotion errors and confidence with local candidates, then confirm on fresh human-rated natural speech. Keep machine labels explicitly marked and separate from human ground truth; review disagreements and low-confidence cases. Only expand the training library after a teacher passes this evaluation. No cloud inference or new labelling run was performed for this update.</p>
<p><strong>Commercial training rights require a separate check.</strong> The <a href="https://openai.com/policies/services-agreement/">OpenAI Services Agreement, §3.3 and “Permitted Exception”</a>, restricts using outputs to develop competing models; its classifier exception excludes models distributed or made commercially available to third parties. This does not determine whether your specific app is competing, but we cannot assume permission to train a shipped phone model from those outputs. Confirm applicable contractual rights before bulk teacher labelling, or use a teacher whose terms expressly permit this use. A paid cloud benchmark and a free-to-run phone model are separate cost decisions.</p>'''

HOLDOUT = '''<p><strong>Yes: training and testing stay separate by person, not just by recording.</strong> The split already exists; we would reuse its training portion for the new experiment, without moving test clips into training.</p>
<div class="flow"><div>Train weights<strong>1,849 clips</strong>Agreement-only subset of the 4,489 training clips from 55 actors</div><div>Select checkpoint<strong>1,476 clips</strong>18 different development actors; broad emotions and disagreements retained</div><div>Compare regression<strong>1,476 clips</strong>18 different test actors; 1,341 unique winners plus 135 ties</div><div>Independent acceptance<strong>New recordings</strong>Fresh unseen speakers; human ratings; not collected yet</div></div>
<p>Compare agreement-only fine-tuning against the existing all-clip fine-tuned model at float precision first. Pick training settings using development data only, freeze the choice, and then compare on the same broad test set. Report all six recalls, macro F1, confusion patterns and actor-level uncertainty; retain tied votes for distribution scoring. Afterward compare float and compressed copies separately.</p>
<p>The existing test has now been inspected and influenced our next question. It remains useful for regression checks, but it is no longer a fresh independent confirmation for the next research round. Reserve a new human-rated set of natural conversations from unseen speakers for final acceptance. Do not filter that acceptance set to actor/listener agreement and do not let a teacher label its own exam.</p>'''
