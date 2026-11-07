# Local phone emotion experiments

This is a research/evaluation branch, not a product release. It contains no nudge
policy or usefulness labels. No hosted inference or paid API is used. Model
quality and actual phone performance are separate gates; consult the generated
report before treating any checkpoint as reliable.

## Data and labels

`benchmarks/cremad-v2.jsonl` is the frozen, versioned index: original audio SHA256,
actor, sentence, intended emotion, human audio-only votes, and speaker-disjoint
partition. `benchmarks/NOTICE.md` records attribution and database licences.
The official source supplies audio separately. There are 7,441 usable recordings:
4,489 training, 1,476 development and 1,476 test. One source WAV is all zero and
is quarantined. Test includes 135 tied labels: these remain in distribution
metrics and are excluded, with counts, from single-label metrics.

Ground truth means **perceived vocal emotion according to these listeners**.
It does not establish a speaker's internal emotional state. The corpus is acted
English. No spontaneous-conversation or other-language validation is claimed.

The two training cohorts are `all_votes` and `actor_agrees`. The latter only
filters training speakers; development and test remain broad and identical.
The full crowd distribution is used for training, preserving minority votes.

## Reproduce locally

Run commands from the repository root. Python 3.12 and the exact package versions
in `requirements.lock.txt` were used on Apple Silicon. Install into `.venv`:

```sh
UV_NATIVE_TLS=1 uv venv --python 3.12 .venv
UV_NATIVE_TLS=1 uv pip sync --python .venv/bin/python eval/codex/requirements.lock.txt
```

Fetch the public corpus into a **new** directory (do not overwrite an existing
investigation). `GIT_LFS_SKIP_SMUDGE=1` avoids downloading video and MP3 copies:

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/CheyneyComputerScience/CREMA-D.git eval/codex/data/crema-d
GIT_LFS_SKIP_SMUDGE=1 git -C eval/codex/data/crema-d checkout 1658cd342dff90010aa843eaeebd53610a08b1dc
git -C eval/codex/data/crema-d lfs pull --include='AudioWAV/*' --exclude=''
.venv/bin/python -m eval.codex.benchmark
.venv/bin/python -m eval.codex.verify_benchmark
```

Download only official pinned model files; these are local weights, not API calls:

```sh
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 .venv/bin/python -c "from huggingface_hub import snapshot_download; from eval.codex.model import REVISIONS, ROOT; [snapshot_download('openai/whisper-'+name, revision=revision, allow_patterns=['config.json','preprocessor_config.json','model.safetensors'], cache_dir=ROOT/'artifacts/hf') for name,revision in REVISIONS.items()]"
```

On the original machine, Python downloads used
`SSL_CERT_FILE=$HOME/.cache/macos-ca-bundle.pem` and
`REQUESTS_CA_BUNDLE=$HOME/.cache/macos-ca-bundle.pem`. These point to the existing
local certificate bundle; no TLS verification was disabled.

The original manifest retains a machine-local path in its quarantined-file error
message, so its whole-file checksum can differ on another computer. The portable
frozen index has no absolute paths. `verify_benchmark` checks every retained label,
vote, speaker, split and audio hash against that index. New runs bind to the local
manifest checksum; preserved historical run hashes are never rewritten.

Each model first passes a real-audio smoke test, then extracts resumable features:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size tiny --limit 6
HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size tiny
.venv/bin/python -m eval.codex.train --size tiny --out eval/codex/results/tiny-all-reproduction
.venv/bin/python -m eval.codex.train --size tiny --cohort actor_agrees --out eval/codex/results/tiny-agrees-reproduction
.venv/bin/python -m eval.codex.finetune --source eval/codex/results/tiny-all-reproduction --out eval/codex/results/tiny-finetuned-reproduction
.venv/bin/python -m eval.codex.export --run eval/codex/results/tiny-finetuned-reproduction
```

Use `base` or `small` in place of `tiny` for the other fixed candidates. All run
directories are immutable; use a new name for a new experiment. Feature extraction
can resume; it checks audio hashes even for cache hits. Keep logs for failed runs.
No batch failure is silently converted to a neutral label or a zero score.

After model settings are frozen, evaluate once:

```sh
.venv/bin/python -m eval.codex.freeze --runs tiny-finetuned-reproduction --out eval/codex/results/reproduction-freeze.json
.venv/bin/python -m eval.codex.evaluate --run eval/codex/results/tiny-finetuned-reproduction --freeze eval/codex/results/reproduction-freeze.json
```

The script stores per-clip predictions, per-emotion precision/recall/F1, Cohen's
kappa, vote-distribution scores, and speaker-cluster bootstrap intervals. It also
reports actor-agreement and crowd-consensus subsets, without replacing the full
test result. Bootstrap resamples actors, not nominally independent clips.

For ongoing regression, run a **new candidate directory** against the same frozen
index and compare the saved reports. Repeated tuning based on this test would
turn it into development data; reserve new independent speakers/corpora for
future acceptance decisions. Publisher pretraining exposure cannot be ruled out.

## Export and phone integration contract

Exports contain the **complete audio encoder plus six-class head**, not just the
tiny classification layer. `input_contract.json` and `frontend-constants.npz`
specify the 16 kHz mono log-Mel frontend. Float export parity and int8 quality
are checked on all development clips. `validation.json` records actual bytes,
hashes, inference timing, and read/preprocess/inference timing.

The ONNX graph accepts fixed six-second padded feature tensors plus valid-frame
lengths. Short clips are padded; long clips are rejected rather than silently
truncated. Six seconds is a maximum accepted clip window, not a claim of
sub-second detection latency. Microphone collection time is additional.

Run an existing local export on a real clip:

```sh
.venv/bin/python -m eval.codex.infer --model eval/codex/results/tiny-finetuned-v1/export/emotion-int8.onnx --audio eval/codex/data/crema-d/AudioWAV/1001_IEO_NEU_XX.wav
```

Silence returns `no_speech`, never a fabricated emotion. Other unsupported inputs
fail explicitly. This CLI uses the reference Python frontend; an independent
Android/iPhone frontend still needs parity tests and actual device latency,
memory, battery and sustained-stream measurements. No app code was changed.

## Machine-labelled training expansion

Qwen2.5-Omni is tested locally against human **development** votes with identical
audio and transcript-only cases. It uses single forward-pass letter scores,
not generated reasoning; therefore no completion budget or truncation is involved.
These conditional scores are not calibrated probabilities of true emotion.

`pseudolabel.admit_rows` composes quality, provenance, model-revision, licence,
protected-speaker/audio and confidence gates. It is an **admission utility**, not
an implemented mass-audio collection/labelling job. It rejects teachers below the
predeclared provisional bar (macro F1 0.75, every-class recall 0.60), incomplete
metrics and nonfinite values. Class confidence thresholds are exploratory,
selected on development clips, not new-corpus precision guarantees.

No machine-labelled pool should be scaled unless a teacher qualifies. New-domain
human validation remains necessary even then. Machine labels must always remain
distinguishable from human votes. No test speaker, original clip, or crop of a
protected recording may enter the training pool; future segment manifests must
retain their parent recording and speaker IDs.

## Tests

The visible overview chart is generated from saved test metrics with
`python -m eval.codex.plot_comparisons` (optional plotting dependency:
Matplotlib 3.11.2), before running `python -m eval.codex.report`.
Plotting can use a separate environment; it does not change model dependencies.
The report builder checks the plotted data and SVG checksum before embedding it.
The SVG, PNG, and underlying numerical values are saved as
`reports/modification-overview.*`. The figure compares changes separately;
agreement-only training is not presented as a stage before all-clip fine-tuning.

```sh
.venv/bin/python -m pytest -q -c /dev/null eval/codex/tests
```

Only these targeted tests are used while data jobs run. No whole-repository suite
is run, and no existing tracked application data is modified.

## Overnight extensions (2026-11-06)

The original freeze and nine-candidate results remain unchanged. Agreement-only
fine-tuning reused the previously inspected English test only as regression
evidence; its separate freeze is `reports/agreement-v2-freeze.json`. Development
results determine selection. The class-balancing experiment is predeclared in
`reports/balanced-training-plan.json`; its inverse-square-root weights use only
training human votes. Calibration is reported separately because weighting
changes the optimized distribution. Epoch zero means the starting classifier
was retained, not a successful weighted fine-tune.

Native audio screens use exactly the same 120 development clips (20 per emotion,
18 speakers) for Qwen3-Omni MLX, Liquid, EmotionThinker and MERaLiON. See
`reports/native-screen-comparison.json`, per-run configuration/raw responses,
`reports/local-model-weight-inventory.json` and runtime inventories. Native
reasoning uses a 40,000-token budget and fails on malformed, empty or truncated
answers. One-hot native answers are not calibrated probability estimates.
MERaLiON's publisher lists CREMA-D as training data; its screen is an overlap
diagnostic, not independent validation. No new teacher qualified.

Two external human-rated corpora were downloaded from their publishers and
checksum-verified. Their protocols were committed before inference with the
unchanged original small int8 export:

- [Emozionalmente](https://zenodo.org/records/12616095), CC BY 4.0: acted Italian,
  five ratings per clip. `reports/emozionalmente-protocol.json` selects 953
  publisher-test clips from 86 actors. All seven labels are considered before
  excluding tied/surprise winners and clips exceeding six seconds.
- [Morgan Emotional Speech Set](https://zenodo.org/records/7378320), Shae Morgan,
  CC BY 4.0 according to publisher API metadata: acted English, six actors,
  1,800 recordings with four-category human distributions in the downloadable
  spreadsheet. `reports/mess-protocol.json` selects 1,270 angry/happy/sad
  plurality clips, excluding 530 calm/tied clips. Calm is not neutral. All six
  model outputs remain eligible; an unsupported predicted emotion counts wrong.
  Macro metrics average only the three evaluated target emotions. All original
  distributions and actor intentions are in `reports/mess-human-ratings.json`.

Both diagnostics show poor transfer. These corpora are acted, their rating tasks
differ, and foundation-model pretraining exposure is unknown. They do not prove
natural-conversation quality. After inspection they are reusable regression sets,
not fresh acceptance evidence for later tuning. The 15-clip MESS float/int8 check
is an implementation check, not a new benchmark. No MESS training occurred.

Audit/evaluation entry points are `external_data`, `external_evaluate`,
`mess_data`, and `mess_evaluate` under `eval.codex`. They require the ignored
publisher downloads/model artifacts and deliberately refuse to overwrite frozen
protocols or existing evaluation directories. Do not delete evidence to rerun.
The normal report rebuild also verifies these raw predictions and writes the
new transfer heatmap (`reports/transfer-chart.svg` and JSON). Current active-job
details are in `OVERNIGHT-20261106.md`; all app, nudge and device work is excluded.
