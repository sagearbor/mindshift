# Gemini audio analyzer

The script reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the repository-root
gitignored `.env`. Keep the key out of commands, logs and Git. Each computer
needs its own local `.env`; copying the repository does not copy credentials.

From the repository root on the other computer, use Python 3.12 and pip inside
a new virtual environment (do not use macOS system Python 3.9):

```sh
python3.12 -m venv eval/codex/artifacts/gemini-runtime
eval/codex/artifacts/gemini-runtime/bin/python -m pip install -r eval/codex/gemini-requirements.txt
eval/codex/artifacts/gemini-runtime/bin/python -m unittest discover -s eval/codex/gemini_tests -v
```

If Python 3.12 is unavailable, uv can provision it. Alternative setup:

```sh
UV_SYSTEM_CERTS=1 uv venv --python 3.12 eval/codex/artifacts/gemini-runtime
UV_SYSTEM_CERTS=1 uv pip install --python eval/codex/artifacts/gemini-runtime/bin/python -r eval/codex/gemini-requirements.txt
eval/codex/artifacts/gemini-runtime/bin/python eval/codex/audio_emotion_analyzer.py --help
eval/codex/artifacts/gemini-runtime/bin/python -m unittest discover -s eval/codex/gemini_tests -v
```

`pygame` is optional and needed only for `--play` or `--interactive`.
Do not recreate the runtime if it already works.

After confirming the API project is on the free tier, or obtaining explicit
authorization for paid usage, this command uploads one local sample:

```sh
eval/codex/artifacts/gemini-runtime/bin/python eval/codex/audio_emotion_analyzer.py eval/codex/data/gemini-manual-20261106/clip-01.wav --model gemini-3.8-flash --summary
```

The nine WAV files in `data/gemini-manual-20261106` are explicitly tracked in
Git, despite the broader data-folder ignore rule. Their attribution and source
hashes are alongside them. The environment, full datasets, model weights and
`.env` are NOT tracked. Only the Gemini smoke test needs this small environment;
the larger research workflow has additional dependencies and downloads.

The user confirmed the replacement key belongs to a free-tier project.
Live test: clip-01 completed (neutral; humans 70% happy), clip-02 returned
503 high demand, clip-03 was skipped. No model substitution occurred.
Clips 04–09 are selected easy examples: two each angry, happy and sad, all
unanimously rated by the original human panel. They have NOT been sent to Gemini.
See `reports/gemini-free-tier-smoke-20261106.json` for the exact outcomes.

Next: run one easy clip, then continue if successful. Provider errors must not
be counted as emotion predictions. Add bounded retries for temporary 503 errors
and preserve full response/usage metadata before any large batch. Keep the
40,000-token output budget for reasoning models. Do not send answer keys to
Gemini. These examples establish functionality only, not labeling quality.
