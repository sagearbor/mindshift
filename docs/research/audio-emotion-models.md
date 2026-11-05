# Commercial emotion-model audit — 2026-11-05

Scope: voice emotion only, phone first. No nudge labels, nudge policy, hosted inference or paid API. Sources checked against publisher repositories/model cards, not old project status notes.

## Models selected for actual experiments

| Candidate | Weight permission evidence | Role | Decision |
|---|---|---|---|
| OpenAI Whisper tiny/base encoders | [Official README](https://github.com/openai/whisper#license) explicitly covers code and model weights; [MIT licence](https://github.com/openai/whisper/blob/main/LICENSE), “free of charge” | Frozen representation plus six-emotion head trained here; no transcription decoder in export | Proceed, retain MIT attribution. These are downloadable local weights, not an OpenAI API or its generated-output terms. |
| Qwen2.5-Omni-7B | [Publisher model card](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), Apache-2.0 | Possible offline teacher, not phone model | Eligible licence candidate; not validated as an emotion teacher here yet. Card lists MELD aggregate 0.570, not per-emotion natural-speech numbers. |
| emotion2vec+ large | [Publisher card](https://huggingface.co/emotion2vec/emotion2vec_plus_large): custom `model-license`, linked to FunASR; [current model agreement](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE) | Possible offline teacher | Excluded from the commercial training path pending unambiguous weight terms. The research code’s MIT badge is not evidence that these weights are MIT. Custom terms contain restrictions/revision clauses; do not silently substitute a code licence. |
| WavLM base-plus | [Publisher repository](https://huggingface.co/microsoft/wavlm-base-plus) | Possible encoder comparator | HF metadata does not specify a licence. Verify exact weight permission before introducing it. |

Whisper is a speech-recognition pretrained encoder, not an already validated emotion classifier. Its phone suitability is established only when we export and measure our entire encoder/head pipeline. Published per-emotion natural-speech results for these particular Whisper classifiers: **not found** (they are new experiments).

Pinned Whisper revisions: tiny `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, base `e37978b90ca9030d5170a5c07aadb050351a65bb`. Exact tensor counts and artifact sizes are measured by the export code, rather than inferred from full transcription model names.

## Data permission and human truth

[CREMA-D official licence](https://github.com/CheyneyComputerScience/CREMA-D/blob/master/LICENSE.txt) names the Open Database License for the database and Database Contents License for individual contents. [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) permits commercial use with attribution and applicable database share-alike obligations; [DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/) covers individual contents. Free commercial use does not mean attribution-free or obligation-free. Preserve original notices and identify derived label databases; no raw audio is pushed with this PR. Public use/distribution of a derived database must follow ODbL obligations. Model distribution and database distribution are distinct; this experiment does not purport to settle every downstream legal interpretation.

Official source commit: `1658cd342dff90010aa843eaeebd53610a08b1dc`.

The benchmark reconstructs votes from `finishedResponses.csv`, keeps `queryType=1` (voice-only), applies the authors’ first-click-time filter from `finishedEmoResponses.csv` / `processFinishedResponses.R`, and verifies winning labels against `processedResults/summaryTable.csv`. The official count table’s row IDs beginning with `1` also distinguish audio from face and audiovisual observations. Filename emotion is stored separately and NEVER substituted for listener labels.

All six categories are retained: anger, disgust, fear, happiness, neutral and sadness. Ties stay in distribution scoring; single-label metrics explicitly exclude them. The corpus is acted English, not proof of performance in spontaneous phone conversations. The same 12 sentence texts occur across emotions, enabling a transcript-only control.

Do not use RAVDESS, IEMOCAP, SBCSAE or other restricted corpora for commercial training by inheriting old script assumptions. Permission must be checked for the specific use, including evaluation; “evaluation only” is not an automatic exception to a noncommercial licence.

## Where earlier notes or conversation were wrong

- CREMA-D intended emotion and heard emotion are substantially different. Using filenames as gold labels answers a different question.
- Initial chat calculation used unfiltered completed responses: 3,085 matching out of 6,806 unique winners, 636 ties. **Corrected** using the authors’ filter: 3,099 matching out of 6,798 unique winners (45.59%), 644 ties. Every one of 7,442 reconstructed winner sets matches the official summary.
- Whisper-base “0.855” on a prior RAVDESS split cannot establish six-emotion performance on this human-audio benchmark. Old local artifacts are absent on this fresh checkout.
- An audio encoder can encode information about spoken words. “It cannot read text” does not prove absence of lexical leakage.
- No fixed universal human agreement ceiling is assumed; disagreement is measured on this benchmark.
- A small model file or a fast Mac run does not prove phone latency or battery use. Real-device testing remains a distinct gate.

## Further catalogue work

Qwen3-Omni, Voxtral, Phi-4-multimodal and MiniCPM-o were named in the earlier handoff. Their exact candidate versions, current weight licences and per-emotion natural-speech results remain **not verified in this run**. They are not used as teachers or shipped dependencies. Prioritize reproducible baseline evidence over adding unchecked models.
