# English-first emotion research handoff

User clarification, 2026-11-06: English is the priority. Italian may be a minority
augmentation experiment only if it helps English validation. All custom training
so far used English CREMA-D only (4,489 training clips, or 1,849 in agreement-only
experiments). Italian was evaluation only; MESS English was also evaluation only.

Repository: sagearbor/mindshift. Branch: codex/emotion-eval. Draft PR218:
https://github.com/sagearbor/mindshift/pull/218
Base is docs/emotion-nudge-plan, not main. Completed overnight milestone060bc1b.
Read current raw reports, README.md and the final OVERNIGHT-20261106.md entries.
Do not restart completed jobs. Audio, environments and model weights are ignored
local files; GitHub carries code, configurations, predictions and evidence, not
those binaries. The 15-minute recovery heartbeat is paused.

Current selected experiment: small-finetuned-v1, Whisper small audio encoder
with a trained six-emotion classifier, encoder fine-tuned using human listener
vote distributions, then int8-exported. Complete export94,862,824bytes; tiny
alternative9,942,329bytes. Neither is product-qualified or device-validated.
Original English test accuracy73.53%, macro F1 .633632. New MESS three-emotion
English diagnostic accuracy18.11%, macro F1 .212916. Do not compare these as
identical tasks or claim broad emotion recognition has been achieved.

Next recommendation:
1. Run the user's manual Gemini audio capability check with the three anonymous
   English WAVs below. A large/cloud labeling teacher is acceptable; only the
   final deployed student must be compact. Do not impose phone size on teachers.
2. Save exact Gemini model/version shown by its UI, date, prompt and raw answers.
   Compare to the separately saved key. Three examples are a smoke test only.
   If useful, predeclare a larger English, speaker-diverse, six-emotion screen
   with human ratings before trusting it for labels. No paid calls authorized.
3. Independently audit existing cross-corpus failure, then seek broader English
   human-rated training data. Keep English-only baseline. Any Italian addition
   should be a clearly recorded minority and selected by English development
   results. Existing inspected diagnostic sets are regression evidence, not fresh
   acceptance sets for repeated tuning. Preserve actor intent, human votes,
   machine-label provenance, licensing and speaker/recording separation.
4. Train/compress only after label quality is demonstrated. No nudges, physical
   phone tests, paid services or deployment. Tests targeted to eval/codex/tests.

## Manual Gemini upload files (local, ignored)

- /Users/scb2/PROJECTS/pgh/mindshift/eval/codex/data/gemini-manual-20261106/clip-01.wav
- /Users/scb2/PROJECTS/pgh/mindshift/eval/codex/data/gemini-manual-20261106/clip-02.wav
- /Users/scb2/PROJECTS/pgh/mindshift/eval/codex/data/gemini-manual-20261106/clip-03.wav

Three different English speakers from MESS; deterministic selection without
using model predictions. Filenames conceal labels. Do not upload the answer key:
reports/gemini-manual-answer-key.json. Keep the selected model identity from the
UI rather than trusting a model's self-description.

## Prompt to paste with the three audio attachments

Listen directly to each attached WAV. Judge the emotion conveyed by vocal tone,
pitch, rhythm, intensity and voice quality, rather than the sentence meaning.
Ignore filenames, metadata and any knowledge of public datasets. Analyze each
recording independently; do not assume their emotions differ.

If you cannot access the audio signal and only have text or a transcript, return
audio_unavailable for that file; do not guess. Otherwise return a JSON array with
one object per file: filename, primary_emotion (anger, disgust, fear, happiness,
neutral, sadness, other, or uncertain), alternative_emotion (or null), confidence
(0–100, a subjective estimate), and audible_evidence (one short sentence about
the voice). Label perceived vocal expression, not the person's internal state.
Return only the JSON. Do not browse for the recordings or their labels.
