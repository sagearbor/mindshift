# Emotion / nudge-quality plan (handoff for any agent: Claude Code or Codex)

Written 2026-11-05. Standing rules: owner has no human time (no manual labelling) and pays nothing for data/licences; round prices only; never ship/merge/deploy without the owner's word; PRs #215 and #216 stay unmerged.

## Findings so far (details: owner's local `tmp/*-20261105/REPORT.html`, gitignored, regenerate if needed)
- Shipped instant tier (`apps/mobile/src/live/instantTier.ts`, 16 acoustic descriptors, 3 KB, pure TS): angry-vs-happy only. Cross-corpus AUC 0.795 shipped (0.842 Python bench) vs raw loudness 0.713.
- VoiceCLAP-commercial (110M, CC-BY): ties the tier (0.840), ~100 MB, CONFER result inconclusive (Greek audio, unsupported language). Not integrated.
- Own distilled student (Whisper-base 22 MB int8 / tiny 9.6 MB, 11-23 ms/5 s on one CPU thread): useful only for Arousal; never beat per-speaker loudness with CI excluding 0 on real fixture packs. Limit = teacher label quality.
- Judges disagree (kappa 0.14 on 17 lines): judge trust is the bottleneck.
- OpenAI terms forbid using its outputs to train a shipped model; Gemini unclear (paid tier only); Qwen-Omni (Apache-2.0) allowed. Using any of them as an EVALUATION judge only is fine.
- Older audio-LLM evidence (GPT-4o, Gemini 2.0) is stale. Newest models are untested by us.

## Labelled audio we can test against (human annotations)
| Corpus | Human labels | Licence use |
|---|---|---|
| CREMA-D | ~6 crowd votes per clip, 6 emotions + intensity; sentences carry no emotion words (text-leakage check) | ODbL, commercial OK |
| RAVDESS | acted, 8 emotions | non-commercial, eval only |
| IEMOCAP | 4-9 emotions + dimensions | eval only (SAIL permission) |
| CONFER (Greek) | 10-rater continuous conflict curves | eval only; Greek: use only for language-agnostic tests like loudness |
| SBCSAE, AMI | no emotion labels (used as assumed-calm negatives) | SBCSAE eval only; AMI CC-BY-4.0 |
| Fixture packs A/B | repo fixtures of real sessions | see `scripts/` and `server/tests` |
Download from the public sources; audio is not in git.

## Proposed plan (no human labelling)
1. Judge-trust layer: panel of labellers from different vendors (Gemini paid batch, local Qwen-Omni, optionally GPT-6.1 Terra as a judge only); score each against CREMA-D crowd votes and IEMOCAP/RAVDESS; per-emotion agreement + CI; audio-only vs transcript-only vs both to catch word-reading; adversarial/silent cases; frozen hashed case set with leakage check.
2. Test newest audio models (Gemini newest Flash, GPT-6.1 Terra/realtime audio, Qwen-Omni latest, Apriel Voice 27B if it outputs tone) on step 1's harness; update `emotion-models-and-data-reference` memory note.
3. Only after 1-2: relabel a licence-clean pool (AMI, YODAS2, People's Speech, consented opt-in audio) with the agreed labels; train/distil a small model; ship dark.
4. Port the 16-descriptor tier to watch (Kotlin for Wear OS now; Swift for Apple Watch once a test device exists ~Dec 2026) with the parity test fixtures. Measure battery.
5. Sub-second behaviour: measure 1 s and 2 s sliding windows for any new model; the 5 s window in earlier benches was a test choice, not a limit.

## Split suggestion
- Claude Code (needs the repo, Mac files, big local runs): steps 1, 4, anything touching `server/` or `apps/`.
- Codex / any other AI (web or API only, can clone this file): step 2 literature + model-card checks, GPT-6.1 Terra audio test via the free Codex account on CREMA-D clips, using `tmp/research-prompts/audio-emotion-deep-research-prompt.md` (copy its text into the Codex task).

## Update after Gemini deep-research report (2026-11-05)
- Drop Apriel Voice 27B (CC BY-NC-SA) and LFM3-Audio-2B (950 MB). Kimi K3.1 audio unknown, too big.
- Add to judge-trust harness: (a) text-leakage test with lexically contradictory audio (neutral/negative words in cheerful voice and vice versa, via CREMA-D neutral sentences + TTS), (b) emotion2vec+ as an audio-only labeller/baseline (verify its weight licence first), (c) labels as probability distributions with soft targets; human crowd kappa on natural speech is only ~0.36, so do not demand 100% agreement.
- Whisper-encoder students stay the model path (report's own benchmark: Whisper encoders best, bigger worse). QCNN not needed: our distilled Whisper-tiny is 9.6 MB, inside a 25 MB watch budget.
- Do not use Common Voice as the emotion pool (read speech); keep AMI, YODAS2, People's Speech, consented opt-in audio.
- Compliance task: explicit granular opt-in for voice-emotion analysis, ephemeral processing, no voiceprint retention; EU AI Act forbids emotion inference in workplaces, so do not market the coach for work meetings in the EU.
