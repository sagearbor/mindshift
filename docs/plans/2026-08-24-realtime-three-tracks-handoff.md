# Realtime coaching build — three-track handoff (2026-08-24)

> **Read this if you're a fresh Claude Code session (or Sage) picking up the
> realtime work.** It supersedes the "poker6 diarization status" half of
> [2026-08-24-mac-transition-and-poker6-status.md](2026-08-24-mac-transition-and-poker6-status.md)
> (poker6 is RESOLVED, see §1). Plan of record:
> `~/.claude/plans/glittery-doodling-kay.md` on Sage's Mac (mirrored in §2).

## 0. The one-paragraph version

The app now has an **on-device-first realtime coaching loop** on the phone
(capture → Silero VAD → speaker-ID → on-device STT → local LLM → expo-speech
earpiece, all without the server in the critical path), the **server turned
into async enrichment** (cloud suggestion, audio tone, identity confirmation,
watch relay, latency telemetry), **live sessions flow into the same
over-time analysis** as uploads (Growth/YourDay/Dynamics/Therapist + a cached
"what you could have said" pass), and the **watch nudge** shares one
escalation policy with the phone via golden test vectors and now escalates
on tone, not just volume. Everything is merged to `main`; every suite is
green. What's left is on-device verification on real hardware (§6).

## 1. Poker6 — resolved (PR #129)

The shipped ECAPA k-search was rejecting the real 6th voice by a hair
(marginal cosine 0.301 vs the 0.30 bar; anchor 0.231 vs 0.20). Recalibrated
`STRONG_SEPARATION_COSINE` 0.30→0.32 and `NEW_VOICE_ANCHOR_COSINE` 0.20→0.24
against every real, checked-in fixture: **poker6 6/6 = 100%**, no new deps,
no manual speaker count (the owner's hard rule). Two tests that "protected"
the old values were built on a fixture the repo's own README says never to
use for diarization (`tmp/test_recording.wav`, gain/tempo-modulated single
voice) — repaired to real fixtures. pyannote.audio was evaluated and archived
(`docs/research/poker6-sliding-window/`): 83% at best, ~800 MB of deps,
over-segments 2-speaker audio in auto mode.

## 2. What shipped (in merge order, all squash-merged to `main`)

| PR | What | Key numbers |
|---|---|---|
| #130 Foundation A | canonical `server/nudge_policy.py` (watch module re-exports), golden vectors `server/tests/fixtures/policy_vectors/{nudge_policy,vad_segments}.json`, realtime WS models `TurnLocalEvent`/`ToneFlagEvent`/`SpeakerIdentityEvent`, `SuggestionEvent.suggestion_source` | 14 nudge + 11 VAD vector cases |
| #131 Foundation C | `server/tone_id.py` — wav2vec2-IEMOCAP audio tone, pinned, lazy, `MINDSHIFT_TONE_AUDIO=off\|dark\|on` | **ships DARK**: 40% 4-class / 50% arousal on our acted fixtures — it reads voice identity, not emotion. Gain-invariant, ~150 ms/turn CPU, 758 MB cache. Report: `docs/research/tone-audio/` |
| #132 Foundation B | `scripts/export_ecapa_onnx.py` (+ parity test), multi-person named voiceprints (`voiceprints/{uid}/{person_id}/profile.json`, `identify_speakers_multi`, `/voice/people` CRUD; legacy single print read as `self`) | ONNX 80 MB opset-17, cosine(onnx, torch) = **1.000000**, **~15 ms per 1.5 s clip** on CPU |
| #133 Foundation D | `scripts/make_test_recording_scenes.py` + 3 labeled scenes (2/3/4 voices, self voice, `emotion_coarse`, `expected_nudges`), `test_diarize_scenes.py` | couple 13/13, family3 15/15, meeting4 **11/17 (pinned ceiling — OpenAI TTS voices are near-twins; self still isolated 6/6)**; +6.7 MB; ~$0.13 TTS |
| #134 | watch enroll tests made venv-independent | — |
| #135 Track 1 | Kotlin replays the golden vectors (Gradle `syncPolicyVectors`), PRD §6 haptic schedule (pure mapper + wearApp), `server/watch/relay.py::push_turn_local` (self-turns only; tone raises level; max with dB path), `tone_escalation.json` | 0 vector mismatches; wear emulator boot **662 ms, 0 crashes**, UI rendered |
| #136 Track 3-server | `audio_pipeline.py`: per-stage latency log + `latency_summary` on `session_complete`, `turn_local` handling (Deepgram overlap suppression, cloud suggestion w/ tone context, `partial` streaming previews, server TTS off when local-first), PCM ring buffer enrichment (audio tone → `tone_flag` if surfaced, identity → `speaker_identity`, watch relay) | 37 new tests; legacy clients byte-identical |
| #137 Track 2 | `POST /sessions/live` (idempotent, episodes = recordings with `media_type: none`), `POST /episodes/{id}/reflect` (cached "could have said"), `GET /sessions` for the therapist dashboard, tone + per-person dimensions in `/growth` and the screens | scene-pack ingest test: escalation turns == `expected_nudges`, no non-self leakage |
| #138 Track 3-mobile | `apps/mobile/src/live/`: `prosody.ts`, `ort.ts` (+node impl for Jest), `vad.ts`/`segmenter.ts` (Silero v6 committed, 2.3 MB), `speakerId.ts`, `stt.ts` (expo-speech-recognition), `localLlm.ts` (expo-ai-kit: Gemini Nano → Apple FM → LiteRT-LM bundled → cloud, refusal falls through), `nudgePolicy.ts`, `fastLoop.ts` (earpiece / speaker / therapist modes), wired into `useAudioStream`/`LiveCoachScreen` behind an STT capability gate; `plugins/withOrtGradle9.js` | Silero **0.26 ms/chunk**; all vectors replay; Android debug APK **builds and boots** on `pixel10_api35`, 0 crashes |

Suites on `main` after #138: pytest **1494 passed** (1 known live-Deepgram
network test deselected — pre-existing vendor issue), Jest **105 suites /
1344 tests**, `tsc` clean, Gradle `:shared:allTests` + `:wearApp` unit green.

## 3. What shipped DARK (built, measured, silenced — not forgotten)

- **Audio tone (wav2vec2-IEMOCAP)**: `MINDSHIFT_TONE_AUDIO=dark`. Computed
  and logged server-side per turn, never surfaced, never relayed to the
  watch. Text-tone (from the LLM call) is what drives tone today. Cheapest
  next experiment per the report: per-speaker logit normalization; better
  bet: a naturalistic dimensional arousal/valence model. **Ops note:** with
  voice deps installed the first ≥1 s `turn_local` triggers a one-time
  ~750 MB model fetch in a thread.
- **On-device speaker-ID**: fully implemented in the app; was inert at #138
  because the server served neither the ONNX model nor embeddings — closed
  by the follow-up PR listed in §7 (see there for the measured numbers).
- **Bundled LLM tier**: uses expo-ai-kit's `getRecommendedModel()`; Gemma 3
  1B was rejected because its HF download is gated (401) for end users.

## 4. Decisions worth knowing (so nobody re-litigates them)

- The fast loop never touches the server; the phone is the orchestrator.
  Older/incapable phones (no on-device STT) silently get the legacy server
  path — that IS the fallback, no separate code path.
- Cross-language sharing (Kotlin/TS/Python) is done with **golden test
  vectors**, not shared code. Add a case to the JSON → all three suites
  must pass it.
- One embedding model everywhere (the pinned speechbrain ECAPA, exported to
  ONNX) so phone/batch/watch voiceprints stay compatible. Not WeSpeaker,
  not FluidAudio.
- `turn_local.is_self` is a voiceprint verdict only; the watch escalates on
  self turns only (bias guard cases in `tone_escalation.json`).
- Never speak over live speech (earpiece and speaker-phone); therapist
  mode never speaks; cloud suggestion is voiced only if every local
  provider fell through.
- Native modules ⇒ **dev build**, never Expo Go, and never raw `eas update`
  (AGENTS.md).

## 5. Machine/toolchain state on Sage's new Mac (all installed 2026-08-24)

JDK 17 (`/opt/homebrew/opt/openjdk@17`), Android SDK at
`~/Library/Android/sdk` (platform 35 + 34, build-tools, emulator, `adb`,
cmdline-tools copied into the SDK root so `avdmanager` works), Android
Studio, AVDs `pixel10_api35` and `wear_os5`, `JAVA_HOME`/`ANDROID_HOME` in
`~/.zprofile`; `apps/watch/local.properties` (gitignored). Python venvs in
`tmp/` (`venv-voice` has torch+speechbrain+transformers+onnx/onnxruntime;
`venv-whisper` runs the plain suite; `venv-pyannote` is research-only).
`HF_TOKEN` in `.env`. A LaunchAgent keeps the Mac awake for Sage's account.
**Not installed: Xcode.app** (App Store only; `mas` needs an interactive
sudo). Claude Code's sandbox shell does not source `~/.zprofile` — prefix
`export PATH="/opt/homebrew/bin:$PATH"`.

## 6. Only-Sage items (nothing else is blocked on these)

1. **Xcode**: App Store → Xcode → `sudo xcode-select -s /Applications/Xcode.app`.
   Until then iOS native (SpeechAnalyzer / Apple FM providers via
   expo-ai-kit) is written and Jest-tested but never compiled here.
2. **Pixel 10**: enable Developer options → USB debugging; then from
   `apps/mobile`: `npx expo run:android --device` (dev build; ~10 min first
   time). On first launch the app downloads `ecapa.onnx` (~80 MB) once.
3. **Enroll voices** in the app: yourself ("You") and "Mom" as a named
   person (new multi-person enrollment in Voice settings) before the demo.
4. **Demo run** (speaker-phone mode): start a live session in `speaker`
   mode, call Mom on speaker, watch the on-screen transcript label
   self/Mom, hear suggestions only while you're silent; at session end the
   per-stage latency log prints (target < 1.5 s to first spoken words —
   this number has NOT been measured on real hardware yet) and the session
   appears in Your Day / Growth with the "could have said" reflections.
5. Apple Intelligence toggle on any iPhone 15 Pro+ you test with.

## 7. Next steps, in order

1. Follow-up PR "speaker-ID seam" (`GET /models/ecapa.onnx`,
   `GET /voice/people?include_embeddings=true`, client download/ETag) —
   see the PR for measured ONNX parity/latency under onnxruntime-node.
2. Real-hardware pass on the Pixel 10: Gemini Nano Prompt API refusal
   behaviour on coaching prompts (unknown until tried), on-device STT
   quality, per-stage latency, whether speaker-ID separates self/Mom
   through a speaker-phone mic (poker6/family fixtures say ECAPA can).
3. If Nano refuses: expo-ai-kit's LiteRT-LM tier is the middle rung; cloud
   is the floor. Measure, don't assume.
4. Tone: run the per-speaker-normalization experiment from the tone
   report; flip `MINDSHIFT_TONE_AUDIO=on` only when it beats text-tone on
   the scene pack.
5. Apple Watch: accepted gap (~6 months).
