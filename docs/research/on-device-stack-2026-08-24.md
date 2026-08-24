# On-device realtime coaching stack — research brief (verified 2026-08-24)

Baseline: Expo SDK 57.0.16 = RN 0.86, React 19.2. Everything below needs a **dev build** (no Expo Go).

## Decision table
| Item | Pick | Fallback |
|---|---|---|
| Gemini Nano (Android) | expo-ai-kit 0.14.1 (Pixel 9/10, nano-v3) | custom Kotlin Expo module on `com.google.mlkit:genai-prompt:1.0.0-beta4` |
| Apple Foundation Models | expo-ai-kit | @react-native-ai/apple 0.12 |
| Bundled small LLM (middle tier) | expo-ai-kit's LiteRT-LM + Gemma 3 1B int4 `.litertlm` (on-demand download) — one owner of the LiteRT native lib; do NOT also add react-native-litert-lm | llama.rn 0.12.9 + GGUF |
| ONNX runtime | onnxruntime-react-native 1.24.3, config plugin, CPU/XNNPACK EPs first (CoreML/NNAPI degrade on dynamic shapes → keep inputs fixed-shape) | react-native-nitro-onnxruntime |
| VAD | silero_vad.onnx (v6 repo, MIT, 2.3 MB, opset 16): input f32 [1,576] = 64 context + 512 new @16k, state f32 [2,1,128], sr int64; outputs prob + stateN; <1 ms/chunk. Port from @ricky0123/vad-web `models/v5.ts` but add the official context concat | silero_vad_16k_op15.onnx |
| STT | expo-speech-recognition 56.0.1 (`requiresOnDeviceRecognition`, `continuous`, `interimResults`, `androidTriggerOfflineModelDownload`, `androidRecognitionServicePackage:'com.google.android.as'`); SDK 57 compat unverified; known: #165 stop() in continuous emits ERROR_CLIENT on Android | expo-ai-kit `streamTranscription` (SpeechAnalyzer iOS 26; ML Kit "advanced" on Pixel 10) |
| Speaker embedding | OUR pinned ECAPA exported to ONNX (voiceprint compatibility with server/batch/watch beats everything) | sherpa-onnx WeSpeaker ResNet34 / CAM++ ONNX (26–29 MB, proven on mobile ORT) if ECAPA ops fail on mobile |

## Facts that constrain the design
- Gemini Nano Prompt API: beta ("no SLA"), input <4000 tokens, `generateContentStream()`; structured output is Kotlin-only best-effort → parse JSON defensively. Safety filters non-configurable, categories undocumented → refusal behaviour on relationship-coaching prompts is UNKNOWN; must test on the Pixel 10 and fall through on refusal.
- Apple FM: iOS 26+, A17 Pro+ (iPhone 15 Pro, all 16/17). Guardrails block self-harm/violence/adult on input+output; forum reports over-blocking; spurious guardrailViolation when safety assets aren't downloaded. 4,096-token context on 26.0–26.3. Expect some refusals → fall through.
- Bundled LLM speeds (published): Gemma 3 1B int4 via LiteRT-LM ≈ 55 tok/s CPU on S24 Ultra, ~1.0–1.2 GB RAM. Models ≥2 GB need iOS Extended Virtual Addressing entitlement — stay at 1B.
- expo-audio SDK 57: `useAudioStream({encoding:'float32', sampleRate:16000, channels:1})` → `buffer.data: ArrayBuffer` → `new Float32Array(data)` → `new ort.Tensor('float32', arr, [1,n])`. Actual sampleRate may differ → resample defensively (the repo's PcmFrame already reports the true rate).
- ORT-RN on Expo iOS: copy the Metro asset to `documentDirectory` and strip `file://` before `InferenceSession.create` (issue #27062).

## Unverified (treat as risks, measure on device)
Nano/Apple refusal rates on coaching prompts; binary size of LiteRT-LM; Pixel 10 tok/s; expo-speech-recognition on SDK 57; whether RN Apple modules do real guided generation; per-segment ECAPA latency on phone ORT.

Sources: developers.google.com/ml-kit/genai (prompt, structured-output, evaluate-prompt); github.com/saidkaban/expo-ai-kit; developer.apple.com/documentation/foundationmodels (+ safety doc); github.com/corasan/react-native-foundation-models; github.com/hung-yueh/react-native-litert-lm; huggingface.co/litert-community/Gemma3-1B-IT; github.com/microsoft/onnxruntime js/react_native; onnxruntime.ai/docs/tutorials/mobile; github.com/snakers4/silero-vad; github.com/ricky0123/vad; github.com/jamsch/expo-speech-recognition; github.com/k2-fsa/sherpa-onnx speaker models; github.com/FluidInference/FluidAudio; docs.expo.dev/versions/v57.0.0/sdk/audio.
