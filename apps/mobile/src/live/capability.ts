/**
 * Can this device run the on-device fast loop at all?
 *
 * The gate is client-side STT (the OS recognizer on a phone, the Web Speech
 * API in a browser): without words there is nothing to coach on locally, so
 * the legacy server path (PCM over the WebSocket, cloud STT + LLM + TTS) is
 * the honest choice. Everything else degrades inside the loop
 * (no Silero -> energy VAD; no ECAPA -> no speaker-ID; no local LLM -> the
 * cloud's suggestion event). Synchronous so the hook can read it as initial
 * state; never throws.
 */
import { Platform } from "react-native";
import { onDeviceSttAvailable } from "./expoStt";
import { webSttAvailable } from "./sttWeb";
import { isWebAudioCaptureSupported } from "../utils/webAudioCapture";

export interface LiveCapability {
  capable: boolean;
  reason: string;
}

export function detectLiveCapability(): LiveCapability {
  if (Platform.OS === "web") {
    // The browser build: the Web Speech API is the recognizer (iOS Safari,
    // Chrome) and getUserMedia + AudioWorklet feed the loop. Both are
    // probed without touching them (no prompts here).
    let stt = false;
    let mic = false;
    try {
      stt = webSttAvailable();
      mic = isWebAudioCaptureSupported();
    } catch {
      stt = false;
    }
    if (!mic) return { capable: false, reason: "this browser can't capture the microphone" };
    if (!stt) return { capable: false, reason: "this browser has no speech recognition (use Safari on iPhone, or Chrome)" };
    return { capable: true, reason: "browser speech recognition available" };
  }
  let stt = false;
  try {
    stt = onDeviceSttAvailable();
  } catch {
    stt = false;
  }
  if (!stt) {
    return { capable: false, reason: "on-device speech recognition isn't available here" };
  }
  return { capable: true, reason: "on-device speech recognition available" };
}
