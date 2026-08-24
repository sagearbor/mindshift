/**
 * Can this device run the on-device fast loop at all?
 *
 * The gate is on-device STT: without words there is nothing to coach on
 * locally, so the legacy server path (PCM over the WebSocket, cloud STT +
 * LLM + TTS) is the honest choice. Everything else degrades inside the loop
 * (no Silero -> energy VAD; no ECAPA -> no speaker-ID; no local LLM -> the
 * cloud's suggestion event). Synchronous so the hook can read it as initial
 * state; never throws.
 */
import { Platform } from "react-native";
import { onDeviceSttAvailable } from "./expoStt";

export interface LiveCapability {
  capable: boolean;
  reason: string;
}

export function detectLiveCapability(): LiveCapability {
  if (Platform.OS === "web") {
    return { capable: false, reason: "on-device coaching needs the native app" };
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
