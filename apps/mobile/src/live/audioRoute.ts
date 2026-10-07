/**
 * Is a PRIVATE audio route (Bluetooth / wired headset) connected right now?
 *
 * Earpiece mode must never fall back to the loudspeaker (2026-10-07 dinner
 * test: the owner took the earpiece out and the coach carried on OUT LOUD on
 * the phone speaker — in a work meeting that is a disaster). expo-speech has
 * no output-route control: Android's TextToSpeech plays on whatever the media
 * route is, which becomes the loudspeaker the moment the headset disconnects.
 * So the hook asks this module before EVERY utterance and polls it while an
 * earpiece session runs, and stays silent unless the answer is "private".
 *
 * JS-only, no native build: the probe is expo-audio's (already shipped)
 * `AudioRecorder.getAvailableInputs()`. On Android that is
 * `AudioManager.getDevices(GET_DEVICES_INPUTS)` filtered by expo-audio to the
 * built-in mic, Bluetooth SCO (a connected hands-free headset/earpiece) and a
 * wired headset; on iOS it is `AVAudioSession.availableInputs` (port types
 * `BluetoothHFP`, `HeadsetMic`, …). The recorder object is created once and
 * never prepared or started — Android's MediaRecorder is lazy, so it does not
 * touch the microphone the live stream is using.
 *
 * KNOWN LIMITS of the JS-only probe (all fail SAFE — silence, never the
 * loudspeaker):
 * - It sees INPUT devices. A headset with no microphone (A2DP-only
 *   headphones) and Bluetooth LE Audio earbuds (Android TYPE_BLE_HEADSET,
 *   which expo-audio's filter drops) read as "public" → earpiece mode stays
 *   silent with them. Fixing that needs a native module that reads OUTPUT
 *   devices (Android `AudioManager.getDevices(GET_DEVICES_OUTPUTS)` /
 *   `AudioDeviceCallback`, iOS `AVAudioSession.currentRoute.outputs` +
 *   `routeChangeNotification`) — a native build + runtime bump.
 * - Detection of a mid-utterance disconnect is by polling
 *   (ROUTE_POLL_MS); up to that long of the line in progress can still come
 *   out of the speaker before Speech.stop() cuts it. A native route-change
 *   callback would make it immediate.
 * - The browser exposes no output route at all → "unknown" → silent in
 *   earpiece mode on the web build.
 * - A car kit is NOT private (it is a loudspeaker) and is never counted.
 */
import { Platform } from "react-native";

/** private = a headset is connected; public = only the phone's own mic/speaker;
 *  unknown = the platform cannot say (treated exactly like public). */
export type AudioRouteState = "private" | "public" | "unknown";

export interface AudioRouteProbe {
  /** Synchronous: called right before every utterance. Never throws. */
  check(): AudioRouteState;
  /** Free any native object the probe holds. */
  dispose(): void;
}

/** How often an earpiece session re-checks the route (ms). */
export const ROUTE_POLL_MS = 500;

/**
 * Input port types that mean "a headset worn by the user is connected".
 * Android names come from expo-audio's AudioUtils.kt; iOS names are
 * AVAudioSession.Port raw values. Car audio / USB / AirPlay are excluded on
 * purpose (they can be loudspeakers).
 */
export const PRIVATE_INPUT_TYPES: readonly string[] = [
  // Android (expo-audio AudioUtils.getMapFromDeviceInfo)
  "BluetoothSCO",
  "BluetoothA2DP",
  "MicrophoneWired",
  // iOS (AVAudioSession.Port)
  "BluetoothHFP",
  "BluetoothLE",
  "HeadsetMic",
];

export function classifyInputs(inputs: readonly { type?: unknown }[] | null | undefined): AudioRouteState {
  if (!Array.isArray(inputs)) return "unknown";
  for (const input of inputs) {
    if (typeof input?.type === "string" && PRIVATE_INPUT_TYPES.includes(input.type)) return "private";
  }
  return "public";
}

/** The browser has no route API: always "unknown" (silent in earpiece mode). */
const UNKNOWN_PROBE: AudioRouteProbe = { check: () => "unknown", dispose: () => {} };

type InputLister = { getAvailableInputs(): { type?: unknown }[]; release?: () => void };

/**
 * The production probe. `makeLister` is a seam for tests; by default it is a
 * never-started expo-audio AudioRecorder.
 */
export function createDefaultAudioRouteProbe(makeLister?: () => InputLister): AudioRouteProbe {
  if (!makeLister && Platform.OS === "web") return UNKNOWN_PROBE;
  let lister: InputLister | null = null;
  let broken = false;
  const factory =
    makeLister ??
    (() => {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { AudioModule } = require("expo-audio") as typeof import("expo-audio");
      if (!AudioModule?.AudioRecorder) throw new Error("expo-audio AudioRecorder unavailable");
      return new AudioModule.AudioRecorder({
        extension: ".aac",
        sampleRate: 8000,
        numberOfChannels: 1,
        bitRate: 16000,
      }) as unknown as InputLister;
    });
  return {
    check() {
      if (broken) return "unknown";
      try {
        if (!lister) lister = factory();
        return classifyInputs(lister.getAvailableInputs());
      } catch {
        broken = true;
        return "unknown";
      }
    },
    dispose() {
      try {
        lister?.release?.();
      } catch {
        // Nothing further to free.
      }
      lister = null;
    },
  };
}
