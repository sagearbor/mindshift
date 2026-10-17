/**
 * Is a PRIVATE audio route (Bluetooth / wired headset) connected right now?
 *
 * Earpiece mode must never fall back to the loudspeaker (2026-10-07 dinner
 * test: the owner took the earpiece out and the coach carried on OUT LOUD on
 * the phone speaker — in a work meeting that is a disaster). expo-speech has
 * no output-route control: Android's TextToSpeech plays on whatever the media
 * route is, which becomes the loudspeaker the moment the headset disconnects.
 * So the hook asks this module before EVERY utterance, listens for route
 * changes while an earpiece session runs (polling as a backstop), and stays
 * silent unless the answer is "private".
 *
 * TWO PROBES, best first:
 *
 * 1. NATIVE OUTPUT ROUTE (runtime 1.19.0+, local module
 *    `modules/audio-route`, native name "MindShiftAudioRoute"). Reads the
 *    devices audio actually PLAYS on:
 *    - Android: `AudioManager.getAudioDevicesForAttributes(USAGE_MEDIA)` (API
 *      33+, the route TTS will take) plus `getDevices(GET_DEVICES_OUTPUTS)`,
 *      the communication device + audio mode (API 31+), and events from
 *      `AudioDeviceCallback` / `OnCommunicationDeviceChangedListener`.
 *    - iOS: `AVAudioSession.currentRoute.outputs` and
 *      `routeChangeNotification`.
 *    This sees A2DP-only headphones and Bluetooth LE Audio earbuds
 *    (TYPE_BLE_HEADSET — Pixel 10 + recent Pixel Buds), which the input probe
 *    cannot, and a disconnect arrives as an event instead of on the next poll.
 *
 * 2. INPUT LIST FALLBACK (no native module: the web build, or a 1.18.0
 *    binary that received this JS over the air). expo-audio's
 *    `AudioRecorder.getAvailableInputs()` — a never-started recorder, so it
 *    does not touch the live microphone. It only sees headsets WITH a mic
 *    that expo-audio recognises (Bluetooth SCO / wired / iOS HFP), so A2DP-
 *    only and LE Audio devices read as "public" there (silent, never loud).
 *
 * FAIL-SILENT RULE (both probes): anything unreadable is "unknown", and the
 * hook treats "unknown" exactly like "public". The browser exposes no output
 * route → "unknown" → silent in earpiece mode on the web build.
 *
 * KNOWN LIMIT: a Bluetooth car kit is an A2DP device, and nothing in
 * AudioDeviceInfo / AVAudioSession tells a car from headphones without the
 * BLUETOOTH_CONNECT permission (Android) — so with a car kit as the media
 * route the native probe says "private". The phone's own speaker, BLE
 * speakers, Auracast broadcast, AirPlay, CarPlay (iOS "CarAudio"), HDMI,
 * line-out and generic USB devices are never counted.
 */
import { Platform } from "react-native";

/** private = a headset is connected; public = only the phone's own mic/speaker;
 *  unknown = the platform cannot say (treated exactly like public). */
export type AudioRouteState = "private" | "public" | "unknown";

export interface AudioRouteProbe {
  /** Synchronous: called right before every utterance. Never throws. */
  check(): AudioRouteState;
  /** Route-change events, when the platform has them (native module). The
   *  listener gets the new state; returns an unsubscribe. Optional so test
   *  doubles and older probes without events still satisfy the type. */
  subscribe?(onChange: (state: AudioRouteState) => void): () => void;
  /** Which probe answered — "native" (output route) or "inputs" (fallback). */
  source?(): "native" | "inputs" | "none";
  /** Free any native object / listener the probe holds. */
  dispose(): void;
}

/** How often an earpiece session re-checks the route (ms). A backstop now:
 *  with the native module a disconnect arrives as an event. */
export const ROUTE_POLL_MS = 500;

// ---------------------------------------------------------------------------
// Native output route
// ---------------------------------------------------------------------------

/** What `MindShiftAudioRoute.getRoute()` returns (and `onRouteChange` sends).
 *  Android types are AudioDeviceInfo.TYPE_* names without the prefix
 *  ("BLE_HEADSET"); iOS types are AVAudioSession.Port raw values
 *  ("BluetoothA2DPOutput"). */
export interface NativeRouteSnapshot {
  platform: "android" | "ios";
  /** Every connected output device. */
  outputs: string[];
  /** The device(s) media playback (TTS) routes to right now; null when the OS
   *  cannot say (Android < 13). */
  active: string[] | null;
  /** Android 12+: the communication device (calls / VoIP), else null. */
  communication: string | null;
  /** Android audio mode: "normal" | "ringtone" | "call" | "communication" |
   *  "call_screening" | "unknown"; null on iOS. */
  mode: string | null;
}

export interface NativeAudioRouteModule {
  getRoute(): NativeRouteSnapshot;
  addListener(event: "onRouteChange", listener: (snapshot: NativeRouteSnapshot) => void): { remove(): void };
}

/** The native module's registered name (modules/audio-route). */
export const NATIVE_AUDIO_ROUTE_MODULE = "MindShiftAudioRoute";

/**
 * Output device types that mean "only the wearer hears it".
 * Android: AudioDeviceInfo.TYPE_* (BLE_HEADSET is API 31+, HEARING_AID 28+,
 * USB_HEADSET 26+ — the constants are just ints, so older OS versions simply
 * never report them). iOS: AVAudioSession.Port raw values — USBAudio is
 * included because USB-C wired earbuds (iPhone 15+) report as it.
 * NOT private: BUILTIN_SPEAKER / BUILTIN_EARPIECE / "Speaker" / "Receiver"
 * (the phone itself, not worn), BLE_SPEAKER, BLE_BROADCAST (Auracast),
 * USB_DEVICE, HDMI, line out, dock, AirPlay, CarAudio.
 */
export const PRIVATE_OUTPUT_TYPES: readonly string[] = [
  // Android
  "BLUETOOTH_A2DP",
  "BLUETOOTH_SCO",
  "BLE_HEADSET",
  "WIRED_HEADSET",
  "WIRED_HEADPHONES",
  "USB_HEADSET",
  "HEARING_AID",
  // iOS
  "Headphones",
  "BluetoothA2DPOutput",
  "BluetoothHFP",
  "BluetoothLE",
  "USBAudio",
];

const isPrivateType = (t: unknown) => typeof t === "string" && PRIVATE_OUTPUT_TYPES.includes(t);
const isStringArray = (v: unknown): v is string[] => Array.isArray(v) && v.every((x) => typeof x === "string");

export function classifyOutputRoute(snapshot: NativeRouteSnapshot | null | undefined): AudioRouteState {
  if (!snapshot || typeof snapshot !== "object" || !isStringArray(snapshot.outputs)) return "unknown";
  const { outputs, active, communication, mode } = snapshot;
  // In a call / VoIP session the communication device can carry our audio too:
  // if it is not private, do not trust the media route alone.
  if ((mode === "call" || mode === "communication") && typeof communication === "string" && !isPrivateType(communication)) {
    return "public";
  }
  if (isStringArray(active) && active.length > 0) {
    // The OS told us where media plays: private only if ALL of it is private.
    return active.every(isPrivateType) ? "private" : "public";
  }
  // Android < 13: no route answer. A connected headset takes media routing,
  // so any private output counts.
  return outputs.some(isPrivateType) ? "private" : "public";
}

function defaultLoadNative(): NativeAudioRouteModule | null {
  if (Platform.OS === "web") return null;
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { requireOptionalNativeModule } = require("expo") as typeof import("expo");
  return requireOptionalNativeModule<NativeAudioRouteModule>(NATIVE_AUDIO_ROUTE_MODULE);
}

// ---------------------------------------------------------------------------
// Input-list fallback (runtime 1.18.0 / no native module)
// ---------------------------------------------------------------------------

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

type InputLister = { getAvailableInputs(): { type?: unknown }[]; release?: () => void };

const NO_UNSUBSCRIBE = () => {};

function inputListProbe(makeLister: () => InputLister): AudioRouteProbe {
  let lister: InputLister | null = null;
  let broken = false;
  return {
    check() {
      if (broken) return "unknown";
      try {
        if (!lister) lister = makeLister();
        return classifyInputs(lister.getAvailableInputs());
      } catch {
        broken = true;
        return "unknown";
      }
    },
    subscribe: () => NO_UNSUBSCRIBE,
    source: () => "inputs",
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

function nativeProbe(native: NativeAudioRouteModule): AudioRouteProbe {
  const subscriptions = new Set<{ remove(): void }>();
  return {
    check() {
      try {
        return classifyOutputRoute(native.getRoute());
      } catch {
        return "unknown";
      }
    },
    subscribe(onChange) {
      let sub: { remove(): void } | null = null;
      try {
        sub = native.addListener("onRouteChange", (snapshot) => {
          let state: AudioRouteState;
          try {
            state = classifyOutputRoute(snapshot);
          } catch {
            state = "unknown";
          }
          onChange(state);
        });
        subscriptions.add(sub);
      } catch {
        return NO_UNSUBSCRIBE; // no events: the poll still covers it
      }
      return () => {
        if (!sub) return;
        subscriptions.delete(sub);
        try {
          sub.remove();
        } catch {
          // Already gone.
        }
        sub = null;
      };
    },
    source: () => "native",
    dispose() {
      for (const sub of subscriptions) {
        try {
          sub.remove();
        } catch {
          // Already gone.
        }
      }
      subscriptions.clear();
    },
  };
}

/** The browser has no route API: always "unknown" (silent in earpiece mode). */
const UNKNOWN_PROBE: AudioRouteProbe = {
  check: () => "unknown",
  subscribe: () => NO_UNSUBSCRIBE,
  source: () => "none",
  dispose: () => {},
};

/**
 * The production probe: the native output-route module when the binary has it,
 * else the input list. `makeLister` and `loadNative` are seams for tests.
 */
export function createDefaultAudioRouteProbe(
  makeLister?: () => InputLister,
  loadNative: () => NativeAudioRouteModule | null = defaultLoadNative,
): AudioRouteProbe {
  let native: NativeAudioRouteModule | null = null;
  try {
    native = loadNative();
  } catch {
    native = null;
  }
  if (native && typeof native.getRoute === "function") return nativeProbe(native);
  if (!makeLister && Platform.OS === "web") return UNKNOWN_PROBE;
  return inputListProbe(
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
      }),
  );
}
