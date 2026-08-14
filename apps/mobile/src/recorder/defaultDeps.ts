import { Platform } from "react-native";
import { setAudioModeAsync } from "expo-audio";
import * as Battery from "expo-battery";
import type { AudioRecorderDeps } from "./AudioRecordScreen";
import { RECORDER_ENGINE } from "./engine";
import { ExpoRecorderFs } from "./expoFs";
import { ExpoPcmSource } from "./expoPcmSource";
import {
  formatForPlatform,
  makeExpoRecorderFactory,
} from "./expoRecorderPort";
import { RecorderSessionStore } from "./sessionStore";
import { STREAM_WAV_FORMAT } from "./streamSession";

/** Battery level as a 0..1 fact, or null when the platform can't say (some
 *  simulators report -1) — never a guess. */
async function getBatteryLevel(): Promise<number | null> {
  try {
    const level = await Battery.getBatteryLevelAsync();
    return typeof level === "number" && level >= 0 && level <= 1 ? level : null;
  } catch {
    return null;
  }
}

/**
 * Configure the OS audio session around a recording session.
 * `allowsBackgroundRecording` + `doNotMix` is what keeps a screen-off,
 * hour-long session alive (paired with the expo-audio config plugin's
 * `enableBackgroundRecording`, a NATIVE change — see app.json).
 * Deactivation mirrors utils/audioMode.setPlaybackMode so later media replay
 * isn't silenced on Android.
 */
async function configureAudioSession(active: boolean): Promise<void> {
  if (active) {
    await setAudioModeAsync({
      allowsRecording: true,
      playsInSilentMode: true,
      allowsBackgroundRecording: true,
      interruptionMode: "doNotMix",
    });
    return;
  }
  await setAudioModeAsync({
    allowsRecording: false,
    playsInSilentMode: true,
    allowsBackgroundRecording: false,
  });
}

/** The production wiring for the audio recorder: expo-audio capture,
 *  expo-file-system persistence, expo-battery preflight. Built lazily so
 *  tests (which inject fakes) and web never construct native objects.
 *
 *  Engine comes from the RECORDER_ENGINE constant — default "stream" (v2
 *  gapless: continuous PCM stream into WAV files on BOTH platforms). Both
 *  wirings ship so flipping that one constant (or a deps override) falls
 *  back to the v1 segmented recorder on a device where the realtime stream
 *  API misbehaves. */
export function defaultAudioRecorderDeps(): AudioRecorderDeps {
  const engine = RECORDER_ENGINE;
  return {
    store: new RecorderSessionStore(new ExpoRecorderFs()),
    makeRecorder: makeExpoRecorderFactory(),
    makePcmSource: () => new ExpoPcmSource(),
    engine,
    // Preflight storage math must match what the active engine writes:
    // stream = 16 kHz WAV everywhere; v1 = per-platform (AAC on Android).
    format:
      engine === "stream" ? STREAM_WAV_FORMAT : formatForPlatform(Platform.OS),
    getBatteryLevel,
    configureAudioSession,
  };
}
