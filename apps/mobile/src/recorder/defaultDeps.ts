import { Platform } from "react-native";
import {
  requestNotificationPermissionsAsync,
  setAudioModeAsync,
} from "expo-audio";
import * as Battery from "expo-battery";
import type { AudioRecorderDeps } from "./AudioRecordScreen";
import { ExpoRecorderFs } from "./expoFs";
import {
  formatForPlatform,
  makeExpoRecorderFactory,
} from "./expoRecorderPort";
import { RecorderSessionStore } from "./sessionStore";

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
 * The audio-session plan for a recording, given the platform and whether the
 * NOTIFICATION permission is granted. Android's background recording runs a
 * foreground service whose persistent notification is mandatory — without the
 * permission, `prepareToRecordAsync` REJECTS outright (found on-device,
 * v1.16.0). Denied → degrade to a foreground-only session that still records
 * (screen must stay on), honestly flagged via `backgroundCapable`. iOS uses
 * UIBackgroundModes audio and needs no notification permission.
 */
export function recordingSessionPlan(
  os: string,
  notificationsGranted: boolean,
): {
  backgroundCapable: boolean;
  mode: {
    allowsRecording: true;
    playsInSilentMode: true;
    allowsBackgroundRecording: boolean;
    interruptionMode: "doNotMix";
  };
} {
  const backgroundCapable = os !== "android" || notificationsGranted;
  return {
    backgroundCapable,
    mode: {
      allowsRecording: true,
      playsInSilentMode: true,
      allowsBackgroundRecording: backgroundCapable,
      interruptionMode: "doNotMix",
    },
  };
}

/**
 * Configure the OS audio session around a recording session.
 * `allowsBackgroundRecording` + `doNotMix` is what keeps a screen-off,
 * hour-long session alive (paired with the expo-audio config plugin's
 * `enableBackgroundRecording`, a NATIVE change — see app.json).
 * Deactivation mirrors utils/audioMode.setPlaybackMode so later media replay
 * isn't silenced on Android.
 */
async function configureAudioSession(
  active: boolean,
): Promise<{ backgroundCapable: boolean } | void> {
  if (active) {
    let notificationsGranted = true;
    if (Platform.OS === "android") {
      try {
        const p = await requestNotificationPermissionsAsync();
        notificationsGranted = p.granted;
      } catch {
        notificationsGranted = false;
      }
    }
    const plan = recordingSessionPlan(Platform.OS, notificationsGranted);
    await setAudioModeAsync(plan.mode);
    return { backgroundCapable: plan.backgroundCapable };
  }
  await setAudioModeAsync({
    allowsRecording: false,
    playsInSilentMode: true,
    allowsBackgroundRecording: false,
  });
}

/** The production wiring for the audio recorder: expo-audio capture,
 *  expo-file-system persistence, expo-battery preflight. Built lazily so
 *  tests (which inject fakes) and web never construct native objects. */
export function defaultAudioRecorderDeps(): AudioRecorderDeps {
  return {
    store: new RecorderSessionStore(new ExpoRecorderFs()),
    makeRecorder: makeExpoRecorderFactory(),
    format: formatForPlatform(Platform.OS),
    getBatteryLevel,
    configureAudioSession,
  };
}
