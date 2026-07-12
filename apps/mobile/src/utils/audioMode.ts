import { Platform } from "react-native";
import { setAudioModeAsync } from "expo-audio";

/**
 * Central control of the device audio SESSION (category/mode), distinct from any
 * one player's volume. The Live Coach mic-capture flow and media replay want the
 * session configured differently, and — critically — the two must not leak into
 * each other:
 *
 * A recording-oriented session (allowsRecording) leaves the OS audio session in
 * a record configuration. On Android that silences subsequent media playback
 * (expo-video plays but you hear nothing) until the session is reset to a
 * playback configuration. That is exactly the "replay has no sound after using
 * Live Coach" bug: the live flow set the record session and nothing ever put it
 * back. These two helpers make the transition explicit and symmetric.
 *
 * Web has no configurable native audio session, so both calls no-op there
 * (calling setAudioModeAsync on web is unnecessary and its options are native).
 */

/**
 * Configure the session for microphone capture (Live Coach). `playsInSilentMode`
 * keeps coaching audio audible even with the ringer silenced.
 */
export async function setRecordingMode(): Promise<void> {
  if (Platform.OS === "web") return;
  await setAudioModeAsync({
    allowsRecording: true,
    playsInSilentMode: true,
  });
}

/**
 * Configure the session for media PLAYBACK (replay). Turning `allowsRecording`
 * back off is what actually restores audible playback after a recording session;
 * without it, a Pixel plays the replay silently.
 */
export async function setPlaybackMode(): Promise<void> {
  if (Platform.OS === "web") return;
  await setAudioModeAsync({
    allowsRecording: false,
    playsInSilentMode: true,
  });
}
