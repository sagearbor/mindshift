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

/**
 * In-app call (Call mode): keep the record-oriented session AND choose where
 * the other person's voice comes out. Android: expo-audio maps
 * `shouldRouteThroughEarpiece` onto AudioManager's mode + speakerphone
 * (earpiece = MODE_IN_COMMUNICATION + speakerphone off; speaker =
 * MODE_NORMAL + speakerphone on) — react-native-webrtc has no route API of
 * its own. iOS ignores the flag (WebRTC's own audio session decides; the
 * native iOS app is not a call target — Mom uses Safari). Web no-ops.
 */
export async function setCallAudioRoute(route: "speaker" | "earpiece"): Promise<void> {
  if (Platform.OS === "web") return;
  await setAudioModeAsync({
    allowsRecording: true,
    playsInSilentMode: true,
    shouldRouteThroughEarpiece: route === "earpiece",
  });
}
