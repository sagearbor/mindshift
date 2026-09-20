/**
 * Production wiring for the Journal mode (journalRecorder.ts) — the native
 * pieces the hook injects through its `makeJournal` seam:
 *
 * - the loop's models: the SAME builders the fast loop uses (Silero VAD,
 *   the ECAPA embedder + the enrolled voiceprints — defaultDeps.ts). No STT,
 *   no LLM: the journal never loads them.
 * - the store: `Paths.cache/journal/` (journalStore.ts).
 * - the uploader: the app's chunked upload JOB (api/client.ts
 *   postAnalyzeUploadChunkedJob) for every file, whatever its size — the
 *   parts are short PUTs and the completion is a 202, which survives the
 *   app being in the background; the direct `/analyze/upload` is one
 *   multi-minute synchronous request that Android routinely kills there.
 *   Title "Journal — <date> <start–end>", `consent: true`, `store: true`,
 *   `source_type: "journal"` (an extra field today's server ignores; the
 *   title and context carry the provenance meanwhile).
 * - the audio session for an all-day mic: `allowsBackgroundRecording` +
 *   `shouldPlayInBackground` on top of the recording mode the live loop
 *   sets, and the Android 13+ notification permission expo-audio's
 *   foreground service needs (expo-notifications is NOT installed; the
 *   permission request is expo-audio's own).
 *
 * Honest limit (read before relying on the background): expo-audio's
 * foreground service ("Recording audio" notification) is attached to its
 * file `AudioRecorder`, not to the PCM `AudioStream` the Live Coach mic
 * uses. `createBackgroundMicHold` therefore starts a throw-away file
 * recorder alongside the stream on Android so that service (and the mic
 * privilege it carries) exists while the journal runs — best effort,
 * unverified on a device as of this writing, and released with its file
 * deleted on stop. If it fails to start, the journal still runs in the
 * foreground.
 */
import { Platform } from "react-native";
import {
  AudioModule,
  requestNotificationPermissionsAsync,
  setAudioModeAsync,
} from "expo-audio";
import { File as FSFile, Paths } from "expo-file-system";
import { postAnalyzeUploadChunkedJob } from "../api/client";
import { buildSpeakerId, buildVad } from "./defaultDeps";
import { journalContext, journalTitle, JournalRecorder, type JournalState } from "./journalRecorder";
import { openDefaultJournalStore, type ClosedJournalFile } from "./journalStore";

export const JOURNAL_UPLOAD_MIME = "audio/wav";

/** Upload one closed journal file as a stored recording. Throws on failure
 *  (the recorder keeps the file and retries at the next boundary). */
export async function uploadJournalFile(file: ClosedJournalFile): Promise<void> {
  const name = file.uri.split("/").pop() || "journal.wav";
  await postAnalyzeUploadChunkedJob(file.uri, name, JOURNAL_UPLOAD_MIME, file.bytes, {
    consent: true,
    store: true,
    title: journalTitle(file),
    context: journalContext(file),
    sourceType: "journal",
  });
}

export interface JournalHandlers {
  onState?: (state: JournalState) => void;
}

/** Build a production JournalRecorder. Throws with the honest reason when
 *  the models or the store can't be had (the hook shows it). */
export async function createDefaultJournalRecorder(handlers: JournalHandlers): Promise<JournalRecorder> {
  const opened = openDefaultJournalStore();
  if (!opened.store) throw new Error(opened.reason);
  const [{ vad }, speaker] = await Promise.all([buildVad(), buildSpeakerId()]);
  if (!speaker.labeler) {
    throw new Error(`speaker-ID unavailable (${speaker.capability.reason})`);
  }
  return new JournalRecorder({
    vad,
    embedder: speaker.embedder,
    labeler: speaker.labeler,
    store: opened.store,
    upload: uploadJournalFile,
    onState: handlers.onState,
  });
}

/** What the audio session was told, for the screen/diagnostics. */
export interface JournalAudioSessionResult {
  /** Android 13+: whether the notification permission expo-audio's
   *  foreground service needs was granted; null where not applicable. */
  notificationsGranted: boolean | null;
}

/**
 * Configure the shared audio session for an all-day mic: the live loop's
 * recording mode plus the background flags. On Android the notification
 * permission is requested first (expo-audio's foreground service posts its
 * "Recording audio" notification; without the permission it refuses to
 * start on Android 13+). Web no-ops.
 */
export async function prepareJournalAudioSession(): Promise<JournalAudioSessionResult> {
  if (Platform.OS === "web") return { notificationsGranted: null };
  let notificationsGranted: boolean | null = null;
  if (Platform.OS === "android") {
    try {
      const res = await requestNotificationPermissionsAsync();
      notificationsGranted = Boolean(res?.granted);
    } catch {
      notificationsGranted = false;
    }
  }
  await setAudioModeAsync({
    allowsRecording: true,
    playsInSilentMode: true,
    shouldPlayInBackground: true,
    allowsBackgroundRecording: true,
  });
  return { notificationsGranted };
}

/** A best-effort handle on the Android foreground recording service. */
export interface BackgroundMicHold {
  /** Stop the throw-away recorder and delete its file. Never throws. */
  release(): Promise<void>;
}

/** Throw-away recorder options: the cheapest thing MediaRecorder writes. */
const HOLD_RECORDING_OPTIONS: Record<string, unknown> = {
  extension: ".aac",
  sampleRate: 8000,
  numberOfChannels: 1,
  bitRate: 16000,
  isMeteringEnabled: false,
  outputFormat: "aac_adts",
  audioEncoder: "aac",
};

/**
 * Android only: start expo-audio's file recorder (which is what its
 * foreground service is attached to) so the app holds a microphone
 * foreground service while the journal's PCM stream runs. Returns null
 * where it does not apply or could not start — the journal runs without
 * it. Unverified on a device; see the module comment.
 */
export async function createBackgroundMicHold(): Promise<BackgroundMicHold | null> {
  if (Platform.OS !== "android") return null;
  let recorder: InstanceType<typeof AudioModule.AudioRecorder> | null = null;
  try {
    recorder = new AudioModule.AudioRecorder(HOLD_RECORDING_OPTIONS);
    await recorder.prepareToRecordAsync(HOLD_RECORDING_OPTIONS);
    recorder.record();
  } catch (err) {
    console.log("[journal] background mic hold unavailable:", err instanceof Error ? err.message : String(err));
    try {
      recorder?.release();
    } catch {
      // Nothing to free.
    }
    return null;
  }
  const held = recorder;
  return {
    async release() {
      let uri: string | null = null;
      try {
        await held.stop();
        uri = held.uri;
      } catch {
        // Already stopped by the OS.
      }
      try {
        held.release();
      } catch {
        // Nothing further to free.
      }
      if (uri) {
        try {
          const f = new FSFile(uri);
          if (f.exists) f.delete();
        } catch {
          // A stray cache file; the cache is reclaimable.
        }
      }
    },
  };
}

/** Where the journal files live (for diagnostics / a future "clear" affordance). */
export function journalDirectoryUri(): string | null {
  if (Platform.OS === "web") return null;
  try {
    const base = Paths.cache.uri;
    return `${base.endsWith("/") ? base.slice(0, -1) : base}/journal`;
  } catch {
    return null;
  }
}
