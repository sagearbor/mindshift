import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Platform,
} from "react-native";
import {
  CameraView,
  useCameraPermissions,
  useMicrophonePermissions,
} from "expo-camera";
import * as MediaLibrary from "expo-media-library";
import { File as FSFile } from "expo-file-system";
import type { RecordedFile } from "../store/recorderStore";

// House colors.
const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";
const DANGER = "#DC2626";

/** Hard cap on an in-app recording. Owner decision: 10 minutes, auto-stopped, so
 *  files stay small and predictable (paired with the 480p quality preset). */
export const MAX_RECORDING_SECONDS = 600;

/**
 * Seconds remaining before the hard cap, floored and never negative. Pure and
 * exported so the cap logic is unit-testable without the camera. `elapsed` is a
 * float (the ticking clock); the cap defaults to MAX_RECORDING_SECONDS.
 */
export function remainingSeconds(
  elapsed: number,
  cap: number = MAX_RECORDING_SECONDS,
): number {
  return Math.max(0, cap - Math.floor(elapsed));
}

/**
 * Whether the elapsed time has reached the cap (so recording must auto-stop).
 * Pure and exported for unit testing.
 */
export function isAtCap(
  elapsed: number,
  cap: number = MAX_RECORDING_SECONDS,
): boolean {
  return Math.floor(elapsed) >= cap;
}

/** Format a whole-second count as m:ss (e.g. 65 → "1:05"). Pure/exported. */
export function formatClock(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${rem.toString().padStart(2, "0")}`;
}

interface RecordScreenProps {
  /** Return to the Session screen without a recording (cancel / back). */
  onBack: () => void;
  /** Hand the finished, camera-roll-saved recording to the Session upload flow. */
  onComplete: (file: RecordedFile) => void;
}

/**
 * In-app video recording. Records at 480p with a hard 10-minute cap, SAVES the
 * clip to the camera roll (the linchpin: the user's cloud backup — e.g. Google
 * Photos — only backs up camera-roll items, which is what later enables
 * attach-HD replay), then hands the file to the Session screen's existing
 * upload/analyze flow.
 *
 * Honest by construction: every permission denial shows an inline message with a
 * grant retry (never a silent black screen), the timer counts up with the
 * remaining time visible, and on web — where expo-camera video recording is
 * unreliable — we show a plain note instead of a broken camera.
 */
export default function RecordScreen({ onBack, onComplete }: RecordScreenProps) {
  const [camPerm, requestCamPerm] = useCameraPermissions();
  const [micPerm, requestMicPerm] = useMicrophonePermissions();
  const [mediaPerm, requestMediaPerm] = MediaLibrary.usePermissions({
    writeOnly: true,
  });

  const cameraRef = useRef<CameraView>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);
  // Synchronous re-entry latch: `recording` state lags a render, so a
  // same-frame double tap could start two recordAsync loops and leak the
  // first timer interval (review MINOR-2).
  const recordLatchRef = useRef(false);

  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [cappedNote, setCappedNote] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A finished clip we couldn't save to the camera roll. We refuse to silently
  // drop it: the user can retry the save or analyze it anyway (HD-later just
  // won't be available). Holds the local URI until resolved.
  const [saveFailedUri, setSaveFailedUri] = useState<string | null>(null);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Best-effort local-file size so the Session screen can route large clips down
  // the chunked-upload path (a 10-min 480p video can exceed the direct ceiling).
  const readSize = (uri: string): number | undefined => {
    try {
      const size = new FSFile(uri).size;
      return typeof size === "number" && size > 0 ? size : undefined;
    } catch {
      return undefined;
    }
  };

  const buildFile = (uri: string): RecordedFile => ({
    uri,
    name: `mindshift-${Date.now()}.mp4`,
    mimeType: "video/mp4",
    size: readSize(uri),
  });

  // Save the finished clip to the camera roll, then hand it off. Saving is the
  // whole point (cloud backup → attach-HD later), so a save failure is surfaced
  // honestly rather than swallowed.
  const finishRecording = useCallback(
    async (uri: string) => {
      try {
        await MediaLibrary.Asset.create(uri);
      } catch {
        if (mountedRef.current) setSaveFailedUri(uri);
        return;
      }
      // If the user backed out mid-recording and recordAsync resolved later,
      // the clip is safely in the camera roll but the handoff must NOT fire —
      // an unconsumed preselect would ambush the next Session visit with a
      // stale file (review MINOR-1).
      if (mountedRef.current) onComplete(buildFile(uri));
    },
    [onComplete],
  );

  const handleRecord = useCallback(async () => {
    if (recording || recordLatchRef.current) return;
    recordLatchRef.current = true;
    setError(null);
    setCappedNote(false);
    setRecording(true);
    setElapsed(0);
    // Count up once per second; auto-stop the instant we hit the cap.
    timerRef.current = setInterval(() => {
      setElapsed((prev) => {
        const next = prev + 1;
        if (isAtCap(next)) {
          setCappedNote(true);
          cameraRef.current?.stopRecording();
          stopTimer();
        }
        return next;
      });
    }, 1000);
    try {
      // Resolves when stopRecording is called, the cap (maxDuration) is hit, or
      // the preview stops. The 480p quality is set on the CameraView below.
      const result = await cameraRef.current?.recordAsync({
        maxDuration: MAX_RECORDING_SECONDS,
      });
      stopTimer();
      if (result?.uri) {
        await finishRecording(result.uri);
      } else if (mountedRef.current) {
        setError("Recording didn’t produce a file. Please try again.");
      }
    } catch {
      if (mountedRef.current) {
        setError("Recording failed. Please try again.");
      }
    } finally {
      recordLatchRef.current = false;
      if (mountedRef.current) setRecording(false);
      stopTimer();
    }
  }, [recording, finishRecording, stopTimer]);

  const handleStop = useCallback(() => {
    cameraRef.current?.stopRecording();
    stopTimer();
  }, [stopTimer]);

  // --- Web: in-app recording isn't reliable; show an honest note, not a camera.
  if (Platform.OS === "web") {
    return (
      <View style={styles.flex}>
        <Header onBack={onBack} />
        <View style={styles.centered}>
          <Text style={styles.noteTitle}>Recording is mobile-only</Text>
          <Text style={styles.noteText} testID="record-web-note">
            In-app recording is mobile-only for now — upload a file or paste a
            link instead.
          </Text>
        </View>
      </View>
    );
  }

  // --- Save-failed: a real clip we couldn't back up. Honest recovery, no silent
  // drop.
  if (saveFailedUri) {
    return (
      <View style={styles.flex}>
        <Header onBack={onBack} />
        <View style={styles.centered} testID="save-error">
          <Text style={styles.noteTitle}>Couldn’t save to your camera roll</Text>
          <Text style={styles.noteText}>
            The recording is fine, but we couldn’t add it to your camera roll —
            so HD replay later (via your cloud backup) may not be available. You
            can try saving again, or analyze it now anyway.
          </Text>
          <TouchableOpacity
            testID="save-retry-button"
            style={styles.primaryButton}
            onPress={() => {
              const uri = saveFailedUri;
              setSaveFailedUri(null);
              void finishRecording(uri);
            }}
          >
            <Text style={styles.primaryButtonText}>Try saving again</Text>
          </TouchableOpacity>
          <TouchableOpacity
            testID="analyze-anyway-button"
            style={styles.secondaryButton}
            onPress={() => {
              const uri = saveFailedUri;
              setSaveFailedUri(null);
              onComplete(buildFile(uri));
            }}
          >
            <Text style={styles.secondaryButtonText}>Analyze it now anyway</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // --- Permission gate: honest, per-permission, with grant retries. Never a
  // black screen.
  const missing: { key: string; label: string; grant: () => void }[] = [];
  if (!camPerm?.granted) {
    missing.push({
      key: "camera",
      label: "Camera access is needed to record video.",
      grant: () => void requestCamPerm(),
    });
  }
  if (!micPerm?.granted) {
    missing.push({
      key: "mic",
      label: "Microphone access is needed to capture the conversation.",
      grant: () => void requestMicPerm(),
    });
  }
  if (!mediaPerm?.granted) {
    missing.push({
      key: "media",
      label:
        "Media-library access is needed to save the recording to your camera roll (so it can back up to your cloud for HD replay later).",
      grant: () => void requestMediaPerm(),
    });
  }

  if (missing.length > 0) {
    return (
      <View style={styles.flex}>
        <Header onBack={onBack} />
        <View style={styles.centered} testID="permission-gate">
          <Text style={styles.noteTitle}>A little access first</Text>
          {missing.map((m) => (
            <View key={m.key} style={styles.permRow} testID={`perm-${m.key}`}>
              <Text style={styles.noteText}>{m.label}</Text>
              <TouchableOpacity
                testID={`grant-${m.key}`}
                style={styles.primaryButton}
                onPress={m.grant}
              >
                <Text style={styles.primaryButtonText}>Grant access</Text>
              </TouchableOpacity>
            </View>
          ))}
        </View>
      </View>
    );
  }

  const remaining = remainingSeconds(elapsed);

  return (
    <View style={styles.flex}>
      <Header onBack={onBack} />
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        mode="video"
        videoQuality="480p"
        facing="front"
        testID="camera-view"
      >
        <View style={styles.overlay} pointerEvents="box-none">
          {/* Live elapsed timer counting up, with remaining time visible. */}
          <View style={styles.timerPill} testID="record-timer">
            <Text style={styles.timerText}>
              {formatClock(elapsed)} / {formatClock(MAX_RECORDING_SECONDS)}
            </Text>
            <Text style={styles.timerRemaining}>
              {formatClock(remaining)} left
            </Text>
          </View>

          {cappedNote && (
            <Text style={styles.capNote} testID="record-cap-note">
              Reached the 10-minute limit — recording stopped automatically.
            </Text>
          )}

          {error && (
            <Text style={styles.recordError} testID="record-error">
              {error}
            </Text>
          )}

          <View style={styles.controls} pointerEvents="box-none">
            {recording ? (
              <TouchableOpacity
                testID="stop-button"
                style={styles.stopButton}
                onPress={handleStop}
              >
                <View style={styles.stopIcon} />
                <Text style={styles.controlLabel}>Stop</Text>
              </TouchableOpacity>
            ) : (
              <TouchableOpacity
                testID="record-button"
                style={styles.recordButton}
                onPress={() => void handleRecord()}
              >
                <View style={styles.recordIcon} />
                <Text style={styles.controlLabel}>Record</Text>
              </TouchableOpacity>
            )}
          </View>
        </View>
      </CameraView>
    </View>
  );
}

function Header({ onBack }: { onBack: () => void }) {
  return (
    <View style={styles.header}>
      <TouchableOpacity testID="record-back" onPress={onBack}>
        <Text style={styles.backText}>‹ Back</Text>
      </TouchableOpacity>
      <Text style={styles.headerTitle}>Record video</Text>
      <View style={styles.headerSpacer} />
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: "#000000" },
  header: {
    flexDirection: "row",
    alignItems: "center",
    paddingTop: 56,
    paddingBottom: 12,
    paddingHorizontal: 16,
    backgroundColor: "#FFFFFF",
    borderBottomWidth: 1,
    borderBottomColor: "#E5E7EB",
  },
  backText: { fontSize: 16, color: PRIMARY, fontWeight: "600", width: 64 },
  headerTitle: {
    flex: 1,
    textAlign: "center",
    fontSize: 17,
    fontWeight: "700",
    color: INK,
  },
  headerSpacer: { width: 64 },
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
    backgroundColor: "#F9FAFB",
  },
  noteTitle: { fontSize: 18, fontWeight: "700", color: INK, marginBottom: 10 },
  noteText: {
    fontSize: 14,
    lineHeight: 20,
    color: MUTED,
    textAlign: "center",
    marginBottom: 14,
  },
  permRow: { alignItems: "center", marginBottom: 18, maxWidth: 360 },
  primaryButton: {
    backgroundColor: PRIMARY,
    paddingVertical: 12,
    paddingHorizontal: 28,
    borderRadius: 10,
    marginBottom: 10,
  },
  primaryButtonText: { color: "#FFFFFF", fontSize: 15, fontWeight: "600" },
  secondaryButton: {
    paddingVertical: 12,
    paddingHorizontal: 28,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#D1D5DB",
  },
  secondaryButtonText: { color: MUTED, fontSize: 15, fontWeight: "600" },
  camera: { flex: 1 },
  overlay: {
    flex: 1,
    justifyContent: "space-between",
    paddingVertical: 24,
    paddingHorizontal: 16,
  },
  timerPill: {
    alignSelf: "center",
    alignItems: "center",
    backgroundColor: "rgba(0,0,0,0.55)",
    borderRadius: 16,
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  timerText: { color: "#FFFFFF", fontSize: 20, fontWeight: "700" },
  timerRemaining: { color: "#E5E7EB", fontSize: 12, marginTop: 2 },
  capNote: {
    alignSelf: "center",
    color: "#FDE68A",
    fontSize: 13,
    fontWeight: "600",
    textAlign: "center",
    marginTop: 12,
  },
  recordError: {
    alignSelf: "center",
    color: "#FCA5A5",
    fontSize: 13,
    fontWeight: "600",
    textAlign: "center",
    marginTop: 12,
  },
  controls: { alignItems: "center" },
  recordButton: { alignItems: "center" },
  recordIcon: {
    width: 68,
    height: 68,
    borderRadius: 34,
    backgroundColor: DANGER,
    borderWidth: 4,
    borderColor: "#FFFFFF",
  },
  stopButton: { alignItems: "center" },
  stopIcon: {
    width: 68,
    height: 68,
    borderRadius: 12,
    backgroundColor: DANGER,
    borderWidth: 4,
    borderColor: "#FFFFFF",
  },
  controlLabel: {
    color: "#FFFFFF",
    fontSize: 13,
    fontWeight: "600",
    marginTop: 8,
  },
});
