import React from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { JOURNAL_QUIET_NOTE_SECONDS, type JournalState } from "../live/journalRecorder";
import { useDevModeStore } from "../store/devModeStore";

/** Whether the account has an owner voiceprint the journal can match
 *  against: "ok" | "missing" (say "enroll your voice first") | "checking"
 *  (people list still loading) | "unknown" (the list could not be fetched —
 *  the hook enforces the real gate at Start). */
export type JournalGate = "ok" | "missing" | "checking" | "unknown";

export const JOURNAL_PRIVACY_NOTE =
  "Only stretches the phone believes are your voice are kept; other people's speech is discarded before it is written.";
export const JOURNAL_ENROLL_NOTE =
  "Enroll your voice first — the journal keeps only the stretches that match your voiceprint.";

interface Props {
  state: JournalState;
  sessionActive: boolean;
  gate: JournalGate;
  onRetryUploads?: () => void;
  /** Wall clock, injectable for tests. */
  now?: () => number;
}

/** "1:02:03" / "4:05". */
export function formatClock(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

/** "3.2 min" / "45 s". */
export function formatMinutes(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)} s`;
  return `${(seconds / 60).toFixed(1)} min`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function timeOfDay(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/**
 * The Journal mode's whole on-screen story: the explainer + honest gates
 * while idle, the minimal live counters while listening (elapsed, how
 * often / how long you were heard, when last, the file size, uploads), and
 * the final tally after Stop. No transcript, no suggestions — the journal
 * has none until the server analyzes the uploaded files.
 */
export default function JournalPanel({ state, sessionActive, gate, onRetryUploads, now = Date.now }: Props) {
  // Developer mode off: no file-size or VAD-internals lines — the journal
  // reads as elapsed / times heard / uploads, in plain words.
  const devMode = useDevModeStore((s) => s.devMode);
  const running = sessionActive || state.status === "starting" || state.status === "listening" || state.status === "stopping";
  const quietSeconds =
    state.status === "listening" && state.startedAt !== null
      ? (now() - (state.lastSelfAt ?? state.startedAt)) / 1000
      : 0;
  const uploads = state.uploads;
  const uploadLine =
    uploads.inFlight
      ? `uploading… (${uploads.sent} sent, ${uploads.pending} waiting)`
      : uploads.sent + uploads.pending + uploads.failed === 0
        ? "uploads every 30 min and at Stop"
        : `${uploads.sent} sent · ${uploads.pending} waiting${uploads.failed > 0 ? ` · ${uploads.failed} failed (kept for retry)` : ""}`;

  return (
    <View style={styles.card} testID="journal-panel">
      {state.error ? (
        <View style={styles.errorBanner} testID="journal-error">
          <Text style={styles.errorText}>{state.error}</Text>
        </View>
      ) : null}

      {!running && state.status !== "stopped" ? (
        <>
          <Text style={styles.title}>Journal — listen for my voice</Text>
          <Text style={styles.line}>
            Keeps the mic open all day and listens for your voice. Nothing is transcribed
            or coached while it listens; the stretches where you speak are saved and uploaded
            later as a recording, so the usual analysis and report cards run on them.
          </Text>
          {gate === "missing" ? (
            <Text style={styles.gate} testID="journal-gate">
              {JOURNAL_ENROLL_NOTE}
            </Text>
          ) : gate === "checking" ? (
            <Text style={styles.hint} testID="journal-gate-checking">
              Checking for your voiceprint…
            </Text>
          ) : null}
          <Text style={styles.privacy} testID="journal-privacy">
            {JOURNAL_PRIVACY_NOTE}
          </Text>
          <Text style={styles.hint} testID="journal-background-note">
            Leave this screen open. Locking the screen may pause the microphone on some
            phones — check the listening timer when you come back.
          </Text>
        </>
      ) : null}

      {running || state.status === "stopped" ? (
        <View style={styles.stats}>
          <Text style={styles.statLine} testID="journal-elapsed">
            {state.status === "stopped" ? "Listened" : state.status === "starting" ? "Starting…" : "Listening"}{" "}
            {formatClock(state.listeningSeconds)}
          </Text>
          <Text style={styles.statLine} testID="journal-self">
            {state.selfCount === 0
              ? "You haven't been heard yet"
              : `You spoke ${state.selfCount} time${state.selfCount === 1 ? "" : "s"} · ${formatMinutes(state.selfSeconds)}`}
          </Text>
          <Text style={styles.statLine} testID="journal-last-heard">
            {state.lastSelfAt === null ? "Last heard you: not yet" : `Last heard you: ${timeOfDay(state.lastSelfAt)}`}
          </Text>
          {state.status === "listening" && quietSeconds >= JOURNAL_QUIET_NOTE_SECONDS ? (
            <Text style={styles.hint} testID="journal-quiet-note">
              Haven&apos;t heard you for {Math.floor(quietSeconds / 60)} min — that&apos;s normal, still listening.
            </Text>
          ) : null}
          {devMode ? (
            <Text style={styles.statLine} testID="journal-size">
              {state.status === "stopped"
                ? `${state.filesClosed} journal file${state.filesClosed === 1 ? "" : "s"}`
                : `Journal file ${formatBytes(state.fileBytes)}${state.filesClosed > 0 ? ` · ${state.filesClosed} closed` : ""}`}
            </Text>
          ) : null}
          <Text style={styles.hint} testID="journal-uploads">
            {uploadLine}
          </Text>
          {devMode && state.vadDegraded ? (
            <Text style={styles.hint} testID="journal-vad-degraded">
              Voice detection fell back to the energy rule (Silero failed).
            </Text>
          ) : null}
          {uploads.pending > 0 && !uploads.inFlight && onRetryUploads ? (
            <TouchableOpacity testID="journal-retry-uploads" style={styles.retry} onPress={onRetryUploads}>
              <Text style={styles.retryText}>Upload now</Text>
            </TouchableOpacity>
          ) : null}
          {running ? (
            <Text style={styles.privacy} testID="journal-privacy">
              {JOURNAL_PRIVACY_NOTE}
            </Text>
          ) : null}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    marginHorizontal: 16,
    marginVertical: 6,
    padding: 14,
    gap: 6,
  },
  title: {
    fontSize: 15,
    fontWeight: "700",
    color: "#111827",
  },
  line: {
    fontSize: 13.5,
    lineHeight: 19,
    color: "#374151",
  },
  gate: {
    fontSize: 13.5,
    lineHeight: 19,
    fontWeight: "600",
    color: "#B45309",
  },
  privacy: {
    fontSize: 12.5,
    lineHeight: 17,
    color: "#4B5563",
    fontStyle: "italic",
  },
  hint: {
    fontSize: 12.5,
    lineHeight: 17,
    color: "#6B7280",
  },
  stats: {
    gap: 4,
  },
  statLine: {
    fontSize: 15,
    fontWeight: "600",
    color: "#1F2937",
  },
  errorBanner: {
    backgroundColor: "#FEE2E2",
    borderLeftWidth: 4,
    borderLeftColor: "#EF4444",
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
  },
  errorText: {
    fontSize: 13,
    color: "#991B1B",
  },
  retry: {
    alignSelf: "flex-start",
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#4A90D9",
    backgroundColor: "#EFF6FF",
  },
  retryText: {
    color: "#4A90D9",
    fontSize: 13,
    fontWeight: "700",
  },
});
