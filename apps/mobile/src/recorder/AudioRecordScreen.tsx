import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
  Platform,
} from "react-native";
import {
  getRecordingPermissionsAsync,
  requestRecordingPermissionsAsync,
} from "expo-audio";
import { formatClock } from "../screens/recordTiming";
import { RECORDER_ENGINE } from "./engine";
import type { RecorderEngineKind } from "./engine";
import { runPreflight } from "./preflight";
import type { PreflightReport } from "./preflight";
import type { PcmSourceFactory } from "./pcmSource";
import {
  SegmentedAudioSession,
  DEFAULT_SEGMENT_MS,
  DEFAULT_RESUME_RETRY_MS,
} from "./segmentedSession";
import type { SessionPhase } from "./segmentedSession";
import type { RecorderSessionStore } from "./sessionStore";
import { StreamAudioSession } from "./streamSession";
import type {
  RecordedAudioFile,
  RecorderFactory,
  SegmentFormat,
} from "./types";

// House colors (match RecordScreenNative).
const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";
const DANGER = "#DC2626";
const AMBER = "#B45309";

/** Hard cap on one audio session. Chosen so the stitched file stays under the
 *  200 MB upload ceiling even on the byte-heavy iOS WAV path (~32 KB/s →
 *  ~187 MB at 100 min). */
export const MAX_AUDIO_SESSION_MS = 100 * 60 * 1000;

/** Everything platform-specific the screen needs, injectable for tests. When
 *  the prop is omitted the production defaults (expo-audio recorder,
 *  expo-file-system storage, expo-battery) are loaded lazily at mount. */
export interface AudioRecorderDeps {
  store: RecorderSessionStore;
  makeRecorder: RecorderFactory;
  format: SegmentFormat;
  getBatteryLevel: () => Promise<number | null>;
  /** Configure the OS audio session for recording (true) / back to playback
   *  (false). Activation may report `backgroundCapable: false` (Android with
   *  the notification permission denied) — recording still works, but only
   *  with the screen on, and the UI must say so. A void resolve (older fakes)
   *  means background-capable. */
  configureAudioSession: (
    active: boolean,
  ) => Promise<{ backgroundCapable: boolean } | void>;
  segmentMs?: number;
  resumeRetryMs?: number;
  maxSessionMs?: number;
  /** Which engine records. Omitted → "stream" (v2 gapless) when a PcmSource
   *  factory is available, else the v1 segmented recorder — so older test
   *  wirings (and any caller without stream capture) keep v1 semantics
   *  without touching this field. */
  engine?: RecorderEngineKind;
  /** Continuous-PCM capture for the stream engine. Production wires the
   *  expo-audio realtime stream; tests inject fakes. */
  makePcmSource?: PcmSourceFactory;
  /** Stream engine only: flush-to-disk cadence (the crash-loss bound). */
  flushMs?: number;
  /** Stream engine only: silent-stall watchdog (see streamSession). */
  stallMs?: number;
}

/** Engine choice for a deps bundle (see AudioRecorderDeps.engine). */
function resolveEngineKind(deps: AudioRecorderDeps | null): RecorderEngineKind {
  if (!deps) return RECORDER_ENGINE;
  if (deps.engine) return deps.engine;
  return deps.makePcmSource ? RECORDER_ENGINE : "recorder";
}

interface AudioRecordScreenProps {
  /** Return to the Analyze screen without a recording. */
  onBack: () => void;
  /** Hand the finished, stitched audio file to the upload flow. */
  onComplete: (file: RecordedAudioFile) => void;
  /** Test seam; production uses the lazy defaults. */
  deps?: AudioRecorderDeps;
}

type UiState = "preflight" | "recording" | "finishing";

/**
 * Crash-hardened AUDIO-ONLY recording. The therapist use case: a ~1-hour
 * session recorded on the phone that must never be lost wholesale — audio is
 * finalized to durable storage every ~5 minutes, so a crash costs at most the
 * current segment (and the recovery prompt offers the rest back on next
 * launch).
 */
export default function AudioRecordScreen({
  onBack,
  onComplete,
  deps,
}: AudioRecordScreenProps) {
  // Resolve platform defaults lazily so tests (and web) never touch native
  // modules. The ref pins one instance for the component's lifetime.
  const depsRef = useRef<AudioRecorderDeps | null>(deps ?? null);
  if (!depsRef.current && Platform.OS !== "web") {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    depsRef.current = require("./defaultDeps").defaultAudioRecorderDeps();
  }

  const [micGranted, setMicGranted] = useState<boolean | null>(null);
  const [report, setReport] = useState<PreflightReport | null>(null);
  const [ui, setUi] = useState<UiState>("preflight");
  const [phase, setPhase] = useState<SessionPhase>("idle");
  const [elapsedMs, setElapsedMs] = useState(0);
  const [segmentsSaved, setSegmentsSaved] = useState(0);
  const [savedMs, setSavedMs] = useState(0);
  const [backgroundCapable, setBackgroundCapable] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const sessionRef = useRef<SegmentedAudioSession | StreamAudioSession | null>(
    null,
  );
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);
  const stoppingRef = useRef(false);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Mic permission check on mount.
  useEffect(() => {
    mountedRef.current = true;
    if (Platform.OS === "web") return;
    void (async () => {
      try {
        const p = await getRecordingPermissionsAsync();
        if (mountedRef.current) setMicGranted(p.granted);
      } catch {
        if (mountedRef.current) setMicGranted(false);
      }
    })();
    return () => {
      mountedRef.current = false;
      stopTimer();
      // Leaving mid-recording: salvage the live segment and leave the session
      // recoverable — never a silent loss.
      const session = sessionRef.current;
      if (session && session.phase !== "done") {
        void session.abandon().catch(() => {});
        void depsRef.current?.configureAudioSession(false).catch(() => {});
      }
    };
  }, [stopTimer]);

  // Preflight once the mic is granted: storage + battery facts.
  useEffect(() => {
    const d = depsRef.current;
    if (!d || micGranted !== true) return;
    void (async () => {
      let battery: number | null = null;
      try {
        battery = await d.getBatteryLevel();
      } catch {
        battery = null;
      }
      if (!mountedRef.current) return;
      setReport(
        runPreflight({
          freeBytes: d.store.freeBytes(),
          batteryLevel: battery,
          bytesPerSecond: d.format.bytesPerSecond,
          plannedSeconds: (d.maxSessionMs ?? MAX_AUDIO_SESSION_MS) / 1000,
        }),
      );
    })();
  }, [micGranted]);

  const syncFromSession = useCallback(() => {
    const session = sessionRef.current;
    if (!session || !mountedRef.current) return;
    setPhase(session.phase);
    setElapsedMs(session.elapsedMs());
    setSegmentsSaved(session.segmentsSaved);
    setSavedMs(session.savedDurationMs);
  }, []);

  const handleStop = useCallback(async () => {
    const d = depsRef.current;
    const session = sessionRef.current;
    if (!d || !session || stoppingRef.current) return;
    stoppingRef.current = true;
    stopTimer();
    if (mountedRef.current) setUi("finishing");
    try {
      const file = await session.stopAndFinish();
      sessionRef.current = null;
      void d.configureAudioSession(false).catch(() => {});
      onComplete(file);
    } catch {
      // Stitching/finishing failed: the segments are still on disk as a
      // recoverable session — say so instead of pretending nothing happened.
      stoppingRef.current = false;
      if (mountedRef.current) {
        setUi("recording");
        setError(
          "Couldn’t assemble the recording — your audio is saved and will be offered for recovery on the Analyze screen.",
        );
      }
    }
  }, [onComplete, stopTimer]);

  const handleStart = useCallback(async () => {
    const d = depsRef.current;
    if (!d || sessionRef.current) return;
    setError(null);
    const session =
      resolveEngineKind(d) === "stream"
        ? new StreamAudioSession({
            // A missing factory here means engine was FORCED to "stream"
            // without capture deps — production wiring for that case is the
            // lazily-loaded expo-audio realtime stream.
            makeSource:
              d.makePcmSource ??
              (() => {
                // eslint-disable-next-line @typescript-eslint/no-var-requires
                const { ExpoPcmSource } = require("./expoPcmSource");
                return new ExpoPcmSource();
              }),
            store: d.store,
            segmentMs: d.segmentMs,
            flushMs: d.flushMs,
            resumeRetryMs: d.resumeRetryMs,
            stallMs: d.stallMs,
          })
        : new SegmentedAudioSession({
            makeRecorder: d.makeRecorder,
            store: d.store,
            format: d.format,
            segmentMs: d.segmentMs ?? DEFAULT_SEGMENT_MS,
            resumeRetryMs: d.resumeRetryMs ?? DEFAULT_RESUME_RETRY_MS,
          });
    try {
      const mode = await d.configureAudioSession(true);
      if (mountedRef.current) {
        setBackgroundCapable(mode?.backgroundCapable !== false);
      }
      await session.start();
    } catch (e) {
      if (mountedRef.current) {
        // Keep the honest detail visible — a swallowed cause is how the vc29
        // upload bug stayed undiagnosable for a month.
        const detail =
          e instanceof Error && e.message ? ` (${e.message})` : "";
        setError(`Couldn’t start the microphone. Please try again.${detail}`);
      }
      return;
    }
    sessionRef.current = session;
    setUi("recording");
    syncFromSession();
    const maxMs = d.maxSessionMs ?? MAX_AUDIO_SESSION_MS;
    timerRef.current = setInterval(() => {
      void (async () => {
        const live = sessionRef.current;
        if (!live) return;
        await live.tick();
        syncFromSession();
        if (live.elapsedMs() >= maxMs && !stoppingRef.current) {
          await handleStop();
        }
      })();
    }, 1000);
  }, [handleStop, syncFromSession]);

  // --- Web: no expo-audio recorder implementation; honest note, no fake UI.
  if (Platform.OS === "web") {
    return (
      <View style={styles.flex}>
        <Header onBack={onBack} />
        <View style={styles.centered} testID="audio-web-note">
          <Text style={styles.noteTitle}>Audio recording is mobile-only</Text>
          <Text style={styles.noteText}>
            In-app audio recording isn’t available in the browser — use the
            mobile app, or upload a file instead.
          </Text>
        </View>
      </View>
    );
  }

  // --- Permission gate: honest, with a grant retry. Never a dead screen.
  if (micGranted !== true) {
    return (
      <View style={styles.flex}>
        <Header onBack={onBack} />
        <View style={styles.centered} testID="audio-permission-gate">
          <Text style={styles.noteTitle}>A little access first</Text>
          <Text style={styles.noteText}>
            Microphone access is needed to record the conversation.
          </Text>
          <TouchableOpacity
            testID="grant-audio-mic"
            style={styles.primaryButton}
            onPress={() =>
              void (async () => {
                try {
                  const p = await requestRecordingPermissionsAsync();
                  if (mountedRef.current) setMicGranted(p.granted);
                } catch {
                  if (mountedRef.current) setMicGranted(false);
                }
              })()
            }
          >
            <Text style={styles.primaryButtonText}>Grant access</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // Honest per-engine disclosure. The stream engine records gaplessly and
  // flushes every few seconds; the v1 engine restarts the recorder every
  // ~5 min with a small audible gap — each gets its own truthful line.
  const engineKind = resolveEngineKind(depsRef.current);
  const disclosure = (
    <Text style={styles.disclosure} testID="gap-disclosure">
      {engineKind === "stream"
        ? "Gapless recording: audio is saved every few seconds — a crash can lose at most a moment."
        : "Crash-proof recording: audio is saved every 5 min — a brief ~0.2s gap at each save point."}
    </Text>
  );

  // --- Preflight card: storage + battery facts before a long session.
  if (ui === "preflight") {
    return (
      <View style={styles.flex}>
        <Header onBack={onBack} />
        <View style={styles.centered}>
          <Text style={styles.noteTitle}>Record audio</Text>
          <Text style={styles.noteText}>
            Audio-only recording for long sessions — smaller files, works with
            the screen off, easier on the battery.
          </Text>
          {disclosure}
          {report?.storage.message && (
            <Text
              style={[
                styles.warning,
                report.storage.blocking && styles.warningBlocking,
              ]}
              testID="storage-warning"
            >
              {report.storage.message}
            </Text>
          )}
          {report?.battery.message && (
            <Text style={styles.warning} testID="battery-warning">
              {report.battery.message}
            </Text>
          )}
          {error && (
            <Text style={styles.error} testID="audio-start-error">
              {error}
            </Text>
          )}
          <TouchableOpacity
            testID="start-audio-recording"
            style={[
              styles.recordButtonWrap,
              report?.storage.blocking === true && styles.disabledButton,
            ]}
            disabled={report?.storage.blocking === true}
            onPress={() => void handleStart()}
          >
            <View style={styles.recordIcon} />
            <Text style={styles.controlLabel}>Record</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  if (ui === "finishing") {
    return (
      <View style={styles.flex}>
        <Header onBack={onBack} />
        <View style={styles.centered} testID="audio-finishing">
          <ActivityIndicator color={PRIMARY} />
          <Text style={styles.noteText}>Assembling your recording…</Text>
        </View>
      </View>
    );
  }

  // --- Live recording UI.
  const savedMin = Math.floor(savedMs / 60000);
  return (
    <View style={styles.flex}>
      <Header onBack={onBack} />
      <View style={styles.centered}>
        <Text style={styles.timer} testID="audio-timer">
          {formatClock(Math.floor(elapsedMs / 1000))}
        </Text>
        <Text style={styles.segmentStatus} testID="segment-status">
          {segmentsSaved === 0
            ? engineKind === "stream"
              ? "First save in a few seconds…"
              : "First save in a few minutes…"
            : `${segmentsSaved} segment${segmentsSaved === 1 ? "" : "s"} safe on this phone (${savedMin} min)`}
        </Text>
        {!backgroundCapable && (
          <Text style={styles.interruption} testID="audio-foreground-note">
            Notifications are off, so recording only works with the screen on —
            keep the app open. (Allow notifications to record screen-off.)
          </Text>
        )}
        {phase === "interrupted" && (
          <Text style={styles.interruption} testID="interruption-banner">
            Recording interrupted — something took the microphone (a call?).
            Everything so far is saved; trying to resume automatically…
          </Text>
        )}
        {error && (
          <Text style={styles.error} testID="audio-record-error">
            {error}
          </Text>
        )}
        {disclosure}
        <Text style={styles.screenOffNote}>
          You can turn the screen off — recording continues.
        </Text>
        <TouchableOpacity
          testID="stop-audio-recording"
          style={styles.recordButtonWrap}
          onPress={() => void handleStop()}
        >
          <View style={styles.stopIcon} />
          <Text style={styles.controlLabel}>Stop &amp; use recording</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

function Header({ onBack }: { onBack: () => void }) {
  return (
    <View style={styles.header}>
      <TouchableOpacity testID="audio-record-back" onPress={onBack}>
        <Text style={styles.backText}>‹ Back</Text>
      </TouchableOpacity>
      <Text style={styles.headerTitle}>Record audio</Text>
      <View style={styles.headerSpacer} />
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: "#F9FAFB" },
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
  },
  noteTitle: { fontSize: 18, fontWeight: "700", color: INK, marginBottom: 10 },
  noteText: {
    fontSize: 14,
    lineHeight: 20,
    color: MUTED,
    textAlign: "center",
    marginBottom: 14,
    maxWidth: 340,
  },
  disclosure: {
    fontSize: 12.5,
    lineHeight: 18,
    color: MUTED,
    textAlign: "center",
    marginBottom: 12,
    maxWidth: 340,
  },
  warning: {
    fontSize: 13,
    lineHeight: 19,
    color: AMBER,
    textAlign: "center",
    marginBottom: 10,
    maxWidth: 340,
    fontWeight: "600",
  },
  warningBlocking: { color: DANGER },
  error: {
    fontSize: 13,
    lineHeight: 19,
    color: DANGER,
    textAlign: "center",
    marginBottom: 10,
    maxWidth: 340,
    fontWeight: "600",
  },
  interruption: {
    fontSize: 13,
    lineHeight: 19,
    color: AMBER,
    textAlign: "center",
    marginBottom: 12,
    maxWidth: 340,
    fontWeight: "600",
  },
  screenOffNote: {
    fontSize: 12.5,
    color: MUTED,
    textAlign: "center",
    marginBottom: 18,
  },
  timer: { fontSize: 44, fontWeight: "700", color: INK, marginBottom: 6 },
  segmentStatus: {
    fontSize: 13,
    color: "#059669",
    fontWeight: "600",
    marginBottom: 14,
    textAlign: "center",
  },
  primaryButton: {
    backgroundColor: PRIMARY,
    paddingVertical: 12,
    paddingHorizontal: 28,
    borderRadius: 10,
    marginBottom: 10,
  },
  primaryButtonText: { color: "#FFFFFF", fontSize: 15, fontWeight: "600" },
  recordButtonWrap: { alignItems: "center", marginTop: 8 },
  disabledButton: { opacity: 0.4 },
  recordIcon: {
    width: 68,
    height: 68,
    borderRadius: 34,
    backgroundColor: DANGER,
    borderWidth: 4,
    borderColor: "#FFFFFF",
  },
  stopIcon: {
    width: 68,
    height: 68,
    borderRadius: 12,
    backgroundColor: DANGER,
    borderWidth: 4,
    borderColor: "#FFFFFF",
  },
  controlLabel: {
    color: INK,
    fontSize: 13,
    fontWeight: "600",
    marginTop: 8,
  },
});
