import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Platform,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

import { enrollVoiceDirect, type DirectEnrollResult } from "../api/client";
import {
  concatTakesToWav,
  takeDurationMs,
  type PhraseTake,
} from "../recorder/guidedCapture";
import type {
  PcmFrame,
  PcmSource,
  PcmSourceFactory,
} from "../recorder/pcmSource";

/**
 * The four prompted phrases. Neutral, conversational, and together phonetically
 * varied (they cover the vowel space plus fricatives, plosives, nasals and
 * glides) so ~20 seconds of reading gives the embedder a rounded sample of the
 * voice. Each reads aloud in roughly five seconds.
 */
export const PHRASES: string[] = [
  "Hi, it's me — I'm teaching this app what my voice sounds like.",
  "Yesterday evening we cooked dinner together and talked about the weekend.",
  "Please pass the water jug before the soup gets cold, would you?",
  "When the weather turns bright and clear, we like to walk down by the river.",
];

/** A take shorter than this holds too little audio to be worth uploading —
 *  the user is asked to read the phrase again rather than silently keeping a
 *  useless clip (the server would 422 the total anyway; failing early is
 *  kinder). */
export const MIN_TAKE_MS = 1000;

/** Everything platform-specific, injectable for tests. Production defaults are
 *  loaded lazily so this module never touches native code at import time. */
export interface VoiceTrainingDeps {
  /** Continuous mic capture — the same seam the v2 recorder engine uses. */
  makeSource: PcmSourceFactory;
  /** Persist the finished wav; returns the handle enrollVoiceDirect accepts
   *  (a file URI on native, a File on web). */
  saveWav: (bytes: Uint8Array) => Promise<string | File>;
  enroll: (
    file: string | File,
    name: string,
    person?: { personId: string; displayName?: string | null },
  ) => Promise<DirectEnrollResult>;
  getPermission: () => Promise<boolean>;
  requestPermission: () => Promise<boolean>;
}

function defaultDeps(): VoiceTrainingDeps {
  return {
    makeSource: () => {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const { ExpoPcmSource } = require("../recorder/expoPcmSource");
      return new ExpoPcmSource();
    },
    saveWav: async (bytes: Uint8Array) => {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const { File, Paths } = require("expo-file-system");
      const base: string = Paths.cache.uri;
      const uri = `${base.endsWith("/") ? base : `${base}/`}guided-enrollment-${Date.now()}.wav`;
      new File(uri).write(bytes);
      return uri;
    },
    enroll: (file, name, person) => enrollVoiceDirect(file, name, person),
    getPermission: async () => {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const { getRecordingPermissionsAsync } = require("expo-audio");
      try {
        return (await getRecordingPermissionsAsync()).granted === true;
      } catch {
        return false;
      }
    },
    requestPermission: async () => {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const { requestRecordingPermissionsAsync } = require("expo-audio");
      try {
        return (await requestRecordingPermissionsAsync()).granted === true;
      } catch {
        return false;
      }
    },
  };
}

interface VoiceTrainingFlowProps {
  /** Enrollment succeeded and the user tapped Done — `count` is the server's
   *  real sample count (the parent refetches the profile for detail). */
  onDone: (count: number) => void;
  onCancel: () => void;
  /** People labeling: train ANOTHER person's voice ("Mom") instead of the
   *  owner's. Omitted → the owner ("This is me"), exactly as before. The
   *  phrases are read by that person; the copy names them. */
  person?: { personId: string; displayName: string };
  /** Test seam; production uses the lazy defaults. */
  deps?: VoiceTrainingDeps;
}

type Stage = "phrase" | "uploading" | "success" | "error";

/**
 * Guided voice enrollment ("Train my voice"): read four short phrases, each
 * recorded through the v2 PcmSource seam, concatenated into one wav and
 * uploaded to POST /voice/enroll-direct. No dependency on having any analyzed
 * recording — this is the from-scratch path that sits beside "This is me".
 *
 * Honesty: mic permission is a real gate with a retry (never a dead screen);
 * a too-quiet/too-short take is called out and re-recorded; upload failures
 * show the server's actual reason (422 "not enough speech", 503 unavailable)
 * or an honest network message with a retry. Nothing is faked, ever.
 */
export default function VoiceTrainingFlow({
  onDone,
  onCancel,
  person,
  deps,
}: VoiceTrainingFlowProps) {
  // "your voice" for the owner; "Mom's voice" for a partner.
  const whose = person ? `${person.displayName}’s` : "your";
  const personRef = useRef(person);
  personRef.current = person;
  const depsRef = useRef<VoiceTrainingDeps | null>(deps ?? null);
  if (!depsRef.current) depsRef.current = defaultDeps();

  const [permGranted, setPermGranted] = useState<boolean | null>(null);
  const [phraseIndex, setPhraseIndex] = useState(0);
  const [stage, setStage] = useState<Stage>("phrase");
  const [recording, setRecording] = useState(false);
  const [recordedMs, setRecordedMs] = useState(0);
  const [takeNote, setTakeNote] = useState<string | null>(null);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [enrollCount, setEnrollCount] = useState<number | null>(null);

  const mountedRef = useRef(true);
  const sourceRef = useRef<PcmSource | null>(null);
  const chunksRef = useRef<Int16Array[]>([]);
  const rateRef = useRef<number | null>(null);
  const mixedRateRef = useRef(false);
  const takesRef = useRef<PhraseTake[]>([]);

  /** Never throws — releasing a dead native stream must not take the UI down. */
  const stopSource = useCallback(() => {
    const source = sourceRef.current;
    sourceRef.current = null;
    if (!source) return;
    try {
      source.stop();
    } catch {
      // Already dead; nothing further to release.
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    if (Platform.OS !== "web") {
      void depsRef.current!.getPermission().then((granted) => {
        if (mountedRef.current) setPermGranted(granted);
      });
    }
    return () => {
      mountedRef.current = false;
      stopSource();
    };
  }, [stopSource]);

  const onFrame = useCallback((frame: PcmFrame) => {
    if (frame.samples.length === 0) return;
    if (rateRef.current === null) {
      rateRef.current = frame.sampleRate;
    } else if (frame.sampleRate !== rateRef.current) {
      // The hardware switched rates mid-phrase (headset (dis)connect). The
      // take is discarded on stop with an honest note — never mis-rated audio.
      mixedRateRef.current = true;
      return;
    }
    chunksRef.current.push(frame.samples);
    if (mountedRef.current) {
      setRecordedMs(
        takeDurationMs({
          chunks: chunksRef.current,
          sampleRate: rateRef.current,
        }),
      );
    }
  }, []);

  const handleRecord = useCallback(async () => {
    if (sourceRef.current) return;
    setTakeNote(null);
    chunksRef.current = [];
    rateRef.current = null;
    mixedRateRef.current = false;
    setRecordedMs(0);
    const source = depsRef.current!.makeSource();
    try {
      await source.start(onFrame);
    } catch (e) {
      // Keep the honest detail visible (the vc29 lesson: never swallow a cause).
      const detail = e instanceof Error && e.message ? ` (${e.message})` : "";
      if (mountedRef.current) {
        setTakeNote(`Couldn’t start the microphone. Please try again.${detail}`);
      }
      return;
    }
    sourceRef.current = source;
    if (mountedRef.current) setRecording(true);
  }, [onFrame]);

  const uploadTakes = useCallback(async () => {
    setStage("uploading");
    setErrorText(null);
    let wav: Uint8Array;
    try {
      wav = concatTakesToWav(takesRef.current);
    } catch {
      // Rates diverged across phrases (mic/headset changed between takes) or
      // nothing usable was captured — honest reset, never a detuned upload.
      if (mountedRef.current) {
        setErrorText(
          "The phrases were recorded with different microphone settings and " +
            "can’t be combined — please start over.",
        );
        setStage("error");
      }
      return;
    }
    try {
      const file = await depsRef.current!.saveWav(wav);
      // The person is passed only when training someone else's voice, so
      // the owner's upload is byte-for-byte the pre-existing call.
      const result = personRef.current
        ? await depsRef.current!.enroll(file, "guided-enrollment.wav", personRef.current)
        : await depsRef.current!.enroll(file, "guided-enrollment.wav");
      if (mountedRef.current) {
        setEnrollCount(result.enroll_count);
        setStage("success");
      }
    } catch (e) {
      if (!mountedRef.current) return;
      const status = (e as { status?: number }).status;
      const msg = e instanceof Error ? e.message : "";
      // A real server reason (422 not enough speech, 413 too large, 503 voice
      // ID unavailable) is shown verbatim; transport failures get an honest
      // generic line. Both offer a retry — the takes are still in memory.
      const hasDetail =
        typeof status === "number" && status > 0 && msg.length > 0 &&
        !msg.startsWith("API error");
      setErrorText(
        hasDetail
          ? msg
          : "Couldn’t upload your voice sample — check your connection and try again.",
      );
      setStage("error");
    }
  }, []);

  const handleStop = useCallback(async () => {
    stopSource();
    setRecording(false);
    const take: PhraseTake = {
      chunks: chunksRef.current,
      sampleRate: rateRef.current ?? 0,
    };
    chunksRef.current = [];
    rateRef.current = null;
    if (mixedRateRef.current) {
      mixedRateRef.current = false;
      setTakeNote(
        "The microphone changed its settings mid-phrase — please record this phrase again.",
      );
      return;
    }
    if (takeDurationMs(take) < MIN_TAKE_MS) {
      setTakeNote(
        "We didn’t hear enough — please read the whole phrase aloud and try again.",
      );
      return;
    }
    setTakeNote(null);
    takesRef.current = [...takesRef.current, take];
    if (takesRef.current.length >= PHRASES.length) {
      await uploadTakes();
      return;
    }
    setPhraseIndex(takesRef.current.length);
  }, [stopSource, uploadTakes]);

  const handleStartOver = useCallback(() => {
    stopSource();
    takesRef.current = [];
    chunksRef.current = [];
    rateRef.current = null;
    mixedRateRef.current = false;
    setRecording(false);
    setRecordedMs(0);
    setTakeNote(null);
    setErrorText(null);
    setPhraseIndex(0);
    setStage("phrase");
  }, [stopSource]);

  const handleCancel = useCallback(() => {
    stopSource();
    onCancel();
  }, [onCancel, stopSource]);

  const cancelLink = (
    <TouchableOpacity
      testID="vt-cancel"
      accessibilityRole="button"
      style={styles.cancelButton}
      onPress={handleCancel}
    >
      <Text style={styles.cancelText}>Cancel</Text>
    </TouchableOpacity>
  );

  // --- Web: the PcmSource capture engine is mobile-only; say so honestly. ---
  if (Platform.OS === "web") {
    return (
      <View style={styles.container} testID="voice-training-flow">
        <Text style={styles.note} testID="vt-web-note">
          Voice training records with the phone microphone and isn’t available
          in the browser — use the mobile app.
        </Text>
        {cancelLink}
      </View>
    );
  }

  // --- Permission gate: honest, with a grant retry. Never a dead screen. ---
  if (permGranted !== true) {
    return (
      <View style={styles.container} testID="voice-training-flow">
        <View testID="vt-permission-gate">
          <Text style={styles.note}>
            Microphone access is needed to record the training phrases.
          </Text>
          <TouchableOpacity
            testID="vt-grant-mic"
            accessibilityRole="button"
            style={styles.primaryButton}
            onPress={() =>
              void (async () => {
                const granted = await depsRef.current!.requestPermission();
                if (mountedRef.current) setPermGranted(granted);
              })()
            }
          >
            <Text style={styles.primaryButtonText}>Grant access</Text>
          </TouchableOpacity>
        </View>
        {cancelLink}
      </View>
    );
  }

  if (stage === "uploading") {
    return (
      <View style={styles.container} testID="voice-training-flow">
        <View style={styles.uploadingRow} testID="vt-uploading">
          <ActivityIndicator size="small" color="#4A90D9" />
          <Text style={styles.note}>{`Teaching the app ${whose} voice…`}</Text>
        </View>
      </View>
    );
  }

  if (stage === "success") {
    return (
      <View style={styles.container} testID="voice-training-flow">
        <View testID="vt-success">
          <Text style={styles.successTitle}>Voice trained</Text>
          <Text style={styles.note}>
            {`${person ? `${person.displayName}’s` : "Your"} voice profile now blends ${enrollCount} sample${enrollCount === 1 ? "" : "s"}. ` +
              `MindShift stores a numeric voice signature, never ${person ? "the" : "your"} audio.`}
          </Text>
          <Text style={styles.note} testID="vt-success-catchup-note">
            {`Training ${whose} voice doesn’t relabel recordings you’ve already ` +
              "stored — open “Your growth” and tap “Catch up my past " +
              "recordings” to match it against everything you recorded before " +
              "today."}
          </Text>
        </View>
        <TouchableOpacity
          testID="vt-success-done"
          accessibilityRole="button"
          style={styles.primaryButton}
          onPress={() => onDone(enrollCount ?? 0)}
        >
          <Text style={styles.primaryButtonText}>Done</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (stage === "error") {
    return (
      <View style={styles.container} testID="voice-training-flow">
        <Text style={styles.errorText} testID="vt-error">
          {errorText}
        </Text>
        <TouchableOpacity
          testID="vt-retry-upload"
          accessibilityRole="button"
          style={styles.primaryButton}
          onPress={() => void uploadTakes()}
        >
          <Text style={styles.primaryButtonText}>Try again</Text>
        </TouchableOpacity>
        <TouchableOpacity
          testID="vt-start-over"
          accessibilityRole="button"
          style={styles.secondaryButton}
          onPress={handleStartOver}
        >
          <Text style={styles.secondaryButtonText}>Start over</Text>
        </TouchableOpacity>
        {cancelLink}
      </View>
    );
  }

  // --- The phrase card: read → record → stop → next. ---
  return (
    <View style={styles.container} testID="voice-training-flow">
      <Text style={styles.progress} testID="vt-progress">
        {`Phrase ${phraseIndex + 1} of ${PHRASES.length}`}
      </Text>
      <Text style={styles.phrase} testID="vt-phrase">
        {`“${PHRASES[phraseIndex]}”`}
      </Text>
      {recording ? (
        <>
          <Text style={styles.recordingStatus} testID="vt-elapsed">
            {`Recording… ${(recordedMs / 1000).toFixed(1)}s`}
          </Text>
          <TouchableOpacity
            testID="vt-stop"
            accessibilityRole="button"
            accessibilityLabel="Stop recording this phrase"
            style={styles.primaryButton}
            onPress={() => void handleStop()}
          >
            <Text style={styles.primaryButtonText}>Stop</Text>
          </TouchableOpacity>
        </>
      ) : (
        <>
          {takeNote ? (
            <Text style={styles.takeNote} testID="vt-take-note">
              {takeNote}
            </Text>
          ) : null}
          <TouchableOpacity
            testID="vt-record"
            accessibilityRole="button"
            accessibilityLabel="Record this phrase"
            style={styles.primaryButton}
            onPress={() => void handleRecord()}
          >
            <Text style={styles.primaryButtonText}>
              {takeNote ? "Record again" : "Record"}
            </Text>
          </TouchableOpacity>
        </>
      )}
      {cancelLink}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: "#F0F1F3",
  },
  progress: {
    fontSize: 12.5,
    fontWeight: "700",
    letterSpacing: 0.4,
    textTransform: "uppercase",
    color: "#9CA3AF",
    marginBottom: 6,
  },
  phrase: {
    fontSize: 16,
    lineHeight: 23,
    fontWeight: "600",
    color: "#1F2937",
    marginBottom: 10,
  },
  note: {
    fontSize: 13.5,
    lineHeight: 19,
    color: "#6B7280",
    marginBottom: 10,
  },
  recordingStatus: {
    fontSize: 13,
    fontWeight: "600",
    color: "#DC2626",
    marginBottom: 10,
  },
  takeNote: {
    fontSize: 13,
    lineHeight: 19,
    fontWeight: "600",
    color: "#B45309",
    marginBottom: 10,
  },
  errorText: {
    fontSize: 13,
    lineHeight: 19,
    fontWeight: "600",
    color: "#DC2626",
    marginBottom: 10,
  },
  successTitle: {
    fontSize: 15,
    fontWeight: "700",
    color: "#059669",
    marginBottom: 4,
  },
  uploadingRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  primaryButton: {
    backgroundColor: "#4A90D9",
    borderRadius: 10,
    minHeight: 44,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 16,
    marginBottom: 8,
  },
  primaryButtonText: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "700",
  },
  secondaryButton: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 10,
    minHeight: 44,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 16,
    marginBottom: 8,
  },
  secondaryButtonText: {
    color: "#1F2937",
    fontSize: 14,
    fontWeight: "600",
  },
  cancelButton: {
    minHeight: 40,
    alignItems: "center",
    justifyContent: "center",
  },
  cancelText: {
    color: "#6B7280",
    fontSize: 14,
    fontWeight: "600",
  },
});
