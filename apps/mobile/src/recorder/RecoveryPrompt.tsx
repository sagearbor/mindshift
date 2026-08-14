import React, { useEffect, useRef, useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet, Platform } from "react-native";
import type { RecorderSessionStore } from "./sessionStore";
import type { RecordedAudioFile, RecoverableSession } from "./types";

interface RecoveryPromptProps {
  /** Test seam; production lazily builds the expo-file-system-backed store. */
  store?: RecorderSessionStore;
  /** The recovered, stitched audio file — ready for the upload flow. */
  onRecovered: (file: RecordedAudioFile) => void;
}

/**
 * The other half of crash-hardening: on the Analyze screen, look for sessions
 * that never finished (app crash, battery death, force-kill mid-recording)
 * and offer the saved audio back. Nothing is recovered or deleted without the
 * user choosing.
 */
export default function RecoveryPrompt({
  store,
  onRecovered,
}: RecoveryPromptProps) {
  const storeRef = useRef<RecorderSessionStore | null>(store ?? null);
  const [pending, setPending] = useState<RecoverableSession[]>([]);
  const [orphans, setOrphans] = useState<RecordedAudioFile[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      if (!storeRef.current) {
        if (Platform.OS === "web") return; // no native filesystem to scan
        // Lazy production wiring — never constructed under test injection.
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const { RecorderSessionStore: Store } = require("./sessionStore");
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const { ExpoRecorderFs } = require("./expoFs");
        storeRef.current = new Store(new ExpoRecorderFs());
      }
      setPending(storeRef.current?.listRecoverable() ?? []);
      setOrphans(storeRef.current?.listOrphanStitched() ?? []);
    } catch {
      // A failed scan means we can't OFFER recovery — the segments (if any)
      // stay untouched on disk for a later launch to find.
    }
  }, []);

  const current = pending[0];
  const orphan = current ? null : (orphans[0] ?? null);
  if (!current && !orphan) return null;

  // Orphaned stitched output (recovered earlier, never uploaded): offer it
  // the same way — audio must never be stranded without a UI path back.
  if (orphan) {
    const mb = Math.max(1, Math.round((orphan.size ?? 0) / (1024 * 1024)));
    return (
      <View style={styles.card} testID="orphan-prompt">
        <Text style={styles.title}>
          Found a recovered recording that was never analyzed ({mb} MB) — use
          it?
        </Text>
        <Text style={styles.body}>
          This audio was rescued from an earlier session but its upload never
          completed. It’s still saved on this phone.
        </Text>
        <View style={styles.row}>
          <TouchableOpacity
            testID="orphan-use"
            style={styles.recoverButton}
            onPress={() => {
              setOrphans((rest) => rest.slice(1));
              onRecovered(orphan);
            }}
          >
            <Text style={styles.recoverText}>Use it</Text>
          </TouchableOpacity>
          <TouchableOpacity
            testID="orphan-discard"
            style={styles.discardButton}
            onPress={() => {
              try {
                storeRef.current?.discardOrphan(orphan.uri);
              } catch {
                // Even if deletion failed we stop prompting this launch.
              }
              setOrphans((rest) => rest.slice(1));
            }}
          >
            <Text style={styles.discardText}>Discard</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  const minutes = Math.max(1, Math.round(current.totalDurationMs / 60000));

  const handleRecover = () => {
    const s = storeRef.current;
    if (!s) return;
    try {
      const file = s.finishToFile(current.manifest);
      setPending((rest) => rest.slice(1));
      setError(null);
      onRecovered(file);
    } catch {
      // Stitching failed — the session stays on disk; be honest about it.
      setError(
        "Couldn’t assemble that recording. Its audio is still saved on this phone — you can try again later.",
      );
    }
  };

  const handleDiscard = () => {
    const s = storeRef.current;
    if (!s) return;
    try {
      s.discard(current.manifest.sessionId);
    } catch {
      // Even if deletion failed we stop prompting this launch.
    }
    setError(null);
    setPending((rest) => rest.slice(1));
  };

  return (
    <View style={styles.card} testID="recovery-prompt">
      <Text style={styles.title}>
        Found an unfinished recording ({minutes} min) — recover?
      </Text>
      <Text style={styles.body}>
        A recording session didn’t finish (the app may have closed
        mid-session). The audio saved up to that point is intact.
      </Text>
      {error && (
        <Text style={styles.error} testID="recovery-error">
          {error}
        </Text>
      )}
      <View style={styles.row}>
        <TouchableOpacity
          testID="recovery-recover"
          style={styles.recoverButton}
          onPress={handleRecover}
        >
          <Text style={styles.recoverText}>Recover</Text>
        </TouchableOpacity>
        <TouchableOpacity
          testID="recovery-discard"
          style={styles.discardButton}
          onPress={handleDiscard}
        >
          <Text style={styles.discardText}>Discard</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    borderWidth: 1,
    borderColor: "#F59E0B",
    backgroundColor: "#FFFBEB",
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
  },
  title: { fontSize: 14.5, fontWeight: "700", color: "#92400E", marginBottom: 6 },
  body: { fontSize: 13, lineHeight: 19, color: "#78350F", marginBottom: 10 },
  error: {
    fontSize: 13,
    lineHeight: 19,
    color: "#DC2626",
    fontWeight: "600",
    marginBottom: 10,
  },
  row: { flexDirection: "row", gap: 10 },
  recoverButton: {
    flex: 1,
    backgroundColor: "#B45309",
    borderRadius: 8,
    paddingVertical: 10,
    alignItems: "center",
  },
  recoverText: { color: "#FFFFFF", fontSize: 14, fontWeight: "700" },
  discardButton: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    paddingVertical: 10,
    alignItems: "center",
    backgroundColor: "#FFFFFF",
  },
  discardText: { color: "#6B7280", fontSize: 14, fontWeight: "600" },
});
