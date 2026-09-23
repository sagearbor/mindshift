import React, { useEffect, useRef, useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet, Platform } from "react-native";
import type { RecorderSessionStore } from "./sessionStore";

interface RecoveryHomeCardProps {
  /** Test seam; production lazily builds the expo-file-system-backed store
   *  exactly as RecoveryPrompt does. */
  store?: RecorderSessionStore;
  /** Where the recovery actually happens: the Analyze screen's RecoveryPrompt. */
  onOpen: () => void;
}

/**
 * Home's pointer to an unfinished recording. The real recovery card lives on
 * Analyze (RecoveryPrompt), but after a crash the app reopens on HOME with no
 * hint that anything was saved — a user who never thinks to open Analyze
 * never finds their audio (UX walk 2026-09-23). This card only says it exists
 * and takes them there; it recovers or deletes nothing itself.
 */
export default function RecoveryHomeCard({ store, onOpen }: RecoveryHomeCardProps) {
  const storeRef = useRef<RecorderSessionStore | null>(store ?? null);
  const [count, setCount] = useState(0);

  useEffect(() => {
    try {
      if (!storeRef.current) {
        if (Platform.OS === "web") return; // no native filesystem to scan
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const { RecorderSessionStore: Store } = require("./sessionStore");
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const { ExpoRecorderFs } = require("./expoFs");
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const { currentRecordingOwnerUid } = require("./owner");
        storeRef.current = new Store(new ExpoRecorderFs(), currentRecordingOwnerUid);
      }
      const s = storeRef.current;
      setCount((s?.listRecoverable().length ?? 0) + (s?.listOrphanStitched().length ?? 0));
    } catch {
      // A failed scan means we can't point at anything; Analyze will re-scan.
    }
  }, []);

  if (count === 0) return null;
  return (
    <TouchableOpacity
      testID="recovery-home-card"
      accessibilityRole="button"
      style={styles.card}
      onPress={onOpen}
    >
      <Text style={styles.title}>
        {count === 1
          ? "An unfinished recording was saved on this phone"
          : `${count} unfinished recordings were saved on this phone`}
      </Text>
      <Text style={styles.body}>Open Analyze to recover or discard it.</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#FEF3C7",
    borderColor: "#FDE68A",
    borderWidth: 1,
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
  },
  title: { fontSize: 14, fontWeight: "700", color: "#92400E" },
  body: { fontSize: 13, color: "#92400E", marginTop: 2 },
});
