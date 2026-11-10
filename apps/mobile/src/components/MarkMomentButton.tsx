/**
 * "Mark this moment" on the Live Coach screen (Developer mode + Emotion dev
 * tester ON). Tapping it keeps the previous 45 s and the next 15 s of the
 * running session's audio on this phone with a user mark in the emotion log,
 * independent of what the models said. Renders nothing when the tester is off.
 */
import { useEffect, useState } from "react";
import { StyleSheet, Text, TouchableOpacity } from "react-native";
import * as Haptics from "expo-haptics";
import type { DevStatus } from "../emotion/devHub";
import { emotionDevHub } from "../emotion/hub";

export function markButtonState(s: DevStatus): "hidden" | "disabled" | "ready" {
  if (s.state === "off") return "hidden";
  return s.state === "running" && !s.snapshot.paused ? "ready" : "disabled";
}

export function MarkMomentButton() {
  const [status, setStatus] = useState<DevStatus>(() => emotionDevHub.status());
  const [note, setNote] = useState<string | null>(null);
  useEffect(() => emotionDevHub.subscribe(setStatus), []);
  useEffect(() => {
    if (!note) return;
    const t = setTimeout(() => setNote(null), 4000);
    return () => clearTimeout(t);
  }, [note]);

  const state = markButtonState(status);
  if (state === "hidden") return null;

  const press = () => {
    const id = emotionDevHub.mark();
    setNote(id ? "Marked. Keeping the 45 s before and the next 15 s." : "Not listening right now, nothing marked.");
    if (id) void Promise.resolve(Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success)).catch(() => {});
  };

  return (
    <>
      <TouchableOpacity
        testID="mark-moment"
        accessibilityRole="button"
        disabled={state !== "ready"}
        style={[styles.btn, state !== "ready" && styles.off]}
        onPress={press}
      >
        <Text style={styles.text}>Mark this moment</Text>
      </TouchableOpacity>
      {note ? (
        <Text style={styles.note} testID="mark-moment-note">
          {note}
        </Text>
      ) : null}
    </>
  );
}

const styles = StyleSheet.create({
  btn: { marginHorizontal: 16, marginTop: 8, paddingVertical: 14, borderRadius: 12, backgroundColor: "#7C3AED", alignItems: "center" },
  off: { backgroundColor: "#9CA3AF" },
  text: { color: "#FFFFFF", fontSize: 16, fontWeight: "700" },
  note: { marginHorizontal: 16, marginTop: 6, fontSize: 13, color: "#6B7280" },
});
