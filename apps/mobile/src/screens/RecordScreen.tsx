import React from "react";
import { View, Text, TouchableOpacity, StyleSheet, Platform } from "react-native";
import type { RecordedFile } from "../store/recorderStore";

// Re-export the pure helpers so existing imports/tests keep one entry point.
export {
  MAX_RECORDING_SECONDS,
  remainingSeconds,
  isAtCap,
  formatClock,
} from "./recordTiming";

export interface RecordScreenProps {
  onBack: () => void;
  onComplete: (file: RecordedFile) => void;
}

/**
 * Platform gate for in-app recording.
 *
 * expo-media-library has NO web implementation — merely importing it on web
 * throws "Cannot find native module 'ExpoMediaLibraryNext'" at bundle load and
 * blanks the entire app (the same failure class as the original web
 * blank-screen bug). So the native implementation lives in
 * RecordScreenNative.tsx and is require()d ONLY off-web, at render time; the
 * web build never executes that module.
 */
export default function RecordScreen(props: RecordScreenProps) {
  if (Platform.OS === "web") {
    return (
      <View style={styles.center} testID="record-web-note">
        <Text style={styles.note}>
          In-app recording is mobile-only for now — upload a file or paste a
          link instead.
        </Text>
        <TouchableOpacity
          style={styles.backButton}
          onPress={props.onBack}
          testID="record-web-back"
        >
          <Text style={styles.backText}>Back</Text>
        </TouchableOpacity>
      </View>
    );
  }
  // Lazy, native-only module resolution (see doc comment above).
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const Native = require("./RecordScreenNative").default;
  return <Native {...props} />;
}

const styles = StyleSheet.create({
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 32,
    backgroundColor: "#F9FAFB",
  },
  note: {
    fontSize: 15,
    color: "#6B7280",
    textAlign: "center",
    marginBottom: 20,
  },
  backButton: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 10,
    paddingVertical: 12,
    paddingHorizontal: 24,
  },
  backText: {
    color: "#1F2937",
    fontWeight: "600",
  },
});
