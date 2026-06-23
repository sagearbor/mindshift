import React, { useState, useCallback } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Switch,
  StyleSheet,
} from "react-native";
import EmpathySlider from "../components/EmpathySlider";
import SuggestionCard from "../components/SuggestionCard";
import LiveTranscript from "../components/LiveTranscript";
import { useAudioStream } from "../hooks/useAudioStream";

const STATUS_COLORS: Record<string, string> = {
  idle: "#9CA3AF",
  connecting: "#F59E0B",
  live: "#10B981",
  disconnected: "#EF4444",
};

export default function LiveCoachScreen() {
  const {
    isRecording,
    transcript,
    suggestions,
    connectionStatus,
    transcriptionMessage,
    startSession,
    stopSession,
    sendEmpathyUpdate,
  } = useAudioStream();

  const [empathyLevel, setEmpathyLevel] = useState(50);
  const [coachMode, setCoachMode] = useState<"earpiece" | "visual">(
    "visual",
  );

  const handleToggle = useCallback(async () => {
    if (isRecording) {
      await stopSession();
    } else {
      const sessionId = `live-${Date.now()}`;
      await startSession(sessionId, empathyLevel);
    }
  }, [isRecording, stopSession, startSession, empathyLevel]);

  const handleEmpathyChange = useCallback(
    (value: number) => {
      setEmpathyLevel(value);
      sendEmpathyUpdate(value);
    },
    [sendEmpathyUpdate],
  );

  const statusColor = STATUS_COLORS[connectionStatus] || STATUS_COLORS.idle;

  return (
    <View style={styles.container}>
      {/* Header with connection status */}
      <View style={styles.header}>
        <Text style={styles.heading}>Live Coach</Text>
        <View style={styles.statusRow}>
          <View
            style={[styles.statusDot, { backgroundColor: statusColor }]}
            testID="connection-status"
          />
          <Text style={[styles.statusText, { color: statusColor }]}>
            {connectionStatus}
          </Text>
        </View>
      </View>

      {/* Transcription availability banner */}
      {transcriptionMessage ? (
        <View style={styles.banner} testID="transcription-banner">
          <Text style={styles.bannerText}>
            Transcription unavailable: {transcriptionMessage}
          </Text>
        </View>
      ) : null}

      {/* Coach mode toggle */}
      <View style={styles.modeRow}>
        <Text style={styles.modeLabel}>Coach mode:</Text>
        <TouchableOpacity
          testID="mode-earpiece"
          style={[
            styles.modeButton,
            coachMode === "earpiece" && styles.modeButtonActive,
          ]}
          onPress={() => setCoachMode("earpiece")}
        >
          <Text
            style={[
              styles.modeButtonText,
              coachMode === "earpiece" && styles.modeButtonTextActive,
            ]}
          >
            Earpiece
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          testID="mode-visual"
          style={[
            styles.modeButton,
            coachMode === "visual" && styles.modeButtonActive,
          ]}
          onPress={() => setCoachMode("visual")}
        >
          <Text
            style={[
              styles.modeButtonText,
              coachMode === "visual" && styles.modeButtonTextActive,
            ]}
          >
            Visual
          </Text>
        </TouchableOpacity>
      </View>

      {/* Empathy slider */}
      <EmpathySlider
        value={empathyLevel}
        onValueChange={handleEmpathyChange}
      />

      {/* Live transcript */}
      <LiveTranscript entries={transcript} />

      {/* Suggestions */}
      {suggestions.length > 0 && (
        <ScrollView
          style={styles.suggestionsContainer}
          horizontal={false}
          testID="suggestions-list"
        >
          <Text style={styles.suggestionsTitle}>Suggestions</Text>
          {suggestions.map((s, i) => (
            <SuggestionCard key={i} text={s.text} tone={s.tone} />
          ))}
        </ScrollView>
      )}

      {/* Start/Stop button */}
      <TouchableOpacity
        testID="mic-toggle"
        style={[
          styles.micButton,
          isRecording && styles.micButtonRecording,
        ]}
        onPress={handleToggle}
      >
        <Text style={styles.micButtonText}>
          {isRecording ? "Stop Listening" : "Start Listening"}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F9FAFB",
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: 8,
  },
  heading: {
    fontSize: 24,
    fontWeight: "700",
    color: "#111827",
  },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  statusText: {
    fontSize: 13,
    fontWeight: "600",
    textTransform: "capitalize",
  },
  banner: {
    backgroundColor: "#FEF3C7",
    borderLeftWidth: 4,
    borderLeftColor: "#F59E0B",
    marginHorizontal: 16,
    marginBottom: 4,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
  },
  bannerText: {
    fontSize: 13,
    color: "#92400E",
  },
  modeRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 8,
    gap: 8,
  },
  modeLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: "#374151",
    marginRight: 4,
  },
  modeButton: {
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#FFFFFF",
  },
  modeButtonActive: {
    backgroundColor: "#4A90D9",
    borderColor: "#4A90D9",
  },
  modeButtonText: {
    fontSize: 13,
    fontWeight: "500",
    color: "#6B7280",
  },
  modeButtonTextActive: {
    color: "#FFFFFF",
  },
  suggestionsContainer: {
    maxHeight: 200,
    paddingBottom: 8,
  },
  suggestionsTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#1F2937",
    paddingHorizontal: 16,
    paddingTop: 8,
    marginBottom: 4,
  },
  micButton: {
    backgroundColor: "#4A90D9",
    marginHorizontal: 16,
    marginVertical: 12,
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: "center",
  },
  micButtonRecording: {
    backgroundColor: "#EF4444",
  },
  micButtonText: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "700",
  },
});
