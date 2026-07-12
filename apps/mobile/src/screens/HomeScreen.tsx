import React from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

/**
 * The two-mode home screen. There are exactly two things you'd do with this
 * app — coach a conversation live, or analyze one afterwards — so the home
 * screen is exactly two huge buttons, a compact "past recordings" entry, and
 * a small corner affordance for everything else (Advanced). No forms, no
 * settings, no clutter: users may open this mid-conflict and stressed, so the
 * primary targets are enormous and unambiguous.
 */
interface HomeScreenProps {
  onLiveCoach: () => void;
  onAnalyze: () => void;
  onOpenRecordings: () => void;
  onOpenAdvanced: () => void;
}

export default function HomeScreen({
  onLiveCoach,
  onAnalyze,
  onOpenRecordings,
  onOpenAdvanced,
}: HomeScreenProps) {
  return (
    <View style={styles.container} testID="home-screen">
      {/* Top bar: wordmark + the small Advanced corner affordance. */}
      <View style={styles.topBar}>
        <Text style={styles.wordmark}>MindShift</Text>
        <TouchableOpacity
          testID="home-advanced-button"
          accessibilityRole="button"
          accessibilityLabel="Advanced"
          style={styles.advancedButton}
          onPress={onOpenAdvanced}
          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        >
          <Text style={styles.advancedGlyph}>⋯</Text>
        </TouchableOpacity>
      </View>

      {/* The two modes. */}
      <View style={styles.cards}>
        <TouchableOpacity
          testID="home-live-coach"
          accessibilityRole="button"
          style={[styles.card, styles.liveCard]}
          onPress={onLiveCoach}
          activeOpacity={0.85}
        >
          <Text style={styles.liveCardBadge}>LIVE</Text>
          <Text style={styles.liveCardTitle}>Live Coach</Text>
          <Text style={styles.liveCardSub}>
            Real-time coaching in your ear while you talk.
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          testID="home-analyze"
          accessibilityRole="button"
          style={[styles.card, styles.analyzeCard]}
          onPress={onAnalyze}
          activeOpacity={0.85}
        >
          <Text style={styles.analyzeCardBadge}>AFTERWARDS</Text>
          <Text style={styles.analyzeCardTitle}>Analyze a Conversation</Text>
          <Text style={styles.analyzeCardSub}>
            Record, upload, or paste a link — get the full breakdown.
          </Text>
        </TouchableOpacity>
      </View>

      {/* Compact history entry point — the recordings/replay flow is a
          flagship feature and stays one tap from home. */}
      <TouchableOpacity
        testID="home-recordings-link"
        accessibilityRole="button"
        style={styles.recordingsRow}
        onPress={onOpenRecordings}
      >
        <Text style={styles.recordingsRowText}>▶ Past recordings</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 24,
    paddingBottom: 20,
    backgroundColor: "#F9FAFB",
  },
  topBar: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 16,
  },
  wordmark: {
    fontSize: 22,
    fontWeight: "700",
    color: "#111827",
  },
  advancedButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: "#E5E7EB",
    backgroundColor: "#FFFFFF",
  },
  advancedGlyph: {
    fontSize: 22,
    lineHeight: 24,
    color: "#6B7280",
    fontWeight: "700",
  },
  cards: {
    flex: 1,
    gap: 16,
  },
  card: {
    flex: 1,
    borderRadius: 20,
    padding: 24,
    justifyContent: "flex-end",
  },
  liveCard: {
    backgroundColor: "#4A90D9",
  },
  liveCardBadge: {
    position: "absolute",
    top: 20,
    left: 24,
    fontSize: 13,
    fontWeight: "800",
    letterSpacing: 2,
    color: "rgba(255,255,255,0.85)",
  },
  liveCardTitle: {
    fontSize: 30,
    fontWeight: "800",
    color: "#FFFFFF",
    marginBottom: 6,
  },
  liveCardSub: {
    fontSize: 15,
    lineHeight: 21,
    color: "rgba(255,255,255,0.92)",
  },
  analyzeCard: {
    backgroundColor: "#FFFFFF",
    borderWidth: 1.5,
    borderColor: "#D1D5DB",
  },
  analyzeCardBadge: {
    position: "absolute",
    top: 20,
    left: 24,
    fontSize: 13,
    fontWeight: "800",
    letterSpacing: 2,
    color: "#9CA3AF",
  },
  analyzeCardTitle: {
    fontSize: 30,
    fontWeight: "800",
    color: "#111827",
    marginBottom: 6,
  },
  analyzeCardSub: {
    fontSize: 15,
    lineHeight: 21,
    color: "#6B7280",
  },
  recordingsRow: {
    marginTop: 16,
    minHeight: 52,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#FFFFFF",
    alignItems: "center",
    justifyContent: "center",
  },
  recordingsRowText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#4A90D9",
  },
});
