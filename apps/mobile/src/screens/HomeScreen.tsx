import React from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

import GrowthStrip from "../components/GrowthStrip";
import HeroWipe from "../components/HeroWipe";

/**
 * The two-mode home screen. There are exactly two things you'd do with this
 * app — coach a conversation live, or analyze one afterwards — so the home
 * screen is exactly two huge buttons, a narrow self-fetching "Your growth"
 * sparkline strip, and a compact history row ("Your day" timeline + "past
 * recordings"). No forms, no settings clutter here: users may open this
 * mid-conflict and stressed, so the primary targets are enormous and
 * unambiguous.
 *
 * Task N3 (P3-10): the wordmark + Settings corner affordance that used to
 * live here moved up into AppChrome's persistent top bar (hamburger +
 * wordmark + avatar, wrapping every primary screen including this one) —
 * Settings is now reached via the avatar menu or the hamburger's catalog, so
 * there's no redundant "⋯" here anymore.
 */
interface HomeScreenProps {
  onLiveCoach: () => void;
  onAnalyze: () => void;
  onOpenRecordings: () => void;
  onOpenYourDay: () => void;
  onOpenGrowth: () => void;
}

export default function HomeScreen({
  onLiveCoach,
  onAnalyze,
  onOpenRecordings,
  onOpenYourDay,
  onOpenGrowth,
}: HomeScreenProps) {
  return (
    <View style={styles.container} testID="home-screen">
      {/* Web-only hero banner (Task P3-4b) — renders nothing on native. */}
      <HeroWipe />

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

      {/* "Your growth" — a narrow, self-fetching sparkline of the user's own
          per-recording scores over time. Renders nothing while loading or when
          growth isn't available (signed out / storage disabled / not enrolled
          server), so the two-mode design is undisturbed by default. */}
      <GrowthStrip onPress={onOpenGrowth} />

      {/* Compact history entry points — small third row under the two mode
          cards: the day timeline (Companion P1) and the recordings/replay
          flow, each one tap from home without disturbing the two-mode design. */}
      <View style={styles.historyRow}>
        <TouchableOpacity
          testID="home-your-day-link"
          accessibilityRole="button"
          style={styles.recordingsRow}
          onPress={onOpenYourDay}
        >
          <Text style={styles.recordingsRowText}>☀ Your day</Text>
        </TouchableOpacity>
        <TouchableOpacity
          testID="home-recordings-link"
          accessibilityRole="button"
          style={styles.recordingsRow}
          onPress={onOpenRecordings}
        >
          <Text style={styles.recordingsRowText}>▶ Past recordings</Text>
        </TouchableOpacity>
      </View>
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
  historyRow: {
    flexDirection: "row",
    gap: 12,
  },
  recordingsRow: {
    flex: 1,
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
