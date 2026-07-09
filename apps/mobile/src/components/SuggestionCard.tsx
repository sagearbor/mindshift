import React from "react";
import { View, Text, StyleSheet } from "react-native";

export interface Suggestion {
  text: string;
  tone: string;
  /** True when the server chose not to voice this suggestion (speak: false).
   *  Dims the card instead of hiding it — the advice is still worth reading. */
  muted?: boolean;
}

const TONE_COLORS: Record<string, string> = {
  empathetic: "#10B981",
  validating: "#6366F1",
  balanced: "#F59E0B",
  assertive: "#EF4444",
  warm: "#F97316",
  constructive: "#0EA5E9",
  neutral: "#6B7280",
};

export function getToneColor(tone: string): string {
  return TONE_COLORS[tone.toLowerCase()] || TONE_COLORS.neutral;
}

export default function SuggestionCard({
  text,
  tone,
  muted = false,
}: Suggestion) {
  const badgeColor = getToneColor(tone);

  return (
    <View
      style={[styles.card, muted && styles.cardMuted]}
      testID="suggestion-card"
    >
      <View style={[styles.badge, { backgroundColor: badgeColor }]}>
        <Text style={styles.badgeText}>{tone}</Text>
      </View>
      <Text style={styles.text}>{text}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 16,
    marginVertical: 6,
    marginHorizontal: 16,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  cardMuted: {
    opacity: 0.5,
  },
  badge: {
    alignSelf: "flex-start",
    paddingVertical: 3,
    paddingHorizontal: 10,
    borderRadius: 12,
    marginBottom: 8,
  },
  badgeText: {
    fontSize: 12,
    fontWeight: "600",
    color: "#FFFFFF",
    textTransform: "capitalize",
  },
  text: {
    fontSize: 15,
    lineHeight: 22,
    color: "#1F2937",
  },
});
