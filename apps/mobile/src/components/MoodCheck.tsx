import React from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

interface Props {
  /** Before a session starts, or after one ends — only the copy differs. */
  phase: "before" | "after";
  /** 1–9, or null (not answered / skipped). Controlled — the screen owns
   *  where this lives (moodStore.ts). */
  value: number | null;
  /** A number tap reports 1–9; the Skip tap reports null. */
  onChange: (value: number | null) => void;
}

const NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9];

const TITLE: Record<Props["phase"], string> = {
  before: "How are you feeling right now?",
  after: "How are you feeling now?",
};

/**
 * CANDOR's single outcome item ("To what extent do you feel positive
 * feelings … or negative feelings … right now?", 1–9) as a compact,
 * one-tap row — the app's therapy-evidence primitive (post-conversation
 * mood improved for 66% of people in CANDOR, median +1). Matches the card
 * styling of SessionSummaryCard / LivePreflightPanel.
 */
export default function MoodCheck({ phase, value, onChange }: Props) {
  return (
    <View style={styles.card} testID={`mood-check-${phase}`}>
      <Text style={styles.title}>{TITLE[phase]}</Text>
      <View style={styles.row}>
        <Text style={styles.endLabel} accessibilityElementsHidden>
          😞
        </Text>
        {NUMBERS.map((n) => {
          const selected = value === n;
          return (
            <TouchableOpacity
              key={n}
              testID={`mood-check-option-${n}`}
              accessibilityRole="button"
              accessibilityLabel={`${n} out of 9`}
              accessibilityState={{ selected }}
              style={[styles.option, selected && styles.optionSelected]}
              onPress={() => onChange(n)}
            >
              <Text style={[styles.optionText, selected && styles.optionTextSelected]}>{n}</Text>
            </TouchableOpacity>
          );
        })}
        <Text style={styles.endLabel} accessibilityElementsHidden>
          😊
        </Text>
      </View>
      <TouchableOpacity
        testID="mood-check-skip"
        accessibilityRole="button"
        style={styles.skip}
        onPress={() => onChange(null)}
      >
        <Text style={styles.skipText}>Skip</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    marginHorizontal: 16,
    marginVertical: 6,
    padding: 12,
    gap: 8,
  },
  title: {
    fontSize: 13.5,
    fontWeight: "700",
    color: "#374151",
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 3,
  },
  endLabel: {
    fontSize: 16,
  },
  option: {
    width: 26,
    height: 26,
    borderRadius: 13,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#F9FAFB",
    alignItems: "center",
    justifyContent: "center",
  },
  optionSelected: {
    borderColor: "#4A90D9",
    backgroundColor: "#4A90D9",
  },
  optionText: {
    fontSize: 12,
    fontWeight: "600",
    color: "#374151",
  },
  optionTextSelected: {
    color: "#FFFFFF",
  },
  skip: {
    alignSelf: "flex-end",
  },
  skipText: {
    fontSize: 12,
    color: "#9CA3AF",
    fontWeight: "600",
  },
});
