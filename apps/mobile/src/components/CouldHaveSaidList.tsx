import React from "react";
import { View, Text, StyleSheet } from "react-native";

import type { CouldHaveSaid } from "../api/client";
import { toneChipColors } from "../screens/toneTrends";

const INK = "#1F2937";
const MUTED = "#6B7280";
const PRIMARY = "#4A90D9";

interface CouldHaveSaidListProps {
  items: CouldHaveSaid[];
  /** The transcript the reflections index into (turn_index → text), so each
   *  card can quote what was actually said. Missing indexes just omit the
   *  quote — never a fabricated line. */
  turns?: { text: string }[] | null;
  testID?: string;
}

/**
 * "What you could have said" — the post-session reflection over the user's
 * OWN turns, shared by ReplayScreen and the therapist SessionDetail. One
 * card per reflected turn: the original words (quoted, verbatim), the
 * suggested alternative, the one-line why, and a tone-read chip naming how
 * the original landed. The list is the server's cached reflection; this
 * component never asks the model itself.
 */
export default function CouldHaveSaidList({
  items,
  turns,
  testID = "could-have-said",
}: CouldHaveSaidListProps) {
  if (items.length === 0) return null;
  return (
    <View testID={testID}>
      {items.map((item) => {
        const original = turns?.[item.turn_index]?.text;
        const chip = item.tone_read ? toneChipColors(item.tone_read) : null;
        return (
          <View
            key={item.turn_index}
            style={styles.card}
            testID={`${testID}-${item.turn_index}`}
          >
            <View style={styles.headRow}>
              <Text style={styles.turnLabel}>Turn {item.turn_index + 1}</Text>
              {chip && item.tone_read ? (
                <View style={[styles.chip, { backgroundColor: chip.bg }]}>
                  <Text style={[styles.chipText, { color: chip.fg }]}>
                    {item.tone_read}
                  </Text>
                </View>
              ) : null}
            </View>
            {original ? (
              <Text style={styles.original} numberOfLines={3}>
                “{original}”
              </Text>
            ) : null}
            <Text style={styles.suggestion}>{item.could_have_said}</Text>
            {item.why ? <Text style={styles.why}>{item.why}</Text> : null}
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    borderLeftWidth: 4,
    borderLeftColor: PRIMARY,
    padding: 12,
    marginBottom: 8,
  },
  headRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 4,
  },
  turnLabel: {
    fontSize: 11.5,
    fontWeight: "700",
    color: MUTED,
    textTransform: "uppercase",
    letterSpacing: 0.4,
  },
  chip: {
    borderRadius: 10,
    paddingVertical: 2,
    paddingHorizontal: 8,
  },
  chipText: {
    fontSize: 11.5,
    fontWeight: "600",
  },
  original: {
    fontSize: 13,
    color: MUTED,
    fontStyle: "italic",
    marginBottom: 6,
  },
  suggestion: {
    fontSize: 14.5,
    lineHeight: 20,
    color: INK,
    fontWeight: "600",
  },
  why: {
    fontSize: 12.5,
    color: MUTED,
    marginTop: 4,
  },
});
