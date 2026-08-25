import React, { useRef, useEffect, useMemo } from "react";
import { View, Text, ScrollView, StyleSheet, TouchableOpacity } from "react-native";
import type { TranscriptEntry } from "../hooks/useAudioStream";
import { getSpeakerColor } from "../utils/speakerColors";

interface Props {
  entries: TranscriptEntry[];
  /** Mid-call naming: tapping a column header or a bubble's speaker opens
   *  "Who is this?" for that voice. Absent → plain text, as before. */
  onSpeakerPress?: (entry: TranscriptEntry) => void;
  /** Whether the entry's speaker already has a name (no "who?" hint then). */
  isNamed?: (entry: TranscriptEntry) => boolean;
}

/** Left/right column assignment: the first two voices heard, in order; any
 *  later voice goes right (a third participant is rare in therapist mode). */
export function columnOf(entries: TranscriptEntry[]): {
  left: string | null;
  right: string | null;
  side: (speaker: string) => "left" | "right";
} {
  const order: string[] = [];
  for (const e of entries) {
    if (!order.includes(e.speaker)) order.push(e.speaker);
    if (order.length === 2) break;
  }
  const left = order[0] ?? null;
  const right = order[1] ?? null;
  return {
    left,
    right,
    side: (speaker) => (speaker === left ? "left" : "right"),
  };
}

/**
 * Therapist mode's transcript: the two people in the room in two labelled
 * columns (bubbles left/right), so the observer can follow who said what at a
 * glance. Nothing is spoken in this mode; this view is the whole output.
 */
export default function TherapistTranscript({ entries, onSpeakerPress, isNamed }: Props) {
  const scrollRef = useRef<ScrollView>(null);
  const columns = useMemo(() => columnOf(entries), [entries]);
  // The first entry for a column's speaker — what a header tap names.
  const entryFor = (speaker: string | null) =>
    speaker === null ? undefined : entries.find((e) => e.speaker === speaker);
  const hintFor = (speaker: string | null) => {
    if (!onSpeakerPress || speaker === null) return false;
    const e = entryFor(speaker);
    return e ? !isNamed?.(e) : false;
  };

  useEffect(() => {
    if (scrollRef.current && entries.length > 0) {
      scrollRef.current.scrollToEnd({ animated: true });
    }
  }, [entries.length]);

  if (entries.length === 0) {
    return (
      <View style={styles.emptyContainer} testID="therapist-transcript-empty">
        <Text style={styles.emptyText}>Waiting for the two of them to talk…</Text>
      </View>
    );
  }

  return (
    <View style={styles.wrap} testID="therapist-transcript">
      <View style={styles.headerRow}>
        <TouchableOpacity
          testID="therapist-column-left-tap"
          accessibilityRole="button"
          disabled={!onSpeakerPress || columns.left === null}
          onPress={() => {
            const e = entryFor(columns.left);
            if (e) onSpeakerPress?.(e);
          }}
        >
          <Text
            style={[styles.headerLeft, { color: getSpeakerColor(columns.left ?? "") }]}
            testID="therapist-column-left"
          >
            {columns.left ?? "—"}
            {hintFor(columns.left) ? <Text style={styles.headerHint}> · who?</Text> : null}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          testID="therapist-column-right-tap"
          accessibilityRole="button"
          disabled={!onSpeakerPress || columns.right === null}
          onPress={() => {
            const e = entryFor(columns.right);
            if (e) onSpeakerPress?.(e);
          }}
        >
          <Text
            style={[styles.headerRight, { color: getSpeakerColor(columns.right ?? "") }]}
            testID="therapist-column-right"
          >
            {hintFor(columns.right) ? <Text style={styles.headerHint}>who? · </Text> : null}
            {columns.right ?? "…"}
          </Text>
        </TouchableOpacity>
      </View>
      <ScrollView ref={scrollRef} style={styles.scroll}>
        {entries.map((entry, i) => {
          const side = columns.side(entry.speaker);
          const color = getSpeakerColor(entry.speaker);
          const isLatest = i === entries.length - 1;
          return (
            <View
              key={i}
              style={[styles.line, side === "right" ? styles.lineRight : styles.lineLeft]}
              testID={`therapist-turn-${i}-${side}`}
            >
              <View
                style={[
                  styles.bubble,
                  side === "right" ? styles.bubbleRight : styles.bubbleLeft,
                  isLatest && { borderColor: color },
                ]}
              >
                {onSpeakerPress ? (
                  <TouchableOpacity
                    testID={`therapist-turn-${i}-speaker`}
                    accessibilityRole="button"
                    accessibilityLabel={`Who is ${entry.speaker}?`}
                    onPress={() => onSpeakerPress(entry)}
                  >
                    <Text style={[styles.speaker, { color }]}>{entry.speaker}</Text>
                  </TouchableOpacity>
                ) : (
                  <Text style={[styles.speaker, { color }]}>{entry.speaker}</Text>
                )}
                <Text style={styles.text}>{entry.text}</Text>
              </View>
            </View>
          );
        })}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    flex: 1,
    paddingHorizontal: 12,
  },
  headerRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: 6,
    paddingBottom: 4,
    borderBottomWidth: 1,
    borderBottomColor: "#E5E7EB",
  },
  headerLeft: {
    fontSize: 12,
    fontWeight: "700",
    textTransform: "uppercase",
  },
  headerRight: {
    fontSize: 12,
    fontWeight: "700",
    textTransform: "uppercase",
    textAlign: "right",
  },
  headerHint: {
    fontSize: 10,
    fontWeight: "500",
    color: "#9CA3AF",
    textTransform: "none",
  },
  scroll: {
    flex: 1,
  },
  emptyContainer: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 40,
  },
  emptyText: {
    fontSize: 15,
    color: "#9CA3AF",
    fontStyle: "italic",
  },
  line: {
    flexDirection: "row",
    marginVertical: 4,
  },
  lineLeft: {
    justifyContent: "flex-start",
    paddingRight: "22%",
  },
  lineRight: {
    justifyContent: "flex-end",
    paddingLeft: "22%",
  },
  bubble: {
    borderRadius: 10,
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderWidth: 1,
    borderColor: "transparent",
    maxWidth: "100%",
  },
  bubbleLeft: {
    backgroundColor: "#F3F4F6",
  },
  bubbleRight: {
    backgroundColor: "#EEF2FF",
  },
  speaker: {
    fontSize: 11,
    fontWeight: "700",
    marginBottom: 2,
    textTransform: "uppercase",
  },
  text: {
    fontSize: 15,
    lineHeight: 21,
    color: "#1F2937",
  },
});
