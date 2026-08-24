import React, { useRef, useEffect, useMemo } from "react";
import { View, Text, ScrollView, StyleSheet } from "react-native";
import type { TranscriptEntry } from "../hooks/useAudioStream";
import { getSpeakerColor } from "../utils/speakerColors";

interface Props {
  entries: TranscriptEntry[];
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
export default function TherapistTranscript({ entries }: Props) {
  const scrollRef = useRef<ScrollView>(null);
  const columns = useMemo(() => columnOf(entries), [entries]);

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
        <Text
          style={[styles.headerLeft, { color: getSpeakerColor(columns.left ?? "") }]}
          testID="therapist-column-left"
        >
          {columns.left ?? "—"}
        </Text>
        <Text
          style={[styles.headerRight, { color: getSpeakerColor(columns.right ?? "") }]}
          testID="therapist-column-right"
        >
          {columns.right ?? "…"}
        </Text>
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
                <Text style={[styles.speaker, { color }]}>{entry.speaker}</Text>
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
