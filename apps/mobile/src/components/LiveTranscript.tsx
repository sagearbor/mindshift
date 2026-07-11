import React, { useRef, useEffect } from "react";
import { View, Text, ScrollView, StyleSheet } from "react-native";
import type { TranscriptEntry } from "../hooks/useAudioStream";
// getSpeakerColor now lives in a shared util so the HeatChart keys speakers to
// the same hues as the live transcript. Behavior here is unchanged.
import { getSpeakerColor } from "../utils/speakerColors";

interface LiveTranscriptProps {
  entries: TranscriptEntry[];
}

export default function LiveTranscript({ entries }: LiveTranscriptProps) {
  const scrollRef = useRef<ScrollView>(null);

  useEffect(() => {
    if (scrollRef.current && entries.length > 0) {
      scrollRef.current.scrollToEnd({ animated: true });
    }
  }, [entries.length]);

  if (entries.length === 0) {
    return (
      <View style={styles.emptyContainer} testID="live-transcript-empty">
        <Text style={styles.emptyText}>
          Waiting for conversation...
        </Text>
      </View>
    );
  }

  return (
    <ScrollView
      ref={scrollRef}
      style={styles.container}
      testID="live-transcript"
    >
      {entries.map((entry, i) => {
        const color = getSpeakerColor(entry.speaker);
        const isLatest = i === entries.length - 1;
        return (
          <View
            key={i}
            style={[styles.entry, isLatest && styles.latestEntry]}
          >
            <Text style={[styles.speaker, { color }]}>
              {entry.speaker}
            </Text>
            <Text style={[styles.text, isLatest && styles.latestText]}>
              {entry.text}
            </Text>
          </View>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 16,
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
  entry: {
    marginBottom: 10,
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 8,
    backgroundColor: "#F9FAFB",
  },
  latestEntry: {
    backgroundColor: "#EEF2FF",
    borderLeftWidth: 3,
    borderLeftColor: "#4A90D9",
  },
  speaker: {
    fontSize: 12,
    fontWeight: "700",
    marginBottom: 2,
    textTransform: "uppercase",
  },
  text: {
    fontSize: 15,
    lineHeight: 22,
    color: "#1F2937",
  },
  latestText: {
    fontWeight: "500",
  },
});
