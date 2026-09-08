import React, { useRef, useEffect } from "react";
import { useDevModeStore } from "../store/devModeStore";
import { View, Text, ScrollView, StyleSheet, TouchableOpacity } from "react-native";
import type { TranscriptEntry } from "../hooks/useAudioStream";
// getSpeakerColor now lives in a shared util so the HeatChart keys speakers to
// the same hues as the live transcript. Behavior here is unchanged.
import { getSpeakerColor } from "../utils/speakerColors";

interface LiveTranscriptProps {
  entries: TranscriptEntry[];
  /** Mid-call naming: tapping a speaker label opens "Who is this?" for that
   *  voice. Absent → labels are plain text, exactly as before. */
  onSpeakerPress?: (entry: TranscriptEntry) => void;
  /** Whether the entry's speaker already has a name (then no "who?" hint —
   *  the label stays tappable to correct it). Default: unnamed. */
  isNamed?: (entry: TranscriptEntry) => boolean;
}

export default function LiveTranscript({ entries, onSpeakerPress, isNamed }: LiveTranscriptProps) {
  const scrollRef = useRef<ScrollView>(null);
  const devMode = useDevModeStore((s) => s.devMode);

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
            {onSpeakerPress ? (
              <TouchableOpacity
                testID={`live-transcript-speaker-${i}`}
                accessibilityRole="button"
                accessibilityLabel={`Who is ${entry.speaker}?`}
                onPress={() => onSpeakerPress(entry)}
                hitSlop={{ top: 6, bottom: 6, left: 6, right: 6 }}
              >
                <Text style={[styles.speaker, { color }]}>
                  {entry.speaker}
                  {isNamed?.(entry) ? null : <Text style={styles.speakerHint}> · who?</Text>}
                </Text>
              </TouchableOpacity>
            ) : (
              <Text style={[styles.speaker, { color }]}>
                {entry.speaker}
              </Text>
            )}
            <Text style={[styles.text, isLatest && styles.latestText]}>
              {entry.text}
              {devMode && entry.activation !== undefined ? (
                // Developer mode: vocal-activation probability of the user's
                // own line (live/activation.ts, dark classifier).
                <Text style={styles.devTag} testID={`live-transcript-activation-${i}`}>
                  {`  ⚡${Math.round(entry.activation * 100)}%`}
                </Text>
              ) : null}
              {devMode && entry.overlapSeconds !== undefined ? (
                // Developer mode: longest mixed-voice run the single-mic
                // overlap probe saw in this turn (live/overlapProbe.ts, dark).
                <Text style={styles.devTag} testID={`live-transcript-overlap-${i}`}>
                  {`  ⟂${entry.overlapSeconds.toFixed(1)}s`}
                </Text>
              ) : null}
            </Text>
          </View>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  devTag: { fontSize: 11, color: "#9CA3AF" },
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
  speakerHint: {
    fontSize: 10,
    fontWeight: "500",
    color: "#9CA3AF",
    textTransform: "none",
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
