import React, { useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import type { WordMetrics, WordMetricsSpeaker } from "../api/client";
import { getSpeakerColor } from "../utils/speakerColors";
import { speakerLabel, type SpeakerLabels } from "../utils/speakerLabels";
import { barPct } from "../screens/glanceSummary";

const INK = "#1F2937";
const MUTED = "#6B7280";
const TRACK = "#EEF1F5";

/** The five emotion densities, in a stable render order, each with a plain label
 *  and its dot/bar color. */
const EMOTIONS: {
  key: keyof Pick<
    WordMetricsSpeaker,
    "anger_rate" | "fear_rate" | "sadness_rate" | "joy_rate" | "trust_rate"
  >;
  label: string;
  color: string;
}[] = [
  { key: "joy_rate", label: "joy", color: "#F59E0B" },
  { key: "trust_rate", label: "trust", color: "#10B981" },
  { key: "sadness_rate", label: "sadness", color: "#4A90D9" },
  { key: "fear_rate", label: "fear", color: "#8B5CF6" },
  { key: "anger_rate", label: "anger", color: "#DC2626" },
];

/**
 * Largest finite rate across the given accessors and speakers — the shared scale
 * for a group of bars so the LONGEST bar fills the track and the rest read
 * proportionally against it. Returns 0 when nothing is measurable (all null),
 * which yields empty bars rather than a fabricated scale.
 */
export function maxRate(
  speakers: WordMetricsSpeaker[],
  keys: (keyof WordMetricsSpeaker)[],
): number {
  let m = 0;
  for (const s of speakers) {
    for (const k of keys) {
      const v = s[k];
      if (typeof v === "number" && Number.isFinite(v) && v > m) m = v;
    }
  }
  return m;
}

/**
 * Method lines for the "How is this counted?" expando, rendered verbatim. Each
 * entry of the server's `method` map becomes one {key, value} line; non-string
 * values are stringified defensively so an unexpected shape still shows
 * something honest rather than crashing.
 */
export function methodLines(
  method: Record<string, unknown>,
): { key: string; value: string }[] {
  return Object.entries(method).map(([key, value]) => ({
    key,
    value: typeof value === "string" ? value : JSON.stringify(value),
  }));
}

interface WordPatternsPanelProps {
  wordMetrics?: WordMetrics;
  speakerLabels?: SpeakerLabels;
}

/**
 * "📖 Word patterns" — the transparent, hand-checkable metrics panel. Collapsed
 * by default (this is the audit layer, not the headline). Renders defensively:
 * when `wordMetrics` is absent (old server / old analysis) the whole panel is
 * hidden, so nothing regresses.
 *
 * Inside: the star metric is the I-vs-you pronoun focus per speaker, in plain
 * language ("Talks about own feelings" vs "Points at the other person") with the
 * technical name in small print; then compact emotion-word density bars; then an
 * expandable "How is this counted?" that shows the server's `method` verbatim —
 * the whole point being that a therapist can check the numbers by hand.
 */
export default function WordPatternsPanel({
  wordMetrics,
  speakerLabels,
}: WordPatternsPanelProps) {
  const [open, setOpen] = useState(false);
  const [methodOpen, setMethodOpen] = useState(false);

  // Defensive: absent on old servers — hide the whole panel.
  if (!wordMetrics || Object.keys(wordMetrics.speakers).length === 0) {
    return null;
  }

  const entries = Object.entries(wordMetrics.speakers);
  const speakerStats = entries.map(([, s]) => s);
  const pronounMax = maxRate(speakerStats, ["i_rate", "you_rate", "we_rate"]);
  const emotionMax = maxRate(
    speakerStats,
    EMOTIONS.map((e) => e.key),
  );
  const method = methodLines(wordMetrics.method);

  return (
    <View style={styles.card} testID="word-patterns-panel">
      <TouchableOpacity
        testID="word-patterns-toggle"
        style={styles.headerRow}
        onPress={() => setOpen((v) => !v)}
        accessibilityRole="button"
      >
        <Text style={styles.title}>📖 Word patterns</Text>
        <Text style={styles.chevron}>{open ? "▲" : "▼"}</Text>
      </TouchableOpacity>

      {!open ? (
        <Text style={styles.subtitle}>
          The plain word counts behind the read — tap to open.
        </Text>
      ) : (
        <View style={styles.body} testID="word-patterns-body">
          {entries.map(([id, s]) => {
            const label = speakerLabel(id, speakerLabels);
            const color = getSpeakerColor(id);
            const lowSample =
              s.low_sample === true || s.i_rate === null;
            return (
              <View
                key={id}
                style={styles.speakerBlock}
                testID={`word-patterns-speaker-${id}`}
              >
                <View style={styles.speakerHeader}>
                  <View style={[styles.swatch, { backgroundColor: color }]} />
                  <Text style={styles.speakerName}>{label}</Text>
                  <Text style={styles.wordCount}>{s.word_count} words</Text>
                </View>

                {lowSample ? (
                  <Text
                    style={styles.lowSample}
                    testID={`word-patterns-lowsample-${id}`}
                  >
                    Not enough words here to count patterns honestly.
                  </Text>
                ) : (
                  <>
                    {/* Star metric: I-vs-you focus, plain language up front. */}
                    <MetricBar
                      plain="Talks about own feelings"
                      technical="I-statements / 100 words"
                      rate={s.i_rate}
                      max={pronounMax}
                      color={color}
                      testID={`word-patterns-i-${id}`}
                    />
                    <MetricBar
                      plain="Points at the other person"
                      technical="you-statements / 100 words"
                      rate={s.you_rate}
                      max={pronounMax}
                      color="#B25E09"
                      testID={`word-patterns-you-${id}`}
                    />
                    <MetricBar
                      plain="Talks about us together"
                      technical="we-statements / 100 words"
                      rate={s.we_rate}
                      max={pronounMax}
                      color="#1B7A4B"
                      testID={`word-patterns-we-${id}`}
                    />

                    {/* Emotion-word densities: compact colored bars. */}
                    <Text style={styles.emotionHeading}>
                      Emotion words / 100 words
                    </Text>
                    <View style={styles.emotionRow}>
                      {EMOTIONS.map((e) => (
                        <EmotionDot
                          key={e.key}
                          label={e.label}
                          rate={s[e.key]}
                          max={emotionMax}
                          color={e.color}
                          testID={`word-patterns-emotion-${e.label}-${id}`}
                        />
                      ))}
                    </View>
                  </>
                )}
              </View>
            );
          })}

          {/* Auditability: the server's own method text, verbatim. */}
          {method.length > 0 && (
            <View style={styles.methodBlock}>
              <TouchableOpacity
                testID="word-patterns-method-toggle"
                onPress={() => setMethodOpen((v) => !v)}
                accessibilityRole="button"
              >
                <Text style={styles.methodToggle}>
                  {methodOpen ? "Hide" : "How is this counted?"}
                </Text>
              </TouchableOpacity>
              {methodOpen && (
                <View style={styles.methodBody} testID="word-patterns-method">
                  {method.map((m) => (
                    <Text key={m.key} style={styles.methodLine}>
                      <Text style={styles.methodKey}>{m.key}: </Text>
                      {m.value}
                    </Text>
                  ))}
                </View>
              )}
            </View>
          )}
        </View>
      )}
    </View>
  );
}

/** One plain-language labeled bar for a pronoun-focus rate. A null rate renders
 *  a muted "—" rather than a zero bar. */
function MetricBar({
  plain,
  technical,
  rate,
  max,
  color,
  testID,
}: {
  plain: string;
  technical: string;
  rate: number | null;
  max: number;
  color: string;
  testID: string;
}) {
  return (
    <View style={styles.metricBar} testID={testID}>
      <View style={styles.metricLabelWrap}>
        <Text style={styles.metricPlain}>{plain}</Text>
        <Text style={styles.metricTechnical}>{technical}</Text>
      </View>
      <View style={styles.metricTrack}>
        {rate !== null && (
          <View
            style={[
              styles.metricFill,
              { width: `${barPct(rate, max)}%`, backgroundColor: color },
            ]}
          />
        )}
        <Text style={styles.metricValue}>
          {rate === null ? "—" : rate.toFixed(1)}
        </Text>
      </View>
    </View>
  );
}

/** A compact emotion-density chip: a colored bar scaled to the panel's max, the
 *  rate printed, and the emotion name. Null rate → a muted dash. */
function EmotionDot({
  label,
  rate,
  max,
  color,
  testID,
}: {
  label: string;
  rate: number | null;
  max: number;
  color: string;
  testID: string;
}) {
  return (
    <View style={styles.emotionItem} testID={testID}>
      <View style={styles.emotionBarTrack}>
        {rate !== null && (
          <View
            style={[
              styles.emotionBarFill,
              { height: `${barPct(rate, max)}%`, backgroundColor: color },
            ]}
          />
        )}
      </View>
      <Text style={styles.emotionValue}>
        {rate === null ? "—" : rate.toFixed(1)}
      </Text>
      <Text style={styles.emotionLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  title: { fontSize: 16, fontWeight: "700", color: INK },
  chevron: { fontSize: 12, color: MUTED },
  subtitle: {
    fontSize: 13,
    color: MUTED,
    marginTop: 6,
  },
  body: { marginTop: 12 },
  speakerBlock: {
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#F3F4F6",
  },
  speakerHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 10,
  },
  swatch: { width: 12, height: 12, borderRadius: 3 },
  speakerName: { fontSize: 15, fontWeight: "700", color: INK, flex: 1 },
  wordCount: { fontSize: 12, color: MUTED },
  lowSample: {
    fontSize: 13,
    color: MUTED,
    fontStyle: "italic",
  },
  metricBar: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
  },
  metricLabelWrap: { width: 150, paddingRight: 8 },
  metricPlain: { fontSize: 13, fontWeight: "600", color: INK },
  metricTechnical: { fontSize: 10, color: "#9CA3AF" },
  metricTrack: {
    flex: 1,
    height: 22,
    borderRadius: 6,
    backgroundColor: TRACK,
    justifyContent: "center",
    overflow: "hidden",
  },
  metricFill: {
    position: "absolute",
    left: 0,
    top: 0,
    bottom: 0,
    borderRadius: 6,
    minWidth: 3,
  },
  metricValue: {
    marginLeft: 8,
    fontSize: 12,
    fontWeight: "800",
    color: INK,
  },
  emotionHeading: {
    fontSize: 11,
    fontWeight: "700",
    color: MUTED,
    textTransform: "uppercase",
    marginTop: 6,
    marginBottom: 8,
  },
  emotionRow: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  emotionItem: {
    alignItems: "center",
    flex: 1,
  },
  emotionBarTrack: {
    width: 18,
    height: 40,
    borderRadius: 5,
    backgroundColor: TRACK,
    justifyContent: "flex-end",
    overflow: "hidden",
  },
  emotionBarFill: {
    width: "100%",
    borderRadius: 5,
    minHeight: 2,
  },
  emotionValue: {
    fontSize: 11,
    fontWeight: "800",
    color: INK,
    marginTop: 4,
  },
  emotionLabel: { fontSize: 10, color: MUTED },
  methodBlock: { marginTop: 4 },
  methodToggle: {
    fontSize: 13,
    fontWeight: "700",
    color: "#4A90D9",
  },
  methodBody: {
    marginTop: 8,
    padding: 12,
    backgroundColor: "#F9FAFB",
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  methodLine: {
    fontSize: 12.5,
    lineHeight: 18,
    color: "#374151",
    marginBottom: 6,
  },
  methodKey: { fontWeight: "700", color: INK },
});
