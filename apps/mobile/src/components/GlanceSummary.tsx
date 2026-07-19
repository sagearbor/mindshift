import React from "react";
import { View, Text, StyleSheet } from "react-native";
import type { AnalyzePerSpeaker, AnalyzePerTurn } from "../api/client";
import type { SpeakerLabels } from "../utils/speakerLabels";
import {
  buildSpeakerBars,
  barPct,
  deriveVerdict,
  verdictColors,
} from "../screens/glanceSummary";

const INK = "#1F2937";
const MUTED = "#6B7280";
const TRACK = "#EEF1F5";

interface GlanceSummaryProps {
  perSpeaker: Record<string, AnalyzePerSpeaker>;
  perTurn: AnalyzePerTurn[];
  speakerLabels?: SpeakerLabels;
  /** Per-turn timing (index-aligned) when the conversation was timed, so the
   *  verdict can anchor a heated moment to its clock position. Absent for pasted
   *  transcripts — the verdict then simply omits the timestamp. */
  turnsTiming?: { start_time: number; end_time: number }[] | null;
}

/**
 * The glanceable summary — the headline act at the top of the analysis results.
 * A fresh-eyes user should read the outcome here in one look: a warm one-line
 * verdict chip, then per-speaker horizontal bars (average heat on the calm→red
 * ramp, talk share, and a plain tally of four-horsemen markers vs repair
 * attempts). No axes, big value labels on the bars. The honest, detailed
 * time-axis chart still sits below for anyone who wants it.
 *
 * Everything is derived from the analysis the server already returned — no
 * fabricated numbers, no zoomed axes. A narrow, calm conversation reads as calm
 * (green, low bars) rather than being blown up into false drama.
 */
export default function GlanceSummary({
  perSpeaker,
  perTurn,
  speakerLabels,
  turnsTiming,
}: GlanceSummaryProps) {
  const bars = buildSpeakerBars(perSpeaker, speakerLabels);
  const verdict = deriveVerdict(perTurn, turnsTiming);

  if (bars.length === 0) return null;

  return (
    <View style={styles.card} testID="glance-summary">
      <Text style={styles.title}>At a glance</Text>

      {verdict && (
        <View
          testID="glance-verdict"
          style={[
            styles.verdictChip,
            { backgroundColor: verdictColors(verdict.tone).bg },
          ]}
        >
          <Text
            style={[
              styles.verdictText,
              { color: verdictColors(verdict.tone).fg },
            ]}
          >
            {verdict.text}
          </Text>
        </View>
      )}

      {bars.map((b) => (
        <View key={b.id} style={styles.speakerBlock} testID={`glance-row-${b.id}`}>
          <View style={styles.speakerHeader}>
            <View style={[styles.swatch, { backgroundColor: b.color }]} />
            <Text style={styles.speakerName}>{b.label}</Text>
          </View>

          {/* Average heat — colored by the calm→amber→red ramp, value on the bar. */}
          <BarLine
            label="Heat"
            valueText={String(Math.round(b.avgHeat))}
            pct={barPct(b.avgHeat, 100)}
            color={b.heatBarColor}
            testID={`glance-heat-${b.id}`}
          />

          {/* Talk share — house blue, share as a percent. */}
          <BarLine
            label="Talk"
            valueText={`${Math.round(b.talkShare * 100)}%`}
            pct={barPct(b.talkShare, 1)}
            color={b.color}
            testID={`glance-talk-${b.id}`}
          />

          {/* Markers vs repairs — a plain, honest tally (no bar; these are counts,
              not a proportion). Repairs read as the warmer, hopeful number. */}
          <View style={styles.tallyRow} testID={`glance-tally-${b.id}`}>
            <Text style={styles.tallyItem}>
              <Text style={styles.tallyNumHarsh}>{b.horsemenTotal}</Text>
              <Text style={styles.tallyLabel}>
                {" "}
                harsh {b.horsemenTotal === 1 ? "moment" : "moments"}
              </Text>
            </Text>
            <Text style={styles.tallyDot}>·</Text>
            <Text style={styles.tallyItem}>
              <Text style={styles.tallyNumRepair}>{b.repairAttempts}</Text>
              <Text style={styles.tallyLabel}>
                {" "}
                repair {b.repairAttempts === 1 ? "attempt" : "attempts"}
              </Text>
            </Text>
          </View>
        </View>
      ))}
    </View>
  );
}

/** One labeled horizontal bar with the value printed on it. */
function BarLine({
  label,
  valueText,
  pct,
  color,
  testID,
}: {
  label: string;
  valueText: string;
  pct: number;
  color: string;
  testID: string;
}) {
  return (
    <View style={styles.barLine} testID={testID}>
      <Text style={styles.barLabel}>{label}</Text>
      <View style={styles.barTrack}>
        <View
          style={[
            styles.barFill,
            { width: `${pct}%`, backgroundColor: color },
          ]}
        />
        <Text style={styles.barValue}>{valueText}</Text>
      </View>
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
  title: {
    fontSize: 16,
    fontWeight: "700",
    color: INK,
    marginBottom: 12,
  },
  verdictChip: {
    alignSelf: "flex-start",
    borderRadius: 999,
    paddingHorizontal: 14,
    paddingVertical: 8,
    marginBottom: 16,
  },
  verdictText: {
    fontSize: 15,
    fontWeight: "700",
  },
  speakerBlock: {
    marginBottom: 16,
  },
  speakerHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 8,
  },
  swatch: { width: 12, height: 12, borderRadius: 3 },
  speakerName: { fontSize: 15, fontWeight: "700", color: INK },
  barLine: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
  },
  barLabel: {
    width: 44,
    fontSize: 12,
    fontWeight: "600",
    color: MUTED,
  },
  barTrack: {
    flex: 1,
    height: 24,
    borderRadius: 6,
    backgroundColor: TRACK,
    justifyContent: "center",
    overflow: "hidden",
  },
  barFill: {
    position: "absolute",
    left: 0,
    top: 0,
    bottom: 0,
    borderRadius: 6,
    // A minimum visible sliver so a zero/near-zero value still reads as a bar.
    minWidth: 3,
  },
  barValue: {
    marginLeft: 10,
    fontSize: 13,
    fontWeight: "800",
    color: INK,
  },
  tallyRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginTop: 2,
    paddingLeft: 44,
  },
  tallyItem: { fontSize: 13, color: MUTED },
  tallyNumHarsh: { fontWeight: "800", color: "#B25E09" },
  tallyNumRepair: { fontWeight: "800", color: "#1B7A4B" },
  tallyLabel: { color: MUTED },
  tallyDot: { color: "#D1D5DB", fontSize: 13 },
});
