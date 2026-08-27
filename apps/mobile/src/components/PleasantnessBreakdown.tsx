import React, { useMemo } from "react";
import { View, Text, StyleSheet } from "react-native";
import type { AnalyzePerTurn } from "../api/client";
import {
  DIMENSIONS,
  WEIGHTS,
  roundHalfUp,
  type Dimension,
} from "../live/pleasantness";
import { speakerLabel, type SpeakerLabels } from "../utils/speakerLabels";

/**
 * The post-session pleasantness breakdown — the FIVE PRD §6 dimensions the
 * live on-device coach already shows (warmth, constructiveness, calmness,
 * respect, engagement), now rendered for a batch-analyzed conversation from
 * the SAME numbers the server computed via the ONE shared scorer
 * (server/pleasantness.py ↔ live/pleasantness.ts). This REPLACES the single
 * "heat" line as the headline of Conversation Dynamics — heat was measured to
 * be a weak signal (five dims explain only ~56% of it).
 *
 * Each speaker gets a stacked-contribution bar: the bar is divided into five
 * lanes whose WIDTHS are the dimension weights (30/25/20/15/10 %), and each
 * lane is FILLED from the left in proportion to that dimension's mean score
 * (so a lane's filled area = weight_i × dim_i, and the filled bands together
 * sum to the composite score out of 100). A dimension nobody could measure
 * (e.g. engagement with only one voice in the window) is HATCHED, never
 * silently redistributed or shown as 0 — the composite number beside the bar
 * is the honest renormalized mean over only the dimensions actually measured.
 */

// One fixed colour per dimension, tying the lane, the legend swatch, and the
// number together. Calm, distinct hues (not the speaker palette, which colours
// people — these colour the five qualities).
export const DIMENSION_COLORS: Record<Dimension, string> = {
  warmth: "#EC6A4E",
  constructiveness: "#4A90D9",
  calmness: "#10B981",
  respect: "#8B5CF6",
  engagement: "#F59E0B",
};

export const DIMENSION_LABELS: Record<Dimension, string> = {
  warmth: "Warmth",
  constructiveness: "Constructiveness",
  calmness: "Calmness",
  respect: "Respect",
  engagement: "Engagement",
};

export interface DimAggregate {
  dim: Dimension;
  /** Mean 0–100 over this speaker's turns that measured the dim; null when none. */
  mean: number | null;
  /** Raw PRD weight for this dim (fraction of the full bar's width). */
  weight: number;
  /** weight × mean/100 — the share of the full 0–100 bar this dim fills; null
   *  when unmeasured. The measured contributions sum to composite/100. */
  contribution: number | null;
}

export interface SpeakerBreakdown {
  speaker: string;
  dims: DimAggregate[]; // always in DIMENSIONS order, all five present
  /** Renormalized weighted mean over the measured dims (0–100); null when the
   *  speaker had no measured content dimension at all. */
  composite: number | null;
  /** Whether ANY of the five dimensions was measurable for this speaker. */
  measured: boolean;
}

function meanOf(values: number[]): number | null {
  if (values.length === 0) return null;
  return roundHalfUp(values.reduce((a, b) => a + b, 0) / values.length);
}

/**
 * Aggregate a batch analysis's per-turn dims into one breakdown per speaker
 * (first-appearance order), plus an "__overall" row across everyone. Pure —
 * exported for unit testing. Reads only `per_turn[].dims`; a turn/dim the
 * server left null contributes nothing (never a 0-guess).
 */
export function aggregateBreakdowns(perTurn: AnalyzePerTurn[]): {
  perSpeaker: SpeakerBreakdown[];
  overall: SpeakerBreakdown;
} {
  const order: string[] = [];
  const bySpeaker = new Map<string, Record<Dimension, number[]>>();
  const overall: Record<Dimension, number[]> = emptyBuckets();

  for (const pt of perTurn) {
    const dims = pt.dims;
    if (!dims) continue;
    const speaker = pt.speaker;
    if (!bySpeaker.has(speaker)) {
      bySpeaker.set(speaker, emptyBuckets());
      order.push(speaker);
    }
    const buckets = bySpeaker.get(speaker)!;
    for (const d of DIMENSIONS) {
      const v = dims[d];
      if (typeof v === "number" && Number.isFinite(v)) {
        buckets[d].push(v);
        overall[d].push(v);
      }
    }
  }

  const perSpeaker = order.map((sp) => buildBreakdown(sp, bySpeaker.get(sp)!));
  return { perSpeaker, overall: buildBreakdown("__overall", overall) };
}

function emptyBuckets(): Record<Dimension, number[]> {
  return {
    warmth: [],
    constructiveness: [],
    calmness: [],
    respect: [],
    engagement: [],
  };
}

function buildBreakdown(
  speaker: string,
  buckets: Record<Dimension, number[]>,
): SpeakerBreakdown {
  const dims: DimAggregate[] = DIMENSIONS.map((d) => {
    const mean = meanOf(buckets[d]);
    return {
      dim: d,
      mean,
      weight: WEIGHTS[d],
      contribution: mean === null ? null : (WEIGHTS[d] * mean) / 100,
    };
  });
  // Composite = renormalized weighted mean over the measured dims — the SAME
  // rule the scorer uses, so it matches the per-turn composite the server sent.
  let weighted = 0;
  let weightSum = 0;
  for (const a of dims) {
    if (a.mean === null) continue;
    weighted += a.weight * a.mean;
    weightSum += a.weight;
  }
  return {
    speaker,
    dims,
    composite: weightSum > 0 ? roundHalfUp(weighted / weightSum) : null,
    measured: weightSum > 0,
  };
}

function compositeColor(score: number | null): string {
  if (score === null) return "#9CA3AF";
  if (score >= 65) return "#10B981";
  if (score >= 40) return "#F59E0B";
  return "#EF4444";
}

/** A diagonal-stripe "not measured" fill — a recognizable hatch that can never
 *  be mistaken for a real value. */
function HatchFill({ testID }: { testID?: string }) {
  return (
    <View style={styles.hatch} testID={testID}>
      {[0, 1, 2, 3, 4, 5].map((i) => (
        <View key={i} style={[styles.hatchLine, { left: i * 8 - 6 }]} />
      ))}
    </View>
  );
}

/** One speaker's five-lane stacked-contribution bar. */
function BreakdownBar({ dims }: { dims: DimAggregate[] }) {
  return (
    <View style={styles.bar} testID="pleasantness-bar">
      {dims.map((a) => (
        <View
          key={a.dim}
          // Lane width is the dimension's weight — the five lanes tile the
          // whole bar (30/25/20/15/10 %).
          style={[styles.lane, { flexGrow: a.weight, flexShrink: a.weight, flexBasis: 0 }]}
          testID={`lane-${a.dim}`}
        >
          {a.mean === null ? (
            <HatchFill testID={`lane-${a.dim}-hatch`} />
          ) : (
            <View
              style={[
                styles.laneFill,
                { width: `${a.mean}%`, backgroundColor: DIMENSION_COLORS[a.dim] },
              ]}
              testID={`lane-${a.dim}-fill`}
            />
          )}
        </View>
      ))}
    </View>
  );
}

export interface PleasantnessBreakdownProps {
  perTurn: AnalyzePerTurn[];
  speakerLabels?: SpeakerLabels;
  testID?: string;
}

export default function PleasantnessBreakdown({
  perTurn,
  speakerLabels,
  testID = "pleasantness-breakdown",
}: PleasantnessBreakdownProps) {
  const { perSpeaker } = useMemo(() => aggregateBreakdowns(perTurn), [perTurn]);
  const anyMeasured = perSpeaker.some((s) => s.measured);

  return (
    <View style={styles.card} testID={testID}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>Pleasantness</Text>
        <Text style={styles.subtitle}>
          higher = warmer, calmer, more constructive
        </Text>
      </View>

      {!anyMeasured ? (
        <Text style={styles.empty} testID={`${testID}-empty`}>
          No per-turn tone was measured for this conversation yet — the five
          dimensions appear once the analysis reads each turn's tone.
        </Text>
      ) : null}

      {perSpeaker.map((s) => (
        <View
          key={s.speaker}
          style={styles.speakerRow}
          testID={`${testID}-row-${s.speaker}`}
        >
          <View style={styles.speakerHead}>
            <Text style={styles.speakerName} numberOfLines={1}>
              {speakerLabel(s.speaker, speakerLabels)}
            </Text>
            <Text
              style={[styles.composite, { color: compositeColor(s.composite) }]}
              testID={`${testID}-score-${s.speaker}`}
            >
              {s.composite === null ? "—" : String(s.composite)}
            </Text>
          </View>
          <BreakdownBar dims={s.dims} />
        </View>
      ))}

      {/* Legend — every dimension, with its colour + weight; a hatched swatch
          marks any dimension that no speaker could measure. */}
      <View style={styles.legend}>
        {DIMENSIONS.map((d) => {
          const everMeasured = perSpeaker.some(
            (s) => s.dims.find((a) => a.dim === d)?.mean !== null,
          );
          return (
            <View key={d} style={styles.legendItem} testID={`legend-${d}`}>
              {everMeasured ? (
                <View
                  style={[styles.legendSwatch, { backgroundColor: DIMENSION_COLORS[d] }]}
                />
              ) : (
                <View style={styles.legendSwatch}>
                  <HatchFill testID={`legend-${d}-hatch`} />
                </View>
              )}
              <Text style={styles.legendLabel}>
                {DIMENSION_LABELS[d]}{" "}
                <Text style={styles.legendWeight}>
                  {Math.round(WEIGHTS[d] * 100)}%
                </Text>
              </Text>
            </View>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    padding: 16,
    marginBottom: 16,
    gap: 10,
  },
  headerRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "baseline",
    justifyContent: "space-between",
    gap: 6,
  },
  title: { fontSize: 16, fontWeight: "700", color: "#1F2937" },
  subtitle: { fontSize: 11, color: "#6B7280" },
  empty: { fontSize: 12.5, color: "#6B7280", fontStyle: "italic" },
  speakerRow: { gap: 6 },
  speakerHead: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  speakerName: { fontSize: 14, fontWeight: "700", color: "#1F2937", flexShrink: 1 },
  composite: {
    fontSize: 22,
    fontWeight: "800",
    fontVariant: ["tabular-nums"],
    marginLeft: 8,
  },
  bar: {
    flexDirection: "row",
    height: 26,
    borderRadius: 6,
    overflow: "hidden",
    backgroundColor: "#F3F4F6",
  },
  lane: {
    height: "100%",
    backgroundColor: "#F3F4F6",
    borderRightWidth: 1,
    borderRightColor: "#FFFFFF",
    overflow: "hidden",
  },
  laneFill: { height: "100%" },
  hatch: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "#E5E7EB",
    overflow: "hidden",
  },
  hatchLine: {
    position: "absolute",
    top: -8,
    width: 2,
    height: 44,
    backgroundColor: "#C4C9D1",
    transform: [{ rotate: "20deg" }],
  },
  legend: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 2,
  },
  legendItem: { flexDirection: "row", alignItems: "center", gap: 5 },
  legendSwatch: {
    width: 12,
    height: 12,
    borderRadius: 3,
    overflow: "hidden",
    backgroundColor: "#E5E7EB",
  },
  legendLabel: { fontSize: 11.5, color: "#374151" },
  legendWeight: { fontSize: 11.5, color: "#9CA3AF" },
});
