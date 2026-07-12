import React, { useState } from "react";
import { View, Text, StyleSheet, TouchableOpacity } from "react-native";
import Svg, { Polyline, Circle } from "react-native-svg";
import type { AnalyzePerTurn } from "../api/client";
import { getSpeakerColor } from "../utils/speakerColors";

// House colors.
const AMBER = "#F59E0B"; // spikes / triggers
const INK = "#1F2937";
const MUTED = "#6B7280";

const HEAT_MIN = 0;
const HEAT_MAX = 100;

export interface ChartPoint {
  index: number; // turn index across the WHOLE conversation
  heat: number; // 0–100
  isSpike: boolean;
  x: number; // pixel x
  y: number; // pixel y
}

export interface SpeakerLine {
  speaker: string;
  color: string;
  points: ChartPoint[]; // that speaker's turns only, in conversation order
}

export interface MapOptions {
  width: number;
  height: number;
  padding: number;
  /** Total turns in the conversation, used to normalize x. When omitted, it's
   *  derived from the max turn index in `perTurn`. */
  totalTurns?: number;
}

/**
 * Pure point-mapping: turns the backend's flat per_turn array into one polyline
 * per speaker, in pixel space. Exported so the geometry can be unit-tested
 * directly without rendering.
 *
 * Key rule (from the spec): x is the turn index across the ENTIRE conversation,
 * not each speaker's own running count — so a speaker's line "carries across
 * gaps", i.e. its points sit at their true conversational positions and we only
 * connect *consecutive* points belonging to that speaker. This is what makes
 * two speakers' lines visibly interleave and cross as the conversation heats up.
 */
export function mapTurnsToLines(
  perTurn: AnalyzePerTurn[],
  opts: MapOptions,
): SpeakerLine[] {
  const { width, height, padding } = opts;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;

  // Normalize x against the last turn index so the line spans the full width.
  const maxIndex =
    opts.totalTurns !== undefined
      ? opts.totalTurns - 1
      : perTurn.reduce((m, t) => Math.max(m, t.index), 0);

  const xFor = (index: number) =>
    padding + (maxIndex <= 0 ? chartWidth / 2 : (index / maxIndex) * chartWidth);
  const yFor = (heat: number) => {
    const clamped = Math.max(HEAT_MIN, Math.min(HEAT_MAX, heat));
    return (
      padding +
      (chartHeight -
        ((clamped - HEAT_MIN) / (HEAT_MAX - HEAT_MIN)) * chartHeight)
    );
  };

  // Group by speaker, preserving first-seen order for a stable legend/z-order.
  const order: string[] = [];
  const bySpeaker = new Map<string, ChartPoint[]>();
  for (const t of perTurn) {
    if (!bySpeaker.has(t.speaker)) {
      bySpeaker.set(t.speaker, []);
      order.push(t.speaker);
    }
    bySpeaker.get(t.speaker)!.push({
      index: t.index,
      heat: t.heat,
      isSpike: t.is_spike,
      x: xFor(t.index),
      y: yFor(t.heat),
    });
  }

  return order.map((speaker) => ({
    speaker,
    color: getSpeakerColor(speaker),
    points: bySpeaker.get(speaker)!,
  }));
}

interface HeatChartProps {
  perTurn: AnalyzePerTurn[];
  // The original transcript, index-aligned with perTurn. The backend's per_turn
  // carries no text (only heat/markers), so the inspector resolves each turn's
  // words from here by index. Optional so the chart still renders without it.
  turns?: { speaker: string; text: string }[];
  height?: number;
}

/**
 * Per-speaker heat over the course of the conversation, modeled on
 * ToneSparkline (Svg + Polyline + Circle). One line per speaker, x = turn index
 * across the whole conversation, y = heat 0–100. Spike turns get a larger amber
 * dot. Tapping any point selects that turn and reveals a "turn inspector" card
 * below the chart (speaker, text, heat, markers, trigger phrase).
 *
 * Width is measured via onLayout so the chart is full-width responsive.
 */
export default function HeatChart({
  perTurn,
  turns,
  height = 180,
}: HeatChartProps) {
  const [width, setWidth] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);

  const padding = 16;
  // Only compute geometry once we've measured a width (first layout pass).
  const lines =
    width > 0 ? mapTurnsToLines(perTurn, { width, height, padding }) : [];

  const selectedTurn =
    selected !== null
      ? perTurn.find((t) => t.index === selected) ?? null
      : null;

  return (
    <View testID="heat-chart">
      {/* Legend: color swatch + speaker name, one row. */}
      <View style={styles.legend}>
        {lines.map((line) => (
          <View key={line.speaker} style={styles.legendItem}>
            <View
              style={[styles.swatch, { backgroundColor: line.color }]}
              testID={`legend-swatch-${line.speaker}`}
            />
            <Text style={styles.legendText}>{line.speaker}</Text>
          </View>
        ))}
      </View>

      {/* Chart surface — onLayout gives us the responsive width. */}
      <View
        style={{ height }}
        onLayout={(e) => setWidth(e.nativeEvent.layout.width)}
      >
        {width > 0 && (
          <Svg width={width} height={height}>
            {lines.map((line) => (
              <Polyline
                key={`line-${line.speaker}`}
                testID={`heat-line-${line.speaker}`}
                points={line.points.map((p) => `${p.x},${p.y}`).join(" ")}
                fill="none"
                stroke={line.color}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            ))}
            {/* Points last so they sit above the lines. Spikes are larger and
                amber; the selected point gets a ring. */}
            {lines.flatMap((line) =>
              line.points.map((p) => (
                <Circle
                  key={`pt-${line.speaker}-${p.index}`}
                  testID={
                    p.isSpike
                      ? `heat-spike-${p.index}`
                      : `heat-point-${p.index}`
                  }
                  cx={p.x}
                  cy={p.y}
                  r={p.isSpike ? 6 : 4}
                  fill={p.isSpike ? AMBER : line.color}
                  stroke={selected === p.index ? INK : "none"}
                  strokeWidth={selected === p.index ? 2 : 0}
                  onPress={() => setSelected(p.index)}
                />
              )),
            )}
          </Svg>
        )}
      </View>

      {/* Tap targets: SVG circles are tiny/unreliable to hit, so we also render
          a row of invisible full-height touch columns, one per turn, as a
          reliable scrubber. Tapping a column selects that turn. */}
      <View style={styles.scrubberRow} testID="heat-scrubber">
        {perTurn.map((t) => (
          <TouchableOpacity
            key={`scrub-${t.index}`}
            testID={`scrub-${t.index}`}
            style={styles.scrubCell}
            onPress={() => setSelected(t.index)}
            accessibilityLabel={`Turn ${t.index + 1}, ${t.speaker}, heat ${t.heat}`}
          />
        ))}
      </View>

      {/* Turn inspector — shows the selected turn's detail. */}
      {selectedTurn && (
        <View style={styles.inspector} testID="turn-inspector">
          <View style={styles.inspectorHeader}>
            <Text
              style={[
                styles.inspectorSpeaker,
                { color: getSpeakerColor(selectedTurn.speaker) },
              ]}
            >
              {selectedTurn.speaker}
            </Text>
            <Text style={styles.inspectorHeat}>heat {selectedTurn.heat}</Text>
          </View>
          <Text style={styles.inspectorText}>
            {turns?.[selectedTurn.index]?.text ?? ""}
          </Text>
          {selectedTurn.markers.length > 0 && (
            <View style={styles.chipRow}>
              {selectedTurn.markers.map((m) => (
                <View key={m} style={styles.chip}>
                  <Text style={styles.chipText}>{m.replace(/_/g, " ")}</Text>
                </View>
              ))}
            </View>
          )}
          {selectedTurn.trigger_phrase && (
            <Text style={styles.trigger} testID="turn-inspector-trigger">
              Trigger: “{selectedTurn.trigger_phrase}”
            </Text>
          )}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  legend: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 14,
    marginBottom: 8,
  },
  legendItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  swatch: {
    width: 12,
    height: 12,
    borderRadius: 3,
  },
  legendText: {
    fontSize: 13,
    color: MUTED,
    fontWeight: "600",
  },
  scrubberRow: {
    flexDirection: "row",
    marginTop: 6,
  },
  scrubCell: {
    flex: 1,
    height: 24,
  },
  inspector: {
    marginTop: 10,
    padding: 12,
    borderRadius: 10,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  inspectorHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 4,
  },
  inspectorSpeaker: {
    fontSize: 13,
    fontWeight: "700",
    textTransform: "uppercase",
  },
  inspectorHeat: {
    fontSize: 13,
    fontWeight: "600",
    color: AMBER,
  },
  inspectorText: {
    fontSize: 15,
    lineHeight: 21,
    color: INK,
  },
  chipRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
    marginTop: 8,
  },
  chip: {
    backgroundColor: "#F3F4F6",
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  chipText: {
    fontSize: 11,
    color: MUTED,
    fontWeight: "600",
  },
  trigger: {
    marginTop: 8,
    fontSize: 13,
    color: AMBER,
    fontStyle: "italic",
  },
});
