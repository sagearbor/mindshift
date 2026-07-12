import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
} from "react-native";
import Svg, { Polyline, Circle } from "react-native-svg";
import type { AnalyzePerTurn, SimulatedTurn, Voice } from "../api/client";
import { getSpeakerColor } from "../utils/speakerColors";

// The baseline prosody label per dimension — a turn at baseline on a dimension
// isn't noteworthy, so we don't render a chip for it. This keeps the inspector
// to "up to three" chips that actually say something (e.g. loud + fast), rather
// than three always-on chips two of which read "normal".
const VOICE_BASELINE: Record<keyof Voice, string> = {
  energy_label: "normal",
  pitch_label: "mid",
  rate_label: "normal",
};

/** Non-baseline prosody labels for a turn, as {kind,label} chips in a stable
 *  order (energy, pitch, rate). Empty when voice is absent or all-baseline. */
export function voiceChipsFor(voice: Voice | null | undefined): {
  kind: "energy" | "pitch" | "rate";
  label: string;
}[] {
  if (!voice) return [];
  const chips: { kind: "energy" | "pitch" | "rate"; label: string }[] = [];
  if (voice.energy_label !== VOICE_BASELINE.energy_label)
    chips.push({ kind: "energy", label: voice.energy_label });
  // pitch_label is null when the turn had too little voiced speech to measure
  // — no reading means no chip, never an empty one.
  if (voice.pitch_label !== null && voice.pitch_label !== VOICE_BASELINE.pitch_label)
    chips.push({ kind: "pitch", label: voice.pitch_label });
  if (voice.rate_label !== VOICE_BASELINE.rate_label)
    chips.push({ kind: "rate", label: voice.rate_label });
  return chips;
}

// House colors.
const AMBER = "#F59E0B"; // spikes / triggers
const INK = "#1F2937";
const MUTED = "#6B7280";
const PRIMARY = "#4A90D9";
const DANGER = "#DC2626";

// Simulated overlay: each speaker's own color, drawn dashed at reduced opacity
// so it reads as a hypothetical laid over the real (solid, full-opacity) lines.
const SIM_OPACITY = 0.55;
const SIM_DASH = "6,4";

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

/**
 * Pure geometry for the "what-if" simulated overlay. Mirrors mapTurnsToLines
 * exactly for x/y so a simulated point at conversation index `i` lands at the
 * SAME x as the real point at index `i` — the caller passes `totalTurns` (the
 * real conversation's length) so both share one x-scale, and the dashed overlay
 * (which only spans pivot_index → last turn) aligns perfectly with the solid
 * lines beneath it. Grouped per speaker so each gets a dashed line in its own
 * color. Simulated points carry no spike/marker data, so isSpike is always
 * false. Exported for direct unit testing of the alignment + grouping.
 */
export function mapSimulatedToLines(
  simulated: SimulatedTurn[],
  opts: MapOptions,
): SpeakerLine[] {
  const { width, height, padding } = opts;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;

  const maxIndex =
    opts.totalTurns !== undefined
      ? opts.totalTurns - 1
      : simulated.reduce((m, t) => Math.max(m, t.index), 0);

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

  const order: string[] = [];
  const bySpeaker = new Map<string, ChartPoint[]>();
  for (const t of simulated) {
    if (!bySpeaker.has(t.speaker)) {
      bySpeaker.set(t.speaker, []);
      order.push(t.speaker);
    }
    bySpeaker.get(t.speaker)!.push({
      index: t.index,
      heat: t.heat,
      isSpike: false,
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

  // --- "What if" simulated overlay (all optional; the chart is fully usable
  // without any of these) ---
  /** Simulated per-turn heat (pivot → last turn) to overlay as dashed lines.
   *  Null/undefined = no simulation available. */
  simulated?: SimulatedTurn[] | null;
  /** Whether the overlay is currently shown (toggle state owned by the parent).
   *  When false the dashed lines + "simulated" legend entry hide, without any
   *  refetch. */
  showSimulation?: boolean;
  /** Toggle the overlay on/off (the chip near the legend). */
  onToggleSimulation?: () => void;
  /** Fire a counterfactual for the currently selected turn (the pivot). */
  onWhatIf?: (pivotIndex: number) => void;
  /** True while a counterfactual request is in flight (button spinner). */
  whatIfLoading?: boolean;
  /** The turn index the in-flight/errored request pertains to, so loading and
   *  error states only attach to the inspector for that turn. */
  whatIfPivotIndex?: number | null;
  /** Honest inline error from the last counterfactual attempt (null = none). */
  whatIfError?: string | null;
  /** Retry the last failed counterfactual. */
  onRetryWhatIf?: () => void;
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
  simulated,
  showSimulation = true,
  onToggleSimulation,
  onWhatIf,
  whatIfLoading = false,
  whatIfPivotIndex = null,
  whatIfError = null,
  onRetryWhatIf,
}: HeatChartProps) {
  const [width, setWidth] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);

  const padding = 16;
  // Only compute geometry once we've measured a width (first layout pass).
  // totalTurns is passed EXPLICITLY so the real and simulated lines share one
  // x-scale by construction, not by the (currently true) coincidence that
  // server turn indexes are contiguous 0..n-1.
  const lines =
    width > 0
      ? mapTurnsToLines(perTurn, {
          width,
          height,
          padding,
          totalTurns: perTurn.length,
        })
      : [];

  // Simulated overlay lines share the SAME x-scale as the real lines by pinning
  // totalTurns to the real conversation length — so the dashed segment lands
  // exactly over the solid one from the pivot onward.
  const overlayActive = !!simulated && simulated.length > 0 && showSimulation;
  const simLines =
    overlayActive && width > 0
      ? mapSimulatedToLines(simulated!, {
          width,
          height,
          padding,
          totalTurns: perTurn.length,
        })
      : [];

  const selectedTurn =
    selected !== null
      ? perTurn.find((t) => t.index === selected) ?? null
      : null;

  // Non-baseline prosody chips for the selected turn (empty when no voice data).
  const voiceChips = voiceChipsFor(selectedTurn?.voice);

  // Loading/error only belong to the inspector when they pertain to the turn
  // currently selected (the pivot the parent is acting on).
  const isPivotSelected = selected !== null && selected === whatIfPivotIndex;
  const showWhatIfLoading = whatIfLoading && isPivotSelected;
  const showWhatIfError = !!whatIfError && isPivotSelected && !whatIfLoading;

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

        {/* Dashed-line legend entry, only while the overlay is visible. */}
        {overlayActive && (
          <View style={styles.legendItem} testID="legend-simulated">
            <View style={styles.dashSwatch}>
              <View style={styles.dashSeg} />
              <View style={styles.dashSeg} />
            </View>
            <Text style={styles.legendText}>simulated</Text>
          </View>
        )}

        {/* Toggle chip: show/hide the overlay without refetching. Present
            whenever a simulation exists (even when currently hidden). */}
        {!!simulated && simulated.length > 0 && (
          <TouchableOpacity
            testID="simulation-toggle"
            style={styles.simToggle}
            onPress={onToggleSimulation}
          >
            <Text style={styles.simToggleText}>
              {showSimulation ? "Simulation ✕" : "Show simulation"}
            </Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Chart surface — onLayout gives us the responsive width. */}
      <View
        style={{ height }}
        onLayout={(e) => setWidth(e.nativeEvent.layout.width)}
      >
        {width > 0 && (
          <Svg width={width} height={height}>
            {/* Simulated overlay FIRST so the real (solid) lines sit on top of
                the dashed hypothetical. Same per-speaker color, dashed, reduced
                opacity. */}
            {simLines.map((line) => (
              <Polyline
                key={`sim-line-${line.speaker}`}
                testID={`sim-line-${line.speaker}`}
                points={line.points.map((p) => `${p.x},${p.y}`).join(" ")}
                fill="none"
                stroke={line.color}
                strokeWidth={2}
                strokeDasharray={SIM_DASH}
                strokeOpacity={SIM_OPACITY}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            ))}
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
          {(selectedTurn.markers.length > 0 || voiceChips.length > 0) && (
            <View style={styles.chipRow}>
              {/* Behavioral markers: filled chips. */}
              {selectedTurn.markers.map((m) => (
                <View key={m} style={styles.chip}>
                  <Text style={styles.chipText}>{m.replace(/_/g, " ")}</Text>
                </View>
              ))}
              {/* Voice/prosody: outline chips, visually distinct from the filled
                  marker chips. Only the non-baseline dimensions appear; nothing
                  when voice is null/absent (old servers / degraded prosody). */}
              {voiceChips.map((c) => (
                <View
                  key={`voice-${c.kind}`}
                  testID={`voice-chip-${c.kind}`}
                  style={styles.voiceChip}
                >
                  <Text style={styles.voiceChipText}>{c.label}</Text>
                </View>
              ))}
            </View>
          )}
          {selectedTurn.trigger_phrase && (
            <Text style={styles.trigger} testID="turn-inspector-trigger">
              Trigger: “{selectedTurn.trigger_phrase}”
            </Text>
          )}

          {/* "What if this was said differently?" — the counterfactual entry
              point. Only shown when the parent wired up onWhatIf. */}
          {onWhatIf && (
            <View style={styles.whatIfBlock}>
              <TouchableOpacity
                testID="what-if-button"
                style={[
                  styles.whatIfButton,
                  showWhatIfLoading && styles.whatIfButtonLoading,
                ]}
                disabled={showWhatIfLoading}
                onPress={() => onWhatIf(selectedTurn.index)}
              >
                {showWhatIfLoading ? (
                  <View style={styles.whatIfLoadingRow}>
                    <ActivityIndicator size="small" color={PRIMARY} />
                    <Text style={styles.whatIfButtonText}>Imagining…</Text>
                  </View>
                ) : (
                  <Text style={styles.whatIfButtonText}>
                    What if this was said differently?
                  </Text>
                )}
              </TouchableOpacity>

              {/* Honest inline error — never a fabricated simulation. */}
              {showWhatIfError && (
                <View style={styles.whatIfError} testID="what-if-error">
                  <Text style={styles.whatIfErrorText}>{whatIfError}</Text>
                  <TouchableOpacity
                    testID="what-if-retry"
                    onPress={onRetryWhatIf}
                  >
                    <Text style={styles.whatIfRetryText}>Try again</Text>
                  </TouchableOpacity>
                </View>
              )}
            </View>
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
  // Outline (unfilled) chip so prosody reads as a different KIND of tag than the
  // filled marker chips sitting beside it.
  voiceChip: {
    backgroundColor: "transparent",
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 12,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  voiceChipText: {
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
  // Dashed legend swatch: two short segments with a gap, echoing the overlay.
  dashSwatch: {
    width: 16,
    height: 12,
    flexDirection: "row",
    alignItems: "center",
    gap: 3,
    opacity: SIM_OPACITY,
  },
  dashSeg: {
    width: 6,
    height: 2,
    borderRadius: 1,
    backgroundColor: MUTED,
  },
  simToggle: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#F9FAFB",
  },
  simToggleText: {
    fontSize: 12,
    fontWeight: "600",
    color: MUTED,
  },
  whatIfBlock: {
    marginTop: 12,
  },
  whatIfButton: {
    backgroundColor: "#EFF6FF",
    borderWidth: 1,
    borderColor: PRIMARY,
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 14,
    alignItems: "center",
  },
  whatIfButtonLoading: {
    opacity: 0.7,
  },
  whatIfLoadingRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  whatIfButtonText: {
    fontSize: 14,
    fontWeight: "700",
    color: PRIMARY,
  },
  whatIfError: {
    marginTop: 8,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  },
  whatIfErrorText: {
    flex: 1,
    fontSize: 13,
    color: DANGER,
  },
  whatIfRetryText: {
    fontSize: 13,
    fontWeight: "700",
    color: PRIMARY,
  },
});
