import React from "react";
import { View, Text, StyleSheet } from "react-native";
import Svg, { Circle, Line, Polyline, Text as SvgText } from "react-native-svg";

import type { GrowthPoint } from "../api/client";
import {
  TREND_MIN_POINTS,
  dateTicks,
  movingAverage,
  scoreToY,
  scoredPoints,
  timeMsToX,
  timeToX,
  timeWindow,
} from "./growthSeries";

const DEFAULT_COLOR = "#4A90D9";
const TREND_COLOR = "#9CB8D9";
const GRID_COLOR = "#E5E7EB";
const MUTED = "#6B7280";

/** Left gutter reserved for the y tick labels ("100" needs ~20px at 10pt). */
const Y_GUTTER = 30;
/** Bottom gutter reserved for the date tick labels. */
const X_GUTTER = 18;
/** Y gridlines + tick labels — the score scale is fixed 0–100. */
const Y_TICKS = [0, 25, 50, 75, 100];
/** Rough width of one date label; caps how many x ticks fit the plot. */
const X_TICK_PX = 56;

interface GrowthChartProps {
  /** Growth points ASCENDING by timestamp (the API's order). */
  points: GrowthPoint[];
  width: number;
  height: number;
  color?: string;
  dotRadius?: number;
  /** When given, each dot becomes tappable (→ that recording). */
  onPressPoint?: (point: GrowthPoint) => void;
  /** Draw real axes: labeled 0–100 gridlines on the left, date ticks under
   *  the plot, and a "Score (0–100) ↑" caption above. Off by default so the
   *  home strip's bare sparkline keeps its exact geometry. */
  axes?: boolean;
}

/**
 * The "Your growth" dot chart — shared by the home strip (small, passive) and
 * GrowthScreen (large, tappable dots, `axes`).
 *
 * Dots are per-recording `my_score` values on a TIME axis (a quiet month reads
 * as a quiet month, not as adjacent dots). Identified recordings whose stored
 * analysis has no usable score are GAPS — no dot, no interpolation, never a
 * zero. The moving-average trend line only appears at ≥ TREND_MIN_POINTS
 * scored points; below that a "trend" would be noise dressed up as signal.
 */
export default function GrowthChart({
  points,
  width,
  height,
  color = DEFAULT_COLOR,
  dotRadius = 3,
  onPressPoint,
  axes = false,
}: GrowthChartProps) {
  const scored = scoredPoints(points);
  const window = timeWindow(scored);
  if (scored.length === 0 || window === null || width <= 0) {
    return (
      <View style={{ width: Math.max(0, width), height }} testID="growth-chart-empty" />
    );
  }

  const padding = dotRadius + 2;
  // With axes on, the plot is inset by the two gutters; without them it is the
  // whole SVG (unchanged sparkline geometry).
  const plotLeft = axes ? Y_GUTTER : 0;
  const plotHeight = axes ? height - X_GUTTER : height;
  const geom = { width: width - plotLeft, padding };
  const xy = scored.map((p) => ({
    point: p,
    x: plotLeft + timeToX(p.timestamp, window, geom),
    y: scoreToY(p.my_score as number, plotHeight, padding),
  }));

  const trend =
    scored.length >= TREND_MIN_POINTS
      ? movingAverage(scored.map((p) => p.my_score as number))
      : null;

  const xTicks = axes
    ? dateTicks(window, Math.max(2, Math.floor(geom.width / X_TICK_PX)))
    : [];

  const svg = (
    <View style={[styles.container, { width, height }]} testID="growth-chart">
      <Svg width={width} height={height}>
        {axes
          ? Y_TICKS.map((v) => {
              const y = scoreToY(v, plotHeight, padding);
              return (
                <React.Fragment key={`y-${v}`}>
                  <Line
                    testID={`growth-axis-y-grid-${v}`}
                    x1={plotLeft}
                    x2={width}
                    y1={y}
                    y2={y}
                    stroke={GRID_COLOR}
                    strokeWidth={1}
                  />
                  <SvgText
                    testID={`growth-axis-y-tick-${v}`}
                    x={plotLeft - 4}
                    y={y + 3.5}
                    fontSize={10}
                    fill={MUTED}
                    textAnchor="end"
                  >
                    {String(v)}
                  </SvgText>
                </React.Fragment>
              );
            })
          : null}
        {xTicks.map((tick) => {
          const x = plotLeft + timeMsToX(tick.t, window, geom);
          // Keep edge labels inside the (clipped) SVG.
          const anchor =
            x < plotLeft + X_TICK_PX / 2
              ? "start"
              : x > width - X_TICK_PX / 2
                ? "end"
                : "middle";
          return (
            <SvgText
              key={`x-${tick.t}`}
              testID="growth-axis-x-tick"
              x={x}
              y={plotHeight + 12}
              fontSize={10}
              fill={MUTED}
              textAnchor={anchor}
            >
              {tick.label}
            </SvgText>
          );
        })}
        {trend ? (
          <Polyline
            testID="growth-trend"
            points={xy
              .map((d, i) => `${d.x},${scoreToY(trend[i], plotHeight, padding)}`)
              .join(" ")}
            fill="none"
            stroke={TREND_COLOR}
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ) : null}
        {xy.map((d) => (
          <Circle
            key={d.point.recording_id}
            testID={`growth-dot-${d.point.recording_id}`}
            cx={d.x}
            cy={d.y}
            r={onPressPoint ? dotRadius + 2 : dotRadius}
            fill={color}
            onPress={
              onPressPoint ? () => onPressPoint(d.point) : undefined
            }
          />
        ))}
      </Svg>
    </View>
  );

  if (!axes) return svg;

  // Same plain caption HeatChart uses for its y axis ("Heat (0–100) ↑"), so
  // the two charts read the same way.
  return (
    <View style={{ width }}>
      <Text style={styles.axisLabel} testID="growth-axis-y-label">
        Score (0–100) ↑
      </Text>
      {svg}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    overflow: "hidden",
  },
  axisLabel: {
    fontSize: 11,
    color: MUTED,
    fontWeight: "600",
    marginBottom: 4,
  },
});
