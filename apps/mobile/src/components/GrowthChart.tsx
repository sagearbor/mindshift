import React from "react";
import { View, StyleSheet } from "react-native";
import Svg, { Circle, Polyline } from "react-native-svg";

import type { GrowthPoint } from "../api/client";
import {
  TREND_MIN_POINTS,
  movingAverage,
  scoreToY,
  scoredPoints,
  timeToX,
  timeWindow,
} from "./growthSeries";

const DEFAULT_COLOR = "#4A90D9";
const TREND_COLOR = "#9CB8D9";

interface GrowthChartProps {
  /** Growth points ASCENDING by timestamp (the API's order). */
  points: GrowthPoint[];
  width: number;
  height: number;
  color?: string;
  dotRadius?: number;
  /** When given, each dot becomes tappable (→ that recording). */
  onPressPoint?: (point: GrowthPoint) => void;
}

/**
 * The "Your growth" dot chart — shared by the home strip (small, passive) and
 * GrowthScreen (large, tappable dots).
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
}: GrowthChartProps) {
  const scored = scoredPoints(points);
  const window = timeWindow(scored);
  if (scored.length === 0 || window === null || width <= 0) {
    return (
      <View style={{ width: Math.max(0, width), height }} testID="growth-chart-empty" />
    );
  }

  const padding = dotRadius + 2;
  const geom = { width, padding };
  const xy = scored.map((p) => ({
    point: p,
    x: timeToX(p.timestamp, window, geom),
    y: scoreToY(p.my_score as number, height, padding),
  }));

  const trend =
    scored.length >= TREND_MIN_POINTS
      ? movingAverage(scored.map((p) => p.my_score as number))
      : null;

  return (
    <View style={[styles.container, { width, height }]} testID="growth-chart">
      <Svg width={width} height={height}>
        {trend ? (
          <Polyline
            testID="growth-trend"
            points={xy
              .map((d, i) => `${d.x},${scoreToY(trend[i], height, padding)}`)
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
}

const styles = StyleSheet.create({
  container: {
    overflow: "hidden",
  },
});
