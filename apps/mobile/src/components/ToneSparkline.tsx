import React from "react";
import { View, StyleSheet } from "react-native";
import Svg, { Polyline, Circle } from "react-native-svg";

interface ToneSparklineProps {
  scores: number[];
  width?: number;
  height?: number;
  color?: string;
}

export default function ToneSparkline({
  scores,
  width = 120,
  height = 40,
  color = "#4A90D9",
}: ToneSparklineProps) {
  if (scores.length === 0) {
    return <View style={{ width, height }} testID="tone-sparkline-empty" />;
  }

  const padding = 4;
  const chartWidth = width - padding * 2;
  const chartHeight = height - padding * 2;

  const maxScore = 100;
  const minScore = 0;

  const points = scores.map((score, i) => {
    const x =
      scores.length === 1
        ? chartWidth / 2
        : (i / (scores.length - 1)) * chartWidth;
    const y = chartHeight - ((score - minScore) / (maxScore - minScore)) * chartHeight;
    return `${padding + x},${padding + y}`;
  });

  const lastPoint = scores[scores.length - 1];
  const lastX =
    scores.length === 1
      ? chartWidth / 2
      : chartWidth;
  const lastY =
    chartHeight -
    ((lastPoint - minScore) / (maxScore - minScore)) * chartHeight;

  return (
    <View style={[styles.container, { width, height }]} testID="tone-sparkline">
      <Svg width={width} height={height}>
        <Polyline
          points={points.join(" ")}
          fill="none"
          stroke={color}
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        <Circle
          cx={padding + lastX}
          cy={padding + lastY}
          r={3}
          fill={color}
        />
      </Svg>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    overflow: "hidden",
  },
});
