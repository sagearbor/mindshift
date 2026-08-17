import React from "react";
import Svg, { Line } from "react-native-svg";
import {
  type IconProps,
  DEFAULT_ICON_COLOR,
  DEFAULT_ICON_SIZE,
  ICON_STROKE_WIDTH,
} from "./types";

// Five bars of varying height, centered on y=12 — a still snapshot of a
// waveform, matching the "Analyze a conversation" destination.
const BARS: { x: number; half: number }[] = [
  { x: 3, half: 2 },
  { x: 7, half: 6 },
  { x: 11, half: 10 },
  { x: 15, half: 5 },
  { x: 19, half: 3 },
];

/** Analyze. A row of waveform bars. */
export default function Waveform({
  size = DEFAULT_ICON_SIZE,
  color = DEFAULT_ICON_COLOR,
  testID,
}: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" testID={testID}>
      {BARS.map((b) => (
        <Line
          key={b.x}
          x1={b.x}
          y1={12 - b.half}
          x2={b.x}
          y2={12 + b.half}
          stroke={color}
          strokeWidth={ICON_STROKE_WIDTH}
          strokeLinecap="round"
        />
      ))}
    </Svg>
  );
}
