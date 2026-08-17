import React from "react";
import Svg, { Circle, Line } from "react-native-svg";
import {
  type IconProps,
  DEFAULT_ICON_COLOR,
  DEFAULT_ICON_SIZE,
  ICON_STROKE_WIDTH,
} from "./types";

// Six teeth at 60-degree increments around the body, precomputed so render
// stays a flat map with no per-frame trig.
const TEETH: { x1: number; y1: number; x2: number; y2: number }[] = [
  { x1: 19, y1: 12, x2: 22, y2: 12 },
  { x1: 15.5, y1: 18.1, x2: 17, y2: 20.7 },
  { x1: 8.5, y1: 18.1, x2: 7, y2: 20.7 },
  { x1: 5, y1: 12, x2: 2, y2: 12 },
  { x1: 8.5, y1: 5.9, x2: 7, y2: 3.3 },
  { x1: 15.5, y1: 5.9, x2: 17, y2: 3.3 },
];

/** Settings. A gear — body ring, center hole, six teeth. */
export default function Gear({
  size = DEFAULT_ICON_SIZE,
  color = DEFAULT_ICON_COLOR,
  testID,
}: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" testID={testID}>
      {TEETH.map((t, i) => (
        <Line
          key={i}
          x1={t.x1}
          y1={t.y1}
          x2={t.x2}
          y2={t.y2}
          stroke={color}
          strokeWidth={ICON_STROKE_WIDTH}
          strokeLinecap="round"
        />
      ))}
      <Circle cx="12" cy="12" r="7" stroke={color} strokeWidth={ICON_STROKE_WIDTH} />
      <Circle cx="12" cy="12" r="2.5" stroke={color} strokeWidth={ICON_STROKE_WIDTH} />
    </Svg>
  );
}
