import React from "react";
import Svg, { Line } from "react-native-svg";
import {
  type IconProps,
  DEFAULT_ICON_COLOR,
  DEFAULT_ICON_SIZE,
  ICON_STROKE_WIDTH,
} from "./types";

// Named ListIcon (not List) to avoid colliding with react-native's/JS's
// built-in List-ish globals and to keep the file grep-able by icon purpose.
const ROWS = [6, 12, 18];

/** Recordings. A stacked list — bullet + line per row. */
export default function ListIcon({
  size = DEFAULT_ICON_SIZE,
  color = DEFAULT_ICON_COLOR,
  testID,
}: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" testID={testID}>
      {ROWS.map((y) => (
        <React.Fragment key={y}>
          <Line
            x1="4"
            y1={y}
            x2="4.01"
            y2={y}
            stroke={color}
            strokeWidth={ICON_STROKE_WIDTH + 0.5}
            strokeLinecap="round"
          />
          <Line
            x1="8"
            y1={y}
            x2="21"
            y2={y}
            stroke={color}
            strokeWidth={ICON_STROKE_WIDTH}
            strokeLinecap="round"
          />
        </React.Fragment>
      ))}
    </Svg>
  );
}
