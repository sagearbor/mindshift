import React from "react";
import Svg, { Rect, Path, Line } from "react-native-svg";
import {
  type IconProps,
  DEFAULT_ICON_COLOR,
  DEFAULT_ICON_SIZE,
  ICON_STROKE_WIDTH,
} from "./types";

/** Set up your watch. A watch face with straps and hands. */
export default function Watch({
  size = DEFAULT_ICON_SIZE,
  color = DEFAULT_ICON_COLOR,
  testID,
}: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" testID={testID}>
      <Path
        d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <Path
        d="M9 18v2a1 1 0 0 0 1 1h4a1 1 0 0 0 1-1v-2"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <Rect
        x="6"
        y="6"
        width="12"
        height="12"
        rx="3"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
      />
      <Line
        x1="12"
        y1="9.5"
        x2="12"
        y2="12"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
      <Line
        x1="12"
        y1="12"
        x2="14"
        y2="13.2"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
    </Svg>
  );
}
