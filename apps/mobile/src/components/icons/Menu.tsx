import React from "react";
import Svg, { Line } from "react-native-svg";
import {
  type IconProps,
  DEFAULT_ICON_COLOR,
  DEFAULT_ICON_SIZE,
  ICON_STROKE_WIDTH,
} from "./types";

/** Chrome glyph: hamburger — opens the full destination catalog. */
export default function Menu({
  size = DEFAULT_ICON_SIZE,
  color = DEFAULT_ICON_COLOR,
  testID,
}: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" testID={testID}>
      <Line
        x1="4"
        y1="6"
        x2="20"
        y2="6"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
      <Line
        x1="4"
        y1="12"
        x2="20"
        y2="12"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
      <Line
        x1="4"
        y1="18"
        x2="20"
        y2="18"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
    </Svg>
  );
}
