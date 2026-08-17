import React from "react";
import Svg, { Rect, Path, Line } from "react-native-svg";
import {
  type IconProps,
  DEFAULT_ICON_COLOR,
  DEFAULT_ICON_SIZE,
  ICON_STROKE_WIDTH,
} from "./types";

/** Coach (live coach). A mic capsule on a stand — speech, live conversation. */
export default function Mic({
  size = DEFAULT_ICON_SIZE,
  color = DEFAULT_ICON_COLOR,
  testID,
}: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" testID={testID}>
      <Rect
        x="9"
        y="2"
        width="6"
        height="12"
        rx="3"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
      />
      <Path
        d="M5 11a7 7 0 0 0 14 0"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
      <Line
        x1="12"
        y1="18"
        x2="12"
        y2="22"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
      <Line
        x1="8"
        y1="22"
        x2="16"
        y2="22"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
    </Svg>
  );
}
