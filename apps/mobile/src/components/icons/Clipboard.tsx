import React from "react";
import Svg, { Rect, Line } from "react-native-svg";
import {
  type IconProps,
  DEFAULT_ICON_COLOR,
  DEFAULT_ICON_SIZE,
  ICON_STROKE_WIDTH,
} from "./types";

/** Therapist dashboard. A clipboard with rows of notes. */
export default function Clipboard({
  size = DEFAULT_ICON_SIZE,
  color = DEFAULT_ICON_COLOR,
  testID,
}: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" testID={testID}>
      <Rect
        x="5"
        y="4"
        width="14"
        height="17"
        rx="2"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
      />
      <Rect
        x="9"
        y="2"
        width="6"
        height="4"
        rx="1"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
      />
      <Line
        x1="8"
        y1="11"
        x2="16"
        y2="11"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
      <Line
        x1="8"
        y1="15"
        x2="16"
        y2="15"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
      <Line
        x1="8"
        y1="19"
        x2="13"
        y2="19"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
    </Svg>
  );
}
