import React from "react";
import Svg, { Rect, Path, Circle } from "react-native-svg";
import {
  type IconProps,
  DEFAULT_ICON_COLOR,
  DEFAULT_ICON_SIZE,
  ICON_STROKE_WIDTH,
} from "./types";

/** Chrome glyph: camera — the future selfie/avatar-capture flow. */
export default function Camera({
  size = DEFAULT_ICON_SIZE,
  color = DEFAULT_ICON_COLOR,
  testID,
}: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" testID={testID}>
      <Path
        d="M8 7l1.5-2h5L16 7"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <Rect
        x="3"
        y="7"
        width="18"
        height="13"
        rx="2"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
      />
      <Circle
        cx="12"
        cy="13.5"
        r="3.5"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
      />
    </Svg>
  );
}
