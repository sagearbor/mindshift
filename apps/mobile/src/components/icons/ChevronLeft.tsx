import React from "react";
import Svg, { Polyline } from "react-native-svg";
import {
  type IconProps,
  DEFAULT_ICON_COLOR,
  DEFAULT_ICON_SIZE,
  ICON_STROKE_WIDTH,
} from "./types";

/** Chrome glyph: back — pops a pushed detail screen. */
export default function ChevronLeft({
  size = DEFAULT_ICON_SIZE,
  color = DEFAULT_ICON_COLOR,
  testID,
}: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" testID={testID}>
      <Polyline
        points="15,4 7,12 15,20"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}
