import React from "react";
import Svg, { Line } from "react-native-svg";
import {
  type IconProps,
  DEFAULT_ICON_COLOR,
  DEFAULT_ICON_SIZE,
  ICON_STROKE_WIDTH,
} from "./types";

/** Chrome glyph: close — dismisses a menu/sheet/full-screen catalog. */
export default function Close({
  size = DEFAULT_ICON_SIZE,
  color = DEFAULT_ICON_COLOR,
  testID,
}: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" testID={testID}>
      <Line
        x1="5"
        y1="5"
        x2="19"
        y2="19"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
      <Line
        x1="19"
        y1="5"
        x2="5"
        y2="19"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
    </Svg>
  );
}
