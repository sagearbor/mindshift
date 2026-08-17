import React from "react";
import Svg, { Circle, Path, Line } from "react-native-svg";
import {
  type IconProps,
  DEFAULT_ICON_COLOR,
  DEFAULT_ICON_SIZE,
  ICON_STROKE_WIDTH,
} from "./types";

/** Voice profile. A person silhouette beside a small waveform — "this
 *  person's voice, as data". */
export default function Voice({
  size = DEFAULT_ICON_SIZE,
  color = DEFAULT_ICON_COLOR,
  testID,
}: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" testID={testID}>
      <Circle cx="8" cy="7" r="3" stroke={color} strokeWidth={ICON_STROKE_WIDTH} />
      <Path
        d="M3 20c0-3.31 2.24-6 5-6s5 2.69 5 6"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
      <Line
        x1="16"
        y1="10"
        x2="16"
        y2="14"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
      <Line
        x1="19"
        y1="7"
        x2="19"
        y2="17"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
      <Line
        x1="22"
        y1="10"
        x2="22"
        y2="14"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
    </Svg>
  );
}
