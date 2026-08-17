import React from "react";
import Svg, { Path } from "react-native-svg";
import {
  type IconProps,
  DEFAULT_ICON_COLOR,
  DEFAULT_ICON_SIZE,
  ICON_STROKE_WIDTH,
} from "./types";

/** Show tutorial. An open book — two pages meeting at a center spine. */
export default function Book({
  size = DEFAULT_ICON_SIZE,
  color = DEFAULT_ICON_COLOR,
  testID,
}: IconProps) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" testID={testID}>
      <Path
        d="M12 6c-1.8-1.2-4-2-6.5-2A2.5 2.5 0 0 0 3 6.5v11A2.5 2.5 0 0 0 5.5 20c2.5 0 4.7.8 6.5 2"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <Path
        d="M12 6c1.8-1.2 4-2 6.5-2A2.5 2.5 0 0 1 21 6.5v11a2.5 2.5 0 0 1-2.5 2.5c-2.5 0-4.7.8-6.5 2"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <Path
        d="M12 6v16"
        stroke={color}
        strokeWidth={ICON_STROKE_WIDTH}
        strokeLinecap="round"
      />
    </Svg>
  );
}
