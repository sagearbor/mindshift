/**
 * Icon registry (Task N2 of P3-10). `ICONS` maps every destination's
 * `IconId` (from src/nav/destinations.ts, Task N1) to its SVG component —
 * typed as `Record<IconId, ...>` so a missing/misspelled key is a TYPE
 * ERROR at build time, not a silent blank icon at runtime.
 *
 * Chrome glyphs (hamburger, back, close, camera) aren't destinations — they
 * have no IconId — so they're exported separately below, not part of the
 * registry map.
 */
import type { ComponentType } from "react";
import type { IconId } from "../../nav/destinations";
import type { IconProps } from "./types";

import Mic from "./Mic";
import Waveform from "./Waveform";
import ListIcon from "./ListIcon";
import Trendline from "./Trendline";
import Watch from "./Watch";
import Voice from "./Voice";
import Clipboard from "./Clipboard";
import Gear from "./Gear";
import Book from "./Book";

import Menu from "./Menu";
import ChevronLeft from "./ChevronLeft";
import Close from "./Close";
import Camera from "./Camera";
import Home from "./Home";

export type { IconProps } from "./types";
export {
  DEFAULT_ICON_COLOR,
  DEFAULT_ICON_SIZE,
  ICON_STROKE_WIDTH,
} from "./types";

export {
  Mic,
  Waveform,
  ListIcon,
  Trendline,
  Watch,
  Voice,
  Clipboard,
  Gear,
  Book,
  Menu,
  ChevronLeft,
  Close,
  Camera,
  Home,
};

/** Every destination icon, keyed by the registry's IconId union. Adding a new
 *  IconId to destinations.ts without adding it here is a compile error. */
export const ICONS: Record<IconId, ComponentType<IconProps>> = {
  mic: Mic,
  waveform: Waveform,
  list: ListIcon,
  trendline: Trendline,
  watch: Watch,
  voice: Voice,
  clipboard: Clipboard,
  gear: Gear,
  book: Book,
};

/** Chrome glyphs — not destinations, so not part of `ICONS`, but kept in one
 *  place for N3's top bar / catalog dismissal / future selfie flow. */
export const CHROME_ICONS = {
  menu: Menu,
  back: ChevronLeft,
  close: Close,
  camera: Camera,
  home: Home,
} as const;

/** Look up a destination's icon component by id. */
export function getIcon(id: IconId): ComponentType<IconProps> {
  return ICONS[id];
}
