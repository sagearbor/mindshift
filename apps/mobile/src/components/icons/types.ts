/**
 * Shared prop contract for every icon in this directory (Task N2 of P3-10).
 * Plain stroke icons: 24x24 viewBox, 2px stroke, round caps/joins, no fill —
 * consistent house style so a box full of mixed destinations still reads as
 * one icon set. `color` stands in for react-native-svg's lack of a real
 * `currentColor` — every icon defaults to the brand ink so an icon dropped
 * into ordinary body text looks intentional without a color prop.
 */
export interface IconProps {
  /** Width and height in px (viewBox is always 24x24, so this just scales
   *  the rendered SVG). Defaults to 24 — the size the glyphs were drawn at. */
  size?: number;
  /** Stroke color. Defaults to the house ink (`#1F2937`, matching
   *  HeatChart's INK) rather than a hardcoded brand blue, so an icon reads
   *  correctly wherever it's dropped (nav chrome, catalog list, tab bar)
   *  without every call site having to pass a color. */
  color?: string;
  testID?: string;
}

/** House ink — the default stroke color shared by every icon. */
export const DEFAULT_ICON_COLOR = "#1F2937";

/** The size every icon is drawn at (matches the 24x24 viewBox). */
export const DEFAULT_ICON_SIZE = 24;

/** Shared stroke weight — every icon uses the same 2px line so the set reads
 *  as one family regardless of glyph complexity. */
export const ICON_STROKE_WIDTH = 2;
