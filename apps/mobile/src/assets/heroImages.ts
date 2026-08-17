/**
 * The six owner-curated hero images (Task P3-4b), pre-downscaled to ~1280px
 * wide JPEGs in apps/mobile/assets/hero/ — the originals in
 * assets/brand/hero/ are ~2MB PNGs each and are never shipped in the web
 * bundle. `require()` (not a static `import`) matches the dynamic-require
 * pattern already used for platform-conditional modules elsewhere in this
 * codebase (see RecordScreen.tsx, AudioRecordScreen.tsx) and sidesteps the
 * need for an ambient `declare module "*.jpg"` — @types/node's `require`
 * already returns `any`.
 */
export const HERO_IMAGES = [
  require("../../assets/hero/hero-1.jpg"),
  require("../../assets/hero/hero-2.jpg"),
  require("../../assets/hero/hero-3.jpg"),
  require("../../assets/hero/hero-4.jpg"),
  require("../../assets/hero/hero-5.jpg"),
  require("../../assets/hero/hero-6.jpg"),
];
