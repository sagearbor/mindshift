/**
 * NATIVE stub for the web-only hero wipe-reveal (Task P3-4b — mobile is
 * deferred per the plan). The real implementation lives in HeroWipe.web.tsx.
 *
 * This is a genuine platform-file split (matching GoogleSignInButton's
 * .native.tsx/.web.tsx pair), not a runtime `Platform.OS` check inside a
 * single file. That distinction matters: Metro resolves exactly one file per
 * bundle for a bare `./HeroWipe` import and never even looks at the other
 * platform's file — so HeroWipe.web.tsx's imports (heroImages.ts and its six
 * ~300KB-ish JPEGs, heroWipeSchedule.ts, heroWipeEffects.ts) are never
 * pulled into the native dependency graph at all, and therefore never ship
 * in an iOS/Android build OR an OTA update payload. A `Platform.OS === "web"`
 * guard inside one shared file (the previous version of this component)
 * only prevents those modules from *executing* at runtime — the top-level
 * `import` that pulls them into the bundle already ran by the time any
 * runtime check could stop it, so Metro still bundles the assets for every
 * platform regardless. Compare RecordScreen.tsx's dynamic, branch-local
 * `require()` (a different, narrower technique that relies on the
 * Platform.OS branch itself gating the require call, not a static
 * module-level import).
 *
 * The default (no `--platform` override) jest test environment resolves the
 * "native" haste platform (see @react-native/jest-preset's
 * `haste.platforms`), so this is also the file every screen test —
 * including HomeScreen's — exercises unless a test imports
 * `HeroWipe.web` directly by its explicit filename.
 */
export default function HeroWipe() {
  return null;
}
