/**
 * The complete registry of navigable destinations (Task N1 — the foundation
 * for P3-10's customizable home & nav). This is the single source of truth
 * the hamburger's full catalog (N3), the bottom tab bar (N3), the home boxes
 * (N4), and the "Home screen design" editor (N5) all read from.
 *
 * Derived from the REAL current app — App.tsx's hand-rolled `Screen` union
 * and AdvancedScreen's (Settings') actual rows — not from the original P3-10
 * proposal's aspirational set. Nothing here is a screen that doesn't exist
 * yet.
 *
 * `screen` deliberately mirrors the *shape* of the corresponding App.tsx
 * `Screen` union member (same `name` literal, same extra fields) without
 * importing that type — `Screen` isn't exported (App.tsx stays untouched in
 * this task) and the two are structurally compatible, so N3's chrome can
 * hand a destination's `screen` straight to `setScreen()` when it's wired up.
 */

/** Every destination's stable id. Used as the persisted vocabulary for
 *  layoutStore's tabSlots/homeBoxes — never rename an existing id (it'd
 *  silently orphan a user's saved layout; add a new id instead). */
export type DestId =
  | "coach"
  | "analyze"
  | "recordings"
  | "growth"
  | "watchSetup"
  | "voiceProfile"
  | "people"
  | "therapistDashboard"
  | "settings"
  | "tutorial";

/** A navigation target shaped like the App.tsx `Screen` union member it
 *  corresponds to — deliberately NOT always identical. Two ways a variant
 *  can relate to its Screen counterpart, each documented per-variant below:
 *
 *   1. Exact match, no extra fields — "live-coach", "analyze", "growth",
 *      "advanced". Handed straight to `setScreen()` with no massaging.
 *   2. "recordings" carries the reduced `returnTo` its Screen counterpart
 *      needs ("home" | "analyze", not the full recursive `Screen`) — the
 *      registry statically knows both legal origins, so it can supply the
 *      real value itself.
 *   3. "watch-setup" / "onboarding" / "dashboard" deliberately OMIT
 *      `returnTo` here even though their Screen counterparts require
 *      `returnTo: Screen` — the registry is static data with no way to know
 *      which screen the hamburger catalog (or a tab/box) was actually
 *      opened FROM, so it can't fill that field in. App.tsx's
 *      `handleNavigate` patches it in instead, from whatever screen is
 *      current at nav time, before calling `setScreen()` — see that
 *      function's comment for the three-case early-return that does it.
 *
 *  Verified against App.tsx's real `Screen` union as part of Task N3 (P3-10)
 *  and re-verified for Task N5: every variant below is intentionally either
 *  an exact match or a documented, `handleNavigate`-patched exception —
 *  never silent drift. App.tsx exports `Screen` and either hands a
 *  `DestScreen` straight to `setScreen()` (case 1/2) or spreads it with a
 *  patched `returnTo` first (case 3), so an UNDOCUMENTED shape mismatch (a
 *  Screen variant's required fields changing without this type or
 *  `handleNavigate` following) still fails the TypeScript build right there,
 *  not silently at runtime. */
export type DestScreen =
  | { name: "live-coach" }
  | { name: "analyze" }
  | { name: "recordings"; returnTo: "home" | "analyze" }
  | { name: "growth" }
  | { name: "watch-setup" }
  | { name: "advanced" }
  | { name: "onboarding" }
  | { name: "dashboard" }
  // People labeling — case 3 like watch-setup: `returnTo` is patched in by
  // App.tsx's handleNavigate from whatever screen is current.
  | { name: "people" };

/** Icon key naming — N2 will name its SVG components with these exact
 *  strings (mic/coach, waveform/analyze, list/recordings, trendline/growth,
 *  watch, voice, clipboard/dashboard, gear/settings, book/tutorial). This
 *  registry only *names* the icon; N2 supplies the actual glyph. */
export type IconId =
  | "mic"
  | "waveform"
  | "list"
  | "trendline"
  | "watch"
  | "voice"
  | "clipboard"
  | "gear"
  | "book";

export interface Destination {
  id: DestId;
  title: string;
  iconId: IconId;
  /** Where navigating to this destination lands, in App.tsx's Screen shape. */
  screen: DestScreen;
  /** True if this destination is a legitimate candidate for the bottom tab
   *  bar / home boxes (the user's frequent, self-contained actions). False
   *  means "catalog-only" — still reachable via the hamburger's full list,
   *  but not offerable as a tab/box slot. See each entry's comment for why. */
  primaryEligible: boolean;
}

export const DESTINATIONS: readonly Destination[] = [
  {
    id: "coach",
    title: "Live Coach",
    iconId: "mic",
    screen: { name: "live-coach" },
    // Primary: one of the app's two core daily modes (HomeScreen's own
    // top-billed card).
    primaryEligible: true,
  },
  {
    id: "analyze",
    title: "Analyze a Conversation",
    iconId: "waveform",
    screen: { name: "analyze" },
    // Primary: the app's other core daily mode.
    primaryEligible: true,
  },
  {
    id: "recordings",
    title: "Recordings",
    iconId: "list",
    screen: { name: "recordings", returnTo: "home" },
    // Primary: HomeScreen already surfaces this as a compact history entry
    // point — it's a frequent, self-contained destination, not a settings
    // action.
    primaryEligible: true,
  },
  {
    id: "growth",
    title: "Your Growth",
    iconId: "trendline",
    screen: { name: "growth" },
    // Primary: the owner explicitly wants home to be able to lean on the
    // growth trend/history (P3-9 RESOLVED); HomeScreen already surfaces a
    // self-fetching growth strip on the primary surface.
    primaryEligible: true,
  },
  {
    id: "watchSetup",
    title: "Set up your watch",
    iconId: "watch",
    screen: { name: "watch-setup" },
    // Catalog-only: a one-time setup flow, not a repeated daily action —
    // it's a Settings row today (AdvancedScreen's "Your tools" section), not
    // something a user taps every session.
    primaryEligible: false,
  },
  {
    id: "voiceProfile",
    title: "Voice profile",
    iconId: "voice",
    // Not a separate pushed screen in App.tsx today — voice profile
    // management lives inline inside AdvancedScreen ("Voice" section). This
    // is a section anchor: until Settings supports deep-linking to a scroll
    // position, navigating here just opens Settings itself.
    screen: { name: "advanced" },
    // Catalog-only: infrequent account-management action, and not even an
    // independently addressable screen yet.
    primaryEligible: false,
  },
  {
    id: "people",
    title: "People",
    iconId: "voice",
    // People labeling: everyone whose voice the app recognizes. A pushed
    // screen (App.tsx `people`, dynamic returnTo via handleNavigate).
    screen: { name: "people" },
    // Catalog-only: account management (add/rename/forget voices), not a
    // daily action — same reasoning as voiceProfile above.
    primaryEligible: false,
  },
  {
    id: "therapistDashboard",
    title: "Dashboard",
    iconId: "clipboard",
    screen: { name: "dashboard" },
    // Primary (2026-08-19 primary-eligible-expand): the owner asked whether
    // a destination like this should be addable to the bottom bar / home
    // boxes too — it's a self-contained, meaningful screen a user might
    // reasonably want quick access to, same reasoning as growth/coach/
    // analyze already being primary-eligible. Safe to flip unconditionally:
    // App.tsx's `dashboard` Screen variant has exactly ONE push site
    // (`onOpenDashboard`, plus the equivalent hamburger-catalog case in
    // `handleNavigate`), both always attaching a dynamic `returnTo` — unlike
    // "recordings" below, there's no second genuinely-different origin that
    // needs to stay pushed, so no instance-predicate special case is needed
    // in App.tsx's `isPrimary()`; it just becomes unconditionally primary by
    // name, same as growth/coach/analyze (see App.tsx's PRIMARY_SCREEN_NAMES
    // comment and its `dashboard` render case, and backHandler.ts's
    // `backTarget` — both updated to match: no more on-screen back button,
    // and hardware back now goes Home like the other primary screens
    // instead of popping through `returnTo`).
    primaryEligible: true,
  },
  {
    id: "settings",
    title: "Settings",
    iconId: "gear",
    screen: { name: "advanced" },
    // Catalog-only by design: Settings is reached via the avatar menu / the
    // hamburger's catalog, not offered as a tab/box slot (it would be an odd
    // recursive "customize your customization" primary action).
    primaryEligible: false,
  },
  {
    id: "tutorial",
    title: "Show tutorial",
    iconId: "book",
    screen: { name: "onboarding" },
    // Still catalog-only (2026-08-19 primary-eligible-expand investigated
    // and deliberately DID NOT flip this one, unlike therapistDashboard
    // above). Three reasons, not just "rarely revisited":
    //  1. The FIRST-LAUNCH auto-shown walkthrough never goes through this
    //     registry or App.tsx's isPrimary()/PRIMARY_SCREEN_NAMES at all —
    //     it's a separate top-level gate in App.tsx (the `onboardingSeen`
    //     check) rendered before the Screen union is even reached. So
    //     flipping this flag would only affect RE-ENTRY (Settings' "Show
    //     tutorial" row / the hamburger catalog), never the first-launch
    //     flow.
    //  2. OnboardingScreen is a focused, linear, skippable card carousel
    //     (Skip / Back / Next / "Get started" — no `onBack` prop at all,
    //     unlike every other pushed screen). Wrapping it in full AppChrome
    //     (hamburger + avatar + tab bar) would let a user tap away to
    //     another tab mid-walkthrough, undermining the one-thing-at-a-time
    //     intent, while not even gaining a back button it doesn't
    //     effectively already have via Skip.
    //  3. Unlike "recordings" (whose two origins carry two different
    //     literal `returnTo` values — "home" vs "analyze" — decidable
    //     per-instance in isPrimary()), EVERY reachable path to the
    //     `onboarding` pushed screen (tab tap, hamburger catalog, Settings'
    //     row) goes through the same `handleNavigate` dynamic-`returnTo`
    //     patch, so there's no static shape to distinguish "opened as a
    //     tab" from "opened from the catalog" the way recordings' instance
    //     predicate does. The recordings-style special case isn't cleanly
    //     representable here without deeper Screen-union restructuring, so
    //     it's left catalog-only for now rather than force-fit.
    primaryEligible: false,
  },
];

const DESTINATIONS_BY_ID: ReadonlyMap<DestId, Destination> = new Map(
  DESTINATIONS.map((d) => [d.id, d]),
);

/** Look up a destination by id, or undefined if it's not a known id (e.g. a
 *  stale/corrupt persisted value from a prior app version). */
export function getDestination(id: string): Destination | undefined {
  return DESTINATIONS_BY_ID.get(id as DestId);
}

/** Every destination usable as a tab-bar/home-box slot, in registry order —
 *  the source list the "Home screen design" editor (N5) offers to add from. */
export const PRIMARY_ELIGIBLE_DESTINATIONS: readonly Destination[] =
  DESTINATIONS.filter((d) => d.primaryEligible);

/** True iff `id` is both a known destination and eligible for tabs/boxes. */
export function isPrimaryEligible(id: string): id is DestId {
  return DESTINATIONS_BY_ID.get(id as DestId)?.primaryEligible === true;
}
