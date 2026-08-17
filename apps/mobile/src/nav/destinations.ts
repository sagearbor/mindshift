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
  | "therapistDashboard"
  | "settings"
  | "tutorial";

/** A navigation target shaped exactly like the App.tsx `Screen` union member
 *  it corresponds to. Only the fields those specific members need are
 *  present here — e.g. "recordings" carries `returnTo` because that member
 *  requires it.
 *
 *  Verified against App.tsx's real `Screen` union as part of Task N3 (P3-10)
 *  — every variant below still matches its Screen counterpart exactly, no
 *  drift found. This isn't just a one-time check: App.tsx now exports
 *  `Screen` and hands a `DestScreen` straight to `setScreen()`
 *  (App.tsx's `handleNavigate`), so a future drift (a Screen variant's shape
 *  changing without this type following) fails the TypeScript build right
 *  there, not silently at runtime. */
export type DestScreen =
  | { name: "live-coach" }
  | { name: "analyze" }
  | { name: "recordings"; returnTo: "home" | "analyze" }
  | { name: "growth" }
  | { name: "watch-setup" }
  | { name: "advanced" }
  | { name: "onboarding" }
  | { name: "dashboard" };

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
    id: "therapistDashboard",
    title: "Dashboard",
    iconId: "clipboard",
    screen: { name: "dashboard" },
    // Catalog-only: a Settings ("Your tools") row for reviewing saved
    // sessions, not a frequent primary action for most users.
    primaryEligible: false,
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
    // Catalog-only: a rarely-revisited walkthrough (AdvancedScreen's "Show
    // tutorial" row), not a daily action.
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
