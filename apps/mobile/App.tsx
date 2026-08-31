import React, { useEffect, useRef, useState } from "react";
import { ActivityIndicator, Linking, Platform, StyleSheet } from "react-native";
// SafeAreaProvider/SafeAreaView come from react-native-safe-area-context (NOT
// react-native): the RN SafeAreaView is a no-op on Android, so under Expo's
// edge-to-edge every screen rendered its header UNDER the status bar — which not
// only looked wrong but ate taps on the top-corner nav buttons (the reported
// "Home / Recordings buttons do nothing" bug). The context version reads the real
// Android/iOS insets and pads the content clear of the system bars.
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";
import HomeScreen from "./src/screens/HomeScreen";
import AnalyzeScreen from "./src/screens/AnalyzeScreen";
import AdvancedScreen from "./src/screens/AdvancedScreen";
import PeopleScreen from "./src/screens/PeopleScreen";
import HomeDesignScreen from "./src/screens/HomeDesignScreen";
import WatchSetupScreen from "./src/screens/WatchSetupScreen";
import SessionScreen from "./src/screens/SessionScreen";
import TherapistDashboard from "./src/screens/TherapistDashboard";
import SessionDetail from "./src/screens/SessionDetail";
import DynamicsScreen from "./src/screens/DynamicsScreen";
import ReplayScreen from "./src/screens/ReplayScreen";
import RecordScreen from "./src/screens/RecordScreen";
import AvatarCaptureScreen from "./src/screens/AvatarCaptureScreen";
import RecordingsScreen from "./src/screens/RecordingsScreen";
import YourDayScreen from "./src/screens/YourDayScreen";
import GrowthScreen from "./src/screens/GrowthScreen";
import LiveCoachScreen from "./src/screens/LiveCoachScreen";
import LoginScreen from "./src/screens/LoginScreen";
import OnboardingScreen from "./src/screens/OnboardingScreen";
import UpdateBanner from "./src/components/UpdateBanner";
import AppChrome, { type AppChromeHandle } from "./src/components/AppChrome";
import PushedScreenChrome from "./src/components/PushedScreenChrome";
import { useAndroidBackHandler } from "./src/nav/useAndroidBackHandler";
import { PRIMARY_ELIGIBLE_DESTINATIONS, type DestScreen } from "./src/nav/destinations";
import { useAuthStore, initAuth } from "./src/store/authStore";
import { useLayoutStore } from "./src/store/layoutStore";
import { useAvatarStore } from "./src/store/avatarStore";
import { useSessionStore } from "./src/store/sessionStore";
import { useRecorderStore } from "./src/store/recorderStore";
import { getOnboardingSeen, setOnboardingSeen } from "./src/utils/onboardingStorage";
import { parseCallLink, type ParsedCallLink } from "./src/nav/callLink";
import { parseJournalLink, type JournalLinkAction } from "./src/nav/journalLink";
import type { AnalyzeResult } from "./src/api/client";

// --- Two-mode navigation -----------------------------------------------------
// The home screen is a radically simple choice between the app's two modes:
// Live Coach (real-time earbud coaching) and Analyze a Conversation
// (everything after-the-fact: record / upload / link / past recordings).
// Everything else is reachable via AppChrome's hamburger catalog or avatar
// menu (Task N3) — the "advanced" screen key below still names the
// destination internally (AdvancedScreen), but the user-facing title is
// "Settings".
//
// Navigation stays the same hand-rolled screen union as before (no nav lib):
// every screen is either PRIMARY (wrapped in AppChrome, no back button of
// its own) or PUSHED (its existing back affordance, carries enough state to
// get back) — see PRIMARY_SCREEN_NAMES below for the exact rule.

/** Where the text-tools (Session) screen should return to: it's pushed both
 *  from Analyze ("Work with text") and from Live Coach's review handoff. */
type SessionReturn = "home" | "analyze";

/** Where a replay should return to. A recordings-origin replay must restore
 *  the recordings list *with its own* back target, so it carries one. */
type ReplayReturn =
  | { name: "recordings"; returnTo: "home" | "analyze" }
  | { name: "analyze" }
  | { name: "session"; returnTo: SessionReturn }
  // The "Your Day" episode timeline (Companion P1) opens replays too.
  | { name: "your-day" }
  // "Your growth" dot-taps open the backing recording's replay.
  | { name: "growth" }
  // The voice-profile card's per-sample "Play" opens the source recording.
  | { name: "advanced" };

// Exported (Task N3) so the pure Android back-handler logic
// (src/nav/backHandler.ts) and the registry's DestScreen (src/nav/
// destinations.ts) can be checked against the real union at compile time —
// see destinations.ts's DestScreen comment for how that guard works.
export type Screen =
  | { name: "home" }
  // `joinCode`: an in-app call invite (mindshift://call/<code>, or the web
  // route /call/<code>) opens Live Coach in Call mode with an Answer button.
  // `journalAction`: a "Hey Google, start my journal" deep link
  // (mindshift://journal/start|stop) — the screen selects Journal mode and
  // starts/stops it, honoring its own gates (src/nav/journalLink.ts).
  | {
      name: "live-coach";
      joinCode?: string;
      joinRole?: "participant" | "therapist";
      journalAction?: JournalLinkAction;
    }
  // The Analyze mode hub: record / upload / link + relationship context.
  | { name: "analyze" }
  // Everything that doesn't fit the two modes: Settings (dashboard, watch
  // setup, voice profile, About, log out).
  //
  // N7 fix round 1 (IMPORTANT 2): `returnTo` mirrors watch-setup/onboarding/
  // dashboard below — Settings and Voice profile both route here and are
  // reachable from every primary screen's avatar menu (not just Settings'
  // own rows), so back needs to land wherever it was actually opened from,
  // not always Home. Optional (defaulting to Home when absent) so every
  // existing Home-originated `setScreen({ name: "advanced" })` call stays
  // valid unchanged.
  | { name: "advanced"; returnTo?: Screen }
  // Task N5 of P3-10: Settings' "Home screen design" editor — arrange
  // layoutStore's tabSlots/homeBoxes. Only reachable from Settings' own row
  // today (not the hamburger catalog — editing your own customization isn't
  // a registry destination, same reasoning as "settings" itself being
  // catalog-only, see destinations.ts), so the DESTINATION is always
  // "advanced" — no dynamic-destination patching needed like watch-setup/
  // onboarding/dashboard below.
  //
  // N7 fix round 1 re-review (IMPORTANT 2, second pass): `returnTo` still
  // carries the WHOLE `advanced` screen it was pushed from (with ITS OWN
  // `returnTo`) — same pattern `detail` uses for `dashboard` below — so
  // popping back to it restores that chain instead of resetting to a bare
  // `{name:"advanced"}` that's forgotten where Settings itself was opened
  // from.
  | { name: "home-design"; returnTo: Screen }
  // People labeling: everyone whose voice the app recognizes — add / rename /
  // forget. Pushed from Settings' "People" row and the hamburger catalog;
  // `returnTo` carries wherever it was launched from (same dynamic-returnTo
  // pattern as watch-setup below).
  | { name: "people"; returnTo: Screen }
  // Phase 3 Slice 1: install the watch app + redeem its pairing code.
  // Pushed from Settings' "Set up your watch" row (returnTo "advanced") AND
  // (Task N3 fix round 1) from the hamburger catalog on any primary screen —
  // `returnTo` carries whichever screen it was actually launched from, so
  // back lands there instead of always landing in Settings.
  | { name: "watch-setup"; returnTo: Screen }
  // Task P3-7: the first-launch onboarding walkthrough, re-entered from
  // Settings' "Show tutorial" row (returnTo "advanced") or, since the
  // hamburger catalog fix, from any primary screen. (The auto-shown-once
  // launch of this same screen is a separate top-level gate below, not part
  // of this pushed-screen union — see `onboardingSeen`.)
  | { name: "onboarding"; returnTo: Screen }
  // Task N6 of P3-10: the selfie-capture flow (front camera -> preview ->
  // Use/Retake), reached from the avatar menu's "Set profile photo" row or
  // Settings' matching Account-section row (both returnTo "advanced" or
  // whichever primary screen the avatar menu was opened from).
  | { name: "avatar-capture"; returnTo: Screen }
  // The text tools (paste/type a transcript, suggestions). Pushed from
  // Analyze and from Live Coach's post-session review handoff.
  | { name: "session"; returnTo: SessionReturn }
  // Reachable from Settings' "Dashboard" row (returnTo "advanced") or the
  // hamburger catalog from any primary screen — same returnTo treatment as
  // watch-setup/onboarding above.
  | { name: "dashboard"; returnTo: Screen }
  // `returnTo` here is the FULL dashboard screen (itself carrying its own
  // returnTo) it was pushed from, so back pops straight to it unchanged —
  // same pattern as replay's returnTo below.
  | { name: "detail"; sessionId: string; returnTo: Screen }
  // Post-session Conversation Dynamics analysis.
  //
  // `initialData` is a ready-made analysis handed over from the
  // recording-upload flow — when present, DynamicsScreen renders it directly
  // instead of re-POSTing /analyze. Absent for the text-tools "Analyze
  // dynamics" button, which analyzes the store transcript on mount.
  //
  // `recordingId` is the server-assigned id of a *stored* recording (set only
  // when the upload flow's consent+store both landed as true); undefined
  // otherwise. Carried through so DynamicsScreen can offer a Replay affordance.
  //
  // `cameFromRecorder` marks an analysis whose file was just recorded in-app
  // (and saved to the camera roll). When true AND the recording was stored,
  // DynamicsScreen offers the "attach HD source later" popup.
  //
  // `returnTo` records which screen pushed it (analyze vs. text tools).
  | {
      name: "dynamics";
      initialData?: AnalyzeResult;
      recordingId?: string | null;
      cameFromRecorder?: boolean;
      returnTo: { name: "analyze" } | { name: "session"; returnTo: SessionReturn };
    }
  // Stored-recordings list — reachable from Home (compact history entry) and
  // from the Analyze screen.
  | { name: "recordings"; returnTo: "home" | "analyze" }
  // "Your Day" (Companion P1): the day timeline of recorded conversations and
  // their episodes. Reachable from Home's compact history row; episode taps
  // push the existing replay, which returns here.
  | { name: "your-day" }
  // "Your growth": the full score-over-time chart behind Home's growth strip.
  // Dot taps push the existing replay, which returns here.
  | { name: "growth" }
  // In-app 480p video recording. On success it hands the recorded file to the
  // Analyze upload flow (via the recorder store).
  | { name: "record" }
  // Media replay with the synced heat graph. `openAttach` opens the
  // attach-HD-source input immediately (from the Dynamics popup).
  | {
      name: "replay";
      recordingId: string;
      returnTo: ReplayReturn;
      openAttach?: boolean;
    };

/**
 * PRIMARY vs PUSHED screens (Task N3 of P3-10). A screen is PRIMARY — wrapped
 * in AppChrome's persistent top bar (hamburger + wordmark + avatar) and
 * configurable tab bar, with no back button of its own — iff it's Home, or
 * its Screen variant needs no extra context beyond `name` to render AND that
 * same shape is one of the registry's primary-eligible destinations
 * (src/nav/destinations.ts: PRIMARY_ELIGIBLE_DESTINATIONS — coach, analyze,
 * recordings, growth, and (2026-08-19 primary-eligible-expand) dashboard).
 * Derived from that list (plus "home") rather than hand-typed, so a future
 * registry change can't silently drift from it.
 *
 * "recordings" is deliberately FILTERED OUT of this name-derived set: unlike
 * the other primary-eligible destinations, its Screen/DestScreen shape
 * always carries a `returnTo` — whether it's actually PRIMARY depends on
 * that field's VALUE, not just the screen's name. `isPrimary` below handles
 * it as a one-off: `returnTo: "home"` (Home's own history row, a configured
 * tab/box slot, or the hamburger catalog — all of which default to "home"
 * per the registry) is primary and gets full chrome, no back button;
 * `returnTo: "analyze"` (pushed from Analyze's own "past recordings" row)
 * stays pushed with its existing dedicated back button, completely
 * unchanged.
 *
 * "dashboard" (2026-08-19 primary-eligible-expand: flipped to
 * primaryEligible in the registry, so it belongs in this name-derived set,
 * NOT filtered out like recordings) does NOT need recordings' instance-
 * predicate treatment: it has exactly one push site in App.tsx
 * (`onOpenDashboard`, plus the equivalent `handleNavigate` hamburger-catalog
 * case) and both always attach a dynamic `returnTo` — there's no second,
 * genuinely different origin the way recordings has "home" vs "analyze". So
 * it's unconditionally primary by name, same as growth/coach/analyze; its
 * render case below no longer passes `onBack` (AppChrome supplies the way
 * back now), and backHandler.ts's `backTarget` now sends hardware back Home
 * for it too, matching the other primary screens, instead of popping
 * through `returnTo`.
 *
 * "onboarding"/tutorial was investigated for the same flip and deliberately
 * LEFT catalog-only (primaryEligible: false) — see the reasoning on that
 * destination in src/nav/destinations.ts. Summary: the first-launch
 * walkthrough bypasses this Screen union/isPrimary entirely (a separate
 * top-level gate below, before renderScreen() is ever called), so only
 * re-entry (Settings' "Show tutorial" row / the hamburger catalog) would be
 * affected — and OnboardingScreen is a focused, no-onBack card carousel
 * where full AppChrome (letting a user tab away mid-walkthrough) would
 * undermine the point, without recordings' static returnTo values available
 * to represent an instance-predicate distinction (every reachable path
 * patches `returnTo` dynamically via the same `handleNavigate` code, so
 * there's nothing to key an instance check off of).
 *
 * Every other pushed screen (session, dynamics, watch-setup, onboarding,
 * detail, your-day, record, replay, advanced, home-design, avatar-capture)
 * either isn't in the registry at all or isn't primary-eligible, so it's
 * pushed by definition — same full-screen layout, same back affordance as
 * before this task (though watch-setup/onboarding/dashboard's `returnTo` is
 * still dynamic rather than hardcoded — see the Task N3 fix-round-1
 * comments on those cases; `dashboard`'s `returnTo` now only matters for
 * restoring the chain when something like `detail` is pushed from it, not
 * for rendering its own back button, since it no longer has one).
 */
const PRIMARY_SCREEN_NAMES: ReadonlySet<Screen["name"]> = new Set<
  Screen["name"]
>([
  "home",
  ...PRIMARY_ELIGIBLE_DESTINATIONS.filter((d) => d.id !== "recordings").map(
    (d) => d.screen.name,
  ),
]);

/** PRIMARY vs PUSHED is a predicate over the screen INSTANCE, not just its
 *  name — see PRIMARY_SCREEN_NAMES's comment for why "recordings" needs the
 *  extra `returnTo` check below rather than a blanket name match. */
function isPrimary(screen: Screen): boolean {
  return (
    PRIMARY_SCREEN_NAMES.has(screen.name) ||
    (screen.name === "recordings" && screen.returnTo === "home")
  );
}

/** The URL the app was opened with, for the web build (window.location) —
 *  overridable by tests. Native reads Linking below. */
function initialWebUrl(): string | null {
  if (Platform.OS !== "web") return null;
  const loc = (globalThis as { location?: { href?: string } }).location;
  return loc?.href ?? null;
}

interface AppProps {
  /** Test seam: the URL the app was launched with. */
  initialUrl?: string | null;
}

export default function App({ initialUrl }: AppProps = {}) {
  const [screen, setScreen] = useState<Screen>({ name: "home" });

  // In-app call invites (src/nav/callLink.ts). The code waits here until the
  // auth + onboarding gates below let a screen render, then opens Live Coach
  // in Call mode. Web: the path at load; native: the launch URL + later
  // `url` events (the app already open when the link is tapped).
  const [pendingCall, setPendingCall] = useState<ParsedCallLink | null>(() =>
    parseCallLink(initialUrl !== undefined ? initialUrl : initialWebUrl()),
  );
  // "Hey Google, start my journal" (src/nav/journalLink.ts): same waiting
  // pattern as pendingCall — mindshift://journal/start|stop holds here until
  // the gates below let a screen render, then opens Live Coach with the
  // action for the screen to execute (its own gates still apply there).
  const [pendingJournal, setPendingJournal] = useState<JournalLinkAction | null>(() =>
    parseJournalLink(initialUrl !== undefined ? initialUrl : initialWebUrl()),
  );
  useEffect(() => {
    if (Platform.OS === "web") return;
    let cancelled = false;
    Linking.getInitialURL()
      .then((url) => {
        const parsed = parseCallLink(url);
        if (parsed && !cancelled) setPendingCall(parsed);
        const journal = parseJournalLink(url);
        if (journal && !cancelled) setPendingJournal(journal);
      })
      .catch(() => {});
    const sub = Linking.addEventListener("url", ({ url }) => {
      const parsed = parseCallLink(url);
      if (parsed) setPendingCall(parsed);
      const journal = parseJournalLink(url);
      if (journal) setPendingJournal(journal);
    });
    return () => {
      cancelled = true;
      sub.remove();
    };
  }, []);

  // Task N3: AppChrome's hamburger catalog / account menu overlays own their
  // open/closed state internally (encapsulated, easier to test in
  // isolation) — but Android hardware-back needs to close THEM first,
  // before falling through to screen navigation or double-back-to-exit
  // (fix round 1, CRITICAL 1: back used to navigate/exit right underneath
  // an open overlay). `closeOverlays()` returns true iff it actually closed
  // something, so the back hook knows whether to swallow the press.
  const chromeRef = useRef<AppChromeHandle>(null);
  const closeChromeOverlays = () => chromeRef.current?.closeOverlays() ?? false;

  // Task N3: Android hardware-back — pushed screens pop to their launching
  // screen, Home double-back-exits with a toast hint. Pure decision logic
  // lives in src/nav/backHandler.ts (unit-tested there); this hook is just
  // the BackHandler/ToastAndroid wiring, a no-op off Android.
  useAndroidBackHandler(screen, setScreen, closeChromeOverlays);

  // Start listening to Firebase auth state once, on mount. Task N3 fix
  // round 1 (IMPORTANT 2): layoutStore.hydrate() was defined back in N1 but
  // never actually called anywhere — AppChrome's tab bar was silently
  // running on hardcoded defaults forever, never loading a saved layout.
  // Hydrate here too, once, on mount — before any primary screen (and its
  // tab bar) can render, so the fast local read (SecureStore/localStorage)
  // has every chance to land before the auth+onboarding gates below even
  // resolve, minimizing any default-then-real flash. hydrate() itself is
  // idempotent and respects its own `hydrated` flag (see layoutStore.ts).
  useEffect(() => {
    initAuth();
    void useLayoutStore.getState().hydrate();
    // Task N6: load the persisted selfie uri (if any) before the top bar's
    // avatar slot first renders — same rationale as layoutStore's hydrate
    // above, minimizing any no-photo-then-photo flash.
    void useAvatarStore.getState().hydrate();
  }, []);

  const user = useAuthStore((s) => s.user);
  const initializing = useAuthStore((s) => s.initializing);
  const signOut = useAuthStore((s) => s.signOut);
  const avatarUri = useAvatarStore((s) => s.uri);

  // N7 fix round 1 (IMPORTANT 3): avatarStore's persisted uri has no uid in
  // it and nothing cleared it on sign-out — on a shared device, account B
  // signing in would see account A's selfie in the top bar. Both real
  // sign-out entry points (the avatar menu's "Log out" and Settings' own
  // row, wired below) go through this one helper instead of calling
  // authStore's signOut() directly, reusing the same clear-state
  // +best-effort-file-delete path "Remove photo" already built
  // (avatarStore.removePhoto).
  const handleSignOut = () => {
    useAvatarStore.getState().removePhoto();
    void signOut();
  };

  // Task P3-7: has this account seen the first-launch onboarding walkthrough?
  // null = not checked yet (persisted read in flight); true/false once known.
  // Re-checked whenever the signed-in uid changes (e.g. a second account on
  // the same device gets its own first-launch tutorial).
  const [onboardingSeen, setOnboardingSeenState] = useState<boolean | null>(
    null,
  );
  useEffect(() => {
    if (!user) {
      setOnboardingSeenState(null);
      return;
    }
    let cancelled = false;
    getOnboardingSeen().then((seen) => {
      if (!cancelled) setOnboardingSeenState(seen);
    });
    return () => {
      cancelled = true;
    };
  }, [user?.uid]);

  // Once signed in and past onboarding, an invite opens Live Coach in Call
  // mode. The web URL is cleaned so a reload doesn't re-offer the call.
  const ready = Boolean(user) && onboardingSeen === true;
  useEffect(() => {
    if (!pendingCall || !ready) return;
    setPendingCall(null);
    setScreen({ name: "live-coach", joinCode: pendingCall.code, joinRole: pendingCall.role });
    if (Platform.OS === "web") {
      try {
        (globalThis as { history?: { replaceState?: (a: unknown, b: string, c: string) => void } }).history?.replaceState?.(
          {},
          "",
          "/",
        );
      } catch {
        // No history API: the URL just stays.
      }
    }
  }, [pendingCall, ready]);

  // Journal deep link: same gate-waiting as the call invite above. The screen
  // does the actual selecting/starting (and shows its gate messages when a
  // gate fails) — this only navigates and hands over the action.
  useEffect(() => {
    if (!pendingJournal || !ready) return;
    setPendingJournal(null);
    setScreen({ name: "live-coach", journalAction: pendingJournal });
  }, [pendingJournal, ready]);

  // Cold start: wait for the first auth-state resolution before deciding which
  // surface to show, so we never flash the wrong screen.
  if (initializing) {
    return (
      <SafeAreaProvider>
        <SafeAreaView
          style={[styles.container, styles.center]}
          testID="auth-loading"
        >
          <ActivityIndicator size="large" color="#4A90D9" />
        </SafeAreaView>
      </SafeAreaProvider>
    );
  }

  // Auth gate: an unauthenticated user only ever sees the login screen.
  if (!user) {
    return (
      <SafeAreaProvider>
        <SafeAreaView style={styles.container}>
          <LoginScreen />
        </SafeAreaView>
      </SafeAreaProvider>
    );
  }

  // Onboarding gate: hold the same loading spinner as the auth gate above
  // while the persisted seen-flag loads (fast, local), then show the
  // walkthrough full-screen — once — before any other app chrome.
  if (onboardingSeen === null) {
    return (
      <SafeAreaProvider>
        <SafeAreaView
          style={[styles.container, styles.center]}
          testID="onboarding-loading"
        >
          <ActivityIndicator size="large" color="#4A90D9" />
        </SafeAreaView>
      </SafeAreaProvider>
    );
  }
  if (onboardingSeen === false) {
    return (
      <SafeAreaProvider>
        <SafeAreaView style={styles.container}>
          <OnboardingScreen
            onFinish={() => {
              setOnboardingSeenState(true);
              void setOnboardingSeen(true);
            }}
          />
        </SafeAreaView>
      </SafeAreaProvider>
    );
  }

  const renderScreen = () => {
    switch (screen.name) {
      case "home":
        // Task N4: the box grid navigates through the same `handleNavigate`
        // callback AppChrome's tab bar and hamburger catalog use (defined
        // below, in scope by closure — see its own comment for why that's
        // safe despite the declaration order).
        return (
          <HomeScreen
            onNavigate={handleNavigate}
            onOpenYourDay={() => setScreen({ name: "your-day" })}
          />
        );
      case "live-coach":
        // No onBack: Live Coach is a PRIMARY screen (see PRIMARY_SCREEN_NAMES
        // above) — AppChrome already provides a way back to Home, so a
        // dedicated back button here would just duplicate it.
        return (
          <LiveCoachScreen
            joinCode={screen.joinCode ?? null}
            joinRole={screen.joinRole ?? "participant"}
            onJoinCodeConsumed={() => setScreen({ name: "live-coach" })}
            journalAction={screen.journalAction ?? null}
            onJournalActionConsumed={() => setScreen({ name: "live-coach" })}
            onReviewTranscript={(turns) => {
              // Hand the finished live conversation to the text tools, where
              // Get Suggestions / Analyze dynamics work off the loaded turns.
              useSessionStore.getState().loadTurns(turns);
              setScreen({ name: "session", returnTo: "home" });
            }}
          />
        );
      case "analyze":
        // No onBack — Analyze is PRIMARY (see PRIMARY_SCREEN_NAMES above);
        // same reasoning as Live Coach.
        return (
          <AnalyzeScreen
            onAnalyzeDynamics={(initialData, recordingId, cameFromRecorder) =>
              setScreen({
                name: "dynamics",
                initialData,
                recordingId,
                cameFromRecorder,
                returnTo: { name: "analyze" },
              })
            }
            onOpenRecordings={() =>
              setScreen({ name: "recordings", returnTo: "analyze" })
            }
            onRecordVideo={() => setScreen({ name: "record" })}
            onOpenTextTools={() =>
              setScreen({ name: "session", returnTo: "analyze" })
            }
          />
        );
      case "advanced":
        return (
          <AdvancedScreen
            // N7 fix round 1 (IMPORTANT 2): back lands wherever this was
            // actually launched from (any primary screen's avatar menu, or
            // the hamburger catalog), defaulting to Home when absent —
            // same dynamic-returnTo pattern as watch-setup/onboarding/
            // dashboard below.
            onBack={() => setScreen(screen.returnTo ?? { name: "home" })}
            // N7 fix round 1 re-review (IMPORTANT 2, second pass): every one
            // of these six outbound pushes used to hardcode a bare
            // `returnTo: { name: "advanced" }`, discarding whatever
            // `returnTo` THIS `advanced` screen was itself carrying (e.g.
            // opened from Live Coach's avatar menu). Passing `returnTo:
            // screen` wholesale — the same idiom `detail` already uses for
            // `dashboard` below — carries that chain through so backing all
            // the way out restores the real origin instead of resetting to
            // Home one hop early.
            onOpenDashboard={() =>
              setScreen({ name: "dashboard", returnTo: screen })
            }
            onSignOut={handleSignOut}
            onOpenReplay={(id) =>
              setScreen({
                name: "replay",
                recordingId: id,
                returnTo: screen,
              })
            }
            onOpenWatchSetup={() =>
              setScreen({ name: "watch-setup", returnTo: screen })
            }
            onOpenTutorial={() =>
              setScreen({ name: "onboarding", returnTo: screen })
            }
            onOpenHomeDesign={() =>
              setScreen({ name: "home-design", returnTo: screen })
            }
            onOpenPeople={() => setScreen({ name: "people", returnTo: screen })}
            onSetProfilePhoto={() =>
              setScreen({
                name: "avatar-capture",
                returnTo: screen,
              })
            }
          />
        );
      case "home-design":
        // N7 fix round 1 re-review: pop back to the WHOLE `advanced` screen
        // this was pushed from (with its own `returnTo` intact) rather than
        // a bare reset — see the Screen union's comment on this variant.
        return <HomeDesignScreen onBack={() => setScreen(screen.returnTo)} />;
      case "people":
        // People labeling: pops back to wherever it was launched from
        // (Settings' row, or the hamburger catalog from any primary screen).
        return (
          <PeopleScreen
            onBack={() => setScreen(screen.returnTo)}
            onOpenReplay={(id) =>
              setScreen({ name: "replay", recordingId: id, returnTo: { name: "advanced" } })
            }
          />
        );
      case "watch-setup":
        // Task N3 fix round 1: returns to wherever it was actually launched
        // from (Settings' own row, or now the hamburger catalog from any
        // primary screen) instead of always landing back in Settings.
        return <WatchSetupScreen onBack={() => setScreen(screen.returnTo)} />;
      case "onboarding":
        // Re-entry from Settings' "Show tutorial" row (or, since the
        // hamburger catalog fix, any primary screen). Doesn't touch the
        // persisted seen-flag — it's already true by the time this is
        // reachable — this is just a manual replay. Returns to its origin
        // (Task N3 fix round 1), not always Settings.
        return (
          <OnboardingScreen onFinish={() => setScreen(screen.returnTo)} />
        );
      case "session":
        // The text tools. Back returns to whichever screen pushed it (Analyze
        // or, after a live session's review handoff, Home). Narrow returnTo to
        // a concrete screen so the discriminated union stays exact (a bare
        // { name: returnTo } widens both variants).
        return (
          <SessionScreen
            onBack={() =>
              setScreen(
                screen.returnTo === "analyze"
                  ? { name: "analyze" as const }
                  : { name: "home" as const },
              )
            }
            onAnalyzeDynamics={() =>
              setScreen({
                name: "dynamics",
                returnTo: { name: "session", returnTo: screen.returnTo },
              })
            }
          />
        );
      case "dynamics":
        // Post-session analysis. initialData (from the upload flow) skips the
        // on-mount fetch. If a recording backs this analysis, it shows a Replay
        // entry point that pushes the ReplayScreen for that id.
        return (
          <DynamicsScreen
            onBack={() => setScreen(screen.returnTo)}
            initialData={screen.initialData}
            recordingId={screen.recordingId}
            cameFromRecorder={screen.cameFromRecorder}
            onReplay={(id) =>
              setScreen({
                name: "replay",
                recordingId: id,
                returnTo: screen.returnTo,
              })
            }
            onAttachSource={(id) =>
              setScreen({
                name: "replay",
                recordingId: id,
                returnTo: screen.returnTo,
                openAttach: true,
              })
            }
          />
        );
      case "recordings":
        // Task N3 fix round 1: `returnTo: "home"` makes this PRIMARY (see
        // isPrimary above) — AppChrome already provides a way back, so no
        // dedicated back button then. `returnTo: "analyze"` (pushed from
        // Analyze's own "past recordings" row) stays pushed, unchanged.
        return (
          <RecordingsScreen
            onBack={
              screen.returnTo === "analyze"
                ? () => setScreen({ name: "analyze" })
                : undefined
            }
            onSelectRecording={(id) =>
              setScreen({
                name: "replay",
                recordingId: id,
                returnTo: { name: "recordings", returnTo: screen.returnTo },
              })
            }
          />
        );
      case "growth":
        // No onBack — Growth is PRIMARY (see PRIMARY_SCREEN_NAMES above);
        // same reasoning as Live Coach/Analyze.
        return (
          <GrowthScreen
            onOpenRecording={(id) =>
              setScreen({
                name: "replay",
                recordingId: id,
                returnTo: { name: "growth" },
              })
            }
            onOpenRecordings={() =>
              setScreen({ name: "recordings", returnTo: "home" })
            }
          />
        );
      case "your-day":
        return (
          <YourDayScreen
            onBack={() => setScreen({ name: "home" })}
            onOpenReplay={(id) =>
              setScreen({
                name: "replay",
                recordingId: id,
                returnTo: { name: "your-day" },
              })
            }
          />
        );
      case "record":
        return (
          <RecordScreen
            onBack={() => setScreen({ name: "analyze" })}
            onComplete={(file) => {
              // Hand the recorded clip to the Analyze upload flow and return
              // there; AnalyzeScreen consumes it from the recorder store.
              useRecorderStore.getState().setPendingFile(file);
              setScreen({ name: "analyze" });
            }}
          />
        );
      case "avatar-capture":
        // Task N6: both cancel and a successful save return to whichever
        // screen launched the capture flow (the avatar menu or the Settings
        // row can be opened from any primary screen).
        return (
          <AvatarCaptureScreen
            onBack={() => setScreen(screen.returnTo)}
            onSaved={() => setScreen(screen.returnTo)}
          />
        );
      case "replay":
        return (
          <ReplayScreen
            recordingId={screen.recordingId}
            onBack={() => setScreen(screen.returnTo)}
            initialAttachOpen={screen.openAttach}
          />
        );
      case "dashboard":
        // No onBack (2026-08-19 primary-eligible-expand) — Dashboard is now
        // PRIMARY (see PRIMARY_SCREEN_NAMES above), same reasoning as
        // Live Coach/Analyze/Growth: AppChrome already provides a way back,
        // so a dedicated back button would be redundant. `screen.returnTo`
        // is still carried and still matters for `detail` below: pushing
        // "detail" from here hands it this WHOLE dashboard screen (with its
        // own `returnTo`) so ITS back restores the chain correctly, even
        // though dashboard itself no longer renders a back button.
        return (
          <TherapistDashboard
            onSelectSession={(id) =>
              setScreen({ name: "detail", sessionId: id, returnTo: screen })
            }
          />
        );
      case "detail":
        return (
          <SessionDetail
            sessionId={screen.sessionId}
            onBack={() => setScreen(screen.returnTo)}
          />
        );
    }
  };

  // Task N3: hand a destination straight to setScreen for every destination
  // whose DestScreen shape already matches its Screen counterpart exactly
  // (verified compile-time AND by hand — see destinations.ts's DestScreen
  // comment). watch-setup/dashboard/onboarding/advanced are the exceptions
  // (Task N3 fix round 1, IMPORTANT 4; N7 fix round 1, IMPORTANT 2 added
  // "advanced"): the registry can't know what screen the catalog was opened
  // FROM (it's static data), so their `returnTo` is attached here instead,
  // from whatever screen is current right now. Each check narrows `dest`'s
  // type via an early return, so the final `setScreen(dest)` below stays a
  // straight, uncast hand-off for everything else.
  const handleNavigate = (dest: DestScreen) => {
    if (dest.name === "watch-setup") {
      setScreen({ ...dest, returnTo: screen });
      return;
    }
    if (dest.name === "dashboard") {
      setScreen({ ...dest, returnTo: screen });
      return;
    }
    if (dest.name === "onboarding") {
      setScreen({ ...dest, returnTo: screen });
      return;
    }
    if (dest.name === "advanced") {
      setScreen({ ...dest, returnTo: screen });
      return;
    }
    if (dest.name === "people") {
      setScreen({ ...dest, returnTo: screen });
      return;
    }
    setScreen(dest);
  };

  // AppChrome's wordmark tap and every pushed screen's new Home affordance
  // (PushedScreenChrome, below) both go through this one handler — a pushed
  // screen's Home tap must land exactly where the wordmark tap already does,
  // not a separate reimplementation of "go home".
  const goHome = () => setScreen({ name: "home" });

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.container}>
        {/* Sits above every screen: a downloaded OTA update surfaces a subtle
            "restart to apply" bar here, and stays out of the way otherwise. */}
        <UpdateBanner />
        {isPrimary(screen) ? (
          <AppChrome
            ref={chromeRef}
            screenName={screen.name}
            onNavigate={handleNavigate}
            onGoHome={goHome}
            onSignOut={handleSignOut}
            user={user}
            avatarUri={avatarUri}
          >
            {renderScreen()}
          </AppChrome>
        ) : (
          // Additive only (see PushedScreenChrome's own comment): a thin
          // "Home" tap target above the screen's own content, alongside its
          // existing back button — not a replacement for it. Does not touch
          // isPrimary/PRIMARY_SCREEN_NAMES or any screen's own onBack/
          // backHandler.ts wiring.
          <PushedScreenChrome onGoHome={goHome}>
            {renderScreen()}
          </PushedScreenChrome>
        )}
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F9FAFB",
  },
  center: {
    alignItems: "center",
    justifyContent: "center",
  },
});
