import React, { useEffect, useState } from "react";
import { ActivityIndicator, StyleSheet } from "react-native";
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
import WatchSetupScreen from "./src/screens/WatchSetupScreen";
import SessionScreen from "./src/screens/SessionScreen";
import TherapistDashboard from "./src/screens/TherapistDashboard";
import SessionDetail from "./src/screens/SessionDetail";
import DynamicsScreen from "./src/screens/DynamicsScreen";
import ReplayScreen from "./src/screens/ReplayScreen";
import RecordScreen from "./src/screens/RecordScreen";
import RecordingsScreen from "./src/screens/RecordingsScreen";
import YourDayScreen from "./src/screens/YourDayScreen";
import GrowthScreen from "./src/screens/GrowthScreen";
import LiveCoachScreen from "./src/screens/LiveCoachScreen";
import LoginScreen from "./src/screens/LoginScreen";
import OnboardingScreen from "./src/screens/OnboardingScreen";
import UpdateBanner from "./src/components/UpdateBanner";
import AppChrome from "./src/components/AppChrome";
import { useAndroidBackHandler } from "./src/nav/useAndroidBackHandler";
import type { DestScreen } from "./src/nav/destinations";
import { useAuthStore, initAuth } from "./src/store/authStore";
import { useSessionStore } from "./src/store/sessionStore";
import { useRecorderStore } from "./src/store/recorderStore";
import { getOnboardingSeen, setOnboardingSeen } from "./src/utils/onboardingStorage";
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
  | { name: "live-coach" }
  // The Analyze mode hub: record / upload / link + relationship context.
  | { name: "analyze" }
  // Everything that doesn't fit the two modes: Settings (dashboard, watch
  // setup, voice profile, About, log out).
  | { name: "advanced" }
  // Phase 3 Slice 1: install the watch app + redeem its pairing code.
  // Pushed from Settings' "Set up your watch" row; returns there.
  | { name: "watch-setup" }
  // Task P3-7: the first-launch onboarding walkthrough, re-entered from
  // Settings' "Show tutorial" row; returns there. (The auto-shown-once
  // launch of this same screen is a separate top-level gate below, not part
  // of this pushed-screen union — see `onboardingSeen`.)
  | { name: "onboarding" }
  // The text tools (paste/type a transcript, suggestions). Pushed from
  // Analyze and from Live Coach's post-session review handoff.
  | { name: "session"; returnTo: SessionReturn }
  | { name: "dashboard" }
  | { name: "detail"; sessionId: string }
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
 * recordings, growth).
 *
 * That mechanically EXCLUDES "recordings" from this set even though it IS
 * primary-eligible for tab/box slots: its Screen/DestScreen shape carries a
 * required `returnTo`, meaning it can only be reached WITH a known
 * "came from" screen — exactly the definition of PUSHED. So it keeps its
 * existing dedicated back button (returning to `returnTo`) completely
 * unchanged, whether reached from Home, Analyze, or a configured tab/box
 * slot (which uses the registry's own default,
 * `{ name: "recordings", returnTo: "home" }`).
 *
 * Every other pushed screen (session, dynamics, watch-setup, onboarding,
 * dashboard, detail, your-day, record, replay, advanced) either isn't in the
 * registry at all or isn't primary-eligible, so it's pushed by definition —
 * same full-screen layout, same back affordance as before this task.
 */
const PRIMARY_SCREEN_NAMES: ReadonlySet<Screen["name"]> = new Set([
  "home",
  "live-coach",
  "analyze",
  "growth",
]);

export default function App() {
  const [screen, setScreen] = useState<Screen>({ name: "home" });

  // Task N3: Android hardware-back — pushed screens pop to their launching
  // screen, Home double-back-exits with a toast hint. Pure decision logic
  // lives in src/nav/backHandler.ts (unit-tested there); this hook is just
  // the BackHandler/ToastAndroid wiring, a no-op off Android.
  useAndroidBackHandler(screen, setScreen);

  // Start listening to Firebase auth state once, on mount.
  useEffect(() => {
    initAuth();
  }, []);

  const user = useAuthStore((s) => s.user);
  const initializing = useAuthStore((s) => s.initializing);
  const signOut = useAuthStore((s) => s.signOut);

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
        return (
          <HomeScreen
            onLiveCoach={() => setScreen({ name: "live-coach" })}
            onAnalyze={() => setScreen({ name: "analyze" })}
            onOpenRecordings={() =>
              setScreen({ name: "recordings", returnTo: "home" })
            }
            onOpenYourDay={() => setScreen({ name: "your-day" })}
            onOpenGrowth={() => setScreen({ name: "growth" })}
          />
        );
      case "live-coach":
        // No onBack: Live Coach is a PRIMARY screen (see PRIMARY_SCREEN_NAMES
        // above) — AppChrome already provides a way back to Home, so a
        // dedicated back button here would just duplicate it.
        return (
          <LiveCoachScreen
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
            onBack={() => setScreen({ name: "home" })}
            onOpenDashboard={() => setScreen({ name: "dashboard" })}
            onSignOut={() => {
              void signOut();
            }}
            onOpenReplay={(id) =>
              setScreen({
                name: "replay",
                recordingId: id,
                returnTo: { name: "advanced" },
              })
            }
            onOpenWatchSetup={() => setScreen({ name: "watch-setup" })}
            onOpenTutorial={() => setScreen({ name: "onboarding" })}
          />
        );
      case "watch-setup":
        return <WatchSetupScreen onBack={() => setScreen({ name: "advanced" })} />;
      case "onboarding":
        // Re-entry from Settings' "Show tutorial" row. Doesn't touch the
        // persisted seen-flag — it's already true by the time this is
        // reachable — this is just a manual replay.
        return (
          <OnboardingScreen onFinish={() => setScreen({ name: "advanced" })} />
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
        return (
          <RecordingsScreen
            onBack={() =>
              setScreen(
                screen.returnTo === "analyze"
                  ? { name: "analyze" as const }
                  : { name: "home" as const },
              )
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
      case "replay":
        return (
          <ReplayScreen
            recordingId={screen.recordingId}
            onBack={() => setScreen(screen.returnTo)}
            initialAttachOpen={screen.openAttach}
          />
        );
      case "dashboard":
        return (
          <TherapistDashboard
            onBack={() => setScreen({ name: "advanced" })}
            onSelectSession={(id) => setScreen({ name: "detail", sessionId: id })}
          />
        );
      case "detail":
        return (
          <SessionDetail
            sessionId={screen.sessionId}
            onBack={() => setScreen({ name: "dashboard" })}
          />
        );
    }
  };

  // Task N3: hand a destination straight to setScreen — DestScreen's shape
  // is verified (compile-time AND by hand) to match Screen exactly for every
  // variant it defines, so this needs no cast. See destinations.ts's
  // DestScreen comment.
  const handleNavigate = (dest: DestScreen) => setScreen(dest);
  const isPrimary = PRIMARY_SCREEN_NAMES.has(screen.name);

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.container}>
        {/* Sits above every screen: a downloaded OTA update surfaces a subtle
            "restart to apply" bar here, and stays out of the way otherwise. */}
        <UpdateBanner />
        {isPrimary ? (
          <AppChrome
            screenName={screen.name}
            onNavigate={handleNavigate}
            onGoHome={() => setScreen({ name: "home" })}
            onSignOut={() => void signOut()}
            user={user}
          >
            {renderScreen()}
          </AppChrome>
        ) : (
          renderScreen()
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
