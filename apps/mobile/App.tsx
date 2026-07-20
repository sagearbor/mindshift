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
import SessionScreen from "./src/screens/SessionScreen";
import TherapistDashboard from "./src/screens/TherapistDashboard";
import SessionDetail from "./src/screens/SessionDetail";
import DynamicsScreen from "./src/screens/DynamicsScreen";
import ReplayScreen from "./src/screens/ReplayScreen";
import RecordScreen from "./src/screens/RecordScreen";
import RecordingsScreen from "./src/screens/RecordingsScreen";
import YourDayScreen from "./src/screens/YourDayScreen";
import LiveCoachScreen from "./src/screens/LiveCoachScreen";
import LoginScreen from "./src/screens/LoginScreen";
import UpdateBanner from "./src/components/UpdateBanner";
import { useAuthStore, initAuth } from "./src/store/authStore";
import { useSessionStore } from "./src/store/sessionStore";
import { useRecorderStore } from "./src/store/recorderStore";
import type { AnalyzeResult } from "./src/api/client";

// --- Two-mode navigation -----------------------------------------------------
// The home screen is a radically simple choice between the app's two modes:
// Live Coach (real-time earbud coaching) and Analyze a Conversation
// (everything after-the-fact: record / upload / link / past recordings).
// Everything else lives behind the small Advanced corner affordance.
//
// Navigation stays the same hand-rolled screen union as before (no nav lib):
// every non-home screen is "pushed" and carries enough state to get back.

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
  | { name: "your-day" };

type Screen =
  | { name: "home" }
  | { name: "live-coach" }
  // The Analyze mode hub: record / upload / link + relationship context.
  | { name: "analyze" }
  // Everything that doesn't fit the two modes (dashboard, sign out).
  | { name: "advanced" }
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

export default function App() {
  const [screen, setScreen] = useState<Screen>({ name: "home" });

  // Start listening to Firebase auth state once, on mount.
  useEffect(() => {
    initAuth();
  }, []);

  const user = useAuthStore((s) => s.user);
  const initializing = useAuthStore((s) => s.initializing);
  const signOut = useAuthStore((s) => s.signOut);

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
            onOpenAdvanced={() => setScreen({ name: "advanced" })}
          />
        );
      case "live-coach":
        return (
          <LiveCoachScreen
            onBack={() => setScreen({ name: "home" })}
            onReviewTranscript={(turns) => {
              // Hand the finished live conversation to the text tools, where
              // Get Suggestions / Analyze dynamics work off the loaded turns.
              useSessionStore.getState().loadTurns(turns);
              setScreen({ name: "session", returnTo: "home" });
            }}
          />
        );
      case "analyze":
        return (
          <AnalyzeScreen
            onBack={() => setScreen({ name: "home" })}
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
          />
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

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.container}>
        {/* Sits above every screen: a downloaded OTA update surfaces a subtle
            "restart to apply" bar here, and stays out of the way otherwise. */}
        <UpdateBanner />
        {renderScreen()}
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
