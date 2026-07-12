import React, { useEffect, useState } from "react";
import {
  SafeAreaView,
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import SessionScreen from "./src/screens/SessionScreen";
import TherapistDashboard from "./src/screens/TherapistDashboard";
import SessionDetail from "./src/screens/SessionDetail";
import DynamicsScreen from "./src/screens/DynamicsScreen";
import ReplayScreen from "./src/screens/ReplayScreen";
import RecordScreen from "./src/screens/RecordScreen";
import RecordingsScreen from "./src/screens/RecordingsScreen";
import LiveCoachScreen from "./src/screens/LiveCoachScreen";
import LoginScreen from "./src/screens/LoginScreen";
import { useAuthStore, initAuth } from "./src/store/authStore";
import { useSessionStore } from "./src/store/sessionStore";
import { useRecorderStore } from "./src/store/recorderStore";
import type { AnalyzeResult } from "./src/api/client";

type Screen =
  | { name: "session" }
  | { name: "live-coach" }
  | { name: "dashboard" }
  | { name: "detail"; sessionId: string }
  // Pushed on top of the Session tab (like "detail"): the tab bar is hidden and
  // onBack returns to the Session screen. Not itself a tab.
  //
  // `initialData` is a ready-made analysis handed over from the recording-upload
  // flow — when present, DynamicsScreen renders it directly instead of
  // re-POSTing /analyze (the transcript came from the upload, not the store's
  // text path). Absent for the normal "Analyze dynamics" button, which analyzes
  // the store transcript on mount as before.
  //
  // `recordingId` is the server-assigned id of a *stored* recording (set only
  // when the upload flow's consent+store both landed as true); undefined
  // otherwise. Carried through so DynamicsScreen can offer a Replay affordance
  // that pushes the ReplayScreen for that id.
  //
  // `cameFromRecorder` marks an analysis whose file was just recorded in-app
  // (and saved to the camera roll). When true AND the recording was stored,
  // DynamicsScreen offers the "attach HD source later" popup.
  | {
      name: "dynamics";
      initialData?: AnalyzeResult;
      recordingId?: string | null;
      cameFromRecorder?: boolean;
    }
  // Stored-recordings list, pushed over the Session tab.
  | { name: "recordings" }
  // In-app 480p video recording, pushed over the Session tab; the tab bar is
  // hidden and onBack returns to Session. On success it hands the recorded file
  // to the Session upload flow (via the recorder store).
  | { name: "record" }
  // Media replay with the synced heat graph. Pushed; `returnTo` records where
  // the user came from so Back lands them there. `openAttach` opens the
  // attach-HD-source input immediately (from the Dynamics popup's "Attach link
  // now").
  | {
      name: "replay";
      recordingId: string;
      returnTo: "session" | "recordings";
      openAttach?: boolean;
    };

export default function App() {
  const [screen, setScreen] = useState<Screen>({ name: "session" });

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
      <SafeAreaView style={[styles.container, styles.center]} testID="auth-loading">
        <ActivityIndicator size="large" color="#4A90D9" />
      </SafeAreaView>
    );
  }

  // Auth gate: an unauthenticated user only ever sees the login screen.
  if (!user) {
    return (
      <SafeAreaView style={styles.container}>
        <LoginScreen />
      </SafeAreaView>
    );
  }

  const renderScreen = () => {
    switch (screen.name) {
      case "session":
        return (
          <SessionScreen
            onAnalyzeDynamics={(initialData, recordingId, cameFromRecorder) =>
              setScreen({
                name: "dynamics",
                initialData,
                recordingId,
                cameFromRecorder,
              })
            }
            onOpenRecordings={() => setScreen({ name: "recordings" })}
            onRecordVideo={() => setScreen({ name: "record" })}
          />
        );
      case "dynamics":
        // Post-session analysis, pushed over the Session tab; back returns there.
        // initialData (from the upload flow) skips the on-mount fetch. If a
        // recording backs this analysis (recordingId present, or one surfaces
        // from DynamicsScreen's own fetched context), it shows a Replay entry
        // point that pushes the ReplayScreen for that id.
        return (
          <DynamicsScreen
            onBack={() => setScreen({ name: "session" })}
            initialData={screen.initialData}
            recordingId={screen.recordingId}
            cameFromRecorder={screen.cameFromRecorder}
            onReplay={(id) =>
              setScreen({ name: "replay", recordingId: id, returnTo: "session" })
            }
            onAttachSource={(id) =>
              setScreen({
                name: "replay",
                recordingId: id,
                returnTo: "session",
                openAttach: true,
              })
            }
          />
        );
      case "recordings":
        return (
          <RecordingsScreen
            onBack={() => setScreen({ name: "session" })}
            onSelectRecording={(id) =>
              setScreen({
                name: "replay",
                recordingId: id,
                returnTo: "recordings",
              })
            }
          />
        );
      case "record":
        return (
          <RecordScreen
            onBack={() => setScreen({ name: "session" })}
            onComplete={(file) => {
              // Hand the recorded clip to the Session upload flow and return to
              // the Session screen, which consumes it from the recorder store.
              useRecorderStore.getState().setPendingFile(file);
              setScreen({ name: "session" });
            }}
          />
        );
      case "replay": {
        // Narrow returnTo to a concrete screen so the discriminated union stays
        // exact (a bare { name: returnTo } widens both variants).
        const back =
          screen.returnTo === "recordings"
            ? { name: "recordings" as const }
            : { name: "session" as const };
        return (
          <ReplayScreen
            recordingId={screen.recordingId}
            onBack={() => setScreen(back)}
            initialAttachOpen={screen.openAttach}
          />
        );
      }
      case "live-coach":
        return (
          <LiveCoachScreen
            onReviewTranscript={(turns) => {
              // Hand the finished live conversation to the async-review store
              // and jump to the Session screen, where Get Suggestions works
              // off the loaded turns. Mirrors the dashboard's onSelectSession.
              useSessionStore.getState().loadTurns(turns);
              setScreen({ name: "session" });
            }}
          />
        );
      case "dashboard":
        return (
          <TherapistDashboard
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
    <SafeAreaView style={styles.container}>
      {renderScreen()}
      {/* Bottom tab bar — hidden on pushed sub-screens (detail, dynamics,
          recordings, record, replay). */}
      {screen.name !== "detail" &&
        screen.name !== "dynamics" &&
        screen.name !== "recordings" &&
        screen.name !== "record" &&
        screen.name !== "replay" && (
        <View style={styles.tabBar}>
          <TouchableOpacity
            testID="tab-session"
            style={[styles.tab, screen.name === "session" && styles.tabActive]}
            onPress={() => setScreen({ name: "session" })}
          >
            <Text
              style={[
                styles.tabText,
                screen.name === "session" && styles.tabTextActive,
              ]}
            >
              Session
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            testID="tab-live-coach"
            style={[styles.tab, screen.name === "live-coach" && styles.tabActive]}
            onPress={() => setScreen({ name: "live-coach" })}
          >
            <Text
              style={[
                styles.tabText,
                screen.name === "live-coach" && styles.tabTextActive,
              ]}
            >
              Live Coach
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            testID="tab-dashboard"
            style={[styles.tab, screen.name === "dashboard" && styles.tabActive]}
            onPress={() => setScreen({ name: "dashboard" })}
          >
            <Text
              style={[
                styles.tabText,
                screen.name === "dashboard" && styles.tabTextActive,
              ]}
            >
              Dashboard
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            testID="tab-sign-out"
            style={styles.tab}
            onPress={() => {
              void signOut();
            }}
          >
            <Text style={styles.tabText}>Sign Out</Text>
          </TouchableOpacity>
        </View>
      )}
    </SafeAreaView>
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
  tabBar: {
    flexDirection: "row",
    borderTopWidth: 1,
    borderTopColor: "#E5E7EB",
    backgroundColor: "#FFFFFF",
  },
  tab: {
    flex: 1,
    paddingVertical: 12,
    alignItems: "center",
  },
  tabActive: {
    borderTopWidth: 2,
    borderTopColor: "#4A90D9",
  },
  tabText: {
    fontSize: 14,
    color: "#6B7280",
  },
  tabTextActive: {
    color: "#4A90D9",
    fontWeight: "600",
  },
});
