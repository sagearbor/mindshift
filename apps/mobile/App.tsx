import React, { useState } from "react";
import { SafeAreaView, View, Text, TouchableOpacity, StyleSheet } from "react-native";
import SessionScreen from "./src/screens/SessionScreen";
import TherapistDashboard from "./src/screens/TherapistDashboard";
import SessionDetail from "./src/screens/SessionDetail";
import LiveCoachScreen from "./src/screens/LiveCoachScreen";

type Screen =
  | { name: "session" }
  | { name: "live-coach" }
  | { name: "dashboard" }
  | { name: "detail"; sessionId: string };

export default function App() {
  const [screen, setScreen] = useState<Screen>({ name: "session" });

  const renderScreen = () => {
    switch (screen.name) {
      case "session":
        return <SessionScreen />;
      case "live-coach":
        return <LiveCoachScreen />;
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
      {/* Bottom tab bar */}
      {screen.name !== "detail" && (
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
