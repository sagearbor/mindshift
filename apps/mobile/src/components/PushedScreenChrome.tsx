import React from "react";
import { View, TouchableOpacity, StyleSheet } from "react-native";

import { CHROME_ICONS } from "./icons";

const HomeIcon = CHROME_ICONS.home;

interface PushedScreenChromeProps {
  /** Same handler AppChrome's wordmark tap already uses
   *  (`() => setScreen({ name: "home" })` in App.tsx) — reused as-is, not
   *  reimplemented, so this affordance always lands wherever the wordmark
   *  tap does. */
  onGoHome: () => void;
  children: React.ReactNode;
}

/**
 * A thin, always-present "Home" tap target rendered ABOVE every PUSHED
 * screen (the `else` branch of App.tsx's `isPrimary(screen) ? ... :
 * renderScreen()`). Purely additive: it sits alongside each screen's own
 * back button, not in place of it — every screen's own "← Back"/"‹ Back"
 * stays exactly where it already is, unchanged.
 *
 * Deliberately minimal — a utility affordance, not a second chrome bar
 * competing with each screen's own header — so it's a single small row with
 * just a Home icon, following AppChrome's topBar spacing/color conventions.
 */
export default function PushedScreenChrome({
  onGoHome,
  children,
}: PushedScreenChromeProps) {
  return (
    <View style={styles.container} testID="pushed-chrome">
      <View style={styles.bar}>
        <TouchableOpacity
          testID="pushed-chrome-home-button"
          accessibilityRole="button"
          accessibilityLabel="Home"
          style={styles.homeButton}
          onPress={onGoHome}
          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        >
          <HomeIcon size={20} />
        </TouchableOpacity>
      </View>
      <View style={styles.content}>{children}</View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  bar: {
    flexDirection: "row",
    justifyContent: "flex-end",
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: 4,
  },
  homeButton: {
    width: 36,
    height: 36,
    alignItems: "center",
    justifyContent: "center",
  },
  content: {
    flex: 1,
  },
});
