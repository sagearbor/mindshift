import React from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

import HeroWipe from "../components/HeroWipe";
import HomeBoxGrid from "../components/HomeBoxGrid";
import { useLayoutStore } from "../store/layoutStore";
import type { DestScreen } from "../nav/destinations";

/**
 * Home (Task N4 of P3-10). The old fixed "two huge mode cards" main area is
 * gone — per the owner's locked architecture (P3-9 RESOLVED) the main area
 * is now `layoutStore.homeBoxes` (0–4 user-arranged icon+label shortcuts,
 * see HomeBoxGrid), configurable via Settings → "Home screen design"
 * (Task N5). Every destination the old cards covered (Live Coach, Analyze,
 * Recordings, Growth) stays reachable even when a box doesn't cover it —
 * via the tab bar (AppChrome), the hamburger's full catalog, or a box —
 * see __tests__/App.test.tsx's "migration honesty" test.
 *
 * "Your day" (the day timeline) has no registry destination of its own —
 * see nav/destinations.ts's DestId comment for why — so it can't be a
 * configurable box or tab. It keeps its own direct link below the box grid
 * so that affordance isn't silently lost in this rework.
 */
interface HomeScreenProps {
  /** Hand a tapped box's destination straight to App.tsx's setScreen — the
   *  same callback AppChrome's tab bar and hamburger catalog already use. */
  onNavigate: (screen: DestScreen) => void;
  onOpenYourDay: () => void;
}

export default function HomeScreen({ onNavigate, onOpenYourDay }: HomeScreenProps) {
  const homeBoxes = useLayoutStore((s) => s.homeBoxes);

  return (
    <View style={styles.container} testID="home-screen">
      {/* Web-only hero banner (Task P3-4b) — renders nothing on native. */}
      <HeroWipe />

      <View style={styles.gridWrap}>
        <HomeBoxGrid boxes={homeBoxes} onNavigate={onNavigate} />
      </View>

      <TouchableOpacity
        testID="home-your-day-link"
        accessibilityRole="button"
        style={styles.yourDayRow}
        onPress={onOpenYourDay}
      >
        <Text style={styles.yourDayRowText}>☀ Your day</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 24,
    paddingBottom: 20,
    backgroundColor: "#F9FAFB",
  },
  gridWrap: {
    flex: 1,
  },
  yourDayRow: {
    marginTop: 16,
    minHeight: 52,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#FFFFFF",
    alignItems: "center",
    justifyContent: "center",
  },
  yourDayRowText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#4A90D9",
  },
});
