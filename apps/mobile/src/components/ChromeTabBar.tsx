import React from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

import type { Screen } from "../../App";
import { getDestination, type DestScreen } from "../nav/destinations";
import { useLayoutStore } from "../store/layoutStore";
import { getIcon } from "./icons";

interface ChromeTabBarProps {
  /** Current screen name, to highlight the matching tab (pushed screens
   *  match no tab, so every tab just renders inactive). */
  screenName: Screen["name"];
  onNavigate: (screen: DestScreen) => void;
  importantForAccessibility?: "auto" | "no-hide-descendants";
}

/** The bottom tab bar (layoutStore.tabSlots; hidden at 0 slots). Shared by
 *  AppChrome (primary screens) and PushedScreenChrome (pushed screens) so
 *  every screen reachable from the app keeps the tabs. */
export default function ChromeTabBar({
  screenName,
  onNavigate,
  importantForAccessibility = "auto",
}: ChromeTabBarProps) {
  const tabSlots = useLayoutStore((s) => s.tabSlots);
  if (tabSlots.length === 0) return null;
  return (
    <View
      style={styles.tabBar}
      testID="chrome-tab-bar"
      importantForAccessibility={importantForAccessibility}
    >
      {tabSlots.map((id) => {
        const dest = getDestination(id);
        if (!dest) return null; // stale persisted id — drop it silently
        const Icon = getIcon(dest.iconId);
        const active = dest.screen.name === screenName;
        return (
          <TouchableOpacity
            key={id}
            testID={`chrome-tab-${id}`}
            accessibilityRole="button"
            accessibilityLabel={dest.title}
            accessibilityState={{ selected: active }}
            style={styles.tab}
            onPress={() => onNavigate(dest.screen)}
          >
            <Icon size={22} color={active ? "#4A90D9" : "#6B7280"} />
            <Text
              style={[styles.tabLabel, active && styles.tabLabelActive]}
              numberOfLines={1}
            >
              {dest.title}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    flexDirection: "row",
    borderTopWidth: 1,
    borderTopColor: "#E5E7EB",
    backgroundColor: "#FFFFFF",
    paddingTop: 8,
    paddingBottom: 12,
    paddingHorizontal: 8,
  },
  tab: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 2,
    paddingHorizontal: 4,
  },
  tabLabel: {
    fontSize: 11,
    fontWeight: "600",
    color: "#6B7280",
  },
  tabLabelActive: {
    color: "#4A90D9",
  },
});
