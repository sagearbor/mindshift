import React from "react";
import { View, Text, TouchableOpacity, ScrollView, StyleSheet } from "react-native";

import { DESTINATIONS, type DestScreen } from "../nav/destinations";
import { getIcon, CHROME_ICONS } from "./icons";

const Close = CHROME_ICONS.close;

interface DestinationCatalogProps {
  onSelect: (screen: DestScreen) => void;
  onClose: () => void;
}

/**
 * The hamburger's full destination catalog (Task N3) — the safety net that
 * makes tab/box customization safe (P3-9 RESOLVED): EVERY registry
 * destination is always listed here, regardless of what the user has (or
 * hasn't) put on the tab bar or home boxes. "Home" itself isn't a registry
 * destination — see AppChrome.tsx's comment on the wordmark tap, which is
 * how Home stays reachable instead.
 *
 * A plain absolutely-positioned full-screen overlay rather than RN's
 * <Modal> — Modal's portal behavior is inconsistent on react-native-web, and
 * this app renders the same source on RN + web from one codebase.
 */
export default function DestinationCatalog({
  onSelect,
  onClose,
}: DestinationCatalogProps) {
  return (
    <View
      style={styles.overlay}
      testID="chrome-catalog"
      // Full-screen overlay covering everything behind it — tell assistive
      // tech this is the modal now (iOS VoiceOver's accessibilityViewIsModal;
      // AppChrome separately marks the background no-hide-descendants for
      // Android/general TalkBack).
      accessibilityViewIsModal
    >
      <View style={styles.header}>
        <Text style={styles.title}>Everything</Text>
        <TouchableOpacity
          testID="chrome-catalog-close"
          accessibilityRole="button"
          accessibilityLabel="Close"
          style={styles.closeButton}
          onPress={onClose}
          hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
        >
          <Close size={22} />
        </TouchableOpacity>
      </View>
      <ScrollView contentContainerStyle={styles.list}>
        {DESTINATIONS.map((dest) => {
          const Icon = getIcon(dest.iconId);
          return (
            <TouchableOpacity
              key={dest.id}
              testID={`chrome-catalog-item-${dest.id}`}
              accessibilityRole="button"
              accessibilityLabel={dest.title}
              style={styles.row}
              onPress={() => onSelect(dest.screen)}
            >
              <Icon size={22} />
              <Text style={styles.rowTitle}>{dest.title}</Text>
            </TouchableOpacity>
          );
        })}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "#F9FAFB",
    zIndex: 30,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 20,
    paddingTop: 24,
    paddingBottom: 12,
  },
  title: {
    fontSize: 20,
    fontWeight: "700",
    color: "#111827",
  },
  closeButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center",
  },
  list: {
    paddingHorizontal: 20,
    paddingBottom: 24,
    gap: 8,
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 14,
    minHeight: 52,
    borderRadius: 12,
    paddingHorizontal: 14,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  rowTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#111827",
  },
});
