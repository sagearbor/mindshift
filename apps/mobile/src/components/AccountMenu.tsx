import React from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

import type { AvatarUser } from "./Avatar";

interface AccountMenuProps {
  user: AvatarUser | null;
  onOpenSettings: () => void;
  /** Open the selfie-capture flow (Task N6 of P3-10 — "Set profile photo"). */
  onSetProfilePhoto: () => void;
  onSignOut: () => void;
  onClose: () => void;
}

/**
 * The avatar's compact account menu (Task N3, "Set profile photo" added in
 * Task N6): a "signed in as" line, Set profile photo, Settings, and Log out
 * — wired to the same signOut App.tsx already uses. A small anchored panel
 * plus a full-screen transparent backdrop that closes it on an outside tap,
 * not RN's <Modal> (see DestinationCatalog.tsx's comment on why: RN+web
 * from one codebase).
 */
export default function AccountMenu({
  user,
  onOpenSettings,
  onSetProfilePhoto,
  onSignOut,
  onClose,
}: AccountMenuProps) {
  const label = user?.email || user?.displayName || "this account";

  return (
    <>
      <TouchableOpacity
        testID="chrome-account-backdrop"
        accessibilityLabel="Close account menu"
        style={styles.backdrop}
        activeOpacity={1}
        onPress={onClose}
      />
      <View
        style={styles.panel}
        testID="chrome-account-menu"
        accessibilityViewIsModal
      >
        <Text style={styles.signedIn} testID="chrome-account-email">
          Signed in as {label}
        </Text>
        <TouchableOpacity
          testID="chrome-account-photo"
          accessibilityRole="button"
          style={styles.row}
          onPress={onSetProfilePhoto}
        >
          <Text style={styles.rowTitle}>Set profile photo</Text>
        </TouchableOpacity>
        <TouchableOpacity
          testID="chrome-account-settings"
          accessibilityRole="button"
          style={styles.row}
          onPress={onOpenSettings}
        >
          <Text style={styles.rowTitle}>Settings</Text>
        </TouchableOpacity>
        <TouchableOpacity
          testID="chrome-account-sign-out"
          accessibilityRole="button"
          style={styles.row}
          onPress={onSignOut}
        >
          <Text style={[styles.rowTitle, styles.signOutText]}>Log out</Text>
        </TouchableOpacity>
      </View>
    </>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 30,
  },
  panel: {
    position: "absolute",
    // AppChrome's top bar is 64px tall (12 top padding + 40 icon button +
    // 12 bottom padding — see AppChrome.tsx's topBar/iconButton styles);
    // this used to say 56, sitting slightly inside the bar instead of
    // right below it.
    top: 64,
    right: 16,
    minWidth: 220,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    backgroundColor: "#FFFFFF",
    paddingVertical: 8,
    zIndex: 31,
    shadowColor: "#000",
    shadowOpacity: 0.12,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
    elevation: 6,
  },
  signedIn: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    fontSize: 12,
    color: "#6B7280",
  },
  row: {
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  rowTitle: {
    fontSize: 15,
    fontWeight: "600",
    color: "#111827",
  },
  signOutText: {
    color: "#DC2626",
  },
});
