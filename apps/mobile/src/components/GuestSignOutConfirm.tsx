import React from "react";
import { View, Text, TouchableOpacity, ScrollView, StyleSheet } from "react-native";

/**
 * The confirmation a GUEST gets before "Log out" — and only a guest.
 *
 * Why this exists: for an anonymous ("Continue as guest") account, Log out is
 * a one-way door. There is no email and no password on the account, so there
 * is no credential to sign back in with; the next "Continue as guest" mints a
 * brand-new anonymous uid and everything recorded under the old one is out of
 * reach for good. Settings offered that behind a single unconfirmed tap, two
 * rows above a "Delete my account" flow that makes you TYPE the word DELETE.
 * The banner at the top of every screen already warns that the data lives only
 * on this device's account — and then the app put losing it behind the least
 * friction it has.
 *
 * A signed-in account with an email loses nothing by logging out and is never
 * shown this (App.tsx gates on `user.isAnonymous`). Friction there would be
 * pure nagging.
 *
 * Deliberately an in-app card, not `Alert.alert`, for the same reason the
 * delete-account flow is (see AdvancedScreen.tsx): react-native-web has no
 * working Alert, and this app runs on web from the same codebase — a
 * destructive confirmation that silently no-ops on one platform is worse than
 * no confirmation at all. Rendered as an overlay above the whole app so BOTH
 * Log out entry points (the avatar menu and Settings' own row) are covered by
 * one implementation rather than two.
 *
 * The copy never says "deleted": logging out does not erase the guest's
 * sessions server-side, it strands them under a uid nobody can authenticate
 * as again. Saying "out of reach" is the true statement; saying "deleted"
 * would be a convenient one.
 */
export interface GuestSignOutConfirmProps {
  /** Dismiss and stay signed in. */
  onCancel: () => void;
  /** Proceed with the real sign-out. */
  onConfirm: () => void;
}

export const GUEST_SIGN_OUT_TITLE = "Log out of this guest session?";

export const GUEST_SIGN_OUT_BODY =
  "This guest account has no email and no password, so there is no way to " +
  "sign back in to it. Logging out doesn't delete what's on it — it puts it " +
  "out of reach for good. Next time, “Continue as guest” starts a " +
  "brand-new, empty account.";

export const GUEST_SIGN_OUT_KEEP =
  "To keep it instead: tap “Create account” in the guest bar at the " +
  "top. Same account and same recordings — it just gains a way back in.";

export default function GuestSignOutConfirm({
  onCancel,
  onConfirm,
}: GuestSignOutConfirmProps) {
  return (
    <View style={styles.overlay} testID="guest-sign-out-confirm">
      {/* A tap outside cancels — the safe outcome, same as the account
          menu's backdrop. Never the destructive one. */}
      <TouchableOpacity
        testID="guest-sign-out-backdrop"
        accessibilityLabel="Cancel logging out"
        style={styles.backdrop}
        activeOpacity={1}
        onPress={onCancel}
      />
      <View style={styles.card} accessibilityViewIsModal>
        <ScrollView contentContainerStyle={styles.cardContent}>
          <Text style={styles.title} testID="guest-sign-out-title">
            {GUEST_SIGN_OUT_TITLE}
          </Text>
          <Text style={styles.body} testID="guest-sign-out-body">
            {GUEST_SIGN_OUT_BODY}
          </Text>

          <Text style={styles.heading}>What you'd be leaving behind</Text>
          <Text style={styles.item} testID="guest-sign-out-scope">
            • Every session and recording made as this guest{"\n"}
            • Your voiceprint and everyone you've named{"\n"}
            • This account's settings and history on this phone
          </Text>

          <Text style={styles.keep} testID="guest-sign-out-keep">
            {GUEST_SIGN_OUT_KEEP}
          </Text>

          <TouchableOpacity
            testID="guest-sign-out-confirm-button"
            accessibilityRole="button"
            style={styles.dangerButton}
            onPress={onConfirm}
          >
            <Text style={styles.dangerButtonText}>
              Log out and lose this guest session
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            testID="guest-sign-out-cancel"
            accessibilityRole="button"
            style={styles.cancelButton}
            onPress={onCancel}
          >
            <Text style={styles.cancelText}>Cancel — stay signed in</Text>
          </TouchableOpacity>
        </ScrollView>
      </View>
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
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
    // Above AppChrome's account menu (zIndex 31) so tapping Log out in that
    // menu doesn't leave the menu painted over this.
    zIndex: 40,
  },
  backdrop: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(17, 24, 39, 0.55)",
  },
  card: {
    width: "100%",
    maxWidth: 420,
    maxHeight: "85%",
    borderRadius: 16,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#FECACA",
    shadowColor: "#000",
    shadowOpacity: 0.2,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 6 },
    elevation: 8,
  },
  cardContent: {
    padding: 20,
    gap: 10,
  },
  title: {
    fontSize: 17,
    fontWeight: "700",
    color: "#B91C1C",
  },
  body: {
    fontSize: 13,
    lineHeight: 19,
    color: "#374151",
  },
  heading: {
    fontSize: 13,
    fontWeight: "700",
    color: "#111827",
    marginTop: 4,
  },
  item: {
    fontSize: 13,
    lineHeight: 20,
    color: "#374151",
  },
  keep: {
    fontSize: 13,
    lineHeight: 19,
    color: "#065F46",
  },
  dangerButton: {
    marginTop: 6,
    backgroundColor: "#DC2626",
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: "center",
  },
  dangerButtonText: {
    color: "#FFFFFF",
    fontSize: 14,
    fontWeight: "700",
  },
  cancelButton: {
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#D1D5DB",
  },
  cancelText: {
    color: "#111827",
    fontSize: 14,
    fontWeight: "600",
  },
});
