import React from "react";
import { TouchableOpacity, Text, StyleSheet } from "react-native";

/**
 * Disabled "Continue with Apple" placeholder — groundwork ahead of the iPhone/
 * App Store push. Apple REQUIRES Sign in with Apple whenever another
 * third-party sign-in (here, Google) is offered, so the visible button goes
 * in now while the SDK wiring itself is next month's work.
 *
 * Deliberately honest, same as the gauge web dashboard's precedent: visibly
 * disabled, clearly labeled "coming soon", no onPress, no SDK/auth-store
 * wiring. Shape/size matches GoogleButtonBase so the two sit consistently on
 * LoginScreen; same component renders on web and native (no platform split
 * needed — there's no real behavior to diverge on yet).
 */
export default function AppleSignInButton() {
  return (
    <TouchableOpacity
      testID="apple-button"
      style={[styles.appleButton, styles.buttonDisabled]}
      disabled
      accessibilityState={{ disabled: true }}
      accessibilityLabel="Continue with Apple — coming soon"
    >
      <Text style={styles.appleButtonText}>
        {""} Continue with Apple
      </Text>
      <Text style={styles.comingSoonText}>(coming soon)</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  appleButton: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
    backgroundColor: "#FFFFFF",
    marginTop: 12,
  },
  appleButtonText: {
    color: "#1F2937",
    fontSize: 15,
    fontWeight: "600",
  },
  comingSoonText: {
    color: "#9CA3AF",
    fontSize: 12,
    marginTop: 2,
  },
  buttonDisabled: {
    opacity: 0.5,
  },
});
