import React from "react";
import { TouchableOpacity, Text, StyleSheet } from "react-native";

/**
 * Presentational "Continue with Google" button, shared by the web and native
 * wiring variants (GoogleSignInButton.web / .native). It owns no auth logic —
 * just the look, the disabled state, and the testID the tests key off of.
 */
export default function GoogleButtonBase({
  onPress,
  disabled,
  label = "Continue with Google",
}: {
  onPress: () => void;
  disabled?: boolean;
  label?: string;
}) {
  return (
    <TouchableOpacity
      testID="google-button"
      style={[styles.googleButton, disabled && styles.buttonDisabled]}
      disabled={disabled}
      onPress={onPress}
    >
      <Text style={styles.googleButtonText}>{label}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  googleButton: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
    backgroundColor: "#FFFFFF",
  },
  googleButtonText: {
    color: "#1F2937",
    fontSize: 15,
    fontWeight: "600",
  },
  buttonDisabled: {
    opacity: 0.5,
  },
});
