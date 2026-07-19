import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  StyleSheet,
} from "react-native";
import { useAuthStore } from "../store/authStore";
import { googleSignInConfigured } from "../auth/firebaseConfig";
import GoogleSignInButton from "../components/GoogleSignInButton";

type Mode = "signIn" | "signUp";

export default function LoginScreen() {
  const [mode, setMode] = useState<Mode>("signIn");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const signIn = useAuthStore((s) => s.signIn);
  const signUp = useAuthStore((s) => s.signUp);
  const sendPasswordReset = useAuthStore((s) => s.sendPasswordReset);
  const busy = useAuthStore((s) => s.busy);
  const error = useAuthStore((s) => s.error);
  const notice = useAuthStore((s) => s.notice);
  const clearError = useAuthStore((s) => s.clearError);

  const submit = () => {
    const fn = mode === "signIn" ? signIn : signUp;
    // Errors surface via the store's `error` state; swallow the rejection here
    // so an unhandled promise doesn't crash the screen.
    void fn(email, password).catch(() => {});
  };

  const toggleMode = () => {
    clearError();
    setMode((m) => (m === "signIn" ? "signUp" : "signIn"));
  };

  const forgotPassword = () => {
    // sendPasswordReset handles the empty-email and error states honestly and
    // surfaces them via the store; nothing to await for the UI here.
    void sendPasswordReset(email);
  };

  const canSubmit = email.trim().length > 0 && password.length > 0 && !busy;

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <View style={styles.container} testID="login-screen">
        <Text style={styles.brand}>MindShift</Text>
        <Text style={styles.subtitle}>
          {mode === "signIn"
            ? "Sign in to continue"
            : "Create your account"}
        </Text>

        <TextInput
          testID="email-input"
          style={styles.input}
          placeholder="Email"
          placeholderTextColor="#9CA3AF"
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="email-address"
          textContentType="emailAddress"
          value={email}
          onChangeText={(v) => {
            if (error || notice) clearError();
            setEmail(v);
          }}
        />

        <TextInput
          testID="password-input"
          style={styles.input}
          placeholder="Password"
          placeholderTextColor="#9CA3AF"
          autoCapitalize="none"
          autoCorrect={false}
          secureTextEntry
          textContentType="password"
          value={password}
          onChangeText={(v) => {
            if (error || notice) clearError();
            setPassword(v);
          }}
        />

        {mode === "signIn" ? (
          <TouchableOpacity
            testID="forgot-password"
            style={styles.forgot}
            onPress={forgotPassword}
            disabled={busy}
          >
            <Text style={styles.forgotText}>Forgot password?</Text>
          </TouchableOpacity>
        ) : null}

        {error ? (
          <Text testID="auth-error" style={styles.error}>
            {error}
          </Text>
        ) : null}

        {notice ? (
          <Text testID="auth-notice" style={styles.notice}>
            {notice}
          </Text>
        ) : null}

        <TouchableOpacity
          testID="submit-button"
          style={[styles.primaryButton, !canSubmit && styles.buttonDisabled]}
          onPress={submit}
          disabled={!canSubmit}
        >
          {busy ? (
            <ActivityIndicator color="#FFFFFF" />
          ) : (
            <Text style={styles.primaryButtonText}>
              {mode === "signIn" ? "Sign In" : "Sign Up"}
            </Text>
          )}
        </TouchableOpacity>

        <TouchableOpacity
          testID="toggle-mode"
          style={styles.toggle}
          onPress={toggleMode}
        >
          <Text style={styles.toggleText}>
            {mode === "signIn"
              ? "Don't have an account? Sign up"
              : "Already have an account? Sign in"}
          </Text>
        </TouchableOpacity>

        <View style={styles.dividerRow}>
          <View style={styles.divider} />
          <Text style={styles.dividerText}>or</Text>
          <View style={styles.divider} />
        </View>

        {googleSignInConfigured ? (
          <GoogleSignInButton />
        ) : (
          <Text testID="google-unconfigured" style={styles.googleUnconfigured}>
            Google sign-in isn't configured yet.
          </Text>
        )}
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  container: {
    flex: 1,
    justifyContent: "center",
    paddingHorizontal: 28,
    backgroundColor: "#F9FAFB",
  },
  brand: {
    fontSize: 34,
    fontWeight: "700",
    color: "#312E81",
    textAlign: "center",
  },
  subtitle: {
    fontSize: 15,
    color: "#6B7280",
    textAlign: "center",
    marginTop: 6,
    marginBottom: 28,
  },
  input: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 10,
    padding: 14,
    fontSize: 15,
    marginBottom: 12,
    color: "#1F2937",
    backgroundColor: "#FFFFFF",
  },
  error: {
    color: "#B91C1C",
    fontSize: 13,
    marginBottom: 12,
    textAlign: "center",
  },
  notice: {
    color: "#047857",
    fontSize: 13,
    marginBottom: 12,
    textAlign: "center",
  },
  forgot: {
    alignSelf: "flex-end",
    marginBottom: 12,
  },
  forgotText: {
    color: "#4A90D9",
    fontSize: 13,
  },
  primaryButton: {
    backgroundColor: "#4A90D9",
    paddingVertical: 15,
    borderRadius: 10,
    alignItems: "center",
    marginTop: 4,
  },
  primaryButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  toggle: {
    marginTop: 16,
    alignItems: "center",
  },
  toggleText: {
    color: "#4A90D9",
    fontSize: 14,
  },
  dividerRow: {
    flexDirection: "row",
    alignItems: "center",
    marginVertical: 24,
  },
  divider: {
    flex: 1,
    height: 1,
    backgroundColor: "#E5E7EB",
  },
  dividerText: {
    marginHorizontal: 12,
    color: "#9CA3AF",
    fontSize: 13,
  },
  googleUnconfigured: {
    color: "#9CA3AF",
    fontSize: 13,
    textAlign: "center",
  },
});
