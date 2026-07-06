import React, { useEffect, useState } from "react";
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
import * as WebBrowser from "expo-web-browser";
import * as Google from "expo-auth-session/providers/google";
import { useAuthStore } from "../store/authStore";
import { googleOAuth, googleSignInConfigured } from "../auth/firebaseConfig";

// Completes the auth session if the app was opened via the OAuth redirect
// (no-op otherwise). Must run at module scope per expo-auth-session docs.
WebBrowser.maybeCompleteAuthSession();

type Mode = "signIn" | "signUp";

export default function LoginScreen() {
  const [mode, setMode] = useState<Mode>("signIn");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const signIn = useAuthStore((s) => s.signIn);
  const signUp = useAuthStore((s) => s.signUp);
  const busy = useAuthStore((s) => s.busy);
  const error = useAuthStore((s) => s.error);
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
            if (error) clearError();
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
            if (error) clearError();
            setPassword(v);
          }}
        />

        {error ? (
          <Text testID="auth-error" style={styles.error}>
            {error}
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
          <GoogleSignInButton busy={busy} />
        ) : (
          <Text testID="google-unconfigured" style={styles.googleUnconfigured}>
            Google sign-in isn't configured yet.
          </Text>
        )}
      </View>
    </KeyboardAvoidingView>
  );
}

/**
 * Google sign-in button, isolated into its own component so the
 * expo-auth-session `useAuthRequest` hook only runs when Google is actually
 * configured. On web that hook throws synchronously ("Client Id property
 * `webClientId` must be defined to use Google auth on this platform") when the
 * client id is undefined — which crashes the entire app to a blank screen.
 * Rendering this component is gated on `googleSignInConfigured`, so the hook
 * stays out of the React tree entirely until the OAuth client ids exist.
 */
function GoogleSignInButton({ busy }: { busy: boolean }) {
  const signInWithGoogleIdToken = useAuthStore((s) => s.signInWithGoogleIdToken);
  const clearError = useAuthStore((s) => s.clearError);

  const [request, response, promptAsync] = Google.useAuthRequest({
    webClientId: googleOAuth.webClientId,
    iosClientId: googleOAuth.iosClientId,
    androidClientId: googleOAuth.androidClientId,
  });

  // On a successful Google OAuth round-trip, exchange the Google ID token for a
  // Firebase session. The provider surfaces the id_token in params (implicit
  // flow) or on the authentication object depending on platform.
  useEffect(() => {
    if (response?.type !== "success") return;
    const googleIdToken =
      response.params?.id_token ?? response.authentication?.idToken;
    if (googleIdToken) {
      void signInWithGoogleIdToken(googleIdToken);
    }
  }, [response, signInWithGoogleIdToken]);

  return (
    <TouchableOpacity
      testID="google-button"
      style={[styles.googleButton, (!request || busy) && styles.buttonDisabled]}
      disabled={!request || busy}
      onPress={() => {
        clearError();
        void promptAsync();
      }}
    >
      <Text style={styles.googleButtonText}>Continue with Google</Text>
    </TouchableOpacity>
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
  googleUnconfigured: {
    color: "#9CA3AF",
    fontSize: 13,
    textAlign: "center",
  },
});
