import React, { useEffect, useState } from "react";
import { Platform, StyleSheet } from "react-native";
import * as AppleAuthentication from "expo-apple-authentication";
import * as Crypto from "expo-crypto";
import { useAuthStore } from "../store/authStore";

/**
 * Sign in with Apple — required, not optional. App Store Review Guideline 4.8
 * makes it mandatory for any app offering a third-party sign-in, and MindShift
 * offers Google, so shipping without this is an automatic rejection. (The
 * button this replaced was a visibly disabled "coming soon" placeholder, which
 * is its own rejection risk under Guideline 2.1 App Completeness.)
 *
 * **The nonce is the part that is easy to get wrong.** We generate a random
 * raw nonce, hand Apple its SHA-256 *hash*, and hand Firebase the *raw* value;
 * Firebase hashes it again and checks the result against the hash Apple
 * embedded in the signed token. Swapping the two — a natural mistake, since
 * both are "the nonce" — fails as `auth/invalid-credential`. This proves the
 * token was minted for this sign-in attempt and cannot be replayed.
 *
 * Renders nothing off iOS: `isAvailableAsync` is false on Android, and Apple's
 * web flow is a different (redirect) mechanism needing a Services ID and a
 * private key we deliberately have not configured — see AppleSignInButton.web.
 */
export default function AppleSignInButton() {
  const signInWithAppleIdToken = useAuthStore((s) => s.signInWithAppleIdToken);
  const setState = useAuthStore.setState;
  const busy = useAuthStore((s) => s.busy);
  const [available, setAvailable] = useState(Platform.OS === "ios");

  useEffect(() => {
    if (Platform.OS !== "ios") return;
    let cancelled = false;
    // False on an iPad running an old iOS, and in some simulators.
    void AppleAuthentication.isAvailableAsync()
      .then((ok) => {
        if (!cancelled) setAvailable(ok);
      })
      .catch(() => {
        if (!cancelled) setAvailable(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!available) return null;

  const onPress = async () => {
    setState({ error: null, notice: null });
    try {
      const rawNonce = Crypto.randomUUID();
      const hashedNonce = await Crypto.digestStringAsync(
        Crypto.CryptoDigestAlgorithm.SHA256,
        rawNonce,
      );
      const credential = await AppleAuthentication.signInAsync({
        requestedScopes: [
          AppleAuthentication.AppleAuthenticationScope.EMAIL,
          AppleAuthentication.AppleAuthenticationScope.FULL_NAME,
        ],
        nonce: hashedNonce,
      });
      if (!credential.identityToken) {
        setState({
          error: "Apple didn't return a sign-in token. Please try again.",
        });
        return;
      }
      await signInWithAppleIdToken(credential.identityToken, rawNonce);
    } catch (err) {
      // Dismissing the Apple sheet is a normal outcome, not an error to report.
      if ((err as { code?: string })?.code === "ERR_REQUEST_CANCELED") return;
      setState({ error: "Couldn't sign in with Apple. Please try again." });
    }
  };

  return (
    <AppleAuthentication.AppleAuthenticationButton
      testID="apple-button"
      accessibilityLabel="Continue with Apple"
      buttonType={
        AppleAuthentication.AppleAuthenticationButtonType.SIGN_IN
      }
      buttonStyle={
        AppleAuthentication.AppleAuthenticationButtonStyle.BLACK
      }
      cornerRadius={10}
      style={[styles.button, busy && styles.busy]}
      onPress={() => {
        if (busy) return;
        void onPress();
      }}
    />
  );
}

const styles = StyleSheet.create({
  // Height and margin match GoogleButtonBase so the two sit as a pair. Apple
  // renders the label and glyph itself — its Human Interface Guidelines do not
  // allow restyling the button's content.
  button: {
    width: "100%",
    height: 48,
    marginTop: 12,
  },
  busy: {
    opacity: 0.5,
  },
});
