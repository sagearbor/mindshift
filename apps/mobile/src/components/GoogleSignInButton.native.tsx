import React, { useEffect } from "react";
import {
  GoogleSignin,
  isSuccessResponse,
  isErrorWithCode,
  statusCodes,
} from "@react-native-google-signin/google-signin";
import { useAuthStore } from "../store/authStore";
import { googleOAuth } from "../auth/firebaseConfig";
import GoogleButtonBase from "./GoogleButtonBase";

/**
 * NATIVE (Android) Google sign-in via @react-native-google-signin/google-signin
 * — Expo's officially recommended package now that expo-auth-session's Google
 * provider is deprecated. It surfaces the modern Android Credential Manager /
 * Play-services flow and works with EAS managed builds through its config
 * plugin (see app.json); it is NOT available in Expo Go.
 *
 * The native SDK returns a Google ID token whose audience is the **web** client
 * id passed to configure(). We hand that token to the store, which mints a
 * Firebase credential from it — the exact same signInWithCredential path the web
 * popup ends at, so both platforms converge on one Firebase identity (and the
 * account-linking recovery in the store).
 */
export default function GoogleSignInButton() {
  const signInWithGoogleIdToken = useAuthStore((s) => s.signInWithGoogleIdToken);
  const setError = useAuthStore.setState;
  const busy = useAuthStore((s) => s.busy);

  useEffect(() => {
    // webClientId sets the ID-token audience Firebase verifies. Gated upstream
    // by googleSignInConfigured, so it is defined whenever this renders.
    GoogleSignin.configure({ webClientId: googleOAuth.webClientId });
  }, []);

  const onPress = async () => {
    setError({ error: null, notice: null });
    try {
      await GoogleSignin.hasPlayServices({ showPlayServicesUpdateDialog: true });
      const response = await GoogleSignin.signIn();
      if (!isSuccessResponse(response)) return; // user cancelled the sheet
      const idToken = response.data.idToken;
      if (!idToken) {
        setError({
          error:
            "Google didn't return a sign-in token. Check the app's OAuth setup.",
        });
        return;
      }
      await signInWithGoogleIdToken(idToken);
    } catch (err) {
      if (isErrorWithCode(err)) {
        if (err.code === statusCodes.SIGN_IN_CANCELLED) return; // dismissed
        if (err.code === statusCodes.IN_PROGRESS) return; // double-tap
        if (err.code === statusCodes.PLAY_SERVICES_NOT_AVAILABLE) {
          setError({
            error: "Google Play services isn't available or is out of date.",
          });
          return;
        }
      }
      setError({ error: "Couldn't sign in with Google. Please try again." });
    }
  };

  return (
    <GoogleButtonBase
      disabled={busy}
      onPress={() => {
        void onPress();
      }}
    />
  );
}
