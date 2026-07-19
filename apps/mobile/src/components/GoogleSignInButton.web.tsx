import React from "react";
import { useAuthStore } from "../store/authStore";
import GoogleButtonBase from "./GoogleButtonBase";

/**
 * WEB Google sign-in. The Firebase JS SDK's signInWithPopup resolves the entire
 * OAuth round-trip in the browser — no OAuth client id and no expo-auth-session
 * dance needed, only the project's Firebase config plus the Google provider
 * enabled (and this origin authorized) in the Firebase console. If the provider
 * is off, the popup rejects with auth/operation-not-allowed, which the store
 * turns into an honest "isn't enabled yet" message.
 */
export default function GoogleSignInButton() {
  const signInWithGooglePopup = useAuthStore((s) => s.signInWithGooglePopup);
  const busy = useAuthStore((s) => s.busy);

  return (
    <GoogleButtonBase
      disabled={busy}
      onPress={() => {
        void signInWithGooglePopup();
      }}
    />
  );
}
