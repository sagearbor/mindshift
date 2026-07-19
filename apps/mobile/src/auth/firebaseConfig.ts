import { Platform } from "react-native";

/**
 * Firebase Web client configuration for the `arborfam-hub` project.
 *
 * These are PUBLIC client identifiers (the same values Firebase ships inside a
 * web app's HTML) — they are safe to embed in the bundle and are NOT secrets.
 * Real protection comes from the backend verifying the signed ID token, plus
 * Firebase's own provider allowlists / authorized-domain settings.
 *
 * Values can be overridden per-build via EXPO_PUBLIC_* env vars (e.g. to point
 * a fork at a different Firebase project) without touching code.
 */
export const firebaseConfig = {
  apiKey:
    process.env.EXPO_PUBLIC_FIREBASE_API_KEY ??
    "AIzaSyAJA-C1dpMqpjmM9A7GIGb-IfsOJSl7XS4",
  authDomain:
    process.env.EXPO_PUBLIC_FIREBASE_AUTH_DOMAIN ??
    "arborfam-hub.firebaseapp.com",
  projectId: process.env.EXPO_PUBLIC_FIREBASE_PROJECT_ID ?? "arborfam-hub",
  storageBucket:
    process.env.EXPO_PUBLIC_FIREBASE_STORAGE_BUCKET ??
    "arborfam-hub.firebasestorage.app",
  messagingSenderId:
    process.env.EXPO_PUBLIC_FIREBASE_MESSAGING_SENDER_ID ?? "664594784582",
  appId:
    process.env.EXPO_PUBLIC_FIREBASE_APP_ID ??
    "1:664594784582:web:553831ba12a3617cc60547",
};

/**
 * OAuth client IDs for Google sign-in.
 *
 * WEB uses the Firebase JS SDK's `signInWithPopup(GoogleAuthProvider)`, which
 * needs NO client id beyond the Firebase config above — the popup rides the
 * project's own OAuth client and authorized-domain allowlist. So the web path
 * is always "configured" as far as the bundle is concerned; a not-enabled
 * provider surfaces at runtime as an honest `auth/operation-not-allowed`.
 *
 * NATIVE (Android) uses `@react-native-google-signin/google-signin`, which does
 * need a **web** client id: `GoogleSignin.configure({ webClientId })` sets the
 * audience of the ID token Firebase later verifies. The Android OAuth client
 * (registered with the build's SHA-1) is matched by package name at Google's
 * end and is not referenced here. Until the web client id is provided via
 * EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID, native Google sign-in degrades to an honest
 * "not configured yet" state instead of failing cryptically.
 */
export const googleOAuth = {
  webClientId: process.env.EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID,
  iosClientId: process.env.EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID,
  androidClientId: process.env.EXPO_PUBLIC_GOOGLE_ANDROID_CLIENT_ID,
};

/**
 * True when Google sign-in has the minimum config it needs to be offered.
 *  - web:    always (signInWithPopup needs only the Firebase config).
 *  - native: only once a web client id exists for GoogleSignin.configure().
 */
export const googleSignInConfigured =
  Platform.OS === "web" ? true : Boolean(googleOAuth.webClientId);
