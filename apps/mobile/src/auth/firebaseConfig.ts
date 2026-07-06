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
 * OAuth client IDs for Google sign-in via expo-auth-session. These do NOT
 * exist until the owner enables the Google provider in the Firebase console
 * (which auto-creates the Web OAuth client) and registers native OAuth clients.
 * Until then they are undefined and the Google button degrades to an honest
 * "not configured yet" state instead of failing cryptically.
 *
 *  - WEB client id  = the ID-token *audience* expo-auth-session requests; it is
 *    also the client the Firebase credential is minted against. Required.
 *  - iOS / Android client ids = only needed for the native standalone builds.
 */
export const googleOAuth = {
  webClientId: process.env.EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID,
  iosClientId: process.env.EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID,
  androidClientId: process.env.EXPO_PUBLIC_GOOGLE_ANDROID_CLIENT_ID,
};

/** True when Google sign-in has the minimum config it needs to be offered. */
export const googleSignInConfigured = Boolean(googleOAuth.webClientId);
