import { create } from "zustand";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithCredential,
  signOut as fbSignOut,
  onIdTokenChanged,
  GoogleAuthProvider,
  type User,
} from "firebase/auth";
import { auth } from "../auth/firebase";
import { setCachedToken, setTokenProvider } from "../auth/authToken";

/** The slice of the Firebase user the UI actually needs. */
export interface AuthUser {
  uid: string;
  email: string | null;
  displayName: string | null;
}

interface AuthState {
  /** Null when signed out. */
  user: AuthUser | null;
  /** True until the first onIdTokenChanged fires — gates the app on a cold
   *  start so we neither flash the login screen at an already-signed-in user
   *  nor the app at a signed-out one. */
  initializing: boolean;
  /** Last auth error, as an honest human-readable message (never a raw code). */
  error: string | null;
  /** True while a sign-in / sign-up / google call is in flight. */
  busy: boolean;

  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  /** Exchange a Google OAuth ID token (from expo-auth-session) for a Firebase
   *  session. Kept separate from the OAuth dance so the store stays UI-free. */
  signInWithGoogleIdToken: (googleIdToken: string) => Promise<void>;
  signOut: () => Promise<void>;
  clearError: () => void;
}

function toAuthUser(user: User): AuthUser {
  return { uid: user.uid, email: user.email, displayName: user.displayName };
}

/**
 * Map a Firebase auth error to an honest, user-facing message. We never show a
 * raw `auth/...` code, and never invent detail we don't have — unknown codes
 * fall through to a plain generic line.
 */
function authErrorMessage(err: unknown): string {
  const code =
    typeof err === "object" && err !== null && "code" in err
      ? String((err as { code: unknown }).code)
      : "";
  switch (code) {
    case "auth/invalid-email":
      return "That email address isn't valid.";
    case "auth/missing-password":
      return "Please enter your password.";
    case "auth/weak-password":
      return "Password is too weak — use at least 6 characters.";
    case "auth/email-already-in-use":
      return "An account already exists for that email. Try signing in.";
    case "auth/invalid-credential":
    case "auth/wrong-password":
    case "auth/user-not-found":
      return "Incorrect email or password.";
    case "auth/too-many-requests":
      return "Too many attempts. Please wait a moment and try again.";
    case "auth/network-request-failed":
      return "Network error — check your connection and try again.";
    case "auth/operation-not-allowed":
      return "This sign-in method isn't enabled for the app yet.";
    default:
      return "Something went wrong signing you in. Please try again.";
  }
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  initializing: true,
  error: null,
  busy: false,

  signIn: async (email, password) => {
    set({ busy: true, error: null });
    try {
      await signInWithEmailAndPassword(auth, email.trim(), password);
    } catch (err) {
      set({ error: authErrorMessage(err) });
      throw err;
    } finally {
      set({ busy: false });
    }
  },

  signUp: async (email, password) => {
    set({ busy: true, error: null });
    try {
      await createUserWithEmailAndPassword(auth, email.trim(), password);
    } catch (err) {
      set({ error: authErrorMessage(err) });
      throw err;
    } finally {
      set({ busy: false });
    }
  },

  signInWithGoogleIdToken: async (googleIdToken) => {
    set({ busy: true, error: null });
    try {
      const credential = GoogleAuthProvider.credential(googleIdToken);
      await signInWithCredential(auth, credential);
    } catch (err) {
      set({ error: authErrorMessage(err) });
      throw err;
    } finally {
      set({ busy: false });
    }
  },

  signOut: async () => {
    // onIdTokenChanged clears user + token; just ask Firebase to sign out.
    await fbSignOut(auth);
  },

  clearError: () => set({ error: null }),
}));

/**
 * Wire Firebase's auth lifecycle into the store and the token accessor.
 * Idempotent — safe to call from module load and/or App mount.
 *
 * onIdTokenChanged fires on sign-in, sign-out, AND every silent token refresh
 * (~hourly), so the cached token the WS/REST layers read is always current.
 */
let started = false;
export function initAuth(): void {
  if (started) return;
  started = true;

  // REST calls get a *fresh* token (force-refreshing near expiry) straight
  // from the current Firebase user.
  setTokenProvider(async () => {
    const current = auth.currentUser;
    return current ? current.getIdToken() : null;
  });

  onIdTokenChanged(auth, async (user: User | null) => {
    if (user) {
      const token = await user.getIdToken();
      setCachedToken(token);
      useAuthStore.setState({ user: toAuthUser(user), initializing: false });
    } else {
      setCachedToken(null);
      useAuthStore.setState({ user: null, initializing: false });
    }
  });
}
