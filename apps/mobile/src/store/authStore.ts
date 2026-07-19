import { create } from "zustand";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithCredential,
  signInWithPopup,
  linkWithCredential,
  sendPasswordResetEmail,
  signOut as fbSignOut,
  onIdTokenChanged,
  GoogleAuthProvider,
  type AuthCredential,
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
  /** A non-error, informational message (e.g. "reset email sent", "Google
   *  linked"). Kept separate from `error` so the UI can style it neutrally. */
  notice: string | null;
  /** True while a sign-in / sign-up / google call is in flight. */
  busy: boolean;
  /** When a Google sign-in hits an existing email/password account, we stash the
   *  pending Google credential here (plus its email) so the very next successful
   *  email/password sign-in links Google onto that same account — Firebase's
   *  documented `account-exists-with-different-credential` recovery. */
  pendingGoogleCredential: AuthCredential | null;
  pendingGoogleEmail: string | null;

  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  /** WEB Google sign-in: Firebase popup. Resolves the whole OAuth dance in the
   *  SDK — no client id needed beyond the Firebase config. */
  signInWithGooglePopup: () => Promise<void>;
  /** NATIVE Google sign-in: exchange a Google OAuth ID token (from
   *  @react-native-google-signin) for a Firebase session. Kept separate from the
   *  OAuth dance so the store stays UI-free. */
  signInWithGoogleIdToken: (googleIdToken: string) => Promise<void>;
  /** Send a Firebase password-reset email for `email`. */
  sendPasswordReset: (email: string) => Promise<void>;
  signOut: () => Promise<void>;
  clearError: () => void;
}

function toAuthUser(user: User): AuthUser {
  return { uid: user.uid, email: user.email, displayName: user.displayName };
}

/** Extract the error code from an unknown Firebase error, or "". */
function errorCode(err: unknown): string {
  return typeof err === "object" && err !== null && "code" in err
    ? String((err as { code: unknown }).code)
    : "";
}

/** The email Firebase attaches to an account-exists error, if present. */
function errorEmail(err: unknown): string | null {
  if (typeof err === "object" && err !== null && "customData" in err) {
    const data = (err as { customData?: { email?: unknown } }).customData;
    if (data && typeof data.email === "string") return data.email;
  }
  return null;
}

/**
 * Map a Firebase auth error to an honest, user-facing message. We never show a
 * raw `auth/...` code, and never invent detail we don't have — unknown codes
 * fall through to a plain generic line.
 */
function authErrorMessage(err: unknown): string {
  const code = errorCode(err);
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
    case "auth/user-disabled":
      return "This account has been disabled.";
    default:
      return "Something went wrong signing you in. Please try again.";
  }
}

/** Codes that mean "the OAuth popup was dismissed", not a real failure. */
const POPUP_CANCELLED = new Set([
  "auth/popup-closed-by-user",
  "auth/cancelled-popup-request",
  "auth/user-cancelled",
]);

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  initializing: true,
  error: null,
  notice: null,
  busy: false,
  pendingGoogleCredential: null,
  pendingGoogleEmail: null,

  signIn: async (email, password) => {
    set({ busy: true, error: null, notice: null });
    try {
      await signInWithEmailAndPassword(auth, email.trim(), password);
      // If a Google sign-in earlier collided with this email, the user has now
      // proven ownership by password — link Google onto the SAME account so
      // both providers reach one identity from here on.
      await linkPendingGoogleCredential(email.trim(), set, get);
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

  signInWithGooglePopup: async () => {
    set({ busy: true, error: null, notice: null });
    try {
      const provider = new GoogleAuthProvider();
      await signInWithPopup(auth, provider);
      set({ pendingGoogleCredential: null, pendingGoogleEmail: null });
    } catch (err) {
      handleGoogleSignInError(err, null, set);
    } finally {
      set({ busy: false });
    }
  },

  signInWithGoogleIdToken: async (googleIdToken) => {
    set({ busy: true, error: null, notice: null });
    try {
      const credential = GoogleAuthProvider.credential(googleIdToken);
      await signInWithCredential(auth, credential);
      set({ pendingGoogleCredential: null, pendingGoogleEmail: null });
    } catch (err) {
      // Pass the credential we built as the fallback to stash for linking — on
      // native the error may not round-trip it via credentialFromError.
      handleGoogleSignInError(
        err,
        GoogleAuthProvider.credential(googleIdToken),
        set,
      );
    } finally {
      set({ busy: false });
    }
  },

  sendPasswordReset: async (email) => {
    const trimmed = email.trim();
    if (!trimmed) {
      set({ error: "Enter your email first.", notice: null });
      return;
    }
    set({ busy: true, error: null, notice: null });
    try {
      await sendPasswordResetEmail(auth, trimmed);
      set({ notice: `Reset email sent to ${trimmed} — check your inbox.` });
    } catch (err) {
      set({ error: authErrorMessage(err) });
    } finally {
      set({ busy: false });
    }
  },

  signOut: async () => {
    // onIdTokenChanged clears user + token; just ask Firebase to sign out.
    await fbSignOut(auth);
  },

  clearError: () => set({ error: null, notice: null }),
}));

type SetState = (partial: Partial<AuthState>) => void;
type GetState = () => AuthState;

/**
 * Shared handling for a failed Google sign-in (popup or native id-token).
 *
 * The one case we recover from is `account-exists-with-different-credential`:
 * the email already has an email/password account. We stash the pending Google
 * credential + email and post an honest, actionable notice; the next successful
 * email/password sign-in links Google onto that account (see signIn ->
 * linkPendingGoogleCredential). A dismissed popup is silent. Anything else
 * becomes an honest error.
 */
function handleGoogleSignInError(
  err: unknown,
  fallbackCredential: AuthCredential | null,
  set: SetState,
): void {
  const code = errorCode(err);
  if (code === "auth/account-exists-with-different-credential") {
    const pending = GoogleAuthProvider.credentialFromError(
      err as Parameters<typeof GoogleAuthProvider.credentialFromError>[0],
    );
    const email = errorEmail(err);
    set({
      pendingGoogleCredential: pending ?? fallbackCredential,
      pendingGoogleEmail: email,
      notice: email
        ? `You already have an account for ${email}. Sign in with your password below and we'll link Google to it.`
        : "You already have an account with this email. Sign in with your password below and we'll link Google to it.",
    });
    return;
  }
  if (POPUP_CANCELLED.has(code)) return; // user dismissed — not an error
  set({ error: authErrorMessage(err) });
}

/**
 * If a Google credential is pending for `email`, link it onto the now
 * signed-in user. Best-effort: a benign "already linked" outcome is treated as
 * success, and any other link failure leaves the user signed in (via password)
 * with an honest notice rather than blocking them.
 */
async function linkPendingGoogleCredential(
  email: string,
  set: SetState,
  get: GetState,
): Promise<void> {
  const { pendingGoogleCredential, pendingGoogleEmail } = get();
  if (!pendingGoogleCredential) return;
  // Only link when the just-authenticated email is the one Google collided on.
  if (
    pendingGoogleEmail &&
    pendingGoogleEmail.toLowerCase() !== email.toLowerCase()
  ) {
    return;
  }
  const current = auth.currentUser;
  set({ pendingGoogleCredential: null, pendingGoogleEmail: null });
  if (!current) return;
  try {
    await linkWithCredential(current, pendingGoogleCredential);
    set({ notice: "Google is now linked to your account." });
  } catch (err) {
    const code = errorCode(err);
    if (
      code === "auth/provider-already-linked" ||
      code === "auth/credential-already-in-use"
    ) {
      set({ notice: "Google is now linked to your account." });
      return;
    }
    set({
      notice: "Signed in. We couldn't link Google this time — you can try again.",
    });
  }
}

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
