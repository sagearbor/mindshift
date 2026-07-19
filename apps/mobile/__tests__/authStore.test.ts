import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithCredential,
  signInWithPopup,
  linkWithCredential,
  sendPasswordResetEmail,
  signOut as fbSignOut,
  GoogleAuthProvider,
} from "firebase/auth";
import { useAuthStore, initAuth } from "../src/store/authStore";
import { getCachedToken } from "../src/auth/authToken";

const signInMock = signInWithEmailAndPassword as jest.Mock;
const signUpMock = createUserWithEmailAndPassword as jest.Mock;
const signInCredMock = signInWithCredential as jest.Mock;
const signInPopupMock = signInWithPopup as jest.Mock;
const linkMock = linkWithCredential as jest.Mock;
const resetMock = sendPasswordResetEmail as jest.Mock;
const signOutMock = fbSignOut as jest.Mock;
const credentialMock = GoogleAuthProvider.credential as jest.Mock;
const credentialFromErrorMock = GoogleAuthProvider.credentialFromError as jest.Mock;

/** The mock state exposed by jest-setup's firebase/auth mock. */
interface FirebaseAuthMock {
  currentUser: unknown;
  idTokenListener: ((user: unknown) => void) | null;
}
const authMock = (globalThis as Record<string, unknown>)
  .__firebaseAuthMock as FirebaseAuthMock;

/** A fake Firebase user whose getIdToken resolves to `token`. */
function fakeUser(token: string) {
  return {
    uid: "user-1",
    email: "a@b.com",
    displayName: "Ada",
    getIdToken: jest.fn().mockResolvedValue(token),
  };
}

beforeEach(() => {
  signInMock.mockReset().mockResolvedValue(undefined);
  signUpMock.mockReset().mockResolvedValue(undefined);
  signInCredMock.mockReset().mockResolvedValue(undefined);
  signInPopupMock.mockReset().mockResolvedValue(undefined);
  linkMock.mockReset().mockResolvedValue(undefined);
  resetMock.mockReset().mockResolvedValue(undefined);
  signOutMock.mockReset().mockResolvedValue(undefined);
  credentialMock.mockClear();
  credentialFromErrorMock.mockClear();
  authMock.currentUser = null;
  useAuthStore.setState({
    user: null,
    initializing: true,
    error: null,
    notice: null,
    busy: false,
    pendingGoogleCredential: null,
    pendingGoogleEmail: null,
  });
});

describe("authStore — email/password", () => {
  it("signIn calls Firebase and clears busy on success", async () => {
    await useAuthStore.getState().signIn("  a@b.com ", "pw123456");

    expect(signInMock).toHaveBeenCalledWith(
      expect.anything(),
      "a@b.com", // trimmed
      "pw123456",
    );
    expect(useAuthStore.getState().busy).toBe(false);
    expect(useAuthStore.getState().error).toBeNull();
  });

  it("signUp maps a Firebase error code to an honest message", async () => {
    signUpMock.mockRejectedValue({ code: "auth/email-already-in-use" });

    await expect(
      useAuthStore.getState().signUp("a@b.com", "pw"),
    ).rejects.toBeDefined();

    expect(useAuthStore.getState().error).toMatch(/already exists/i);
    // Never leaks the raw code.
    expect(useAuthStore.getState().error).not.toMatch(/auth\//);
    expect(useAuthStore.getState().busy).toBe(false);
  });

  it("maps invalid-credential to a generic 'incorrect email or password'", async () => {
    signInMock.mockRejectedValue({ code: "auth/invalid-credential" });
    await expect(
      useAuthStore.getState().signIn("a@b.com", "wrong"),
    ).rejects.toBeDefined();
    expect(useAuthStore.getState().error).toMatch(/incorrect email or password/i);
  });
});

describe("authStore — Google", () => {
  it("exchanges a Google ID token for a Firebase credential (native)", async () => {
    await useAuthStore.getState().signInWithGoogleIdToken("google-tok");

    expect(credentialMock).toHaveBeenCalledWith("google-tok");
    expect(signInCredMock).toHaveBeenCalledTimes(1);
    expect(useAuthStore.getState().error).toBeNull();
  });

  it("signs in with the Firebase popup (web)", async () => {
    await useAuthStore.getState().signInWithGooglePopup();

    expect(signInPopupMock).toHaveBeenCalledTimes(1);
    expect(useAuthStore.getState().error).toBeNull();
    expect(useAuthStore.getState().pendingGoogleCredential).toBeNull();
  });

  it("a dismissed popup is not surfaced as an error", async () => {
    signInPopupMock.mockRejectedValue({ code: "auth/popup-closed-by-user" });
    await useAuthStore.getState().signInWithGooglePopup();
    expect(useAuthStore.getState().error).toBeNull();
    expect(useAuthStore.getState().busy).toBe(false);
  });
});

describe("authStore — account linking (exists with different credential)", () => {
  it("stashes the pending Google credential + email and posts a notice", async () => {
    const pendingCred = { providerId: "google.com", pending: true };
    signInPopupMock.mockRejectedValue({
      code: "auth/account-exists-with-different-credential",
      customData: { email: "linda@example.com" },
      __pendingCred: pendingCred,
    });

    await useAuthStore.getState().signInWithGooglePopup();

    const s = useAuthStore.getState();
    expect(credentialFromErrorMock).toHaveBeenCalled();
    expect(s.pendingGoogleCredential).toBe(pendingCred);
    expect(s.pendingGoogleEmail).toBe("linda@example.com");
    expect(s.notice).toMatch(/linda@example\.com/);
    expect(s.error).toBeNull(); // it's a recoverable state, not an error
  });

  it("links the pending Google credential onto the account after password sign-in", async () => {
    // Arrange: a Google collision left a pending credential for this email.
    const pendingCred = { providerId: "google.com", pending: true };
    useAuthStore.setState({
      pendingGoogleCredential: pendingCred as never,
      pendingGoogleEmail: "linda@example.com",
    });
    const user = { uid: "u1", email: "linda@example.com", displayName: null };
    authMock.currentUser = user;

    // Act: the user proves ownership with their password.
    await useAuthStore.getState().signIn("linda@example.com", "pw123456");

    // Assert: Google was linked onto the SAME (now current) user, pending cleared.
    expect(signInMock).toHaveBeenCalledTimes(1);
    expect(linkMock).toHaveBeenCalledWith(user, pendingCred);
    const s = useAuthStore.getState();
    expect(s.pendingGoogleCredential).toBeNull();
    expect(s.pendingGoogleEmail).toBeNull();
    expect(s.notice).toMatch(/linked/i);
  });

  it("does not link when there is no pending credential", async () => {
    authMock.currentUser = { uid: "u1", email: "a@b.com", displayName: null };
    await useAuthStore.getState().signIn("a@b.com", "pw123456");
    expect(linkMock).not.toHaveBeenCalled();
  });

  it("treats an already-linked provider as success, not an error", async () => {
    useAuthStore.setState({
      pendingGoogleCredential: { providerId: "google.com" } as never,
      pendingGoogleEmail: "linda@example.com",
    });
    authMock.currentUser = { uid: "u1", email: "linda@example.com", displayName: null };
    linkMock.mockRejectedValue({ code: "auth/provider-already-linked" });

    await useAuthStore.getState().signIn("linda@example.com", "pw123456");

    expect(useAuthStore.getState().notice).toMatch(/linked/i);
    expect(useAuthStore.getState().error).toBeNull();
  });
});

describe("authStore — password reset", () => {
  it("sends a reset email and posts an honest confirmation", async () => {
    await useAuthStore.getState().sendPasswordReset("  linda@example.com ");

    expect(resetMock).toHaveBeenCalledWith(expect.anything(), "linda@example.com");
    expect(useAuthStore.getState().notice).toMatch(/reset email sent to linda@example\.com/i);
    expect(useAuthStore.getState().error).toBeNull();
  });

  it("refuses an empty email without calling Firebase", async () => {
    await useAuthStore.getState().sendPasswordReset("   ");

    expect(resetMock).not.toHaveBeenCalled();
    expect(useAuthStore.getState().error).toMatch(/enter your email first/i);
  });

  it("surfaces a Firebase reset error as an honest message", async () => {
    resetMock.mockRejectedValue({ code: "auth/invalid-email" });
    await useAuthStore.getState().sendPasswordReset("nope");
    expect(useAuthStore.getState().error).toMatch(/isn't valid/i);
    expect(useAuthStore.getState().error).not.toMatch(/auth\//);
  });
});

describe("authStore — auth state listener (initAuth)", () => {
  it("populates user + cached token on sign-in and clears both on sign-out", async () => {
    initAuth();
    expect(typeof authMock.idTokenListener).toBe("function");

    // Firebase reports a signed-in user.
    const user = fakeUser("id-token-listener");
    authMock.currentUser = user;
    await authMock.idTokenListener!(user);

    expect(useAuthStore.getState().user).toEqual({
      uid: "user-1",
      email: "a@b.com",
      displayName: "Ada",
    });
    expect(useAuthStore.getState().initializing).toBe(false);
    expect(getCachedToken()).toBe("id-token-listener");

    // Firebase reports sign-out.
    authMock.currentUser = null;
    await authMock.idTokenListener!(null);

    expect(useAuthStore.getState().user).toBeNull();
    expect(getCachedToken()).toBeNull();
  });

  it("signOut delegates to Firebase", async () => {
    await useAuthStore.getState().signOut();
    expect(signOutMock).toHaveBeenCalledTimes(1);
  });
});
