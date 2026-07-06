import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithCredential,
  signOut as fbSignOut,
  GoogleAuthProvider,
} from "firebase/auth";
import { useAuthStore, initAuth } from "../src/store/authStore";
import { getCachedToken } from "../src/auth/authToken";

const signInMock = signInWithEmailAndPassword as jest.Mock;
const signUpMock = createUserWithEmailAndPassword as jest.Mock;
const signInCredMock = signInWithCredential as jest.Mock;
const signOutMock = fbSignOut as jest.Mock;
const credentialMock = GoogleAuthProvider.credential as jest.Mock;

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
  signOutMock.mockReset().mockResolvedValue(undefined);
  credentialMock.mockClear();
  authMock.currentUser = null;
  useAuthStore.setState({
    user: null,
    initializing: true,
    error: null,
    busy: false,
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
  it("exchanges a Google ID token for a Firebase credential", async () => {
    await useAuthStore.getState().signInWithGoogleIdToken("google-tok");

    expect(credentialMock).toHaveBeenCalledWith("google-tok");
    expect(signInCredMock).toHaveBeenCalledTimes(1);
    expect(useAuthStore.getState().error).toBeNull();
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
