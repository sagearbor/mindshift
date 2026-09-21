import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import LoginScreen from "../src/screens/LoginScreen";
import { useAuthStore } from "../src/store/authStore";
import {
  signInWithEmailAndPassword,
  sendPasswordResetEmail,
} from "firebase/auth";

const signInMock = signInWithEmailAndPassword as jest.Mock;
const resetMock = sendPasswordResetEmail as jest.Mock;

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

beforeEach(() => {
  signInMock.mockReset().mockResolvedValue(undefined);
  resetMock.mockReset().mockResolvedValue(undefined);
  useAuthStore.setState({
    user: null,
    initializing: false,
    error: null,
    notice: null,
    busy: false,
    pendingGoogleCredential: null,
    pendingGoogleEmail: null,
  });
});

describe("LoginScreen", () => {
  it("submits email + password through the store's signIn", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<LoginScreen />);
    });

    act(() => {
      queryId(comp, "email-input")!.props.onChangeText("user@example.com");
      queryId(comp, "password-input")!.props.onChangeText("secret123");
    });

    // signInWithEmailAndPassword is called synchronously at the top of the
    // store's signIn(), so awaiting the async act settles the rest of the chain.
    await act(async () => {
      queryId(comp, "submit-button")!.props.onPress();
    });

    expect(signInMock).toHaveBeenCalledWith(
      expect.anything(),
      "user@example.com",
      "secret123",
    );
    act(() => comp.unmount());
  });

  it("shows the store's honest error message", () => {
    act(() => {
      useAuthStore.setState({ error: "Incorrect email or password." });
    });
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<LoginScreen />);
    });
    expect(queryId(comp, "auth-error")!.props.children).toMatch(/incorrect/i);
    act(() => comp.unmount());
  });

  it("sends a password reset for the entered email and shows the confirmation", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<LoginScreen />);
    });
    act(() => {
      queryId(comp, "email-input")!.props.onChangeText("linda@example.com");
    });

    await act(async () => {
      queryId(comp, "forgot-password")!.props.onPress();
    });

    expect(resetMock).toHaveBeenCalledWith(expect.anything(), "linda@example.com");
    expect(queryId(comp, "auth-notice")!.props.children).toMatch(
      /reset email sent to linda@example\.com/i,
    );
    act(() => comp.unmount());
  });

  it("asks for an email first when Forgot password is tapped with none entered", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<LoginScreen />);
    });

    await act(async () => {
      queryId(comp, "forgot-password")!.props.onPress();
    });

    expect(resetMock).not.toHaveBeenCalled();
    expect(queryId(comp, "auth-error")!.props.children).toMatch(
      /enter your email first/i,
    );
    act(() => comp.unmount());
  });

  it("degrades to an honest 'not configured' state when Google isn't set up", () => {
    // No EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID in the test env.
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<LoginScreen />);
    });
    expect(queryId(comp, "google-unconfigured")).toBeTruthy();
    expect(queryId(comp, "google-button")).toBeNull();
    act(() => comp.unmount());
  });

  it("offers a working 'Continue with Apple' (Guideline 4.8 is mandatory given Google)", async () => {
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<LoginScreen />);
    });

    const appleButton = queryId(comp, "apple-button");
    expect(appleButton).toBeTruthy();
    // The old placeholder was disabled with no handler. Both are now the
    // failure condition: a dead "coming soon" control is itself a rejection
    // risk under Guideline 2.1, and 4.8 needs a real sign-in.
    expect(appleButton!.props.disabled).toBeFalsy();
    expect(typeof appleButton!.props.onPress).toBe("function");

    act(() => comp.unmount());
  });

  it("hands Firebase the RAW nonce and Apple the HASHED one, never the reverse", async () => {
    const appleMock = (globalThis as Record<string, unknown>)
      .__appleAuthMock as { signInAsync: jest.Mock };
    const signInWithAppleIdToken = jest.fn().mockResolvedValue(undefined);
    useAuthStore.setState({ signInWithAppleIdToken });

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<LoginScreen />);
    });
    await act(async () => {
      queryId(comp, "apple-button")!.props.onPress();
    });

    // Apple signs the hash; Firebase re-hashes the raw value and compares.
    expect(appleMock.signInAsync).toHaveBeenCalledWith(
      expect.objectContaining({ nonce: "sha256(raw-nonce-uuid)" }),
    );
    expect(signInWithAppleIdToken).toHaveBeenCalledWith(
      "apple-id-token",
      "raw-nonce-uuid",
    );

    act(() => comp.unmount());
  });

  it("treats a dismissed Apple sheet as a cancel, not an error", async () => {
    const appleMock = (globalThis as Record<string, unknown>)
      .__appleAuthMock as { signInAsync: jest.Mock };
    appleMock.signInAsync.mockRejectedValueOnce({
      code: "ERR_REQUEST_CANCELED",
    });

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<LoginScreen />);
    });
    await act(async () => {
      queryId(comp, "apple-button")!.props.onPress();
    });

    expect(useAuthStore.getState().error).toBeNull();

    act(() => comp.unmount());
  });

  it("is unaffected by the web hero banner on native (Task P3-4b, owner: both placements)", () => {
    // Default jest resolution is native (HeroWipe.native.tsx, a null stub —
    // see heroWipeHomeIntegration.test.tsx for the require.cache proof this
    // never even loads the web implementation or its images). The banner
    // slot still renders (empty), and the rest of the form is untouched.
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<LoginScreen />);
    });
    expect(queryId(comp, "hero-wipe")).toBeNull();
    expect(queryId(comp, "login-root")).toBeTruthy();
    expect(queryId(comp, "login-screen")).toBeTruthy();
    expect(queryId(comp, "email-input")).toBeTruthy();
    expect(queryId(comp, "submit-button")).toBeTruthy();
    act(() => comp.unmount());
  });
});
