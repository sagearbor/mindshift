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
});
