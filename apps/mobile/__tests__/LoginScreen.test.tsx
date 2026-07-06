import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import LoginScreen from "../src/screens/LoginScreen";
import { useAuthStore } from "../src/store/authStore";
import { signInWithEmailAndPassword } from "firebase/auth";

const signInMock = signInWithEmailAndPassword as jest.Mock;

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

beforeEach(() => {
  signInMock.mockReset().mockResolvedValue(undefined);
  useAuthStore.setState({
    user: null,
    initializing: false,
    error: null,
    busy: false,
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
