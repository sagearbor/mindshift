import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import * as SecureStore from "expo-secure-store";
import App from "../App";
import { useAuthStore } from "../src/store/authStore";
import { useSessionStore } from "../src/store/sessionStore";
import { useAnalyzeStore } from "../src/store/analyzeStore";

/**
 * Task P3-7 wiring at the App level: the walkthrough auto-shows once after
 * sign-in (gated on the persisted seen-flag, read through
 * src/utils/onboardingStorage — expo-secure-store on native, which is what
 * these tests exercise since jest's default RN haste platform is native),
 * and is separately re-runnable from Settings' "Show tutorial" row without
 * touching that persisted flag. App.test.tsx's navigation tests all default
 * the flag to "already seen" (see its beforeEach) so they land straight on
 * Home; this file covers the flag's two other states.
 */

/** The firebase/auth mock state from jest-setup. */
interface FirebaseAuthMock {
  currentUser: unknown;
  idTokenListener: ((user: unknown) => void | Promise<void>) | null;
}
const authMock = (globalThis as Record<string, unknown>)
  .__firebaseAuthMock as FirebaseAuthMock;

function fakeUser() {
  return {
    uid: "u1",
    email: "user@example.com",
    displayName: "Test User",
    getIdToken: jest.fn().mockResolvedValue("id-token"),
  };
}

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

/** Sign in and flush the (separate, expo-secure-store-backed) onboarding
 *  check — same two-step flush App.test.tsx's signIn() uses. */
async function signIn(comp: renderer.ReactTestRenderer) {
  const user = fakeUser();
  await act(async () => {
    authMock.currentUser = user;
    await authMock.idTokenListener?.(user);
  });
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

beforeEach(() => {
  authMock.currentUser = null;
  jest.clearAllMocks();
  useAuthStore.setState({
    user: null,
    initializing: true,
    error: null,
    busy: false,
  });
  act(() => {
    useSessionStore.setState({ turns: [], suggestions: [], loading: false });
    useAnalyzeStore.setState({ relationship: null });
  });
});

describe("App onboarding gate — first launch", () => {
  it("shows the walkthrough (not Home) when the seen-flag has never been set", async () => {
    (SecureStore.getItemAsync as jest.Mock).mockResolvedValue(null);

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    expect(queryId(comp, "onboarding-screen")).toBeTruthy();
    expect(queryId(comp, "home-live-coach")).toBeNull();
    act(() => comp.unmount());
  });

  it("Skip persists seen=true and reveals Home", async () => {
    (SecureStore.getItemAsync as jest.Mock).mockResolvedValue(null);
    (SecureStore.setItemAsync as jest.Mock).mockResolvedValue(undefined);

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);
    expect(queryId(comp, "onboarding-screen")).toBeTruthy();

    await act(async () => {
      queryId(comp, "onboarding-skip")!.props.onPress();
    });

    expect(queryId(comp, "onboarding-screen")).toBeNull();
    expect(queryId(comp, "home-live-coach")).toBeTruthy();
    expect(SecureStore.setItemAsync).toHaveBeenCalledWith(
      expect.any(String),
      "true",
    );
    act(() => comp.unmount());
  });

  it("Get started on the last card persists seen=true and reveals Home", async () => {
    (SecureStore.getItemAsync as jest.Mock).mockResolvedValue(null);
    (SecureStore.setItemAsync as jest.Mock).mockResolvedValue(undefined);

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    for (let i = 0; i < 3; i++) {
      await act(async () => {
        queryId(comp, "onboarding-next")!.props.onPress();
      });
    }
    expect(queryId(comp, "onboarding-get-started")).toBeTruthy();

    await act(async () => {
      queryId(comp, "onboarding-get-started")!.props.onPress();
    });

    expect(queryId(comp, "onboarding-screen")).toBeNull();
    expect(queryId(comp, "home-live-coach")).toBeTruthy();
    expect(SecureStore.setItemAsync).toHaveBeenCalledWith(
      expect.any(String),
      "true",
    );
    act(() => comp.unmount());
  });

  it("does not show the walkthrough when the seen-flag is already true", async () => {
    (SecureStore.getItemAsync as jest.Mock).mockResolvedValue("true");

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    expect(queryId(comp, "onboarding-screen")).toBeNull();
    expect(queryId(comp, "home-live-coach")).toBeTruthy();
    act(() => comp.unmount());
  });
});

describe("App onboarding gate — re-entry from Settings", () => {
  it("Settings → Show tutorial re-runs the walkthrough without touching the seen-flag, and Skip returns to Settings", async () => {
    // Already onboarded — Home is reached directly.
    (SecureStore.getItemAsync as jest.Mock).mockResolvedValue("true");

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);
    expect(queryId(comp, "home-live-coach")).toBeTruthy();

    await act(async () => {
      queryId(comp, "home-advanced-button")!.props.onPress();
    });
    const tutorialRow = queryId(comp, "advanced-show-tutorial");
    expect(tutorialRow).toBeTruthy();

    await act(async () => {
      tutorialRow!.props.onPress();
    });
    expect(queryId(comp, "onboarding-screen")).toBeTruthy();

    (SecureStore.setItemAsync as jest.Mock).mockClear();
    await act(async () => {
      queryId(comp, "onboarding-skip")!.props.onPress();
    });

    // Back on Settings, not Home — re-entry returns to where it was opened.
    expect(queryId(comp, "onboarding-screen")).toBeNull();
    expect(queryId(comp, "settings-heading")).toBeTruthy();
    // A manual replay is not a "you've now seen it" write — it was already
    // true before this re-entry.
    expect(SecureStore.setItemAsync).not.toHaveBeenCalled();

    act(() => comp.unmount());
  });
});
