import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import App from "../App";
import { useAuthStore } from "../src/store/authStore";
import { useSessionStore } from "../src/store/sessionStore";

const mockFetch = global.fetch as jest.Mock;

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

/** First node carrying the given testID, or null. */
function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

beforeEach(() => {
  authMock.currentUser = null;
  useAuthStore.setState({
    user: null,
    initializing: true,
    error: null,
    busy: false,
  });
});

describe("App auth gate", () => {
  it("shows a loading state until the first auth resolution", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    // No auth event yet: neither login nor the app, just the spinner.
    expect(queryId(comp, "auth-loading")).toBeTruthy();
    expect(queryId(comp, "login-screen")).toBeNull();
    expect(queryId(comp, "tab-session")).toBeNull();
    act(() => comp.unmount());
  });

  it("shows the login screen when signed out", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });

    // Firebase resolves auth state to "no user".
    await act(async () => {
      await authMock.idTokenListener?.(null);
    });

    expect(queryId(comp, "login-screen")).toBeTruthy();
    // The app's tab tree is NOT reachable while signed out.
    expect(queryId(comp, "tab-session")).toBeNull();
    act(() => comp.unmount());
  });

  it("reaches the app (tab tree) when signed in", async () => {
    const user = fakeUser();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });

    await act(async () => {
      authMock.currentUser = user;
      await authMock.idTokenListener?.(user);
    });

    // The existing coaching UX is intact: the tab bar + Session screen render,
    // and the login screen is gone.
    expect(queryId(comp, "tab-session")).toBeTruthy();
    expect(queryId(comp, "tab-sign-out")).toBeTruthy();
    expect(queryId(comp, "login-screen")).toBeNull();
    act(() => comp.unmount());
  });

  it("navigates from the Session tab to the pushed Dynamics screen", async () => {
    const user = fakeUser();
    // Enough turns for the analyze button to appear.
    act(() => {
      useSessionStore.setState({
        turns: [
          { speaker: "Alice", text: "a" },
          { speaker: "Bob", text: "b" },
          { speaker: "Alice", text: "c" },
          { speaker: "Bob", text: "d" },
        ],
      });
    });
    // DynamicsScreen fetches /analyze on mount; return a contract-valid, minimal
    // result so it lands on its content view.
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        per_turn: [],
        per_speaker: {},
        dynamics: {
          coupling: { strength: null, leader: null, description: "" },
          deescalation: { who_first: null, follow_rate: null, description: "" },
          triggers: [],
          requests: [],
        },
        narrative: "",
      }),
    });

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await act(async () => {
      authMock.currentUser = user;
      await authMock.idTokenListener?.(user);
    });

    // Press "Analyze dynamics →" and confirm we land on the pushed screen with
    // the tab bar hidden.
    await act(async () => {
      queryId(comp, "analyze-dynamics-button")?.props.onPress();
    });
    expect(queryId(comp, "dynamics-back")).toBeTruthy();
    expect(queryId(comp, "tab-session")).toBeNull();
    act(() => comp.unmount());
  });
});
