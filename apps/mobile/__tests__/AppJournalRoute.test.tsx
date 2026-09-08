import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import * as SecureStore from "expo-secure-store";
import App from "../App";
import { useAuthStore } from "../src/store/authStore";
import { useLayoutStore } from "../src/store/layoutStore";
import { useAvatarStore } from "../src/store/avatarStore";

/**
 * The mindshift://journal/start deep link ("Hey Google, start my journal"):
 * waits behind the sign-in gate like a call invite, then opens Live Coach
 * with Journal mode selected. With no enrolled owner voiceprint (this test's
 * server answers with no people) the screen lands on the journal panel with
 * its gate visible instead of silently starting — the gate-respecting path,
 * end to end through the real screen.
 */
interface FirebaseAuthMock {
  currentUser: unknown;
  idTokenListener: ((user: unknown) => void | Promise<void>) | null;
}
const authMock = (globalThis as Record<string, unknown>).__firebaseAuthMock as FirebaseAuthMock;

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

async function signIn() {
  const user = { uid: "u1", email: "user@example.com", displayName: "T", getIdToken: jest.fn().mockResolvedValue("t") };
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
  (SecureStore.getItemAsync as jest.Mock).mockResolvedValue("true");
  // The people endpoint must answer with a REAL empty roster: since the
  // 2026-08-31 hardening a 200 without a people array reads as "server not
  // ready" (gate "unknown", no banner) — this test is about the honest
  // "missing" gate, which needs people: [] with no self.
  (global.fetch as jest.Mock).mockImplementation(async (url: unknown) => ({
    ok: true,
    status: 200,
    json: async () =>
      String(url).includes("/voice/people")
        ? { available: true, storage_enabled: true, people: [] }
        : {},
  }));
  useAuthStore.setState({ user: null, initializing: true, error: null, busy: false });
  act(() => {
    useLayoutStore.getState().resetToDefaults();
    useAvatarStore.setState({ uri: null, hydrated: false });
  });
});

describe("App — journal deep link", () => {
  it("mindshift://journal/start opens Live Coach in Journal mode once signed in", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App initialUrl="mindshift://journal/start" />);
    });
    // Signed out: the action waits behind the login gate.
    await act(async () => {
      await authMock.idTokenListener?.(null);
    });
    expect(queryId(comp, "login-screen")).toBeTruthy();
    await signIn();
    // Extra settles for the screen's own mode-load + people-gate promises
    // (and App's developer-mode hydrate, one more async effect in the chain).
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(queryId(comp, "home-screen")).toBeNull();
    // Journal mode is selected (its panel renders), and — with no enrolled
    // owner print on this fake server — the gate message shows instead of a
    // silently started session.
    expect(queryId(comp, "journal-panel")).toBeTruthy();
    expect(queryId(comp, "journal-gate")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("a plain launch still lands on Home with no journal panel", async () => {
    let plain!: renderer.ReactTestRenderer;
    act(() => {
      plain = renderer.create(<App initialUrl={null} />);
    });
    await signIn();
    expect(queryId(plain, "home-screen")).toBeTruthy();
    expect(queryId(plain, "journal-panel")).toBeNull();
    act(() => plain.unmount());
  });
});
