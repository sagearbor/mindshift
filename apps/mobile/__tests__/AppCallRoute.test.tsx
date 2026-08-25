import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import * as SecureStore from "expo-secure-store";
import App from "../App";
import { useAuthStore } from "../src/store/authStore";
import { useLayoutStore } from "../src/store/layoutStore";
import { useAvatarStore } from "../src/store/avatarStore";

/**
 * The /call/<code> route (web) and the mindshift://call/<code> deep link:
 * the code waits for sign-in, then opens Live Coach in Call mode with an
 * Answer button; a plain launch still lands on Home.
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
  (global.fetch as jest.Mock).mockResolvedValue({ ok: true, status: 200, json: async () => ({}) });
  useAuthStore.setState({ user: null, initializing: true, error: null, busy: false });
  act(() => {
    useLayoutStore.getState().resetToDefaults();
    useAvatarStore.setState({ uri: null, hydrated: false });
  });
});

describe("App — call invite route", () => {
  it("opens Live Coach in Call mode with an Answer button once signed in", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App initialUrl="https://arborfam-hub.web.app/call/K7M2PQ" />);
    });
    // Signed out: the invite waits behind the login gate.
    await act(async () => {
      await authMock.idTokenListener?.(null);
    });
    expect(queryId(comp, "login-screen")).toBeTruthy();
    await signIn();
    expect(queryId(comp, "call-invited")).toBeTruthy();
    expect(queryId(comp, "call-answer")).toBeTruthy();
    expect(JSON.stringify(comp.toJSON())).toContain("K7M2PQ");
    act(() => comp.unmount());
  });

  it("accepts the app's own scheme too, and a plain launch lands on Home", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App initialUrl="mindshift://call/ABC123" />);
    });
    await signIn();
    expect(queryId(comp, "call-answer")).toBeTruthy();
    act(() => comp.unmount());

    let plain!: renderer.ReactTestRenderer;
    act(() => {
      plain = renderer.create(<App initialUrl={null} />);
    });
    await signIn();
    expect(queryId(plain, "home-screen")).toBeTruthy();
    expect(queryId(plain, "call-answer")).toBeNull();
    act(() => plain.unmount());
  });
});
