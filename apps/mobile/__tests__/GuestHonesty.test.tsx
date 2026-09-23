/**
 * Guest mode's two honesty gaps, both found on an emulator and both on the
 * path every app-store reviewer walks.
 *
 * 1. "Log out" was a one-tap, unconfirmed, IRREVERSIBLE account loss for a
 *    guest: an anonymous account has no credential to sign back in with, so
 *    the next "Continue as guest" mints a different uid and the old one's
 *    sessions are unreachable forever. Two rows below it, "Delete my account"
 *    makes you type DELETE. A signed-up account loses nothing by logging out
 *    and must NOT be given the same friction.
 *
 * 2. A guest was never told the quota existed until it bit mid-conversation.
 *    The numbers are the server's (server/guest_quota.py, env-configurable)
 *    and the server already sends them; the app now states them BEFORE a
 *    session, from the wire, never from a constant.
 */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import * as SecureStore from "expo-secure-store";
import { signOut as fbSignOut } from "firebase/auth";
import App from "../App";
import GuestBanner, {
  GUEST_LIMITS_UNKNOWN_TEXT,
  guestLimitsLine,
} from "../src/components/GuestBanner";
import { useAuthStore, type AuthUser } from "../src/store/authStore";
import { useAvatarStore } from "../src/store/avatarStore";
import { useLayoutStore } from "../src/store/layoutStore";
import {
  GUEST_LIMITS_KEY,
  UNKNOWN_GUEST_LIMITS,
  loadGuestLimits,
  readGuestLimits,
  useGuestLimitsStore,
} from "../src/store/guestLimitsStore";

interface FirebaseAuthMock {
  currentUser: unknown;
  idTokenListener: ((user: unknown) => void | Promise<void>) | null;
}
const authMock = (globalThis as Record<string, unknown>)
  .__firebaseAuthMock as FirebaseAuthMock;

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

const GUEST: AuthUser = {
  uid: "guest-1",
  email: null,
  displayName: null,
  isAnonymous: true,
};

/** A guest's Firebase user: no email, no displayName, isAnonymous true. */
function fakeGuest() {
  return {
    uid: "guest-1",
    email: null,
    displayName: null,
    isAnonymous: true,
    getIdToken: jest.fn().mockResolvedValue("id-token"),
  };
}

function fakeAccount() {
  return {
    uid: "u1",
    email: "user@example.com",
    displayName: "Test User",
    isAnonymous: false,
    getIdToken: jest.fn().mockResolvedValue("id-token"),
  };
}

/** Boot App and resolve auth as `user`, landing on Home. Mirrors the helper
 *  in App.test.tsx (onboarding-seen is defaulted to true in beforeEach). */
async function boot(user: unknown) {
  let comp!: renderer.ReactTestRenderer;
  act(() => {
    comp = renderer.create(<App />);
  });
  await act(async () => {
    authMock.currentUser = user;
    await authMock.idTokenListener?.(user);
  });
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
  expect(queryId(comp, "home-screen")).toBeTruthy();
  return comp;
}

/** Open the avatar menu and tap its "Log out" row. */
async function tapAvatarLogOut(comp: renderer.ReactTestRenderer) {
  await act(async () => {
    queryId(comp, "chrome-avatar-button")!.props.onPress();
  });
  await act(async () => {
    queryId(comp, "chrome-account-sign-out")!.props.onPress();
  });
}

beforeEach(() => {
  authMock.currentUser = null;
  (SecureStore.getItemAsync as jest.Mock).mockResolvedValue("true");
  (SecureStore.setItemAsync as jest.Mock).mockResolvedValue(undefined);
  (fbSignOut as jest.Mock).mockClear();
  useAuthStore.setState({
    user: null,
    initializing: true,
    error: null,
    notice: null,
    busy: false,
  });
  act(() => {
    useLayoutStore.getState().resetToDefaults();
    useAvatarStore.setState({ uri: null, hydrated: false });
    useGuestLimitsStore.setState({ ...UNKNOWN_GUEST_LIMITS });
  });
});

describe("Log out — a guest is warned, an account holder is not", () => {
  it("asks a GUEST first, and does not sign them out until they confirm", async () => {
    const comp = await boot(fakeGuest());

    await tapAvatarLogOut(comp);

    const card = queryId(comp, "guest-sign-out-confirm");
    expect(card).toBeTruthy();
    // Nothing has happened yet — the tap opened a question, not a door.
    expect(fbSignOut as jest.Mock).not.toHaveBeenCalled();

    // The warning has to say the thing that makes this irreversible, and must
    // not claim the data is "deleted" (it isn't — it's stranded).
    const body = String(queryId(comp, "guest-sign-out-body")!.props.children);
    expect(body).toMatch(/no way to sign back in/i);
    expect(body).toMatch(/doesn't delete/i);
    // ...and it has to name the way out that keeps everything.
    expect(
      String(queryId(comp, "guest-sign-out-keep")!.props.children),
    ).toMatch(/create account/i);

    act(() => comp.unmount());
  });

  it("signs the guest out for real once they confirm", async () => {
    const comp = await boot(fakeGuest());
    await tapAvatarLogOut(comp);

    await act(async () => {
      queryId(comp, "guest-sign-out-confirm-button")!.props.onPress();
    });

    expect(fbSignOut as jest.Mock).toHaveBeenCalledTimes(1);
    expect(queryId(comp, "guest-sign-out-confirm")).toBeNull();
    act(() => comp.unmount());
  });

  it("cancelling keeps the guest signed in", async () => {
    const comp = await boot(fakeGuest());
    await tapAvatarLogOut(comp);

    await act(async () => {
      queryId(comp, "guest-sign-out-cancel")!.props.onPress();
    });

    expect(fbSignOut as jest.Mock).not.toHaveBeenCalled();
    expect(queryId(comp, "guest-sign-out-confirm")).toBeNull();
    act(() => comp.unmount());
  });

  it("a signed-up account logs straight out — no confirm, no new friction", async () => {
    const comp = await boot(fakeAccount());

    await tapAvatarLogOut(comp);

    expect(queryId(comp, "guest-sign-out-confirm")).toBeNull();
    expect(fbSignOut as jest.Mock).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });

  it("guards Settings' own Log out row the same way (one gate, both entry points)", async () => {
    const comp = await boot(fakeGuest());
    await act(async () => {
      queryId(comp, "chrome-avatar-button")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "chrome-account-settings")!.props.onPress();
    });

    await act(async () => {
      queryId(comp, "advanced-sign-out")!.props.onPress();
    });

    expect(queryId(comp, "guest-sign-out-confirm")).toBeTruthy();
    expect(fbSignOut as jest.Mock).not.toHaveBeenCalled();
    act(() => comp.unmount());
  });
});

describe("guestLimitsStore — the numbers come off the wire", () => {
  it("reads only whole positive integers out of a frame", () => {
    expect(
      readGuestLimits({ max_sessions_per_day: 5, max_session_minutes: 12 }),
    ).toEqual({ maxSessionsPerDay: 5, maxSessionMinutes: 12 });
    // Junk is refused rather than repeated to the user.
    expect(
      readGuestLimits({ max_sessions_per_day: 0, max_session_minutes: -1 }),
    ).toEqual(UNKNOWN_GUEST_LIMITS);
    expect(
      readGuestLimits({ max_sessions_per_day: "3", max_session_minutes: 1.5 }),
    ).toEqual(UNKNOWN_GUEST_LIMITS);
    expect(readGuestLimits(null)).toEqual(UNKNOWN_GUEST_LIMITS);
    expect(readGuestLimits("guest_limit")).toEqual(UNKNOWN_GUEST_LIMITS);
  });

  it("learns, persists, and never lets a partial frame erase what it knows", async () => {
    const setItem = SecureStore.setItemAsync as jest.Mock;
    setItem.mockClear();

    act(() => {
      useGuestLimitsStore
        .getState()
        .learnFromServer({ max_sessions_per_day: 5, max_session_minutes: 12 });
    });
    expect(useGuestLimitsStore.getState().maxSessionsPerDay).toBe(5);
    expect(useGuestLimitsStore.getState().maxSessionMinutes).toBe(12);
    expect(setItem).toHaveBeenCalledWith(
      GUEST_LIMITS_KEY,
      JSON.stringify({ max_sessions_per_day: 5, max_session_minutes: 12 }),
    );

    // A frame with only one number updates only that one.
    act(() => {
      useGuestLimitsStore.getState().learnFromServer({ max_session_minutes: 8 });
    });
    expect(useGuestLimitsStore.getState().maxSessionsPerDay).toBe(5);
    expect(useGuestLimitsStore.getState().maxSessionMinutes).toBe(8);

    // A frame with neither changes nothing at all.
    act(() => {
      useGuestLimitsStore.getState().learnFromServer({ message: "nope" });
    });
    expect(useGuestLimitsStore.getState().maxSessionsPerDay).toBe(5);
    expect(useGuestLimitsStore.getState().maxSessionMinutes).toBe(8);
  });

  it("hydrates the cached numbers, and reads unknown when there are none", async () => {
    const getItem = SecureStore.getItemAsync as jest.Mock;

    getItem.mockResolvedValue(
      JSON.stringify({ max_sessions_per_day: 2, max_session_minutes: 6 }),
    );
    await act(async () => {
      await useGuestLimitsStore.getState().hydrate();
    });
    expect(useGuestLimitsStore.getState().maxSessionsPerDay).toBe(2);
    expect(useGuestLimitsStore.getState().maxSessionMinutes).toBe(6);

    // Unreadable storage is "we don't know", never a fabricated default.
    getItem.mockResolvedValue("not json");
    expect(await loadGuestLimits()).toEqual(UNKNOWN_GUEST_LIMITS);
    getItem.mockRejectedValue(new Error("keystore locked"));
    expect(await loadGuestLimits()).toEqual(UNKNOWN_GUEST_LIMITS);
  });
});

describe("GuestBanner — the limits are stated before one bites", () => {
  it("renders the SERVER's numbers, not constants", () => {
    // 7/4 is nothing like the 3/10 defaults on purpose: an app that hardcoded
    // the defaults would render "3"/"10" here and fail.
    act(() => {
      useAuthStore.setState({ user: GUEST, initializing: false });
      useGuestLimitsStore.setState({
        maxSessionsPerDay: 7,
        maxSessionMinutes: 4,
      });
    });

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<GuestBanner />);
    });

    const line = String(queryId(comp, "guest-limits-text")!.props.children);
    expect(line).toBe("Guest limit: 7 live sessions a day, 4 minutes each.");
    expect(line).not.toMatch(/\b3\b|\b10\b/);

    act(() => comp.unmount());
  });

  it("says the limits exist without inventing them when the server hasn't said", () => {
    act(() => {
      useAuthStore.setState({ user: GUEST, initializing: false });
    });
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<GuestBanner />);
    });

    const line = String(queryId(comp, "guest-limits-text")!.props.children);
    expect(line).toBe(GUEST_LIMITS_UNKNOWN_TEXT);
    // The honest default names no number at all.
    expect(line).not.toMatch(/\d/);

    act(() => comp.unmount());
  });

  it("stays absent for a signed-up account", () => {
    act(() => {
      useAuthStore.setState({
        user: {
          uid: "u1",
          email: "user@example.com",
          displayName: "Test User",
          isAnonymous: false,
        },
        initializing: false,
      });
      useGuestLimitsStore.setState({
        maxSessionsPerDay: 7,
        maxSessionMinutes: 4,
      });
    });
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<GuestBanner />);
    });
    expect(queryId(comp, "guest-limits-text")).toBeNull();
    act(() => comp.unmount());
  });

  it("reads correctly for every shape of what the server told us", () => {
    expect(guestLimitsLine(3, 10)).toBe(
      "Guest limit: 3 live sessions a day, 10 minutes each.",
    );
    expect(guestLimitsLine(1, 1)).toBe(
      "Guest limit: 1 live session a day, 1 minute each.",
    );
    expect(guestLimitsLine(3, null)).toBe("Guest limit: 3 live sessions a day.");
    expect(guestLimitsLine(null, 10)).toBe("Guest limit: 10 minutes each.");
    expect(guestLimitsLine(null, null)).toBe(GUEST_LIMITS_UNKNOWN_TEXT);
  });
});
