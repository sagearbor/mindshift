/**
 * Guest mode ("Continue as guest" — Firebase Anonymous Auth) on the phone.
 *
 * Three promises, each proved from the real components rather than from the
 * store in isolation:
 *
 * 1. The login screen offers the guest path, and tapping it calls Firebase's
 *    `signInAnonymously` — not a fake local flag, not a skipped auth gate.
 * 2. A guest sees the persistent banner (with the honest sentence about where
 *    their data lives), a signed-up account does not, and the banner's
 *    "Create account" upgrade LINKS a credential onto the same anonymous uid
 *    so nothing recorded as a guest is stranded.
 * 3. In-app Calls are absent for a guest — the chip is not rendered at all,
 *    because a call needs a second real account on the other end — while
 *    every other session mode is still there.
 */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import {
  signInAnonymously,
  linkWithCredential,
  EmailAuthProvider,
} from "firebase/auth";
import LoginScreen from "../src/screens/LoginScreen";
import GuestBanner, { GUEST_BANNER_TEXT } from "../src/components/GuestBanner";
import LiveModePicker, {
  LIVE_MODE_OPTIONS,
} from "../src/components/LiveModePicker";
import { useAuthStore, type AuthUser } from "../src/store/authStore";

const anonMock = signInAnonymously as jest.Mock;
const linkMock = linkWithCredential as jest.Mock;
const emailCredMock = (EmailAuthProvider as unknown as {
  credential: jest.Mock;
}).credential;

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

/** The firebase/auth mock's shared state (see jest-setup.ts). */
function authMock(): { currentUser: unknown } {
  return (globalThis as Record<string, unknown>).__firebaseAuthMock as {
    currentUser: unknown;
  };
}

const GUEST: AuthUser = {
  uid: "guest-1",
  email: null,
  displayName: null,
  isAnonymous: true,
};
const REAL: AuthUser = {
  uid: "u1",
  email: "sage@example.com",
  displayName: "Sage",
  isAnonymous: false,
};

function resetStore(user: AuthUser | null) {
  useAuthStore.setState({
    user,
    initializing: false,
    error: null,
    notice: null,
    busy: false,
    pendingGoogleCredential: null,
    pendingGoogleEmail: null,
  });
}

beforeEach(() => {
  anonMock.mockReset().mockResolvedValue(undefined);
  linkMock.mockReset().mockResolvedValue(undefined);
  emailCredMock.mockClear();
  authMock().currentUser = null;
  resetStore(null);
});

describe("LoginScreen — Continue as guest", () => {
  it("offers the guest button and explains what a guest session is", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<LoginScreen />);
    });

    const button = queryId(comp, "continue-as-guest");
    expect(button).toBeTruthy();
    expect(button!.props.accessibilityRole).toBe("button");
    expect(button!.props.accessibilityLabel).toMatch(/continue as guest/i);

    // The button must not be the ONLY thing that says what this costs.
    expect(queryId(comp, "guest-explainer")!.props.children).toBeTruthy();

    act(() => comp.unmount());
  });

  it("signs in anonymously through Firebase when tapped", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<LoginScreen />);
    });

    await act(async () => {
      queryId(comp, "continue-as-guest")!.props.onPress();
    });

    expect(anonMock).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });

  it("shows the store's honest error when anonymous sign-in is disabled", async () => {
    // Firebase reports a disabled Anonymous provider as admin-restricted.
    anonMock.mockRejectedValue({ code: "auth/admin-restricted-operation" });
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<LoginScreen />);
    });

    await act(async () => {
      queryId(comp, "continue-as-guest")!.props.onPress();
    });

    expect(queryId(comp, "auth-error")!.props.children).toMatch(
      /isn't enabled for the app yet/i,
    );
    act(() => comp.unmount());
  });
});

describe("GuestBanner", () => {
  it("renders nothing for a signed-up account", () => {
    resetStore(REAL);
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<GuestBanner />);
    });
    expect(queryId(comp, "guest-banner")).toBeNull();
    act(() => comp.unmount());
  });

  it("renders nothing when signed out", () => {
    resetStore(null);
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<GuestBanner />);
    });
    expect(queryId(comp, "guest-banner")).toBeNull();
    act(() => comp.unmount());
  });

  it("tells a guest where their data lives, and offers the upgrade", () => {
    resetStore(GUEST);
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<GuestBanner />);
    });

    expect(queryId(comp, "guest-banner")).toBeTruthy();
    expect(queryId(comp, "guest-banner-text")!.props.children).toBe(
      GUEST_BANNER_TEXT,
    );
    // The exact promises the copy has to make, asserted rather than assumed.
    expect(GUEST_BANNER_TEXT).toMatch(/this device/i);
    expect(GUEST_BANNER_TEXT).toMatch(/create an account/i);

    // Collapsed until asked for — the banner must not eat the screen.
    expect(queryId(comp, "guest-upgrade-form")).toBeNull();
    expect(queryId(comp, "guest-create-account")).toBeTruthy();

    act(() => comp.unmount());
  });

  it("links an email/password credential onto the SAME anonymous uid", async () => {
    resetStore(GUEST);
    const guestUser = { uid: "guest-1", isAnonymous: true };
    authMock().currentUser = guestUser;

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<GuestBanner />);
    });

    act(() => {
      queryId(comp, "guest-create-account")!.props.onPress();
    });
    expect(queryId(comp, "guest-upgrade-form")).toBeTruthy();

    act(() => {
      queryId(comp, "guest-email-input")!.props.onChangeText("new@example.com");
      queryId(comp, "guest-password-input")!.props.onChangeText("secret123");
    });

    await act(async () => {
      queryId(comp, "guest-upgrade-submit")!.props.onPress();
    });

    expect(emailCredMock).toHaveBeenCalledWith("new@example.com", "secret123");
    // linkWithCredential, NOT createUserWithEmailAndPassword: the uid must
    // survive so the guest's recordings come with them.
    expect(linkMock).toHaveBeenCalledTimes(1);
    expect(linkMock.mock.calls[0][0]).toBe(guestUser);
    expect(queryId(comp, "guest-notice")!.props.children).toMatch(
      /account created/i,
    );

    act(() => comp.unmount());
  });

  it("says plainly that two accounts cannot be merged", async () => {
    resetStore(GUEST);
    authMock().currentUser = { uid: "guest-1", isAnonymous: true };
    linkMock.mockRejectedValue({ code: "auth/email-already-in-use" });

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<GuestBanner />);
    });
    act(() => {
      queryId(comp, "guest-create-account")!.props.onPress();
    });
    act(() => {
      queryId(comp, "guest-email-input")!.props.onChangeText("taken@example.com");
      queryId(comp, "guest-password-input")!.props.onChangeText("secret123");
    });
    await act(async () => {
      queryId(comp, "guest-upgrade-submit")!.props.onPress();
    });

    const message = String(queryId(comp, "guest-error")!.props.children);
    expect(message).toMatch(/can't merge/i);
    // It must name the real alternative rather than just failing.
    expect(message).toMatch(/different email|log out/i);

    act(() => comp.unmount());
  });
});

describe("LiveModePicker — Calls hidden for guests", () => {
  it("drops only the Call chip and keeps every other mode", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <LiveModePicker
          value="earpiece"
          onChange={jest.fn()}
          hiddenModes={["call"]}
        />,
      );
    });

    expect(queryId(comp, "session-mode-call")).toBeNull();
    for (const o of LIVE_MODE_OPTIONS.filter((x) => x.mode !== "call")) {
      expect(queryId(comp, `session-mode-${o.mode}`)).toBeTruthy();
    }

    act(() => comp.unmount());
  });

  it("still renders all five modes when nothing is hidden (the signed-up case)", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <LiveModePicker value="earpiece" onChange={jest.fn()} />,
      );
    });
    for (const o of LIVE_MODE_OPTIONS) {
      expect(queryId(comp, `session-mode-${o.mode}`)).toBeTruthy();
    }
    act(() => comp.unmount());
  });

  it("falls back to a VISIBLE mode's hint when the selected one is hidden", () => {
    // A guest whose remembered mode was "call": the hint must describe a chip
    // that is actually on screen, never the one that was removed.
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <LiveModePicker
          value="call"
          onChange={jest.fn()}
          hiddenModes={["call"]}
        />,
      );
    });
    const callHint = LIVE_MODE_OPTIONS.find((o) => o.mode === "call")!.hint;
    expect(queryId(comp, "session-mode-hint")!.props.children).not.toBe(
      callHint,
    );
    act(() => comp.unmount());
  });
});
