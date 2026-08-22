import React from "react";
import { Alert, Linking } from "react-native";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import WatchSetupScreen from "../src/screens/WatchSetupScreen";
import { claimWatchPairing, disconnectWatch } from "../src/api/watchPairing";
import { getMe } from "../src/api/me";

jest.mock("../src/api/watchPairing", () => ({
  claimWatchPairing: jest.fn(),
  disconnectWatch: jest.fn(),
}));
jest.mock("../src/api/me", () => ({
  getMe: jest.fn(),
}));
const mockClaim = claimWatchPairing as jest.Mock;
const mockDisconnect = disconnectWatch as jest.Mock;
const mockGetMe = getMe as jest.Mock;

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

function textOf(node: ReactTestInstance): string {
  return node
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");
}

async function render(onBack = jest.fn()) {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<WatchSetupScreen onBack={onBack} />);
  });
  return comp;
}

beforeEach(() => {
  mockClaim.mockReset();
  mockDisconnect.mockReset();
  mockGetMe.mockReset();
  // Default: honest "unknown" until a test overrides it — matches the
  // screen's own default before the fetch resolves/rejects.
  mockGetMe.mockResolvedValue({
    account_id: "u1",
    email: "a@example.com",
    legacy: false,
    has_paired_watch: false,
  });
  jest.spyOn(Linking, "openURL").mockResolvedValue(true);
});

afterEach(() => {
  jest.restoreAllMocks();
});

describe("WatchSetupScreen", () => {
  it("renders both numbered steps, the install button, pair section, and wires back", async () => {
    const onBack = jest.fn();
    const comp = await render(onBack);

    expect(queryId(comp, "watch-setup-screen")).toBeTruthy();
    expect(queryId(comp, "watch-install-button")).toBeTruthy();
    expect(queryId(comp, "watch-pair-code-input")).toBeTruthy();
    expect(queryId(comp, "watch-pair-button")).toBeTruthy();
    // No stale success/error/opened state up front.
    expect(queryId(comp, "watch-install-opened")).toBeNull();
    expect(queryId(comp, "watch-pair-success")).toBeNull();
    expect(queryId(comp, "watch-pair-error")).toBeNull();

    act(() => queryId(comp, "watch-setup-back")!.props.onPress());
    expect(onBack).toHaveBeenCalledTimes(1);

    act(() => comp.unmount());
  });

  it("opens the Play listing for the watch app on Install and confirms it stepped forward", async () => {
    const comp = await render();

    await act(async () => {
      queryId(comp, "watch-install-button")!.props.onPress();
    });

    expect(Linking.openURL).toHaveBeenCalledWith(
      "https://play.google.com/store/apps/details?id=com.sagearbor.gauge.wear",
    );
    // Step 1 steps back visually and confirms what happened — the button
    // stays usable (re-opening Play is legitimate) rather than disappearing.
    expect(queryId(comp, "watch-install-opened")).toBeTruthy();
    expect(queryId(comp, "watch-install-button")).toBeTruthy();

    act(() => comp.unmount());
  });

  it("disables Pair until exactly 6 characters are entered, and uppercases input", async () => {
    mockClaim.mockReturnValue(new Promise(() => {})); // never resolves in this test
    const comp = await render();
    const input = queryId(comp, "watch-pair-code-input")!;
    const button = () => queryId(comp, "watch-pair-button")!;

    expect(button().props.disabled).toBe(true);

    act(() => input.props.onChangeText("ab3"));
    expect(button().props.disabled).toBe(true);
    expect(mockClaim).not.toHaveBeenCalled();

    act(() => input.props.onChangeText("ab3xyz"));
    // Auto-caps client-side regardless of the TextInput's own autoCapitalize.
    expect(queryId(comp, "watch-pair-code-input")!.props.value).toBe("AB3XYZ");

    act(() => comp.unmount());
  });

  it("auto-submits the moment a valid 6-character code is typed", async () => {
    let resolveClaim!: (v: unknown) => void;
    mockClaim.mockReturnValue(
      new Promise((resolve) => {
        resolveClaim = resolve;
      }),
    );
    const comp = await render();
    const input = queryId(comp, "watch-pair-code-input")!;

    act(() => input.props.onChangeText("abc123"));

    expect(mockClaim).toHaveBeenCalledTimes(1);
    expect(mockClaim).toHaveBeenCalledWith("ABC123");
    // Loading state on the Pair button while the request is in flight.
    expect(queryId(comp, "watch-pair-button")!.props.disabled).toBe(true);

    await act(async () => {
      resolveClaim({ ok: true });
    });
    expect(textOf(queryId(comp, "watch-pair-success")!)).toContain(
      "Watch paired",
    );

    act(() => comp.unmount());
  });

  it("does not call the client while fewer than 6 characters are entered", async () => {
    const comp = await render();
    const input = queryId(comp, "watch-pair-code-input")!;

    act(() => input.props.onChangeText("ABC12"));

    expect(mockClaim).not.toHaveBeenCalled();
    act(() => comp.unmount());
  });

  it("shows the success state with an honest next-step note (no polling implied)", async () => {
    mockClaim.mockResolvedValue({ ok: true });
    const comp = await render();
    const input = queryId(comp, "watch-pair-code-input")!;

    await act(async () => {
      input.props.onChangeText("abc123");
    });

    expect(textOf(queryId(comp, "watch-pair-success")!)).toContain(
      "Watch paired",
    );
    expect(textOf(queryId(comp, "watch-pair-success")!)).toMatch(/10 s/i);
    act(() => comp.unmount());
  });

  it("locks the code input and Pair button after a successful claim (review Minor 1)", async () => {
    mockClaim.mockResolvedValue({ ok: true });
    const comp = await render();
    const input = queryId(comp, "watch-pair-code-input")!;

    await act(async () => {
      input.props.onChangeText("abc123");
    });

    expect(mockClaim).toHaveBeenCalledTimes(1);
    // Both are locked so a stray re-tap/re-edit can't fire a redundant claim
    // that would 409 and burn one of the account's lockout-budget attempts.
    expect(queryId(comp, "watch-pair-code-input")!.props.editable).toBe(false);
    expect(queryId(comp, "watch-pair-button")!.props.disabled).toBe(true);

    // Even an explicit press does nothing further — no second claim call.
    await act(async () => {
      queryId(comp, "watch-pair-button")!.props.onPress();
    });
    expect(mockClaim).toHaveBeenCalledTimes(1);

    act(() => comp.unmount());
  });

  it("gives the code field an accessibility label (review Minor 2)", async () => {
    const comp = await render();
    const input = queryId(comp, "watch-pair-code-input")!;

    expect(input.props.accessibilityLabel).toBe(
      "6-character pairing code from your watch",
    );

    act(() => comp.unmount());
  });

  it("shows the server's honest error and lets the user retry via the Pair button", async () => {
    mockClaim.mockResolvedValue({
      ok: false,
      detail: "too many failed pairing attempts on this account",
    });
    const comp = await render();
    const input = queryId(comp, "watch-pair-code-input")!;

    await act(async () => {
      input.props.onChangeText("abc123");
    });

    expect(textOf(queryId(comp, "watch-pair-error")!)).toContain(
      "too many failed pairing attempts on this account",
    );
    expect(queryId(comp, "watch-pair-success")).toBeNull();
    // Not disabled — the user can correct the code and try again.
    expect(queryId(comp, "watch-pair-button")!.props.disabled).toBe(false);

    // Explicit retry works too, not just auto-submit.
    mockClaim.mockResolvedValueOnce({ ok: true });
    await act(async () => {
      queryId(comp, "watch-pair-button")!.props.onPress();
    });
    expect(mockClaim).toHaveBeenLastCalledWith("ABC123");
    expect(textOf(queryId(comp, "watch-pair-success")!)).toContain(
      "Watch paired",
    );

    act(() => comp.unmount());
  });

  it("clears a stale result when the code is edited", async () => {
    mockClaim.mockResolvedValueOnce({
      ok: false,
      detail: "too many failed pairing attempts on this account",
    });
    const comp = await render();
    const input = () => queryId(comp, "watch-pair-code-input")!;

    await act(async () => {
      input().props.onChangeText("abc123");
    });
    expect(queryId(comp, "watch-pair-error")).toBeTruthy();

    act(() => input().props.onChangeText("abc12"));
    expect(queryId(comp, "watch-pair-error")).toBeNull();

    act(() => comp.unmount());
  });

  it("shows an honest fallback message if the client call itself throws", async () => {
    mockClaim.mockRejectedValue(new Error("network down"));
    const comp = await render();
    const input = queryId(comp, "watch-pair-code-input")!;

    await act(async () => {
      input.props.onChangeText("abc123");
    });

    expect(queryId(comp, "watch-pair-error")).toBeTruthy();
    act(() => comp.unmount());
  });
});

describe("WatchSetupScreen — paired-state awareness", () => {
  it("shows the default first-time heading and no disconnect action while unpaired", async () => {
    mockGetMe.mockResolvedValue({
      account_id: "u1",
      email: "a@example.com",
      legacy: false,
      has_paired_watch: false,
    });
    const comp = await render();

    expect(textOf(queryId(comp, "watch-setup-heading")!)).toBe(
      "Set up your watch",
    );
    expect(queryId(comp, "watch-setup-disconnect-button")).toBeNull();
    // Install/pair flow stays fully visible even before /me resolves.
    expect(queryId(comp, "watch-install-button")).toBeTruthy();
    expect(queryId(comp, "watch-pair-code-input")).toBeTruthy();

    act(() => comp.unmount());
  });

  it("switches to 'Add another watch' heading and offers disconnect once /me reports has_paired_watch", async () => {
    mockGetMe.mockResolvedValue({
      account_id: "u1",
      email: "a@example.com",
      legacy: false,
      has_paired_watch: true,
    });
    const comp = await render();

    expect(textOf(queryId(comp, "watch-setup-heading")!)).toContain(
      "Add another watch",
    );
    expect(queryId(comp, "watch-setup-disconnect-button")).toBeTruthy();
    // The install/pair flow (steps 1 + 2) stays fully usable — adding a
    // second watch is a legitimate case, not a blocked one.
    expect(queryId(comp, "watch-install-button")).toBeTruthy();
    expect(queryId(comp, "watch-pair-code-input")).toBeTruthy();

    act(() => comp.unmount());
  });

  it("stays on the honest default first-time heading if /me can't be fetched", async () => {
    mockGetMe.mockRejectedValue(new Error("network down"));
    const comp = await render();

    expect(textOf(queryId(comp, "watch-setup-heading")!)).toBe(
      "Set up your watch",
    );
    expect(queryId(comp, "watch-setup-disconnect-button")).toBeNull();

    act(() => comp.unmount());
  });

  it("confirms before disconnecting, with accurate non-alarming copy", async () => {
    mockGetMe.mockResolvedValue({
      account_id: "u1",
      email: "a@example.com",
      legacy: false,
      has_paired_watch: true,
    });
    mockDisconnect.mockReturnValue(new Promise(() => {})); // never resolves here
    const alertSpy = jest.spyOn(Alert, "alert").mockImplementation(() => {});
    const comp = await render();

    act(() => queryId(comp, "watch-setup-disconnect-button")!.props.onPress());

    expect(alertSpy).toHaveBeenCalled();
    const [title, message] = alertSpy.mock.calls[0];
    expect(title).toMatch(/disconnect/i);
    expect(message).toMatch(/stop being able to sign in/i);
    // Accurate, non-alarming: explicitly says data/recordings are safe.
    expect(message).toMatch(/recordings.*safe|safe.*recordings/i);
    expect(mockDisconnect).not.toHaveBeenCalled();

    alertSpy.mockRestore();
    act(() => comp.unmount());
  });

  it("on confirm, calls disconnectWatch and reverts to the first-time state without a restart", async () => {
    mockGetMe.mockResolvedValue({
      account_id: "u1",
      email: "a@example.com",
      legacy: false,
      has_paired_watch: true,
    });
    mockDisconnect.mockResolvedValue({ disconnected: true, count: 1 });
    const alertSpy = jest.spyOn(Alert, "alert").mockImplementation(() => {});
    const comp = await render();

    act(() => queryId(comp, "watch-setup-disconnect-button")!.props.onPress());
    const buttons = alertSpy.mock.calls[0][2] as Array<{
      style?: string;
      onPress?: () => void;
    }>;
    const destructive = buttons.find((b) => b.style === "destructive")!;
    await act(async () => destructive.onPress!());

    expect(mockDisconnect).toHaveBeenCalledTimes(1);
    expect(textOf(queryId(comp, "watch-setup-heading")!)).toBe(
      "Set up your watch",
    );
    expect(queryId(comp, "watch-setup-disconnect-button")).toBeNull();

    alertSpy.mockRestore();
    act(() => comp.unmount());
  });

  it("shows an honest error and stays paired if the disconnect call fails", async () => {
    mockGetMe.mockResolvedValue({
      account_id: "u1",
      email: "a@example.com",
      legacy: false,
      has_paired_watch: true,
    });
    mockDisconnect.mockRejectedValue(new Error("API error: 500"));
    const alertSpy = jest.spyOn(Alert, "alert").mockImplementation(() => {});
    const comp = await render();

    act(() => queryId(comp, "watch-setup-disconnect-button")!.props.onPress());
    const buttons = alertSpy.mock.calls[0][2] as Array<{
      style?: string;
      onPress?: () => void;
    }>;
    const destructive = buttons.find((b) => b.style === "destructive")!;
    await act(async () => destructive.onPress!());

    expect(mockDisconnect).toHaveBeenCalledTimes(1);
    // Still paired — the UI never claims success it didn't get.
    expect(textOf(queryId(comp, "watch-setup-heading")!)).toContain(
      "Add another watch",
    );
    expect(queryId(comp, "watch-setup-disconnect-button")).toBeTruthy();
    // A second Alert.alert call reports the failure.
    expect(alertSpy).toHaveBeenCalledTimes(2);

    alertSpy.mockRestore();
    act(() => comp.unmount());
  });

  it("cancel does not call disconnectWatch and leaves paired state untouched", async () => {
    mockGetMe.mockResolvedValue({
      account_id: "u1",
      email: "a@example.com",
      legacy: false,
      has_paired_watch: true,
    });
    const alertSpy = jest.spyOn(Alert, "alert").mockImplementation(() => {});
    const comp = await render();

    act(() => queryId(comp, "watch-setup-disconnect-button")!.props.onPress());
    const buttons = alertSpy.mock.calls[0][2] as Array<{
      style?: string;
      onPress?: () => void;
    }>;
    const cancel = buttons.find((b) => b.style === "cancel")!;
    if (cancel.onPress) act(() => cancel.onPress!());

    expect(mockDisconnect).not.toHaveBeenCalled();
    expect(queryId(comp, "watch-setup-disconnect-button")).toBeTruthy();

    alertSpy.mockRestore();
    act(() => comp.unmount());
  });
});
