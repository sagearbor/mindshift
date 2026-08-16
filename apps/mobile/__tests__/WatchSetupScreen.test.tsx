import React from "react";
import { Linking } from "react-native";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import WatchSetupScreen from "../src/screens/WatchSetupScreen";
import { claimWatchPairing } from "../src/api/watchPairing";

jest.mock("../src/api/watchPairing", () => ({
  claimWatchPairing: jest.fn(),
}));
const mockClaim = claimWatchPairing as jest.Mock;

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
