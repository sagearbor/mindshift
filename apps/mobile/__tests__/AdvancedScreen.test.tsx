import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import AdvancedScreen from "../src/screens/AdvancedScreen";
import { useAuthStore } from "../src/store/authStore";
import type { OtaStatus } from "../src/utils/otaUpdate";

// The About section reads OTA status through this hook; drive it per-test. The
// `mock`-prefixed name is the only ref jest.mock's hoisted factory may close over.
let mockOta: OtaStatus;
jest.mock("../src/utils/otaUpdate", () => ({
  __esModule: true,
  useOtaStatus: () => mockOta,
  restartToApplyUpdate: jest.fn(),
}));

function baseOta(overrides: Partial<OtaStatus> = {}): OtaStatus {
  return {
    supported: false,
    isEmbeddedLaunch: true,
    channel: null,
    createdAt: null,
    runtimeVersion: "1.14.0",
    updateId: null,
    isUpdatePending: false,
    errored: false,
    ...overrides,
  };
}

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

/** Concatenated string content rendered under a node (its Text leaves). */
function textOf(node: ReactTestInstance): string {
  return node
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");
}

describe("AdvancedScreen", () => {
  beforeEach(() => {
    mockOta = baseOta();
    act(() => {
      useAuthStore.setState({ user: null });
    });
  });

  it("renders the dashboard entry, sign out, and back — and wires each press", () => {
    const onBack = jest.fn();
    const onOpenDashboard = jest.fn();
    const onSignOut = jest.fn();

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AdvancedScreen
          onBack={onBack}
          onOpenDashboard={onOpenDashboard}
          onSignOut={onSignOut}
        />,
      );
    });

    act(() => queryId(comp, "advanced-dashboard")!.props.onPress());
    expect(onOpenDashboard).toHaveBeenCalledTimes(1);

    act(() => queryId(comp, "advanced-sign-out")!.props.onPress());
    expect(onSignOut).toHaveBeenCalledTimes(1);

    act(() => queryId(comp, "advanced-back")!.props.onPress());
    expect(onBack).toHaveBeenCalledTimes(1);

    act(() => comp.unmount());
  });

  it("About shows version, build, account email, backend host, and the honest store-build OTA line", () => {
    process.env.EXPO_PUBLIC_API_URL = "https://mindshift-api.example.run.app";
    act(() => {
      useAuthStore.setState({
        user: { uid: "u1", email: "tester@example.com", displayName: "T" },
      });
    });
    mockOta = baseOta({ supported: false });

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AdvancedScreen
          onBack={jest.fn()}
          onOpenDashboard={jest.fn()}
          onSignOut={jest.fn()}
        />,
      );
    });

    expect(queryId(comp, "about-section")).toBeTruthy();
    // Mocked expo-application / expo-constants from jest-setup.
    expect(textOf(queryId(comp, "about-version")!)).toContain("1.14.0");
    expect(textOf(queryId(comp, "about-build")!)).toContain("29");
    expect(textOf(queryId(comp, "about-account")!)).toContain(
      "tester@example.com",
    );
    expect(textOf(queryId(comp, "about-backend")!)).toContain(
      "mindshift-api.example.run.app",
    );
    // No OTA module in this build → an honest store-build line, not a fake channel.
    expect(textOf(queryId(comp, "about-update")!)).toContain(
      "Store build (no OTA yet)",
    );

    act(() => comp.unmount());
  });

  it("About reports a downloaded OTA update with its publish time and channel", () => {
    mockOta = baseOta({
      supported: true,
      isEmbeddedLaunch: false,
      channel: "production",
      createdAt: new Date("2026-07-19T15:30:00Z"),
      updateId: "abc",
    });

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AdvancedScreen
          onBack={jest.fn()}
          onOpenDashboard={jest.fn()}
          onSignOut={jest.fn()}
        />,
      );
    });

    const updateText = textOf(queryId(comp, "about-update")!);
    expect(updateText).toContain("Updated");
    expect(updateText).toContain("production channel");

    act(() => comp.unmount());
  });

  it("About falls back honestly when signed in without an email on file", () => {
    act(() => {
      useAuthStore.setState({
        user: { uid: "u2", email: null, displayName: null },
      });
    });

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AdvancedScreen
          onBack={jest.fn()}
          onOpenDashboard={jest.fn()}
          onSignOut={jest.fn()}
        />,
      );
    });

    expect(textOf(queryId(comp, "about-account")!)).toContain(
      "No email on this account",
    );

    act(() => comp.unmount());
  });
});
