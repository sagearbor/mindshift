import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import UpdateBanner from "../src/components/UpdateBanner";
import type { OtaStatus } from "../src/utils/otaUpdate";

// Drive the banner purely off the OTA status hook and capture the restart call.
// The `mock`-prefixed names are the only refs jest.mock's hoisted factory may
// close over.
let mockStatus: OtaStatus;
const mockRestart = jest.fn();

jest.mock("../src/utils/otaUpdate", () => ({
  __esModule: true,
  useOtaStatus: () => mockStatus,
  restartToApplyUpdate: () => mockRestart(),
}));

function baseStatus(overrides: Partial<OtaStatus> = {}): OtaStatus {
  return {
    supported: true,
    isEmbeddedLaunch: false,
    channel: "production",
    createdAt: null,
    runtimeVersion: "1.14.0",
    updateId: "u-1",
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

describe("UpdateBanner", () => {
  beforeEach(() => {
    mockRestart.mockReset().mockResolvedValue(undefined);
    mockStatus = baseStatus();
  });

  it("renders nothing when no update is pending", () => {
    mockStatus = baseStatus({ isUpdatePending: false });
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<UpdateBanner />);
    });
    expect(queryId(comp, "update-banner")).toBeNull();
    act(() => comp.unmount());
  });

  it("shows the restart affordance once an update is pending", () => {
    mockStatus = baseStatus({ isUpdatePending: true });
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<UpdateBanner />);
    });
    expect(queryId(comp, "update-banner")).toBeTruthy();
    expect(queryId(comp, "update-banner-restart")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("relaunches into the update when Restart is pressed", () => {
    mockStatus = baseStatus({ isUpdatePending: true });
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<UpdateBanner />);
    });
    act(() => {
      queryId(comp, "update-banner-restart")!.props.onPress();
    });
    expect(mockRestart).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });

  it("recovers to a pressable button if the relaunch rejects", async () => {
    mockStatus = baseStatus({ isUpdatePending: true });
    mockRestart.mockRejectedValueOnce(new Error("reload failed"));
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<UpdateBanner />);
    });
    await act(async () => {
      queryId(comp, "update-banner-restart")!.props.onPress();
    });
    // Button is still present and enabled (not stuck on a spinner) after failure.
    const button = queryId(comp, "update-banner-restart")!;
    expect(button.props.disabled).toBe(false);
    act(() => comp.unmount());
  });
});
