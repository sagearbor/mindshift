import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import * as Camera from "expo-camera";
// Force the expo-media-library mock factory to run: the component now
// lazy-requires the native module (web-bundle fix), so nothing else imports
// it before the top-level mock-global grab below.
import "expo-media-library";
import RecordScreen, {
  remainingSeconds,
  isAtCap,
  formatClock,
  MAX_RECORDING_SECONDS,
} from "../src/screens/RecordScreen";

// Spies from the jest-setup wholesale mocks.
const cameraMock = (globalThis as Record<string, unknown>).__expoCameraMock as {
  recordAsync: jest.Mock;
  stopRecording: jest.Mock;
};
const mediaMock = (globalThis as Record<string, unknown>)
  .__expoMediaLibraryMock as { create: jest.Mock };

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

const grantedHook = () => [
  { granted: true, status: "granted", canAskAgain: true },
  jest.fn().mockResolvedValue({ granted: true, status: "granted" }),
  jest.fn().mockResolvedValue({ granted: true, status: "granted" }),
];

beforeEach(() => {
  cameraMock.recordAsync.mockReset();
  cameraMock.recordAsync.mockResolvedValue({ uri: "file:///recorded.mp4" });
  cameraMock.stopRecording.mockReset();
  mediaMock.create.mockReset();
  mediaMock.create.mockResolvedValue({ id: "asset-1", uri: "ph://asset-1" });
  (Camera.useCameraPermissions as jest.Mock).mockImplementation(grantedHook);
  (Camera.useMicrophonePermissions as jest.Mock).mockImplementation(grantedHook);
});

// --- Pure cap/timer helpers (unit tests, no camera) ---
describe("recording cap helpers", () => {
  it("remainingSeconds counts down from the cap and never goes negative", () => {
    expect(remainingSeconds(0)).toBe(MAX_RECORDING_SECONDS);
    expect(remainingSeconds(1.4)).toBe(MAX_RECORDING_SECONDS - 1);
    expect(remainingSeconds(MAX_RECORDING_SECONDS)).toBe(0);
    // Past the cap clamps to 0, never negative.
    expect(remainingSeconds(MAX_RECORDING_SECONDS + 30)).toBe(0);
    // Honors a custom cap.
    expect(remainingSeconds(10, 60)).toBe(50);
  });

  it("isAtCap is true only once elapsed reaches the cap", () => {
    expect(isAtCap(0)).toBe(false);
    expect(isAtCap(MAX_RECORDING_SECONDS - 1)).toBe(false);
    expect(isAtCap(MAX_RECORDING_SECONDS)).toBe(true);
    expect(isAtCap(MAX_RECORDING_SECONDS + 5)).toBe(true);
    expect(isAtCap(60, 60)).toBe(true);
  });

  it("formatClock renders m:ss with the cap at 10:00", () => {
    expect(formatClock(0)).toBe("0:00");
    expect(formatClock(5)).toBe("0:05");
    expect(formatClock(65)).toBe("1:05");
    expect(formatClock(MAX_RECORDING_SECONDS)).toBe("10:00");
  });
});

describe("RecordScreen", () => {
  it("shows an honest permission-denied message with a grant retry (no black screen)", () => {
    const requestCam = jest
      .fn()
      .mockResolvedValue({ granted: true, status: "granted" });
    (Camera.useCameraPermissions as jest.Mock).mockReturnValue([
      { granted: false, status: "denied", canAskAgain: true },
      requestCam,
      jest.fn(),
    ]);

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <RecordScreen onBack={() => {}} onComplete={() => {}} />,
      );
    });

    // The gate (not the camera) is shown, with a specific message + grant button.
    expect(queryId(comp, "permission-gate")).toBeTruthy();
    expect(queryId(comp, "camera-view")).toBeNull();
    expect(queryId(comp, "perm-camera")).toBeTruthy();
    const grant = queryId(comp, "grant-camera");
    expect(grant).toBeTruthy();
    act(() => grant!.props.onPress());
    expect(requestCam).toHaveBeenCalled();
    act(() => comp.unmount());
  });

  it("saves the clip to the camera roll and hands the recorded file to the upload flow", async () => {
    const onComplete = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <RecordScreen onBack={() => {}} onComplete={onComplete} />,
      );
    });

    // Permissions granted → the camera preview renders with a record button.
    expect(queryId(comp, "camera-view")).toBeTruthy();
    expect(queryId(comp, "record-timer")).toBeTruthy();

    // Press Record. The mocked recordAsync resolves with a uri (as if stopped),
    // so the finish path runs: save-to-roll then hand-off.
    await act(async () => {
      queryId(comp, "record-button")!.props.onPress();
    });
    await act(async () => {});

    expect(cameraMock.recordAsync).toHaveBeenCalledTimes(1);
    // recordAsync received the 10-minute maxDuration cap.
    expect(cameraMock.recordAsync.mock.calls[0][0]).toEqual(
      expect.objectContaining({ maxDuration: MAX_RECORDING_SECONDS }),
    );
    // Saved to the camera roll (the cloud-backup linchpin).
    expect(mediaMock.create).toHaveBeenCalledWith("file:///recorded.mp4");
    // Handoff carries the recorded file into the upload flow.
    expect(onComplete).toHaveBeenCalledTimes(1);
    const file = onComplete.mock.calls[0][0];
    expect(file.uri).toBe("file:///recorded.mp4");
    expect(file.mimeType).toBe("video/mp4");
    expect(file.name).toMatch(/^mindshift-\d+\.mp4$/);
    act(() => comp.unmount());
  });

  it("on a save failure, offers recovery instead of silently dropping the clip", async () => {
    mediaMock.create.mockRejectedValueOnce(new Error("disk full"));
    const onComplete = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <RecordScreen onBack={() => {}} onComplete={onComplete} />,
      );
    });

    await act(async () => {
      queryId(comp, "record-button")!.props.onPress();
    });
    await act(async () => {});

    // Save failed → honest recovery UI, no hand-off yet.
    expect(queryId(comp, "save-error")).toBeTruthy();
    expect(onComplete).not.toHaveBeenCalled();

    // "Analyze it now anyway" still hands the file off honestly.
    await act(async () => {
      queryId(comp, "analyze-anyway-button")!.props.onPress();
    });
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onComplete.mock.calls[0][0].uri).toBe("file:///recorded.mp4");
    act(() => comp.unmount());
  });
});
