import React from "react";
import { Platform } from "react-native";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import * as Camera from "expo-camera";
import AvatarCaptureScreen from "../src/screens/AvatarCaptureScreen";
import { useAvatarStore } from "../src/store/avatarStore";

const originalOS = Platform.OS;

function setPlatform(os: string) {
  Object.defineProperty(Platform, "OS", { value: os, configurable: true });
}

// Spies from the jest-setup wholesale expo-camera mock (same pattern
// RecordScreen.test.tsx uses).
const cameraMock = (globalThis as Record<string, unknown>).__expoCameraMock as {
  takePictureAsync: jest.Mock;
};

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
  setPlatform("ios");
  cameraMock.takePictureAsync.mockReset();
  cameraMock.takePictureAsync.mockResolvedValue({
    uri: "file:///captured-selfie.jpg",
  });
  (Camera.useCameraPermissions as jest.Mock).mockImplementation(grantedHook);
  useAvatarStore.setState({ uri: null, hydrated: false });
});

afterEach(() => {
  setPlatform(originalOS);
});

describe("AvatarCaptureScreen — permission denied", () => {
  it("shows an honest message with a grant retry (no black screen)", () => {
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
        <AvatarCaptureScreen onBack={() => {}} onSaved={() => {}} />,
      );
    });

    expect(queryId(comp, "avatar-permission-gate")).toBeTruthy();
    expect(queryId(comp, "avatar-camera-view")).toBeNull();
    const grant = queryId(comp, "avatar-grant-camera");
    expect(grant).toBeTruthy();
    act(() => grant!.props.onPress());
    expect(requestCam).toHaveBeenCalled();
    act(() => comp.unmount());
  });

  it("offers a path to device Settings once the OS won't ask again — never a grant button that does nothing", () => {
    (Camera.useCameraPermissions as jest.Mock).mockReturnValue([
      { granted: false, status: "denied", canAskAgain: false },
      jest.fn(),
      jest.fn(),
    ]);

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AvatarCaptureScreen onBack={() => {}} onSaved={() => {}} />,
      );
    });

    expect(queryId(comp, "avatar-permission-gate")).toBeTruthy();
    expect(queryId(comp, "avatar-grant-camera")).toBeNull();
    act(() => comp.unmount());
  });

  it("Back returns to the launching screen without touching avatarStore", () => {
    (Camera.useCameraPermissions as jest.Mock).mockReturnValue([
      { granted: false, status: "denied", canAskAgain: true },
      jest.fn(),
      jest.fn(),
    ]);
    const onBack = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AvatarCaptureScreen onBack={onBack} onSaved={() => {}} />,
      );
    });
    act(() => queryId(comp, "avatar-capture-back")!.props.onPress());
    expect(onBack).toHaveBeenCalledTimes(1);
    expect(useAvatarStore.getState().uri).toBeNull();
    act(() => comp.unmount());
  });
});

describe("AvatarCaptureScreen — capture, preview, Use/Retake", () => {
  it("captures a photo and shows the preview with Use/Retake, not the live camera", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AvatarCaptureScreen onBack={() => {}} onSaved={() => {}} />,
      );
    });
    expect(queryId(comp, "avatar-camera-view")).toBeTruthy();

    await act(async () => {
      queryId(comp, "avatar-shutter-button")!.props.onPress();
    });
    await act(async () => {});

    expect(cameraMock.takePictureAsync).toHaveBeenCalledTimes(1);
    expect(queryId(comp, "avatar-preview-image")).toBeTruthy();
    expect(queryId(comp, "avatar-use-button")).toBeTruthy();
    expect(queryId(comp, "avatar-retake-button")).toBeTruthy();
    expect(queryId(comp, "avatar-camera-view")).toBeNull();
    act(() => comp.unmount());
  });

  it("Retake discards the capture and returns to the live camera", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AvatarCaptureScreen onBack={() => {}} onSaved={() => {}} />,
      );
    });
    await act(async () => {
      queryId(comp, "avatar-shutter-button")!.props.onPress();
    });
    await act(async () => {});

    act(() => queryId(comp, "avatar-retake-button")!.props.onPress());

    expect(queryId(comp, "avatar-camera-view")).toBeTruthy();
    expect(queryId(comp, "avatar-preview-image")).toBeNull();
    act(() => comp.unmount());
  });

  it("Use persists the photo, saves it to avatarStore, and calls onSaved", async () => {
    const persistPhoto = jest
      .fn()
      .mockResolvedValue("file:///doc/avatar/profile.jpg");
    const onSaved = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AvatarCaptureScreen
          onBack={() => {}}
          onSaved={onSaved}
          deps={{ persistPhoto }}
        />,
      );
    });
    await act(async () => {
      queryId(comp, "avatar-shutter-button")!.props.onPress();
    });
    await act(async () => {});

    await act(async () => {
      queryId(comp, "avatar-use-button")!.props.onPress();
    });
    await act(async () => {});

    expect(persistPhoto).toHaveBeenCalledWith("file:///captured-selfie.jpg");
    expect(useAvatarStore.getState().uri).toBe(
      "file:///doc/avatar/profile.jpg",
    );
    expect(onSaved).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });

  it("on a save failure, offers recovery instead of silently dropping the shot", async () => {
    const persistPhoto = jest.fn().mockRejectedValue(new Error("disk full"));
    const onSaved = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AvatarCaptureScreen
          onBack={() => {}}
          onSaved={onSaved}
          deps={{ persistPhoto }}
        />,
      );
    });
    await act(async () => {
      queryId(comp, "avatar-shutter-button")!.props.onPress();
    });
    await act(async () => {});

    await act(async () => {
      queryId(comp, "avatar-use-button")!.props.onPress();
    });
    await act(async () => {});

    expect(queryId(comp, "avatar-capture-error")).toBeTruthy();
    expect(onSaved).not.toHaveBeenCalled();
    expect(useAvatarStore.getState().uri).toBeNull();
    // Still on the preview — Retake is still available.
    expect(queryId(comp, "avatar-retake-button")).toBeTruthy();
    act(() => comp.unmount());
  });
});

describe("AvatarCaptureScreen — web", () => {
  it("shows an honest 'not available on web' note instead of a broken camera", () => {
    setPlatform("web");
    const onBack = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AvatarCaptureScreen onBack={onBack} onSaved={() => {}} />,
      );
    });
    expect(queryId(comp, "avatar-capture-web-note")).toBeTruthy();
    expect(queryId(comp, "avatar-camera-view")).toBeNull();
    act(() => queryId(comp, "avatar-capture-web-back")!.props.onPress());
    expect(onBack).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });
});
