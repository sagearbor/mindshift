import React from "react";
import { Platform, BackHandler } from "react-native";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import * as Camera from "expo-camera";
import AvatarCaptureScreen from "../src/screens/AvatarCaptureScreen";
import { defaultPersistPhoto } from "../src/screens/AvatarCaptureScreenNative";
import { useAvatarStore } from "../src/store/avatarStore";

// N7 fix round 1 (CRITICAL 1): a minimal local override of the setup's
// expo-file-system mock (same override pattern audioMode.test.ts uses for
// expo-audio) — this file needs Directory/Paths in addition to File, which
// the shared setup mock doesn't provide (it only covers the chunked-upload
// File.open()/readBytes() path). Just enough surface for
// defaultPersistPhoto's real move-into-permanent-storage code to run
// end-to-end without throwing.
jest.mock("expo-file-system", () => {
  class Directory {
    uri: string;
    exists = true;
    constructor(base: { uri: string } | string, name: string) {
      const baseUri = typeof base === "string" ? base : base.uri;
      this.uri = `${baseUri}/${name}`;
    }
    create() {}
  }
  class File {
    uri: string;
    exists = false;
    constructor(source: { uri: string } | string, name?: string) {
      if (typeof source === "string") {
        this.uri = source;
      } else {
        this.uri = name ? `${source.uri}/${name}` : source.uri;
      }
    }
    delete() {}
    moveSync(dest: { uri: string }) {
      (globalThis as Record<string, unknown>).__fileOps = [
        ...(((globalThis as Record<string, unknown>).__fileOps as unknown[]) ?? []),
        { op: "move", from: this.uri, to: dest.uri },
      ];
    }
    copy(dest: { uri: string }) {
      (globalThis as Record<string, unknown>).__fileOps = [
        ...(((globalThis as Record<string, unknown>).__fileOps as unknown[]) ?? []),
        { op: "copy", from: this.uri, to: dest.uri },
      ];
    }
  }
  return {
    __esModule: true,
    Directory,
    File,
    Paths: { document: { uri: "file:///doc" } },
  };
});

// The in-app photo grid reads the roll through expo-media-library/legacy
// (getAssetsAsync / getAssetInfoAsync) — the shared setup only mocks the
// main entry, so mock the legacy subpath here.
jest.mock("expo-media-library/legacy", () => ({
  __esModule: true,
  MediaType: { photo: "photo" },
  SortBy: { creationTime: "creationTime" },
  getAssetsAsync: jest.fn().mockResolvedValue({
    assets: [
      { id: "a1", uri: "file:///roll/one.jpg" },
      { id: "a2", uri: "file:///roll/two.jpg" },
    ],
    endCursor: "a2",
    hasNextPage: false,
    totalCount: 2,
  }),
  getAssetInfoAsync: jest.fn(async (id: string) => ({
    id,
    localUri: `file:///roll/local-${id}.jpg`,
  })),
}));

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

    expect(persistPhoto).toHaveBeenCalledWith("file:///captured-selfie.jpg", "camera");
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

describe("defaultPersistPhoto (N7 fix round 1, CRITICAL 1)", () => {
  it("returns a uri that differs between two sequential captures, even though the underlying file always writes to the same fixed path", async () => {
    // A test that only asserted setPhoto() was called (the shape of the
    // pre-fix coverage) would NOT have caught this bug: it stubbed
    // persistPhoto entirely, so it never exercised the real fixed-path
    // write. This test calls the actual implementation twice — as a retake
    // would — and asserts the two returned uris are genuinely different
    // strings, which is what makes zustand's set({ uri }) actually notify
    // subscribers and what busts Image's uri-keyed cache on both platforms.
    const first = await defaultPersistPhoto("file:///captured-1.jpg");
    const second = await defaultPersistPhoto("file:///captured-2.jpg");
    expect(first).not.toBe(second);
    // Still the same fixed destination file underneath — no accumulation of
    // stale files, just a cache-busted uri.
    expect(first.split("?")[0]).toBe(second.split("?")[0]);
    expect(first.split("?")[0]).toBe("file:///doc/avatar/profile.jpg");
  });
});

describe("AvatarCaptureScreen — hardware back from the preview (N7 fix round 1, IMPORTANT 5)", () => {
  it("does not register a hardware-back listener while on the live camera (no in-progress capture)", () => {
    const addEventListenerSpy = jest.spyOn(BackHandler, "addEventListener");
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AvatarCaptureScreen onBack={() => {}} onSaved={() => {}} />,
      );
    });
    expect(
      addEventListenerSpy.mock.calls.some((c) => c[0] === "hardwareBackPress"),
    ).toBe(false);
    act(() => comp.unmount());
    addEventListenerSpy.mockRestore();
  });

  it("consumes hardware back on the preview and returns to the live camera instead of exiting the whole flow", async () => {
    const addEventListenerSpy = jest.spyOn(BackHandler, "addEventListener");
    const onBack = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AvatarCaptureScreen onBack={onBack} onSaved={() => {}} />,
      );
    });
    await act(async () => {
      queryId(comp, "avatar-shutter-button")!.props.onPress();
    });
    await act(async () => {});
    expect(queryId(comp, "avatar-preview-image")).toBeTruthy();

    const registered = addEventListenerSpy.mock.calls.find(
      (c) => c[0] === "hardwareBackPress",
    );
    expect(registered).toBeTruthy();

    let handled: boolean | undefined;
    act(() => {
      handled = (registered![1] as () => boolean)();
    });

    // The press was consumed (not left for the outer chain's own back
    // handling, which would have popped straight to `returnTo` and
    // silently dropped the just-captured photo).
    expect(handled).toBe(true);
    expect(onBack).not.toHaveBeenCalled();
    // Same effect as tapping Retake: back to the live camera, capture gone.
    expect(queryId(comp, "avatar-camera-view")).toBeTruthy();
    expect(queryId(comp, "avatar-preview-image")).toBeNull();

    act(() => comp.unmount());
    addEventListenerSpy.mockRestore();
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


// ---------------------------------------------------------------------------
// Choose from your photos — no camera needed (the owner asked from a dark room)
// ---------------------------------------------------------------------------

const deniedHook = () => [
  { granted: false, status: "denied", canAskAgain: true },
  jest.fn().mockResolvedValue({ granted: false, status: "denied" }),
  jest.fn().mockResolvedValue({ granted: false, status: "denied" }),
];

async function renderNative(props: Partial<React.ComponentProps<typeof AvatarCaptureScreen>> = {}) {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(
      <AvatarCaptureScreen onBack={jest.fn()} onSaved={jest.fn()} {...props} />,
    );
  });
  return comp;
}

describe("AvatarCaptureScreen — choose from your photos", () => {
  beforeEach(() => {
    (globalThis as Record<string, unknown>).__fileOps = [];
  });

  it("is offered even when the camera is denied — a photo you already have needs no camera", async () => {
    (Camera.useCameraPermissions as jest.Mock).mockImplementation(deniedHook);
    const comp = await renderNative();
    expect(queryId(comp, "avatar-permission-gate")).not.toBeNull();
    const choose = queryId(comp, "avatar-choose-library");
    expect(choose).not.toBeNull();
    await act(async () => choose!.props.onPress());
    expect(queryId(comp, "photo-library-picker")).not.toBeNull();
    expect(queryId(comp, "photo-library-thumb-a1")).not.toBeNull();
    expect(queryId(comp, "photo-library-thumb-a2")).not.toBeNull();
  });

  it("picking a photo shows the preview with 'Choose another', and Use COPIES it (never moves the user's original)", async () => {
    const onSaved = jest.fn();
    const persistPhoto = jest.fn(async (uri: string, source?: string) =>
      defaultPersistPhoto(uri, source as "camera" | "library"),
    );
    const comp = await renderNative({ onSaved, deps: { persistPhoto } });
    await act(async () => queryId(comp, "avatar-choose-library")!.props.onPress());
    await act(async () => queryId(comp, "photo-library-thumb-a2")!.props.onPress());
    // Preview shows the resolved local file (iOS ph:// assets need localUri).
    expect(queryId(comp, "avatar-preview-image")!.props.source.uri).toBe(
      "file:///roll/local-a2.jpg",
    );
    expect(queryId(comp, "avatar-retake-button")).not.toBeNull();
    await act(async () => queryId(comp, "avatar-use-button")!.props.onPress());
    expect(persistPhoto).toHaveBeenCalledWith("file:///roll/local-a2.jpg", "library");
    const ops = (globalThis as Record<string, unknown>).__fileOps as { op: string; from: string }[];
    expect(ops).toEqual([
      { op: "copy", from: "file:///roll/local-a2.jpg", to: "file:///doc/avatar/profile.jpg" },
    ]);
    expect(useAvatarStore.getState().uri?.split("?")[0]).toBe("file:///doc/avatar/profile.jpg");
    expect(onSaved).toHaveBeenCalledTimes(1);
  });

  it("a camera capture still MOVES its temp file", async () => {
    await defaultPersistPhoto("file:///captured-selfie.jpg");
    const ops = (globalThis as Record<string, unknown>).__fileOps as { op: string }[];
    expect(ops.map((o) => o.op)).toEqual(["move"]);
  });

  it("'Choose another' returns to the photo grid, not the camera", async () => {
    const comp = await renderNative();
    await act(async () => queryId(comp, "avatar-choose-library")!.props.onPress());
    await act(async () => queryId(comp, "photo-library-thumb-a1")!.props.onPress());
    await act(async () => queryId(comp, "avatar-retake-button")!.props.onPress());
    expect(queryId(comp, "photo-library-picker")).not.toBeNull();
    expect(queryId(comp, "avatar-camera-view")).toBeNull();
  });

  it("Back from the grid returns to the camera without choosing", async () => {
    const comp = await renderNative();
    await act(async () => queryId(comp, "avatar-choose-library")!.props.onPress());
    await act(async () => queryId(comp, "avatar-back")!.props.onPress());
    expect(queryId(comp, "avatar-camera-view")).not.toBeNull();
    expect(queryId(comp, "avatar-preview-image")).toBeNull();
  });
});
