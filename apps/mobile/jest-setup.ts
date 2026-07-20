// First-render transforms of react-native + expo modules can be slow on cold
// CI/sandbox workers; the 5s default trips on the first test of a suite even
// though the work itself is fast. Give every test more headroom.
jest.setTimeout(30000);

// Mock react-native-safe-area-context for tests. The redesign wraps the app in
// a SafeAreaProvider + SafeAreaView so screen headers clear the Android status
// bar; under react-test-renderer there's no native module, so we passthrough the
// provider/view as plain Views and report zero insets deterministically.
jest.mock("react-native-safe-area-context", () => {
  const React = require("react");
  const { View } = require("react-native");
  const insets = { top: 0, right: 0, bottom: 0, left: 0 };
  const frame = { x: 0, y: 0, width: 320, height: 640 };
  const SafeAreaInsetsContext = React.createContext(insets);
  const SafeAreaFrameContext = React.createContext(frame);
  const passthrough = (props: Record<string, unknown>) =>
    React.createElement(View, props, props.children as React.ReactNode);
  return {
    __esModule: true,
    SafeAreaProvider: passthrough,
    SafeAreaView: passthrough,
    SafeAreaInsetsContext,
    SafeAreaFrameContext,
    SafeAreaConsumer: SafeAreaInsetsContext.Consumer,
    useSafeAreaInsets: () => insets,
    useSafeAreaFrame: () => frame,
    initialWindowMetrics: { insets, frame },
  };
});

// Mock @react-native-community/slider for tests
jest.mock("@react-native-community/slider", () => {
  const React = require("react");
  const { View } = require("react-native");
  return {
    __esModule: true,
    default: (props: Record<string, unknown>) =>
      React.createElement(View, { testID: props.testID }),
  };
});

// Mock react-native-svg for tests
jest.mock("react-native-svg", () => {
  const React = require("react");
  const { View } = require("react-native");
  const createMock = (name: string) => (props: Record<string, unknown>) =>
    React.createElement(View, { testID: name, ...props }, props.children);
  return {
    __esModule: true,
    default: createMock("Svg"),
    Svg: createMock("Svg"),
    Polyline: createMock("Polyline"),
    Circle: createMock("Circle"),
    Line: createMock("Line"),
    Rect: createMock("Rect"),
    Path: createMock("Path"),
    G: createMock("G"),
    Text: createMock("SvgText"),
  };
});

// Mock react-native Share API
jest.mock("react-native/Libraries/Share/Share", () => ({
  share: jest.fn().mockResolvedValue({ action: "sharedAction" }),
}));

// Mock expo-audio for tests. The realtime PCM stream API (useAudioStream)
// captures the onBuffer callback on `globalThis.__expoAudioMock` so tests can
// push synthetic PCM buffers through the pipeline. Test files may override
// this with their own jest.mock("expo-audio", ...) for finer control.
jest.mock("expo-audio", () => {
  type MockBufferCallback = (mockBuffer: unknown) => void;
  const mockState = {
    onBuffer: null as MockBufferCallback | null,
    stream: {
      id: "mock-audio-stream",
      sampleRate: 16000,
      channels: 1,
      isStreaming: false,
      start: jest.fn().mockResolvedValue(undefined),
      stop: jest.fn(),
    },
  };
  (globalThis as Record<string, unknown>).__expoAudioMock = mockState;
  return {
    __esModule: true,
    requestRecordingPermissionsAsync: jest
      .fn()
      .mockResolvedValue({ status: "granted", granted: true }),
    getRecordingPermissionsAsync: jest
      .fn()
      .mockResolvedValue({ status: "granted", granted: true }),
    setAudioModeAsync: jest.fn().mockResolvedValue(undefined),
    useAudioStream: (options?: { onBuffer?: MockBufferCallback }) => {
      mockState.onBuffer = options?.onBuffer ?? null;
      return { stream: mockState.stream, isStreaming: false };
    },
  };
});

// Mock expo-document-picker. Default: the user cancels the picker, so screens
// that merely import it render fine. Tests that exercise the upload flow
// override getDocumentAsync with their own mockResolvedValueOnce.
jest.mock("expo-document-picker", () => ({
  __esModule: true,
  getDocumentAsync: jest.fn().mockResolvedValue({ canceled: true, assets: null }),
}));

// Mock expo-file-system for tests. The chunked-upload path (postAnalyzeUploadChunked)
// reads byte ranges via the modern `File(uri).open().readBytes()` handle API. The
// mock handle serves slices out of `globalThis.__fsMockBytes` (a Uint8Array the
// test sets to stand in for the on-disk file), honoring the seekable `offset`.
jest.mock("expo-file-system", () => {
  class FileHandle {
    offset = 0;
    size: number | null = null;
    close = jest.fn();
    readBytes(length: number): Uint8Array {
      const buf =
        ((globalThis as Record<string, unknown>).__fsMockBytes as
          | Uint8Array
          | undefined) ?? new Uint8Array(0);
      return buf.slice(this.offset, this.offset + length);
    }
    writeBytes() {}
  }
  class File {
    uri: string;
    constructor(uri: string) {
      this.uri = uri;
    }
    open() {
      return new FileHandle();
    }
  }
  return {
    __esModule: true,
    File,
    FileMode: { ReadWrite: "rw", ReadOnly: "r", WriteOnly: "w" },
  };
});

// Mock expo-video for tests. `useVideoPlayer` returns a controllable mock
// player exposed on `globalThis.__expoVideoMock` so tests can drive
// currentTime / playing / duration; `VideoView` renders a plain host View.
// (Most screen tests mock the MediaPlayer component wholesale; this keeps the
// module resolvable for anything that imports expo-video directly.)
jest.mock("expo-video", () => {
  const React = require("react");
  const { View } = require("react-native");
  const player = {
    currentTime: 0,
    duration: 0,
    playing: false,
    loop: false,
    muted: false,
    play: jest.fn(function (this: Record<string, unknown>) {
      this.playing = true;
    }),
    pause: jest.fn(function (this: Record<string, unknown>) {
      this.playing = false;
    }),
    seekBy: jest.fn(),
    replace: jest.fn(),
    addListener: jest.fn(() => ({ remove: jest.fn() })),
  };
  (globalThis as Record<string, unknown>).__expoVideoMock = player;
  return {
    __esModule: true,
    useVideoPlayer: (_source: unknown, setup?: (p: unknown) => void) => {
      if (setup) setup(player);
      return player;
    },
    createVideoPlayer: () => player,
    VideoView: (props: Record<string, unknown>) =>
      React.createElement(View, { testID: props.testID ?? "video-view" }),
  };
});

// Mock expo-camera for tests. `CameraView` is a host view that forwards a ref
// exposing `recordAsync`/`stopRecording` spies (recordAsync resolves to a fixed
// recorded-file uri so the finish/handoff path can be driven without a real
// camera). The permission hooks default to granted; tests override them to
// exercise the denial gate. Spies live on `globalThis.__expoCameraMock`.
jest.mock("expo-camera", () => {
  const React = require("react");
  const { View } = require("react-native");
  const recordAsync = jest
    .fn()
    .mockResolvedValue({ uri: "file:///recorded.mp4" });
  const stopRecording = jest.fn();
  (globalThis as Record<string, unknown>).__expoCameraMock = {
    recordAsync,
    stopRecording,
  };
  const CameraView = React.forwardRef(
    (props: Record<string, unknown>, ref: unknown) => {
      React.useImperativeHandle(ref, () => ({ recordAsync, stopRecording }));
      return React.createElement(
        View,
        { testID: props.testID ?? "camera-view" },
        props.children as React.ReactNode,
      );
    },
  );
  const grantedHook = () => [
    { granted: true, status: "granted", canAskAgain: true },
    jest.fn().mockResolvedValue({ granted: true, status: "granted" }),
    jest.fn().mockResolvedValue({ granted: true, status: "granted" }),
  ];
  return {
    __esModule: true,
    CameraView,
    useCameraPermissions: jest.fn(grantedHook),
    useMicrophonePermissions: jest.fn(grantedHook),
  };
});

// Mock expo-media-library for tests. `Asset.create` (the SDK-57 save-to-roll
// entry point) is a spy on `globalThis.__expoMediaLibraryMock`; `usePermissions`
// defaults to granted. Tests assert the save was attempted on stop.
jest.mock("expo-media-library", () => {
  const create = jest
    .fn()
    .mockResolvedValue({ id: "asset-1", uri: "ph://asset-1" });
  (globalThis as Record<string, unknown>).__expoMediaLibraryMock = { create };
  const grantedHook = () => [
    { granted: true, status: "granted", canAskAgain: true },
    jest.fn().mockResolvedValue({ granted: true, status: "granted" }),
    jest.fn().mockResolvedValue({ granted: true, status: "granted" }),
  ];
  return {
    __esModule: true,
    Asset: { create },
    usePermissions: jest.fn(grantedHook),
    requestPermissionsAsync: jest
      .fn()
      .mockResolvedValue({ granted: true, status: "granted" }),
    getPermissionsAsync: jest
      .fn()
      .mockResolvedValue({ granted: true, status: "granted" }),
  };
});

// Mock expo-speech (free on-device TTS) for tests. speak/stop/isSpeakingAsync
// are plain jest.fn()s so tests can assert what would have been spoken.
jest.mock("expo-speech", () => ({
  __esModule: true,
  speak: jest.fn(),
  stop: jest.fn().mockResolvedValue(undefined),
  isSpeakingAsync: jest.fn().mockResolvedValue(false),
  getAvailableVoicesAsync: jest.fn().mockResolvedValue([]),
  maxSpeechInputLength: 4000,
}));

// --- Firebase Auth mocks (tests never hit real Firebase) ---
// The real firebase/app + firebase/auth are replaced so no network/keystore is
// touched. onIdTokenChanged stores the listener and the signed-in user on
// `globalThis.__firebaseAuthMock` so tests can drive auth-state transitions.
jest.mock("firebase/app", () => ({
  __esModule: true,
  initializeApp: jest.fn(() => ({ name: "[DEFAULT]" })),
  getApps: jest.fn(() => []),
  getApp: jest.fn(() => ({ name: "[DEFAULT]" })),
}));

jest.mock("firebase/auth", () => {
  const state = { currentUser: null, idTokenListener: null };
  (globalThis as Record<string, unknown>).__firebaseAuthMock = state;
  const authInstance = {
    get currentUser() {
      return state.currentUser;
    },
  };
  return {
    __esModule: true,
    initializeAuth: jest.fn(() => authInstance),
    getAuth: jest.fn(() => authInstance),
    browserLocalPersistence: { type: "LOCAL" },
    getReactNativePersistence: jest.fn(() => ({ type: "REACT_NATIVE" })),
    onIdTokenChanged: jest.fn((_auth, cb) => {
      state.idTokenListener = cb;
      return () => {
        state.idTokenListener = null;
      };
    }),
    signInWithEmailAndPassword: jest.fn().mockResolvedValue(undefined),
    createUserWithEmailAndPassword: jest.fn().mockResolvedValue(undefined),
    signInWithCredential: jest.fn().mockResolvedValue(undefined),
    signInWithPopup: jest.fn().mockResolvedValue(undefined),
    linkWithCredential: jest.fn().mockResolvedValue(undefined),
    sendPasswordResetEmail: jest.fn().mockResolvedValue(undefined),
    signOut: jest.fn().mockResolvedValue(undefined),
    // Constructable provider with static helpers, matching firebase/auth's API.
    GoogleAuthProvider: Object.assign(
      jest.fn(() => ({ providerId: "google.com" })),
      {
        credential: jest.fn((idToken: string) => ({
          providerId: "google.com",
          idToken,
        })),
        // Tests stash the pending credential on the error as `__pendingCred`.
        credentialFromError: jest.fn(
          (err: { __pendingCred?: unknown }) => err?.__pendingCred ?? null,
        ),
      },
    ),
  };
});

// Mock @react-native-google-signin/google-signin (native Google button). Tests
// drive the sign-in outcome via `globalThis.__googleSigninMock`.
jest.mock("@react-native-google-signin/google-signin", () => {
  const mock = {
    configure: jest.fn(),
    hasPlayServices: jest.fn().mockResolvedValue(true),
    signIn: jest
      .fn()
      .mockResolvedValue({ type: "success", data: { idToken: "g-id-token" } }),
  };
  (globalThis as Record<string, unknown>).__googleSigninMock = mock;
  const statusCodes = {
    SIGN_IN_CANCELLED: "SIGN_IN_CANCELLED",
    IN_PROGRESS: "IN_PROGRESS",
    PLAY_SERVICES_NOT_AVAILABLE: "PLAY_SERVICES_NOT_AVAILABLE",
    SIGN_IN_REQUIRED: "SIGN_IN_REQUIRED",
  };
  return {
    __esModule: true,
    GoogleSignin: mock,
    statusCodes,
    isSuccessResponse: (r: { type?: string } | null) => r?.type === "success",
    isErrorWithCode: (e: { code?: unknown } | null) =>
      Boolean(e && typeof e === "object" && "code" in e),
  };
});

// Mock expo-secure-store (Firebase persistence backend on native).
jest.mock("expo-secure-store", () => ({
  __esModule: true,
  getItemAsync: jest.fn().mockResolvedValue(null),
  setItemAsync: jest.fn().mockResolvedValue(undefined),
  deleteItemAsync: jest.fn().mockResolvedValue(undefined),
}));

// Mock expo-updates (EAS Update / OTA). The default here models a normal STORE
// build with no OTA applied yet: updates disabled at rest, running the embedded
// bundle, nothing pending. Tests that exercise the "update ready" banner or the
// "OTA applied" About line mock ../src/utils/otaUpdate (or useUpdates) directly.
jest.mock("expo-updates", () => ({
  __esModule: true,
  isEnabled: false,
  isEmbeddedLaunch: true,
  channel: null,
  createdAt: null,
  runtimeVersion: "1.14.0",
  updateId: null,
  reloadAsync: jest.fn().mockResolvedValue(undefined),
  checkForUpdateAsync: jest
    .fn()
    .mockResolvedValue({ isAvailable: false, manifest: null }),
  fetchUpdateAsync: jest
    .fn()
    .mockResolvedValue({ isNew: false, manifest: null }),
  useUpdates: () => ({
    currentlyRunning: {
      updateId: undefined,
      channel: undefined,
      createdAt: undefined,
      isEmbeddedLaunch: true,
      isEmergencyLaunch: false,
      emergencyLaunchReason: null,
      runtimeVersion: "1.14.0",
    },
    isUpdateAvailable: false,
    isUpdatePending: false,
    isChecking: false,
    isDownloading: false,
    availableUpdate: undefined,
    checkError: undefined,
    downloadError: undefined,
  }),
}));

// Mock expo-application (native app/build identity for the About section).
jest.mock("expo-application", () => ({
  __esModule: true,
  nativeApplicationVersion: "1.14.0",
  nativeBuildVersion: "29",
  applicationName: "MindShift",
  applicationId: "com.sagearbor.mindshift.app",
}));

// Mock expo-constants so the About section reads a deterministic manifest even
// when expo-application returns null (e.g. web). No src module used Constants
// before the About section, so this global default is safe.
jest.mock("expo-constants", () => ({
  __esModule: true,
  default: {
    expoConfig: {
      version: "1.14.0",
      android: { versionCode: 29 },
    },
  },
}));

// Mock fetch globally
global.fetch = jest.fn();
