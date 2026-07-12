// First-render transforms of react-native + expo modules can be slow on cold
// CI/sandbox workers; the 5s default trips on the first test of a suite even
// though the work itself is fast. Give every test more headroom.
jest.setTimeout(30000);

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
    signOut: jest.fn().mockResolvedValue(undefined),
    GoogleAuthProvider: {
      credential: jest.fn((idToken) => ({
        providerId: "google.com",
        idToken,
      })),
    },
  };
});

// Mock expo-secure-store (Firebase persistence backend on native).
jest.mock("expo-secure-store", () => ({
  __esModule: true,
  getItemAsync: jest.fn().mockResolvedValue(null),
  setItemAsync: jest.fn().mockResolvedValue(undefined),
  deleteItemAsync: jest.fn().mockResolvedValue(undefined),
}));

// Mock expo-web-browser + the Google auth-session provider so no OAuth
// browser session is opened during tests.
jest.mock("expo-web-browser", () => ({
  __esModule: true,
  maybeCompleteAuthSession: jest.fn(),
  warmUpAsync: jest.fn(),
  coolDownAsync: jest.fn(),
}));

jest.mock("expo-auth-session/providers/google", () => ({
  __esModule: true,
  useAuthRequest: jest.fn(() => [null, null, jest.fn()]),
}));

// Mock fetch globally
global.fetch = jest.fn();
