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

// Mock fetch globally
global.fetch = jest.fn();
