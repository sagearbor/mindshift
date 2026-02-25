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

// Mock expo-av for audio tests
jest.mock("expo-av", () => ({
  Audio: {
    requestPermissionsAsync: jest
      .fn()
      .mockResolvedValue({ status: "granted" }),
    setAudioModeAsync: jest.fn().mockResolvedValue(undefined),
    Recording: jest.fn().mockImplementation(() => ({
      prepareToRecordAsync: jest.fn().mockResolvedValue(undefined),
      startAsync: jest.fn().mockResolvedValue(undefined),
      stopAndUnloadAsync: jest.fn().mockResolvedValue(undefined),
      setOnRecordingStatusUpdate: jest.fn(),
    })),
    RecordingOptionsPresets: {
      HIGH_QUALITY: {},
    },
  },
}));

// Mock fetch globally
global.fetch = jest.fn();
