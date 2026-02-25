import React from "react";
import renderer, { act } from "react-test-renderer";

const mockUseAudioStream = jest.fn();

jest.mock("../src/hooks/useAudioStream", () => ({
  useAudioStream: () => mockUseAudioStream(),
}));

import LiveCoachScreen from "../src/screens/LiveCoachScreen";

const defaultHookState = {
  isRecording: false,
  transcript: [],
  suggestions: [],
  speakerLabel: "",
  connectionStatus: "idle" as const,
  startSession: jest.fn(),
  stopSession: jest.fn(),
  sendEmpathyUpdate: jest.fn(),
};

beforeEach(() => {
  mockUseAudioStream.mockReturnValue({ ...defaultHookState });
});

describe("LiveCoachScreen", () => {
  it("renders initial idle state", () => {
    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer.create(<LiveCoachScreen />).toJSON();
    });
    expect(tree).toMatchSnapshot();
  });

  it("renders recording state with transcript and suggestions", () => {
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      isRecording: true,
      connectionStatus: "live",
      speakerLabel: "Speaker B",
      transcript: [
        {
          speaker: "Speaker A",
          text: "You never listen to me.",
          timestamp: 1000,
        },
        {
          speaker: "Speaker B",
          text: "I'm trying my best.",
          timestamp: 2000,
        },
      ],
      suggestions: [
        {
          text: "I hear that you feel unheard. Can you help me understand what you need?",
          tone: "empathetic",
        },
      ],
    });

    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer.create(<LiveCoachScreen />).toJSON();
    });
    expect(tree).toMatchSnapshot();
  });

  it("renders disconnected state", () => {
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      connectionStatus: "disconnected",
    });

    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer.create(<LiveCoachScreen />).toJSON();
    });
    expect(tree).toMatchSnapshot();
  });
});
