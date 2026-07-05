import React from "react";
import renderer, { act } from "react-test-renderer";

const mockUseAudioStream = jest.fn();

jest.mock("../src/hooks/useAudioStream", () => ({
  useAudioStream: () => mockUseAudioStream(),
}));

import LiveCoachScreen from "../src/screens/LiveCoachScreen";

const defaultHookState = {
  isRecording: false,
  sessionActive: false,
  transcript: [],
  suggestions: [],
  speakerLabel: "",
  connectionStatus: "idle" as const,
  transcriptionAvailable: true,
  transcriptionMessage: "",
  micError: "",
  speechAvailable: true,
  speechEnabled: false,
  setSpeechEnabled: jest.fn(),
  startSession: jest.fn(),
  stopSession: jest.fn(),
  sendEmpathyUpdate: jest.fn(),
};

beforeEach(() => {
  mockUseAudioStream.mockReturnValue({ ...defaultHookState });
});

describe("LiveCoachScreen", () => {
  it("renders initial idle state", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<LiveCoachScreen />);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders recording state with transcript and suggestions", () => {
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      isRecording: true,
      sessionActive: true,
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

    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<LiveCoachScreen />);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders disconnected state", () => {
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      connectionStatus: "disconnected",
    });

    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<LiveCoachScreen />);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("shows the mic error banner when capture fails", () => {
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      micError: "Microphone permission denied — enable microphone access.",
    });

    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    const banner = root!.root.findByProps({ testID: "mic-error-banner" });
    expect(banner).toBeTruthy();
    // The honest failure message is shown verbatim.
    const text = JSON.stringify(root!.toJSON());
    expect(text).toContain("Microphone permission denied");
  });

  it("hides the mic error banner when there is no error", () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    expect(
      root!.root.findAllByProps({ testID: "mic-error-banner" }),
    ).toHaveLength(0);
  });

  it("wires coach mode to speech: visual on mount, earpiece enables speaking", () => {
    const setSpeechEnabled = jest.fn();
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      setSpeechEnabled,
    });

    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    // Default mode is visual — speech starts disabled, honestly silent.
    expect(setSpeechEnabled).toHaveBeenLastCalledWith(false);

    act(() => {
      root!.root.findByProps({ testID: "mode-earpiece" }).props.onPress();
    });
    expect(setSpeechEnabled).toHaveBeenLastCalledWith(true);

    act(() => {
      root!.root.findByProps({ testID: "mode-visual" }).props.onPress();
    });
    expect(setSpeechEnabled).toHaveBeenLastCalledWith(false);
  });

  it("shows an honest note when earpiece is selected but TTS is unavailable", () => {
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      speechAvailable: false,
    });

    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    // Visual mode: no note (nothing was promised aloud).
    expect(
      root!.root.findAllByProps({ testID: "speech-unavailable-note" }),
    ).toHaveLength(0);

    act(() => {
      root!.root.findByProps({ testID: "mode-earpiece" }).props.onPress();
    });
    // Count host nodes only — RN <Text> also yields a composite node
    // carrying the same testID.
    const notes = root!.root.findAll(
      (node) =>
        node.props.testID === "speech-unavailable-note" &&
        typeof node.type === "string",
    );
    expect(notes).toHaveLength(1);
  });

  it("hides the unavailable note when TTS works in earpiece mode", () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    act(() => {
      root!.root.findByProps({ testID: "mode-earpiece" }).props.onPress();
    });
    expect(
      root!.root.findAllByProps({ testID: "speech-unavailable-note" }),
    ).toHaveLength(0);
  });
});
