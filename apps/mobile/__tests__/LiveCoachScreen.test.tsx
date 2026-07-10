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
  selfSpeaker: "Speaker A" as string | null,
  setSelfSpeaker: jest.fn(),
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
  sendInterjectUpdate: jest.fn(),
};

/** Build a "response" feed entry (the shape the hook now exposes). */
function responseEntry(
  texts: string[],
  { id = 1, tone = "empathetic", muted = false } = {},
) {
  return { id, kind: "response" as const, texts, tone, muted, timestamp: id };
}

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
        responseEntry([
          "I hear that you feel unheard. Can you help me understand what you need?",
        ]),
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

  it("moving the interject slider updates local state and notifies the hook", () => {
    const sendInterjectUpdate = jest.fn();
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      sendInterjectUpdate,
    });

    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    act(() => {
      root!.root
        .findByProps({ testID: "interject-slider" })
        .props.onValueChange(75);
    });

    expect(sendInterjectUpdate).toHaveBeenCalledWith(75);
  });

  it("passes the chosen interject level into startSession", async () => {
    const startSession = jest.fn().mockResolvedValue(undefined);
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      startSession,
    });

    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    act(() => {
      root!.root
        .findByProps({ testID: "interject-slider" })
        .props.onValueChange(30);
    });
    await act(async () => {
      await root!.root.findByProps({ testID: "mic-toggle" }).props.onPress();
    });

    expect(startSession).toHaveBeenCalledWith(
      expect.any(String),
      50,
      30,
    );
  });

  it("dims muted suggestions instead of hiding them", () => {
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      // Two separate feed entries: newest first, the older one muted.
      suggestions: [
        responseEntry(["Spoken advice."], { id: 2, tone: "balanced" }),
        responseEntry(["Quiet aside."], {
          id: 1,
          tone: "balanced",
          muted: true,
        }),
      ],
    });

    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    // Host nodes only — RN's <View> also yields a composite node carrying
    // the same testID.
    const cards = root!.root.findAll(
      (node) =>
        node.props.testID === "suggestion-card" &&
        typeof node.type === "string",
    );
    expect(cards).toHaveLength(2);
    expect(cards[0].props.style).not.toContainEqual(
      expect.objectContaining({ opacity: 0.5 }),
    );
    expect(cards[1].props.style).toContainEqual(
      expect.objectContaining({ opacity: 0.5 }),
    );
  });

  it("identity chip appears with a session and toggles the self speaker", () => {
    const setSelfSpeaker = jest.fn();
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      sessionActive: true,
      connectionStatus: "live",
      selfSpeaker: "Speaker A",
      setSelfSpeaker,
    });

    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    const chip = root!.root.findByProps({ testID: "self-speaker-chip" });
    expect(chip).toBeTruthy();

    act(() => chip.props.onPress());
    // Currently "Speaker A" → toggles to "Speaker B".
    expect(setSelfSpeaker).toHaveBeenCalledWith("Speaker B");
  });

  it("hides the identity chip before any session or transcript", () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    expect(
      root!.root.findAllByProps({ testID: "self-speaker-chip" }),
    ).toHaveLength(0);
  });

  it("shows the idle explainer only when idle with no transcript", () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    expect(
      root!.root.findAllByProps({ testID: "idle-explainer" }).length,
    ).toBeGreaterThan(0);

    // During a live session it is gone.
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      sessionActive: true,
      connectionStatus: "live",
    });
    let live: renderer.ReactTestRenderer;
    act(() => {
      live = renderer.create(<LiveCoachScreen />);
    });
    expect(
      live!.root.findAllByProps({ testID: "idle-explainer" }),
    ).toHaveLength(0);
  });

  it("review button shows only after a session ends with a transcript, and hands off the mapped turns", () => {
    const onReviewTranscript = jest.fn();
    const transcript = [
      { speaker: "Speaker A", text: "You never listen.", timestamp: 1 },
      { speaker: "Speaker B", text: "I'm trying.", timestamp: 2 },
    ];

    // Live session in progress: no review button yet.
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      sessionActive: true,
      transcript,
    });
    let live: renderer.ReactTestRenderer;
    act(() => {
      live = renderer.create(
        <LiveCoachScreen onReviewTranscript={onReviewTranscript} />,
      );
    });
    expect(
      live!.root.findAllByProps({ testID: "review-transcript-button" }),
    ).toHaveLength(0);

    // Session ended with a transcript: the button appears and hands off the
    // turns mapped to {speaker, text} (timestamps dropped).
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      sessionActive: false,
      transcript,
    });
    let ended: renderer.ReactTestRenderer;
    act(() => {
      ended = renderer.create(
        <LiveCoachScreen onReviewTranscript={onReviewTranscript} />,
      );
    });
    const button = ended!.root.findByProps({
      testID: "review-transcript-button",
    });
    act(() => button.props.onPress());
    expect(onReviewTranscript).toHaveBeenCalledWith([
      { speaker: "Speaker A", text: "You never listen." },
      { speaker: "Speaker B", text: "I'm trying." },
    ]);
  });

  it("renders a nudge entry as a compact banner, not a suggestion card", () => {
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      sessionActive: true,
      connectionStatus: "live",
      suggestions: [
        {
          id: 1,
          kind: "nudge" as const,
          texts: ["ease up"],
          tone: "balanced",
          muted: false,
          timestamp: 1,
        },
      ],
    });

    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    expect(
      root!.root.findAllByProps({ testID: "nudge-banner" }).length,
    ).toBeGreaterThan(0);
    // A nudge is NOT a full SuggestionCard stack.
    expect(
      root!.root.findAllByProps({ testID: "suggestion-card" }),
    ).toHaveLength(0);
    expect(JSON.stringify(root!.toJSON())).toContain("ease up");
  });
});
