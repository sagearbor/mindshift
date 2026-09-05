import React from "react";
import renderer, { act } from "react-test-renderer";

const mockUseAudioStream = jest.fn();

jest.mock("../src/hooks/useAudioStream", () => ({
  useAudioStream: () => mockUseAudioStream(),
}));

// The screen's mount-time reads: enrolled people ("who's here"), the
// therapist link (end-of-session share), and the persisted mode. All
// deterministic here; the real modules are covered by their own suites.
const mockListVoicePeople = jest.fn();
jest.mock("../src/api/liveSessions", () => ({
  listVoicePeople: () => mockListVoicePeople(),
}));
const mockGetTherapistLink = jest.fn();
jest.mock("../src/api/therapist", () => ({
  getTherapistLink: () => mockGetTherapistLink(),
}));
const mockLoadLiveMode = jest.fn();
const mockSaveLiveMode = jest.fn();
jest.mock("../src/live/modePrefs", () => ({
  loadLiveMode: (uid: string | null) => mockLoadLiveMode(uid),
  saveLiveMode: (uid: string | null, mode: string) => mockSaveLiveMode(uid, mode),
}));
jest.mock("../src/api/client", () => ({
  postShare: jest.fn(),
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
  liveCapable: false,
  liveCapabilityReason: "on-device speech recognition isn't available here",
  liveMode: false,
  setLiveMode: jest.fn(),
  sessionMode: "earpiece" as const,
  setSessionMode: jest.fn(),
  liveStatus: "",
  nudgeFlash: null,
  clearNudgeFlash: jest.fn(),
  latencySummary: "",
  toneFlags: [],
  preflight: null,
  runPreflight: jest.fn(),
  escalationCount: 0,
  sessionSummary: null,
  lastEpisode: null,
};

/** Build a "response" feed entry (the shape the hook now exposes). */
function responseEntry(
  texts: string[],
  { id = 1, tone = "empathetic", muted = false, source = undefined as undefined | "on-device" | "cloud" } = {},
) {
  return { id, kind: "response" as const, texts, tone, muted, timestamp: id, ...(source ? { source } : {}) };
}

const flush = () => act(async () => { await Promise.resolve(); });

import { useDevModeStore } from "../src/store/devModeStore";

beforeEach(() => {
  // Most of this suite asserts the full diagnostic surface (and its
  // snapshots predate developer mode) — run it as the owner does, dev ON.
  useDevModeStore.setState({ devMode: true });
  mockUseAudioStream.mockReturnValue({ ...defaultHookState });
  mockListVoicePeople.mockReset().mockResolvedValue({ people: [], error: null });
  mockGetTherapistLink.mockReset().mockResolvedValue({ linked: false });
  mockLoadLiveMode.mockReset().mockResolvedValue("earpiece");
  mockSaveLiveMode.mockReset().mockResolvedValue(undefined);
});

describe("LiveCoachScreen — developer mode off (clean tester surface)", () => {
  it("plain status word; capability, latency and mode-row chrome hidden", async () => {
    useDevModeStore.setState({ devMode: false });
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      liveStatus: "On-device: Silero VAD · speaker-ID off (no model)",
      latencySummary: "p50 1200ms to speak",
    });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    const json = JSON.stringify(root!.toJSON());
    expect(json).toContain("ready"); // friendly word, not the raw "idle"
    expect(json).not.toContain("speaker-ID");
    expect(json).not.toContain("p50 1200ms");
    expect(root!.root.findAllByProps({ testID: "live-mode-row" })).toHaveLength(0);
    expect(root!.root.findAllByProps({ testID: "live-status" })).toHaveLength(0);
    expect(root!.root.findByProps({ testID: "preflight-plain" })).toBeTruthy();
  });
});

describe("LiveCoachScreen", () => {
  it("renders initial idle state", async () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders recording state with transcript and suggestions", async () => {
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
    await flush();
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders disconnected state", async () => {
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      connectionStatus: "disconnected",
    });

    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("shows the mic error banner when capture fails", async () => {
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      micError: "Microphone permission denied — enable microphone access.",
    });

    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    const banner = root!.root.findByProps({ testID: "mic-error-banner" });
    expect(banner).toBeTruthy();
    // The honest failure message is shown verbatim.
    const text = JSON.stringify(root!.toJSON());
    expect(text).toContain("Microphone permission denied");
  });

  it("hides the mic error banner when there is no error", async () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(
      root!.root.findAllByProps({ testID: "mic-error-banner" }),
    ).toHaveLength(0);
  });

  it("the mode decides speech: earpiece/speaker speak, therapist never does", async () => {
    const setSpeechEnabled = jest.fn();
    mockUseAudioStream.mockReturnValue({ ...defaultHookState, setSpeechEnabled });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(setSpeechEnabled).toHaveBeenLastCalledWith(true);

    mockUseAudioStream.mockReturnValue({ ...defaultHookState, setSpeechEnabled, sessionMode: "speaker" });
    act(() => root!.update(<LiveCoachScreen />));
    expect(setSpeechEnabled).toHaveBeenLastCalledWith(true);

    mockUseAudioStream.mockReturnValue({ ...defaultHookState, setSpeechEnabled, sessionMode: "therapist" });
    act(() => root!.update(<LiveCoachScreen />));
    expect(setSpeechEnabled).toHaveBeenLastCalledWith(false);
  });

  it("mode picker: shows each mode's one-line hint, persists the choice per account, locks while live", async () => {
    const setSessionMode = jest.fn();
    mockUseAudioStream.mockReturnValue({ ...defaultHookState, setSessionMode });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    // The persisted mode is applied on mount.
    expect(mockLoadLiveMode).toHaveBeenCalled();
    expect(setSessionMode).toHaveBeenCalledWith("earpiece");
    expect(JSON.stringify(root!.toJSON())).toContain("the coach whispers to you privately");

    act(() => {
      root!.root.findByProps({ testID: "session-mode-speaker" }).props.onPress();
    });
    expect(setSessionMode).toHaveBeenLastCalledWith("speaker");
    expect(mockSaveLiveMode).toHaveBeenLastCalledWith(null, "speaker");

    mockUseAudioStream.mockReturnValue({ ...defaultHookState, setSessionMode, sessionMode: "therapist" });
    act(() => root!.update(<LiveCoachScreen />));
    expect(JSON.stringify(root!.toJSON())).toContain("nothing is ever spoken");

    mockUseAudioStream.mockReturnValue({ ...defaultHookState, setSessionMode, sessionActive: true });
    act(() => root!.update(<LiveCoachScreen />));
    expect(root!.root.findByProps({ testID: "session-mode-speaker" }).props.disabled).toBe(true);
  });

  it("applies a remembered mode ('In person' — stored as `speaker` — on Sage's phone)", async () => {
    mockLoadLiveMode.mockResolvedValue("speaker");
    const setSessionMode = jest.fn();
    mockUseAudioStream.mockReturnValue({ ...defaultHookState, setSessionMode });
    act(() => {
      renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(setSessionMode).toHaveBeenCalledWith("speaker");
  });

  it("shows an honest note when a spoken mode is selected but TTS is unavailable", async () => {
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      speechAvailable: false,
    });

    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(root!.root.findByProps({ testID: "speech-unavailable-note" })).toBeTruthy();

    // Therapist mode never speaks, so there's nothing to warn about.
    mockUseAudioStream.mockReturnValue({ ...defaultHookState, speechAvailable: false, sessionMode: "therapist" });
    act(() => root!.update(<LiveCoachScreen />));
    expect(root!.root.findAllByProps({ testID: "speech-unavailable-note" })).toHaveLength(0);
  });

  it("hides the unavailable note when TTS works", async () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(root!.root.findAllByProps({ testID: "speech-unavailable-note" })).toHaveLength(0);
  });

  it("moving the interject slider updates local state and notifies the hook", async () => {
    const sendInterjectUpdate = jest.fn();
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      sendInterjectUpdate,
    });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    const slider = root!.root.findByProps({ testID: "interject-slider" });
    act(() => {
      slider.props.onValueChange(70.4);
    });
    expect(sendInterjectUpdate).toHaveBeenCalledWith(70);
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
    await flush();
    act(() => {
      root!.root.findByProps({ testID: "interject-slider" }).props.onValueChange(30);
    });
    await act(async () => {
      await root!.root.findByProps({ testID: "mic-toggle" }).props.onPress();
    });
    expect(startSession).toHaveBeenCalledWith(expect.stringMatching(/^live-/), 50, 30);
  });

  it("dims muted suggestions instead of hiding them", async () => {
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      // Two separate feed entries: newest first, the older one muted.
      suggestions: [
        responseEntry(["Spoken advice."], { id: 2, tone: "balanced" }),
        responseEntry(["Quiet aside."], { id: 1, tone: "balanced", muted: true }),
      ],
    });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    // Host nodes only — RN's <View> also yields a composite node carrying
    // the same testID.
    const cards = root!.root.findAll(
      (node) => node.props.testID === "suggestion-card" && typeof node.type === "string",
    );
    expect(cards).toHaveLength(2);
    expect(cards[0].props.style).not.toContainEqual(expect.objectContaining({ opacity: 0.5 }));
    expect(cards[1].props.style).toContainEqual(expect.objectContaining({ opacity: 0.5 }));
  });

  it("tags each suggestion with where it came from (on-device first, cloud augments)", async () => {
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      sessionActive: true,
      connectionStatus: "live",
      suggestions: [
        responseEntry(["Cloud's take."], { id: 2, source: "cloud" }),
        responseEntry(["Phone's take."], { id: 1, source: "on-device" }),
      ],
    });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    const tagCloud = root!.root.findByProps({ testID: "suggestion-source-2" });
    const tagLocal = root!.root.findByProps({ testID: "suggestion-source-1" });
    expect(JSON.stringify(tagCloud.props.children)).toContain("cloud");
    expect(JSON.stringify(tagLocal.props.children)).toContain("on-device");
  });

  it("identity chip appears with a session and toggles the self speaker", async () => {
    const setSelfSpeaker = jest.fn();
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      sessionActive: true,
      selfSpeaker: "Speaker A",
      setSelfSpeaker,
    });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    act(() => {
      root!.root.findByProps({ testID: "self-speaker-chip" }).props.onPress();
    });
    expect(setSelfSpeaker).toHaveBeenCalledWith("Speaker B");
  });

  it("hides the identity chip before any session — and always in therapist mode", async () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(root!.root.findAllByProps({ testID: "self-speaker-chip" })).toHaveLength(0);
    mockUseAudioStream.mockReturnValue({ ...defaultHookState, sessionActive: true, sessionMode: "therapist" });
    act(() => root!.update(<LiveCoachScreen />));
    expect(root!.root.findAllByProps({ testID: "self-speaker-chip" })).toHaveLength(0);
  });

  it("shows the pre-flight panel + explainer only when idle with no transcript", async () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(root!.root.findByProps({ testID: "idle-explainer" })).toBeTruthy();
    expect(root!.root.findByProps({ testID: "live-preflight" })).toBeTruthy();

    mockUseAudioStream.mockReturnValue({ ...defaultHookState, sessionActive: true });
    act(() => root!.update(<LiveCoachScreen />));
    expect(root!.root.findAllByProps({ testID: "idle-explainer" })).toHaveLength(0);
    expect(root!.root.findAllByProps({ testID: "live-preflight" })).toHaveLength(0);
  });

  it("the BEFORE mood check shows while idle and disappears once a session starts", async () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(root!.root.findByProps({ testID: "mood-check-before" })).toBeTruthy();

    mockUseAudioStream.mockReturnValue({ ...defaultHookState, sessionActive: true });
    act(() => root!.update(<LiveCoachScreen />));
    expect(root!.root.findAllByProps({ testID: "mood-check-before" })).toHaveLength(0);
  });

  it("pre-flight tells the truth: no on-device STT here, so the server labels voices", async () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    const text = JSON.stringify(root!.toJSON());
    expect(text).toContain("on-device speech recognition isn't available here");
    expect(text).toContain("server labels voices by speaking order");
    expect(defaultHookState.runPreflight).not.toHaveBeenCalled();
  });

  it("pre-flight on a capable phone: probes on mount and shows speaker-ID + LLM honestly", async () => {
    const runPreflight = jest.fn().mockResolvedValue(undefined);
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      liveCapable: true,
      liveMode: true,
      runPreflight,
      preflight: {
        status: "ready",
        capabilities: {
          vad: "silero",
          speakerId: { active: false, reason: "server has no ECAPA export (503)", enrolled: 0, model: null, droppedForModel: 0 },
          llm: ["os", "bundled", "cloud"],
        },
      },
    });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(runPreflight).toHaveBeenCalled();
    const text = JSON.stringify(root!.toJSON());
    expect(text).toContain("server has no ECAPA export (503)");
    expect(text).toContain("os → bundled");
    expect(text).toContain("Silero VAD");
  });

  it("who's here: lists the enrolled people, or says nobody is enrolled", async () => {
    mockListVoicePeople.mockResolvedValue({
      people: [
        { personId: "self", displayName: "You", isSelf: true, enrollCount: 3 },
        { personId: "mom", displayName: "Mom", isSelf: false, enrollCount: 2 },
      ],
      error: null,
    });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(root!.root.findByProps({ testID: "whos-here-self" })).toBeTruthy();
    expect(root!.root.findByProps({ testID: "whos-here-mom" })).toBeTruthy();

    mockListVoicePeople.mockResolvedValue({ people: [], error: null });
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(root!.root.findByProps({ testID: "whos-here-empty" })).toBeTruthy();
  });

  it("review button shows only after a session ends with a transcript, and hands off the mapped turns", async () => {
    const onReviewTranscript = jest.fn();
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      sessionActive: false,
      transcript: [
        { speaker: "Speaker A", text: "Hi", timestamp: 1, startTime: 0.5, endTime: 1.2 },
        { speaker: "Speaker B", text: "Hey", timestamp: 2 },
      ],
    });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen onReviewTranscript={onReviewTranscript} />);
    });
    await flush();
    act(() => {
      root!.root.findByProps({ testID: "review-transcript-button" }).props.onPress();
    });
    expect(onReviewTranscript).toHaveBeenCalledWith([
      { speaker: "Speaker A", text: "Hi", start_time: 0.5, end_time: 1.2 },
      { speaker: "Speaker B", text: "Hey" },
    ]);

    mockUseAudioStream.mockReturnValue({ ...defaultHookState, sessionActive: true, transcript: [{ speaker: "A", text: "x", timestamp: 1 }] });
    act(() => root!.update(<LiveCoachScreen onReviewTranscript={onReviewTranscript} />));
    expect(root!.root.findAllByProps({ testID: "review-transcript-button" })).toHaveLength(0);
  });

  it("renders a nudge entry as a compact banner, not a suggestion card", async () => {
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      suggestions: [{ id: 1, kind: "nudge" as const, texts: ["ease up"], tone: "balanced", muted: false, timestamp: 1 }],
    });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(root!.root.findByProps({ testID: "nudge-banner" })).toBeTruthy();
    expect(
      root!.root.findAllByProps({ testID: "suggestion-card" }),
    ).toHaveLength(0);
    expect(JSON.stringify(root!.toJSON())).toContain("ease up");
  });

  it("hides the on-device switch when the device can't run the fast loop", async () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(root!.root.findAllByProps({ testID: "live-mode-row" })).toHaveLength(0);
  });

  it("offers the on-device switch when capable, locked while live, with the loaded status + tone flag", async () => {
    const setLiveMode = jest.fn();
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      liveCapable: true,
      liveMode: true,
      setLiveMode,
    });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    const sw = root!.root.findByProps({ testID: "live-mode-switch" });
    expect(sw.props.value).toBe(true);
    expect(sw.props.disabled).toBe(false);

    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      sessionActive: true,
      liveCapable: true,
      liveMode: true,
      setLiveMode,
      sessionMode: "speaker",
      liveStatus: "On-device: Silero VAD · speaker-ID on (2 enrolled) · LLM os → bundled → cloud",
      escalationCount: 2,
      toneFlags: [{ type: "tone_flag", session_id: "s", speaker: "Mom", start_time: 0, end_time: 1, source: "text", scores: {}, label: "hurt", confidence: 0.7 }],
    });
    act(() => {
      root!.update(<LiveCoachScreen />);
    });
    expect(root!.root.findByProps({ testID: "live-mode-switch" }).props.disabled).toBe(true);
    const rendered = JSON.stringify(root!.toJSON());
    expect(rendered).toContain("On-device: Silero VAD");
    expect(rendered).toContain("Mom: hurt (text tone)");
    expect(rendered).toContain("escalations: 2");
  });

  it("therapist mode: two-column transcript, 'observing' strip, no self chip", async () => {
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      sessionActive: true,
      connectionStatus: "live",
      sessionMode: "therapist",
      transcript: [
        { speaker: "Sage", text: "I felt ignored.", timestamp: 1 },
        { speaker: "Mom", text: "I didn't mean to.", timestamp: 2 },
      ],
    });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(root!.root.findByProps({ testID: "therapist-transcript" })).toBeTruthy();
    expect(root!.root.findByProps({ testID: "therapist-turn-0-left" })).toBeTruthy();
    expect(root!.root.findByProps({ testID: "therapist-turn-1-right" })).toBeTruthy();
    expect(root!.root.findAllByProps({ testID: "live-transcript" })).toHaveLength(0);
    expect(JSON.stringify(root!.toJSON())).toContain("observing");
  });

  it("flashes a nudge and clears it after a moment; shows the latency headline after a session", async () => {
    jest.useFakeTimers();
    const clearNudgeFlash = jest.fn();
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      liveCapable: true,
      liveMode: true,
      nudgeFlash: { channel: "A", level: 2, t: 3, vectors: ["aggressive_tone", "yelling"] },
      clearNudgeFlash,
      latencySummary: "[fastLoop] 3 turns, median segment-end→speak 640 ms",
    });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    const rendered = JSON.stringify(root!.toJSON());
    expect(root!.root.findByProps({ testID: "nudge-flash" })).toBeTruthy();
    expect(rendered).toContain("Easy — level 2 (aggressive tone, yelling)");
    expect(rendered).toContain("median segment-end");
    act(() => {
      jest.advanceTimersByTime(1500);
    });
    expect(clearNudgeFlash).toHaveBeenCalled();
    jest.useRealTimers();
  });

  it("session end: summary card with the measured numbers and 'Share with my therapist' when linked", async () => {
    mockGetTherapistLink.mockResolvedValue({ linked: true, therapist_email: "mom@example.com", status: "accepted", auto_share: false });
    mockUseAudioStream.mockReturnValue({
      ...defaultHookState,
      transcript: [{ speaker: "You", text: "hi", timestamp: 1 }],
      sessionSummary: {
        durationMs: 134000,
        turnsBySpeaker: [{ speaker: "You", turns: 3 }, { speaker: "Mom", turns: 2 }],
        totalTurns: 5,
        escalations: 1,
        firstWordsMedianMs: 640,
        firstWordsBestMs: 410,
        spokenTurns: 2,
        topProvider: "os",
      },
      lastEpisode: { episodeId: "ep-1", postStatus: "created", sharedWith: [] },
    });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    const text = root!.root
      .findAll((n) => typeof n.type === "string")
      .flatMap((n) => n.children)
      .filter((c): c is string => typeof c === "string")
      .join("");
    expect(root!.root.findByProps({ testID: "session-summary" })).toBeTruthy();
    expect(text).toContain("2m 14s");
    expect(text).toContain("640 ms");
    expect(text).toContain("You: 3 · Mom: 2 · via os");
    expect(root!.root.findByProps({ testID: "summary-share-therapist" })).toBeTruthy();
    // The AFTER mood check shows alongside the summary.
    expect(root!.root.findByProps({ testID: "mood-check-after" })).toBeTruthy();
  });
});
