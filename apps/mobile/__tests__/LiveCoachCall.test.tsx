import React from "react";
import renderer, { act } from "react-test-renderer";

/**
 * Live Coach in Call mode (the screen + CallPanel over a mocked hook):
 * the mode chips (In person / Call), the pre-flight explainer, Start a
 * call / Join with code / Answer, the in-call header and controls.
 */
const mockUseAudioStream = jest.fn();
jest.mock("../src/hooks/useAudioStream", () => ({
  useAudioStream: () => mockUseAudioStream(),
}));
jest.mock("../src/api/liveSessions", () => ({
  listVoicePeople: jest.fn().mockResolvedValue({ people: [], error: null }),
}));
jest.mock("../src/api/therapist", () => ({
  getTherapistLink: jest.fn().mockResolvedValue({ linked: false }),
}));
const mockLoadLiveMode = jest.fn();
jest.mock("../src/live/modePrefs", () => ({
  loadLiveMode: () => mockLoadLiveMode(),
  saveLiveMode: jest.fn().mockResolvedValue(undefined),
}));
jest.mock("../src/api/client", () => ({ postShare: jest.fn(), listVoicePeople: jest.fn() }));

import LiveCoachScreen from "../src/screens/LiveCoachScreen";
import { LIVE_MODE_OPTIONS } from "../src/components/LiveModePicker";
import { IDLE_CALL_VIEW, type CallView } from "../src/live/call/types";
import CallPanel, { CALL_MODE_EXPLAINER, formatElapsed } from "../src/components/CallPanel";

const base = {
  isRecording: false,
  sessionActive: false,
  transcript: [] as unknown[],
  suggestions: [] as unknown[],
  speakerLabel: "",
  selfSpeaker: "Speaker A",
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
  liveCapabilityReason: "n/a",
  liveMode: false,
  setLiveMode: jest.fn(),
  sessionMode: "call" as const,
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
  call: IDLE_CALL_VIEW,
  startCall: jest.fn(),
  joinCall: jest.fn(),
  hangUp: jest.fn(),
  setCallMuted: jest.fn(),
  callRoute: "speaker" as const,
  setCallRoute: jest.fn(),
};

const flush = () => act(async () => { await Promise.resolve(); });
const text = (root: renderer.ReactTestRenderer) => JSON.stringify(root.toJSON());

beforeEach(() => {
  mockUseAudioStream.mockReturnValue({ ...base });
  mockLoadLiveMode.mockReset().mockResolvedValue("call");
});

describe("Live Coach — Call mode", () => {
  it("offers four modes: the old speaker-phone is now 'In person', plus 'Call'", () => {
    expect(LIVE_MODE_OPTIONS.map((o) => [o.mode, o.label])).toEqual([
      ["earpiece", "Earpiece"],
      ["speaker", "In person"],
      ["therapist", "Therapist"],
      ["call", "Call"],
    ]);
  });

  it("idle: explains why the app places the call, and starts / joins one", async () => {
    const startCall = jest.fn();
    const joinCall = jest.fn();
    mockUseAudioStream.mockReturnValue({ ...base, startCall, joinCall });
    let root!: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(root.root.findByProps({ testID: "call-explainer" }).props.children).toBe(CALL_MODE_EXPLAINER);
    expect(text(root)).toContain("MindShift places the call itself");
    // No Start Listening button in Call mode — the panel drives it.
    expect(root.root.findAllByProps({ testID: "mic-toggle" })).toHaveLength(0);
    // The identity chip is meaningless in a call (only "you" on this mic).
    expect(root.root.findAllByProps({ testID: "self-speaker-chip" })).toHaveLength(0);

    act(() => root.root.findByProps({ testID: "call-start" }).props.onPress());
    expect(startCall).toHaveBeenCalledWith(50, 0);

    const join = root.root.findByProps({ testID: "call-join" });
    expect(join.props.disabled).toBe(true);
    act(() => root.root.findByProps({ testID: "call-code-input" }).props.onChangeText("  K7M2PQ "));
    expect(root.root.findByProps({ testID: "call-join" }).props.disabled).toBe(false);
    act(() => root.root.findByProps({ testID: "call-join" }).props.onPress());
    expect(joinCall).toHaveBeenCalledWith("K7M2PQ", 50, 0);
  });

  it("an invite link opens Call mode with one Answer tap", async () => {
    const joinCall = jest.fn();
    const setSessionMode = jest.fn();
    const consumed = jest.fn();
    mockLoadLiveMode.mockResolvedValue("earpiece");
    mockUseAudioStream.mockReturnValue({ ...base, joinCall, setSessionMode });
    let root!: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen joinCode="K7M2PQ" onJoinCodeConsumed={consumed} />);
    });
    await flush();
    // The invite overrides the remembered mode for this visit.
    expect(setSessionMode).toHaveBeenCalledWith("call");
    expect(setSessionMode).not.toHaveBeenCalledWith("earpiece");
    expect(root.root.findByProps({ testID: "call-invited" }).props.children.join("")).toBe(
      "You've been invited to a call (code K7M2PQ).",
    );
    expect(root.root.findAllByProps({ testID: "call-start" })).toHaveLength(0);
    act(() => root.root.findByProps({ testID: "call-answer" }).props.onPress());
    expect(joinCall).toHaveBeenCalledWith("K7M2PQ", 50, 0, "participant");
    expect(consumed).toHaveBeenCalled();
  });

  it("shows the failure reason honestly", async () => {
    mockUseAudioStream.mockReturnValue({
      ...base,
      call: { ...IDLE_CALL_VIEW, status: "failed", error: "couldn't start a call: this server has no in-app calls yet (404)" },
    });
    let root!: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(root.root.findByProps({ testID: "call-error" }).props.children).toContain("no in-app calls yet");
  });

  it("in a call: header with name · status · timer, invite while waiting, mute / route / hang up", async () => {
    const hangUp = jest.fn();
    const setCallMuted = jest.fn();
    const setCallRoute = jest.fn();
    const waiting: CallView = { ...IDLE_CALL_VIEW, status: "waiting", callId: "c1", joinCode: "K7M2PQ", joinUrl: "https://arborfam-hub.web.app/call/K7M2PQ" };
    mockUseAudioStream.mockReturnValue({ ...base, sessionActive: true, isRecording: true, connectionStatus: "live", call: waiting, hangUp, setCallMuted, setCallRoute });
    let root!: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveCoachScreen />);
    });
    await flush();
    expect(root.root.findByProps({ testID: "call-header" }).props.children).toBe("You · waiting for them");
    expect(root.root.findByProps({ testID: "call-invite-code" }).props.children.join("")).toBe("Code: K7M2PQ");
    expect(text(root)).toContain("https://arborfam-hub.web.app/call/K7M2PQ");
    // The bottom button hangs up in Call mode.
    expect(text(root)).toContain("Hang up");
    act(() => root.root.findByProps({ testID: "mic-toggle" }).props.onPress());
    expect(hangUp).toHaveBeenCalledTimes(1);

    const connected: CallView = { ...waiting, status: "connected", peers: [{ uid: "b", label: "Speaker B", displayName: "Mom", role: "participant", connected: true, iceRestarts: 0 }], connectedAt: Date.now() - 192_000 };
    mockUseAudioStream.mockReturnValue({ ...base, sessionActive: true, isRecording: true, connectionStatus: "live", call: connected, hangUp, setCallMuted, setCallRoute });
    act(() => root.update(<LiveCoachScreen />));
    expect(root.root.findByProps({ testID: "call-header" }).props.children).toMatch(/^You · connected · 03:1[0-9]$/);
    expect(root.root.findByProps({ testID: "call-peer-b" }).props.children.join("")).toBe("Mom · connected");
    expect(root.root.findAllByProps({ testID: "call-invite" })).toHaveLength(0);
    act(() => root.root.findByProps({ testID: "call-mute" }).props.onPress());
    expect(setCallMuted).toHaveBeenCalledWith(true);
    act(() => root.root.findByProps({ testID: "call-route" }).props.onPress());
    expect(setCallRoute).toHaveBeenCalledWith("earpiece");
    act(() => root.root.findByProps({ testID: "call-hangup" }).props.onPress());
    expect(hangUp).toHaveBeenCalledTimes(2);

    // The other person's turns render under their name like any transcript line.
    mockUseAudioStream.mockReturnValue({
      ...base,
      sessionActive: true,
      call: connected,
      transcript: [{ speaker: "Mom", text: "You never call me.", timestamp: 1 }],
    });
    act(() => root.update(<LiveCoachScreen />));
    expect(text(root)).toContain("You never call me.");
    expect(text(root)).toContain("Mom");
  });

  it("formats the timer and reports 'reconnecting'", () => {
    expect(formatElapsed(0)).toBe("00:00");
    expect(formatElapsed(192_400)).toBe("03:12");
    let root!: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(
        <CallPanel
          call={{ ...IDLE_CALL_VIEW, status: "reconnecting", peers: [{ uid: "b", label: "Speaker B", displayName: "Mom", role: "participant", connected: false, iceRestarts: 1 }], connectedAt: 5 }}
          sessionActive
          onStart={jest.fn()}
          onJoin={jest.fn()}
          onHangUp={jest.fn()}
          onToggleMute={jest.fn()}
          now={() => 65_005}
        />,
      );
    });
    expect(root.root.findByProps({ testID: "call-header" }).props.children).toBe("You · reconnecting · 01:05");
    expect(root.root.findByProps({ testID: "call-reconnecting" })).toBeTruthy();
  });
});
