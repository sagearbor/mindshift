import React from "react";
import renderer, { act } from "react-test-renderer";

/**
 * The journal deep-link actions on LiveCoachScreen ("Hey Google, start my
 * journal" — mindshift://journal/start|stop, App.tsx hands them over as the
 * `journalAction` prop): start selects Journal mode and starts the session
 * through the hook seam once the voiceprint gate resolves; a missing owner
 * print lands with the gate message visible and does NOT start; stop stops a
 * running journal session and nothing else.
 */
const mockUseAudioStream = jest.fn();
jest.mock("../src/hooks/useAudioStream", () => ({
  useAudioStream: () => mockUseAudioStream(),
}));
const mockListVoicePeople = jest.fn();
jest.mock("../src/api/liveSessions", () => ({
  listVoicePeople: () => mockListVoicePeople(),
}));
jest.mock("../src/api/therapist", () => ({
  getTherapistLink: jest.fn().mockResolvedValue({ linked: false }),
}));
const mockLoadLiveMode = jest.fn();
const mockSaveLiveMode = jest.fn();
jest.mock("../src/live/modePrefs", () => ({
  loadLiveMode: (uid: string | null) => mockLoadLiveMode(uid),
  saveLiveMode: (uid: string | null, mode: string) => mockSaveLiveMode(uid, mode),
}));
jest.mock("../src/api/client", () => ({
  postShare: jest.fn(),
  listVoicePeople: jest.fn().mockResolvedValue({ people: [] }),
}));

import LiveCoachScreen from "../src/screens/LiveCoachScreen";
import { IDLE_JOURNAL_STATE, type JournalState } from "../src/live/journalRecorder";

function makeHook(overrides: Record<string, unknown> = {}) {
  return {
    isRecording: false,
    sessionActive: false,
    transcript: [],
    suggestions: [],
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
    liveCapable: true,
    liveCapabilityReason: "ok",
    liveMode: true,
    setLiveMode: jest.fn(),
    sessionMode: "journal" as const,
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
    journal: IDLE_JOURNAL_STATE as JournalState,
    retryJournalUploads: jest.fn(),
    ...overrides,
  };
}

const SELF_PERSON = { personId: "self", displayName: "You", isSelf: true, enrollCount: 2, settings: 2 };
const OTHER_PERSON = { personId: "p2", displayName: "Mom", isSelf: false, enrollCount: 1, settings: 1 };

const flush = () => act(async () => { await Promise.resolve(); });

function render(props: React.ComponentProps<typeof LiveCoachScreen>) {
  let root: renderer.ReactTestRenderer;
  act(() => {
    root = renderer.create(<LiveCoachScreen {...props} />);
  });
  return root!;
}

function has(root: renderer.ReactTestRenderer, testID: string): boolean {
  return root.root.findAllByProps({ testID }).length > 0;
}

beforeEach(() => {
  mockListVoicePeople.mockReset().mockResolvedValue({ people: [SELF_PERSON, OTHER_PERSON], error: null });
  mockLoadLiveMode.mockReset().mockResolvedValue("earpiece");
  mockSaveLiveMode.mockReset().mockResolvedValue(undefined);
});

describe("LiveCoachScreen — journal deep-link actions", () => {
  it("start: selects Journal mode and starts the session once the gate is ok", async () => {
    const hook = makeHook();
    mockUseAudioStream.mockReturnValue(hook);
    const consumed = jest.fn();
    const root = render({ journalAction: "start", onJournalActionConsumed: consumed });
    await flush();
    // The visit-scoped mode override (never persisted).
    expect(hook.setSessionMode).toHaveBeenCalledWith("journal");
    expect(mockSaveLiveMode).not.toHaveBeenCalled();
    expect(hook.startSession).toHaveBeenCalledTimes(1);
    expect(hook.startSession).toHaveBeenCalledWith(expect.stringMatching(/^live-\d+$/), 50, 0);
    expect(consumed).toHaveBeenCalled();
    // A re-render with the prop still set (parent hasn't cleared it yet) must
    // not start a second session.
    act(() => {
      root.update(<LiveCoachScreen journalAction="start" onJournalActionConsumed={consumed} />);
    });
    await flush();
    expect(hook.startSession).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
  });

  it("start: the remembered-mode load also lands on journal, not the saved mode", async () => {
    const hook = makeHook();
    mockUseAudioStream.mockReturnValue(hook);
    mockLoadLiveMode.mockResolvedValue("therapist");
    render({ journalAction: "start" });
    await flush();
    // Both the loader and the override pick journal; the saved mode never wins.
    expect(hook.setSessionMode).toHaveBeenCalledWith("journal");
    expect(hook.setSessionMode).not.toHaveBeenCalledWith("therapist");
  });

  it("start with no owner voiceprint: lands on the gate message, does NOT start", async () => {
    const hook = makeHook();
    mockUseAudioStream.mockReturnValue(hook);
    mockListVoicePeople.mockResolvedValue({ people: [OTHER_PERSON], error: null });
    const consumed = jest.fn();
    const root = render({ journalAction: "start", onJournalActionConsumed: consumed });
    await flush();
    expect(hook.startSession).not.toHaveBeenCalled();
    expect(consumed).toHaveBeenCalled();
    // Journal mode is selected and its gate message is on screen.
    expect(has(root, "journal-panel")).toBe(true);
    expect(has(root, "journal-gate")).toBe(true);
  });

  it("start while the gate is still checking: waits (neither starts nor consumes)", async () => {
    const hook = makeHook();
    mockUseAudioStream.mockReturnValue(hook);
    mockListVoicePeople.mockReturnValue(new Promise(() => {})); // never resolves
    const consumed = jest.fn();
    render({ journalAction: "start", onJournalActionConsumed: consumed });
    await flush();
    expect(hook.startSession).not.toHaveBeenCalled();
    expect(consumed).not.toHaveBeenCalled();
  });

  it("start waits for the mode override to land in the hook", async () => {
    // The hook still reports earpiece (the override hasn't round-tripped):
    // the executor must not start a journal session under the wrong mode.
    const hook = makeHook({ sessionMode: "earpiece" as const });
    mockUseAudioStream.mockReturnValue(hook);
    render({ journalAction: "start" });
    await flush();
    expect(hook.setSessionMode).toHaveBeenCalledWith("journal");
    expect(hook.startSession).not.toHaveBeenCalled();
  });

  it("start while a journal session is already running: consumes without restarting", async () => {
    const hook = makeHook({ sessionActive: true, isRecording: true });
    mockUseAudioStream.mockReturnValue(hook);
    const consumed = jest.fn();
    render({ journalAction: "start", onJournalActionConsumed: consumed });
    await flush();
    expect(hook.startSession).not.toHaveBeenCalled();
    expect(consumed).toHaveBeenCalled();
  });

  it("stop: stops a running journal session", async () => {
    const hook = makeHook({ sessionActive: true, isRecording: true });
    mockUseAudioStream.mockReturnValue(hook);
    const consumed = jest.fn();
    render({ journalAction: "stop", onJournalActionConsumed: consumed });
    await flush();
    expect(hook.stopSession).toHaveBeenCalledTimes(1);
    expect(consumed).toHaveBeenCalled();
  });

  it("stop with nothing running: a consumed no-op", async () => {
    const hook = makeHook();
    mockUseAudioStream.mockReturnValue(hook);
    const consumed = jest.fn();
    render({ journalAction: "stop", onJournalActionConsumed: consumed });
    await flush();
    expect(hook.stopSession).not.toHaveBeenCalled();
    expect(consumed).toHaveBeenCalled();
  });

  it("stop never touches a running non-journal session (and doesn't re-mode it)", async () => {
    const hook = makeHook({ sessionActive: true, sessionMode: "earpiece" as const });
    mockUseAudioStream.mockReturnValue(hook);
    const consumed = jest.fn();
    render({ journalAction: "stop", onJournalActionConsumed: consumed });
    await flush();
    expect(hook.stopSession).not.toHaveBeenCalled();
    expect(hook.setSessionMode).not.toHaveBeenCalledWith("journal");
    expect(consumed).toHaveBeenCalled();
  });
});
