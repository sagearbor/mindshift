import React from "react";
import renderer, { act } from "react-test-renderer";

/**
 * LiveCoachScreen in JOURNAL mode: the picker offers it, the idle screen
 * shows the explainer + privacy note and the honest "enroll your voice
 * first" gate (Start disabled), the coaching UI (sliders, transcript,
 * keep-audio / speak-aloud / scoreboard rows) is gone, and while listening
 * the panel shows the counters the owner asked for.
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
import { LIVE_MODE_OPTIONS } from "../src/components/LiveModePicker";
import { IDLE_JOURNAL_STATE, type JournalState } from "../src/live/journalRecorder";
import { JOURNAL_ENROLL_NOTE, JOURNAL_PRIVACY_NOTE } from "../src/components/JournalPanel";

const baseHook = {
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
};

const SELF_PERSON = { personId: "self", displayName: "You", isSelf: true, enrollCount: 2, settings: 2 };
const OTHER_PERSON = { personId: "p2", displayName: "Mom", isSelf: false, enrollCount: 1, settings: 1 };

const flush = () => act(async () => { await Promise.resolve(); });

function render() {
  let root: renderer.ReactTestRenderer;
  act(() => {
    root = renderer.create(<LiveCoachScreen />);
  });
  return root!;
}

function has(root: renderer.ReactTestRenderer, testID: string): boolean {
  return root.root.findAllByProps({ testID }).length > 0;
}

/** Every string rendered under the node with this testID, joined. */
function textOf(root: renderer.ReactTestRenderer, testID: string): string {
  const node = root.root.findByProps({ testID });
  const out: string[] = [];
  const walk = (children: unknown) => {
    if (typeof children === "string" || typeof children === "number") out.push(String(children));
    else if (Array.isArray(children)) children.forEach(walk);
  };
  walk(node.props.children);
  for (const n of node.findAll(() => true)) walk(n.props.children);
  return out.join("");
}

beforeEach(() => {
  mockUseAudioStream.mockReturnValue({ ...baseHook });
  mockListVoicePeople.mockReset().mockResolvedValue({ people: [SELF_PERSON, OTHER_PERSON], error: null });
  mockLoadLiveMode.mockReset().mockResolvedValue("journal");
  mockSaveLiveMode.mockReset().mockResolvedValue(undefined);
});

describe("LiveCoachScreen — Journal mode", () => {
  it("offers Journal in the mode picker with its one-line hint", () => {
    const journal = LIVE_MODE_OPTIONS.find((o) => o.mode === "journal");
    expect(journal?.label).toBe("Journal");
    expect(journal?.hint).toMatch(/keeps only what you say/);
    expect(journal?.hint).toMatch(/no coaching, no transcription until later/);
  });

  it("idle with an enrolled owner: explainer + privacy note, coaching UI gone, Start Journal enabled", async () => {
    const root = render();
    await flush();
    expect(has(root, "journal-panel")).toBe(true);
    expect(has(root, "journal-privacy")).toBe(true);
    const json = JSON.stringify(root.toJSON());
    expect(json).toContain(JOURNAL_PRIVACY_NOTE);
    expect(json).toContain("Journal — listen for my voice");
    expect(has(root, "journal-gate")).toBe(false);
    // None of the coaching chrome.
    for (const id of ["keep-audio-row", "speak-aloud-row", "scoreboard-row", "live-mode-row", "idle-explainer", "self-speaker-chip"]) {
      expect(has(root, id)).toBe(false);
    }
    expect(json).not.toContain("Empathy");
    const toggle = root.root.findByProps({ testID: "mic-toggle" });
    expect(toggle.props.disabled).toBe(false);
    expect(textOf(root, "mic-toggle")).toContain("Start Journal");
    act(() => {
      toggle.props.onPress();
    });
    expect(baseHook.startSession).toHaveBeenCalled();
    // The mode itself is still switchable (and persisted) while idle.
    act(() => {
      root.root.findByProps({ testID: "session-mode-earpiece" }).props.onPress();
    });
    expect(baseHook.setSessionMode).toHaveBeenCalledWith("earpiece");
    expect(mockSaveLiveMode).toHaveBeenCalledWith(null, "earpiece");
  });

  it("without an owner voiceprint: says 'enroll your voice first' and disables Start", async () => {
    mockListVoicePeople.mockResolvedValue({ people: [OTHER_PERSON], error: null });
    const root = render();
    await flush();
    expect(textOf(root, "journal-gate")).toContain(JOURNAL_ENROLL_NOTE);
    const toggle = root.root.findByProps({ testID: "mic-toggle" });
    expect(toggle.props.disabled).toBe(true);
  });

  it("an owner print with no recording pooled into it counts as missing", async () => {
    mockListVoicePeople.mockResolvedValue({ people: [{ ...SELF_PERSON, enrollCount: 0, settings: 0 }], error: null });
    const root = render();
    await flush();
    expect(has(root, "journal-gate")).toBe(true);
    expect(root.root.findByProps({ testID: "mic-toggle" }).props.disabled).toBe(true);
  });

  it("when the people list can't be fetched, Start stays enabled (the hook enforces the gate)", async () => {
    mockListVoicePeople.mockResolvedValue({ people: [], error: "people unreachable" });
    const root = render();
    await flush();
    expect(has(root, "journal-gate")).toBe(false);
    expect(root.root.findByProps({ testID: "mic-toggle" }).props.disabled).toBe(false);
  });

  it("while listening: elapsed, how often you spoke, last heard, file size, uploads, Stop Journal", async () => {
    const now = Date.now();
    mockUseAudioStream.mockReturnValue({
      ...baseHook,
      sessionActive: true,
      isRecording: true,
      journal: {
        ...IDLE_JOURNAL_STATE,
        status: "listening",
        startedAt: now - 3725_000,
        listeningSeconds: 3725,
        selfCount: 12,
        selfSeconds: 271,
        lastSelfAt: now - 30_000,
        fileBytes: 2.5 * 1024 * 1024,
        fileStartedAt: now - 600_000,
        filesClosed: 2,
        uploads: { pending: 1, sent: 1, failed: 1, lastError: "network down", inFlight: false },
      } satisfies JournalState,
    });
    const root = render();
    await flush();
    const text = (id: string) => textOf(root, id);
    expect(text("journal-elapsed")).toContain("1:02:05");
    expect(text("journal-self")).toContain("You spoke 12 times");
    expect(text("journal-self")).toContain("4.5 min");
    expect(text("journal-last-heard")).toContain("Last heard you:");
    expect(text("journal-size")).toContain("2.5 MB");
    expect(text("journal-size")).toContain("2 closed");
    expect(text("journal-uploads")).toContain("1 sent · 1 waiting · 1 failed (kept for retry)");
    expect(has(root, "journal-quiet-note")).toBe(false);
    expect(has(root, "journal-privacy")).toBe(true);
    // Nothing coaching-shaped while it runs either.
    expect(has(root, "session-strip")).toBe(false);
    expect(has(root, "keep-audio-row")).toBe(false);
    const toggle = root.root.findByProps({ testID: "mic-toggle" });
    expect(textOf(root, "mic-toggle")).toContain("Stop Journal");
    act(() => {
      toggle.props.onPress();
    });
    expect(baseHook.stopSession).toHaveBeenCalled();
    act(() => {
      root.root.findByProps({ testID: "journal-retry-uploads" }).props.onPress();
    });
    expect(baseHook.retryJournalUploads).toHaveBeenCalled();
  });

  it("says when it hasn't heard you for 10 minutes — and keeps listening", async () => {
    const now = Date.now();
    mockUseAudioStream.mockReturnValue({
      ...baseHook,
      sessionActive: true,
      journal: {
        ...IDLE_JOURNAL_STATE,
        status: "listening",
        startedAt: now - 900_000,
        listeningSeconds: 900,
        lastSelfAt: now - 720_000,
        selfCount: 1,
        selfSeconds: 3,
      } satisfies JournalState,
    });
    const root = render();
    await flush();
    expect(textOf(root, "journal-quiet-note")).toContain("12");
    expect(textOf(root, "journal-quiet-note")).toContain("still listening");
  });

  it("shows the journal's error (a refused start) in place", async () => {
    mockUseAudioStream.mockReturnValue({
      ...baseHook,
      journal: { ...IDLE_JOURNAL_STATE, error: "Enroll your voice first — the journal keeps only the stretches that match your voiceprint." },
    });
    const root = render();
    await flush();
    expect(textOf(root, "journal-error")).toContain("Enroll your voice first");
  });
});
