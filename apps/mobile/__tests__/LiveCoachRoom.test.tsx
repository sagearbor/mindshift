import React from "react";
import renderer, { act } from "react-test-renderer";

/**
 * LiveCoachScreen in ROOM mode: the picker offers it; idle shows the room
 * explainer with the consent note and the library picker (no coaching UI);
 * a running session shows the persistent consent banner, the latest spoken
 * answer and the library cards in the big layout, and Stop is one tap away.
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
jest.mock("../src/live/modePrefs", () => ({
  loadLiveMode: jest.fn().mockResolvedValue("room"),
  saveLiveMode: jest.fn().mockResolvedValue(undefined),
}));
jest.mock("../src/api/client", () => ({
  postShare: jest.fn(),
  listVoicePeople: jest.fn().mockResolvedValue({ people: [] }),
}));
let mockLibrarySaved: string[] = [];
jest.mock("../src/live/librarySelection", () => ({
  ...jest.requireActual("../src/live/librarySelection"),
  defaultLibrarySelectionStore: () => ({ load: () => mockLibrarySaved, save: jest.fn() }),
}));
jest.mock("../src/api/library", () => ({
  ...jest.requireActual("../src/api/library"),
  listLibrary: jest.fn().mockResolvedValue({ items: [], retrieval_available: true }),
}));

import LiveCoachScreen from "../src/screens/LiveCoachScreen";
import { LIVE_MODE_OPTIONS } from "../src/components/LiveModePicker";
import { ROOM_BANNER_TITLE, ROOM_CONSENT_NOTE } from "../src/components/RoomPanel";
import { IDLE_ROOM_STATE, type RoomViewState } from "../src/live/roomMode";
import { IDLE_JOURNAL_STATE } from "../src/live/journalRecorder";
import { useDevModeStore } from "../src/store/devModeStore";

const baseHook = {
  isRecording: false,
  sessionActive: false,
  transcript: [] as { speaker: string; text: string; timestamp: number }[],
  suggestions: [],
  speakerLabel: "",
  selfSpeaker: null as string | null,
  setSelfSpeaker: jest.fn(),
  connectionStatus: "idle" as const,
  transcriptionAvailable: true,
  transcriptionMessage: "",
  micError: "",
  speechAvailable: true,
  speechEnabled: true,
  setSpeechEnabled: jest.fn(),
  startSession: jest.fn(),
  stopSession: jest.fn(),
  sendEmpathyUpdate: jest.fn(),
  sendInterjectUpdate: jest.fn(),
  liveCapable: true,
  liveCapabilityReason: "ok",
  liveMode: true,
  setLiveMode: jest.fn(),
  sessionMode: "room" as const,
  setSessionMode: jest.fn(),
  setRelationship: jest.fn(),
  setSessionContext: jest.fn(),
  setLibraryItemIds: jest.fn(),
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
  journal: IDLE_JOURNAL_STATE,
  retryJournalUploads: jest.fn(),
  room: IDLE_ROOM_STATE as RoomViewState,
};

const LIVE_ROOM: RoomViewState = {
  ...IDLE_ROOM_STATE,
  serverConfirmed: true,
  cards: [
    {
      id: "c1",
      title: "Acme · Q3",
      fact: "Q3 2026: Acme bought 120 seats of the Pro plan for $48,000.",
      sourceItemId: "item-acme",
      sourceTitle: "Acme account",
      t: 8,
    },
  ],
  answers: [
    {
      question: "what's Acme's churn rate?",
      text: "I don't have that in the library or in this conversation.",
      known: false,
      sourceItemIds: [],
      t: 63,
    },
  ],
};

const __trees: renderer.ReactTestRenderer[] = [];
function render() {
  let root: renderer.ReactTestRenderer;
  act(() => {
    root = renderer.create(<LiveCoachScreen />);
  });
  __trees.push(root!);
  return root!;
}
const flush = () => act(async () => { await Promise.resolve(); });
const has = (root: renderer.ReactTestRenderer, testID: string) =>
  root.root.findAllByProps({ testID }).length > 0;

afterEach(async () => {
  await act(async () => {});
  act(() => {
    for (const t of __trees.splice(0)) {
      try {
        t.unmount();
      } catch {
        // already unmounted
      }
    }
  });
});

beforeEach(() => {
  useDevModeStore.setState({ devMode: true });
  mockLibrarySaved = [];
  mockUseAudioStream.mockReturnValue({ ...baseHook });
});

const COACHING_CHROME = [
  "coach-options-toggle",
  "coach-options-sliders",
  "self-speaker-chip",
  "live-mode-row",
  "idle-explainer",
  "mood-check-before",
  "suggestions-list",
  "session-strip",
  "live-transcript",
];

describe("LiveCoachScreen — Room mode", () => {
  it("is offered in the picker, off unless chosen", () => {
    const room = LIVE_MODE_OPTIONS.find((o) => o.mode === "room");
    expect(room?.label).toBe("Room");
    expect(room?.hint).toMatch(/MindShift/);
    expect(room?.hint).toMatch(/No personal coaching/);
    expect(LIVE_MODE_OPTIONS[0].mode).toBe("earpiece");
  });

  it("idle: explainer with the consent note, the library picker, no coaching UI", async () => {
    const root = render();
    await flush();
    expect(has(root, "room-explainer")).toBe(true);
    expect(JSON.stringify(root.toJSON())).toContain(ROOM_CONSENT_NOTE);
    expect(has(root, "room-consent-banner")).toBe(false);
    expect(has(root, "room-no-library")).toBe(true);
    expect(has(root, "library-picker")).toBe(true);
    for (const id of COACHING_CHROME) expect(has(root, id)).toBe(false);
    const json = JSON.stringify(root.toJSON());
    expect(json).toContain("Start Room Assistant");
    expect(json).not.toContain("Empathy");
  });

  it("live: consent banner, the answer and the card in the big layout, Stop", async () => {
    mockLibrarySaved = ["item-acme"];
    mockUseAudioStream.mockReturnValue({
      ...baseHook,
      sessionActive: true,
      isRecording: true,
      connectionStatus: "live",
      room: LIVE_ROOM,
      transcript: [{ speaker: "Speaker B", text: "Acme closed their Q3 order.", timestamp: 1 }],
    });
    const root = render();
    await flush();
    const json = JSON.stringify(root.toJSON());
    expect(has(root, "room-consent-banner")).toBe(true);
    expect(json).toContain(ROOM_BANNER_TITLE);
    expect(json).toContain("Room — listening");
    expect(json).toContain("Q3 2026: Acme bought 120 seats of the Pro plan for $48,000.");
    expect(json).toContain("Acme account");
    expect(has(root, "room-answer-unknown")).toBe(true);
    expect(has(root, "room-no-library")).toBe(false);
    for (const id of COACHING_CHROME) expect(has(root, id)).toBe(false);
    // Transcript is optional: hidden until asked for.
    expect(has(root, "room-transcript")).toBe(false);
    act(() => root.root.findByProps({ testID: "room-transcript-toggle" }).props.onPress());
    expect(JSON.stringify(root.toJSON())).toContain("Acme closed their Q3 order.");
    // Stop is the fixed footer button.
    const stop = root.root.findByProps({ testID: "mic-toggle" });
    expect(JSON.stringify(root.toJSON())).toContain("Stop Room Assistant");
    await act(async () => {
      await stop.props.onPress();
    });
    expect(baseHook.stopSession).toHaveBeenCalled();
  });

  it("the speak-answers switch drives the hook's speech gate", async () => {
    const root = render();
    await flush();
    baseHook.setSpeechEnabled.mockClear();
    act(() => root.root.findByProps({ testID: "room-speak-switch" }).props.onValueChange(false));
    expect(baseHook.setSpeechEnabled).toHaveBeenLastCalledWith(false);
  });
});
