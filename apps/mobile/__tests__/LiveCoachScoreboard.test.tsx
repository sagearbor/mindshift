import React from "react";
import renderer, { act, type ReactTestInstance } from "react-test-renderer";

/**
 * LiveCoachScreen — the opt-in pleasantness scoreboard (persisted per
 * account) and mid-call naming (speaker chips / transcript labels →
 * "Who is this?" → the hook's labelSpeaker).
 */
const mockUseAudioStream = jest.fn();
jest.mock("../src/hooks/useAudioStream", () => ({
  useAudioStream: () => mockUseAudioStream(),
}));
const mockListVoicePeople = jest.fn();
jest.mock("../src/api/liveSessions", () => ({
  listVoicePeople: () => mockListVoicePeople(),
}));
const mockGetTherapistLink = jest.fn();
jest.mock("../src/api/therapist", () => ({
  getTherapistLink: () => mockGetTherapistLink(),
}));
jest.mock("../src/live/modePrefs", () => ({
  loadLiveMode: jest.fn(() => Promise.resolve("earpiece")),
  saveLiveMode: jest.fn(() => Promise.resolve()),
}));
const mockLoadScoreboard = jest.fn();
const mockSaveScoreboard = jest.fn();
jest.mock("../src/live/scoreboardPrefs", () => ({
  loadScoreboardVisible: (uid: string | null) => mockLoadScoreboard(uid),
  saveScoreboardVisible: (uid: string | null, on: boolean) => mockSaveScoreboard(uid, on),
}));
const mockClientPeople = jest.fn();
jest.mock("../src/api/client", () => ({
  postShare: jest.fn(),
  listVoicePeople: () => mockClientPeople(),
  patchSpeakerLabels: jest.fn(),
  enrollPersonFromRecording: jest.fn(),
}));

import LiveCoachScreen from "../src/screens/LiveCoachScreen";
import type { Scoreboard } from "../src/live/pleasantness";

const board: Scoreboard = {
  people: [
    { speaker: "Speaker A", current: 74, series: [70, 78], scoredTurns: 2 },
    { speaker: "Speaker B", current: 61, series: [61], scoredTurns: 1 },
  ],
  lead: { speaker: "Speaker A", margin: 13 },
};

const mockLabelSpeaker = jest.fn();

function hookState(over: Record<string, unknown> = {}) {
  return {
    isRecording: true,
    sessionActive: true,
    transcript: [
      { speaker: "Speaker A", speakerId: "Speaker A", text: "hi", timestamp: 1 },
      { speaker: "Speaker B", speakerId: "Speaker B", text: "you never call", timestamp: 2 },
    ],
    suggestions: [],
    speakerLabel: "Speaker B",
    selfSpeaker: "Speaker A",
    setSelfSpeaker: jest.fn(),
    connectionStatus: "live",
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
    sessionMode: "earpiece",
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
    speakerNames: {},
    displayNameOf: (s: string) => s,
    labelSpeaker: mockLabelSpeaker,
    scoreboard: board,
    ...over,
  };
}

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => typeof n.type === "string" && n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}
function queryAny(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}
function textOf(node: ReactTestInstance | null): string {
  if (!node) return "";
  return node
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");
}
const flush = () => act(async () => {
  await Promise.resolve();
  await Promise.resolve();
});

async function mount() {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<LiveCoachScreen />);
  });
  await flush();
  return comp;
}

beforeEach(() => {
  mockUseAudioStream.mockReturnValue(hookState());
  mockListVoicePeople.mockReset().mockResolvedValue({ people: [], error: null });
  mockGetTherapistLink.mockReset().mockResolvedValue({ linked: false });
  mockLoadScoreboard.mockReset().mockResolvedValue(false);
  mockSaveScoreboard.mockReset().mockResolvedValue(undefined);
  mockClientPeople.mockReset().mockResolvedValue({
    available: true,
    storage_enabled: true,
    people: [
      { available: true, storage_enabled: true, enrolled: true, enroll_count: 2, person_id: "mom", display_name: "Mom", is_self: false, samples: [] },
    ],
  });
  mockLabelSpeaker.mockReset().mockResolvedValue({ text: "Mom is labeled for the rest of this call.", enrolled: false, seconds: 2 });
});

describe("LiveCoachScreen scoreboard", () => {
  it("is off by default and shows nothing until toggled; the toggle is remembered per account", async () => {
    const comp = await mount();
    expect(queryId(comp, "scoreboard")).toBeNull();
    const sw = queryAny(comp, "scoreboard-switch")!;
    expect(sw.props.value).toBe(false);
    await act(async () => {
      sw.props.onValueChange(true);
    });
    expect(mockSaveScoreboard).toHaveBeenCalledWith(null, true);
    const panel = queryId(comp, "scoreboard");
    expect(panel).not.toBeNull();
    expect(textOf(queryId(comp, "scoreboard-score-Speaker A"))).toBe("74");
    expect(textOf(queryId(comp, "scoreboard-score-Speaker B"))).toBe("61");
    expect(textOf(queryId(comp, "scoreboard-lead"))).toBe("Speaker A +13 — leading with kindness.");
  });

  it("opens with the remembered choice and names people on the board", async () => {
    mockLoadScoreboard.mockResolvedValue(true);
    mockUseAudioStream.mockReturnValue(
      hookState({
        speakerNames: { "Speaker B": { personId: "mom", displayName: "Mom", isSelf: false } },
        displayNameOf: (s: string) => (s === "Speaker B" ? "Mom" : s),
      }),
    );
    const comp = await mount();
    expect(queryAny(comp, "scoreboard-switch")!.props.value).toBe(true);
    expect(textOf(queryId(comp, "scoreboard-row-Speaker B"))).toContain("Mom");
  });

  it("says why there are no scores on the legacy (server) path", async () => {
    mockLoadScoreboard.mockResolvedValue(true);
    mockUseAudioStream.mockReturnValue(hookState({ liveMode: false, scoreboard: null }));
    const comp = await mount();
    expect(textOf(queryId(comp, "scoreboard-empty"))).toMatch(/need on-device coaching/);
  });
});

describe("LiveCoachScreen mid-call naming", () => {
  it("shows a chip per voice (named ones without the hint) and opens 'Who is this?' from a chip", async () => {
    const comp = await mount();
    const chips = queryId(comp, "speaker-chips")!;
    expect(textOf(chips)).toContain("Speaker A · who?");
    expect(textOf(chips)).toContain("Speaker B · who?");
    await act(async () => {
      queryAny(comp, "speaker-chip-Speaker B")!.props.onPress();
    });
    await flush();
    const sheet = queryId(comp, "who-sheet");
    expect(sheet).not.toBeNull();
    expect(textOf(queryId(comp, "who-subtitle"))).toBe("Currently “Speaker B”");
    expect(mockClientPeople).toHaveBeenCalled();
    // Picking the enrolled person hands the choice to the hook, live.
    await act(async () => {
      queryAny(comp, "who-person-mom")!.props.onPress();
    });
    await flush();
    expect(mockLabelSpeaker).toHaveBeenCalledWith("Speaker B", {
      personId: "mom",
      displayName: "Mom",
      isSelf: false,
      isNew: false,
    });
    expect(textOf(queryId(comp, "who-done-text"))).toBe("Mom is labeled for the rest of this call.");
  });

  it("a named voice's chip drops the hint; transcript labels open the sheet too", async () => {
    mockUseAudioStream.mockReturnValue(
      hookState({
        transcript: [
          { speaker: "Speaker A", speakerId: "Speaker A", text: "hi", timestamp: 1 },
          { speaker: "Mom", speakerId: "Speaker B", text: "you never call", timestamp: 2 },
        ],
        speakerNames: { "Speaker B": { personId: "mom", displayName: "Mom", isSelf: false } },
        displayNameOf: (s: string) => (s === "Speaker B" ? "Mom" : s),
      }),
    );
    const comp = await mount();
    const chips = queryId(comp, "speaker-chips")!;
    expect(textOf(chips)).toContain("Speaker A · who?");
    expect(textOf(chips)).toContain("Mom");
    expect(textOf(chips)).not.toContain("Mom · who?");
    await act(async () => {
      queryAny(comp, "live-transcript-speaker-0")!.props.onPress();
    });
    await flush();
    expect(textOf(queryId(comp, "who-subtitle"))).toBe("Currently “Speaker A”");
  });

  it("the therapist view's column headers open the sheet", async () => {
    mockUseAudioStream.mockReturnValue(hookState({ sessionMode: "therapist" }));
    const comp = await mount();
    await act(async () => {
      queryAny(comp, "therapist-column-right-tap")!.props.onPress();
    });
    await flush();
    expect(textOf(queryId(comp, "who-subtitle"))).toBe("Currently “Speaker B”");
  });

  it("'New person…' asks the hook to create + learn them", async () => {
    const comp = await mount();
    await act(async () => {
      queryAny(comp, "speaker-chip-Speaker B")!.props.onPress();
    });
    await flush();
    await act(async () => {
      queryAny(comp, "who-new-person")!.props.onPress();
    });
    await act(async () => {
      queryAny(comp, "who-name-input")!.props.onChangeText("Aunt Béa");
    });
    await act(async () => {
      queryAny(comp, "who-save-name")!.props.onPress();
    });
    await flush();
    expect(mockLabelSpeaker).toHaveBeenCalledWith("Speaker B", {
      personId: "aunt-bea",
      displayName: "Aunt Béa",
      isSelf: false,
      isNew: true,
    });
  });
});
