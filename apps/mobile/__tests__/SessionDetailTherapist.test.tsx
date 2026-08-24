/** SessionDetail as the THERAPIST sees a patient's shared session: the
 *  therapist panel (escalation markers, named people, private notes) renders
 *  for a shared session and not for the viewer's own. */
import React from "react";
import renderer, { act } from "react-test-renderer";
import SessionDetail from "../src/screens/SessionDetail";
import { useDashboardStore, type SavedSession } from "../src/store/dashboardStore";
import { getSessionNote, putSessionNote } from "../src/api/therapist";

jest.mock("../src/api/client", () => ({
  listDashboardSessions: jest.fn(() => new Promise(() => {})),
}));
jest.mock("../src/api/therapist", () => ({
  getSessionNote: jest.fn(),
  putSessionNote: jest.fn(),
}));
const mockGetNote = getSessionNote as jest.Mock;
const mockPutNote = putSessionNote as jest.Mock;

const shared: SavedSession = {
  id: "e1",
  recordingId: "e1",
  date: "2026-08-24T18:05:00+00:00",
  role: "sage@example.com",
  shared: true,
  source: "live",
  mode: "speaker",
  avgPleasantness: 60,
  analysisStatus: "full",
  turns: [
    { speaker: "You", text: "Hey Mom.", toneScores: { pleasantness: 85 }, isSelf: true, escalated: false },
    { speaker: "Mom", text: "You never call.", toneScores: { pleasantness: 70 }, isSelf: false },
    { speaker: "You", text: "I was working!", toneScores: { pleasantness: 40 }, isSelf: true, toneLabel: "frustrated", escalated: true },
  ],
  toneSummary: {
    self_speaker: "Speaker A",
    self: { turns: 2, scored_turns: 2, labels: { frustrated: 1 }, mean: {}, escalation_turns: [2], escalation_count: 1 },
    audio: null,
    audio_tone_surfaced: false,
    people: [
      { speaker: "Speaker B", person_id: "mom", display_name: "Mom", their_turns: 1, self_turns: 2,
        turns: 2, scored_turns: 2, labels: {}, mean: {}, escalation_turns: [2], escalation_count: 1 },
    ],
  },
  couldHaveSaid: [
    { turn_index: 2, could_have_said: "Work swallowed me — I'm sorry.", why: "Owns it.", tone_read: "defensive" },
  ],
};

const flush = () => act(async () => { await Promise.resolve(); });

beforeEach(() => {
  mockGetNote.mockReset().mockResolvedValue({ episode_id: "e1", text: "", updated_at: null });
  mockPutNote.mockReset().mockResolvedValue({ episode_id: "e1", text: "Note.", updated_at: "now" });
});

describe("SessionDetail — therapist view", () => {
  it("a shared session shows the tone timeline, escalation markers, named people, reflections and notes", async () => {
    act(() => {
      useDashboardStore.setState({ sessions: [shared], selectedSessionId: null, roleFilter: null, loading: false });
    });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<SessionDetail sessionId="e1" onBack={jest.fn()} />);
    });
    await flush();
    expect(root!.root.findByProps({ testID: "therapist-session-panel" })).toBeTruthy();
    expect(root!.root.findByProps({ testID: "escalation-markers" })).toBeTruthy();
    const t = root!.root
      .findAll((n) => typeof n.type === "string")
      .flatMap((n) => n.children)
      .filter((c): c is string => typeof c === "string")
      .join("");
    expect(t).toContain("1 escalation on the patient's turns (turn 3)");
    expect(t).toContain("Mom: 1 turn");
    expect(t).toContain("Work swallowed me");
    expect(mockGetNote).toHaveBeenCalledWith("e1");
    act(() => {
      root!.root.findByProps({ testID: "therapist-note-input" }).props.onChangeText("Note.");
    });
    await act(async () => {
      await root!.root.findByProps({ testID: "therapist-note-save" }).props.onPress();
    });
    expect(mockPutNote).toHaveBeenCalledWith("e1", "Note.");
  });

  it("the viewer's own session has no therapist panel", async () => {
    act(() => {
      useDashboardStore.setState({ sessions: [{ ...shared, id: "own", recordingId: "own", role: "You", shared: false }], selectedSessionId: null, roleFilter: null, loading: false });
    });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<SessionDetail sessionId="own" onBack={jest.fn()} />);
    });
    await flush();
    expect(root!.root.findAllByProps({ testID: "therapist-session-panel" })).toHaveLength(0);
    expect(mockGetNote).not.toHaveBeenCalled();
  });
});
