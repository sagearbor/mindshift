import React from "react";
import renderer, { act } from "react-test-renderer";
import TherapistSessionPanel, {
  escalationTurns,
  namedPeople,
} from "../src/components/TherapistSessionPanel";
import type { SavedSession } from "../src/store/dashboardStore";

jest.mock("../src/api/therapist", () => ({
  getSessionNote: jest.fn(),
  putSessionNote: jest.fn(),
}));

const session: SavedSession = {
  id: "e1",
  recordingId: "e1",
  date: "2026-08-24T18:05:00+00:00",
  role: "sage@example.com",
  shared: true,
  source: "live",
  mode: "speaker",
  avgPleasantness: 60,
  turns: [
    { speaker: "You", text: "Hey Mom.", toneScores: { pleasantness: 85 }, isSelf: true, escalated: false },
    { speaker: "Mom", text: "You never call.", toneScores: {}, isSelf: false },
    { speaker: "You", text: "I was working!", toneScores: { pleasantness: 40 }, isSelf: true, escalated: true },
    { speaker: "You", text: "Fine.", toneScores: {}, isSelf: true, escalated: true },
  ],
  toneSummary: {
    self_speaker: "Speaker A",
    self: { turns: 3, scored_turns: 2, labels: {}, mean: {}, escalation_turns: [2, 3], escalation_count: 2 },
    audio: null,
    audio_tone_surfaced: false,
    people: [
      { speaker: "Speaker B", person_id: "mom", display_name: "Mom", their_turns: 1, self_turns: 3,
        turns: 3, scored_turns: 2, labels: {}, mean: {}, escalation_turns: [2, 3], escalation_count: 2 },
    ],
  },
};

const flush = () => act(async () => { await Promise.resolve(); });
/** All rendered text, joined — RN splits interpolated strings into fragments. */
const text = (root: renderer.ReactTestRenderer) =>
  root.root
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");

describe("TherapistSessionPanel", () => {
  it("pure helpers: escalation turn indexes and named people", () => {
    expect(escalationTurns(session)).toEqual([2, 3]);
    expect(namedPeople(session)).toEqual([{ name: "Mom", theirTurns: 1, selfTurns: 3, escalations: 2 }]);
    expect(namedPeople({ ...session, toneSummary: null })).toEqual([]);
  });

  it("renders escalation markers, the named people, and a private note that saves", async () => {
    const loadNote = jest.fn().mockResolvedValue({ text: "Earlier note." });
    const saveNote = jest.fn().mockResolvedValue({ text: "Earlier note. Defensive about work." });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(
        <TherapistSessionPanel session={session} loadNote={loadNote} saveNote={saveNote} />,
      );
    });
    await flush();
    expect(loadNote).toHaveBeenCalledWith("e1");
    const t = text(root!);
    expect(t).toContain("2 escalations on the patient's turns (turns 3, 4)");
    expect(t).toContain("Mom: 1 turn · patient spoke to them 3× · 2 escalations");
    expect(root!.root.findAllByProps({ testID: "escalation-marker-2" })).not.toHaveLength(0);
    const input = root!.root.findByProps({ testID: "therapist-note-input" });
    expect(input.props.value).toBe("Earlier note.");
    expect(root!.root.findByProps({ testID: "therapist-note-save" }).props.disabled).toBe(true);

    act(() => {
      input.props.onChangeText("Earlier note. Defensive about work.");
    });
    expect(JSON.stringify(root!.root.findByProps({ testID: "therapist-note-status" }).props.children)).toContain("Unsaved changes");
    await act(async () => {
      await root!.root.findByProps({ testID: "therapist-note-save" }).props.onPress();
    });
    expect(saveNote).toHaveBeenCalledWith("e1", "Earlier note. Defensive about work.");
    expect(JSON.stringify(root!.root.findByProps({ testID: "therapist-note-status" }).props.children)).toContain("Saved");
  });

  it("a save failure is stated, never pretended", async () => {
    const loadNote = jest.fn().mockResolvedValue({ text: "" });
    const saveNote = jest.fn().mockRejectedValue(new Error("503"));
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(
        <TherapistSessionPanel session={{ ...session, turns: [] }} loadNote={loadNote} saveNote={saveNote} />,
      );
    });
    await flush();
    expect(text(root!)).toContain("No escalations flagged");
    act(() => {
      root!.root.findByProps({ testID: "therapist-note-input" }).props.onChangeText("x");
    });
    await act(async () => {
      await root!.root.findByProps({ testID: "therapist-note-save" }).props.onPress();
    });
    expect(JSON.stringify(root!.root.findByProps({ testID: "therapist-note-status" }).props.children)).toContain("Couldn’t save");
  });
});
