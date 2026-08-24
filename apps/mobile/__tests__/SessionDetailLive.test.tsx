/** The therapist views (TherapistDashboard + SessionDetail) rendering a
 *  Track 2 LIVE session as GET /sessions projects it: the patient label as
 *  the group, mode badge, partial tone scores (no fabricated zeros), the
 *  patient's tone summary card, per-turn tone chips, and the "could have
 *  said" reflection under the patient's own turns. */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import SessionDetail from "../src/screens/SessionDetail";
import TherapistDashboard from "../src/screens/TherapistDashboard";
import { useDashboardStore, type SavedSession } from "../src/store/dashboardStore";
import { listDashboardSessions } from "../src/api/client";

jest.mock("../src/api/client", () => ({
  listDashboardSessions: jest.fn(),
}));
const mockListSessions = listDashboardSessions as jest.Mock;

// Therapist-side reads (patient list, private notes) — deterministic here;
// TherapistDashboardPatients / SessionDetailTherapist cover them.
jest.mock("../src/api/therapist", () => ({
  listPatients: jest.fn(() => Promise.resolve([])),
  acceptPatient: jest.fn(),
  declinePatient: jest.fn(),
  getSessionNote: jest.fn(() => Promise.resolve({ episode_id: "e1", text: "", updated_at: null })),
  putSessionNote: jest.fn(),
}));

/** HOST nodes only (a component that renders null still has a fiber with
 *  its testID prop — matching it would make "renders nothing" untestable). */
function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll(
    (n) => typeof n.type === "string" && n.props?.testID === id,
  );
  return found.length > 0 ? found[0] : null;
}

/** All rendered text under a node, joined — RN splits a Text's children
 *  into string fragments, so a substring check needs them joined. */
function textOf(node: ReactTestInstance): string {
  return node
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");
}

const liveSession: SavedSession = {
  id: "e1",
  recordingId: "e1",
  date: "2026-08-24T18:05:00+00:00",
  role: "patient@example.com",
  shared: true,
  title: "Live session · earpiece",
  source: "live",
  mode: "earpiece",
  avgPleasantness: 60,
  analysisStatus: "full",
  turns: [
    { speaker: "You", text: "Hey Mom, I got your message.", toneScores: { pleasantness: 85, warmth: 80 },
      isSelf: true, toneLabel: "warm", escalated: false, withPerson: "Speaker B" },
    { speaker: "Mom", text: "You never call back.", toneScores: { pleasantness: 75 }, isSelf: false },
    { speaker: "You", text: "I was working, I told you that.", toneScores: { pleasantness: 65, warmth: 20 },
      isSelf: true, toneLabel: "frustrated", escalated: true, withPerson: "Speaker B" },
  ],
  toneSummary: {
    self_speaker: "Speaker A",
    self: { turns: 2, scored_turns: 2, labels: { warm: 1, frustrated: 1 },
            mean: { warmth: 50 }, escalation_turns: [2], escalation_count: 1 },
    audio: null,
    audio_tone_surfaced: false,
    people: [
      { speaker: "Speaker B", person_id: "p-mom", display_name: "Mom", their_turns: 1,
        self_turns: 2, turns: 2, scored_turns: 2, labels: { warm: 1, frustrated: 1 },
        mean: {}, escalation_turns: [2], escalation_count: 1 },
    ],
  },
  couldHaveSaid: [
    { turn_index: 2, could_have_said: "I hear you. Work swallowed me — I'm sorry.",
      why: "Owns it without defending.", tone_read: "defensive" },
  ],
};

// A live session BEFORE its batch analysis: no pleasantness anywhere.
const pendingSession: SavedSession = {
  ...liveSession,
  id: "e2",
  role: "You",
  shared: false,
  avgPleasantness: null,
  analysisStatus: "lite",
  couldHaveSaid: null,
  turns: liveSession.turns.map((t) => ({ ...t, toneScores: t.isSelf ? { warmth: 50 } : {} })),
};

beforeEach(() => {
  mockListSessions.mockReset();
  mockListSessions.mockImplementation(() => new Promise(() => {}));
  act(() => {
    useDashboardStore.setState({
      sessions: [liveSession, pendingSession],
      selectedSessionId: null,
      roleFilter: null,
      loading: false,
    });
  });
});

describe("TherapistDashboard — live sessions", () => {
  it("groups by patient, badges live sessions, and never fabricates a score", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<TherapistDashboard onSelectSession={jest.fn()} />);
    });
    const tree = textOf(comp.root);
    expect(tree).toContain("patient@example.com");
    expect(queryId(comp, "filter-patient@example.com")).toBeTruthy();
    expect(queryId(comp, "filter-You")).toBeTruthy();
    const badge = queryId(comp, "session-e1-live");
    expect(textOf(badge!)).toContain("Live · Earpiece · Live session · earpiece");
    expect(tree).toContain("3 turns · mostly frustrated · 1 escalation");
    // The pending session shows "—", not 0.
    const card2 = queryId(comp, "session-e2");
    expect(textOf(card2!)).toContain("—");
    act(() => comp.unmount());
  });
});

describe("SessionDetail — live session", () => {
  it("shows the mode, the tone card, per-turn chips and the reflection", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<SessionDetail sessionId="e1" onBack={jest.fn()} />);
    });
    const meta = queryId(comp, "session-live-meta");
    expect(textOf(meta!)).toContain("Live session · Earpiece");
    expect(queryId(comp, "session-tone-summary")).toBeTruthy();
    expect(queryId(comp, "session-tone-summary-chip-frustrated")).toBeTruthy();
    expect(queryId(comp, "session-tone-summary-person-Speaker B")).toBeTruthy();
    // Per-turn tone chip with the escalation marker on the frustrated turn.
    expect(textOf(queryId(comp, "turn-0-tone")!)).toContain("warm");
    expect(textOf(queryId(comp, "turn-2-tone")!)).toContain("frustrated ↑");
    expect(queryId(comp, "turn-1-tone")).toBeNull();
    // The reflection sits under turn 2 only.
    expect(queryId(comp, "turn-2-could-have-said-2")).toBeTruthy();
    expect(queryId(comp, "turn-0-could-have-said-0")).toBeNull();
    const tree = textOf(comp.root);
    expect(tree).toContain("I hear you. Work swallowed me — I'm sorry.");
    expect(tree).toContain("Owns it without defending.");
    // Averages only over the dimensions that were measured — warmth from the
    // two self turns (80, 20 → 50), pleasantness over all three (75).
    expect(tree).toContain("50warmth");
    expect(tree).toContain("75pleasantness");
    expect(tree).not.toContain("Empathy:");
    act(() => comp.unmount());
  });

  it("a pending live session has no timeline, no pleasantness, and says so", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<SessionDetail sessionId="e2" onBack={jest.fn()} />);
    });
    const tree = textOf(comp.root);
    expect(textOf(queryId(comp, "session-live-meta")!)).toContain(
      "heat analysis pending",
    );
    expect(tree).not.toContain("Tone Timeline");
    expect(tree).not.toContain("pleasantness");
    // The warmth average is still honest (50 over the two self turns).
    expect(tree).toContain("50warmth");
    act(() => comp.unmount());
  });
});
