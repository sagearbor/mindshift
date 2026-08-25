/**
 * SessionDetail renders the server's PRD §6 scoreboard for a stored live
 * session (patient's own view and the therapist's shared view), naming
 * people by the session's CURRENT labels; a session without one shows no
 * board.
 */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import SessionDetail, { scoreboardOf } from "../src/screens/SessionDetail";
import { useDashboardStore, type SavedSession } from "../src/store/dashboardStore";

jest.mock("../src/api/client", () => ({
  listDashboardSessions: jest.fn(),
  listVoicePeople: jest.fn(() => Promise.resolve({ available: true, storage_enabled: true, people: [] })),
}));
jest.mock("../src/api/therapist", () => ({
  getSessionNote: jest.fn(() => Promise.resolve({ episode_id: "e1", text: "", updated_at: null })),
  putSessionNote: jest.fn(),
}));

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => typeof n.type === "string" && n.props?.testID === id);
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

const session: SavedSession = {
  id: "e1",
  recordingId: "e1",
  date: "2026-08-24T18:05:00+00:00",
  role: "You",
  source: "live",
  mode: "speaker",
  avgPleasantness: null,
  turns: [
    { speaker: "You", speakerId: "Speaker A", text: "hi", toneScores: { warmth: 80, calmness: 90 } },
    { speaker: "Mom", speakerId: "Speaker B", text: "you never call", toneScores: { engagement: 100 } },
  ],
  speakers: [
    { id: "Speaker A", display: "You", labelSource: "enrolled" },
    { id: "Speaker B", display: "Mom", labelSource: "manual-person", personId: "mom" },
  ],
  scoreboard: {
    people: [
      { speaker: "Speaker A", display: "You", current: 84, series: [84], scored_turns: 1 },
      { speaker: "Speaker B", display: "Speaker B", current: null, series: [], scored_turns: 0 },
    ],
    lead: null,
    turns: [84, null],
  },
};

async function mount(s: SavedSession) {
  useDashboardStore.setState({ sessions: [s], selectedSessionId: s.id, roleFilter: null, loading: false });
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<SessionDetail sessionId={s.id} onBack={() => {}} />);
  });
  return comp;
}

describe("SessionDetail scoreboard", () => {
  it("renders the board with the session's current names and the PRD dimensions in the averages", async () => {
    const comp = await mount(session);
    const board = queryId(comp, "session-scoreboard");
    expect(board).not.toBeNull();
    expect(textOf(queryId(comp, "session-scoreboard-score-Speaker A"))).toBe("84");
    expect(textOf(queryId(comp, "session-scoreboard-score-Speaker B"))).toBe("—");
    // The speakers list's label wins over the server's board display.
    expect(textOf(queryId(comp, "session-scoreboard-row-Speaker B"))).toContain("Mom");
    expect(textOf(queryId(comp, "session-scoreboard-lead"))).toBe("You is warming up the room.");
    expect(textOf(comp.root.findAll((n) => typeof n.type === "string" && n.props?.testID === "session-detail")[0])).toContain("calmness");
  });

  it("titles the therapist's copy as the patient's and shows nothing without a board", async () => {
    const shared = await mount({ ...session, id: "e2", shared: true, role: "patient@example.com" });
    expect(textOf(queryId(shared, "session-scoreboard"))).toContain("patient's session");
    const none = await mount({ ...session, id: "e3", scoreboard: null });
    expect(queryId(none, "session-scoreboard")).toBeNull();
  });

  it("scoreboardOf is null for an empty/absent board and tolerates odd values", () => {
    expect(scoreboardOf({ ...session, scoreboard: undefined })).toBeNull();
    expect(scoreboardOf({ ...session, scoreboard: { people: [], lead: null, turns: [] } })).toBeNull();
    const odd = scoreboardOf({
      ...session,
      speakers: [],
      scoreboard: {
        people: [{ speaker: "X", display: "", current: 12, series: [1, null as unknown as number, 3], scored_turns: 2 }],
        lead: { speaker: "X", display: "X", margin: 5 },
        turns: [],
      },
    })!;
    expect(odd.board.people[0]).toEqual({ speaker: "X", current: 12, series: [1, 3], scoredTurns: 2 });
    expect(odd.board.lead).toEqual({ speaker: "X", margin: 5 });
    expect(odd.nameOf("X")).toBe("X");
  });
});
