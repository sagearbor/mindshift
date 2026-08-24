import React from "react";
import renderer, { act, type ReactTestInstance } from "react-test-renderer";
import SessionDetail, { applyLabelsToSession } from "../src/screens/SessionDetail";
import { useDashboardStore, type SavedSession } from "../src/store/dashboardStore";
import { listVoicePeople, patchSpeakerLabels } from "../src/api/client";

jest.mock("../src/api/client", () => ({
  listDashboardSessions: jest.fn(),
  listVoicePeople: jest.fn(),
  patchSpeakerLabels: jest.fn(),
  enrollPersonFromRecording: jest.fn(),
}));

const mockPeople = listVoicePeople as jest.Mock;
const mockPatch = patchSpeakerLabels as jest.Mock;

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

function textOf(node: ReactTestInstance | null): string {
  if (!node) return "";
  const out: string[] = [];
  for (const n of node.findAll((n) => typeof n.type === "string")) {
    const c = n.props?.children;
    if (typeof c === "string") out.push(c);
    else if (Array.isArray(c)) out.push(...c.filter((x): x is string => typeof x === "string"));
  }
  return out.join(" ");
}

const OWN: SavedSession = {
  id: "s1",
  recordingId: "r1",
  date: "2026-08-20T10:00:00Z",
  role: "You",
  shared: false,
  hasAudio: true,
  avgPleasantness: 60,
  speakers: [
    { id: "Speaker A", display: "You", labelSource: "enrolled", personId: null },
    { id: "Speaker B", display: "Speaker B", labelSource: "generic", personId: null },
  ],
  turns: [
    { speaker: "You", speakerId: "Speaker A", labelSource: "enrolled", personId: null, text: "hi", toneScores: { pleasantness: 60 } },
    { speaker: "Speaker B", speakerId: "Speaker B", labelSource: "generic", personId: null, text: "yo", toneScores: { pleasantness: 60 } },
  ],
};

async function mount(session: SavedSession) {
  act(() => {
    useDashboardStore.setState({ sessions: [session], selectedSessionId: null, roleFilter: null, loading: false });
  });
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<SessionDetail sessionId={session.id} onBack={jest.fn()} />);
  });
  return comp;
}

beforeEach(() => {
  mockPeople.mockReset();
  mockPatch.mockReset();
  mockPeople.mockResolvedValue({
    available: true, storage_enabled: true,
    people: [{ available: true, storage_enabled: true, enrolled: true, enroll_count: 1, person_id: "mom", display_name: "Mom", is_self: false, samples: [] }],
  });
});

describe("applyLabelsToSession", () => {
  it("re-labels turns and the speakers list by raw speaker id", () => {
    const out = applyLabelsToSession(OWN, {
      "Speaker B": { display_label: "Mom", label_source: "manual-person", person_id: "mom" },
    });
    expect(out.turns[1]).toMatchObject({ speaker: "Mom", labelSource: "manual-person", personId: "mom" });
    expect(out.turns[0].speaker).toBe("You"); // untouched
    expect(out.speakers![1]).toMatchObject({ display: "Mom", personId: "mom" });
  });
});

describe("SessionDetail — Who is this?", () => {
  it("offers Who is this? on own sessions, marks enrolled speakers, and re-labels after a save", async () => {
    mockPatch.mockResolvedValue({
      id: "r1",
      manual_speaker_labels: { "Speaker B": "Mom" },
      manual_speaker_people: { "Speaker B": "mom" },
      speaker_labels: {
        "Speaker B": { display_label: "Mom", label_source: "manual-person", person_id: "mom" },
      },
    });
    const comp = await mount(OWN);
    expect(mockPeople).toHaveBeenCalled();
    // The enrolled "You" carries the badge; Speaker B invites naming.
    expect(queryId(comp, "turn-0-enrolled")).toBeTruthy();
    expect(queryId(comp, "turn-1-enrolled")).toBeNull();

    act(() => queryId(comp, "turn-1-who")!.props.onPress());
    expect(queryId(comp, "who-sheet")).toBeTruthy();
    expect(textOf(queryId(comp, "who-subtitle"))).toContain("Speaker B");
    await act(async () => {
      queryId(comp, "who-person-mom")!.props.onPress();
    });
    expect(mockPatch).toHaveBeenCalledWith("r1", { "Speaker B": "Mom" }, { "Speaker B": "mom" });
    // The store row now shows the person's name with the enrolled badge.
    const turn1 = queryId(comp, "turn-1-who");
    expect(textOf(turn1)).toContain("Mom");
    expect(queryId(comp, "turn-1-enrolled")).toBeTruthy();
    expect(useDashboardStore.getState().sessions[0].turns[1].personId).toBe("mom");
    // "Remember this voice" is offered because the server kept audio.
    expect(queryId(comp, "who-remember")).toBeTruthy();
  });

  it("does not offer naming on a shared (therapist-view) session or a legacy row without speaker ids", async () => {
    const shared = await mount({ ...OWN, id: "s2", shared: true, role: "patient@example.com" });
    expect(queryId(shared, "turn-1-who")).toBeNull();
    expect(mockPeople).not.toHaveBeenCalled();

    const legacy = await mount({
      ...OWN, id: "s3", speakers: undefined,
      turns: OWN.turns.map((t) => ({ speaker: t.speaker, text: t.text, toneScores: t.toneScores })),
    });
    expect(queryId(legacy, "turn-1-who")).toBeNull();
    expect(mockPeople).not.toHaveBeenCalled();
  });
});
