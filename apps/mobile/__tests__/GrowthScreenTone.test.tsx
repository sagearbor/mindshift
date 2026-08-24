/** GrowthScreen's Track 2 "How you sound" section: per-day self-tone rows
 *  and the cross-session per-person rows, driven by the /growth fields
 *  (`self_tone` on points, `people` on the result). Hidden entirely when no
 *  live session carried tone. */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import GrowthScreen from "../src/screens/GrowthScreen";
import { catchUpVoice, getGrowth, getVoiceProfile } from "../src/api/client";
import type { GrowthPoint, GrowthResult } from "../src/api/client";
import { dateKeyOfIso } from "../src/screens/dayTimeline";

jest.mock("../src/api/client", () => ({
  getGrowth: jest.fn(),
  getVoiceProfile: jest.fn(),
  catchUpVoice: jest.fn(),
}));
const mockGetGrowth = getGrowth as jest.Mock;
const mockGetVoiceProfile = getVoiceProfile as jest.Mock;
const mockCatchUpVoice = catchUpVoice as jest.Mock;

/** HOST nodes only (a component that renders null still has a fiber with
 *  its testID prop — matching it would make "renders nothing" untestable). */
function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll(
    (n) => typeof n.type === "string" && n.props?.testID === id,
  );
  return found.length > 0 ? found[0] : null;
}

/** Fire a testID'd pressable's onPress — the handler lives on the composite
 *  (TouchableOpacity) node, not the host View queryId returns. */
function press(comp: renderer.ReactTestRenderer, id: string): void {
  const found = comp.root.findAll(
    (n) => n.props?.testID === id && typeof n.props?.onPress === "function",
  );
  if (found.length === 0) throw new Error(`no pressable with testID ${id}`);
  found[0].props.onPress();
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

function pt(overrides: Partial<GrowthPoint> = {}): GrowthPoint {
  return {
    recording_id: "r1",
    timestamp: "2026-08-24T18:05:00+00:00",
    title: "Live session · earpiece",
    my_score: 64,
    partner_names: ["Mom"],
    ...overrides,
  };
}

function result(overrides: Partial<GrowthResult> = {}): GrowthResult {
  return { points: [], total_recordings: 0, identified_recordings: 0, people: [], ...overrides };
}

async function render() {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(
      <GrowthScreen onOpenRecording={() => {}} onOpenRecordings={() => {}} />,
    );
  });
  await act(async () => {});
  return comp;
}

beforeEach(() => {
  mockGetGrowth.mockReset();
  mockGetVoiceProfile.mockReset();
  mockCatchUpVoice.mockReset();
  mockGetVoiceProfile.mockResolvedValue({
    available: true, storage_enabled: true, enrolled: false, enroll_count: 0,
  });
});

describe("GrowthScreen — How you sound", () => {
  it("hides the section when no point carries self tone (uploads only)", async () => {
    mockGetGrowth.mockResolvedValueOnce(
      result({ points: [pt({ self_tone: null }), pt({ recording_id: "r2" })],
               total_recordings: 2, identified_recordings: 2 }),
    );
    const comp = await render();
    expect(queryId(comp, "growth-chart")).toBeTruthy();
    expect(queryId(comp, "growth-tone-section")).toBeNull();
    act(() => comp.unmount());
  });

  it("renders per-day rows and per-person rows from live sessions", async () => {
    const day1 = "2026-08-24T18:05:00";
    const day2 = "2026-08-25T09:00:00";
    mockGetGrowth.mockResolvedValueOnce(result({
      points: [
        pt({
          recording_id: "r1", timestamp: day1, source: "live", mode: "earpiece",
          self_tone: {
            scored_turns: 3, labels: { warm: 1, frustrated: 1, defensive: 1 },
            mean: { warmth: 50 }, escalation_count: 2, people: [],
          },
        }),
        pt({
          recording_id: "r2", timestamp: day2, source: "live", mode: "speaker",
          partner_names: [],
          self_tone: {
            scored_turns: 2, labels: { warm: 2 }, mean: { warmth: 80 },
            escalation_count: 0, people: [],
          },
        }),
      ],
      total_recordings: 2,
      identified_recordings: 2,
      people: [
        { person_id: "p-mom", display_name: "Mom", sessions: 1, scored_turns: 3,
          labels: { warm: 1, frustrated: 1, defensive: 1 }, escalation_count: 2 },
      ],
    }));
    const comp = await render();

    expect(queryId(comp, "growth-tone-section")).toBeTruthy();
    const row1 = queryId(comp, `growth-tone-day-${dateKeyOfIso(day1)}`);
    const row2 = queryId(comp, `growth-tone-day-${dateKeyOfIso(day2)}`);
    expect(row1).toBeTruthy();
    expect(row2).toBeTruthy();
    const text1 = textOf(row1!);
    expect(text1).toContain("2 escalations");
    expect(text1).toContain("defensive ×1");
    const text2 = textOf(row2!);
    expect(text2).toContain("mostly warm · no escalations");

    const mom = queryId(comp, "growth-tone-person-p-mom");
    expect(mom).toBeTruthy();
    const momText = textOf(mom!);
    expect(momText).toContain("with Mom");
    expect(momText).toContain("mostly defensive");
    expect(momText).toContain("1 session");
    act(() => comp.unmount());
  });

  it("narrows the day rows with the partner filter", async () => {
    mockGetGrowth.mockResolvedValueOnce(result({
      points: [
        pt({
          recording_id: "r1", timestamp: "2026-08-24T18:05:00", partner_names: ["Mom"],
          self_tone: { scored_turns: 1, labels: { warm: 1 }, mean: {}, escalation_count: 0, people: [] },
        }),
        pt({
          recording_id: "r2", timestamp: "2026-08-25T18:05:00", partner_names: ["Asher"],
          self_tone: { scored_turns: 1, labels: { frustrated: 1 }, mean: {}, escalation_count: 1, people: [] },
        }),
      ],
      total_recordings: 2,
      identified_recordings: 2,
    }));
    const comp = await render();
    expect(queryId(comp, `growth-tone-day-${dateKeyOfIso("2026-08-24T18:05:00")}`)).toBeTruthy();
    expect(queryId(comp, `growth-tone-day-${dateKeyOfIso("2026-08-25T18:05:00")}`)).toBeTruthy();

    await act(async () => press(comp, "growth-filter-Asher"));
    expect(queryId(comp, `growth-tone-day-${dateKeyOfIso("2026-08-24T18:05:00")}`)).toBeNull();
    expect(queryId(comp, `growth-tone-day-${dateKeyOfIso("2026-08-25T18:05:00")}`)).toBeTruthy();
    act(() => comp.unmount());
  });
});
