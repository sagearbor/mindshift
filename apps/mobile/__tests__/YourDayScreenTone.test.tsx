/** YourDayScreen's Track 2 additions: a live session is badged (with its
 *  coaching mode) and each episode shows the user's OWN tone chip line when
 *  the server carried one — omitted, never "neutral", otherwise. */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import YourDayScreen from "../src/screens/YourDayScreen";
import { listRecordings, getRecordingEpisodes } from "../src/api/client";
import type { Episode, RecordingSummary } from "../src/api/client";

jest.mock("../src/api/client", () => ({
  listRecordings: jest.fn(),
  getRecordingEpisodes: jest.fn(),
}));
const mockList = listRecordings as jest.Mock;
const mockEpisodes = getRecordingEpisodes as jest.Mock;

jest.setTimeout(120000);

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

function todayRec(id: string, overrides: Partial<RecordingSummary> = {}): RecordingSummary {
  const d = new Date();
  d.setHours(9, 0, 0, 0);
  return {
    id,
    created_at: d.toISOString(),
    filename: "live-session",
    title: "Call with Mom",
    media_type: "none",
    duration_seconds: 120,
    has_analysis: true,
    source_type: "live",
    mode: "earpiece",
    ...overrides,
  };
}

function ep(index: number, overrides: Partial<Episode> = {}): Episode {
  return {
    index,
    start_time: 0,
    end_time: 60,
    duration_seconds: 60,
    first_turn_index: 0,
    last_turn_index: 5,
    turn_count: 6,
    speakers: ["Speaker A", "Speaker B"],
    participants: ["You", "Mom"],
    mean_heat: null,
    peak_heat: null,
    summary: "Hey Mom, I got your message.",
    summary_source: "excerpt",
    ...overrides,
  };
}

async function renderScreen() {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<YourDayScreen onOpenReplay={() => {}} onBack={() => {}} />);
  });
  await act(async () => {});
  return comp;
}

beforeEach(() => {
  mockList.mockReset();
  mockEpisodes.mockReset();
});

describe("YourDayScreen — live sessions", () => {
  it("badges a live session and shows the self-tone chip line", async () => {
    mockList.mockResolvedValueOnce([todayRec("r1")]);
    mockEpisodes.mockResolvedValueOnce([
      ep(0, { self_tone_labels: { warm: 1, frustrated: 1, defensive: 1 }, self_escalation_count: 2 }),
    ]);
    const comp = await renderScreen();
    const tree = textOf(comp.root);
    expect(tree).toContain("Call with Mom · live · Earpiece");
    const chip = queryId(comp, "episode-tone-r1-0");
    expect(chip).toBeTruthy();
    expect(textOf(chip!)).toContain("you: defensive ×1, frustrated ×1, warm ×1 · 2 escalations");
    // No heats yet → the honest "heat unknown" copy still shows.
    expect(tree).toContain("heat unknown");
    act(() => comp.unmount());
  });

  it("omits the chip when the episode carries no self tone", async () => {
    mockList.mockResolvedValueOnce([
      todayRec("r2", { source_type: "upload", mode: null, media_type: "audio", title: "Kitchen" }),
    ]);
    mockEpisodes.mockResolvedValueOnce([ep(0, { participants: ["You", "Jordan"] })]);
    const comp = await renderScreen();
    expect(queryId(comp, "episode-tone-r2-0")).toBeNull();
    expect(textOf(comp.root)).not.toContain("· live");
    act(() => comp.unmount());
  });
});
