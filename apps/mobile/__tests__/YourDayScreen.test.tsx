import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import YourDayScreen from "../src/screens/YourDayScreen";
import { listRecordings, getRecordingEpisodes } from "../src/api/client";
import type { Episode, RecordingSummary } from "../src/api/client";
import { HEAT_UNKNOWN_COLOR, heatColor } from "../src/screens/dayTimeline";

jest.mock("../src/api/client", () => ({
  listRecordings: jest.fn(),
  getRecordingEpisodes: jest.fn(),
}));
const mockList = listRecordings as jest.Mock;
const mockEpisodes = getRecordingEpisodes as jest.Mock;

// The first render of a suite pays jest-expo's cold-start transform cost (see
// jest-setup.ts); on a loaded worker this screen's first test can exceed the
// 30s global headroom, so give this suite extra room. The tests themselves
// are fast once warm.
jest.setTimeout(120000);

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

/** A recording created at local `hour` TODAY, so day-bucketing is stable no
 *  matter when/where the suite runs. */
function todayRec(
  id: string,
  hour: number,
  overrides: Partial<RecordingSummary> = {},
): RecordingSummary {
  const d = new Date();
  d.setHours(hour, 0, 0, 0);
  return {
    id,
    created_at: d.toISOString(),
    filename: `${id}.m4a`,
    media_type: "audio",
    duration_seconds: 600,
    has_analysis: true,
    ...overrides,
  };
}

function makeEpisode(index: number, overrides: Partial<Episode> = {}): Episode {
  return {
    index,
    start_time: index * 300,
    end_time: index * 300 + 120,
    duration_seconds: 120,
    first_turn_index: index * 4,
    last_turn_index: index * 4 + 3,
    turn_count: 4,
    speakers: ["Speaker A", "Speaker B"],
    participants: ["You", "Jordan"],
    mean_heat: 30 + index * 40,
    peak_heat: 45 + index * 40,
    summary: `Episode ${index} opening line`,
    summary_source: "excerpt",
    ...overrides,
  };
}

async function renderScreen(
  props: Partial<React.ComponentProps<typeof YourDayScreen>> = {},
) {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(
      <YourDayScreen
        onOpenReplay={props.onOpenReplay ?? (() => {})}
        onBack={props.onBack ?? (() => {})}
      />,
    );
  });
  await act(async () => {});
  return comp;
}

beforeEach(() => {
  mockList.mockReset();
  mockEpisodes.mockReset();
});

describe("YourDayScreen", () => {
  it("shows the honest empty state for a day with no recordings", async () => {
    mockList.mockResolvedValueOnce([]);
    const comp = await renderScreen();

    expect(queryId(comp, "your-day-empty")).toBeTruthy();
    expect(JSON.stringify(comp.toJSON())).toContain(
      "No conversations recorded today.",
    );
    // Nothing to segment → the episodes endpoint is never hit.
    expect(mockEpisodes).not.toHaveBeenCalled();
    act(() => comp.unmount());
  });

  it("renders a multi-episode day: ribbons, participants, summaries", async () => {
    mockList.mockResolvedValueOnce([todayRec("r1", 9)]);
    const eps = [
      makeEpisode(0),
      makeEpisode(1, { mean_heat: null, peak_heat: null }),
    ];
    mockEpisodes.mockResolvedValueOnce(eps);
    const comp = await renderScreen();

    expect(queryId(comp, "your-day-timeline")).toBeTruthy();
    expect(queryId(comp, "day-recording-r1")).toBeTruthy();
    expect(queryId(comp, "episode-r1-0")).toBeTruthy();
    expect(queryId(comp, "episode-r1-1")).toBeTruthy();

    // Heat ribbon color tracks mean heat; unknown heat renders neutral gray.
    const ribbon0 = queryId(comp, "episode-heat-r1-0")!;
    const style0 = JSON.stringify(ribbon0.props.style);
    expect(style0).toContain(heatColor(30));
    const ribbon1 = queryId(comp, "episode-heat-r1-1")!;
    expect(JSON.stringify(ribbon1.props.style)).toContain(HEAT_UNKNOWN_COLOR);

    const tree = JSON.stringify(comp.toJSON());
    expect(tree).toContain("You, Jordan");
    expect(tree).toContain("Episode 0 opening line");
    expect(tree).toContain("Episode 1 opening line");
    expect(tree).toContain("peak heat 45");
    expect(tree).toContain("heat unknown");
    act(() => comp.unmount());
  });

  it("taps an episode → opens the recording's replay", async () => {
    mockList.mockResolvedValueOnce([todayRec("r1", 9)]);
    mockEpisodes.mockResolvedValueOnce([makeEpisode(0), makeEpisode(1)]);
    const onOpenReplay = jest.fn();
    const comp = await renderScreen({ onOpenReplay });

    act(() => queryId(comp, "episode-r1-1")!.props.onPress());
    expect(onOpenReplay).toHaveBeenCalledWith("r1");
    act(() => comp.unmount());
  });

  it("renders an unanalyzed recording honestly, without fetching episodes", async () => {
    mockList.mockResolvedValueOnce([
      todayRec("r2", 10, { has_analysis: false }),
    ]);
    const comp = await renderScreen();

    expect(queryId(comp, "day-recording-r2-unanalyzed")).toBeTruthy();
    expect(mockEpisodes).not.toHaveBeenCalled();
    act(() => comp.unmount());
  });

  it("steps to a past day (empty there) and disables the future arrow", async () => {
    mockList.mockResolvedValue([todayRec("r1", 9)]);
    mockEpisodes.mockResolvedValue([makeEpisode(0)]);
    const comp = await renderScreen();

    // Today: timeline is present, and stepping forward is disabled.
    expect(queryId(comp, "your-day-timeline")).toBeTruthy();
    const next = queryId(comp, "your-day-next")!;
    expect(next.props.disabled ?? next.props.accessibilityState?.disabled).toBe(
      true,
    );

    // Step back a day: no recordings there → honest past-day empty state.
    await act(async () => queryId(comp, "your-day-prev")!.props.onPress());
    await act(async () => {});
    expect(JSON.stringify(comp.toJSON())).toContain(
      "No conversations recorded on this day.",
    );
    // The date title changed and "Jump to today" appeared.
    expect(queryId(comp, "your-day-today")).toBeTruthy();

    // Jump back to today restores the timeline (from cache — list was
    // fetched exactly once).
    await act(async () => queryId(comp, "your-day-today")!.props.onPress());
    await act(async () => {});
    expect(queryId(comp, "your-day-timeline")).toBeTruthy();
    expect(mockList).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });

  it("shows an honest error state and retries", async () => {
    mockList.mockRejectedValueOnce(new Error("API error: 503"));
    const comp = await renderScreen();

    expect(queryId(comp, "your-day-error")).toBeTruthy();
    expect(JSON.stringify(comp.toJSON())).toContain(
      "Replay storage isn’t enabled yet.",
    );

    mockList.mockResolvedValueOnce([]);
    await act(async () => queryId(comp, "your-day-retry")!.props.onPress());
    await act(async () => {});
    expect(queryId(comp, "your-day-empty")).toBeTruthy();
    act(() => comp.unmount());
  });
});
