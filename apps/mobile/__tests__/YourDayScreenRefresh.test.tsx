/** Your Day right after a live session: the server-confirmed episode shows
 *  up immediately (optimistic merge), the server's own row wins once listed,
 *  and pull-to-refresh drops both caches so a freshly-landed reflection /
 *  analysis is re-read. */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import YourDayScreen from "../src/screens/YourDayScreen";
import { listRecordings, getRecordingEpisodes } from "../src/api/client";
import type { RecordingSummary } from "../src/api/client";
import { useLiveEpisodeStore } from "../src/store/liveEpisodeStore";

jest.mock("../src/api/client", () => ({
  listRecordings: jest.fn(),
  getRecordingEpisodes: jest.fn(),
}));
const mockList = listRecordings as jest.Mock;
const mockEpisodes = getRecordingEpisodes as jest.Mock;

jest.setTimeout(120000);

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

function todayIso(hour: number): string {
  const d = new Date();
  d.setHours(hour, 0, 0, 0);
  return d.toISOString();
}

function rec(id: string, hour: number, overrides: Partial<RecordingSummary> = {}): RecordingSummary {
  return {
    id,
    created_at: todayIso(hour),
    filename: `${id}.m4a`,
    media_type: "audio",
    duration_seconds: 600,
    has_analysis: true,
    ...overrides,
  };
}

const flush = () => act(async () => { await new Promise((r) => setTimeout(r, 0)); });

beforeEach(() => {
  mockList.mockReset();
  mockEpisodes.mockReset().mockResolvedValue([]);
  useLiveEpisodeStore.getState().clear();
});

describe("YourDayScreen — after a live session", () => {
  it("shows a just-finished live session even before the list includes it, then defers to the server row", async () => {
    mockList.mockResolvedValue([rec("old", 9)]);
    useLiveEpisodeStore.getState().remember({
      episodeId: "ep-live",
      sessionId: "live-1",
      startedAt: todayIso(14),
      mode: "speaker",
      title: "Live session · speaker",
      turnCount: 5,
      sharedWith: [],
    });
    let comp: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<YourDayScreen onBack={jest.fn()} onOpenReplay={jest.fn()} />);
    });
    await flush();
    expect(queryId(comp!, "day-recording-ep-live")).toBeTruthy();
    expect(queryId(comp!, "day-recording-old")).toBeTruthy();
    // The optimistic row never fabricates episodes: it asked the server.
    expect(mockEpisodes).toHaveBeenCalledWith("ep-live");

    // Pull-to-refresh: the server now lists it (as a live session) — one
    // row, from the server, and both caches were dropped (re-listed).
    mockList.mockResolvedValue([rec("ep-live", 14, { media_type: "none", source_type: "live", mode: "speaker", title: "Live session · speaker" }), rec("old", 9)]);
    await act(async () => {
      queryId(comp!, "your-day-refresh")!.props.onRefresh();
    });
    await flush();
    expect(mockList).toHaveBeenCalledTimes(2);
    expect(comp!.root.findAll((n) => n.props?.testID === "day-recording-ep-live" && typeof n.type === "string")).toHaveLength(1);
    expect(mockEpisodes.mock.calls.filter((c) => c[0] === "ep-live").length).toBeGreaterThanOrEqual(2);
  });

  it("without a confirmed episode nothing is invented", async () => {
    mockList.mockResolvedValue([]);
    let comp: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<YourDayScreen onBack={jest.fn()} onOpenReplay={jest.fn()} />);
    });
    await flush();
    expect(queryId(comp!, "your-day-empty")).toBeTruthy();
  });
});
