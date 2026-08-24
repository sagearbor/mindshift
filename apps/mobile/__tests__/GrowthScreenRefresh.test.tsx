/** Growth pull-to-refresh: re-reads /growth while keeping the current chart
 *  on screen (a live session's batch analysis lands seconds after it ends). */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import GrowthScreen from "../src/screens/GrowthScreen";
import { getGrowth, getVoiceProfile } from "../src/api/client";
import type { GrowthResult } from "../src/api/client";

jest.mock("../src/api/client", () => ({
  getGrowth: jest.fn(),
  getVoiceProfile: jest.fn(),
  catchUpVoice: jest.fn(),
}));
const mockGrowth = getGrowth as jest.Mock;
const mockProfile = getVoiceProfile as jest.Mock;

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

function result(points: number): GrowthResult {
  return {
    points: Array.from({ length: points }, (_, i) => ({
      recording_id: `r${i}`,
      timestamp: `2026-08-2${i}T10:00:00Z`,
      title: `Talk ${i}`,
      my_score: 60 + i,
      partner_names: [],
    })),
    total_recordings: points,
    identified_recordings: points,
    people: [],
  };
}

const flush = () => act(async () => { await Promise.resolve(); });

beforeEach(() => {
  mockGrowth.mockReset();
  mockProfile.mockReset().mockResolvedValue({ available: true, storage_enabled: true, enrolled: true, enroll_count: 1 });
});

describe("GrowthScreen — pull to refresh", () => {
  it("re-fetches growth on pull and reflects the new points", async () => {
    mockGrowth.mockResolvedValueOnce(result(1));
    let comp: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<GrowthScreen onOpenRecording={jest.fn()} onOpenRecordings={jest.fn()} />);
    });
    await flush();
    expect(mockGrowth).toHaveBeenCalledTimes(1);
    const refresh = queryId(comp!, "growth-refresh");
    expect(refresh).toBeTruthy();
    expect(refresh!.props.refreshing).toBe(false);

    mockGrowth.mockResolvedValueOnce(result(2));
    await act(async () => {
      refresh!.props.onRefresh();
    });
    await flush();
    expect(mockGrowth).toHaveBeenCalledTimes(2);
    expect(queryId(comp!, "growth-refresh")!.props.refreshing).toBe(false);
    expect(JSON.stringify(comp!.toJSON())).toContain("2 of 2");
  });
});
