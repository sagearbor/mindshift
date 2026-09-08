/** Growth pull-to-refresh: re-reads /growth while keeping the current chart
 *  on screen (a live session's batch analysis lands seconds after it ends). */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import GrowthScreen from "../src/screens/GrowthScreen";
import { getGrowth, getVoiceProfile } from "../src/api/client";
import type { GrowthResult } from "../src/api/client";

/**
 * Unmount every tree this file creates.
 *
 * react-test-renderer keeps a mounted tree scheduled, and a state update that
 * lands AFTER Jest tears the environment down throws "You are trying to
 * `import` a file after the Jest environment has been torn down" — which is
 * not a test failure but a WORKER CRASH, taking every unrelated suite sharing
 * that worker with it. That is why running the whole suite at once used to
 * fail a handful of random files that each passed on their own.
 */
const __trees: renderer.ReactTestRenderer[] = [];
function track<T extends renderer.ReactTestRenderer>(t: T): T {
  __trees.push(t);
  return t;
}
afterEach(async () => {
  await act(async () => {});
  act(() => {
    for (const t of __trees.splice(0)) {
      try {
        t.unmount();
      } catch {
        // A tree a test already unmounted, or one whose teardown throws, must
        // not fail the test that otherwise passed.
      }
    }
  });
});


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
    gaps: { not_analyzed: 0, not_your_conversation: 0, could_not_find_you: 0 },
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
      comp = track(renderer.create(<GrowthScreen onOpenRecording={jest.fn()} onOpenRecordings={jest.fn()} />));
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
