import React from "react";
import { AppState } from "react-native";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import GrowthScreen from "../src/screens/GrowthScreen";
import { catchUpVoice, getGrowth, getVoiceProfile } from "../src/api/client";
import type { GrowthPoint, GrowthResult, VoiceProfile } from "../src/api/client";

jest.mock("../src/api/client", () => ({
  getGrowth: jest.fn(),
  getVoiceProfile: jest.fn(),
  catchUpVoice: jest.fn(),
}));
const mockGetGrowth = getGrowth as jest.Mock;
const mockGetVoiceProfile = getVoiceProfile as jest.Mock;
const mockCatchUpVoice = catchUpVoice as jest.Mock;

function voiceProfile(overrides: Partial<VoiceProfile> = {}): VoiceProfile {
  return {
    available: true,
    storage_enabled: true,
    enrolled: false,
    enroll_count: 0,
    ...overrides,
  };
}

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

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
    timestamp: "2026-07-01T12:00:00+00:00",
    title: "A talk",
    my_score: 70,
    partner_names: [],
    ...overrides,
  };
}

function result(overrides: Partial<GrowthResult> = {}): GrowthResult {
  return {
    points: [],
    total_recordings: 0,
    identified_recordings: 0,
    people: [],
    ...overrides,
  };
}

/** N scored points, one per day, alternating partners when given. */
function series(n: number, partners: string[][] = []): GrowthPoint[] {
  return Array.from({ length: n }, (_, i) =>
    pt({
      recording_id: `r${i}`,
      timestamp: `2026-07-${String(i + 1).padStart(2, "0")}T12:00:00Z`,
      my_score: 50 + i,
      partner_names: partners[i % Math.max(1, partners.length)] ?? [],
    }),
  );
}

function makeHandlers() {
  return {
    onBack: jest.fn(),
    onOpenRecording: jest.fn(),
    onOpenRecordings: jest.fn(),
  };
}

async function render(handlers = makeHandlers()) {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<GrowthScreen {...handlers} />);
  });
  return comp;
}

beforeEach(() => {
  mockGetGrowth.mockReset();
  mockGetVoiceProfile.mockReset();
  mockCatchUpVoice.mockReset();
  // Default: not enrolled — most existing tests never care about the
  // catch-up affordance, so it stays hidden unless a test opts in.
  mockGetVoiceProfile.mockResolvedValue(voiceProfile());
});

describe("GrowthScreen — load states", () => {
  it("shows a spinner while loading", async () => {
    mockGetGrowth.mockReturnValueOnce(new Promise(() => {}));
    const comp = await render();
    expect(queryId(comp, "growth-loading")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("shows an honest error with retry when the fetch fails", async () => {
    mockGetGrowth
      .mockRejectedValueOnce(new Error("API error: 503"))
      .mockResolvedValueOnce(result({ total_recordings: 1 }));
    const comp = await render();
    expect(queryId(comp, "growth-error")).toBeTruthy();
    await act(async () => queryId(comp, "growth-retry")!.props.onPress());
    expect(mockGetGrowth).toHaveBeenCalledTimes(2);
    expect(queryId(comp, "growth-error")).toBeNull();
    expect(queryId(comp, "growth-empty")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("back is wired", async () => {
    mockGetGrowth.mockResolvedValueOnce(result());
    const handlers = makeHandlers();
    const comp = await render(handlers);
    act(() => queryId(comp, "growth-back")!.props.onPress());
    expect(handlers.onBack).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });
});

describe("GrowthScreen — empty states", () => {
  it("with no recordings at all, explains the full path", async () => {
    mockGetGrowth.mockResolvedValueOnce(result());
    const comp = await render();
    const empty = queryId(comp, "growth-empty")!;
    expect(textOf(empty)).toContain("Analyze and store a conversation");
    act(() => comp.unmount());
  });

  it("with recordings but no identified voice, CTAs into the enrollment flow", async () => {
    mockGetGrowth.mockResolvedValueOnce(result({ total_recordings: 4 }));
    const handlers = makeHandlers();
    const comp = await render(handlers);
    const empty = queryId(comp, "growth-empty")!;
    expect(textOf(empty)).toContain("This is me");
    act(() => queryId(comp, "growth-enroll-cta")!.props.onPress());
    expect(handlers.onOpenRecordings).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });

  it("not enrolled: no catch-up button, and the copy doesn't mention it", async () => {
    mockGetGrowth.mockResolvedValueOnce(result({ total_recordings: 4 }));
    mockGetVoiceProfile.mockResolvedValue(voiceProfile({ enrolled: false }));
    const comp = await render();
    expect(queryId(comp, "growth-catchup-cta")).toBeNull();
    expect(textOf(queryId(comp, "growth-empty")!)).not.toContain("Catch up");
    act(() => comp.unmount());
  });
});

describe("GrowthScreen — catch up my past recordings", () => {
  it("enrolled + zero identified: offers the catch-up button and mentions both paths", async () => {
    mockGetGrowth.mockResolvedValueOnce(result({ total_recordings: 5 }));
    mockGetVoiceProfile.mockResolvedValue(voiceProfile({ enrolled: true }));
    const comp = await render();
    expect(queryId(comp, "growth-catchup-cta")).toBeTruthy();
    const empty = queryId(comp, "growth-empty")!;
    expect(textOf(empty)).toContain("This is me");
    expect(textOf(empty)).toContain("Catch up");
    act(() => comp.unmount());
  });

  it("tapping catch-up shows a pending state, then an honest match result, and refetches growth", async () => {
    mockGetGrowth
      .mockResolvedValueOnce(result({ total_recordings: 5 }))
      .mockResolvedValueOnce(
        result({
          points: series(3),
          total_recordings: 5,
          identified_recordings: 3,
        }),
      );
    mockGetVoiceProfile.mockResolvedValue(voiceProfile({ enrolled: true }));
    let resolveCatchUp!: (v: {
      checked: number;
      newly_identified: number;
      remaining: number;
    }) => void;
    mockCatchUpVoice.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveCatchUp = resolve;
      }),
    );

    const comp = await render();
    act(() => queryId(comp, "growth-catchup-cta")!.props.onPress());
    expect(queryId(comp, "growth-catchup-pending")).toBeTruthy();

    await act(async () =>
      resolveCatchUp({ checked: 5, newly_identified: 3, remaining: 0 }),
    );

    expect(mockGetGrowth).toHaveBeenCalledTimes(2); // refetched after success
    expect(textOf(queryId(comp, "growth-catchup-result")!)).toBe(
      "Found you in 3 of 5 recordings",
    );
    // The refetched growth now has identified points — the chart renders.
    expect(queryId(comp, "growth-empty")).toBeNull();
    expect(queryId(comp, "growth-footer")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("honest zero-match result keeps the empty state and suggests 'This is me'", async () => {
    mockGetGrowth
      .mockResolvedValueOnce(result({ total_recordings: 5 }))
      .mockResolvedValueOnce(result({ total_recordings: 5 }));
    mockGetVoiceProfile.mockResolvedValue(voiceProfile({ enrolled: true }));
    mockCatchUpVoice.mockResolvedValueOnce({
      checked: 5, newly_identified: 0, remaining: 0,
    });

    const comp = await render();
    await act(async () => queryId(comp, "growth-catchup-cta")!.props.onPress());

    expect(textOf(queryId(comp, "growth-catchup-result")!)).toContain(
      "No match found",
    );
    expect(textOf(queryId(comp, "growth-catchup-result")!)).toContain(
      "This is me",
    );
    expect(queryId(comp, "growth-empty")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("a 503 shows the specific 'voice matching unavailable' error, never a silent no-op", async () => {
    mockGetGrowth.mockResolvedValueOnce(result({ total_recordings: 5 }));
    mockGetVoiceProfile.mockResolvedValue(voiceProfile({ enrolled: true }));
    mockCatchUpVoice.mockRejectedValueOnce(
      Object.assign(new Error("Voice ID unavailable"), { status: 503 }),
    );

    const comp = await render();
    await act(async () => queryId(comp, "growth-catchup-cta")!.props.onPress());

    expect(textOf(queryId(comp, "growth-catchup-error")!)).toBe(
      "Voice matching isn't available on the server right now.",
    );
    expect(mockGetGrowth).toHaveBeenCalledTimes(1); // no refetch on failure
    act(() => comp.unmount());
  });

  it("a dropped socket ('Network request failed') says the connection was lost", async () => {
    mockGetGrowth.mockResolvedValueOnce(result({ total_recordings: 5 }));
    mockGetVoiceProfile.mockResolvedValue(voiceProfile({ enrolled: true }));
    mockCatchUpVoice.mockRejectedValueOnce(new TypeError("Network request failed"));

    const comp = await render();
    await act(async () => queryId(comp, "growth-catchup-cta")!.props.onPress());

    expect(textOf(queryId(comp, "growth-catchup-error")!)).toBe(
      "Lost the connection while checking — keep the app open and try again.",
    );
    expect(mockGetGrowth).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });

  it("backgrounded mid catch-up: on return, a failed request still re-reads growth (partial progress is real)", async () => {
    // RN's jest preset mocks AppState.addEventListener as a jest.fn — grab
    // the listener GrowthScreen registers so the test can drive it. Scoped to
    // this test: cleared here, and the subscription is removed on unmount.
    const addListener = AppState.addEventListener as jest.Mock;
    addListener.mockClear();
    const remove = jest.fn();
    addListener.mockReturnValueOnce({ remove });

    mockGetGrowth
      .mockResolvedValueOnce(result({ total_recordings: 5 }))
      .mockResolvedValueOnce(
        result({ points: series(2), total_recordings: 5, identified_recordings: 2 }),
      );
    mockGetVoiceProfile.mockResolvedValue(voiceProfile({ enrolled: true }));
    let rejectCatchUp!: (err: unknown) => void;
    mockCatchUpVoice.mockReturnValueOnce(
      new Promise((_, reject) => {
        rejectCatchUp = reject;
      }),
    );

    const comp = await render();
    expect(addListener).toHaveBeenCalledWith("change", expect.any(Function));
    const onChange = addListener.mock.calls[0][1] as (s: string) => void;

    await act(async () => queryId(comp, "growth-catchup-cta")!.props.onPress());
    expect(queryId(comp, "growth-catchup-pending")).toBeTruthy();

    act(() => onChange("background"));
    act(() => onChange("active"));
    await act(async () => rejectCatchUp(new TypeError("Network request failed")));

    expect(textOf(queryId(comp, "growth-catchup-error")!)).toBe(
      "Lost the connection while checking — keep the app open and try again.",
    );
    expect(mockGetGrowth).toHaveBeenCalledTimes(2); // re-read after settling
    expect(queryId(comp, "growth-dot-r0")).toBeTruthy(); // what DID get identified
    expect(queryId(comp, "growth-catchup-pending")).toBeNull();

    act(() => comp.unmount());
    expect(remove).toHaveBeenCalledTimes(1);
  });

  it("stays reachable after the first identification: partially identified + enrolled shows the CTA next to the footer", async () => {
    // The exact regression: Part A's first "This is me" tap instantly flips
    // identified_recordings from 0 to 1 — the button must NOT become
    // permanently unreachable just because the empty state is gone.
    mockGetGrowth.mockResolvedValueOnce(
      result({
        points: series(2),
        total_recordings: 5,
        identified_recordings: 2,
      }),
    );
    mockGetVoiceProfile.mockResolvedValue(voiceProfile({ enrolled: true }));
    const comp = await render();

    expect(queryId(comp, "growth-empty")).toBeNull(); // the chart renders
    expect(queryId(comp, "growth-footer")).toBeTruthy();
    expect(queryId(comp, "growth-catchup-cta")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("fully identified: no catch-up CTA left to offer", async () => {
    mockGetGrowth.mockResolvedValueOnce(
      result({
        points: series(3),
        total_recordings: 3,
        identified_recordings: 3,
      }),
    );
    mockGetVoiceProfile.mockResolvedValue(voiceProfile({ enrolled: true }));
    const comp = await render();

    expect(queryId(comp, "growth-catchup-cta")).toBeNull();
    act(() => comp.unmount());
  });

  it("not enrolled: no catch-up CTA in the chart view either", async () => {
    mockGetGrowth.mockResolvedValueOnce(
      result({
        points: series(2),
        total_recordings: 5,
        identified_recordings: 2,
      }),
    );
    mockGetVoiceProfile.mockResolvedValue(voiceProfile({ enrolled: false }));
    const comp = await render();

    expect(queryId(comp, "growth-catchup-cta")).toBeNull();
    act(() => comp.unmount());
  });

  it("tapping the footer-adjacent CTA catches up the rest and updates the footer", async () => {
    mockGetGrowth
      .mockResolvedValueOnce(
        result({
          points: series(2),
          total_recordings: 5,
          identified_recordings: 2,
        }),
      )
      .mockResolvedValueOnce(
        result({
          points: series(5),
          total_recordings: 5,
          identified_recordings: 5,
        }),
      );
    mockGetVoiceProfile.mockResolvedValue(voiceProfile({ enrolled: true }));
    mockCatchUpVoice.mockResolvedValueOnce({
      checked: 3, newly_identified: 3, remaining: 0,
    });

    const comp = await render();
    await act(async () => queryId(comp, "growth-catchup-cta")!.props.onPress());

    expect(textOf(queryId(comp, "growth-catchup-result")!)).toBe(
      "Found you in 3 of 3 recordings",
    );
    expect(textOf(queryId(comp, "growth-footer")!)).toBe(
      "5 of 5 recordings identified your voice",
    );
    // Nothing left to catch up — the CTA is gone now.
    expect(queryId(comp, "growth-catchup-cta")).toBeNull();
    act(() => comp.unmount());
  });
});

describe("GrowthScreen — chart", () => {
  it("few points: dots render, no trend line below 5 scored points", async () => {
    mockGetGrowth.mockResolvedValueOnce(
      result({
        points: series(3),
        total_recordings: 4,
        identified_recordings: 3,
      }),
    );
    const comp = await render();
    expect(queryId(comp, "growth-dot-r0")).toBeTruthy();
    expect(queryId(comp, "growth-dot-r2")).toBeTruthy();
    expect(queryId(comp, "growth-trend")).toBeNull();
    act(() => comp.unmount());
  });

  it("draws real axes: a labeled 0–100 y axis and date ticks on the x axis", async () => {
    mockGetGrowth.mockResolvedValueOnce(
      result({
        points: series(3),
        total_recordings: 3,
        identified_recordings: 3,
      }),
    );
    const comp = await render();
    expect(textOf(queryId(comp, "growth-axis-y-label")!)).toBe("Score (0–100) ↑");
    for (const v of [0, 25, 50, 75, 100]) {
      expect(textOf(queryId(comp, `growth-axis-y-tick-${v}`)!)).toBe(String(v));
      expect(queryId(comp, `growth-axis-y-grid-${v}`)).toBeTruthy();
    }
    const xTicks = comp.root.findAll(
      (n) => typeof n.type === "string" && n.props?.testID === "growth-axis-x-tick",
    );
    expect(xTicks.length).toBeGreaterThanOrEqual(1);
    // Day-scale window (Jul 1–3) → day labels, first and last dates present.
    const labels = xTicks.map(textOf);
    expect(labels[0]).toMatch(/^Jul \d+$/);
    expect(labels[labels.length - 1]).toMatch(/^Jul \d+$/);
    expect(queryId(comp, "growth-axis-hint")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("many points: the moving-average trend appears at ≥5 scored points", async () => {
    mockGetGrowth.mockResolvedValueOnce(
      result({
        points: series(6),
        total_recordings: 6,
        identified_recordings: 6,
      }),
    );
    const comp = await render();
    expect(queryId(comp, "growth-trend")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("null scores are gaps — no dot, and they don't count toward the trend threshold", async () => {
    const points = [
      ...series(4),
      pt({
        recording_id: "gap",
        timestamp: "2026-07-20T12:00:00Z",
        my_score: null,
      }),
    ];
    mockGetGrowth.mockResolvedValueOnce(
      result({
        points,
        total_recordings: 5,
        identified_recordings: 5,
      }),
    );
    const comp = await render();
    expect(queryId(comp, "growth-dot-gap")).toBeNull();
    // 4 scored + 1 gap is still < 5 scored — no trend.
    expect(queryId(comp, "growth-trend")).toBeNull();
    act(() => comp.unmount());
  });

  it("tapping a dot opens that recording", async () => {
    mockGetGrowth.mockResolvedValueOnce(
      result({
        points: series(2),
        total_recordings: 2,
        identified_recordings: 2,
      }),
    );
    const handlers = makeHandlers();
    const comp = await render(handlers);
    act(() => queryId(comp, "growth-dot-r1")!.props.onPress());
    expect(handlers.onOpenRecording).toHaveBeenCalledWith("r1");
    act(() => comp.unmount());
  });

  it("shows the honest footer", async () => {
    mockGetGrowth.mockResolvedValueOnce(
      result({
        points: series(2),
        total_recordings: 7,
        identified_recordings: 2,
      }),
    );
    const comp = await render();
    expect(textOf(queryId(comp, "growth-footer")!)).toBe(
      "2 of 7 recordings identified your voice",
    );
    act(() => comp.unmount());
  });
});

describe("GrowthScreen — partner filter", () => {
  const points = [
    pt({
      recording_id: "with-linda",
      timestamp: "2026-07-01T12:00:00Z",
      partner_names: ["Linda"],
    }),
    pt({
      recording_id: "with-sam",
      timestamp: "2026-07-02T12:00:00Z",
      partner_names: ["Sam"],
    }),
    pt({
      recording_id: "anon",
      timestamp: "2026-07-03T12:00:00Z",
      partner_names: [],
    }),
  ];

  it("builds chips from partner names plus the unidentified bucket", async () => {
    mockGetGrowth.mockResolvedValueOnce(
      result({
        points,
        total_recordings: 3,
        identified_recordings: 3,
      }),
    );
    const comp = await render();
    expect(queryId(comp, "growth-filter-all")).toBeTruthy();
    expect(queryId(comp, "growth-filter-Linda")).toBeTruthy();
    expect(queryId(comp, "growth-filter-Sam")).toBeTruthy();
    expect(queryId(comp, "growth-filter-unidentified")).toBeTruthy();

    act(() => queryId(comp, "growth-filter-Linda")!.props.onPress());
    expect(queryId(comp, "growth-dot-with-linda")).toBeTruthy();
    expect(queryId(comp, "growth-dot-with-sam")).toBeNull();
    expect(queryId(comp, "growth-dot-anon")).toBeNull();

    act(() => queryId(comp, "growth-filter-unidentified")!.props.onPress());
    expect(queryId(comp, "growth-dot-anon")).toBeTruthy();
    expect(queryId(comp, "growth-dot-with-linda")).toBeNull();

    act(() => queryId(comp, "growth-filter-all")!.props.onPress());
    expect(queryId(comp, "growth-dot-with-linda")).toBeTruthy();
    expect(queryId(comp, "growth-dot-anon")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("shows no chip row when no partner was ever named", async () => {
    mockGetGrowth.mockResolvedValueOnce(
      result({
        points: series(3),
        total_recordings: 3,
        identified_recordings: 3,
      }),
    );
    const comp = await render();
    expect(queryId(comp, "growth-filter-all")).toBeNull();
    expect(queryId(comp, "growth-filter-unidentified")).toBeNull();
    act(() => comp.unmount());
  });

  it("states plainly when a filter leaves nothing scored", async () => {
    mockGetGrowth.mockResolvedValueOnce(
      result({
        points: [
          pt({ recording_id: "a", partner_names: ["Linda"] }),
          pt({
            recording_id: "b",
            timestamp: "2026-07-02T12:00:00Z",
            partner_names: ["Sam"],
            my_score: null,
          }),
        ],
        total_recordings: 2,
        identified_recordings: 2,
      }),
    );
    const comp = await render();
    act(() => queryId(comp, "growth-filter-Sam")!.props.onPress());
    expect(queryId(comp, "growth-filter-empty")).toBeTruthy();
    act(() => comp.unmount());
  });
});
