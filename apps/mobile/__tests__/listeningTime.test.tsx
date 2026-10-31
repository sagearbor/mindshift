import React from "react";
import { StyleSheet } from "react-native";
import renderer, { act } from "react-test-renderer";
import {
  OUT_OF_TIME_MESSAGE,
  formatLeft,
  noticeCopy,
  noticeCrossed,
  pillText,
  toneFor,
} from "../src/live/listeningTime";
import { remainingNow, useListeningStore } from "../src/store/listeningStore";
import { usePlanLimitStore } from "../src/store/planLimitStore";
import ListeningPill from "../src/components/ListeningPill";
import { getPlan } from "../src/api/plan";

jest.mock("../src/api/plan", () => ({ getPlan: jest.fn(), redeemCode: jest.fn() }));

describe("thresholds and copy", () => {
  test("tone: ok > 10 min, amber at 10, red at 5, out at 0", () => {
    expect(toneFor(601)).toBe("ok");
    expect(toneFor(600)).toBe("amber");
    expect(toneFor(301)).toBe("amber");
    expect(toneFor(300)).toBe("red");
    expect(toneFor(1)).toBe("red");
    expect(toneFor(0)).toBe("out");
  });

  test("notices fire once per threshold as the clock falls", () => {
    expect(noticeCrossed(700, 650)).toBeNull();
    expect(noticeCrossed(601, 600)).toBe(600);
    expect(noticeCrossed(600, 599)).toBeNull(); // already past 600
    expect(noticeCrossed(301, 300)).toBe(300);
    expect(noticeCrossed(61, 60)).toBe(60);
    expect(noticeCrossed(10, 0)).toBeNull();
  });

  test("copy", () => {
    expect(formatLeft(42 * 60)).toBe("42 min");
    expect(formatLeft(125 * 60)).toBe("2 h 05 min");
    expect(formatLeft(45)).toBe("45 s");
    expect(pillText(42 * 60, "lifetime", true)).toBe("Free time left: 42 min");
    expect(pillText(6.5 * 3600, "month", false)).toBe("6.5 h left this month");
    expect(pillText(0, "lifetime", true)).toBe("Out of free time");
    expect(noticeCopy(600, true)).toBe("About 10 minutes of free time left.");
    expect(noticeCopy(60, true)).toContain("Coaching will pause");
    expect(OUT_OF_TIME_MESSAGE).toContain("out of free time");
    expect(OUT_OF_TIME_MESSAGE).toContain("Redeem a code or upgrade");
  });

  test("remainingNow ticks only while running", () => {
    const s = { baseSec: 100, syncedAtMs: 0 };
    expect(remainingNow(s, 30_000, true)).toBe(70);
    expect(remainingNow(s, 30_000, false)).toBe(100);
    expect(remainingNow(s, 500_000, true)).toBe(0);
    expect(remainingNow({ baseSec: null, syncedAtMs: 0 }, 1, true)).toBeNull();
  });
});

describe("ListeningPill", () => {
  beforeEach(() => {
    jest.useFakeTimers();
    usePlanLimitStore.setState({ notice: null });
    (getPlan as jest.Mock).mockReset();
  });
  afterEach(() => jest.useRealTimers());

  const plan = (seconds: number, p = "free", period = "lifetime") => ({
    plan: p,
    listening: { remaining_seconds: seconds, period, limit: "x", allowance_seconds: 3600 },
  });

  async function mount(active: boolean) {
    let tree!: renderer.ReactTestRenderer;
    await act(async () => {
      tree = renderer.create(<ListeningPill active={active} />);
    });
    return tree;
  }
  const text = (t: renderer.ReactTestRenderer) =>
    t.root.findByProps({ testID: "listening-pill-text" }).props.children;

  test("counts down, turns red at 5 min, banners once, opens the sheet at zero", async () => {
    (getPlan as jest.Mock).mockResolvedValueOnce(plan(602)).mockResolvedValueOnce(plan(250)).mockResolvedValue(plan(0));
    const t = await mount(true);
    expect(text(t)).toBe("Free time left: 11 min");
    await act(async () => {
      jest.advanceTimersByTime(3000); // 599 s left: crossed 10 min
    });
    expect(t.root.findByProps({ testID: "listening-banner" })).toBeTruthy();
    expect(JSON.stringify(t.toJSON())).toContain("About 10 minutes of free time left.");
    await act(async () => {
      jest.advanceTimersByTime(180 * 1000); // re-sync: server says 250 s left
    });
    expect(JSON.stringify(t.toJSON())).toContain("About 5 minutes");
    expect(StyleSheet.flatten(t.root.findByProps({ testID: "listening-pill" }).props.style).borderColor).toBe("#b91c1c");
    await act(async () => {
      jest.advanceTimersByTime(260 * 1000);
    });
    expect(text(t)).toBe("Out of free time");
    expect(usePlanLimitStore.getState().notice?.message).toBe(OUT_OF_TIME_MESSAGE);
  });

  test("paid plan shows hours left this month and an idle pill does not tick", async () => {
    (getPlan as jest.Mock).mockResolvedValue(plan(7 * 3600, "basic", "month"));
    const t = await mount(false);
    await act(async () => {
      jest.advanceTimersByTime(60_000);
    });
    expect(text(t)).toBe("7.0 h left this month");
  });

  test("renders nothing when the plan has no listening cap", async () => {
    (getPlan as jest.Mock).mockResolvedValue({ plan: "max", listening: null });
    useListeningStore.getState().clear();
    const t = await mount(true);
    expect(t.toJSON()).toBeNull();
  });
});
