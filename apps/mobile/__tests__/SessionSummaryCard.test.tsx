import React from "react";
import renderer, { act } from "react-test-renderer";
import SessionSummaryCard from "../src/components/SessionSummaryCard";
import type { SessionSummary } from "../src/live/sessionSummary";
import { useDevModeStore } from "../src/store/devModeStore";

jest.mock("../src/api/client", () => ({
  postShare: jest.fn(),
}));

const summary: SessionSummary = {
  durationMs: 134000,
  turnsBySpeaker: [
    { speaker: "You", turns: 3 },
    { speaker: "Mom", turns: 2 },
  ],
  totalTurns: 5,
  escalations: 1,
  firstWordsMedianMs: 640,
  firstWordsBestMs: 410,
  spokenTurns: 2,
  topProvider: "os",
};

const linked = { linked: true, therapist_email: "mom@example.com", status: "accepted" as const, auto_share: true };

/** All rendered text, joined — RN splits interpolated strings into fragments. */
function text(root: renderer.ReactTestRenderer) {
  return root.root
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");
}

describe("SessionSummaryCard", () => {
  // Latency stat + provider tag are developer-mode details.
  beforeEach(() => useDevModeStore.setState({ devMode: true }));
  afterEach(() => useDevModeStore.setState({ devMode: false }));

  it("shows duration, turns, escalations, first-words latency and per-person turns", () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<SessionSummaryCard summary={summary} episode={null} therapist={null} />);
    });
    const t = text(root!);
    expect(t).toContain("2m 14s");
    expect(t).toContain("640 ms");
    expect(t).toContain("best 410 ms · 2 spoken");
    expect(t).toContain("You: 3 · Mom: 2 · via os");
    expect(root!.root.findAllByProps({ testID: "summary-share-therapist" })).toHaveLength(0);
  });

  it("nothing spoken (therapist mode / legacy path) reads as unknown, never 0", () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(
        <SessionSummaryCard
          summary={{ ...summary, firstWordsMedianMs: null, firstWordsBestMs: null, spokenTurns: 0, topProvider: null }}
          episode={null}
          therapist={null}
        />,
      );
    });
    const t = text(root!);
    expect(t).toContain("nothing spoken");
    expect(t).not.toContain("0 ms");
  });

  it("auto-shared at ingest: says so instead of offering the button", () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(
        <SessionSummaryCard
          summary={summary}
          episode={{ episodeId: "ep-1", postStatus: "created", sharedWith: ["Mom@Example.com"] }}
          therapist={linked}
        />,
      );
    });
    expect(text(root!)).toContain("Shared with mom@example.com automatically");
    expect(root!.root.findAllByProps({ testID: "summary-share-therapist" })).toHaveLength(0);
  });

  it("linked but not auto-shared: the button shares the episode with the therapist and confirms", async () => {
    const share = jest.fn().mockResolvedValue({ shares: [] });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(
        <SessionSummaryCard
          summary={summary}
          episode={{ episodeId: "ep-1", postStatus: "created", sharedWith: [] }}
          therapist={{ ...linked, auto_share: false }}
          share={share}
        />,
      );
    });
    await act(async () => {
      await root!.root.findByProps({ testID: "summary-share-therapist" }).props.onPress();
    });
    expect(share).toHaveBeenCalledWith("ep-1", "mom@example.com");
    expect(text(root!)).toContain("Shared with mom@example.com");
  });

  it("share failure surfaces the server's detail; a failed POST is stated honestly", async () => {
    const share = jest.fn().mockRejectedValue(Object.assign(new Error("x"), { detail: "no MindShift account with that email" }));
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(
        <SessionSummaryCard
          summary={summary}
          episode={{ episodeId: "ep-1", postStatus: "created", sharedWith: [] }}
          therapist={linked}
          share={share}
        />,
      );
    });
    await act(async () => {
      await root!.root.findByProps({ testID: "summary-share-therapist" }).props.onPress();
    });
    expect(text(root!)).toContain("no MindShift account with that email");

    act(() => {
      root!.update(
        <SessionSummaryCard
          summary={summary}
          episode={{ episodeId: null, postStatus: "failed", sharedWith: [] }}
          therapist={linked}
        />,
      );
    });
    expect(root!.root.findByProps({ testID: "summary-post-failed" })).toBeTruthy();
    expect(root!.root.findAllByProps({ testID: "summary-share-therapist" })).toHaveLength(0);
  });

  it("developer mode off: duration/turns/people survive, latency stat and provider tag don't", () => {
    useDevModeStore.setState({ devMode: false });
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<SessionSummaryCard summary={summary} episode={null} therapist={null} />);
    });
    const t = text(root!);
    expect(t).toContain("2m 14s");
    expect(t).toContain("You: 3 · Mom: 2");
    expect(t).not.toContain("640 ms");
    expect(t).not.toContain("via os");
  });
});
