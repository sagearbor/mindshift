/** DynamicsScreen's Track 2 "Your tone" card: rendered from a live
 *  analysis's `live.tone_summary`, hidden for a plain text/upload analysis. */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import DynamicsScreen from "../src/screens/DynamicsScreen";
import { useSessionStore } from "../src/store/sessionStore";
import type { AnalyzeResult, ToneSummary } from "../src/api/client";

jest.mock("../src/api/client", () => ({
  postAnalyze: jest.fn(),
  postCounterfactual: jest.fn(),
}));

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

const base: AnalyzeResult = {
  per_turn: [
    { index: 0, speaker: "Speaker A", heat: 20, markers: [], is_spike: false, trigger_phrase: null },
    { index: 1, speaker: "Speaker B", heat: 30, markers: [], is_spike: false, trigger_phrase: null },
  ],
  per_speaker: {},
  dynamics: {
    coupling: { strength: null, leader: null, description: "" },
    deescalation: { who_first: null, follow_rate: null, description: "" },
    triggers: [],
    requests: [],
  },
  narrative: "Quiet.",
};

const summary: ToneSummary = {
  self_speaker: "Speaker A",
  self: {
    turns: 3, scored_turns: 3, labels: { warm: 1, frustrated: 1, defensive: 1 },
    mean: { warmth: 50, defensiveness: null, sarcasm: null, sadness: null, frustration: 42 },
    escalation_turns: [2, 4], escalation_count: 2,
  },
  audio: { turns: 1, labels: { angry: 1 }, escalation_turns: [2], escalation_count: 1 },
  audio_tone_surfaced: false,
  people: [
    { speaker: "Speaker B", person_id: "p-mom", display_name: "Mom", their_turns: 3,
      self_turns: 3, turns: 3, scored_turns: 3, labels: { warm: 1, frustrated: 1, defensive: 1 },
      mean: {}, escalation_turns: [2, 4], escalation_count: 2 },
  ],
};

async function render(data: AnalyzeResult) {
  useSessionStore.setState({
    turns: [
      { speaker: "Speaker A", text: "Hey Mom.", start_time: 0, end_time: 2 },
      { speaker: "Speaker B", text: "Hi.", start_time: 2, end_time: 4 },
    ],
  });
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<DynamicsScreen onBack={() => {}} initialData={data} />);
  });
  await act(async () => {});
  return comp;
}

describe("DynamicsScreen — Your tone", () => {
  it("renders the tone card for a live analysis", async () => {
    const comp = await render({ ...base, live: { tone_summary: summary, mode: "earpiece" } });
    const card = queryId(comp, "dynamics-tone-summary");
    expect(card).toBeTruthy();
    expect(queryId(comp, "dynamics-tone-summary-chip-warm")).toBeTruthy();
    expect(queryId(comp, "dynamics-tone-summary-chip-defensive")).toBeTruthy();
    const line = queryId(comp, "dynamics-tone-summary-line");
    expect(textOf(line!)).toContain("2 escalations");
    const mom = queryId(comp, "dynamics-tone-summary-person-Speaker B");
    expect(textOf(mom!)).toContain("with Mom");
    // Audio tone is NOT surfaced unless the server allowed it.
    expect(queryId(comp, "dynamics-tone-summary-audio")).toBeNull();
    act(() => comp.unmount());
  });

  it("surfaces audio tone only when the server says so", async () => {
    const comp = await render({
      ...base,
      live: { tone_summary: { ...summary, audio_tone_surfaced: true } },
    });
    const audio = queryId(comp, "dynamics-tone-summary-audio");
    expect(audio).toBeTruthy();
    expect(textOf(audio!)).toContain("angry ×1");
    act(() => comp.unmount());
  });

  it("hides the card without a live block, or with no self bucket", async () => {
    let comp = await render(base);
    expect(queryId(comp, "dynamics-tone-summary")).toBeNull();
    act(() => comp.unmount());
    comp = await render({ ...base, live: { tone_summary: { ...summary, self: null } } });
    expect(queryId(comp, "dynamics-tone-summary")).toBeNull();
    act(() => comp.unmount());
  });
});
