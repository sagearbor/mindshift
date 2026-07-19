import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import GlanceSummary from "../src/components/GlanceSummary";
import type { AnalyzePerSpeaker, AnalyzePerTurn } from "../src/api/client";

function speaker(over: Partial<AnalyzePerSpeaker> = {}): AnalyzePerSpeaker {
  return {
    turns: 2,
    talk_share: 0.5,
    avg_heat: 30,
    peak_heat: 40,
    peak_turn_index: 1,
    heat_variance: 100,
    interruptions: null,
    horsemen: { criticism: 0, contempt: 0, defensiveness: 0, stonewalling: 0 },
    repair_attempts: 0,
    repairs_accepted: 0,
    ...over,
  };
}

const perTurn: AnalyzePerTurn[] = [
  { index: 0, speaker: "Alice", heat: 20, markers: [], is_spike: false, trigger_phrase: null },
  { index: 1, speaker: "Bob", heat: 30, markers: [], is_spike: false, trigger_phrase: null },
  { index: 2, speaker: "Alice", heat: 25, markers: [], is_spike: false, trigger_phrase: null },
];

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

describe("GlanceSummary", () => {
  it("renders a verdict chip and a bar row per speaker", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <GlanceSummary
          perSpeaker={{
            Alice: speaker({ avg_heat: 20, talk_share: 0.6, repair_attempts: 1 }),
            Bob: speaker({ avg_heat: 30, talk_share: 0.4 }),
          }}
          perTurn={perTurn}
        />,
      );
    });
    expect(queryId(comp, "glance-summary")).toBeTruthy();
    expect(queryId(comp, "glance-verdict")).toBeTruthy();
    expect(queryId(comp, "glance-row-Alice")).toBeTruthy();
    expect(queryId(comp, "glance-row-Bob")).toBeTruthy();
    // Heat + talk bars and the marker/repair tally are present per speaker.
    expect(queryId(comp, "glance-heat-Alice")).toBeTruthy();
    expect(queryId(comp, "glance-talk-Alice")).toBeTruthy();
    expect(queryId(comp, "glance-tally-Alice")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("applies display labels (e.g. 'You') and shows the value labels on the bars", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <GlanceSummary
          perSpeaker={{ Alice: speaker({ avg_heat: 42, talk_share: 0.75 }) }}
          perTurn={perTurn}
          speakerLabels={{ Alice: { display_label: "You", label_source: "enrolled" } }}
        />,
      );
    });
    const text = JSON.stringify(comp.toJSON());
    expect(text).toContain("You");
    expect(text).toContain("42"); // avg heat value label
    expect(text).toContain("75%"); // talk-share value label
    act(() => comp.unmount());
  });

  it("reads a heated conversation with its verdict tone", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <GlanceSummary
          perSpeaker={{ Alice: speaker(), Bob: speaker({ avg_heat: 70 }) }}
          perTurn={[
            { index: 0, speaker: "Alice", heat: 80, markers: [], is_spike: true, trigger_phrase: null },
            { index: 1, speaker: "Bob", heat: 30, markers: [], is_spike: false, trigger_phrase: null },
            { index: 2, speaker: "Alice", heat: 90, markers: [], is_spike: true, trigger_phrase: null },
          ]}
        />,
      );
    });
    const text = JSON.stringify(comp.toJSON());
    expect(text).toContain("heated");
    act(() => comp.unmount());
  });

  it("renders nothing when there are no speakers", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <GlanceSummary perSpeaker={{}} perTurn={[]} />,
      );
    });
    expect(comp.toJSON()).toBeNull();
    act(() => comp.unmount());
  });
});
