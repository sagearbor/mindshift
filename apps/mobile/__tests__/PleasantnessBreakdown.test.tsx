import React from "react";
import renderer, { act, type ReactTestInstance } from "react-test-renderer";
import PleasantnessBreakdown, {
  aggregateBreakdowns,
} from "../src/components/PleasantnessBreakdown";
import type { AnalyzePerTurn } from "../src/api/client";

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll(
    (n) => typeof n.type === "string" && n.props?.testID === id,
  );
  return found.length > 0 ? found[0] : null;
}

function allIds(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance[] {
  return comp.root.findAll(
    (n) => typeof n.type === "string" && n.props?.testID === id,
  );
}

function textOf(node: ReactTestInstance | null): string {
  if (!node) return "";
  return node
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");
}

/** A per-turn entry with the five dims. Helper keeps the fixtures terse. */
function turn(
  index: number,
  speaker: string,
  dims: Partial<AnalyzePerTurn["dims"]> & object,
  pleasantness: number | null,
): AnalyzePerTurn {
  return {
    index,
    speaker,
    heat: 20,
    markers: [],
    is_spike: false,
    trigger_phrase: null,
    dims: dims as AnalyzePerTurn["dims"],
    pleasantness,
  };
}

async function mount(props: React.ComponentProps<typeof PleasantnessBreakdown>) {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<PleasantnessBreakdown {...props} />);
  });
  return comp;
}

describe("aggregateBreakdowns", () => {
  it("means each dimension over a speaker's turns and renormalizes the composite over measured dims", () => {
    const perTurn: AnalyzePerTurn[] = [
      turn(0, "A", {
        warmth: 60,
        constructiveness: 80,
        calmness: 90,
        respect: 100,
        engagement: 100,
      }, 80),
      turn(1, "A", {
        warmth: 80,
        constructiveness: 80,
        calmness: 90,
        respect: 100,
        engagement: 100,
      }, 86),
    ];
    const { perSpeaker } = aggregateBreakdowns(perTurn);
    expect(perSpeaker).toHaveLength(1);
    const a = perSpeaker[0];
    // warmth mean = (60+80)/2 = 70
    expect(a.dims.find((d) => d.dim === "warmth")!.mean).toBe(70);
    // All five measured → renormalized weighted mean = raw weighted mean.
    // 0.3*70 + 0.25*80 + 0.2*90 + 0.15*100 + 0.1*100 = 21+20+18+15+10 = 84
    expect(a.composite).toBe(84);
    expect(a.measured).toBe(true);
  });

  it("keeps an unmeasurable dimension null (never 0) and renormalizes the composite without it", () => {
    // engagement unmeasured (one voice) — the other four measured.
    const perTurn: AnalyzePerTurn[] = [
      turn(0, "A", {
        warmth: 100,
        constructiveness: 100,
        calmness: 100,
        respect: 100,
        engagement: null,
      }, 100),
    ];
    const { perSpeaker } = aggregateBreakdowns(perTurn);
    const a = perSpeaker[0];
    expect(a.dims.find((d) => d.dim === "engagement")!.mean).toBeNull();
    // Renormalized over the measured 0.9 of weight → still 100, NOT dragged to
    // 90 by treating the missing dim as 0.
    expect(a.composite).toBe(100);
  });

  it("groups per speaker in first-appearance order", () => {
    const perTurn: AnalyzePerTurn[] = [
      turn(0, "B", { warmth: 50 }, 50),
      turn(1, "A", { warmth: 50 }, 50),
    ];
    const { perSpeaker } = aggregateBreakdowns(perTurn);
    expect(perSpeaker.map((s) => s.speaker)).toEqual(["B", "A"]);
  });
});

describe("PleasantnessBreakdown", () => {
  const perTurn: AnalyzePerTurn[] = [
    turn(0, "Speaker A", {
      warmth: 70,
      constructiveness: 90,
      calmness: 90,
      respect: 95,
      engagement: 100,
    }, 86),
    turn(1, "Speaker B", {
      warmth: 20,
      constructiveness: 20,
      calmness: 25,
      respect: 20,
      engagement: null, // unmeasured for this speaker
    }, 21),
  ];

  it("renders each speaker's composite score and a five-lane bar with a fill per dimension", async () => {
    const comp = await mount({ perTurn });
    expect(textOf(queryId(comp, "pleasantness-breakdown-score-Speaker A"))).toBe("86");
    expect(textOf(queryId(comp, "pleasantness-breakdown-score-Speaker B"))).toBe("21");
    // Two bars (one per speaker), each with five lanes.
    expect(allIds(comp, "pleasantness-bar")).toHaveLength(2);
    expect(allIds(comp, "lane-warmth")).toHaveLength(2);
    // Speaker A measured all five → a fill for engagement exists somewhere.
    expect(allIds(comp, "lane-engagement-fill").length).toBeGreaterThan(0);
  });

  it("HATCHES an unmeasured dimension instead of showing it as 0 or redistributing", async () => {
    const comp = await mount({ perTurn });
    // Speaker B's engagement was null → a hatch, not a fill, appears for it.
    expect(allIds(comp, "lane-engagement-hatch").length).toBeGreaterThan(0);
  });

  it("shows an honest empty state when no turn carried dims", async () => {
    const bare: AnalyzePerTurn[] = [
      {
        index: 0,
        speaker: "A",
        heat: 20,
        markers: [],
        is_spike: false,
        trigger_phrase: null,
      },
    ];
    const comp = await mount({ perTurn: bare });
    expect(queryId(comp, "pleasantness-breakdown-empty")).not.toBeNull();
  });
});
