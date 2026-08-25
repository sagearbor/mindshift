import React from "react";
import renderer, { act, type ReactTestInstance } from "react-test-renderer";
import ScoreboardPanel, { leadCopy } from "../src/components/ScoreboardPanel";
import type { Scoreboard } from "../src/live/pleasantness";

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => typeof n.type === "string" && n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

function textOf(node: ReactTestInstance | null): string {
  if (!node) return "";
  return node
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");
}

const board: Scoreboard = {
  people: [
    { speaker: "Speaker A", current: 72, series: [60, 70, 80, 78], scoredTurns: 4 },
    { speaker: "Speaker B", current: 66, series: [66], scoredTurns: 1 },
  ],
  lead: { speaker: "Speaker A", margin: 6 },
};

async function mount(props: React.ComponentProps<typeof ScoreboardPanel>) {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<ScoreboardPanel {...props} />);
  });
  return comp;
}

describe("ScoreboardPanel", () => {
  it("shows each person's current score, a sparkline, and a kind lead line under their names", async () => {
    const nameOf = (s: string) => (s === "Speaker B" ? "Mom" : s === "Speaker A" ? "You" : s);
    const comp = await mount({ board, nameOf });
    expect(textOf(queryId(comp, "scoreboard-score-Speaker A"))).toBe("72");
    expect(textOf(queryId(comp, "scoreboard-score-Speaker B"))).toBe("66");
    expect(textOf(queryId(comp, "scoreboard-row-Speaker B"))).toContain("Mom");
    expect(textOf(queryId(comp, "scoreboard-row-Speaker A"))).toContain("4 turns");
    expect(textOf(queryId(comp, "scoreboard-lead"))).toBe("You +6 — leading with kindness.");
    expect(
      comp.root.findAll((n) => typeof n.type === "string" && n.props?.testID === "tone-sparkline"),
    ).toHaveLength(2);
    expect(queryId(comp, "scoreboard-empty")).toBeNull();
  });

  it("never crowns anyone when it's even, and says so warmly", async () => {
    const even: Scoreboard = { ...board, lead: null };
    const comp = await mount({ board: even });
    expect(textOf(queryId(comp, "scoreboard-lead"))).toBe("Neck and neck — you're both bringing it.");
  });

  it("renders the honest empty state (with the caller's reason) and dashes for unscored people", async () => {
    const empty: Scoreboard = {
      people: [{ speaker: "Speaker A", current: null, series: [], scoredTurns: 0 }],
      lead: null,
    };
    const comp = await mount({ board: empty, emptyText: "Scores need on-device coaching." });
    expect(textOf(queryId(comp, "scoreboard-empty"))).toBe("Scores need on-device coaching.");
    expect(textOf(queryId(comp, "scoreboard-score-Speaker A"))).toBe("—");
    expect(queryId(comp, "scoreboard-lead")).toBeNull();
    const none = await mount({ board: null });
    expect(textOf(queryId(none, "scoreboard-empty"))).toMatch(/Scores appear as people talk/);
  });

  it("leadCopy covers every state", () => {
    const id = (s: string) => s;
    expect(leadCopy(null, id)).toMatch(/Waiting/);
    expect(leadCopy({ people: [{ speaker: "A", current: 50, series: [50], scoredTurns: 1 }], lead: null }, id)).toBe(
      "A is warming up the room.",
    );
    expect(leadCopy(board, id)).toBe("Speaker A +6 — leading with kindness.");
  });
});
