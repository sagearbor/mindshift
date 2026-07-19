import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import WordPatternsPanel, {
  maxRate,
  methodLines,
} from "../src/components/WordPatternsPanel";
import type { WordMetrics, WordMetricsSpeaker } from "../src/api/client";

function wmSpeaker(over: Partial<WordMetricsSpeaker> = {}): WordMetricsSpeaker {
  return {
    i_rate: 8,
    you_rate: 3,
    we_rate: 1,
    anger_rate: 0.5,
    fear_rate: 0.2,
    sadness_rate: 0.4,
    joy_rate: 1.1,
    trust_rate: 0.9,
    word_count: 240,
    ...over,
  };
}

const method = {
  description: "I-statements are first-person singular pronouns per 100 words.",
  source: "deterministic regex over the transcript",
};

const wordMetrics: WordMetrics = {
  speakers: {
    Alice: wmSpeaker({ i_rate: 9, you_rate: 2 }),
    Bob: wmSpeaker({ i_rate: 4, you_rate: 7 }),
  },
  method,
};

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

function press(comp: renderer.ReactTestRenderer, id: string) {
  act(() => {
    comp.root.find((n) => n.props?.testID === id).props.onPress();
  });
}

describe("maxRate", () => {
  it("finds the largest finite rate across accessors, ignoring nulls", () => {
    const speakers = [
      wmSpeaker({ i_rate: 9, you_rate: 2 }),
      wmSpeaker({ i_rate: null, you_rate: 12 }),
    ];
    expect(maxRate(speakers, ["i_rate", "you_rate"])).toBe(12);
  });

  it("returns 0 when nothing is measurable (all null)", () => {
    const speakers = [wmSpeaker({ i_rate: null, you_rate: null, we_rate: null })];
    expect(maxRate(speakers, ["i_rate", "you_rate", "we_rate"])).toBe(0);
  });
});

describe("methodLines", () => {
  it("turns the method map into verbatim key/value lines", () => {
    const lines = methodLines(method);
    expect(lines).toEqual([
      { key: "description", value: method.description },
      { key: "source", value: method.source },
    ]);
  });

  it("stringifies non-string method values defensively", () => {
    const lines = methodLines({ window: 100, flags: ["a", "b"] });
    expect(lines[0]).toEqual({ key: "window", value: "100" });
    expect(lines[1].value).toContain("a");
  });
});

describe("WordPatternsPanel", () => {
  it("hides itself entirely when word_metrics is absent (old server)", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<WordPatternsPanel wordMetrics={undefined} />);
    });
    expect(comp.toJSON()).toBeNull();
    act(() => comp.unmount());
  });

  it("renders collapsed by default, then expands the body on tap", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<WordPatternsPanel wordMetrics={wordMetrics} />);
    });
    // Collapsed: the panel + toggle exist, but not the body.
    expect(queryId(comp, "word-patterns-panel")).toBeTruthy();
    expect(queryId(comp, "word-patterns-body")).toBeNull();

    press(comp, "word-patterns-toggle");

    expect(queryId(comp, "word-patterns-body")).toBeTruthy();
    // Star metric: the I / you bars per speaker, in plain language.
    expect(queryId(comp, "word-patterns-i-Alice")).toBeTruthy();
    expect(queryId(comp, "word-patterns-you-Alice")).toBeTruthy();
    const text = JSON.stringify(comp.toJSON());
    expect(text).toContain("Talks about own feelings");
    expect(text).toContain("Points at the other person");
    act(() => comp.unmount());
  });

  it("shows an honest note for a low-sample speaker instead of noisy bars", () => {
    const lowSample: WordMetrics = {
      speakers: {
        Alice: wmSpeaker(),
        Quiet: wmSpeaker({
          i_rate: null,
          you_rate: null,
          we_rate: null,
          anger_rate: null,
          fear_rate: null,
          sadness_rate: null,
          joy_rate: null,
          trust_rate: null,
          word_count: 4,
          low_sample: true,
        }),
      },
      method,
    };
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<WordPatternsPanel wordMetrics={lowSample} />);
    });
    press(comp, "word-patterns-toggle");
    expect(queryId(comp, "word-patterns-lowsample-Quiet")).toBeTruthy();
    // The low-sample speaker gets no I-bar (nothing fabricated).
    expect(queryId(comp, "word-patterns-i-Quiet")).toBeNull();
    // But the well-sampled speaker still renders bars.
    expect(queryId(comp, "word-patterns-i-Alice")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("renders the method text verbatim under 'How is this counted?'", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<WordPatternsPanel wordMetrics={wordMetrics} />);
    });
    press(comp, "word-patterns-toggle");
    // Method body is behind its own expando.
    expect(queryId(comp, "word-patterns-method")).toBeNull();
    press(comp, "word-patterns-method-toggle");
    expect(queryId(comp, "word-patterns-method")).toBeTruthy();
    expect(JSON.stringify(comp.toJSON())).toContain(method.description);
    act(() => comp.unmount());
  });
});
