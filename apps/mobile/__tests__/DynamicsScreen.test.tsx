import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import DynamicsScreen from "../src/screens/DynamicsScreen";
import { useSessionStore } from "../src/store/sessionStore";
import { postAnalyze, postCounterfactual } from "../src/api/client";
import type { AnalyzeResult, CounterfactualResult } from "../src/api/client";

// Mock the network fns; types are erased so the rest of the module is unused.
jest.mock("../src/api/client", () => ({
  postAnalyze: jest.fn(),
  postCounterfactual: jest.fn(),
}));
const mockAnalyze = postAnalyze as jest.Mock;
const mockCounterfactual = postCounterfactual as jest.Mock;

const fixture: AnalyzeResult = {
  per_turn: [
    { index: 0, speaker: "Alice", heat: 20, markers: [], is_spike: false, trigger_phrase: null },
    { index: 1, speaker: "Bob", heat: 30, markers: ["defensiveness"], is_spike: false, trigger_phrase: null },
    { index: 2, speaker: "Alice", heat: 40, markers: [], is_spike: false, trigger_phrase: null },
    { index: 3, speaker: "Bob", heat: 88, markers: ["contempt"], is_spike: true, trigger_phrase: "you always" },
  ],
  per_speaker: {
    Alice: {
      turns: 2, talk_share: 0.5, avg_heat: 30, peak_heat: 40, peak_turn_index: 2,
      heat_variance: 100, interruptions: null,
      horsemen: { criticism: 1, contempt: 0, defensiveness: 0, stonewalling: 0 },
      repair_attempts: 1, repairs_accepted: 1,
    },
    Bob: {
      turns: 2, talk_share: 0.5, avg_heat: 59, peak_heat: 88, peak_turn_index: 3,
      heat_variance: 841, interruptions: 2,
      horsemen: { criticism: 0, contempt: 1, defensiveness: 1, stonewalling: 0 },
      repair_attempts: 0, repairs_accepted: 0,
    },
  },
  report_cards: {
    Alice: {
      score: 72,
      headline: "Kept reaching for repair.",
      did_well: "Named a concrete need without name-calling.",
      work_on: "Lead with the ask before the grievance.",
    },
    Bob: {
      score: 41,
      headline: "Defensiveness escalated the heat.",
      did_well: "Stayed in the room instead of stonewalling.",
      work_on: "Swap 'you always' for how it lands on you.",
    },
  },
  dynamics: {
    coupling: { strength: 0.7, leader: "Bob", description: "When Bob heats up, Alice follows within a turn." },
    deescalation: { who_first: "Alice", follow_rate: 0.6, description: "Alice tends to reach for repair first." },
    triggers: [
      { phrase: "you always", speaker: "Bob", turn_index: 3, heat_delta: 48 },
      // The server emits triggers for COOLING phrases too — delta is negative.
      { phrase: "I hear you", speaker: "Alice", turn_index: 2, heat_delta: -10 },
    ],
    requests: [{ speaker: "Alice", request: "asked for help with chores", outcome: "deflected" }],
  },
  narrative: "The pair circle the same worn groove: a bid for help, met with defense.",
};

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

const simFixture: CounterfactualResult = {
  pivot_index: 1,
  rewritten_text: "I want to help — what would feel most useful right now?",
  rationale: "Naming a willingness to help instead of defending invites repair.",
  // Spans the pivot (turn 1, Bob) to the last turn, so both speakers appear —
  // two dashed overlay lines.
  simulated_per_turn: [
    { index: 1, speaker: "Bob", heat: 25 },
    { index: 2, speaker: "Alice", heat: 22 },
    { index: 3, speaker: "Bob", heat: 30 },
  ],
  disclaimer: "This is a hypothetical projection, not a prediction of what would have happened.",
};

beforeEach(() => {
  mockAnalyze.mockReset();
  mockCounterfactual.mockReset();
  act(() => {
    useSessionStore.setState({
      turns: [
        { speaker: "Alice", text: "You never help." },
        { speaker: "Bob", text: "I do plenty." },
        { speaker: "Alice", text: "Not with the kids." },
        { speaker: "Bob", text: "You always say that." },
      ],
    });
  });
});

describe("DynamicsScreen", () => {
  it("fetches on mount and renders chart, speaker cards, and narrative", async () => {
    mockAnalyze.mockResolvedValueOnce(fixture);

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<DynamicsScreen onBack={() => {}} />);
    });

    expect(mockAnalyze).toHaveBeenCalledTimes(1);
    // Turns from the store were sent.
    expect(mockAnalyze.mock.calls[0][0]).toHaveLength(4);

    expect(queryId(comp, "dynamics-content")).toBeTruthy();
    expect(queryId(comp, "heat-chart")).toBeTruthy();
    expect(queryId(comp, "speaker-card-Alice")).toBeTruthy();
    expect(queryId(comp, "speaker-card-Bob")).toBeTruthy();
    expect(queryId(comp, "dynamics-narrative")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("formats trigger deltas by direction: sparked +N for heating, cooled −N for cooling", async () => {
    mockAnalyze.mockResolvedValueOnce(fixture);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<DynamicsScreen onBack={() => {}} />);
    });

    const text = JSON.stringify(comp.toJSON());
    expect(text).toContain("sparked +48 heat");
    expect(text).toContain("cooled −10 heat"); // proper minus, never "+-10"
    expect(text).not.toContain("+-10");
    act(() => comp.unmount());
  });

  it("posts start_time/end_time when the turns carry them, and omits them when not", async () => {
    // Live-session turns: timing present.
    act(() => {
      useSessionStore.setState({
        turns: [
          { speaker: "Alice", text: "You never help.", start_time: 0, end_time: 1.4 },
          { speaker: "Bob", text: "I do plenty.", start_time: 1.1, end_time: 2.0 },
        ],
      });
    });
    mockAnalyze.mockResolvedValueOnce(fixture);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<DynamicsScreen onBack={() => {}} />);
    });
    expect(mockAnalyze.mock.calls[0][0]).toEqual([
      { speaker: "Alice", text: "You never help.", start_time: 0, end_time: 1.4 },
      { speaker: "Bob", text: "I do plenty.", start_time: 1.1, end_time: 2.0 },
    ]);
    act(() => comp.unmount());

    // Pasted-transcript flow (real parseTranscript path): no timing — the keys
    // must be absent entirely, never fabricated as 0 (the server then honestly
    // returns interruptions null).
    act(() => {
      useSessionStore.getState().loadTranscript("Alice: typed line");
    });
    mockAnalyze.mockResolvedValueOnce(fixture);
    await act(async () => {
      comp = renderer.create(<DynamicsScreen onBack={() => {}} />);
    });
    expect(mockAnalyze.mock.calls[1][0]).toEqual([
      { speaker: "Alice", text: "typed line" },
    ]);
    expect(mockAnalyze.mock.calls[1][0][0]).not.toHaveProperty("start_time");
    act(() => comp.unmount());
  });

  it("ignores a second invocation while a request is already in flight (one LLM call)", async () => {
    // First (mount) request rejects so the retry button appears.
    mockAnalyze.mockRejectedValueOnce(new Error("API error: 502"));
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<DynamicsScreen onBack={() => {}} />);
    });
    expect(mockAnalyze).toHaveBeenCalledTimes(1);

    // Retry now returns a promise we never resolve — the request stays pending.
    mockAnalyze.mockImplementation(() => new Promise(() => {}));
    const onPress = comp.root.find((n) => n.props?.testID === "dynamics-retry")
      .props.onPress as () => void;

    // Two rapid presses (StrictMode-style double invoke): the in-flight guard
    // must let only the FIRST one through — /analyze is a real, costed LLM call.
    await act(async () => {
      onPress();
      onPress();
    });
    expect(mockAnalyze).toHaveBeenCalledTimes(2); // 1 mount + 1 retry, NOT 3.
    act(() => comp.unmount());
  });

  it("omits the interruptions row when null but shows it when present", async () => {
    mockAnalyze.mockResolvedValueOnce(fixture);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<DynamicsScreen onBack={() => {}} />);
    });

    const text = JSON.stringify(comp.toJSON());
    // Bob has interruptions:2 -> shown; Alice is null -> the label appears once total.
    expect((text.match(/Interruptions/g) || []).length).toBe(1);
    act(() => comp.unmount());
  });

  it("shows an error state with a retry that re-calls postAnalyze", async () => {
    mockAnalyze.mockRejectedValueOnce(new Error("API error: 502"));
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<DynamicsScreen onBack={() => {}} />);
    });

    expect(queryId(comp, "dynamics-error")).toBeTruthy();
    expect(queryId(comp, "dynamics-content")).toBeNull();

    // Retry succeeds this time.
    mockAnalyze.mockResolvedValueOnce(fixture);
    const retry = comp.root.find((n) => n.props?.testID === "dynamics-retry");
    await act(async () => {
      retry.props.onPress();
    });

    expect(mockAnalyze).toHaveBeenCalledTimes(2);
    expect(queryId(comp, "dynamics-content")).toBeTruthy();
    expect(queryId(comp, "dynamics-error")).toBeNull();
    act(() => comp.unmount());
  });

  it("renders a report card per speaker with the absolute score", async () => {
    mockAnalyze.mockResolvedValueOnce(fixture);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<DynamicsScreen onBack={() => {}} />);
    });

    expect(queryId(comp, "report-card-Alice")).toBeTruthy();
    expect(queryId(comp, "report-card-Bob")).toBeTruthy();

    const text = JSON.stringify(comp.toJSON());
    // Scores shown plainly, out of 100, with the did-well / work-on lines.
    expect(text).toContain("72");
    expect(text).toContain("41");
    expect(text).toContain("/100");
    expect(text).toContain("Did well:");
    expect(text).toContain("Work on:");
    expect(text).toContain(fixture.report_cards!.Alice.headline);
    act(() => comp.unmount());
  });

  // Drive the full what-if flow: select a turn, run the counterfactual, and
  // inspect the resulting overlay + card.
  async function mountWithSim(): Promise<renderer.ReactTestRenderer> {
    mockAnalyze.mockResolvedValueOnce(fixture);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<DynamicsScreen onBack={() => {}} />);
    });
    // Drive the chart's onLayout so it measures width and can draw lines.
    const layoutNode = comp.root.findAll(
      (n) => typeof n.props?.onLayout === "function",
    )[0];
    act(() => {
      layoutNode.props.onLayout({ nativeEvent: { layout: { width: 300, height: 180 } } });
    });
    // Select turn 1 (the pivot) via the scrubber.
    act(() => comp.root.find((n) => n.props?.testID === "scrub-1").props.onPress());
    return comp;
  }

  function countIds(comp: renderer.ReactTestRenderer, prefix: string): number {
    const ids = new Set<string>();
    comp.root
      .findAll(
        (n) => typeof n.props?.testID === "string" && n.props.testID.startsWith(prefix),
      )
      .forEach((n) => ids.add(n.props.testID as string));
    return ids.size;
  }

  it("runs a counterfactual and shows dashed overlay lines + the rewritten card and disclaimer verbatim", async () => {
    const comp = await mountWithSim();
    mockCounterfactual.mockResolvedValueOnce(simFixture);

    // Tap the what-if button inside the inspector.
    await act(async () => {
      comp.root.find((n) => n.props?.testID === "what-if-button").props.onPress();
    });

    // The counterfactual was called with the full turns and the selected pivot.
    expect(mockCounterfactual).toHaveBeenCalledTimes(1);
    expect(mockCounterfactual.mock.calls[0][0]).toHaveLength(4);
    expect(mockCounterfactual.mock.calls[0][1]).toBe(1);

    // Two dashed overlay lines (Bob + Alice across pivot→end).
    expect(countIds(comp, "sim-line-")).toBe(2);
    // Real lines remain.
    expect(countIds(comp, "heat-line-")).toBe(2);

    // The "What if" card shows the rewritten text, rationale, and disclaimer
    // verbatim.
    expect(queryId(comp, "what-if-card")).toBeTruthy();
    const text = JSON.stringify(comp.toJSON());
    expect(text).toContain(simFixture.rewritten_text);
    expect(text).toContain(simFixture.rationale);
    expect(text).toContain(simFixture.disclaimer);
    act(() => comp.unmount());
  });

  it("toggles the overlay off and on without refetching", async () => {
    const comp = await mountWithSim();
    mockCounterfactual.mockResolvedValueOnce(simFixture);
    await act(async () => {
      comp.root.find((n) => n.props?.testID === "what-if-button").props.onPress();
    });
    expect(countIds(comp, "sim-line-")).toBe(2);

    // Toggle off — overlay + card disappear, no new network call.
    act(() =>
      comp.root.find((n) => n.props?.testID === "simulation-toggle").props.onPress(),
    );
    expect(countIds(comp, "sim-line-")).toBe(0);
    expect(queryId(comp, "what-if-card")).toBeNull();
    expect(mockCounterfactual).toHaveBeenCalledTimes(1); // no refetch

    // Toggle back on — overlay returns, still one fetch.
    act(() =>
      comp.root.find((n) => n.props?.testID === "simulation-toggle").props.onPress(),
    );
    expect(countIds(comp, "sim-line-")).toBe(2);
    expect(mockCounterfactual).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });

  it("shows an honest inline error with a retry that re-runs the counterfactual", async () => {
    const comp = await mountWithSim();
    mockCounterfactual.mockRejectedValueOnce(new Error("API error: 429"));
    await act(async () => {
      comp.root.find((n) => n.props?.testID === "what-if-button").props.onPress();
    });

    // Error surfaced inline, no overlay/card, no fabricated projection.
    expect(queryId(comp, "what-if-error")).toBeTruthy();
    expect(countIds(comp, "sim-line-")).toBe(0);
    expect(queryId(comp, "what-if-card")).toBeNull();
    const errText = JSON.stringify(comp.toJSON());
    expect(errText).toContain("API error: 429");

    // Retry succeeds — overlay appears.
    mockCounterfactual.mockResolvedValueOnce(simFixture);
    await act(async () => {
      comp.root.find((n) => n.props?.testID === "what-if-retry").props.onPress();
    });
    expect(mockCounterfactual).toHaveBeenCalledTimes(2);
    expect(countIds(comp, "sim-line-")).toBe(2);
    expect(queryId(comp, "what-if-error")).toBeNull();
    act(() => comp.unmount());
  });

  it("renders directly from initialData without calling /analyze (upload flow)", async () => {
    // The upload flow loads the transcript into the store and hands the analysis
    // over as initialData — Dynamics must render it, never re-fetch.
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <DynamicsScreen onBack={() => {}} initialData={fixture} />,
      );
    });

    expect(mockAnalyze).not.toHaveBeenCalled();
    expect(queryId(comp, "dynamics-loading")).toBeNull();
    expect(queryId(comp, "dynamics-content")).toBeTruthy();
    expect(queryId(comp, "speaker-card-Alice")).toBeTruthy();
    expect(queryId(comp, "speaker-card-Bob")).toBeTruthy();
    expect(queryId(comp, "dynamics-narrative")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("shows the Replay button for a stored recording and calls onReplay with the id", async () => {
    // Upload flow: consent+store landed, so App threads the recording id in
    // alongside the ready-made analysis. The Replay entry point must render and
    // hand that exact id back through onReplay.
    const onReplay = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <DynamicsScreen
          onBack={() => {}}
          initialData={fixture}
          recordingId="rec_42"
          onReplay={onReplay}
        />,
      );
    });

    const btn = queryId(comp, "replay-recording-button");
    expect(btn).toBeTruthy();
    act(() => {
      btn!.props.onPress();
    });
    expect(onReplay).toHaveBeenCalledWith("rec_42");
    act(() => comp.unmount());
  });

  it("falls back to the analysis's own recording_id when no recordingId prop was threaded", async () => {
    // The screen's fetched/handed-over context can itself carry recording_id
    // (an UploadAnalyzeResult) — EITHER source enables the Replay entry point.
    const onReplay = jest.fn();
    const uploadShaped = {
      ...fixture,
      turns: [],
      stored: true,
      recording_id: "rec_ctx",
      storage_note: null,
    };
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <DynamicsScreen
          onBack={() => {}}
          initialData={uploadShaped}
          onReplay={onReplay}
        />,
      );
    });

    const btn = queryId(comp, "replay-recording-button");
    expect(btn).toBeTruthy();
    act(() => {
      btn!.props.onPress();
    });
    expect(onReplay).toHaveBeenCalledWith("rec_ctx");
    act(() => comp.unmount());
  });

  it("hides the Replay button when no recording backs the analysis", async () => {
    // Plain transcript analysis: no recordingId prop, no recording_id on the
    // result — nothing to replay, so no button (never a dead affordance).
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <DynamicsScreen
          onBack={() => {}}
          initialData={fixture}
          onReplay={() => {}}
        />,
      );
    });
    expect(queryId(comp, "replay-recording-button")).toBeNull();
    act(() => comp.unmount());
  });

  it("shows the voice_analysis note when the upload result carries one", async () => {
    const withNote = {
      ...fixture,
      voice_analysis: "Voice tone couldn’t be measured for this recording.",
    };
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <DynamicsScreen onBack={() => {}} initialData={withNote} />,
      );
    });
    expect(queryId(comp, "voice-analysis-note")).toBeTruthy();
    expect(JSON.stringify(comp.toJSON())).toContain(
      "Voice tone couldn’t be measured",
    );
    act(() => comp.unmount());
  });

  it("omits the voice_analysis note when absent (normal fetch flow)", async () => {
    mockAnalyze.mockResolvedValueOnce(fixture);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<DynamicsScreen onBack={() => {}} />);
    });
    expect(queryId(comp, "voice-analysis-note")).toBeNull();
    act(() => comp.unmount());
  });

  it("replaces the overlay when a different turn is used as the pivot", async () => {
    const comp = await mountWithSim();
    mockCounterfactual.mockResolvedValueOnce(simFixture);
    await act(async () => {
      comp.root.find((n) => n.props?.testID === "what-if-button").props.onPress();
    });
    expect(countIds(comp, "sim-line-")).toBe(2);

    // Select a different turn (3) and run again — overlay replaced by the new
    // single-speaker projection.
    const secondSim: CounterfactualResult = {
      ...simFixture,
      pivot_index: 3,
      simulated_per_turn: [{ index: 3, speaker: "Bob", heat: 50 }],
    };
    act(() => comp.root.find((n) => n.props?.testID === "scrub-3").props.onPress());
    mockCounterfactual.mockResolvedValueOnce(secondSim);
    await act(async () => {
      comp.root.find((n) => n.props?.testID === "what-if-button").props.onPress();
    });
    expect(mockCounterfactual).toHaveBeenLastCalledWith(expect.any(Array), 3);
    // Only Bob is in the new projection -> one dashed line.
    expect(countIds(comp, "sim-line-")).toBe(1);
    act(() => comp.unmount());
  });
});

describe("DynamicsScreen HD-later popup", () => {
  // initialData short-circuits the on-mount fetch, so these render content
  // directly. The popup only makes sense for a stored, recorder-origin analysis.
  it("shows the popup only for a recorder-origin analysis that was stored", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <DynamicsScreen
          onBack={() => {}}
          initialData={fixture}
          recordingId="rec_1"
          cameFromRecorder
          onAttachSource={() => {}}
        />,
      );
    });
    expect(queryId(comp, "hd-suggest-popup")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("hides the popup when the analysis did not come from the recorder", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <DynamicsScreen
          onBack={() => {}}
          initialData={fixture}
          recordingId="rec_1"
          cameFromRecorder={false}
          onAttachSource={() => {}}
        />,
      );
    });
    expect(queryId(comp, "hd-suggest-popup")).toBeNull();
    act(() => comp.unmount());
  });

  it("hides the popup when nothing was stored (no recording id)", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <DynamicsScreen
          onBack={() => {}}
          initialData={fixture}
          recordingId={null}
          cameFromRecorder
          onAttachSource={() => {}}
        />,
      );
    });
    expect(queryId(comp, "hd-suggest-popup")).toBeNull();
    act(() => comp.unmount());
  });

  it("Later dismisses the popup", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <DynamicsScreen
          onBack={() => {}}
          initialData={fixture}
          recordingId="rec_1"
          cameFromRecorder
          onAttachSource={() => {}}
        />,
      );
    });
    expect(queryId(comp, "hd-suggest-popup")).toBeTruthy();
    act(() => queryId(comp, "hd-suggest-later")!.props.onPress());
    expect(queryId(comp, "hd-suggest-popup")).toBeNull();
    act(() => comp.unmount());
  });

  it("Attach link now jumps to the attach flow for this recording id", () => {
    const onAttachSource = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <DynamicsScreen
          onBack={() => {}}
          initialData={fixture}
          recordingId="rec_1"
          cameFromRecorder
          onAttachSource={onAttachSource}
        />,
      );
    });
    act(() => queryId(comp, "hd-suggest-attach")!.props.onPress());
    expect(onAttachSource).toHaveBeenCalledWith("rec_1");
    // And it dismisses on the way out.
    expect(queryId(comp, "hd-suggest-popup")).toBeNull();
    act(() => comp.unmount());
  });
});

describe("DynamicsScreen glanceable summary + word metrics", () => {
  it("renders the glanceable summary above the detailed chart", async () => {
    mockAnalyze.mockResolvedValueOnce(fixture);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<DynamicsScreen onBack={() => {}} />);
    });
    expect(queryId(comp, "glance-summary")).toBeTruthy();
    expect(queryId(comp, "glance-verdict")).toBeTruthy();
    expect(queryId(comp, "glance-row-Alice")).toBeTruthy();
    expect(queryId(comp, "glance-row-Bob")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("hides the word-patterns panel when the analysis has no word_metrics", async () => {
    mockAnalyze.mockResolvedValueOnce(fixture);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<DynamicsScreen onBack={() => {}} />);
    });
    expect(queryId(comp, "word-patterns-panel")).toBeNull();
    act(() => comp.unmount());
  });

  it("shows the word-patterns panel when word_metrics is present", async () => {
    const withMetrics = {
      ...fixture,
      word_metrics: {
        speakers: {
          Alice: {
            i_rate: 9, you_rate: 2, we_rate: 1,
            anger_rate: 0.3, fear_rate: 0.1, sadness_rate: 0.5, joy_rate: 1.2, trust_rate: 0.8,
            word_count: 210,
          },
          Bob: {
            i_rate: 4, you_rate: 8, we_rate: 0,
            anger_rate: 1.1, fear_rate: 0.2, sadness_rate: 0.3, joy_rate: 0.4, trust_rate: 0.2,
            word_count: 190,
          },
        },
        method: { description: "Counted per 100 words with a deterministic lexicon." },
      },
    };
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <DynamicsScreen onBack={() => {}} initialData={withMetrics} />,
      );
    });
    expect(queryId(comp, "word-patterns-panel")).toBeTruthy();
    act(() => comp.unmount());
  });
});
