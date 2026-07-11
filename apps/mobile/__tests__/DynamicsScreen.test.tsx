import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import DynamicsScreen from "../src/screens/DynamicsScreen";
import { useSessionStore } from "../src/store/sessionStore";
import { postAnalyze } from "../src/api/client";
import type { AnalyzeResult } from "../src/api/client";

// Mock only postAnalyze; types are erased so the rest of the module is unused.
jest.mock("../src/api/client", () => ({
  postAnalyze: jest.fn(),
}));
const mockAnalyze = postAnalyze as jest.Mock;

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

beforeEach(() => {
  mockAnalyze.mockReset();
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
});
