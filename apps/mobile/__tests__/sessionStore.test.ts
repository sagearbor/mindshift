import { useSessionStore } from "../src/store/sessionStore";

// Reset the store to a known baseline before each test.
beforeEach(() => {
  useSessionStore.setState({
    role: "Husband / Wife",
    empathyLevel: 50,
    turns: [],
    suggestions: [],
    loading: false,
  });
});

describe("sessionStore.loadTurns", () => {
  it("sets turns directly and clears any stale suggestions", () => {
    // Seed stale state from a previous review.
    useSessionStore.setState({
      turns: [{ speaker: "Old", text: "old line" }],
      suggestions: [{ text: "stale advice", tone: "balanced" }],
    });

    useSessionStore.getState().loadTurns([
      { speaker: "Speaker A", text: "You never listen." },
      { speaker: "Speaker B", text: "I'm trying my best." },
    ]);

    const { turns, suggestions } = useSessionStore.getState();
    expect(turns).toEqual([
      { speaker: "Speaker A", text: "You never listen." },
      { speaker: "Speaker B", text: "I'm trying my best." },
    ]);
    // The handoff starts a fresh review — old suggestions must not linger.
    expect(suggestions).toEqual([]);
  });

  it("maps to the store's Turn shape, dropping fields outside the Turn contract", () => {
    // Live transcript entries carry a wall-clock timestamp the store's Turn
    // does not — it must not leak through.
    useSessionStore.getState().loadTurns([
      { speaker: "Speaker A", text: "Hello.", timestamp: 123 } as never,
    ]);

    expect(useSessionStore.getState().turns).toEqual([
      { speaker: "Speaker A", text: "Hello." },
    ]);
  });

  it("preserves start_time/end_time when the source turns carry them", () => {
    // A live session hands off timed turns — timing must survive the store so
    // /analyze can compute real interruption stats.
    useSessionStore.getState().loadTurns([
      { speaker: "Speaker A", text: "You never listen.", start_time: 0, end_time: 1.4 },
      { speaker: "Speaker B", text: "I'm trying.", start_time: 1.2, end_time: 2.3 },
    ]);

    expect(useSessionStore.getState().turns).toEqual([
      { speaker: "Speaker A", text: "You never listen.", start_time: 0, end_time: 1.4 },
      { speaker: "Speaker B", text: "I'm trying.", start_time: 1.2, end_time: 2.3 },
    ]);
  });

  it("leaves timing keys absent (not 0) for untimed turns", () => {
    useSessionStore.getState().loadTurns([
      { speaker: "Speaker A", text: "Untimed." },
    ]);
    const [turn] = useSessionStore.getState().turns;
    expect(turn).not.toHaveProperty("start_time");
    expect(turn).not.toHaveProperty("end_time");
  });
});
