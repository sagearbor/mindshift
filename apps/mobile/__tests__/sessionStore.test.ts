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

  it("maps to the store's Turn shape, keeping only speaker + text", () => {
    // Live transcript entries carry a timestamp the store's Turn does not.
    useSessionStore.getState().loadTurns([
      { speaker: "Speaker A", text: "Hello.", timestamp: 123 } as never,
    ]);

    expect(useSessionStore.getState().turns).toEqual([
      { speaker: "Speaker A", text: "Hello." },
    ]);
  });
});
