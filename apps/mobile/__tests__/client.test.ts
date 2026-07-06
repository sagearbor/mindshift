import { postRespond, empathyTone } from "../src/api/client";

const mockFetch = global.fetch as jest.Mock;

beforeEach(() => {
  mockFetch.mockReset();
});

describe("postRespond", () => {
  it("sends the server's field names and maps string suggestions to {text, tone}", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        suggestions: ["I hear you.", "Tell me more."],
        tone_score: { warmth: 70, defensiveness: 10 },
      }),
    });

    const result = await postRespond({
      transcript_turn: "You never listen.",
      role: "Husband / Wife",
      empathy_slider: 90,
      context: "Alice: hi\nBob: hey",
    });

    // Request body matches the server's RespondRequest, not the old shape.
    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/respond$/);
    const body = JSON.parse(init.body);
    expect(body).toEqual({
      transcript_turn: "You never listen.",
      role: "Husband / Wife",
      empathy_slider: 90,
      context: "Alice: hi\nBob: hey",
    });
    expect(body).not.toHaveProperty("turns");
    expect(body).not.toHaveProperty("empathy_level");

    // Bare strings become {text, tone}; tone derives from the slider.
    expect(result.suggestions).toEqual([
      { text: "I hear you.", tone: "validating" },
      { text: "Tell me more.", tone: "validating" },
    ]);
    expect(result.toneScore).toEqual({ warmth: 70, defensiveness: 10 });
  });

  it("throws on a non-OK response", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 502 });
    await expect(
      postRespond({ transcript_turn: "x", role: "r", empathy_slider: 50 }),
    ).rejects.toThrow("API error: 502");
  });
});

describe("empathyTone", () => {
  it("maps slider ranges to coaching stances", () => {
    expect(empathyTone(0)).toBe("assertive");
    expect(empathyTone(20)).toBe("assertive");
    expect(empathyTone(50)).toBe("balanced");
    expect(empathyTone(80)).toBe("empathetic");
    expect(empathyTone(100)).toBe("validating");
  });
});
