import { postRespond, postAnalyze, empathyTone } from "../src/api/client";
import {
  setCachedToken,
  setTokenProvider,
} from "../src/auth/authToken";

const mockFetch = global.fetch as jest.Mock;

beforeEach(() => {
  mockFetch.mockReset();
  // Reset auth token state between tests (module-level singleton).
  setCachedToken(null);
  setTokenProvider(null);
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

    // Signed out (no token set): no Authorization header is attached.
    expect(init.headers).not.toHaveProperty("Authorization");

    // Bare strings become {text, tone}; tone derives from the slider.
    expect(result.suggestions).toEqual([
      { text: "I hear you.", tone: "validating" },
      { text: "Tell me more.", tone: "validating" },
    ]);
    expect(result.toneScore).toEqual({ warmth: 70, defensiveness: 10 });
  });

  it("attaches the Firebase ID token as an Authorization: Bearer header", async () => {
    setTokenProvider(async () => "fresh-id-token-abc");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ suggestions: [], tone_score: {} }),
    });

    await postRespond({
      transcript_turn: "hi",
      role: "Husband / Wife",
      empathy_slider: 50,
    });

    const [, init] = mockFetch.mock.calls[0];
    expect(init.headers.Authorization).toBe("Bearer fresh-id-token-abc");
    expect(init.headers["Content-Type"]).toBe("application/json");
  });

  it("falls back to the cached token when no provider is registered", async () => {
    setCachedToken("cached-token-xyz");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ suggestions: [], tone_score: {} }),
    });

    await postRespond({
      transcript_turn: "hi",
      role: "r",
      empathy_slider: 50,
    });

    const [, init] = mockFetch.mock.calls[0];
    expect(init.headers.Authorization).toBe("Bearer cached-token-xyz");
  });

  it("throws on a non-OK response", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 502 });
    await expect(
      postRespond({ transcript_turn: "x", role: "r", empathy_slider: 50 }),
    ).rejects.toThrow("API error: 502");
  });
});

describe("postAnalyze", () => {
  const smallResult = {
    per_turn: [],
    per_speaker: {},
    dynamics: {
      coupling: { strength: null, leader: null, description: "" },
      deescalation: { who_first: null, follow_rate: null, description: "" },
      triggers: [],
      requests: [],
    },
    narrative: "",
  };

  it("POSTs turns to /analyze and returns the parsed result", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => smallResult });

    const turns = [
      { speaker: "Alice", text: "You never listen." },
      { speaker: "Bob", text: "That's not fair." },
    ];
    const result = await postAnalyze(turns);

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/analyze$/);
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body);
    // Turns are sent verbatim; no context key when none is passed.
    expect(body).toEqual({ turns });
    expect(body).not.toHaveProperty("context");
    expect(result).toEqual(smallResult);
  });

  it("includes context only when provided", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => smallResult });
    await postAnalyze([{ speaker: "A", text: "hi" }], "earlier stuff");
    const [, init] = mockFetch.mock.calls[0];
    expect(JSON.parse(init.body).context).toBe("earlier stuff");
  });

  it("attaches the Firebase ID token as a Bearer header", async () => {
    setTokenProvider(async () => "analyze-token");
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => smallResult });
    await postAnalyze([{ speaker: "A", text: "hi" }]);
    const [, init] = mockFetch.mock.calls[0];
    expect(init.headers.Authorization).toBe("Bearer analyze-token");
  });

  it("throws an honest error on a non-OK response (no fabricated result)", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 429 });
    await expect(postAnalyze([{ speaker: "A", text: "hi" }])).rejects.toThrow(
      "API error: 429",
    );
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
