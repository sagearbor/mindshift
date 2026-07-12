import {
  postRespond,
  postAnalyze,
  postAnalyzeUpload,
  postCounterfactual,
  empathyTone,
} from "../src/api/client";
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
    report_cards: {},
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

describe("postCounterfactual", () => {
  const simResult = {
    pivot_index: 3,
    rewritten_text: "I feel unseen when the chores pile up.",
    rationale: "A softened bid invites repair instead of defense.",
    simulated_per_turn: [
      { index: 3, speaker: "Bob", heat: 40 },
      { index: 4, speaker: "Alice", heat: 30 },
    ],
    disclaimer: "A hypothetical projection, not a prediction.",
  };

  const turns = [
    { speaker: "Alice", text: "You never listen." },
    { speaker: "Bob", text: "That's not fair." },
  ];

  it("POSTs turns + pivot_index to /analyze/counterfactual and returns the result", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => simResult });

    const result = await postCounterfactual(turns, 3);

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/analyze\/counterfactual$/);
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body);
    // Turns verbatim, pivot_index in snake_case, no context key when unset.
    expect(body).toEqual({ turns, pivot_index: 3 });
    expect(body).not.toHaveProperty("context");
    expect(result).toEqual(simResult);
  });

  it("includes context only when provided", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => simResult });
    await postCounterfactual(turns, 1, "earlier context");
    const [, init] = mockFetch.mock.calls[0];
    const body = JSON.parse(init.body);
    expect(body.context).toBe("earlier context");
    expect(body.pivot_index).toBe(1);
  });

  it("attaches the Firebase ID token as a Bearer header", async () => {
    setTokenProvider(async () => "cf-token");
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => simResult });
    await postCounterfactual(turns, 0);
    const [, init] = mockFetch.mock.calls[0];
    expect(init.headers.Authorization).toBe("Bearer cf-token");
    expect(init.headers["Content-Type"]).toBe("application/json");
  });

  it("omits the Authorization header when signed out", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => simResult });
    await postCounterfactual(turns, 0);
    const [, init] = mockFetch.mock.calls[0];
    expect(init.headers).not.toHaveProperty("Authorization");
  });

  it("throws an honest error on a non-OK response (no fabricated result)", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 422 });
    await expect(postCounterfactual(turns, 0)).rejects.toThrow("API error: 422");
  });
});

describe("postAnalyzeUpload", () => {
  // A recorder standing in for FormData so we can inspect the multipart parts
  // without depending on RN FormData internals.
  class RecordingFormData {
    entries: [string, unknown][] = [];
    append(name: string, value: unknown) {
      this.entries.push([name, value]);
    }
  }
  const RealFormData = global.FormData;
  beforeEach(() => {
    (global as { FormData: unknown }).FormData = RecordingFormData;
  });
  afterEach(() => {
    (global as { FormData: unknown }).FormData = RealFormData;
  });

  const uploadResult = {
    per_turn: [],
    per_speaker: {},
    dynamics: {
      coupling: { strength: null, leader: null, description: "" },
      deescalation: { who_first: null, follow_rate: null, description: "" },
      triggers: [],
      requests: [],
    },
    narrative: "",
    turns: [
      { speaker: "Alice", text: "hi", start_time: 0, end_time: 1 },
    ],
    stored: false,
    recording_id: null,
    storage_note: "Storage requires consent.",
  };

  it("builds a multipart FormData with the native file part + context and a Bearer header", async () => {
    setTokenProvider(async () => "up-token");
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => uploadResult });

    const result = await postAnalyzeUpload(
      "file:///rec.m4a",
      "rec.m4a",
      "audio/m4a",
      "kitchen argument",
    );

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/analyze\/upload$/);
    expect(init.method).toBe("POST");

    // The file part is RN's { uri, name, type } descriptor.
    const body = init.body as InstanceType<typeof RecordingFormData>;
    const fileEntry = body.entries.find((e) => e[0] === "file");
    expect(fileEntry?.[1]).toEqual({
      uri: "file:///rec.m4a",
      name: "rec.m4a",
      type: "audio/m4a",
    });
    // Context is appended verbatim.
    expect(body.entries.find((e) => e[0] === "context")?.[1]).toBe(
      "kitchen argument",
    );
    // Defaults (no options passed): consent false, store true — sent as strings.
    expect(body.entries.find((e) => e[0] === "consent")?.[1]).toBe("false");
    expect(body.entries.find((e) => e[0] === "store")?.[1]).toBe("true");

    // Bearer auth present; Content-Type NOT set manually (fetch must add the
    // multipart boundary itself).
    expect(init.headers.Authorization).toBe("Bearer up-token");
    expect(init.headers["Content-Type"]).toBeUndefined();

    expect(result).toEqual(uploadResult);
  });

  it("omits the context part when none is provided", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => uploadResult });
    await postAnalyzeUpload("file:///rec.m4a", "rec.m4a", "audio/m4a");
    const body = mockFetch.mock.calls[0][1].body as InstanceType<
      typeof RecordingFormData
    >;
    expect(body.entries.some((e) => e[0] === "context")).toBe(false);
    // Signed out: no Authorization header.
    expect(mockFetch.mock.calls[0][1].headers).not.toHaveProperty(
      "Authorization",
    );
  });

  it("serializes consent/store as literal 'true'/'false' strings when provided", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ...uploadResult, stored: true, recording_id: "rec_1", storage_note: null }),
    });
    await postAnalyzeUpload(
      "file:///rec.m4a",
      "rec.m4a",
      "audio/m4a",
      undefined,
      { consent: true, store: true },
    );
    const body = mockFetch.mock.calls[0][1].body as InstanceType<
      typeof RecordingFormData
    >;
    expect(body.entries.find((e) => e[0] === "consent")?.[1]).toBe("true");
    expect(body.entries.find((e) => e[0] === "store")?.[1]).toBe("true");
  });

  it("sends store: false when consent is given but storage is declined", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => uploadResult });
    await postAnalyzeUpload(
      "file:///rec.m4a",
      "rec.m4a",
      "audio/m4a",
      undefined,
      { consent: true, store: false },
    );
    const body = mockFetch.mock.calls[0][1].body as InstanceType<
      typeof RecordingFormData
    >;
    expect(body.entries.find((e) => e[0] === "consent")?.[1]).toBe("true");
    expect(body.entries.find((e) => e[0] === "store")?.[1]).toBe("false");
  });

  it("throws an honest error on a non-OK response (e.g. 503 transcription unconfigured)", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503 });
    await expect(
      postAnalyzeUpload("file:///rec.m4a", "rec.m4a", "audio/m4a"),
    ).rejects.toThrow("API error: 503");
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
