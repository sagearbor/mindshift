import {
  postRespond,
  postAnalyze,
  postAnalyzeUpload,
  postAnalyzeUploadChunked,
  postAnalyzeLink,
  postCounterfactual,
  empathyTone,
  listRecordings,
  getRecording,
  getRecordingMediaUrl,
  getRecordingSourceUrl,
  patchRecordingSource,
  deleteRecording,
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

describe("postAnalyzeUploadChunked", () => {
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
    turns: [{ speaker: "Alice", text: "hi", start_time: 0, end_time: 1 }],
    stored: true,
    recording_id: "rec_9",
    storage_note: null,
  };

  // Fill a Uint8Array of `n` bytes with a recognizable ramp so we can assert the
  // right slice went out on each chunk PUT.
  const ramp = (n: number) => {
    const b = new Uint8Array(n);
    for (let i = 0; i < n; i += 1) b[i] = i % 256;
    return b;
  };

  afterEach(() => {
    delete (globalThis as Record<string, unknown>).__fsMockBytes;
  });

  it("starts, PUTs each chunk (right index/size/body), completes, and reports progress", async () => {
    setTokenProvider(async () => "chunk-token");
    (globalThis as Record<string, unknown>).__fsMockBytes = ramp(250);

    mockFetch
      // POST /uploads/start
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          upload_id: "up1",
          chunk_bytes: 100,
          expected_chunks: 3,
        }),
      })
      // PUT chunks 0,1,2
      .mockResolvedValueOnce({ ok: true, status: 204 })
      .mockResolvedValueOnce({ ok: true, status: 204 })
      .mockResolvedValueOnce({ ok: true, status: 204 })
      // POST /uploads/up1/complete
      .mockResolvedValueOnce({ ok: true, json: async () => uploadResult });

    const progress: number[] = [];
    const result = await postAnalyzeUploadChunked(
      "file:///big.mp4",
      "big.mp4",
      "video/mp4",
      250,
      {
        consent: true,
        store: true,
        context: "kitchen argument",
        onProgress: (f) => progress.push(f),
      },
    );

    // --- start: JSON body with real booleans + context, Bearer auth ---
    const [startUrl, startInit] = mockFetch.mock.calls[0];
    expect(startUrl).toMatch(/\/uploads\/start$/);
    expect(startInit.method).toBe("POST");
    expect(startInit.headers.Authorization).toBe("Bearer chunk-token");
    expect(JSON.parse(startInit.body)).toEqual({
      filename: "big.mp4",
      content_type: "video/mp4",
      total_bytes: 250,
      consent: true,
      store: true,
      context: "kitchen argument",
    });

    // --- three chunk PUTs at the right indexes, sizes, and bytes ---
    const chunk0 = mockFetch.mock.calls[1];
    expect(chunk0[0]).toMatch(/\/uploads\/up1\/chunks\/0$/);
    expect(chunk0[1].method).toBe("PUT");
    expect(chunk0[1].headers["Content-Type"]).toBe("application/octet-stream");
    expect(chunk0[1].headers.Authorization).toBe("Bearer chunk-token");
    expect(chunk0[1].body).toBeInstanceOf(Uint8Array);
    expect((chunk0[1].body as Uint8Array).length).toBe(100);
    expect((chunk0[1].body as Uint8Array)[0]).toBe(0);

    const chunk1 = mockFetch.mock.calls[2];
    expect(chunk1[0]).toMatch(/\/uploads\/up1\/chunks\/1$/);
    expect((chunk1[1].body as Uint8Array).length).toBe(100);
    // Second chunk starts at byte 100 → value 100 % 256.
    expect((chunk1[1].body as Uint8Array)[0]).toBe(100);

    const chunk2 = mockFetch.mock.calls[3];
    expect(chunk2[0]).toMatch(/\/uploads\/up1\/chunks\/2$/);
    // Final short chunk: 250 - 200 = 50 bytes.
    expect((chunk2[1].body as Uint8Array).length).toBe(50);

    // --- complete ---
    const [completeUrl, completeInit] = mockFetch.mock.calls[4];
    expect(completeUrl).toMatch(/\/uploads\/up1\/complete$/);
    expect(completeInit.method).toBe("POST");

    // Progress fired once per chunk, ending at 1.
    expect(progress).toEqual([1 / 3, 2 / 3, 1]);
    expect(result).toEqual(uploadResult);
  });

  it("omits context and sends consent/store as real JSON booleans when off", async () => {
    (globalThis as Record<string, unknown>).__fsMockBytes = ramp(10);
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          upload_id: "u3",
          chunk_bytes: 100,
          expected_chunks: 1,
        }),
      })
      .mockResolvedValueOnce({ ok: true, status: 204 })
      .mockResolvedValueOnce({ ok: true, json: async () => uploadResult });

    await postAnalyzeUploadChunked("file:///s.m4a", "s.m4a", "audio/m4a", 10, {
      consent: false,
      store: false,
    });

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body).toEqual({
      filename: "s.m4a",
      content_type: "audio/m4a",
      total_bytes: 10,
      consent: false,
      store: false,
    });
    expect(body).not.toHaveProperty("context");
    // Single chunk carries the whole (short) file.
    expect((mockFetch.mock.calls[1][1].body as Uint8Array).length).toBe(10);
    // Signed out: no Authorization header on start.
    expect(mockFetch.mock.calls[0][1].headers).not.toHaveProperty(
      "Authorization",
    );
  });

  it("aborts with a DELETE and throws the honest error when a chunk PUT fails", async () => {
    (globalThis as Record<string, unknown>).__fsMockBytes = ramp(150);
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          upload_id: "up2",
          chunk_bytes: 100,
          expected_chunks: 2,
        }),
      })
      .mockResolvedValueOnce({ ok: true, status: 204 }) // chunk 0 OK
      .mockResolvedValueOnce({ ok: false, status: 413 }) // chunk 1 rejected
      .mockResolvedValueOnce({ ok: true, status: 204 }); // DELETE abort

    await expect(
      postAnalyzeUploadChunked("file:///x.mp4", "x.mp4", "video/mp4", 150, {
        consent: true,
        store: true,
      }),
    ).rejects.toThrow("API error: 413");

    // A best-effort abort was issued for the started upload.
    const del = mockFetch.mock.calls[3];
    expect(del[0]).toMatch(/\/uploads\/up2$/);
    expect(del[1].method).toBe("DELETE");
    // The completion call never fired (5th call would be it).
    expect(mockFetch).toHaveBeenCalledTimes(4);
  });

  it("throws without aborting when /uploads/start itself fails (nothing to abort)", async () => {
    (globalThis as Record<string, unknown>).__fsMockBytes = ramp(50);
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503 });
    await expect(
      postAnalyzeUploadChunked("file:///x.mp4", "x.mp4", "video/mp4", 50, {
        consent: true,
        store: true,
      }),
    ).rejects.toThrow("API error: 503");
    // Only the start call was made — no chunk PUTs, no DELETE.
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });
});

describe("postAnalyzeLink", () => {
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
    turns: [{ speaker: "Alice", text: "hi", start_time: 0, end_time: 1 }],
    stored: false,
    recording_id: null,
    storage_note: "Storage requires consent.",
  };

  it("POSTs the url + JSON booleans (and context) with a Bearer header", async () => {
    setTokenProvider(async () => "link-token");
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => uploadResult });

    const result = await postAnalyzeLink("https://drive.google.com/file/d/abc", {
      consent: true,
      store: true,
      context: "date-night recap",
    });

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/analyze\/link$/);
    expect(init.method).toBe("POST");
    expect(init.headers.Authorization).toBe("Bearer link-token");
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(init.body)).toEqual({
      url: "https://drive.google.com/file/d/abc",
      consent: true,
      store: true,
      context: "date-night recap",
    });
    expect(result).toEqual(uploadResult);
  });

  it("omits context when not provided and can be signed out", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => uploadResult });
    await postAnalyzeLink("https://example.com/rec.mp4", {
      consent: false,
      store: false,
    });
    const [, init] = mockFetch.mock.calls[0];
    const body = JSON.parse(init.body);
    expect(body).toEqual({
      url: "https://example.com/rec.mp4",
      consent: false,
      store: false,
    });
    expect(body).not.toHaveProperty("context");
    expect(init.headers).not.toHaveProperty("Authorization");
  });

  it("surfaces the server's user-facing 422 message verbatim (detail + status)", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 422,
      json: async () => ({
        detail:
          "That link isn’t a direct file link — use a direct file URL, a Google Drive share link, or a Google Photos share link of a single video.",
      }),
    });
    await expect(
      postAnalyzeLink("https://example.com/share/xyz", {
        consent: false,
        store: false,
      }),
    ).rejects.toMatchObject({
      status: 422,
      detail:
        "That link isn’t a direct file link — use a direct file URL, a Google Drive share link, or a Google Photos share link of a single video.",
      message:
        "That link isn’t a direct file link — use a direct file URL, a Google Drive share link, or a Google Photos share link of a single video.",
    });
  });

  it("falls back to an `API error: <status>` message when the body has no detail", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => ({}),
    });
    await expect(
      postAnalyzeLink("https://example.com/rec.mp4", {
        consent: false,
        store: false,
      }),
    ).rejects.toThrow("API error: 503");
  });
});

describe("listRecordings", () => {
  it("GETs /recordings with a Bearer token and unwraps the recordings array", async () => {
    setTokenProvider(async () => "rec-token");
    const recordings = [
      {
        id: "r1",
        created_at: "2026-07-01T10:00:00Z",
        filename: "kitchen-fight.m4a",
        media_type: "audio",
        duration_seconds: 182,
        has_analysis: true,
      },
    ];
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ recordings }),
    });

    const result = await listRecordings();

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/recordings$/);
    expect(init.method).toBe("GET");
    expect(init.headers.Authorization).toBe("Bearer rec-token");
    expect(result).toEqual(recordings);
  });

  it("returns [] when the payload has no recordings key", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => ({}) });
    expect(await listRecordings()).toEqual([]);
  });

  it("throws an honest error on 503 (storage not configured)", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503 });
    await expect(listRecordings()).rejects.toThrow("API error: 503");
  });

  it("omits the Authorization header when signed out", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ recordings: [] }),
    });
    await listRecordings();
    const [, init] = mockFetch.mock.calls[0];
    expect(init.headers).not.toHaveProperty("Authorization");
  });
});

describe("getRecording", () => {
  const detail = {
    id: "r1",
    created_at: "2026-07-01T10:00:00Z",
    filename: "kitchen-fight.m4a",
    media_type: "audio",
    duration_seconds: 182,
    has_analysis: true,
    turns: [
      { speaker: "Alice", text: "You never listen.", start_time: 0, end_time: 3 },
      { speaker: "Bob", text: "That's not fair.", start_time: 3, end_time: 6 },
    ],
    analysis: {
      per_turn: [],
      per_speaker: {},
      dynamics: {
        coupling: { strength: null, leader: null, description: "" },
        deescalation: { who_first: null, follow_rate: null, description: "" },
        triggers: [],
        requests: [],
      },
      narrative: "",
    },
  };

  it("GETs /recordings/{id} (id URL-encoded) and returns the detail", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => detail });
    const result = await getRecording("r 1");
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/recordings\/r%201$/);
    expect(init.method).toBe("GET");
    expect(result).toEqual(detail);
  });

  it("throws an honest error on 404", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 404 });
    await expect(getRecording("nope")).rejects.toThrow("API error: 404");
  });
});

describe("getRecordingMediaUrl", () => {
  it("GETs /recordings/{id}/media_url and returns the signed URL payload", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ url: "https://signed.example/abc", expires_in: 600 }),
    });
    const result = await getRecordingMediaUrl("r1");
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/recordings\/r1\/media_url$/);
    expect(init.method).toBe("GET");
    expect(result).toEqual({ url: "https://signed.example/abc", expires_in: 600 });
  });

  it("throws an honest error on 503", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503 });
    await expect(getRecordingMediaUrl("r1")).rejects.toThrow("API error: 503");
  });
});

describe("getRecordingSourceUrl", () => {
  it("GETs /recordings/{id}/source_url and returns the resolved URL payload", async () => {
    const payload = {
      url: "https://lh3.googleusercontent.com/pw/XYZ=dv",
      content_type: "video/mp4",
      expires_hint: "may expire; refetch on failure",
    };
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => payload });
    const result = await getRecordingSourceUrl("r 1");
    const [url, init] = mockFetch.mock.calls[0];
    // id is URL-encoded in the path.
    expect(url).toMatch(/\/recordings\/r%201\/source_url$/);
    expect(init.method).toBe("GET");
    expect(result).toEqual(payload);
  });

  it("throws an honest error on 404 (no remote source)", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 404 });
    await expect(getRecordingSourceUrl("r1")).rejects.toThrow("API error: 404");
  });

  it("throws an honest error on 502 (source unreachable) so the caller falls back", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 502 });
    await expect(getRecordingSourceUrl("r1")).rejects.toThrow("API error: 502");
  });
});

describe("patchRecordingSource", () => {
  it("PATCHes /recordings/{id}/source with { url } and returns the link source", async () => {
    const payload = {
      type: "link",
      url: "https://photos.app.goo.gl/abc",
      original_filename: "kitchen-fight.m4a",
    };
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => payload });
    const result = await patchRecordingSource(
      "r 1",
      "https://photos.app.goo.gl/abc",
    );
    const [url, init] = mockFetch.mock.calls[0];
    // id URL-encoded in the path.
    expect(url).toMatch(/\/recordings\/r%201\/source$/);
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body)).toEqual({
      url: "https://photos.app.goo.gl/abc",
    });
    expect(result).toEqual(payload);
  });

  it("surfaces a 422's user-facing detail verbatim (unusable link)", async () => {
    const detail =
      "That link points to an album, not a single video — share one item.";
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 422,
      json: async () => ({ detail }),
    });
    await expect(
      patchRecordingSource("r1", "https://example.com/album"),
    ).rejects.toMatchObject({ message: detail, status: 422, detail });
  });

  it("throws an honest status error on 404", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
      json: async () => ({}),
    });
    await expect(patchRecordingSource("r1", "https://x")).rejects.toThrow(
      "API error: 404",
    );
  });
});

describe("deleteRecording", () => {
  it("DELETEs /recordings/{id} and resolves on 204", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 204 });
    await expect(deleteRecording("r1")).resolves.toBeUndefined();
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/recordings\/r1$/);
    expect(init.method).toBe("DELETE");
  });

  it("throws an honest error on a non-OK response", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 404 });
    await expect(deleteRecording("r1")).rejects.toThrow("API error: 404");
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
