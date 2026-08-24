/**
 * src/api/liveSessions.ts — `fetchVoiceprints` against the embeddings
 * opt-in (`GET /voice/people?include_embeddings=true`). Global-fetch style
 * like clientLive.test.ts. The server's default response (no `embedding`
 * key) must yield nobody to match — never a guess — and every failure
 * names its cause for the session status line.
 */
import { fetchVoiceprints, parseVoiceprints, VOICEPRINTS_PATH } from "../src/api/liveSessions";
import { setCachedToken, setTokenProvider } from "../src/auth/authToken";

const mockFetch = global.fetch as jest.Mock;

beforeEach(() => {
  mockFetch.mockReset();
  setCachedToken(null);
  setTokenProvider(null);
});

const SELF = { person_id: "self", display_name: "You", is_self: true, embedding: [1, 0, 0], dim: 3, model: "ecapa@rev" };
const MOM = { person_id: "mom", display_name: "Mom", is_self: false, embedding: [0, 1, 0], dim: 3, model: "ecapa@rev" };

describe("fetchVoiceprints", () => {
  it("GETs /voice/people?include_embeddings=true with auth and parses people with prints", async () => {
    setCachedToken("tok");
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ available: true, storage_enabled: true, people: [SELF, MOM] }) });
    const res = await fetchVoiceprints();
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toContain("/voice/people?include_embeddings=true");
    expect(VOICEPRINTS_PATH).toBe("/voice/people?include_embeddings=true");
    expect(init.method).toBe("GET");
    expect(init.headers.Authorization).toBe("Bearer tok");
    expect(init.headers["Content-Type"]).toBeUndefined();
    expect(res.error).toBeNull();
    expect(res.people).toEqual([
      { personId: "self", displayName: "You", isSelf: true, embedding: [1, 0, 0], model: "ecapa@rev", dim: 3 },
      { personId: "mom", displayName: "Mom", isSelf: false, embedding: [0, 1, 0], model: "ecapa@rev", dim: 3 },
    ]);
  });

  it("the default (no-embedding) response yields nobody to match", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ people: [{ person_id: "self", display_name: "You", is_self: true, enrolled: true, dim: 192 }] }),
    });
    expect(await fetchVoiceprints()).toEqual({ people: [], error: null });
  });

  it("names the cause on 404 / 401 / other / network failure", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 404 });
    expect(await fetchVoiceprints()).toEqual({ people: [], error: expect.stringContaining("404") });
    mockFetch.mockResolvedValueOnce({ ok: false, status: 401 });
    expect(await fetchVoiceprints()).toEqual({ people: [], error: expect.stringContaining("not signed in") });
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    expect(await fetchVoiceprints()).toEqual({ people: [], error: expect.stringContaining("500") });
    mockFetch.mockRejectedValueOnce(new Error("Network request failed"));
    expect(await fetchVoiceprints()).toEqual({ people: [], error: expect.stringContaining("Network request failed") });
  });
});

describe("parseVoiceprints", () => {
  it("accepts a bare array or {people}, the legacy `voiceprint` key, and fills display names", () => {
    expect(parseVoiceprints([{ person_id: "p", voiceprint: [0, 1] }])).toEqual([
      { personId: "p", displayName: "p", isSelf: false, embedding: [0, 1], model: null, dim: 2 },
    ]);
    expect(parseVoiceprints({ people: [{ person_id: "self", is_self: true, embedding: [1] }] })[0]).toMatchObject({
      displayName: "You",
      isSelf: true,
    });
    expect(parseVoiceprints(null)).toEqual([]);
    expect(parseVoiceprints({})).toEqual([]);
  });

  it("drops prints whose length disagrees with dim, or that carry non-numbers", () => {
    expect(parseVoiceprints([{ person_id: "a", embedding: [1, 2], dim: 192 }])).toEqual([]);
    expect(parseVoiceprints([{ person_id: "b", embedding: [1, "x"] as unknown as number[] }])).toEqual([]);
    expect(parseVoiceprints([{ person_id: "c", embedding: [1, Number.NaN] }])).toEqual([]);
    expect(parseVoiceprints([{ person_id: "d", embedding: [] }, { embedding: [1] } as never])).toEqual([]);
  });
});
