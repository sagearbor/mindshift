/** postReflect + listDashboardSessions — the Track 2 client calls behind
 *  "What you could have said" and the therapist dashboard. Mirrors
 *  clientGrowth.test.ts's global-fetch style. */
import { listDashboardSessions, postReflect } from "../src/api/client";
import { setCachedToken, setTokenProvider } from "../src/auth/authToken";

const mockFetch = global.fetch as jest.Mock;

beforeEach(() => {
  mockFetch.mockReset();
  setCachedToken(null);
  setTokenProvider(null);
});

describe("postReflect", () => {
  it("POSTs /episodes/{id}/reflect with auth and returns the reflection", async () => {
    setCachedToken("tok");
    const body = {
      episode_id: "e1",
      self_speaker: "Speaker A",
      could_have_said: [
        { turn_index: 2, could_have_said: "I hear you.", why: "Owns it.", tone_read: "defensive" },
      ],
      cached: true,
      reflected_at: "2026-08-24T18:10:00+00:00",
    };
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => body });

    const result = await postReflect("e1");

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toContain("/episodes/e1/reflect");
    expect(url).not.toContain("force");
    expect(init.method).toBe("POST");
    expect(init.headers.Authorization).toBe("Bearer tok");
    expect(result).toEqual(body);
  });

  it("adds ?force=true on demand and defends against a sparse body", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => ({}) });
    const result = await postReflect("e1", true);
    expect(mockFetch.mock.calls[0][0]).toContain("/episodes/e1/reflect?force=true");
    expect(result).toEqual({
      episode_id: "e1",
      self_speaker: "",
      could_have_said: [],
      cached: false,
      reflected_at: null,
    });
  });

  it("throws with the status on a non-OK (422 no identified self)", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 422 });
    await expect(postReflect("e1")).rejects.toThrow("API error: 422");
  });
});

describe("listDashboardSessions", () => {
  it("GETs /sessions with auth and returns the rows", async () => {
    setCachedToken("tok");
    const sessions = [
      {
        id: "s1", date: "2026-08-24T18:05:00+00:00", role: "You", turns: [],
        avgPleasantness: null, source: "live", mode: "earpiece",
      },
    ];
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => ({ sessions }) });
    const result = await listDashboardSessions();
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/sessions$/);
    expect(init.method).toBe("GET");
    expect(init.headers.Authorization).toBe("Bearer tok");
    expect(result).toEqual(sessions);
  });

  it("yields an empty list for a sparse body and throws on non-OK", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => ({}) });
    await expect(listDashboardSessions()).resolves.toEqual([]);
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503 });
    await expect(listDashboardSessions()).rejects.toThrow("API error: 503");
  });
});
