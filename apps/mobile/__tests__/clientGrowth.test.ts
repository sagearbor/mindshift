/** getGrowth + deleteVoiceSample — the new client calls behind "Your growth"
 *  and the voice-profile card. Mirrors client.test.ts's global-fetch style. */
import { deleteVoiceSample, getGrowth } from "../src/api/client";
import { setCachedToken, setTokenProvider } from "../src/auth/authToken";

const mockFetch = global.fetch as jest.Mock;

beforeEach(() => {
  mockFetch.mockReset();
  setCachedToken(null);
  setTokenProvider(null);
});

describe("getGrowth", () => {
  it("GETs /growth and returns the aggregate verbatim", async () => {
    const body = {
      points: [
        {
          recording_id: "r1",
          timestamp: "2026-07-01T12:00:00+00:00",
          title: "Kitchen talk",
          my_score: 70,
          partner_names: ["Linda"],
        },
        {
          recording_id: "r2",
          timestamp: "2026-07-02T12:00:00+00:00",
          title: "Call",
          my_score: null,
          partner_names: [],
        },
      ],
      total_recordings: 5,
      identified_recordings: 2,
    };
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => body });

    const result = await getGrowth();

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toContain("/growth");
    expect(init.method).toBe("GET");
    expect(result).toEqual(body);
    // Null scores pass through as null — gaps, never zeroed.
    expect(result.points[1].my_score).toBeNull();
  });

  it("defends against a sparse body", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => ({}) });
    await expect(getGrowth()).resolves.toEqual({
      points: [],
      total_recordings: 0,
      identified_recordings: 0,
    });
  });

  it("throws with the status on a non-OK (503 storage disabled)", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503 });
    await expect(getGrowth()).rejects.toThrow("API error: 503");
  });
});

describe("deleteVoiceSample", () => {
  it("DELETEs /voice/samples/{id} and returns what remains", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ deleted: true, enrolled: true, enroll_count: 1 }),
    });

    const result = await deleteVoiceSample("s1");

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toContain("/voice/samples/s1");
    expect(init.method).toBe("DELETE");
    expect(result).toEqual({ deleted: true, enrolled: true, enroll_count: 1 });
  });

  it("attaches .status so the card can roll back honestly", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 404 });
    await expect(deleteVoiceSample("gone")).rejects.toMatchObject({
      status: 404,
    });
  });

  it("URL-encodes the sample id", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ deleted: true, enrolled: false, enroll_count: 0 }),
    });
    await deleteVoiceSample("legacy blend/1");
    expect(mockFetch.mock.calls[0][0]).toContain(
      "/voice/samples/legacy%20blend%2F1",
    );
  });
});
