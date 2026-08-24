/** getGrowth + deleteVoiceSample + catchUpVoice — the new client calls behind
 *  "Your growth" and the voice-profile card. Mirrors client.test.ts's
 *  global-fetch style. */
import { catchUpVoice, deleteVoiceSample, getGrowth } from "../src/api/client";
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
    // `people` (Track 2) is always an array — an older server that omits
    // the key yields an empty list, never undefined.
    expect(result).toEqual({ ...body, people: [] });
    // Null scores pass through as null — gaps, never zeroed.
    expect(result.points[1].my_score).toBeNull();
  });

  it("defends against a sparse body", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => ({}) });
    await expect(getGrowth()).resolves.toEqual({
      points: [],
      total_recordings: 0,
      identified_recordings: 0,
      people: [],
    });
  });

  it("passes Track 2 tone/people through verbatim", async () => {
    const body = {
      points: [
        {
          recording_id: "r1",
          timestamp: "2026-08-24T18:05:00+00:00",
          title: "Live session · earpiece",
          my_score: 64,
          partner_names: ["Mom"],
          source: "live",
          mode: "earpiece",
          self_tone: {
            scored_turns: 3,
            labels: { warm: 1, frustrated: 1, defensive: 1 },
            mean: { warmth: 50, frustration: 42 },
            escalation_count: 2,
            people: [],
          },
        },
      ],
      total_recordings: 1,
      identified_recordings: 1,
      people: [
        {
          person_id: "p-mom",
          display_name: "Mom",
          sessions: 1,
          scored_turns: 3,
          labels: { warm: 1, frustrated: 1, defensive: 1 },
          escalation_count: 2,
        },
      ],
    };
    mockFetch.mockResolvedValueOnce({ ok: true, json: async () => body });
    await expect(getGrowth()).resolves.toEqual(body);
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

describe("catchUpVoice", () => {
  it("POSTs /voice/catch-up and returns the checked/newly_identified/remaining counts", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ checked: 5, newly_identified: 3, remaining: 2 }),
    });

    const result = await catchUpVoice();

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toContain("/voice/catch-up");
    expect(init.method).toBe("POST");
    expect(result).toEqual({ checked: 5, newly_identified: 3, remaining: 2 });
  });

  it("throws with .status and the server's honest detail on a non-OK", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => ({ detail: "voice enrollment not available on this server" }),
    });
    await expect(catchUpVoice()).rejects.toMatchObject({
      status: 503,
      message: "voice enrollment not available on this server",
    });
  });

  it("falls back to a generic message when the error body isn't JSON", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: async () => {
        throw new Error("not json");
      },
    });
    await expect(catchUpVoice()).rejects.toMatchObject({
      status: 500,
      message: "API error: 500",
    });
  });
});
