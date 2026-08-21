import { claimWatchPairing, disconnectWatch } from "../src/api/watchPairing";
import { getFreshToken, setCachedToken, setTokenProvider } from "../src/auth/authToken";

const mockFetch = global.fetch as jest.Mock;

beforeEach(() => {
  mockFetch.mockReset();
  setCachedToken(null);
  setTokenProvider(null);
});

describe("claimWatchPairing", () => {
  it("POSTs the code to /me/pair/claim with a fresh Bearer token and resolves ok:true", async () => {
    setCachedToken("id-token-abc");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        status: "claimed",
        pairing_id: "p1",
        account_id: "u1",
      }),
    });

    const result = await claimWatchPairing("ABC123");

    expect(result).toEqual({ ok: true });
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/me\/pair\/claim$/);
    expect(init.method).toBe("POST");
    expect(init.headers.Authorization).toBe("Bearer id-token-abc");
    expect(JSON.parse(init.body)).toEqual({ code: "ABC123" });
  });

  it("maps a 401 (signed out) to an honest, non-thrown failure", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: async () => ({ detail: "not authenticated" }),
    });

    const result = await claimWatchPairing("ABC123");

    expect(result.ok).toBe(false);
    expect(result.detail).toMatch(/signed in/i);
  });

  it("maps a 404 (bad or expired code) to friendly retry copy, not the raw server string", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
      json: async () => ({ detail: "pairing code not found or already used" }),
    });

    const result = await claimWatchPairing("ZZZZZZ");

    expect(result.ok).toBe(false);
    expect(result.detail).toBeTruthy();
    expect(result.detail).not.toBe("pairing code not found or already used");
    expect(result.detail).toMatch(/wrong|expired|try again/i);
  });

  it("maps a 409 (already claimed) to friendly retry copy as well", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 409,
      json: async () => ({ detail: "this code was already claimed" }),
    });

    const result = await claimWatchPairing("ABC123");

    expect(result.ok).toBe(false);
    expect(result.detail).toMatch(/wrong|expired|try again/i);
  });

  it("surfaces the server's 429 lockout message verbatim", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 429,
      json: async () => ({
        detail: "too many failed pairing attempts on this account",
      }),
    });

    const result = await claimWatchPairing("ABC123");

    expect(result).toEqual({
      ok: false,
      detail: "too many failed pairing attempts on this account",
    });
  });

  it("falls back to a generic message on a 429 with no decodable body", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 429,
      json: async () => {
        throw new Error("not json");
      },
    });

    const result = await claimWatchPairing("ABC123");

    expect(result.ok).toBe(false);
    expect(result.detail).toMatch(/too many|try again/i);
  });

  it("falls back to an honest generic message on an unexpected status", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => {
        throw new Error("not json");
      },
    });

    const result = await claimWatchPairing("ABC123");

    expect(result.ok).toBe(false);
    expect(result.detail).toMatch(/503|went wrong/i);
  });
});

describe("disconnectWatch", () => {
  it("DELETEs /me/watch-pairing with a fresh Bearer token and resolves the server's body", async () => {
    setCachedToken("id-token-abc");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ disconnected: true, count: 1 }),
    });

    const result = await disconnectWatch();

    expect(result).toEqual({ disconnected: true, count: 1 });
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/me\/watch-pairing$/);
    expect(init.method).toBe("DELETE");
    expect(init.headers.Authorization).toBe("Bearer id-token-abc");
  });

  it("resolves count:0 the same idempotent way when nothing was paired", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ disconnected: true, count: 0 }),
    });

    const result = await disconnectWatch();

    expect(result).toEqual({ disconnected: true, count: 0 });
  });

  it("throws on a non-OK response, mirroring client.ts's forgetVoice", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: async () => ({ detail: "not authenticated" }),
    });

    await expect(disconnectWatch()).rejects.toThrow("API error: 401");
  });
});
