import { getMe } from "../src/api/me";
import { setCachedToken, setTokenProvider } from "../src/auth/authToken";

const mockFetch = global.fetch as jest.Mock;

beforeEach(() => {
  mockFetch.mockReset();
  setCachedToken(null);
  setTokenProvider(null);
});

describe("getMe", () => {
  it("GETs /me with a fresh Bearer token and returns the parsed body", async () => {
    setCachedToken("id-token-abc");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        account_id: "u1",
        email: "a@example.com",
        legacy: false,
        has_paired_watch: true,
      }),
    });

    const me = await getMe();

    expect(me).toEqual({
      account_id: "u1",
      email: "a@example.com",
      legacy: false,
      has_paired_watch: true,
    });
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/me$/);
    expect(init.method).toBe("GET");
    expect(init.headers.Authorization).toBe("Bearer id-token-abc");
  });

  it("omits the Authorization header when signed out", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        account_id: "default",
        email: null,
        legacy: true,
        has_paired_watch: false,
      }),
    });

    await getMe();

    const [, init] = mockFetch.mock.calls[0];
    expect(init.headers.Authorization).toBeUndefined();
  });

  it("throws an honest error on a non-2xx response, never a fabricated result", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: async () => ({ detail: "not authenticated" }),
    });

    await expect(getMe()).rejects.toThrow(/401/);
  });
});
