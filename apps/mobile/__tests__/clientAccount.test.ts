/** src/api/account.ts — the DELETE /me call. Global-fetch style, same as
 *  clientPeople.test.ts / clientGrowth.test.ts. */
import { deleteAccount, DELETE_CONFIRMATION } from "../src/api/account";
import { setCachedToken, setTokenProvider } from "../src/auth/authToken";

const mockFetch = global.fetch as jest.Mock;

beforeEach(() => {
  mockFetch.mockReset();
  setCachedToken(null);
  setTokenProvider(null);
});

const OK = {
  ok: true,
  status: 200,
  json: async () => ({
    deleted: true,
    firebase_user_deleted: true,
    counts: { recordings: 2 },
  }),
};

describe("deleteAccount", () => {
  it("DELETEs /me with the typed confirmation in the body", async () => {
    mockFetch.mockResolvedValueOnce(OK);
    const res = await deleteAccount();

    expect(mockFetch.mock.calls[0][0]).toMatch(/\/me$/);
    const init = mockFetch.mock.calls[0][1];
    expect(init.method).toBe("DELETE");
    expect(JSON.parse(init.body)).toEqual({ confirm: DELETE_CONFIRMATION });
    expect(res.deleted).toBe(true);
    expect(res.counts.recordings).toBe(2);
  });

  it("force-refreshes the ID token so the server's freshness gate passes", async () => {
    const provider = jest.fn(async (forceRefresh?: boolean) =>
      forceRefresh ? "brand-new-token" : "stale-cached-token",
    );
    setTokenProvider(provider);
    mockFetch.mockResolvedValueOnce(OK);

    await deleteAccount();

    expect(provider).toHaveBeenCalledWith(true);
    expect(mockFetch.mock.calls[0][1].headers.Authorization).toBe(
      "Bearer brand-new-token",
    );
  });

  it("sends no Authorization header when signed out, letting the server 401", async () => {
    mockFetch.mockResolvedValueOnce(OK);
    await deleteAccount();
    expect(mockFetch.mock.calls[0][1].headers.Authorization).toBeUndefined();
  });

  it("surfaces the server's own reason and status on a partial failure", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: async () => ({
        detail: {
          message: "Some of your data could not be deleted",
          failed: ["recordings: RuntimeError"],
        },
      }),
    });

    await expect(deleteAccount()).rejects.toMatchObject({
      status: 500,
      message: "Some of your data could not be deleted",
    });
  });

  it("surfaces a plain-string detail (401 / 422 / 429) verbatim", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: async () => ({ detail: "a freshly issued sign-in token is required" }),
    });
    await expect(deleteAccount()).rejects.toMatchObject({
      status: 401,
      message: "a freshly issued sign-in token is required",
    });
  });

  it("falls back to the status line when the body is not JSON", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 502,
      json: async () => {
        throw new Error("not json");
      },
    });
    await expect(deleteAccount()).rejects.toMatchObject({
      status: 502,
      message: "API error: 502",
    });
  });
});
