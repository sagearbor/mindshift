/**
 * postLiveSessionAudio — attaching the kept session WAV to a saved live
 * episode. Direct multipart under the direct cap; the chunked flow (with
 * `attach_to_recording_id` on complete) above it. Global-fetch style like
 * clientLive.test.ts / the chunked-upload tests in client.test.ts.
 */
import {
  LIVE_AUDIO_DIRECT_MAX_BYTES,
  postLiveSessionAudio,
} from "../src/api/liveSessions";
import { setCachedToken, setTokenProvider } from "../src/auth/authToken";

const mockFetch = global.fetch as jest.Mock;

const okBody = {
  recording_id: "ep-1",
  media_type: "audio" as const,
  duration_seconds: 12.5,
  size_bytes: 400_044,
  stored_variants: ["wav"],
};

function ramp(n: number): Uint8Array {
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = i % 256;
  return out;
}

beforeEach(() => {
  mockFetch.mockReset();
  setCachedToken(null);
  setTokenProvider(null);
  delete (globalThis as Record<string, unknown>).__fsMockBytes;
  delete (globalThis as Record<string, unknown>).__fsMockSize;
});

describe("postLiveSessionAudio (direct)", () => {
  it("POSTs multipart `file` (session.wav) to /sessions/{id}/audio with auth and a request id", async () => {
    setCachedToken("tok");
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => okBody });

    const result = await postLiveSessionAudio("ep-1", "file:///cache/live-audio/live-2.wav", 400_044);

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/sessions\/ep-1\/audio$/);
    expect(init.method).toBe("POST");
    expect(init.headers.Authorization).toBe("Bearer tok");
    expect(init.headers["X-Request-ID"]).toMatch(/^[0-9a-f]{32}$/);
    // fetch must add the multipart boundary itself.
    expect(init.headers["Content-Type"]).toBeUndefined();
    expect(init.body).toBeInstanceOf(FormData);
    const part = (init.body as FormData).get("file") as File;
    expect(part).toBeInstanceOf(Blob);
    expect(part.name).toBe("session.wav");
    expect((init.body as FormData).getAll("file")).toHaveLength(1);
    expect(result).toEqual(okBody);
  });

  it("throws with .status on a non-OK answer (404 unknown episode, 422 undecodable)", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 404 });
    await expect(postLiveSessionAudio("nope", "file:///x.wav", 10)).rejects.toMatchObject({
      message: "API error: 404",
      status: 404,
    });
    mockFetch.mockResolvedValueOnce({ ok: false, status: 422 });
    await expect(postLiveSessionAudio("ep-1", "file:///x.wav", 10)).rejects.toMatchObject({ status: 422 });
  });

  it("a network rejection throws with status 0", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("Network request failed"));
    await expect(postLiveSessionAudio("ep-1", "file:///x.wav", 10)).rejects.toMatchObject({
      status: 0,
      message: expect.stringContaining("Network request failed"),
    });
  });

  it("retries once with a force-refreshed token on 401", async () => {
    const provider = jest.fn(async (force?: boolean) => (force ? "fresh" : "stale"));
    setTokenProvider(provider);
    mockFetch
      .mockResolvedValueOnce({ ok: false, status: 401 })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => okBody });
    await postLiveSessionAudio("ep-1", "file:///x.wav", 10);
    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(mockFetch.mock.calls[0][1].headers.Authorization).toBe("Bearer stale");
    expect(mockFetch.mock.calls[1][1].headers.Authorization).toBe("Bearer fresh");
    expect(provider).toHaveBeenCalledWith(true);
  });
});

describe("postLiveSessionAudio (chunked)", () => {
  it("over the direct cap: /uploads/start → PUT chunks → complete with attach_to_recording_id", async () => {
    setCachedToken("tok");
    const bytes = LIVE_AUDIO_DIRECT_MAX_BYTES + 1;
    // The mock file handle serves slices of this; the server's chunk plan is
    // what drives the loop, so a tiny stand-in file exercises the real path.
    (globalThis as Record<string, unknown>).__fsMockBytes = ramp(250);
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ upload_id: "up1", chunk_bytes: 100, expected_chunks: 3 }),
      })
      .mockResolvedValueOnce({ ok: true, status: 204 })
      .mockResolvedValueOnce({ ok: true, status: 204 })
      .mockResolvedValueOnce({ ok: true, status: 204 })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => okBody });

    const result = await postLiveSessionAudio("ep-1", "file:///cache/live-audio/big.wav", bytes);

    expect(mockFetch).toHaveBeenCalledTimes(5);
    const [startUrl, startInit] = mockFetch.mock.calls[0];
    expect(startUrl).toMatch(/\/uploads\/start$/);
    expect(startInit.method).toBe("POST");
    expect(startInit.headers.Authorization).toBe("Bearer tok");
    expect(startInit.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(startInit.body)).toEqual({
      filename: "session.wav",
      content_type: "audio/wav",
      total_bytes: bytes,
      consent: true,
      store: true,
    });
    const requestId = startInit.headers["X-Request-ID"];
    expect(requestId).toMatch(/^[0-9a-f]{32}$/);

    for (let i = 0; i < 3; i++) {
      const [url, init] = mockFetch.mock.calls[1 + i];
      expect(url).toMatch(new RegExp(`/uploads/up1/chunks/${i}$`));
      expect(init.method).toBe("PUT");
      expect(init.headers["Content-Type"]).toBe("application/octet-stream");
      expect(init.headers["X-Request-ID"]).toBe(requestId);
      expect(init.body).toBeInstanceOf(Uint8Array);
    }
    expect((mockFetch.mock.calls[1][1].body as Uint8Array).length).toBe(100);
    expect((mockFetch.mock.calls[2][1].body as Uint8Array)[0]).toBe(100); // byte 100 of the ramp

    const [completeUrl, completeInit] = mockFetch.mock.calls[4];
    expect(completeUrl).toMatch(/\/uploads\/up1\/complete$/);
    expect(completeInit.method).toBe("POST");
    expect(completeInit.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(completeInit.body)).toEqual({ attach_to_recording_id: "ep-1" });
    expect(result).toEqual(okBody);
  });

  it("a 413 from the direct POST re-routes through the chunked flow", async () => {
    (globalThis as Record<string, unknown>).__fsMockBytes = ramp(50);
    mockFetch
      .mockResolvedValueOnce({ ok: false, status: 413 }) // direct
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ upload_id: "up2", chunk_bytes: 50, expected_chunks: 1 }),
      })
      .mockResolvedValueOnce({ ok: true, status: 204 })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => okBody });
    const result = await postLiveSessionAudio("ep-1", "file:///x.wav", 50);
    expect(mockFetch.mock.calls[0][0]).toMatch(/\/sessions\/ep-1\/audio$/);
    expect(mockFetch.mock.calls[1][0]).toMatch(/\/uploads\/start$/);
    expect(mockFetch.mock.calls[2][0]).toMatch(/\/uploads\/up2\/chunks\/0$/);
    expect(JSON.parse(mockFetch.mock.calls[3][1].body)).toEqual({ attach_to_recording_id: "ep-1" });
    expect(result).toEqual(okBody);
  });

  it("aborts the partial upload (DELETE) and throws with .status when complete fails", async () => {
    (globalThis as Record<string, unknown>).__fsMockBytes = ramp(50);
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ upload_id: "up3", chunk_bytes: 50, expected_chunks: 1 }),
      })
      .mockResolvedValueOnce({ ok: true, status: 204 })
      .mockResolvedValueOnce({ ok: false, status: 404 }) // complete: unknown recording
      .mockResolvedValueOnce({ ok: true, status: 204 }); // DELETE
    await expect(
      postLiveSessionAudio("gone", "file:///x.wav", LIVE_AUDIO_DIRECT_MAX_BYTES + 1),
    ).rejects.toMatchObject({ status: 404 });
    expect(mockFetch).toHaveBeenCalledTimes(4);
    const [delUrl, delInit] = mockFetch.mock.calls[3];
    expect(delUrl).toMatch(/\/uploads\/up3$/);
    expect(delInit.method).toBe("DELETE");
  });

  it("throws without aborting when /uploads/start itself fails", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503 });
    await expect(
      postLiveSessionAudio("ep-1", "file:///x.wav", LIVE_AUDIO_DIRECT_MAX_BYTES + 1),
    ).rejects.toMatchObject({ status: 503 });
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });
});
