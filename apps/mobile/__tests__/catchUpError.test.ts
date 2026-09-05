import {
  CATCH_UP_CONNECTION_LOST,
  CATCH_UP_GENERIC,
  CATCH_UP_STILL_RUNNING,
  CATCH_UP_UNAVAILABLE,
  catchUpErrorMessage,
} from "../src/screens/catchUpError";

function apiError(status: number, message = `API error: ${status}`) {
  return Object.assign(new Error(message), { status });
}

describe("catchUpErrorMessage — one honest sentence per failure kind", () => {
  it("a dropped socket (TypeError / 'Network request failed') is a connection loss, not a server fault", () => {
    expect(catchUpErrorMessage(new TypeError("Network request failed"))).toBe(
      CATCH_UP_CONNECTION_LOST,
    );
    expect(catchUpErrorMessage(new TypeError("Failed to fetch"))).toBe(
      CATCH_UP_CONNECTION_LOST,
    );
    expect(catchUpErrorMessage(new Error("Network request failed"))).toBe(
      CATCH_UP_CONNECTION_LOST,
    );
  });

  it("429 → still running, try again shortly", () => {
    expect(catchUpErrorMessage(apiError(429))).toBe(CATCH_UP_STILL_RUNNING);
  });

  it("503 → voice matching unavailable on the server", () => {
    expect(catchUpErrorMessage(apiError(503, "Voice ID unavailable"))).toBe(
      CATCH_UP_UNAVAILABLE,
    );
    // The client always sets .status, but the message shape alone is enough.
    expect(catchUpErrorMessage(new Error("API error: 503"))).toBe(
      CATCH_UP_UNAVAILABLE,
    );
  });

  it("anything else keeps the generic text, with the status when there is one", () => {
    expect(catchUpErrorMessage(apiError(500))).toBe(`${CATCH_UP_GENERIC} (HTTP 500)`);
    expect(catchUpErrorMessage(new Error("boom"))).toBe(CATCH_UP_GENERIC);
    expect(catchUpErrorMessage(undefined)).toBe(CATCH_UP_GENERIC);
    expect(catchUpErrorMessage("weird")).toBe(CATCH_UP_GENERIC);
  });
});
