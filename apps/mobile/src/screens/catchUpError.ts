/**
 * Honest wording for a failed POST /voice/catch-up (GrowthScreen's "Catch up
 * my past recordings"). The endpoint legitimately runs for a long time — it
 * re-embeds every speaker of up to 25 recordings on a scale-to-zero server —
 * so the ways it fails are different enough that one generic banner misleads:
 * a dropped socket (the OS killed it while the app was backgrounded) is not a
 * server outage, and a 429 means "it's still working", not "it broke".
 *
 * Pure: takes the thrown value, returns the sentence to show.
 */

export const CATCH_UP_CONNECTION_LOST =
  "Lost the connection while checking — keep the app open and try again.";
export const CATCH_UP_STILL_RUNNING = "Still checking — try again in a minute.";
export const CATCH_UP_UNAVAILABLE =
  "Voice matching isn't available on the server right now.";
export const CATCH_UP_GENERIC =
  "Couldn’t check your past recordings. Please try again.";

/** `.status` set by the API client on a non-OK response; falling back to the
 *  "API error: NNN" message shape for anything that lost the property. */
function statusOf(err: unknown): number | null {
  if (typeof err !== "object" || err === null) return null;
  const status = (err as { status?: unknown }).status;
  if (typeof status === "number") return status;
  const message = (err as { message?: unknown }).message;
  const m = typeof message === "string" ? /API error: (\d{3})/.exec(message) : null;
  return m ? Number(m[1]) : null;
}

function isTransportFailure(err: unknown): boolean {
  if (err instanceof TypeError) return true;
  const message =
    typeof err === "object" && err !== null
      ? (err as { message?: unknown }).message
      : err;
  return typeof message === "string" && message.includes("Network request failed");
}

export function catchUpErrorMessage(err: unknown): string {
  if (isTransportFailure(err)) return CATCH_UP_CONNECTION_LOST;
  const status = statusOf(err);
  if (status === 429) return CATCH_UP_STILL_RUNNING;
  if (status === 503) return CATCH_UP_UNAVAILABLE;
  return status === null ? CATCH_UP_GENERIC : `${CATCH_UP_GENERIC} (HTTP ${status})`;
}
