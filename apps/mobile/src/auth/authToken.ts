/**
 * Decouples the token *producer* (authStore/Firebase) from its *consumers*
 * (the REST client and the WebSocket hook). Neither consumer imports Firebase
 * or the store directly, which keeps them synchronous where they must be
 * (the WS `config` frame is sent inside `ws.onopen`, with no chance to await)
 * and trivially testable (a test just calls the setters).
 */

export type TokenProvider = () => Promise<string | null>;

let cachedToken: string | null = null;
let provider: TokenProvider | null = null;

/**
 * Register the source of *fresh* ID tokens — Firebase's
 * `user.getIdToken()`, which transparently refreshes a token nearing its
 * ~1h expiry. Called once when auth initializes.
 */
export function setTokenProvider(next: TokenProvider | null): void {
  provider = next;
}

/**
 * The last-known ID token, synchronously. Used where awaiting is impossible —
 * the WS `config` frame. Kept current by authStore's onIdTokenChanged
 * subscription (which fires on sign-in, sign-out, and every auto-refresh).
 */
export function getCachedToken(): string | null {
  return cachedToken;
}

/** Update the cached token (from onIdTokenChanged / sign-in / sign-out). */
export function setCachedToken(token: string | null): void {
  cachedToken = token;
}

/**
 * A fresh ID token for REST calls: asks Firebase for one (force-refreshing if
 * near expiry) via the registered provider, falling back to the cached token
 * if the provider is absent (tests) or errors. Returns null when signed out.
 */
export async function getFreshToken(): Promise<string | null> {
  if (!provider) return cachedToken;
  try {
    const token = await provider();
    return token ?? cachedToken;
  } catch {
    return cachedToken;
  }
}
