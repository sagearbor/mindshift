/**
 * GET /me — the signed-in user's own account facts from the unified backend
 * (server/watch/routers/rest.py's `MeResponse`, Task P3-6). The Settings
 * screen (apps/mobile/src/screens/AdvancedScreen.tsx) uses `has_paired_watch`
 * so "Set up your watch" can show live paired state instead of guessing.
 *
 * Mirrors watchPairing.ts's authHeaders() shape: a fresh Firebase ID token
 * as Bearer auth, omitted when signed out so the server answers its own 401
 * rather than the client fabricating one.
 *
 * Honest degradation: throws on any non-2xx or network failure. The caller
 * treats a rejection identically to "unknown" (no badge shown) — never a
 * fabricated paired/unpaired guess.
 */
import { getFreshToken } from "../auth/authToken";

const API_URL = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

export interface Me {
  account_id: string;
  email: string | null;
  legacy: boolean;
  has_paired_watch: boolean;
}

async function authHeaders(): Promise<Record<string, string>> {
  const token = await getFreshToken();
  const headers: Record<string, string> = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

/** GET /me. Throws `Error("API error: <status>")` on any non-2xx response —
 *  callers fall back to their existing honest-default UI on rejection. */
export async function getMe(): Promise<Me> {
  const res = await fetch(`${API_URL}/me`, {
    method: "GET",
    headers: await authHeaders(),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return res.json();
}
