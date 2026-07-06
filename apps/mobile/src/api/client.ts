import type { Suggestion } from "../components/SuggestionCard";
import { getFreshToken } from "../auth/authToken";

const API_URL =
  process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Request body for POST /respond. Mirrors the server's RespondRequest
 * (server/main.py): the server wants the single latest utterance as
 * `transcript_turn`, the empathy slider as an int 0–100, the role, and any
 * prior conversation as free-text `context`.
 */
export interface RespondRequest {
  transcript_turn: string;
  role: string;
  empathy_slider: number;
  context?: string;
}

/** Raw JSON returned by the server: suggestions are plain strings. */
interface RespondServerResponse {
  suggestions: string[];
  tone_score: Record<string, number>;
}

/** Parsed result the app consumes: suggestions carried as {text, tone}. */
export interface RespondResult {
  suggestions: Suggestion[];
  toneScore: Record<string, number>;
}

/**
 * Maps the empathy slider (0–100) to the coaching-stance label shown on each
 * suggestion. Mirrors useAudioStream's empathyTone: it describes how the
 * suggestion was generated, not a claim about detected tone (the server's
 * /respond returns suggestions as bare strings with no per-item tone).
 */
export function empathyTone(slider: number): string {
  if (slider <= 20) return "assertive";
  if (slider <= 50) return "balanced";
  if (slider <= 80) return "empathetic";
  return "validating";
}

export async function postRespond(
  payload: RespondRequest,
): Promise<RespondResult> {
  // A fresh Firebase ID token authenticates the request; the backend verifies
  // it and scopes data to the token's uid. Null when signed out — the header is
  // then omitted and the server answers with its own 401.
  const token = await getFreshToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const res = await fetch(`${API_URL}/respond`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }

  const data = (await res.json()) as RespondServerResponse;
  const tone = empathyTone(payload.empathy_slider);
  const suggestions: Suggestion[] = Array.isArray(data.suggestions)
    ? data.suggestions.map((text) => ({ text, tone }))
    : [];

  return { suggestions, toneScore: data.tone_score ?? {} };
}
