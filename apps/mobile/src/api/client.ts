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

// --- POST /analyze (Conversation Dynamics post-session analysis) ------------
// These types mirror the backend's /analyze response contract exactly. They are
// intentionally faithful: nullable fields (interruptions, coupling.strength,
// leader, follow_rate) stay `| null` so the UI can *omit* rather than fabricate
// a value the model couldn't determine — the house "no fake data" rule.

/** One turn as sent to /analyze. start/end times are optional (present only
 *  for diarized live audio; absent for pasted transcripts). */
export interface AnalyzeTurnInput {
  speaker: string;
  text: string;
  start_time?: number;
  end_time?: number;
}

/** Emotional/behavioral markers the backend may attach to a turn. */
export type ConversationMarker =
  | "criticism"
  | "contempt"
  | "defensiveness"
  | "stonewalling"
  | "repair_attempt"
  | "validation";

export interface AnalyzePerTurn {
  index: number;
  speaker: string;
  heat: number; // 0–100
  markers: string[]; // subset of ConversationMarker
  is_spike: boolean;
  trigger_phrase: string | null;
}

export interface HorsemenCounts {
  criticism: number;
  contempt: number;
  defensiveness: number;
  stonewalling: number;
}

export interface AnalyzePerSpeaker {
  turns: number;
  talk_share: number; // 0–1
  avg_heat: number;
  peak_heat: number;
  peak_turn_index: number;
  heat_variance: number;
  interruptions: number | null; // null when the backend can't infer it
  horsemen: HorsemenCounts;
  repair_attempts: number;
  repairs_accepted: number;
}

export interface AnalyzeDynamics {
  coupling: {
    strength: number | null;
    leader: string | null;
    description: string;
  };
  deescalation: {
    who_first: string | null;
    follow_rate: number | null;
    description: string;
  };
  triggers: {
    phrase: string;
    speaker: string;
    turn_index: number;
    heat_delta: number;
  }[];
  requests: { speaker: string; request: string; outcome: string }[];
}

/**
 * Per-speaker "report card" grade. `score` is an ABSOLUTE 0–100 measure of
 * conduct (higher = better) — intentionally comparable across speakers (owner's
 * product decision), so the UI renders it plainly without softening qualifiers.
 */
export interface ReportCard {
  score: number; // 0–100, higher = better conduct
  headline: string;
  did_well: string;
  work_on: string;
}

export interface AnalyzeResult {
  per_turn: AnalyzePerTurn[];
  per_speaker: Record<string, AnalyzePerSpeaker>;
  // Optional at the type level to match runtime reality: a pre-v2 server omits
  // this field entirely (the UI already guards and omits the section).
  report_cards?: Record<string, ReportCard>;
  dynamics: AnalyzeDynamics;
  narrative: string;
}

/** One simulated turn from /analyze/counterfactual: a hypothetical heat value
 *  at that turn's conversation-wide index, from the pivot to the last turn. */
export interface SimulatedTurn {
  index: number;
  speaker: string;
  heat: number; // 0–100
}

/** Result of POST /analyze/counterfactual — a "what if this turn had been said
 *  differently" projection. Never fabricated client-side; comes wholesale from
 *  the server, disclaimer included, and is rendered verbatim. */
export interface CounterfactualResult {
  pivot_index: number;
  rewritten_text: string;
  rationale: string;
  simulated_per_turn: SimulatedTurn[];
  disclaimer: string;
}

/**
 * POST /analyze — post-session "Conversation Dynamics" analysis. Follows
 * postRespond exactly: a fresh Firebase ID token as Bearer auth (omitted when
 * signed out, so the server answers its own 401), JSON body, and a thrown
 * `API error: <status>` on any non-OK response so the caller can surface an
 * honest error state (401/429/502/413) rather than a fabricated result.
 */
export async function postAnalyze(
  turns: AnalyzeTurnInput[],
  context?: string,
): Promise<AnalyzeResult> {
  const token = await getFreshToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const res = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    headers,
    // Only include `context` when provided — keeps the body shape minimal and
    // matches the optional field in the contract.
    body: JSON.stringify(context !== undefined ? { turns, context } : { turns }),
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }

  return (await res.json()) as AnalyzeResult;
}

/**
 * POST /analyze/counterfactual — the "what if this turn had been said
 * differently?" projection. Follows postAnalyze exactly: fresh Firebase ID
 * token as Bearer auth (omitted when signed out so the server answers its own
 * 401), JSON body carrying the full `turns` (same shape incl. optional
 * start/end times) plus the `pivot_index` to rewrite, and a thrown
 * `API error: <status>` on any non-OK response (401/422/429/502/413) so the
 * caller can show an honest inline error rather than a fabricated simulation.
 */
export async function postCounterfactual(
  turns: AnalyzeTurnInput[],
  pivotIndex: number,
  context?: string,
): Promise<CounterfactualResult> {
  const token = await getFreshToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const res = await fetch(`${API_URL}/analyze/counterfactual`, {
    method: "POST",
    headers,
    body: JSON.stringify(
      context !== undefined
        ? { turns, pivot_index: pivotIndex, context }
        : { turns, pivot_index: pivotIndex },
    ),
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }

  return (await res.json()) as CounterfactualResult;
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
