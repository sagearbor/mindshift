import type { Turn } from "../store/sessionStore";
import type { Suggestion } from "../components/SuggestionCard";

const API_URL =
  process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

interface RespondRequest {
  role: string;
  empathy_level: number;
  turns: Turn[];
}

interface RespondResponse {
  suggestions: Suggestion[];
}

export async function postRespond(
  payload: RespondRequest,
): Promise<RespondResponse> {
  const res = await fetch(`${API_URL}/respond`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }

  return res.json();
}
