import { create } from "zustand";
import { postRespond } from "../api/client";
import type { Suggestion } from "../components/SuggestionCard";

export interface Turn {
  speaker: string;
  text: string;
}

/**
 * Parse a pasted/typed conversation into turns for async review. Each non-empty
 * line is a turn; a leading "Name:" prefix (short, word-like) becomes the
 * speaker, otherwise the whole line is the text with an empty speaker. Kept
 * pure and exported so it's unit-testable without the store.
 *
 * The speaker guard is deliberately conservative (<=24 chars, no sentence
 * punctuation) so a mid-sentence colon — "here's the thing: ..." — isn't
 * mistaken for a speaker label.
 */
export function parseTranscript(raw: string): Turn[] {
  const turns: Turn[] = [];
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const m = trimmed.match(/^([A-Za-z][A-Za-z0-9 _\-]{0,23}):\s*(.+)$/);
    if (m) {
      turns.push({ speaker: m[1].trim(), text: m[2].trim() });
    } else {
      turns.push({ speaker: "", text: trimmed });
    }
  }
  return turns;
}

interface SessionState {
  role: string;
  empathyLevel: number;
  turns: Turn[];
  suggestions: Suggestion[];
  loading: boolean;

  setRole: (role: string) => void;
  setEmpathyLevel: (level: number) => void;
  addTurn: (turn: Turn) => void;
  /** Replace all turns from a pasted/typed transcript (async review). */
  loadTranscript: (raw: string) => void;
  /** Load turns directly (e.g. a finished live-coach conversation handed off
   *  for post-session review). Clears any stale suggestions. */
  loadTurns: (turns: { speaker: string; text: string }[]) => void;
  clearTurns: () => void;
  fetchSuggestions: () => Promise<void>;
}

export const useSessionStore = create<SessionState>((set, get) => ({
  role: "Husband / Wife",
  empathyLevel: 50,
  turns: [],
  suggestions: [],
  loading: false,

  setRole: (role) => set({ role }),
  setEmpathyLevel: (level) => set({ empathyLevel: level }),
  addTurn: (turn) => set((s) => ({ turns: [...s.turns, turn] })),
  loadTranscript: (raw) => set({ turns: parseTranscript(raw), suggestions: [] }),
  loadTurns: (turns) =>
    set({
      turns: turns.map((t) => ({ speaker: t.speaker, text: t.text })),
      suggestions: [],
    }),
  clearTurns: () => set({ turns: [], suggestions: [] }),

  fetchSuggestions: async () => {
    const { role, empathyLevel, turns } = get();
    if (turns.length === 0) return;

    // The server coaches on the latest utterance; earlier turns are supplied
    // as free-text context. Send the empathy slider as an int (server expects
    // 0–100).
    const latest = turns[turns.length - 1];
    const context = turns
      .slice(0, -1)
      .map((t) => `${t.speaker}: ${t.text}`)
      .join("\n");

    set({ loading: true });
    try {
      const { suggestions } = await postRespond({
        transcript_turn: latest.text,
        role,
        empathy_slider: Math.round(empathyLevel),
        context,
      });
      set({ suggestions });
    } catch {
      set({ suggestions: [] });
    } finally {
      set({ loading: false });
    }
  },
}));
