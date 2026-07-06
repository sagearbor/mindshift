import { create } from "zustand";
import { postRespond } from "../api/client";
import type { Suggestion } from "../components/SuggestionCard";

export interface Turn {
  speaker: string;
  text: string;
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
