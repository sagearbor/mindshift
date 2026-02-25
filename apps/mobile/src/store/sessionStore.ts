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

    set({ loading: true });
    try {
      const data = await postRespond({ role, empathy_level: empathyLevel, turns });
      set({ suggestions: data.suggestions ?? [] });
    } catch {
      set({ suggestions: [] });
    } finally {
      set({ loading: false });
    }
  },
}));
