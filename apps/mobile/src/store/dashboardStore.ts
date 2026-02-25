import { create } from "zustand";

export interface ToneScores {
  warmth: number;
  constructiveness: number;
  calmness: number;
  respect: number;
  engagement: number;
  pleasantness: number;
}

export interface ScoredTurn {
  speaker: string;
  text: string;
  empathyLevel: number;
  toneScores: ToneScores;
}

export interface SavedSession {
  id: string;
  date: string;
  role: string;
  turns: ScoredTurn[];
  avgPleasantness: number;
}

interface DashboardState {
  sessions: SavedSession[];
  selectedSessionId: string | null;
  roleFilter: string | null;
  loading: boolean;

  setSessions: (sessions: SavedSession[]) => void;
  selectSession: (id: string | null) => void;
  setRoleFilter: (role: string | null) => void;
  fetchSessions: () => Promise<void>;
  exportSession: (id: string) => Promise<string>;
}

const API_URL =
  process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

export const useDashboardStore = create<DashboardState>((set, get) => ({
  sessions: [],
  selectedSessionId: null,
  roleFilter: null,
  loading: false,

  setSessions: (sessions) => set({ sessions }),
  selectSession: (id) => set({ selectedSessionId: id }),
  setRoleFilter: (role) => set({ roleFilter: role }),

  fetchSessions: async () => {
    set({ loading: true });
    try {
      const res = await fetch(`${API_URL}/sessions`);
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      set({ sessions: data.sessions ?? [] });
    } catch {
      set({ sessions: [] });
    } finally {
      set({ loading: false });
    }
  },

  exportSession: async (id: string) => {
    const res = await fetch(`${API_URL}/session/${id}/export`);
    if (!res.ok) throw new Error(`Export error: ${res.status}`);
    const data = await res.json();
    return data.text ?? "";
  },
}));
