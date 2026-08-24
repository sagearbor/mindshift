import { create } from "zustand";

import {
  listDashboardSessions,
  type CouldHaveSaid,
  type ToneSummary,
} from "../api/client";

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
  // Optional (Track 2): a live-session turn has no empathy-slider setting —
  // the slider belongs to the text coaching flow. Absent → not rendered.
  empathyLevel?: number;
  // Partial (Track 2): a server-projected session carries ONLY the
  // dimensions that were actually measured (pleasantness = 100 − heat once
  // the batch analysis ran; warmth from the phone's text tone). The
  // aggregate averages over the keys present, never fills a missing one.
  toneScores: Partial<ToneScores>;
  // Track 2 per-turn facts from the live session (all optional — legacy
  // sessions and older servers omit them).
  isSelf?: boolean;
  toneLabel?: string | null;
  escalated?: boolean;
  audioLabel?: string | null;
  withPerson?: string | null;
}

export interface SavedSession {
  id: string;
  date: string;
  // The PATIENT label the dashboard groups + filters by: "You" for the
  // caller's own episodes, the owner's email for episodes a patient shared
  // (the therapist ← patient navigation is the existing read-only grant).
  role: string;
  turns: ScoredTurn[];
  // Null when no turn carries a heat yet (a live session before its batch
  // analysis lands) — rendered as "—", never as 0.
  avgPleasantness: number | null;
  // Track 2 additions (optional so legacy fixtures/snapshots are untouched).
  recordingId?: string;
  title?: string | null;
  source?: string | null;
  mode?: string | null;
  shared?: boolean;
  toneSummary?: ToneSummary | null;
  couldHaveSaid?: CouldHaveSaid[] | null;
  analysisStatus?: string | null;
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
      // GET /sessions (authenticated) — every analyzed episode the caller
      // owns or was shared, already in this store's shape (Track 2's
      // server-side projection, live_sessions.dashboard_session).
      const sessions = await listDashboardSessions();
      set({ sessions: Array.isArray(sessions) ? (sessions as SavedSession[]) : [] });
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
