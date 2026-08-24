/**
 * Live sessions this app instance just finished and the server confirmed
 * (`POST /sessions/live` → 201). Your Day merges these into its listing so
 * a session shows up the moment it ends — even if the recordings list was
 * fetched a beat earlier — and drops the optimistic copy as soon as the
 * server's own row is present.
 *
 * Only CONFIRMED episodes go in here (the server returned an episode id);
 * a failed/unsupported POST never fabricates a row.
 */
import { create } from "zustand";
import type { RecordingSummary } from "../api/client";

export interface RecentLiveEpisode {
  episodeId: string;
  sessionId: string;
  startedAt: string;
  mode: string;
  title: string;
  turnCount: number;
  /** Therapist emails the server auto-shared it with at ingest. */
  sharedWith: string[];
}

interface LiveEpisodeState {
  recent: RecentLiveEpisode[];
  remember: (episode: RecentLiveEpisode) => void;
  forget: (episodeId: string) => void;
  clear: () => void;
}

const MAX_RECENT = 10;

export const useLiveEpisodeStore = create<LiveEpisodeState>((set) => ({
  recent: [],
  remember: (episode) =>
    set((s) => ({
      recent: [episode, ...s.recent.filter((e) => e.episodeId !== episode.episodeId)].slice(
        0,
        MAX_RECENT,
      ),
    })),
  forget: (episodeId) =>
    set((s) => ({ recent: s.recent.filter((e) => e.episodeId !== episodeId) })),
  clear: () => set({ recent: [] }),
}));

/** The list-row shape Your Day renders, for an episode the server has not
 *  yet returned in `GET /recordings` (same fields `_recording_summary`
 *  serves for a live session: no media, analysis present as "lite"). */
export function asRecordingSummary(e: RecentLiveEpisode): RecordingSummary {
  return {
    id: e.episodeId,
    created_at: e.startedAt,
    filename: "live-session",
    title: e.title,
    media_type: "none",
    duration_seconds: null,
    has_analysis: true,
    source_type: "live",
    mode: e.mode,
  };
}

/** Server rows win; optimistic rows fill only what the server lacks. */
export function mergeRecent(
  serverRows: RecordingSummary[],
  recent: RecentLiveEpisode[],
): RecordingSummary[] {
  const seen = new Set(serverRows.map((r) => r.id));
  const extra = recent.filter((e) => !seen.has(e.episodeId)).map(asRecordingSummary);
  return extra.length ? [...extra, ...serverRows] : serverRows;
}
