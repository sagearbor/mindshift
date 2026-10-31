import { create } from "zustand";
import type { PlanSummary } from "../api/plan";
import type { Period } from "../live/listeningTime";

/** The server's last statement of the binding listening allowance. */
interface ListeningState {
  /** Seconds left AT `syncedAtMs`; null = unknown / not capped. */
  baseSec: number | null;
  syncedAtMs: number;
  period: Period;
  isFree: boolean;
  setFromPlan: (p: PlanSummary, nowMs?: number) => void;
  clear: () => void;
}

export const useListeningStore = create<ListeningState>((set) => ({
  baseSec: null,
  syncedAtMs: 0,
  period: "lifetime",
  isFree: true,
  setFromPlan: (p, nowMs = Date.now()) =>
    set({
      baseSec: p.listening ? p.listening.remaining_seconds : null,
      syncedAtMs: nowMs,
      period: (p.listening?.period as Period) ?? "lifetime",
      isFree: p.plan === "free",
    }),
  clear: () => set({ baseSec: null }),
}));

/** Remaining seconds at `nowMs`, ticking down only while `running`. */
export function remainingNow(
  s: Pick<ListeningState, "baseSec" | "syncedAtMs">,
  nowMs: number,
  running: boolean,
): number | null {
  if (s.baseSec == null) return null;
  const elapsed = running ? Math.max(0, (nowMs - s.syncedAtMs) / 1000) : 0;
  return Math.max(0, s.baseSec - elapsed);
}
