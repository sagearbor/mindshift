import { create } from "zustand";
import { parsePlanLimit, type PlanLimitNotice } from "../api/plan";

/**
 * The most recent plan-limit / quota refusal from the server, shown by
 * <PlanLimitSheet/> (mounted once in App.tsx). API modules call
 * `reportPlanLimit(detail)` when a 429 body parses as a typed limit; it
 * returns the friendly sentence so the caller can use it as its error text.
 */
interface PlanLimitState {
  notice: PlanLimitNotice | null;
  dismiss: () => void;
}

export const usePlanLimitStore = create<PlanLimitState>((set) => ({
  notice: null,
  dismiss: () => set({ notice: null }),
}));

export function reportPlanLimit(detail: unknown): string | null {
  const notice = parsePlanLimit(detail);
  if (!notice) return null;
  usePlanLimitStore.setState({ notice });
  return notice.message;
}
