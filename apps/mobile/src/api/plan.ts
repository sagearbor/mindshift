/**
 * Plans — GET /plan and POST /plan/redeem (server/routers/plan.py, server/plans.py).
 *
 * Honest errors: a failed redeem rejects with an Error carrying `.status` and
 * `.reason` and the server's own friendly sentence as the message. The app
 * never fabricates a plan: when GET /plan fails the card says it is unavailable.
 */
import { getFreshToken } from "../auth/authToken";

const API_URL = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

export interface PlanSummary {
  plan: string;
  label: string;
  expires_at: string | null;
  source: "revenuecat" | "admin" | "promo" | null;
  is_guest: boolean;
  limits: Record<string, number | boolean | null>;
  daily: Record<string, { cap: number | null; used: number }>;
  library: { items: number | null; chars: number | null };
  flags: Record<string, boolean>;
  upgrade_plan: string | null;
  redeem_available: boolean;
  /** Free: lifetime cap; Pro: monthly cap (seconds of cloud transcription). */
  windows?: Record<string, { cap: number; used: number }>;
  /** Personal share code; null for guests / when plans are unavailable. */
  referral?: {
    code: string;
    rewards_earned: number;
    bonus_hours_each: number;
    bonus_hours: number;
    max_bonus_hours: number;
    already_referred: boolean;
  } | null;
  purchase: { available: boolean; message?: string };
}

export type RedeemStatus = "granted" | "already_redeemed" | "covered";

export type PlanApiError = Error & { status?: number; reason?: string };

async function headers(json: boolean): Promise<Record<string, string>> {
  const token = await getFreshToken();
  const h: Record<string, string> = json ? { "Content-Type": "application/json" } : {};
  if (token) h.Authorization = `Bearer ${token}`;
  return h;
}

export async function getPlan(): Promise<PlanSummary> {
  const res = await fetch(`${API_URL}/plan`, { method: "GET", headers: await headers(false) });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return (await res.json()) as PlanSummary;
}

export async function redeemCode(
  code: string,
): Promise<{ status: RedeemStatus; plan: PlanSummary }> {
  const res = await fetch(`${API_URL}/plan/redeem`, {
    method: "POST",
    headers: await headers(true),
    body: JSON.stringify({ code }),
  });
  if (!res.ok) {
    let message = "";
    let reason: string | undefined;
    try {
      const body = (await res.json()) as { detail?: unknown };
      const d = body?.detail as { message?: unknown; reason?: unknown } | string | undefined;
      if (typeof d === "string") message = d;
      else if (d && typeof d.message === "string") message = d.message;
      if (d && typeof d === "object" && typeof d.reason === "string") reason = d.reason;
    } catch {
      // non-JSON body
    }
    const err = new Error(message || "Couldn't redeem that code. Please try again.") as PlanApiError;
    err.status = res.status;
    err.reason = reason;
    throw err;
  }
  return (await res.json()) as { status: RedeemStatus; plan: PlanSummary };
}

/** What a server plan-limit / quota 429 means to a human. */
export interface PlanLimitNotice {
  message: string;
  limit?: string;
  plan?: string;
  upgradePlan?: string | null;
  resetsAt?: string;
}

/**
 * Recognise the typed 429 bodies: `{type: "plan_limit"}` (library caps) and
 * `{type: "quota_notice"}` (daily caps, which now carry plan/upgrade_plan).
 * Returns null for anything else so callers keep their existing handling.
 */
export function parsePlanLimit(detail: unknown): PlanLimitNotice | null {
  if (!detail || typeof detail !== "object") return null;
  const d = detail as Record<string, unknown>;
  if (d.type !== "plan_limit" && d.type !== "quota_notice") return null;
  if (typeof d.message !== "string" || !d.message) return null;
  return {
    message: d.message,
    limit: typeof d.limit === "string" ? d.limit : undefined,
    plan: typeof d.plan === "string" ? d.plan : undefined,
    upgradePlan: typeof d.upgrade_plan === "string" ? d.upgrade_plan : null,
    resetsAt: typeof d.resets_at === "string" ? d.resets_at : undefined,
  };
}
