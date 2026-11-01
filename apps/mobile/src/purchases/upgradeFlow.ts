/**
 * One upgrade flow for the Plan card and the plan-limit sheet: buy (or restore)
 * through the store, then refresh GET /plan until the server has caught up with
 * the RevenueCat webhook. The plan shown is always the server's, never assumed.
 */
import { getPlan, type PlanSummary } from "../api/plan";
import { useAuthStore } from "../store/authStore";
import { useListeningStore } from "../store/listeningStore";
import { purchaseLevel, restore, PurchaseError, friendlyError } from "./purchases";

export interface FlowResult {
  ok: boolean;
  cancelled?: boolean;
  message: string;
  plan?: PlanSummary;
}

const RANK = ["free", "basic", "plus", "pro", "max"];
const rank = (l: string) => RANK.indexOf(l);

export const POLL_MS = 1500;
export const POLL_TRIES = 8;

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** Poll /plan until it reaches `atLeast` (or tries run out); returns the latest. */
export async function refreshPlan(atLeast: string | null, sleepMs = POLL_MS): Promise<PlanSummary | undefined> {
  let latest: PlanSummary | undefined;
  for (let i = 0; i < POLL_TRIES; i++) {
    try {
      latest = await getPlan();
      if (!atLeast || rank(latest.plan) >= rank(atLeast)) break;
    } catch {
      // keep trying; the webhook may simply not have landed yet
    }
    if (i < POLL_TRIES - 1) await sleep(sleepMs);
  }
  if (latest) useListeningStore.getState().setFromPlan(latest);
  return latest;
}

function user() {
  return useAuthStore.getState().user;
}

export async function startUpgrade(level: string, sleepMs = POLL_MS): Promise<FlowResult> {
  try {
    await purchaseLevel(level, user());
  } catch (e) {
    const pe = friendlyError(e);
    return { ok: false, cancelled: pe.kind === "cancelled", message: pe.message };
  }
  const plan = await refreshPlan(level, sleepMs);
  if (plan && rank(plan.plan) >= rank(level)) {
    return { ok: true, message: `You're on ${plan.label}. Thank you!`, plan };
  }
  return {
    ok: true,
    message: "Purchase complete. Your plan will update in a moment; pull to refresh if it doesn't.",
    plan,
  };
}

export async function startRestore(sleepMs = POLL_MS): Promise<FlowResult> {
  try {
    await restore(user());
  } catch (e) {
    const pe: PurchaseError = friendlyError(e);
    return { ok: false, message: pe.message };
  }
  const plan = await refreshPlan(null, sleepMs);
  return {
    ok: true,
    message: plan && plan.plan !== "free" ? `Restored. You're on ${plan.label}.` : "Purchases restored. No active subscription found.",
    plan,
  };
}
