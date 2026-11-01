import React, { useCallback, useEffect, useState } from "react";
import { View, Text, TouchableOpacity, ActivityIndicator, StyleSheet, Share } from "react-native";
import { getPlan, type PlanSummary } from "../api/plan";
import RedeemBox from "./RedeemBox";
import { useListeningStore } from "../store/listeningStore";
import { purchasesEnabled } from "../purchases/purchases";
import { startUpgrade, startRestore, type FlowResult } from "../purchases/upgradeFlow";

function fmtDate(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString();
}

function sourceText(p: PlanSummary): string {
  if (p.plan === "free") return "";
  const until = p.expires_at ? `until ${fmtDate(p.expires_at)}` : "no expiry";
  const how = p.source === "promo" ? "promo code" : p.source === "revenuecat" ? "subscription" : "granted";
  return `${how}, ${until}`;
}

/** What the current plan gives, in plain words (numbers come from the server). */
export function perks(p: PlanSummary): string[] {
  const out: string[] = [];
  const limits = p.limits ?? {};
  const life = p.windows?.stt_lifetime;
  const month = p.windows?.stt_monthly;
  if (life) {
    const left = Math.max(0, life.cap - life.used);
    out.push(`${(left / 3600).toFixed(1)} of ${Math.round(life.cap / 3600)} free transcription hours left`);
  } else if (month) {
    out.push(`${Math.round(month.cap / 3600)} h of cloud transcription a month`);
  }
  const calls = limits.calls;
  if (typeof calls === "number" && calls > 0) out.push(`${calls} calls a day`);
  const items = limits.library_items;
  if (typeof items === "number" && items > 0) out.push(`Library: up to ${items} items`);
  return out;
}

type Tier = NonNullable<PlanSummary["tiers"]>[number];

interface Buy {
  canBuy: boolean;
  busy: boolean;
  onBuy: (level: string) => void;
}

function TierCard({ t, current, buy }: { t: Tier; current?: string; buy?: Buy }) {
  const isCurrent = current === t.level;
  const enabled = !!buy?.canBuy && !isCurrent && !buy.busy;
  const label = isCurrent ? "Current" : buy?.canBuy ? (buy.busy ? "Please wait" : "Subscribe") : "Coming soon";
  return (
    <View style={[styles.tier, t.popular && styles.tierPopular]} testID={`tier-${t.level}`}>
      <View style={{ flex: 1 }}>
        <Text style={styles.plan}>
          {t.label}
          {t.popular ? <Text style={styles.badge} testID="tier-popular">  Most popular</Text> : null}
        </Text>
        <Text style={styles.sub}>
          {`$${t.price_usd}/mo · ${t.hours_per_month} h of listening a month · ${t.library_items} library items`}
        </Text>
      </View>
      <TouchableOpacity
        accessibilityRole="button"
        disabled={!enabled}
        onPress={() => buy?.onBuy(t.level)}
        style={[styles.buy, !enabled && styles.disabled, styles.tierBtn]}
        testID={`tier-${t.level}-buy`}
      >
        <Text style={styles.buyText}>{label}</Text>
      </TouchableOpacity>
    </View>
  );
}

/** Basic / Plus (most popular) / Pro by default; Max under "For business". */
export function TierCards({
  tiers,
  current,
  buy,
  footer,
}: {
  tiers: Tier[];
  current?: string;
  buy?: Buy;
  footer?: React.ReactNode;
}) {
  const [business, setBusiness] = useState(false);
  const main = tiers.filter((t) => !t.business);
  const biz = tiers.filter((t) => t.business);
  if (!main.length) return null;
  return (
    <View testID="tier-cards">
      {main.map((t) => (
        <TierCard key={t.level} t={t} current={current} buy={buy} />
      ))}
      {biz.length ? (
        business ? (
          biz.map((t) => <TierCard key={t.level} t={t} current={current} buy={buy} />)
        ) : (
          <TouchableOpacity testID="tier-business-link" accessibilityRole="link" onPress={() => setBusiness(true)}>
            <Text style={styles.link}>For business</Text>
          </TouchableOpacity>
        )
      ) : null}
      {buy?.canBuy ? footer : <Text style={styles.sub}>Paid plans are coming soon.</Text>}
    </View>
  );
}

/**
 * Settings → Plan: current plan + expiry + what it includes, and a
 * "Redeem a code" box. Subscribe buttons buy through RevenueCat when this build
 * has a key and the server says purchases are on; otherwise they stay "Coming
 * soon" (docs/plans/2026-10-31-paid-plans.md). Never a fake purchase.
 */
export default function PlanCard() {
  const [plan, setPlan] = useState<PlanSummary | null>(null);
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [flowMsg, setFlowMsg] = useState<FlowResult | null>(null);

  const load = useCallback(async () => {
    try {
      setPlan(await getPlan());
      setFailed(false);
    } catch {
      setFailed(true);
    }
  }, []);
  useEffect(() => {
    void load();
  }, [load]);

  const canBuy = !!plan?.purchase?.available && !plan.is_guest && purchasesEnabled();

  const run = async (go: () => Promise<FlowResult>) => {
    if (busy) return;
    setBusy(true);
    setFlowMsg(null);
    try {
      const r = await go();
      setFlowMsg(r);
      if (r.plan) setPlan(r.plan);
    } finally {
      setBusy(false);
    }
  };
  const buy: Buy = { canBuy, busy, onBuy: (level) => void run(() => startUpgrade(level)) };
  const footer = (
    <View>
      <TouchableOpacity
        testID="plan-restore"
        accessibilityRole="button"
        disabled={busy}
        onPress={() => void run(() => startRestore())}
      >
        <Text style={styles.link}>Restore purchases</Text>
      </TouchableOpacity>
      {flowMsg && !flowMsg.cancelled ? (
        <Text style={[styles.sub, !flowMsg.ok && styles.err]} testID="plan-flow-message">
          {flowMsg.message}
        </Text>
      ) : null}
      <Text style={styles.sub}>
        Subscriptions renew monthly until cancelled. Manage or cancel in your App Store / Google Play account settings.
      </Text>
    </View>
  );

  return (
    <View style={styles.card} testID="plan-card">
      {plan ? (
        <>
          <Text style={styles.plan} testID="plan-name">
            {plan.label ?? "Free"} plan
          </Text>
          {sourceText(plan) ? (
            <Text style={styles.sub} testID="plan-source">
              {sourceText(plan)}
            </Text>
          ) : null}
          {perks(plan).map((l) => (
            <Text key={l} style={styles.sub}>
              • {l}
            </Text>
          ))}
        </>
      ) : failed ? (
        <Text style={styles.sub} testID="plan-unavailable">
          Your plan couldn't be loaded right now.
        </Text>
      ) : (
        <ActivityIndicator />
      )}

      <TierCards tiers={plan?.tiers ?? []} current={plan?.plan} buy={buy} footer={footer} />

      {plan?.redeem_available ? (
        <RedeemBox
          onDone={(p) => {
            setPlan(p);
            useListeningStore.getState().setFromPlan(p);
          }}
        />
      ) : plan?.is_guest ? (
        <Text style={styles.sub}>Create an account to redeem a code.</Text>
      ) : null}
      {plan?.referral ? (
        <View style={styles.redeem} testID="plan-referral">
          <Text style={[styles.sub, { flex: 1 }]}>
            Your share code: <Text style={styles.code} selectable testID="plan-referral-code">{plan.referral.code}</Text>
            {"\n"}Invite a friend: you both get {plan.referral.bonus_hours_each} more free hours.
            {plan.referral.bonus_hours > 0
              ? ` Bonus earned: ${plan.referral.bonus_hours} of ${plan.referral.max_bonus_hours} h.`
              : ""}
          </Text>
          <TouchableOpacity
            testID="plan-referral-share"
            accessibilityRole="button"
            style={styles.redeemBtn}
            onPress={() =>
              void Share.share({
                message: `Try MindShift. Enter my code ${plan.referral!.code} in Settings > Plan and we both get ${plan.referral!.bonus_hours_each} more free hours.`,
              })
            }
          >
            <Text style={styles.buyText}>Share</Text>
          </TouchableOpacity>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: { backgroundColor: "#fff", borderRadius: 12, padding: 14, marginBottom: 12 },
  plan: { fontSize: 17, fontWeight: "700", color: "#111" },
  sub: { fontSize: 13, color: "#444", marginTop: 4 },
  buy: { backgroundColor: "#4f46e5", borderRadius: 10, paddingVertical: 10, alignItems: "center", marginTop: 10 },
  disabled: { opacity: 0.5 },
  buyText: { color: "#fff", fontWeight: "700" },
  redeem: { flexDirection: "row", marginTop: 10, alignItems: "center" },
  input: { flex: 1, borderWidth: 1, borderColor: "#ccc", borderRadius: 8, paddingHorizontal: 10, paddingVertical: 8, marginRight: 8 },
  tier: { flexDirection: "row", alignItems: "center", borderWidth: 1, borderColor: "#ddd", borderRadius: 10, padding: 10, marginTop: 8 },
  tierPopular: { borderColor: "#4f46e5", borderWidth: 2 },
  tierBtn: { marginTop: 0, paddingHorizontal: 12 },
  badge: { color: "#4f46e5", fontSize: 12, fontWeight: "700" },
  err: { color: "#b91c1c" },
  link: { color: "#4f46e5", marginTop: 8, fontWeight: "600" },
  code: { fontWeight: "700", letterSpacing: 1 },
  redeemBtn: { backgroundColor: "#4f46e5", borderRadius: 8, paddingHorizontal: 14, paddingVertical: 10 },
});
