import React, { useCallback, useEffect, useState } from "react";
import { View, Text, TextInput, TouchableOpacity, ActivityIndicator, StyleSheet } from "react-native";
import { getPlan, redeemCode, type PlanSummary } from "../api/plan";

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
  const stt = limits.stt_seconds;
  if (typeof stt === "number" && stt > 0) out.push(`Up to ${Math.round(stt / 3600)} h of cloud transcription a day`);
  const calls = limits.calls;
  if (typeof calls === "number" && calls > 0) out.push(`${calls} calls a day`);
  const items = limits.library_items;
  if (typeof items === "number" && items > 0) out.push(`Library: up to ${items} items`);
  return out;
}

/**
 * Settings → Plan: current plan + expiry + what it includes, and a
 * "Redeem a code" box. The purchase button is a placeholder until paid plans
 * ship (docs/plans/2026-10-31-paid-plans.md): no fake purchase.
 */
export default function PlanCard() {
  const [plan, setPlan] = useState<PlanSummary | null>(null);
  const [failed, setFailed] = useState(false);
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

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

  const onRedeem = async () => {
    if (!code.trim() || busy) return;
    setBusy(true);
    setMsg(null);
    try {
      const r = await redeemCode(code.trim());
      setPlan(r.plan);
      setCode("");
      setMsg({
        ok: true,
        text:
          r.status === "granted"
            ? `Code redeemed. You're on ${r.plan.label}.`
            : r.status === "already_redeemed"
              ? "You've already redeemed that code."
              : `You already have ${r.plan.label} — nothing to add.`,
      });
    } catch (e) {
      setMsg({ ok: false, text: e instanceof Error ? e.message : "Couldn't redeem that code." });
    } finally {
      setBusy(false);
    }
  };

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

      <TouchableOpacity accessibilityRole="button" disabled style={[styles.buy, styles.disabled]} testID="plan-upgrade">
        <Text style={styles.buyText}>Upgrade to Pro — coming soon</Text>
      </TouchableOpacity>

      {plan?.redeem_available ? (
        <View style={styles.redeem}>
          <TextInput
            testID="plan-code-input"
            style={styles.input}
            placeholder="Redeem a code"
            autoCapitalize="characters"
            autoCorrect={false}
            value={code}
            onChangeText={setCode}
            onSubmitEditing={onRedeem}
          />
          <TouchableOpacity
            testID="plan-redeem-button"
            accessibilityRole="button"
            style={styles.redeemBtn}
            onPress={onRedeem}
            disabled={busy || !code.trim()}
          >
            <Text style={styles.buyText}>{busy ? "…" : "Redeem"}</Text>
          </TouchableOpacity>
        </View>
      ) : plan?.is_guest ? (
        <Text style={styles.sub}>Create an account to redeem a code.</Text>
      ) : null}
      {msg ? (
        <Text style={[styles.sub, { color: msg.ok ? "#15803d" : "#b91c1c" }]} testID="plan-redeem-msg">
          {msg.text}
        </Text>
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
  redeemBtn: { backgroundColor: "#4f46e5", borderRadius: 8, paddingHorizontal: 14, paddingVertical: 10 },
});
