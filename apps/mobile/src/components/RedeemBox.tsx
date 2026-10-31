import React, { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, StyleSheet } from "react-native";
import { redeemCode, type PlanSummary } from "../api/plan";

export function redeemMessage(status: string, plan: PlanSummary): string {
  if (status === "granted") {
    return plan.plan === "free"
      ? "Code redeemed. Your free hours went up."
      : `Code redeemed. You're on ${plan.label}.`;
  }
  if (status === "already_redeemed") return "You've already redeemed that code.";
  return `You already have ${plan.label} — nothing to add.`;
}

/** The "Have a code?" box, shared by the Plan card and the out-of-time sheet. */
export default function RedeemBox({ onDone }: { onDone?: (plan: PlanSummary) => void }) {
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const onRedeem = async () => {
    if (!code.trim() || busy) return;
    setBusy(true);
    setMsg(null);
    try {
      const r = await redeemCode(code.trim());
      setCode("");
      setMsg({ ok: true, text: redeemMessage(r.status, r.plan) });
      onDone?.(r.plan);
    } catch (e) {
      setMsg({ ok: false, text: e instanceof Error ? e.message : "Couldn't redeem that code." });
    } finally {
      setBusy(false);
    }
  };

  return (
    <View>
      <View style={styles.row}>
        <TextInput
          testID="plan-code-input"
          style={styles.input}
          placeholder="Have a code? (promo or friend's)"
          autoCapitalize="characters"
          autoCorrect={false}
          value={code}
          onChangeText={setCode}
          onSubmitEditing={onRedeem}
        />
        <TouchableOpacity
          testID="plan-redeem-button"
          accessibilityRole="button"
          style={styles.btn}
          onPress={onRedeem}
          disabled={busy || !code.trim()}
        >
          <Text style={styles.btnText}>{busy ? "…" : "Redeem"}</Text>
        </TouchableOpacity>
      </View>
      {msg ? (
        <Text style={[styles.msg, { color: msg.ok ? "#15803d" : "#b91c1c" }]} testID="plan-redeem-msg">
          {msg.text}
        </Text>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: "row", marginTop: 10, alignItems: "center" },
  input: { flex: 1, borderWidth: 1, borderColor: "#ccc", borderRadius: 8, paddingHorizontal: 10, paddingVertical: 8, marginRight: 8 },
  btn: { backgroundColor: "#4f46e5", borderRadius: 8, paddingHorizontal: 14, paddingVertical: 10 },
  btnText: { color: "#fff", fontWeight: "700" },
  msg: { fontSize: 13, marginTop: 4 },
});
