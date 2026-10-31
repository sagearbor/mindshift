import React, { useState } from "react";
import { Modal, View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { usePlanLimitStore } from "../store/planLimitStore";
import { useListeningStore } from "../store/listeningStore";
import RedeemBox from "./RedeemBox";
import { purchasesEnabled } from "../purchases/purchases";
import { startUpgrade } from "../purchases/upgradeFlow";

/**
 * Friendly "you're out of free time / plan limit" sheet, shown when the server
 * refuses something because of the user's plan (typed 429 from server/plans.py
 * / usage_meter, or a quota_notice frame on the live socket) or when the
 * listening clock reaches zero. Nothing the user already has is touched; the
 * sheet says so, offers Redeem a code, and an Upgrade button that opens the
 * same store purchase flow as the Plan card (disabled "coming soon" when this
 * build has no RevenueCat key).
 */
export default function PlanLimitSheet() {
  const notice = usePlanLimitStore((s) => s.notice);
  const dismiss = usePlanLimitStore((s) => s.dismiss);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  if (!notice) return null;
  const level = notice.upgradePlan ?? "basic";
  const canBuy = purchasesEnabled();
  const upgrade = async () => {
    if (busy) return;
    setBusy(true);
    setMsg(null);
    const r = await startUpgrade(level);
    setBusy(false);
    if (r.ok) dismiss();
    else if (!r.cancelled) setMsg(r.message);
  };
  return (
    <Modal transparent animationType="fade" visible onRequestClose={dismiss}>
      <View style={styles.backdrop}>
        <View style={styles.sheet} testID="plan-limit-sheet">
          <Text style={styles.title}>You've reached your plan's limit</Text>
          <Text style={styles.body} testID="plan-limit-message">
            {notice.message}
          </Text>
          <RedeemBox
            onDone={(plan) => {
              useListeningStore.getState().setFromPlan(plan);
              dismiss();
            }}
          />
          <TouchableOpacity
            testID="plan-limit-upgrade"
            accessibilityRole="button"
            disabled={!canBuy || busy}
            onPress={() => void upgrade()}
            style={[styles.upgrade, (!canBuy || busy) && styles.disabled]}
          >
            <Text style={styles.buttonText}>{canBuy ? (busy ? "Please wait" : "Upgrade") : "Upgrade — coming soon"}</Text>
          </TouchableOpacity>
          {msg ? (
            <Text style={styles.body} testID="plan-limit-upgrade-error">
              {msg}
            </Text>
          ) : null}
          <TouchableOpacity
            testID="plan-limit-dismiss"
            accessibilityRole="button"
            style={styles.button}
            onPress={dismiss}
          >
            <Text style={styles.buttonText}>OK</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: "rgba(0,0,0,0.5)", justifyContent: "center", padding: 24 },
  sheet: { backgroundColor: "#fff", borderRadius: 14, padding: 20 },
  title: { fontSize: 18, fontWeight: "700", marginBottom: 8, color: "#111" },
  body: { fontSize: 15, color: "#222", marginBottom: 8 },
  button: { backgroundColor: "#4f46e5", borderRadius: 10, paddingVertical: 12, alignItems: "center", marginTop: 8 },
  upgrade: { backgroundColor: "#4f46e5", borderRadius: 10, paddingVertical: 12, alignItems: "center", marginTop: 10 },
  disabled: { opacity: 0.5 },
  buttonText: { color: "#fff", fontWeight: "700" },
});
