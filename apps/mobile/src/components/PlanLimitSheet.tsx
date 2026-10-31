import React from "react";
import { Modal, View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { usePlanLimitStore } from "../store/planLimitStore";

/**
 * Friendly upgrade message shown when the server refuses something because of
 * the user's plan (typed 429 from server/plans.py / usage_meter). Nothing the
 * user already has is touched; the message says so. Paid purchase is not live
 * yet, so there is no purchase button, only the honest hint.
 */
export default function PlanLimitSheet() {
  const notice = usePlanLimitStore((s) => s.notice);
  const dismiss = usePlanLimitStore((s) => s.dismiss);
  if (!notice) return null;
  return (
    <Modal transparent animationType="fade" visible onRequestClose={dismiss}>
      <View style={styles.backdrop}>
        <View style={styles.sheet} testID="plan-limit-sheet">
          <Text style={styles.title}>You've reached your plan's limit</Text>
          <Text style={styles.body} testID="plan-limit-message">
            {notice.message}
          </Text>
          {notice.upgradePlan ? (
            <Text style={styles.hint}>
              Got a code? Settings → Plan → Redeem a code. Paid plans are coming soon.
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
  hint: { fontSize: 13, color: "#555", marginBottom: 12 },
  button: { backgroundColor: "#4f46e5", borderRadius: 10, paddingVertical: 12, alignItems: "center" },
  buttonText: { color: "#fff", fontWeight: "700" },
});
