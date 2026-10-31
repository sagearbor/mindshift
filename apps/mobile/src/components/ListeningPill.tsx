import React, { useCallback, useEffect, useRef, useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { getPlan } from "../api/plan";
import { useListeningStore, remainingNow } from "../store/listeningStore";
import { usePlanLimitStore } from "../store/planLimitStore";
import {
  OUT_OF_TIME_MESSAGE,
  RESYNC_MS,
  noticeCopy,
  noticeCrossed,
  pillText,
  toneFor,
  type Tone,
} from "../live/listeningTime";

const COLORS: Record<Tone, string> = {
  ok: "#15803d",
  amber: "#b45309",
  red: "#b91c1c",
  out: "#b91c1c",
};

/**
 * Live Coach: "Free time left: 42 min", counting down while a session runs.
 * Starts from the server's number (GET /plan `listening`), ticks locally every
 * second, re-syncs every few minutes, goes amber at 10 min and red at 5, shows
 * one gentle banner at 10 / 5 / 1 min, and at zero opens the out-of-time sheet
 * (never silently dies; nothing is deleted). Renders nothing when the plan has
 * no listening cap or the server can't be reached.
 */
export default function ListeningPill({ active }: { active: boolean }) {
  const base = useListeningStore((s) => s.baseSec);
  const syncedAt = useListeningStore((s) => s.syncedAtMs);
  const period = useListeningStore((s) => s.period);
  const isFree = useListeningStore((s) => s.isFree);
  const [now, setNow] = useState(Date.now());
  const [banner, setBanner] = useState<string | null>(null);
  const lastRef = useRef<number | null>(null);
  const outShown = useRef(false);

  const sync = useCallback(async () => {
    try {
      useListeningStore.getState().setFromPlan(await getPlan());
    } catch {
      // keep ticking from the last server number
    }
  }, []);

  useEffect(() => {
    void sync();
    if (!active) return;
    outShown.current = false;
    const tick = setInterval(() => setNow(Date.now()), 1000);
    const resync = setInterval(() => void sync(), RESYNC_MS);
    return () => {
      clearInterval(tick);
      clearInterval(resync);
    };
  }, [active, sync]);

  const remaining = remainingNow({ baseSec: base, syncedAtMs: syncedAt }, now, active);

  useEffect(() => {
    if (remaining == null) return;
    const prev = lastRef.current;
    lastRef.current = remaining;
    if (prev != null) {
      const crossed = noticeCrossed(prev, remaining);
      if (crossed != null) setBanner(noticeCopy(crossed, isFree));
    }
    if (active && remaining <= 0 && !outShown.current) {
      outShown.current = true;
      usePlanLimitStore.setState({
        notice: { message: OUT_OF_TIME_MESSAGE, limit: "listening", upgradePlan: "basic" },
      });
    }
  }, [remaining, active, isFree]);

  if (remaining == null) return null;
  const tone = toneFor(remaining);
  return (
    <View testID="listening-wrap">
      <View style={[styles.pill, { borderColor: COLORS[tone] }]} testID="listening-pill" accessibilityRole="text">
        <Text style={[styles.text, { color: COLORS[tone] }]} testID="listening-pill-text">
          {pillText(remaining, period, isFree)}
        </Text>
      </View>
      {banner ? (
        <TouchableOpacity
          testID="listening-banner"
          accessibilityRole="alert"
          style={[styles.banner, { backgroundColor: tone === "red" || tone === "out" ? "#fee2e2" : "#fef3c7" }]}
          onPress={() => setBanner(null)}
        >
          <Text style={styles.bannerText}>{banner} (tap to dismiss)</Text>
        </TouchableOpacity>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  pill: { alignSelf: "flex-start", borderWidth: 1, borderRadius: 14, paddingHorizontal: 10, paddingVertical: 3, marginVertical: 4 },
  text: { fontSize: 12, fontWeight: "700" },
  banner: { borderRadius: 8, padding: 8, marginBottom: 6 },
  bannerText: { fontSize: 13, color: "#111" },
});
