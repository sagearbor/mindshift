/**
 * Settings -> Advanced -> "Emotion dev tester". A research instrument for the
 * owner to judge two experimental emotion models while wearing the watch; OFF
 * by default, labelled as experimental, and honest about what it is not (see
 * docs/plans/2026-11-07-emotion-dev-mode.md). Nothing here uploads audio.
 */
import { useCallback, useEffect, useState } from "react";
import { Alert, Share, StyleSheet, Switch, Text, TouchableOpacity, View } from "react-native";
import { DEFAULT_USER_SETTINGS, type EmotionDevUserSettings } from "../emotion/config";
import { loadEmotionDevSettings, saveEmotionDevSettings } from "../emotion/devPrefs";
import type { DevStatus } from "../emotion/devHub";
import { emotionDevHub } from "../emotion/hub";

export function describeStatus(s: DevStatus): string {
  switch (s.state) {
    case "off":
      return "Off.";
    case "starting":
      return "Loading the models…";
    case "waiting-for-audio":
      return "On. Waiting for a Live Coach session to supply microphone audio.";
    case "models-missing":
      return s.message;
    case "error":
      return `Error — ${s.message}`;
    case "running": {
      const sn = s.snapshot;
      const small = s.models.small ? "SMALL running silently" : `SMALL off (${s.models.smallReason ?? "not loaded"})`;
      const last = sn.lastLabel ? `latest ${sn.lastLabel} ${Math.round((sn.lastProb ?? 0) * 100)}%` : "no speech heard yet";
      return `${sn.paused ? "PAUSED. " : ""}TINY nudging, ${small}. ${sn.windows} windows, ${sn.nudges} buzzes, ${last}.`;
    }
  }
}

export function EmotionDevPanel({ userId }: { userId: string | null }) {
  const [settings, setSettings] = useState<EmotionDevUserSettings>(DEFAULT_USER_SETTINGS);
  const [status, setStatus] = useState<DevStatus>({ state: "off" });
  const [usage, setUsage] = useState<{ clips: number; clipBytes: number; logBytes: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    void loadEmotionDevSettings(userId).then((s) => {
      if (cancelled) return;
      setSettings(s);
      void emotionDevHub.configure(s);
    });
    return () => {
      cancelled = true;
    };
  }, [userId]);

  useEffect(() => emotionDevHub.subscribe(setStatus), []);
  useEffect(() => {
    setUsage(emotionDevHub.usage());
  }, [status]);

  const update = useCallback(
    (next: EmotionDevUserSettings) => {
      setSettings(next);
      void saveEmotionDevSettings(userId, next);
      void emotionDevHub.configure(next);
    },
    [userId],
  );

  const paused = status.state === "running" && status.snapshot.paused;
  const running = status.state === "running";

  const confirmDelete = () =>
    Alert.alert("Delete all emotion-tester data?", "Removes every saved clip and the whole log from this phone. Cannot be undone.", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete all",
        style: "destructive",
        onPress: () => {
          emotionDevHub.deleteAll();
          setUsage(emotionDevHub.usage());
        },
      },
    ]);

  const exportLog = async () => {
    const text = emotionDevHub.exportLogText();
    if (!text) {
      Alert.alert("Nothing to export", "The log is empty.");
      return;
    }
    // Explicit export of the LOG (probabilities + flags). Audio clips never leave via this button.
    await Share.share({ message: text, title: "emotion-log.jsonl" }).catch(() => {});
  };

  const mb = (b: number) => (b / (1024 * 1024)).toFixed(1);

  return (
    <View style={styles.box} testID="emotion-dev-panel">
      <View style={styles.switchRow}>
        <View style={styles.switchText}>
          <Text style={styles.title}>Emotion dev tester</Text>
          <Text style={styles.sub}>
            Experimental. Two on-phone voice-emotion models listen to your Live Coach audio; the small one nudges your watch (rate-capped), the big one
            runs silently. They over-predict “neutral” and are NOT validated on real speech — this is for judging them, not for trusting them.
          </Text>
        </View>
        <Switch testID="emotion-dev-switch" value={settings.enabled} onValueChange={(on) => update({ ...settings, enabled: on })} />
      </View>

      <Text style={styles.status} testID="emotion-dev-status">
        {describeStatus(status)}
      </Text>

      {settings.enabled ? (
        <>
          <View style={styles.line}>
            <Text style={styles.label}>Buzzes per hour (max)</Text>
            <View style={styles.stepper}>
              <TouchableOpacity testID="emotion-dev-less" onPress={() => update({ ...settings, maxNudgesPerHour: Math.max(0, settings.maxNudgesPerHour - 1) })}>
                <Text style={styles.stepBtn}>−</Text>
              </TouchableOpacity>
              <Text style={styles.stepVal}>{settings.maxNudgesPerHour}</Text>
              <TouchableOpacity testID="emotion-dev-more" onPress={() => update({ ...settings, maxNudgesPerHour: Math.min(60, settings.maxNudgesPerHour + 1) })}>
                <Text style={styles.stepBtn}>+</Text>
              </TouchableOpacity>
            </View>
          </View>
          <View style={styles.line}>
            <Text style={styles.label}>Show a one-line suggestion on the watch</Text>
            <Switch value={settings.nudgeSuggestion} onValueChange={(v) => update({ ...settings, nudgeSuggestion: v })} />
          </View>

          <View style={styles.buttons}>
            <TouchableOpacity testID="emotion-dev-pause" disabled={!running} style={[styles.btn, !running && styles.btnOff]} onPress={() => emotionDevHub.setPaused(!paused)}>
              <Text style={styles.btnText}>{paused ? "Resume" : "Pause"}</Text>
            </TouchableOpacity>
            <TouchableOpacity testID="emotion-dev-flag-bad" disabled={!running || paused} style={[styles.btn, (!running || paused) && styles.btnOff]} onPress={() => emotionDevHub.flag("bad")}>
              <Text style={styles.btnText}>✗ Wrong</Text>
            </TouchableOpacity>
            <TouchableOpacity testID="emotion-dev-flag-good" disabled={!running || paused} style={[styles.btn, (!running || paused) && styles.btnOff]} onPress={() => emotionDevHub.flag("good")}>
              <Text style={styles.btnText}>✓ Good</Text>
            </TouchableOpacity>
            <TouchableOpacity testID="emotion-dev-flag-ignore" disabled={!running || paused} style={[styles.btn, (!running || paused) && styles.btnOff]} onPress={() => emotionDevHub.flag("ignore")}>
              <Text style={styles.btnText}>Ignore (media)</Text>
            </TouchableOpacity>
          </View>
          <Text style={styles.sub}>
            A flag keeps the 45 s before and 15 s after on this phone only. An un-tapped buzz is logged as “unrated”. Media playing is not detected —
            pause or tap Ignore.
          </Text>

          <Text style={styles.sub} testID="emotion-dev-usage">
            {usage ? `${usage.clips} clips · ${mb(usage.clipBytes)} MB audio · ${mb(usage.logBytes)} MB log` : "No data stored."}
          </Text>
          <View style={styles.buttons}>
            <TouchableOpacity testID="emotion-dev-export" style={styles.btn} onPress={exportLog}>
              <Text style={styles.btnText}>Export log</Text>
            </TouchableOpacity>
            <TouchableOpacity testID="emotion-dev-delete" style={[styles.btn, styles.btnDanger]} onPress={confirmDelete}>
              <Text style={styles.btnText}>Delete all</Text>
            </TouchableOpacity>
            {status.state === "models-missing" ? (
              <TouchableOpacity testID="emotion-dev-retry" style={styles.btn} onPress={() => emotionDevHub.retry()}>
                <Text style={styles.btnText}>Check again</Text>
              </TouchableOpacity>
            ) : null}
          </View>
        </>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  box: { borderWidth: 1, borderColor: "#D1D5DB", borderRadius: 14, backgroundColor: "#FFFFFF", padding: 18, marginBottom: 12 },
  switchRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 12 },
  switchText: { flex: 1 },
  title: { fontSize: 17, fontWeight: "600", color: "#1F2937" },
  sub: { marginTop: 4, fontSize: 13.5, lineHeight: 19, color: "#6B7280" },
  status: { marginTop: 10, fontSize: 14, lineHeight: 20, color: "#111827" },
  line: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginTop: 12, gap: 12 },
  label: { flex: 1, fontSize: 14.5, color: "#1F2937" },
  stepper: { flexDirection: "row", alignItems: "center", gap: 14 },
  stepBtn: { fontSize: 24, color: "#2563EB", paddingHorizontal: 8 },
  stepVal: { fontSize: 17, fontWeight: "600", minWidth: 24, textAlign: "center" },
  buttons: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 12 },
  btn: { backgroundColor: "#2563EB", borderRadius: 10, paddingVertical: 10, paddingHorizontal: 14 },
  btnOff: { backgroundColor: "#9CA3AF" },
  btnDanger: { backgroundColor: "#B91C1C" },
  btnText: { color: "#FFFFFF", fontWeight: "600", fontSize: 14.5 },
});
