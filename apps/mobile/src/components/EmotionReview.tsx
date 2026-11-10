/**
 * Review mode of the Emotion dev tester: plays saved, still-unrated clips back
 * to back and lets the owner rate each with one big tap, one-handed, while
 * walking. THUMBS UP = the models' emotion label was right; THUMBS DOWN =
 * wrong (optionally pick "what was it really?" first). Each tap gives a
 * haptic and moves on to the next clip after a short gap. A clip with no tap
 * plays twice, then is skipped (stays unrated for next time).
 *
 * Screen locked: playback continues (background audio + lock-screen media
 * entry showing what the models said), but the OS lock-screen controls only
 * offer play/pause, so rating needs the screen on. The buttons are therefore
 * huge and each gives a haptic. Nothing here uploads audio.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Modal, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import * as Haptics from "expo-haptics";
import { createAudioPlayer, setAudioModeAsync } from "expo-audio";
import { EMOTION_LABELS, type EmotionLabel } from "../emotion/config";
import { emotionDevHub } from "../emotion/hub";
import { REVIEW_GAP_MS, ReviewSession, describeItem } from "../emotion/reviewSession";

interface Player {
  play(): void;
  pause(): void;
  remove(): void;
  addListener(event: string, cb: (s: { didJustFinish?: boolean }) => void): { remove(): void };
  setActiveForLockScreen?(active: boolean, meta?: { title?: string; artist?: string }): void;
}

const buzz = (kind: "up" | "down" | "soft") => {
  const run = kind === "soft" ? Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light) : Haptics.notificationAsync(kind === "up" ? Haptics.NotificationFeedbackType.Success : Haptics.NotificationFeedbackType.Warning);
  void Promise.resolve(run).catch(() => {});
};

export function EmotionReview({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  const session = useRef<ReviewSession | null>(null);
  const player = useRef<Player | null>(null);
  const gap = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [, bump] = useState(0);
  const rerender = () => bump((n) => n + 1);

  const stopPlayer = useCallback(() => {
    if (gap.current) clearTimeout(gap.current);
    gap.current = null;
    try {
      player.current?.setActiveForLockScreen?.(false);
      player.current?.remove();
    } catch {
      // already released
    }
    player.current = null;
  }, []);

  const playCurrent = useCallback(() => {
    stopPlayer();
    const cur = session.current?.current();
    if (!cur) {
      rerender();
      return;
    }
    try {
      const p = createAudioPlayer({ uri: cur.uri }) as unknown as Player;
      player.current = p;
      p.addListener("playbackStatusUpdate", (st) => {
        if (!st.didJustFinish || player.current !== p) return;
        const act = session.current?.onEnded();
        rerender();
        gap.current = setTimeout(playCurrent, REVIEW_GAP_MS);
        if (act === "done") stopPlayer();
      });
      p.setActiveForLockScreen?.(true, { title: describeItem(cur), artist: "MindShift emotion review" });
      p.play();
    } catch {
      // A clip that cannot play is skipped, never blocks the queue.
      session.current?.skip();
      gap.current = setTimeout(playCurrent, REVIEW_GAP_MS);
    }
    rerender();
  }, [stopPlayer]);

  useEffect(() => {
    if (!visible) return;
    session.current = new ReviewSession(emotionDevHub.reviewQueue(), {
      rate: (id, rating, truth) => emotionDevHub.rate(id, rating, truth),
    });
    void setAudioModeAsync({ allowsRecording: false, playsInSilentMode: true, shouldPlayInBackground: true } as never).catch(() => {});
    playCurrent();
    return () => {
      stopPlayer();
      session.current = null;
    };
  }, [visible, playCurrent, stopPlayer]);

  const s = session.current;
  const cur = s?.current() ?? null;

  const thumb = (up: boolean) => {
    if (!s?.thumb(up)) return;
    buzz(up ? "up" : "down");
    stopPlayer();
    rerender();
    gap.current = setTimeout(playCurrent, REVIEW_GAP_MS);
  };
  const skip = () => {
    s?.skip();
    buzz("soft");
    stopPlayer();
    rerender();
    gap.current = setTimeout(playCurrent, 300);
  };
  const pickTruth = (l: EmotionLabel) => {
    s?.setTruth(l);
    buzz("soft");
    rerender();
  };

  return (
    <Modal visible={visible} animationType="slide" onRequestClose={onClose}>
      <View style={styles.wrap} testID="emotion-review">
        <View style={styles.top}>
          <Text style={styles.count} testID="emotion-review-count">
            {cur && s ? `Clip ${s.position} of ${s.total}` : "All caught up"}
          </Text>
          <TouchableOpacity testID="emotion-review-close" onPress={onClose} accessibilityRole="button" hitSlop={12}>
            <Text style={styles.close}>Done</Text>
          </TouchableOpacity>
        </View>

        {cur ? (
          <>
            <Text style={styles.said} testID="emotion-review-said">
              {describeItem(cur)}
            </Text>
            <Text style={styles.hint}>Thumbs up = the label is right. Thumbs down = wrong.</Text>

            <View style={styles.thumbs}>
              <TouchableOpacity testID="emotion-review-down" accessibilityRole="button" accessibilityLabel="Wrong" style={[styles.thumb, styles.down]} onPress={() => thumb(false)}>
                <Text style={styles.thumbIcon}>👎</Text>
                <Text style={styles.thumbText}>Wrong</Text>
              </TouchableOpacity>
              <TouchableOpacity testID="emotion-review-up" accessibilityRole="button" accessibilityLabel="Right" style={[styles.thumb, styles.up]} onPress={() => thumb(true)}>
                <Text style={styles.thumbIcon}>👍</Text>
                <Text style={styles.thumbText}>Right</Text>
              </TouchableOpacity>
            </View>

            <Text style={styles.hint}>What was it really? (optional, then tap a thumb)</Text>
            <View style={styles.truths}>
              {EMOTION_LABELS.map((l) => (
                <TouchableOpacity key={l} testID={`emotion-review-truth-${l}`} style={[styles.truth, s?.pendingTruth === l && styles.truthOn]} onPress={() => pickTruth(l)}>
                  <Text style={[styles.truthText, s?.pendingTruth === l && styles.truthTextOn]}>{l}</Text>
                </TouchableOpacity>
              ))}
            </View>

            <TouchableOpacity testID="emotion-review-skip" accessibilityRole="button" style={styles.skip} onPress={skip}>
              <Text style={styles.skipText}>Skip</Text>
            </TouchableOpacity>
          </>
        ) : (
          <Text style={styles.said}>
            {s && s.total > 0 ? `Done. ${s.rated} rated, ${s.skipped} skipped (skipped ones come back next time).` : "No unrated saved clips. Turn on Save audio and have a conversation."}
          </Text>
        )}
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, backgroundColor: "#0F172A", padding: 20, paddingTop: 48 },
  top: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  count: { color: "#E5E7EB", fontSize: 18, fontWeight: "600" },
  close: { color: "#93C5FD", fontSize: 18, fontWeight: "600" },
  said: { color: "#FFFFFF", fontSize: 22, lineHeight: 30, marginTop: 24, fontWeight: "600" },
  hint: { color: "#9CA3AF", fontSize: 14, marginTop: 10 },
  thumbs: { flex: 1, flexDirection: "row", gap: 14, marginTop: 18, minHeight: 220 },
  thumb: { flex: 1, borderRadius: 24, alignItems: "center", justifyContent: "center" },
  up: { backgroundColor: "#15803D" },
  down: { backgroundColor: "#B91C1C" },
  thumbIcon: { fontSize: 72 },
  thumbText: { color: "#FFFFFF", fontSize: 24, fontWeight: "700", marginTop: 6 },
  truths: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 8 },
  truth: { paddingVertical: 14, paddingHorizontal: 16, borderRadius: 12, borderWidth: 1, borderColor: "#475569" },
  truthOn: { backgroundColor: "#2563EB", borderColor: "#2563EB" },
  truthText: { color: "#E5E7EB", fontSize: 16, fontWeight: "600" },
  truthTextOn: { color: "#FFFFFF" },
  skip: { marginTop: 14, paddingVertical: 18, borderRadius: 14, backgroundColor: "#334155", alignItems: "center" },
  skipText: { color: "#FFFFFF", fontSize: 18, fontWeight: "600" },
});
