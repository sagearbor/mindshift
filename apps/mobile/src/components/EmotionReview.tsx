/**
 * Review mode of the Emotion dev tester: plays saved clips back to back and
 * lets a labeller (the owner, his mother, anyone) rate each with big one-handed
 * taps, e.g. while walking.
 *
 *  - "Right" = the shown label was right; an emotion chip = it was really THAT
 *    (a thumbs-down and the truth in one tap). Both auto-advance.
 *  - "Emotion + nudge" mode adds one more tap: "Would a nudge help here?"
 *    Yes / No / Not sure. "Emotion only" is one tap per clip.
 *  - Setup: who is labelling (one label per person per clip), which day,
 *    mode, and whether to hide the model's guess (default on for non-owners).
 *  - Skip, Replay, Undo last. A clip with no tap plays twice, then is skipped
 *    (stays unlabelled for next time).
 *
 * Screen locked: playback continues (background audio + lock-screen entry) but
 * rating needs the screen on. Nothing here uploads audio. All touch targets
 * are >= 64 px; the layout is a single column that fits 412 px without
 * horizontal scrolling.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Modal, ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View, useColorScheme } from "react-native";
import * as Haptics from "expo-haptics";
import { createAudioPlayer, setAudioModeAsync } from "expo-audio";
import { EMOTION_LABELS, type EmotionLabel } from "../emotion/config";
import type { ReviewItemFull } from "../emotion/devStore";
import { nativeBackend, reviewPrefs, type ReviewBackend } from "../emotion/reviewBackend";
import {
  ALL_DAYS,
  REVIEW_GAP_MS,
  ReviewSession,
  contextLine,
  defaultDay,
  buildQueue,
  defaultHideGuess,
  describeItem,
  listDays,
  raterKey,
  smallTop,
  tinyTop,
  type Day,
  type RaterKind,
  type ReviewMode,
} from "../emotion/reviewSession";

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

const CHIP_TEXT: Record<EmotionLabel, string> = {
  anger: "Anger",
  disgust: "Disgust",
  fear: "Fear",
  happiness: "Happiness",
  neutral: "Neutral / nothing",
  sadness: "Sadness",
};

const palette = (dark: boolean) =>
  dark
    ? { bg: "#0F172A", text: "#FFFFFF", sub: "#9CA3AF", card: "#1E293B", line: "#475569", btn: "#334155", on: "#2563EB", link: "#93C5FD" }
    : { bg: "#F8FAFC", text: "#0F172A", sub: "#475569", card: "#E2E8F0", line: "#94A3B8", btn: "#CBD5E1", on: "#2563EB", link: "#1D4ED8" };

function Big({ id, label, onPress, style, textStyle, small }: { id: string; label: string; onPress: () => void; style: object; textStyle: object; small?: boolean }) {
  return (
    <TouchableOpacity testID={id} accessibilityRole="button" accessibilityLabel={label} style={[styles.big, small && styles.bigSmall, style]} onPress={onPress}>
      <Text style={[styles.bigText, textStyle]}>{label}</Text>
    </TouchableOpacity>
  );
}

export function EmotionReview({ visible, onClose, backend }: { visible: boolean; onClose: () => void; backend?: ReviewBackend | null }) {
  const c = palette(useColorScheme() !== "light");
  const be = useRef<ReviewBackend | null>(null);
  const session = useRef<ReviewSession | null>(null);
  const player = useRef<Player | null>(null);
  const gap = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [, bump] = useState(0);
  const rerender = () => bump((n) => n + 1);

  const [kind, setKind] = useState<RaterKind>(reviewPrefs.kind);
  const [name, setName] = useState(reviewPrefs.name);
  const [mode, setMode] = useState<ReviewMode>(reviewPrefs.mode);
  const [hideOverride, setHideOverride] = useState<boolean | null>(reviewPrefs.hideGuess);
  const [dayChoice, setDayChoice] = useState<Day | null>(null);
  const [setupOpen, setSetupOpen] = useState(true);
  const [items, setItems] = useState<ReviewItemFull[]>([]);

  const rater = raterKey(kind, name);
  const hideGuess = hideOverride ?? defaultHideGuess(rater);
  const day = dayChoice ?? defaultDay(items, rater);
  const days = useMemo(() => listDays(items, rater), [items, rater]);

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
      p.setActiveForLockScreen?.(true, { title: hideGuessRef.current ? "Clip to label" : describeItem(cur), artist: "MindShift emotion review" });
      p.play();
    } catch {
      // A clip that cannot play is skipped, never blocks the queue.
      session.current?.skip();
      gap.current = setTimeout(playCurrent, REVIEW_GAP_MS);
    }
    rerender();
  }, [stopPlayer]);
  const hideGuessRef = useRef(hideGuess);
  hideGuessRef.current = hideGuess;

  const reload = useCallback(() => {
    be.current = backend ?? be.current ?? nativeBackend();
    setItems(be.current?.items() ?? []);
  }, [backend]);

  // Open: load clips, show setup, no sound until "Start".
  useEffect(() => {
    if (!visible) return;
    reload();
    setSetupOpen(true);
    setDayChoice(null);
    void setAudioModeAsync({ allowsRecording: false, playsInSilentMode: true, shouldPlayInBackground: true } as never).catch(() => {});
    return () => {
      stopPlayer();
      session.current = null;
    };
  }, [visible, reload, stopPlayer]);

  const start = () => {
    reviewPrefs.kind = kind;
    reviewPrefs.name = name;
    reviewPrefs.mode = mode;
    reviewPrefs.hideGuess = hideOverride;
    stopPlayer();
    const fresh = be.current?.items() ?? [];
    setItems(fresh);
    const d = dayChoice ?? defaultDay(fresh, rater);
    const b = be.current;
    session.current = new ReviewSession(
      buildQueue(fresh, rater, d),
      { save: (id, l) => b?.save(id, l), clear: (id, r) => b?.clear(id, r) },
      { rater, mode },
    );
    setSetupOpen(false);
    playCurrent();
  };

  const s = session.current;
  const cur = s?.current() ?? null;

  const after = (haptic: "up" | "down" | "soft", delay = 300) => {
    buzz(haptic);
    stopPlayer();
    rerender();
    gap.current = setTimeout(playCurrent, delay);
  };
  const first = (tap: Parameters<ReviewSession["answer"]>[0]) => {
    const r = s?.answer(tap);
    if (!r) return;
    if (r === "saved") after("soft");
    else {
      buzz("soft");
      rerender();
    }
  };
  const nudge = (v: "yes" | "no" | "unsure") => {
    if (s?.answerNudge(v)) after("soft");
  };
  const skip = () => {
    s?.skip();
    after("soft");
  };
  const undo = () => {
    if (!s?.undo()) return;
    buzz("soft");
    stopPlayer();
    rerender();
    gap.current = setTimeout(playCurrent, 200);
  };
  const replay = () => {
    s?.replay();
    buzz("soft");
    playCurrent();
  };

  const said = cur ? (hideGuess ? "Listen, then tap what you hear." : describeItem(cur)) : "";
  const differ = cur ? smallTop(cur) !== null && smallTop(cur) !== tinyTop(cur) : false;
  const sum = s?.summary();
  const pct = (a: { agree: number; of: number }) => (a.of ? `${Math.round((100 * a.agree) / a.of)}% (${a.agree}/${a.of})` : "n/a");

  return (
    <Modal visible={visible} animationType="slide" onRequestClose={onClose}>
      <View style={[styles.wrap, { backgroundColor: c.bg }]} testID="emotion-review">
        <View style={styles.top}>
          <Text style={[styles.count, { color: c.text }]} testID="review-count">
            {setupOpen ? "Review" : cur && s ? `${s.position} of ${s.total}` : "All caught up"}
          </Text>
          <TouchableOpacity testID="review-close" onPress={onClose} accessibilityRole="button" style={styles.topBtn}>
            <Text style={[styles.close, { color: c.link }]}>Done</Text>
          </TouchableOpacity>
        </View>
        {!setupOpen && s ? (
          <View style={[styles.barBg, { backgroundColor: c.card }]} testID="review-progress">
            <View style={[styles.barFill, { width: `${Math.round(s.ratio * 100)}%`, backgroundColor: c.on }]} />
          </View>
        ) : null}

        <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
          {setupOpen ? (
            <View testID="review-setup">
              <Text style={[styles.h, { color: c.text }]}>Who is labelling?</Text>
              <View style={styles.row}>
                {(["me", "mom", "other"] as const).map((k) => (
                  <Big key={k} id={`review-rater-${k}`} label={k === "me" ? "Me" : k === "mom" ? "Mom" : "Other"} small
                    style={{ backgroundColor: kind === k ? c.on : c.btn, flex: 1 }} textStyle={{ color: kind === k ? "#FFFFFF" : c.text }}
                    onPress={() => { setKind(k); setDayChoice(null); }} />
                ))}
              </View>
              {kind === "other" ? (
                <TextInput testID="review-rater-name" value={name} onChangeText={setName} placeholder="Name" placeholderTextColor={c.sub}
                  style={[styles.input, { color: c.text, borderColor: c.line }]} autoCapitalize="words" />
              ) : null}

              <Text style={[styles.h, { color: c.text }]}>What to label</Text>
              <View style={styles.row}>
                <Big id="review-mode-emotion" label="Emotion only" small style={{ backgroundColor: mode === "emotion" ? c.on : c.btn, flex: 1 }} textStyle={{ color: mode === "emotion" ? "#FFFFFF" : c.text }} onPress={() => setMode("emotion")} />
                <Big id="review-mode-nudge" label="Emotion + nudge" small style={{ backgroundColor: mode === "emotion+nudge" ? c.on : c.btn, flex: 1 }} textStyle={{ color: mode === "emotion+nudge" ? "#FFFFFF" : c.text }} onPress={() => setMode("emotion+nudge")} />
              </View>
              <Text style={[styles.hint, { color: c.sub }]}>{mode === "emotion" ? "One tap per clip." : "Two taps per clip: the emotion, then whether a nudge would help."}</Text>

              <TouchableOpacity testID="review-hide-guess" accessibilityRole="switch" accessibilityState={{ checked: hideGuess }} style={[styles.toggle, { backgroundColor: c.card }]} onPress={() => setHideOverride(!hideGuess)}>
                <Text style={[styles.bigText, { color: c.text }]}>{hideGuess ? "Model's guess: hidden" : "Model's guess: shown"}</Text>
                <Text style={[styles.hint, { color: c.sub }]}>Hidden is best for a second labeller (no bias). Tap to change.</Text>
              </TouchableOpacity>

              <Text style={[styles.h, { color: c.text }]}>Day</Text>
              <View style={styles.row}>
                {days.map((d) => (
                  <Big key={d.day} id={`review-day-${d.day}`} label={`${d.day.slice(5)}  (${d.unrated})`} small
                    style={{ backgroundColor: day === d.day ? c.on : c.btn, minWidth: "30%" }} textStyle={{ color: day === d.day ? "#FFFFFF" : c.text }}
                    onPress={() => setDayChoice(d.day)} />
                ))}
                <Big id="review-day-all" label={`All  (${days.reduce((a, d) => a + d.unrated, 0)})`} small
                  style={{ backgroundColor: day === ALL_DAYS ? c.on : c.btn, minWidth: "30%" }} textStyle={{ color: day === ALL_DAYS ? "#FFFFFF" : c.text }}
                  onPress={() => setDayChoice(ALL_DAYS)} />
              </View>
              <Text style={[styles.hint, { color: c.sub }]}>Numbers are clips you have not labelled yet.</Text>

              <Big id="review-start" label="Start labelling" style={{ backgroundColor: "#15803D", marginTop: 18 }} textStyle={{ color: "#FFFFFF" }} onPress={start} />
              {items.length === 0 ? <Text style={[styles.hint, { color: c.sub }]}>No saved clips yet. Turn on Save audio and have a conversation.</Text> : null}
            </View>
          ) : cur && s ? (
            <>
              <Text style={[styles.said, { color: c.text }]} testID="review-said">{said}</Text>
              <Text style={[styles.hint, { color: c.sub }]} testID="review-context">{contextLine(cur)}</Text>

              {s.stage === "emotion" ? (
                <>
                  {!hideGuess ? (
                    differ ? (
                      <View style={styles.row}>
                        <Big id="review-right-tiny" label="TINY right" style={{ backgroundColor: "#15803D", flex: 1 }} textStyle={{ color: "#FFFFFF" }} onPress={() => first({ right: "tiny" })} />
                        <Big id="review-right-small" label="SMALL right" style={{ backgroundColor: "#15803D", flex: 1 }} textStyle={{ color: "#FFFFFF" }} onPress={() => first({ right: "small" })} />
                      </View>
                    ) : (
                      <Big id="review-right" label="👍  Right" style={{ backgroundColor: "#15803D", minHeight: 96 }} textStyle={{ color: "#FFFFFF", fontSize: 26 }} onPress={() => first({ right: "both" })} />
                    )
                  ) : null}
                  <Text style={[styles.hint, { color: c.sub }]}>{hideGuess ? "What was it?" : "Not right? Tap what it really was:"}</Text>
                  <View style={styles.grid}>
                    {EMOTION_LABELS.map((l) => (
                      <Big key={l} id={`review-chip-${l}`} label={CHIP_TEXT[l]} style={{ backgroundColor: c.btn, width: "48.5%" }} textStyle={{ color: c.text }} onPress={() => first({ truth: l })} />
                    ))}
                  </View>
                </>
              ) : (
                <>
                  <Text style={[styles.h, { color: c.text }]} testID="review-nudge-question">Would a nudge help here?</Text>
                  <Text style={[styles.hint, { color: c.sub }]}>Emotion: {s.pendingTruth ? CHIP_TEXT[s.pendingTruth] : "right"}</Text>
                  <View style={styles.row}>
                    <Big id="review-nudge-yes" label="Yes" style={{ backgroundColor: "#15803D", flex: 1, minHeight: 120 }} textStyle={{ color: "#FFFFFF", fontSize: 24 }} onPress={() => nudge("yes")} />
                    <Big id="review-nudge-no" label="No" style={{ backgroundColor: "#B91C1C", flex: 1, minHeight: 120 }} textStyle={{ color: "#FFFFFF", fontSize: 24 }} onPress={() => nudge("no")} />
                    <Big id="review-nudge-unsure" label="Not sure" style={{ backgroundColor: c.btn, flex: 1, minHeight: 120 }} textStyle={{ color: c.text, fontSize: 20 }} onPress={() => nudge("unsure")} />
                  </View>
                  <Big id="review-back" label="Back (change emotion)" small style={{ backgroundColor: c.card }} textStyle={{ color: c.text }} onPress={() => { s.backToEmotion(); rerender(); }} />
                </>
              )}

              <View style={styles.row}>
                <Big id="review-undo" label="Undo last" small style={{ backgroundColor: c.card, flex: 1, opacity: s.canUndo ? 1 : 0.4 }} textStyle={{ color: c.text }} onPress={undo} />
                <Big id="review-replay" label="Replay" small style={{ backgroundColor: c.card, flex: 1 }} textStyle={{ color: c.text }} onPress={replay} />
                <Big id="review-skip" label="Skip" small style={{ backgroundColor: c.card, flex: 1 }} textStyle={{ color: c.text }} onPress={skip} />
              </View>
            </>
          ) : (
            <View testID="review-summary">
              {s && s.total > 0 && sum ? (
                <>
                  <Text style={[styles.said, { color: c.text }]}>Done: {sum.rated} labelled, {sum.skipped} skipped.</Text>
                  <Text style={[styles.hint, { color: c.sub }]}>Agreement with TINY: {pct(sum.tiny)}</Text>
                  <Text style={[styles.hint, { color: c.sub }]}>Agreement with SMALL: {pct(sum.small)}</Text>
                  <Text style={[styles.hint, { color: c.sub }]}>Skipped clips come back next time.</Text>
                </>
              ) : (
                <Text style={[styles.said, { color: c.text }]}>No unlabelled clips for this day. Pick another day or turn on Save audio.</Text>
              )}
              <View style={styles.row}>
                {s && s.canUndo ? <Big id="review-undo" label="Undo last" small style={{ backgroundColor: c.card, flex: 1 }} textStyle={{ color: c.text }} onPress={undo} /> : null}
                <Big id="review-setup-toggle" label="Setup / other day" small style={{ backgroundColor: c.card, flex: 1 }} textStyle={{ color: c.text }}
                  onPress={() => { stopPlayer(); reload(); setDayChoice(null); setSetupOpen(true); }} />
              </View>
            </View>
          )}
          {!setupOpen && cur ? (
            <TouchableOpacity testID="review-setup-toggle-open" accessibilityRole="button" style={styles.linkBtn} onPress={() => { stopPlayer(); reload(); setSetupOpen(true); }}>
              <Text style={[styles.hint, { color: c.link }]}>Setup ({rater}, {mode === "emotion" ? "emotion only" : "emotion + nudge"})</Text>
            </TouchableOpacity>
          ) : null}
        </ScrollView>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  wrap: { flex: 1, paddingHorizontal: 16, paddingTop: 44 },
  top: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", minHeight: 64 },
  topBtn: { minHeight: 64, minWidth: 64, alignItems: "flex-end", justifyContent: "center" },
  count: { fontSize: 20, fontWeight: "700" },
  close: { fontSize: 18, fontWeight: "600" },
  barBg: { height: 10, borderRadius: 5, overflow: "hidden" },
  barFill: { height: 10, borderRadius: 5 },
  scroll: { paddingBottom: 40, gap: 12 },
  h: { fontSize: 18, fontWeight: "700", marginTop: 14 },
  said: { fontSize: 21, lineHeight: 28, marginTop: 14, fontWeight: "600" },
  hint: { fontSize: 14, marginTop: 6 },
  row: { flexDirection: "row", flexWrap: "wrap", gap: 10, marginTop: 8 },
  grid: { flexDirection: "row", flexWrap: "wrap", gap: 10, justifyContent: "space-between" },
  big: { minHeight: 76, borderRadius: 18, alignItems: "center", justifyContent: "center", paddingHorizontal: 12 },
  bigSmall: { minHeight: 64, borderRadius: 14 },
  bigText: { fontSize: 18, fontWeight: "700", textAlign: "center" },
  input: { minHeight: 64, borderWidth: 1, borderRadius: 14, paddingHorizontal: 14, fontSize: 18, marginTop: 10 },
  toggle: { minHeight: 64, borderRadius: 14, padding: 12, marginTop: 12, justifyContent: "center" },
  linkBtn: { minHeight: 64, alignItems: "center", justifyContent: "center" },
});
