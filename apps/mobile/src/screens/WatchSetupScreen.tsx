import React, { useCallback, useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Linking,
  ActivityIndicator,
} from "react-native";

import { claimWatchPairing } from "../api/watchPairing";

/** Play listing for the watch app — package name is permanent (see
 *  apps/watch/PORTED_FROM_GAUGE.md), branding changes never touch it. Play
 *  handles the remote-install-to-watch device picker on its own; nothing
 *  here talks to the watch directly. */
const WATCH_PLAY_URL =
  "https://play.google.com/store/apps/details?id=com.sagearbor.gauge.wear";

const PAIR_CODE_LENGTH = 6;

/** Uppercase and strip anything that isn't alphanumeric, capped to the
 *  server's 6-character code length — matches the watch's own
 *  code alphabet closely enough for a forgiving input (the server is the
 *  authority on which characters are actually valid). */
function sanitizeCode(raw: string): string {
  return raw
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, "")
    .slice(0, PAIR_CODE_LENGTH);
}

type ClaimOutcome = { ok: true } | { ok: false; detail: string };

interface WatchSetupScreenProps {
  onBack: () => void;
}

/** A small numbered badge ("1", "2") that gives the two cards below an
 *  explicit step order, so it always reads as "Install first, then pair" —
 *  not two equally-weighted unrelated options. */
function StepBadge({ n, done }: { n: number; done: boolean }) {
  return (
    <View style={[styles.badge, done && styles.badgeDone]}>
      <Text style={[styles.badgeText, done && styles.badgeTextDone]}>
        {done ? "✓" : n}
      </Text>
    </View>
  );
}

/**
 * Phase 3 Slice 1 — "Set up your watch": install the watch app via Play's
 * remote-install picker, then redeem the 6-character code the watch shows
 * (watch → Sign in) against the signed-in phone account. This screen never
 * polls anything — the WATCH polls `GET /me/pair/status` and finishes its
 * own sign-in within ~10s of a successful claim here.
 *
 * Deliberately two visually distinct, numbered steps rather than one flat
 * list: Install is the only strongly-emphasized action until it's been
 * tapped at least once, at which point it steps back (outline style, a
 * "Opened ✓" note) and the code field becomes the natural next thing to
 * fill in. Pairing itself both auto-submits the moment a valid 6-character
 * code is typed (least friction for the common case: read the code off the
 * watch, type it, done) AND keeps an explicit Pair button — needed for
 * retrying the same code after a transient failure, and as an unambiguous
 * affordance for anyone who prefers to press something.
 */
export default function WatchSetupScreen({ onBack }: WatchSetupScreenProps) {
  const [code, setCode] = useState("");
  const [claiming, setClaiming] = useState(false);
  const [result, setResult] = useState<ClaimOutcome | null>(null);
  const [installOpened, setInstallOpened] = useState(false);

  const handleInstall = useCallback(() => {
    setInstallOpened(true);
    void Linking.openURL(WATCH_PLAY_URL);
  }, []);

  const attemptPair = useCallback(
    async (candidate: string) => {
      if (candidate.length !== PAIR_CODE_LENGTH || claiming) return;
      setClaiming(true);
      setResult(null);
      try {
        const res = await claimWatchPairing(candidate);
        setResult(res.ok ? { ok: true } : { ok: false, detail: res.detail! });
      } catch {
        // The client itself never throws for expected pairing failures —
        // this is an unexpected transport failure (offline, DNS, etc).
        setResult({
          ok: false,
          detail: "Couldn't reach the server. Check your connection and try again.",
        });
      } finally {
        setClaiming(false);
      }
    },
    [claiming],
  );

  const handleChangeCode = useCallback(
    (raw: string) => {
      const next = sanitizeCode(raw);
      setCode(next);
      // A fresh edit means the previous attempt's outcome no longer
      // describes what's in the box — clear it rather than leaving a stale
      // success/error showing next to a code that's since changed.
      setResult(null);
      if (next.length === PAIR_CODE_LENGTH) {
        // Least-friction path: a full code read straight off the watch
        // needs no extra tap. The button below still works for a manual
        // retry of the same code.
        void attemptPair(next);
      }
    },
    [attemptPair],
  );

  const handlePairPress = useCallback(() => {
    void attemptPair(code);
  }, [attemptPair, code]);

  const pairDisabled = code.length !== PAIR_CODE_LENGTH || claiming;

  return (
    <ScrollView
      style={styles.flex}
      contentContainerStyle={styles.content}
      testID="watch-setup-screen"
    >
      <TouchableOpacity
        testID="watch-setup-back"
        accessibilityRole="button"
        style={styles.backButton}
        onPress={onBack}
      >
        <Text style={styles.backText}>← Back</Text>
      </TouchableOpacity>

      <Text style={styles.heading}>Set up your watch</Text>

      <View style={styles.row}>
        <View style={styles.rowHeader}>
          <StepBadge n={1} done={installOpened} />
          <Text style={styles.rowTitle}>Install the watch app</Text>
        </View>
        <Text style={styles.rowSub}>
          Opens Google Play — choose your watch when prompted.
        </Text>
        <TouchableOpacity
          testID="watch-install-button"
          accessibilityRole="button"
          style={[
            styles.installButton,
            installOpened && styles.installButtonOpened,
          ]}
          onPress={handleInstall}
        >
          <Text
            style={[
              styles.installButtonText,
              installOpened && styles.installButtonTextOpened,
            ]}
          >
            {installOpened ? "Open Google Play again" : "Install the watch app"}
          </Text>
        </TouchableOpacity>
        {installOpened ? (
          <Text style={styles.openedNote} testID="watch-install-opened">
            Opened Google Play — install on your watch, then come back here
            to pair it.
          </Text>
        ) : null}
      </View>

      <View style={styles.row}>
        <View style={styles.rowHeader}>
          <StepBadge n={2} done={result?.ok === true} />
          <Text style={styles.rowTitle}>Pair your watch</Text>
        </View>
        <Text style={styles.rowSub}>
          On the watch, tap “Sign in” — it will show a 6-character code.
          Type it below; it pairs automatically once all 6 are in.
        </Text>

        <TextInput
          testID="watch-pair-code-input"
          style={styles.codeInput}
          value={code}
          onChangeText={handleChangeCode}
          placeholder="CODE"
          placeholderTextColor="#9CA3AF"
          autoCapitalize="characters"
          autoCorrect={false}
          autoComplete="off"
          maxLength={PAIR_CODE_LENGTH}
          editable={!claiming}
        />
        <Text style={styles.codeHint}>6 letters/numbers, from the watch</Text>

        <TouchableOpacity
          testID="watch-pair-button"
          accessibilityRole="button"
          style={[styles.pairButton, pairDisabled && styles.pairButtonDisabled]}
          onPress={handlePairPress}
          disabled={pairDisabled}
        >
          {claiming ? (
            <ActivityIndicator size="small" color="#FFFFFF" />
          ) : (
            <Text style={styles.pairButtonText}>Pair</Text>
          )}
        </TouchableOpacity>

        {result?.ok ? (
          <Text style={styles.successText} testID="watch-pair-success">
            Watch paired ✓ Your watch will finish signing in on its own,
            usually within about 10 seconds.
          </Text>
        ) : null}
        {result && !result.ok ? (
          <Text style={styles.errorText} testID="watch-pair-error">
            {result.detail}
          </Text>
        ) : null}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  flex: {
    flex: 1,
  },
  content: {
    paddingTop: 24,
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  backButton: {
    alignSelf: "flex-start",
    minHeight: 44,
    justifyContent: "center",
    paddingRight: 12,
    marginBottom: 4,
  },
  backText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#4A90D9",
  },
  heading: {
    fontSize: 24,
    fontWeight: "700",
    color: "#111827",
    marginBottom: 20,
  },
  row: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 14,
    backgroundColor: "#FFFFFF",
    padding: 18,
    marginBottom: 12,
  },
  rowHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  badge: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: "#4A90D9",
    alignItems: "center",
    justifyContent: "center",
  },
  badgeDone: {
    backgroundColor: "#10B981",
  },
  badgeText: {
    fontSize: 13,
    fontWeight: "700",
    color: "#FFFFFF",
  },
  badgeTextDone: {
    color: "#FFFFFF",
  },
  rowTitle: {
    fontSize: 17,
    fontWeight: "600",
    color: "#1F2937",
  },
  rowSub: {
    marginTop: 8,
    fontSize: 13.5,
    lineHeight: 19,
    color: "#6B7280",
  },
  installButton: {
    marginTop: 14,
    minHeight: 48,
    borderRadius: 10,
    backgroundColor: "#4A90D9",
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 16,
  },
  // Once Install has been tapped once, it steps back to a secondary/outline
  // style — Pair becomes the natural next primary action instead of two
  // equally-loud solid buttons competing for attention.
  installButtonOpened: {
    backgroundColor: "#FFFFFF",
    borderWidth: 1.5,
    borderColor: "#4A90D9",
  },
  installButtonText: {
    fontSize: 15,
    fontWeight: "700",
    color: "#FFFFFF",
  },
  installButtonTextOpened: {
    color: "#4A90D9",
  },
  openedNote: {
    marginTop: 10,
    fontSize: 12.5,
    lineHeight: 18,
    color: "#10B981",
    fontWeight: "600",
  },
  codeInput: {
    marginTop: 14,
    minHeight: 52,
    borderWidth: 1.5,
    borderColor: "#D1D5DB",
    borderRadius: 10,
    paddingHorizontal: 14,
    fontSize: 22,
    fontWeight: "700",
    letterSpacing: 6,
    color: "#1F2937",
    textAlign: "center",
  },
  codeHint: {
    marginTop: 6,
    fontSize: 12,
    color: "#9CA3AF",
    textAlign: "center",
  },
  pairButton: {
    marginTop: 12,
    minHeight: 48,
    borderRadius: 10,
    backgroundColor: "#4A90D9",
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 16,
  },
  pairButtonDisabled: {
    backgroundColor: "#B9CFE8",
  },
  pairButtonText: {
    fontSize: 15,
    fontWeight: "700",
    color: "#FFFFFF",
  },
  successText: {
    marginTop: 12,
    fontSize: 13.5,
    lineHeight: 19,
    color: "#10B981",
    fontWeight: "600",
  },
  errorText: {
    marginTop: 12,
    fontSize: 13.5,
    lineHeight: 19,
    color: "#DC2626",
  },
});
