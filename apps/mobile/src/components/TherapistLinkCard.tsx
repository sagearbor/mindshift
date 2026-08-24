import React, { useCallback, useEffect, useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  Switch,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import {
  getTherapistLink,
  setTherapistLink,
  setAutoShare,
  unlinkTherapist,
  type TherapistLink,
} from "../api/therapist";

function humanize(err: unknown): string {
  const e = err as { detail?: string; status?: number };
  if (typeof e?.detail === "string" && e.detail) return e.detail;
  if (e?.status === 401) return "Please sign in again.";
  if (e?.status === 503) return "Linking isn’t available right now.";
  return "Something went wrong — please try again.";
}

/**
 * Settings → "My therapist": the patient names ONE therapist account by
 * email. Once linked, "Share sessions automatically" (default on) makes
 * every finished live session and stored recording a normal read-only
 * share to that account at ingest — the same per-episode grant Replay's
 * "Share with…" makes, so each one can still be revoked there. The
 * therapist accepts from their own dashboard; until then the row says
 * "waiting for them to accept" (auto-share already applies — the patient
 * chose the recipient, exactly as a manual share does).
 */
export default function TherapistLinkCard() {
  // null = still loading / couldn't be determined (offline, 401, 503).
  const [link, setLink] = useState<TherapistLink | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getTherapistLink()
      .then((l) => {
        if (!cancelled) setLink(l);
      })
      .catch((e) => {
        if (!cancelled) {
          setLink(null);
          setLoadError(humanize(e));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const submit = useCallback(async () => {
    const trimmed = email.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setError(null);
    try {
      const l = await setTherapistLink(trimmed);
      setLink(l);
      setEmail("");
    } catch (e) {
      setError(humanize(e));
    } finally {
      setBusy(false);
    }
  }, [email, busy]);

  const toggleAuto = useCallback(
    async (on: boolean) => {
      if (!link?.linked) return;
      const previous = link;
      setLink({ ...link, auto_share: on });
      setError(null);
      try {
        setLink(await setAutoShare(on));
      } catch (e) {
        setLink(previous);
        setError(humanize(e));
      }
    },
    [link],
  );

  const unlink = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await unlinkTherapist();
      setLink({ linked: false });
    } catch (e) {
      setError(humanize(e));
    } finally {
      setBusy(false);
    }
  }, [busy]);

  return (
    <View style={styles.card} testID="therapist-link-card">
      <Text style={styles.title}>My therapist</Text>
      {link === null ? (
        <Text style={styles.sub} testID="therapist-link-status">
          {loadError ? `Couldn’t load your therapist link (${loadError})` : "Loading…"}
        </Text>
      ) : link.linked ? (
        <>
          <Text style={styles.sub} testID="therapist-link-status">
            Linked to {link.therapist_email}
            {link.status === "accepted"
              ? " · accepted"
              : " · waiting for them to accept"}
          </Text>
          <View style={styles.switchRow}>
            <View style={styles.switchInfo}>
              <Text style={styles.rowTitle}>Share sessions automatically</Text>
              <Text style={styles.sub}>
                Every live session and recording from now on is shared with them
                (read-only). You can un-share any one from its Replay screen.
              </Text>
            </View>
            <Switch
              testID="therapist-auto-share"
              value={Boolean(link.auto_share)}
              onValueChange={toggleAuto}
            />
          </View>
          <TouchableOpacity
            testID="therapist-unlink"
            accessibilityRole="button"
            style={styles.unlinkButton}
            onPress={unlink}
            disabled={busy}
          >
            <Text style={styles.unlinkText}>Unlink therapist</Text>
          </TouchableOpacity>
        </>
      ) : (
        <>
          <Text style={styles.sub} testID="therapist-link-status">
            Enter your therapist’s MindShift account email. They’ll see your
            sessions — transcript, tone over time, and what you could have said —
            in their Therapist dashboard.
          </Text>
          <View style={styles.inputRow}>
            <TextInput
              testID="therapist-email-input"
              style={styles.input}
              placeholder="therapist@example.com"
              placeholderTextColor="#9CA3AF"
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="email-address"
              value={email}
              onChangeText={setEmail}
              onSubmitEditing={submit}
              editable={!busy}
            />
            <TouchableOpacity
              testID="therapist-link-submit"
              accessibilityRole="button"
              style={[styles.linkButton, (!email.trim() || busy) && styles.linkButtonDisabled]}
              onPress={submit}
              disabled={!email.trim() || busy}
            >
              {busy ? (
                <ActivityIndicator size="small" color="#FFFFFF" />
              ) : (
                <Text style={styles.linkButtonText}>Link</Text>
              )}
            </TouchableOpacity>
          </View>
        </>
      )}
      {error ? (
        <Text style={styles.error} testID="therapist-link-error">
          {error}
        </Text>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 14,
    backgroundColor: "#FFFFFF",
    padding: 18,
    marginBottom: 12,
  },
  title: {
    fontSize: 17,
    fontWeight: "600",
    color: "#1F2937",
  },
  rowTitle: {
    fontSize: 15,
    fontWeight: "600",
    color: "#1F2937",
  },
  sub: {
    marginTop: 4,
    fontSize: 13.5,
    lineHeight: 19,
    color: "#6B7280",
  },
  inputRow: {
    flexDirection: "row",
    gap: 8,
    marginTop: 10,
    alignItems: "center",
  },
  input: {
    flex: 1,
    minHeight: 42,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 10,
    paddingHorizontal: 12,
    fontSize: 15,
    color: "#1F2937",
    backgroundColor: "#F9FAFB",
  },
  linkButton: {
    minHeight: 42,
    minWidth: 72,
    paddingHorizontal: 16,
    borderRadius: 10,
    backgroundColor: "#4A90D9",
    alignItems: "center",
    justifyContent: "center",
  },
  linkButtonDisabled: {
    opacity: 0.5,
  },
  linkButtonText: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "700",
  },
  switchRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: "#F0F1F3",
  },
  switchInfo: {
    flex: 1,
    minWidth: 0,
  },
  unlinkButton: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: "#F0F1F3",
  },
  unlinkText: {
    fontSize: 15,
    fontWeight: "700",
    color: "#DC2626",
  },
  error: {
    marginTop: 8,
    fontSize: 13,
    color: "#DC2626",
  },
});
