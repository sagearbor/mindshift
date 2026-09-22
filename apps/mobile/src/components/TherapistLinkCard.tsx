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
  consentGranted,
  disclosureFor,
  getTherapistLink,
  setTherapistLink,
  setAutoShare,
  setTherapistConsent,
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
 *
 * CONSENT (server/consent.py). Linking is not a silent hand-over: the
 * server records an `episodes` consent the moment the email is submitted,
 * stamped with the `text_version` of the sentence that was supposed to be
 * on screen. So the sentence IS on screen, above the field, and it is the
 * server's wording — `consent.scopes.episodes.disclosure`, which rides on
 * `GET /therapist/link` linked or not. The card never keeps its own copy:
 * a client sentence would drift from the version the stored record cites,
 * and a record citing wording nobody saw is not consent. If the server
 * sends no disclosure (an older build), the card says so and REFUSES to
 * link rather than take an undisclosed agreement.
 *
 * `live` — "they may listen to my calls as they happen" — is a separate,
 * explicit switch with its own disclosure, default OFF. Granting it is what
 * lets that therapist into a call without everyone tapping Approve in the
 * moment; revoking it puts the in-call approval back.
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
    // No disclosure on screen, no consent taken. PUT /therapist/link writes
    // a consent record citing a text_version; submitting without having
    // shown the patient that text would make the record a lie.
    if (disclosureFor(link, "episodes") === "") return;
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
  }, [email, busy, link]);

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

  const toggleLive = useCallback(
    async (on: boolean) => {
      if (!link?.linked) return;
      setError(null);
      try {
        setLink(await setTherapistConsent("live", on));
      } catch (e) {
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

  // The server's wording, verbatim. "" = it sent none (a build older than
  // consent): the card must not invent one, and must not link without it.
  const episodesDisclosure = disclosureFor(link, "episodes");
  const liveDisclosure = disclosureFor(link, "live");
  const liveGranted = consentGranted(link, "live");
  const canSubmit = Boolean(email.trim()) && !busy && episodesDisclosure !== "";

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
          {/* What you already agreed to, in the wording that is in force —
              readable after the fact, not only at the moment of the tap. */}
          {episodesDisclosure ? (
            <View style={styles.disclosureBox} testID="therapist-disclosure">
              <Text style={styles.disclosureLabel}>You agreed:</Text>
              <Text style={styles.disclosureText} testID="therapist-disclosure-episodes">
                {episodesDisclosure}
              </Text>
            </View>
          ) : null}
          {/* `live` is a SEPARATE agreement, default off: being listened to
              as it happens is not the same as a transcript read later. With
              it off, a therapist joining a call still needs everyone on that
              call to tap Approve. */}
          {liveDisclosure ? (
            <View style={styles.switchRow}>
              <View style={styles.switchInfo}>
                <Text style={styles.rowTitle}>Let them listen to my calls</Text>
                <Text style={styles.sub} testID="therapist-disclosure-live">
                  {liveDisclosure}
                </Text>
              </View>
              <Switch
                testID="therapist-live-consent"
                value={liveGranted}
                onValueChange={toggleLive}
              />
            </View>
          ) : null}
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
          {/* The agreement itself, in the server's own words, BEFORE the
              tap that records it. */}
          {episodesDisclosure ? (
            <View style={styles.disclosureBox} testID="therapist-disclosure">
              <Text style={styles.disclosureLabel}>
                By linking them you agree:
              </Text>
              <Text style={styles.disclosureText} testID="therapist-disclosure-episodes">
                {episodesDisclosure}
              </Text>
            </View>
          ) : (
            <Text style={styles.error} testID="therapist-disclosure-missing">
              Couldn’t load what you’d be agreeing to, so linking is off for
              now. Pull to refresh, or try again in a moment.
            </Text>
          )}
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
              style={[styles.linkButton, !canSubmit && styles.linkButtonDisabled]}
              onPress={submit}
              disabled={!canSubmit}
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
  // The disclosure itself — set apart from the card's own prose so it reads
  // as the agreement, not as another hint.
  disclosureBox: {
    marginTop: 12,
    padding: 12,
    borderRadius: 10,
    backgroundColor: "#F3F7FC",
    borderWidth: 1,
    borderColor: "#D6E4F5",
  },
  disclosureLabel: {
    fontSize: 12.5,
    fontWeight: "700",
    color: "#1D4ED8",
    textTransform: "uppercase",
    letterSpacing: 0.4,
  },
  disclosureText: {
    marginTop: 4,
    fontSize: 14,
    lineHeight: 20,
    color: "#1F2937",
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
