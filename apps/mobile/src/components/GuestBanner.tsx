import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import { useAuthStore } from "../store/authStore";
import { useGuestUpgradeStore } from "../store/guestUpgradeStore";
import { useGuestLimitsStore } from "../store/guestLimitsStore";
import { googleSignInConfigured } from "../auth/firebaseConfig";
import GoogleSignInButton from "./GoogleSignInButton";

/** The one line every guest sees, everywhere in the app. Exported so the copy
 *  lives in exactly one place and the test asserts the real string. */
export const GUEST_BANNER_TEXT =
  "Guest session — your data is on this device's account only; create an " +
  "account to keep it";

/** What the banner says while the server has not yet told this device the
 *  quota's numbers. It states that the limits EXIST — the thing a guest was
 *  never told until one of them bit — without inventing what they are. */
export const GUEST_LIMITS_UNKNOWN_TEXT =
  "Guest sessions have a daily limit and a time limit per session.";

function plural(n: number, one: string, many: string): string {
  return n === 1 ? one : many;
}

/**
 * The quota line, built from the numbers the SERVER sent (see
 * store/guestLimitsStore.ts: they ride the `guest_limit` frame and are cached
 * on the device). Never a constant — whatever the deploy's env says is what a
 * guest reads, so the app cannot drift into stating a limit that isn't the one
 * being enforced. Each half is independent, so a server that told us only one
 * of the two states that one and stays quiet about the other.
 */
export function guestLimitsLine(
  maxSessionsPerDay: number | null,
  maxSessionMinutes: number | null,
): string {
  const parts: string[] = [];
  if (maxSessionsPerDay !== null) {
    parts.push(
      `${maxSessionsPerDay} live ` +
        `${plural(maxSessionsPerDay, "session", "sessions")} a day`,
    );
  }
  if (maxSessionMinutes !== null) {
    parts.push(
      `${maxSessionMinutes} ${plural(maxSessionMinutes, "minute", "minutes")} each`,
    );
  }
  if (parts.length === 0) return GUEST_LIMITS_UNKNOWN_TEXT;
  return `Guest limit: ${parts.join(", ")}.`;
}

/**
 * A thin, persistent bar shown above every screen while the signed-in user is
 * an anonymous ("Continue as guest") one.
 *
 * Why persistent rather than a one-time toast: the fact it states is true for
 * the whole session and has a real consequence (log out, or lose the phone,
 * and the recordings are gone). A guest who never sees it until the data is
 * already gone was misled by omission. It is deliberately one line tall and
 * muted, not a modal — it must never stand between a reviewer and the Start
 * button.
 *
 * "Create account" expands an inline upgrade form rather than navigating: the
 * upgrade LINKS a credential onto the same anonymous uid (see
 * authStore.linkGuestToEmailPassword), so everything recorded as a guest stays
 * with the account. Sending the user back to the login screen would sign them
 * out of the guest session and strand exactly that data.
 *
 * Renders nothing at all for a signed-up account, so App.tsx can mount it
 * unconditionally.
 */
export default function GuestBanner() {
  const user = useAuthStore((s) => s.user);
  const busy = useAuthStore((s) => s.busy);
  const error = useAuthStore((s) => s.error);
  const notice = useAuthStore((s) => s.notice);
  const clearError = useAuthStore((s) => s.clearError);
  const linkGuestToEmailPassword = useAuthStore(
    (s) => s.linkGuestToEmailPassword,
  );
  const maxSessionsPerDay = useGuestLimitsStore((s) => s.maxSessionsPerDay);
  const maxSessionMinutes = useGuestLimitsStore((s) => s.maxSessionMinutes);

  const [open, setOpen] = useState(false);
  // Opened from elsewhere — today, the confirm a guest sees when they are about
  // to log out and lose everything. One-shot: consume it so a later render
  // cannot re-open a form the user has closed.
  const upgradeRequested = useGuestUpgradeStore((st) => st.requested);
  const consumeUpgradeRequest = useGuestUpgradeStore((st) => st.consume);
  useEffect(() => {
    if (!upgradeRequested) return;
    setOpen(true);
    consumeUpgradeRequest();
  }, [upgradeRequested, consumeUpgradeRequest]);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  if (!user?.isAnonymous) return null;

  const canSubmit = email.trim().length > 0 && password.length > 0 && !busy;

  const submit = () => {
    // Errors surface via the store's `error`; swallow the rejection so an
    // unhandled promise can't crash the app chrome this sits in.
    void linkGuestToEmailPassword(email, password).catch(() => {});
  };

  return (
    <View style={styles.wrap} testID="guest-banner">
      <View style={styles.row}>
        <View style={styles.textColumn}>
          <Text style={styles.text} testID="guest-banner-text">
            {GUEST_BANNER_TEXT}
          </Text>
          {/* The other half of the honesty. Guest mode is CAPPED, and until
              this line a guest found that out only when a cap cut a
              conversation short mid-sentence. One muted line, no counter and
              no nagging — and the numbers in it are the server's, never
              ours (see guestLimitsLine). */}
          <Text style={styles.limits} testID="guest-limits-text">
            {guestLimitsLine(maxSessionsPerDay, maxSessionMinutes)}
          </Text>
        </View>
        <TouchableOpacity
          testID="guest-create-account"
          accessibilityRole="button"
          accessibilityLabel="Create account"
          accessibilityState={{ expanded: open }}
          style={styles.cta}
          onPress={() => {
            clearError();
            setOpen((v) => !v);
          }}
        >
          <Text style={styles.ctaText}>
            {open ? "Not now" : "Create account"}
          </Text>
        </TouchableOpacity>
      </View>

      {open ? (
        <View style={styles.form} testID="guest-upgrade-form">
          <Text style={styles.formNote}>
            Same session, same recordings — we just add a way to sign back in.
          </Text>
          <TextInput
            testID="guest-email-input"
            style={styles.input}
            placeholder="Email"
            placeholderTextColor="#9CA3AF"
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="email-address"
            textContentType="emailAddress"
            value={email}
            onChangeText={(v) => {
              if (error || notice) clearError();
              setEmail(v);
            }}
          />
          <TextInput
            testID="guest-password-input"
            style={styles.input}
            placeholder="Password"
            placeholderTextColor="#9CA3AF"
            autoCapitalize="none"
            autoCorrect={false}
            secureTextEntry
            textContentType="newPassword"
            value={password}
            onChangeText={(v) => {
              if (error || notice) clearError();
              setPassword(v);
            }}
          />
          {error ? (
            <Text testID="guest-error" style={styles.error}>
              {error}
            </Text>
          ) : null}
          {notice ? (
            <Text testID="guest-notice" style={styles.notice}>
              {notice}
            </Text>
          ) : null}
          <TouchableOpacity
            testID="guest-upgrade-submit"
            style={[styles.submit, !canSubmit && styles.submitDisabled]}
            onPress={submit}
            disabled={!canSubmit}
          >
            {busy ? (
              <ActivityIndicator color="#FFFFFF" />
            ) : (
              <Text style={styles.submitText}>Create account</Text>
            )}
          </TouchableOpacity>
          {/* The same Google button the login screen uses. The store notices
              the current user is anonymous and LINKS Google onto that uid
              instead of signing into a separate account — see
              authStore.signInWithGoogleIdToken. */}
          {googleSignInConfigured ? <GoogleSignInButton /> : null}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    backgroundColor: "#FEF3C7",
    borderBottomWidth: 1,
    borderBottomColor: "#FDE68A",
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  textColumn: {
    flex: 1,
    gap: 2,
  },
  text: {
    color: "#78350F",
    fontSize: 12,
    lineHeight: 16,
  },
  limits: {
    color: "#92400E",
    fontSize: 11,
    lineHeight: 15,
  },
  cta: {
    paddingVertical: 4,
    paddingHorizontal: 10,
    borderRadius: 8,
    backgroundColor: "#92400E",
  },
  ctaText: {
    color: "#FFFFFF",
    fontSize: 12,
    fontWeight: "600",
  },
  form: {
    marginTop: 10,
    gap: 8,
  },
  formNote: {
    color: "#78350F",
    fontSize: 12,
    lineHeight: 16,
  },
  input: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 14,
    color: "#1F2937",
    backgroundColor: "#FFFFFF",
  },
  error: {
    color: "#B91C1C",
    fontSize: 12,
    lineHeight: 16,
  },
  notice: {
    color: "#047857",
    fontSize: 12,
    lineHeight: 16,
  },
  submit: {
    backgroundColor: "#92400E",
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: "center",
  },
  submitDisabled: {
    opacity: 0.5,
  },
  submitText: {
    color: "#FFFFFF",
    fontSize: 14,
    fontWeight: "600",
  },
});
