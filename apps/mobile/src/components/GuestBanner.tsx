import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import { useAuthStore } from "../store/authStore";
import { googleSignInConfigured } from "../auth/firebaseConfig";
import GoogleSignInButton from "./GoogleSignInButton";

/** The one line every guest sees, everywhere in the app. Exported so the copy
 *  lives in exactly one place and the test asserts the real string. */
export const GUEST_BANNER_TEXT =
  "Guest session — your data is on this device's account only; create an " +
  "account to keep it";

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

  const [open, setOpen] = useState(false);
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
        <Text style={styles.text} testID="guest-banner-text">
          {GUEST_BANNER_TEXT}
        </Text>
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
  text: {
    flex: 1,
    color: "#78350F",
    fontSize: 12,
    lineHeight: 16,
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
