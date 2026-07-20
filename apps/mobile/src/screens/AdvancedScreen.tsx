import React, { useCallback, useEffect, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from "react-native";

import Constants from "expo-constants";
import * as Application from "expo-application";

import { forgetVoice, getVoiceProfile } from "../api/client";
import { useAuthStore } from "../store/authStore";
import { useOtaStatus, type OtaStatus } from "../utils/otaUpdate";
import { formatDateTime } from "../utils/dateDisplay";

/** Bare host (no scheme/path) of the configured backend, for the About row. */
function backendHost(): string {
  const raw = process.env.EXPO_PUBLIC_API_URL || "";
  if (!raw) return "localhost:8000 (default)";
  return raw.replace(/^https?:\/\//, "").replace(/\/.*$/, "");
}

/**
 * One honest sentence describing the running JS bundle's OTA state. We never
 * imply an update channel exists when it doesn't: a store build without the
 * expo-updates module (or web) reads "Store build (no OTA yet)".
 */
function otaSummary(ota: OtaStatus): string {
  if (!ota.supported) return "Store build (no OTA yet)";
  if (ota.isEmbeddedLaunch) {
    return ota.channel
      ? `Store build · ${ota.channel} channel (no OTA applied yet)`
      : "Store build (no OTA applied yet)";
  }
  const when = ota.createdAt ? formatDateTime(ota.createdAt.toISOString()) : null;
  const parts: string[] = [];
  if (when) parts.push(`Updated ${when}`);
  if (ota.channel) parts.push(`${ota.channel} channel`);
  const base = parts.length > 0 ? parts.join(" · ") : "OTA update applied";
  return ota.errored ? `${base} · last check failed` : base;
}

/**
 * Everything that doesn't fit the two home modes lives behind the small
 * Advanced affordance: the therapist dashboard and account actions. Nothing
 * here was deleted from the app — only moved out of the way.
 */
interface AdvancedScreenProps {
  onBack: () => void;
  onOpenDashboard: () => void;
  onSignOut: () => void;
}

/** A single copy-friendly label/value row in the About card. The value is
 *  `selectable` so testers can long-press to copy (versions, email, backend). */
function AboutRow({
  testID,
  label,
  value,
  last,
}: {
  testID: string;
  label: string;
  value: string;
  last?: boolean;
}) {
  return (
    <View
      testID={testID}
      style={[styles.aboutRow, last ? styles.aboutRowLast : null]}
    >
      <Text style={styles.aboutLabel}>{label}</Text>
      <Text style={styles.aboutValue} selectable>
        {value}
      </Text>
    </View>
  );
}

export default function AdvancedScreen({
  onBack,
  onOpenDashboard,
  onSignOut,
}: AdvancedScreenProps) {
  // "Forget my voice" — only shown once we confirm a voiceprint exists to delete.
  const [enrolled, setEnrolled] = useState(false);
  const [forgetting, setForgetting] = useState(false);

  // --- About section facts (all honest; a missing value reads "unknown"). ---
  const user = useAuthStore((s) => s.user);
  const ota = useOtaStatus();
  const appVersion =
    Application.nativeApplicationVersion ??
    Constants.expoConfig?.version ??
    "unknown";
  const buildVersion =
    Application.nativeBuildVersion ??
    (Constants.expoConfig?.android?.versionCode != null
      ? String(Constants.expoConfig.android.versionCode)
      : "unknown");
  const accountEmail = user?.email ?? "No email on this account";

  useEffect(() => {
    let cancelled = false;
    getVoiceProfile()
      .then((p) => {
        if (!cancelled) setEnrolled(p.available && p.enrolled);
      })
      .catch(() => {
        if (!cancelled) setEnrolled(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const confirmForget = useCallback(() => {
    Alert.alert(
      "Forget my voice?",
      "This permanently deletes the numeric voice signature MindShift uses to " +
        'label you “You”. Your recordings are not affected. You can re-enroll ' +
        "anytime from a recording.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Forget",
          style: "destructive",
          onPress: () => {
            setForgetting(true);
            forgetVoice()
              .then(() => {
                setEnrolled(false);
                Alert.alert("Voice forgotten", "Your voice signature was deleted.");
              })
              .catch(() => {
                Alert.alert(
                  "Couldn’t forget your voice",
                  "Something went wrong. Please try again.",
                );
              })
              .finally(() => setForgetting(false));
          },
        },
      ],
    );
  }, []);

  return (
    <ScrollView
      style={styles.flex}
      contentContainerStyle={styles.content}
      testID="advanced-screen"
    >
      <TouchableOpacity
        testID="advanced-back"
        accessibilityRole="button"
        style={styles.backButton}
        onPress={onBack}
      >
        <Text style={styles.backText}>← Back</Text>
      </TouchableOpacity>

      <Text style={styles.heading}>Advanced</Text>

      <TouchableOpacity
        testID="advanced-dashboard"
        accessibilityRole="button"
        style={styles.row}
        onPress={onOpenDashboard}
      >
        <Text style={styles.rowTitle}>Therapist Dashboard</Text>
        <Text style={styles.rowSub}>
          Saved coaching sessions grouped by role, with tone trends and export.
        </Text>
      </TouchableOpacity>

      {enrolled ? (
        <TouchableOpacity
          testID="advanced-forget-voice"
          accessibilityRole="button"
          style={styles.row}
          onPress={confirmForget}
          disabled={forgetting}
        >
          <View style={styles.forgetTitleRow}>
            <Text style={styles.rowTitle}>Forget my voice</Text>
            {forgetting ? (
              <ActivityIndicator size="small" color="#6B7280" />
            ) : null}
          </View>
          <Text style={styles.rowSub}>
            Delete the numeric voice signature used to label you “You”. Your
            recordings are kept; only the voiceprint is removed.
          </Text>
        </TouchableOpacity>
      ) : null}

      <Text style={styles.sectionHeading}>About</Text>
      <View style={styles.aboutCard} testID="about-section">
        <AboutRow testID="about-version" label="App version" value={appVersion} />
        <AboutRow testID="about-build" label="Build" value={buildVersion} />
        <AboutRow
          testID="about-update"
          label="Update"
          value={otaSummary(ota)}
        />
        <AboutRow
          testID="about-account"
          label="Signed in as"
          value={accountEmail}
        />
        <AboutRow
          testID="about-backend"
          label="Backend"
          value={backendHost()}
          last
        />
      </View>

      <TouchableOpacity
        testID="advanced-sign-out"
        accessibilityRole="button"
        style={[styles.row, styles.signOutRow]}
        onPress={onSignOut}
      >
        <Text style={[styles.rowTitle, styles.signOutText]}>Sign out</Text>
      </TouchableOpacity>
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
    minHeight: 52,
    justifyContent: "center",
  },
  rowTitle: {
    fontSize: 17,
    fontWeight: "600",
    color: "#1F2937",
  },
  rowSub: {
    marginTop: 4,
    fontSize: 13.5,
    lineHeight: 19,
    color: "#6B7280",
  },
  forgetTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  sectionHeading: {
    fontSize: 13,
    fontWeight: "700",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    color: "#9CA3AF",
    marginTop: 12,
    marginBottom: 8,
  },
  aboutCard: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 14,
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 18,
  },
  aboutRow: {
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#F0F1F3",
  },
  aboutRowLast: {
    borderBottomWidth: 0,
  },
  aboutLabel: {
    fontSize: 12.5,
    fontWeight: "600",
    color: "#6B7280",
  },
  aboutValue: {
    marginTop: 3,
    fontSize: 15,
    color: "#1F2937",
  },
  signOutRow: {
    marginTop: 16,
  },
  signOutText: {
    color: "#DC2626",
  },
});
