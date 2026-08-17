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

import {
  deleteVoiceSample,
  forgetVoice,
  getVoiceProfile,
  listRecordings,
  type VoiceProfile,
  type VoiceSample,
} from "../api/client";
import { getMe } from "../api/me";
import VoiceTrainingFlow from "../components/VoiceTrainingFlow";
import Avatar from "../components/Avatar";
import { useAuthStore } from "../store/authStore";
import { useAvatarStore } from "../store/avatarStore";
import { useOtaStatus, type OtaStatus } from "../utils/otaUpdate";
import { formatDate, formatDateTime } from "../utils/dateDisplay";

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
 * "⋯" affordance on Home: this is the app's Settings destination — the
 * therapist dashboard, watch setup, voice profile management, and account
 * actions, grouped into labeled sections. Nothing here was deleted from the
 * app — only moved out of the way and organized (Phase 0 of the nav
 * redesign; the hamburger + customizable home arrive in a follow-up).
 */
interface AdvancedScreenProps {
  onBack: () => void;
  onOpenDashboard: () => void;
  onSignOut: () => void;
  /** Open a recording's replay — the voice card's per-sample "Play" jumps to
   *  the recording a sample was enrolled from. */
  onOpenReplay: (recordingId: string) => void;
  /** Open the "Set up your watch" screen (Phase 3 Slice 1): install the
   *  watch app via Play + redeem the pairing code it shows. */
  onOpenWatchSetup: () => void;
  /** Re-run the first-launch onboarding walkthrough (Task P3-7). Doesn't
   *  touch the "seen" persistence — this is an explicit replay, not a reset
   *  of whether it auto-shows again on next launch. */
  onOpenTutorial: () => void;
  /** Open the "Home screen design" editor (Task N5 of P3-10) — arrange the
   *  configurable bottom bar and home boxes. */
  onOpenHomeDesign: () => void;
  /** Open the selfie-capture flow (Task N6 of P3-10) — the same flow the
   *  avatar menu's "Set profile photo" row opens. "Remove photo" doesn't
   *  need navigation (see the Account section below) — it clears
   *  avatarStore directly, same as the voice card's "Forget" action calls
   *  the API directly. */
  onSetProfilePhoto: () => void;
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
  onOpenReplay,
  onOpenWatchSetup,
  onOpenTutorial,
  onOpenHomeDesign,
  onSetProfilePhoto,
}: AdvancedScreenProps) {
  // Voice profile card — full detail (per-sample provenance) once loaded.
  const [profile, setProfile] = useState<VoiceProfile | null>(null);
  // recording_id → display title, so a sample can say WHICH recording taught
  // it. null = the list couldn't be fetched (we then say nothing about whether
  // a source recording still exists, rather than guessing "deleted").
  const [recordingTitles, setRecordingTitles] = useState<Record<
    string,
    string
  > | null>(null);
  const [forgetting, setForgetting] = useState(false);
  const [sampleError, setSampleError] = useState<string | null>(null);
  // Guided "Train my voice" flow, opened in place inside the voice card.
  const [training, setTraining] = useState(false);
  // Server-derived paired-watch state (Task P3-6): null covers BOTH "still
  // loading" and "couldn't be determined" (offline, 401, 5xx) — the row's
  // honest default is to say nothing about pairing state, same as before
  // this fetch existed, never a fabricated paired/unpaired guess.
  const [hasPairedWatch, setHasPairedWatch] = useState<boolean | null>(null);

  // --- About section facts (all honest; a missing value reads "unknown"). ---
  const user = useAuthStore((s) => s.user);
  // Task N6: the selfie avatar shown in the top bar — this row previews and
  // manages the same store.
  const avatarUri = useAvatarStore((s) => s.uri);
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
        if (!cancelled) setProfile(p);
      })
      .catch(() => {
        // No card is better than a wrong card — the server may simply not
        // support voice ID, or the user may be offline.
        if (!cancelled) setProfile(null);
      });
    listRecordings()
      .then((rs) => {
        if (!cancelled) {
          setRecordingTitles(
            Object.fromEntries(rs.map((r) => [r.id, r.title || r.filename])),
          );
        }
      })
      .catch(() => {
        if (!cancelled) setRecordingTitles(null);
      });
    getMe()
      .then((me) => {
        if (!cancelled) setHasPairedWatch(me.has_paired_watch);
      })
      .catch(() => {
        // Offline / signed-out / server hiccup: stay at the honest "unknown"
        // default (null) — the row just renders as it always has.
        if (!cancelled) setHasPairedWatch(null);
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
                setProfile((p) =>
                  p ? { ...p, enrolled: false, enroll_count: 0, samples: [] } : p,
                );
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

  /** Delete one enrollment sample — optimistic removal, rolled back with an
   *  honest message when the server says no. */
  const handleDeleteSample = useCallback(
    (sample: VoiceSample) => {
      const previous = profile;
      if (!previous) return;
      setSampleError(null);
      const remaining = (previous.samples ?? []).filter(
        (s) => s.id !== sample.id,
      );
      setProfile({
        ...previous,
        samples: remaining,
        enroll_count: remaining.length,
        enrolled: remaining.length > 0,
      });
      deleteVoiceSample(sample.id)
        .then((res) => {
          // Trust the server's view of what remains (it recomputed the blend).
          setProfile((p) =>
            p
              ? {
                  ...p,
                  enrolled: res.enrolled,
                  enroll_count: res.enroll_count,
                }
              : p,
          );
        })
        .catch(() => {
          setProfile(previous); // roll the optimistic removal back
          setSampleError("Couldn’t delete that sample. Please try again.");
        });
    },
    [profile],
  );

  /** Guided training finished: the server now holds the new sample — refetch
   *  so the card shows the SERVER's view (count + the guided sample's note)
   *  rather than a client-side guess. */
  const handleTrained = useCallback(() => {
    setTraining(false);
    getVoiceProfile()
      .then((p) => setProfile(p))
      .catch(() => {
        // Keep the last known profile; the next visit refetches.
      });
  }, []);

  /** Where a sample came from, honestly: the recording's title when it still
   *  exists, "source recording deleted" when it's provably gone, the legacy
   *  note for the migrated pre-v2 blend, and nothing speculative when the
   *  recordings list couldn't be checked. */
  const sampleSource = (s: VoiceSample): string => {
    if (!s.recording_id) return s.note || "earlier enrollments";
    if (recordingTitles === null) return "from a recording";
    const title = recordingTitles[s.recording_id];
    return title ? `from ${title}` : "source recording deleted";
  };

  const samples = profile?.samples ?? [];

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

      <Text style={styles.heading} testID="settings-heading">
        Settings
      </Text>

      <Text style={styles.sectionHeading} testID="section-your-tools">
        Your tools
      </Text>

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

      <TouchableOpacity
        testID="advanced-watch-setup"
        accessibilityRole="button"
        style={styles.row}
        onPress={onOpenWatchSetup}
      >
        <Text style={styles.rowTitle}>Set up your watch</Text>
        <Text style={styles.rowSub}>
          Install the MindShift watch app and pair it to this account.
        </Text>
        {hasPairedWatch ? (
          <Text style={styles.watchPairedStatus} testID="watch-setup-paired-status">
            ✓ Paired to this account
          </Text>
        ) : null}
      </TouchableOpacity>

      <TouchableOpacity
        testID="advanced-show-tutorial"
        accessibilityRole="button"
        style={styles.row}
        onPress={onOpenTutorial}
      >
        <Text style={styles.rowTitle}>Show tutorial</Text>
        <Text style={styles.rowSub}>
          Replay the short walkthrough of Live Coach, Analyze, your watch,
          and Growth.
        </Text>
      </TouchableOpacity>

      <Text style={styles.sectionHeading} testID="section-appearance">
        Appearance
      </Text>

      <TouchableOpacity
        testID="advanced-home-design"
        accessibilityRole="button"
        style={styles.row}
        onPress={onOpenHomeDesign}
      >
        <Text style={styles.rowTitle}>Home screen design</Text>
        <Text style={styles.rowSub}>
          Arrange your bottom bar and home screen shortcuts.
        </Text>
      </TouchableOpacity>

      {profile && profile.available && profile.storage_enabled ? (
        <>
          <Text style={styles.sectionHeading} testID="section-voice">
            Voice
          </Text>
          <View style={styles.row} testID="voice-profile-card">
            <Text style={styles.rowTitle}>Voice profile</Text>
            {profile.enrolled ? (
              <>
                <Text style={styles.rowSub} testID="voice-profile-status">
                  {[
                    `Enrolled · ${profile.enroll_count} sample` +
                      `${profile.enroll_count === 1 ? "" : "s"}`,
                    formatDate(profile.updated_at)
                      ? `updated ${formatDate(profile.updated_at)}`
                      : null,
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </Text>

                {samples.map((s) => (
                  <View key={s.id} style={styles.sampleRow} testID={`voice-sample-${s.id}`}>
                    <View style={styles.sampleInfo}>
                      <Text style={styles.sampleTitle} numberOfLines={1}>
                        {sampleSource(s)}
                      </Text>
                      <Text style={styles.sampleMeta}>
                        {[s.speaker, s.at ? formatDate(s.at) : null]
                          .filter(Boolean)
                          .join(" · ") || "no details recorded"}
                      </Text>
                    </View>
                    {s.recording_id && recordingTitles?.[s.recording_id] ? (
                      <TouchableOpacity
                        testID={`voice-sample-play-${s.id}`}
                        accessibilityRole="button"
                        accessibilityLabel="Play the recording this sample came from"
                        style={styles.sampleButton}
                        onPress={() => onOpenReplay(s.recording_id as string)}
                      >
                        <Text style={styles.sampleButtonText}>Play</Text>
                      </TouchableOpacity>
                    ) : null}
                    <TouchableOpacity
                      testID={`voice-sample-delete-${s.id}`}
                      accessibilityRole="button"
                      accessibilityLabel="Delete this voice sample"
                      style={styles.sampleButton}
                      onPress={() => handleDeleteSample(s)}
                    >
                      <Text style={styles.sampleDeleteText}>Delete</Text>
                    </TouchableOpacity>
                  </View>
                ))}

                {sampleError ? (
                  <Text style={styles.sampleError} testID="voice-sample-error">
                    {sampleError}
                  </Text>
                ) : null}

                <Text style={styles.addHint} testID="voice-add-sample-hint">
                  Add another sample: open any recording and tap “This is me” on
                  your speaker. More samples make “You” more reliable.
                </Text>

                <TouchableOpacity
                  testID="advanced-forget-voice"
                  accessibilityRole="button"
                  style={styles.forgetButton}
                  onPress={confirmForget}
                  disabled={forgetting}
                >
                  <View style={styles.forgetTitleRow}>
                    <Text style={styles.forgetText}>Forget my voice</Text>
                    {forgetting ? (
                      <ActivityIndicator size="small" color="#6B7280" />
                    ) : null}
                  </View>
                  <Text style={styles.rowSub}>
                    Delete the numeric voice signature used to label you “You”.
                    Your recordings are kept; only the voiceprint is removed.
                  </Text>
                </TouchableOpacity>
              </>
            ) : (
              <Text style={styles.rowSub} testID="voice-profile-status">
                Not enrolled. Train your voice right here with four short
                phrases, or open a recording and tap “This is me” on your
                speaker — MindShift will label you “You” from then on. It stores
                a numeric voice signature, never your audio.
              </Text>
            )}

            {training ? (
              <VoiceTrainingFlow
                onDone={handleTrained}
                onCancel={() => setTraining(false)}
              />
            ) : (
              <TouchableOpacity
                testID="voice-train-button"
                accessibilityRole="button"
                style={styles.trainButton}
                onPress={() => setTraining(true)}
              >
                <Text style={styles.trainText}>
                  {profile.enrolled ? "Add more voice training" : "Train my voice"}
                </Text>
                <Text style={styles.rowSub}>
                  Read four short phrases aloud — no recordings needed first.
                  Works alongside “This is me”; both add samples.
                </Text>
              </TouchableOpacity>
            )}
          </View>
        </>
      ) : null}

      <Text style={styles.sectionHeading} testID="section-about">
        About
      </Text>
      <View style={styles.aboutCard} testID="about-section">
        <Text style={styles.aboutAppName} testID="about-app-name">
          MindShift
        </Text>
        <AboutRow testID="about-version" label="App version" value={appVersion} />
        <AboutRow testID="about-build" label="Build" value={buildVersion} />
        <AboutRow
          testID="about-update"
          label="Update"
          value={otaSummary(ota)}
        />
        {ota.updateId ? (
          <AboutRow
            testID="about-update-id"
            label="Update ID"
            value={ota.updateId}
          />
        ) : null}
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

      <Text style={styles.sectionHeading} testID="section-account">
        Account
      </Text>

      <View style={styles.row} testID="advanced-profile-photo-card">
        <View style={styles.profilePhotoRow}>
          <Avatar
            user={user}
            photoUri={avatarUri}
            size={44}
            testID="advanced-avatar-preview"
          />
          <View style={styles.profilePhotoInfo}>
            <Text style={styles.rowTitle}>Profile photo</Text>
            <Text style={styles.rowSub}>
              {avatarUri
                ? "Shown in the top bar so you know you're signed in as you."
                : "Add a selfie so the top bar shows it's you, not just an initial."}
            </Text>
          </View>
        </View>
        <TouchableOpacity
          testID="advanced-set-profile-photo"
          accessibilityRole="button"
          style={styles.trainButton}
          onPress={onSetProfilePhoto}
        >
          <Text style={styles.trainText}>
            {avatarUri ? "Retake photo" : "Set profile photo"}
          </Text>
        </TouchableOpacity>
        {avatarUri ? (
          <TouchableOpacity
            testID="advanced-remove-profile-photo"
            accessibilityRole="button"
            style={styles.forgetButton}
            onPress={() => useAvatarStore.getState().removePhoto()}
          >
            <Text style={styles.forgetText}>Remove photo</Text>
          </TouchableOpacity>
        ) : null}
      </View>

      <TouchableOpacity
        testID="advanced-sign-out"
        accessibilityRole="button"
        style={[styles.row, styles.signOutRow]}
        onPress={onSignOut}
      >
        <Text style={[styles.rowTitle, styles.signOutText]}>Log out</Text>
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
  watchPairedStatus: {
    marginTop: 6,
    fontSize: 13.5,
    fontWeight: "600",
    color: "#15803D",
  },
  forgetTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  sampleRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: "#F0F1F3",
    marginTop: 10,
  },
  sampleInfo: {
    flex: 1,
    minWidth: 0,
  },
  sampleTitle: {
    fontSize: 14,
    fontWeight: "600",
    color: "#1F2937",
  },
  sampleMeta: {
    marginTop: 2,
    fontSize: 12.5,
    color: "#6B7280",
  },
  sampleButton: {
    minHeight: 36,
    minWidth: 56,
    paddingHorizontal: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    alignItems: "center",
    justifyContent: "center",
  },
  sampleButtonText: {
    fontSize: 13,
    fontWeight: "700",
    color: "#4A90D9",
  },
  sampleDeleteText: {
    fontSize: 13,
    fontWeight: "700",
    color: "#DC2626",
  },
  sampleError: {
    marginTop: 8,
    fontSize: 13,
    color: "#DC2626",
  },
  addHint: {
    marginTop: 12,
    fontSize: 12.5,
    lineHeight: 18,
    color: "#6B7280",
    fontStyle: "italic",
  },
  forgetButton: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: "#F0F1F3",
  },
  trainButton: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: "#F0F1F3",
  },
  trainText: {
    fontSize: 15,
    fontWeight: "700",
    color: "#4A90D9",
  },
  forgetText: {
    fontSize: 15,
    fontWeight: "700",
    color: "#DC2626",
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
  aboutAppName: {
    fontSize: 17,
    fontWeight: "700",
    color: "#1F2937",
    paddingTop: 14,
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
  profilePhotoRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  profilePhotoInfo: {
    flex: 1,
    minWidth: 0,
  },
});
