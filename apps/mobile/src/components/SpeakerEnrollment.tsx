import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
} from "react-native";

import {
  enrollVoice,
  getVoiceProfile,
  type RecordingTurn,
  type VoiceProfile,
} from "../api/client";
import { getSpeakerColor } from "../utils/speakerColors";
import { speakerLabel, type SpeakerLabels } from "../utils/speakerLabels";

const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";
const GOOD = "#0F9D58";

interface SpeakerEnrollmentProps {
  recordingId: string;
  turns: RecordingTurn[];
  /** Per-speaker display labels from the analysis (name → deeper/higher
   *  voice → raw id). Absent → the raw diarization id is shown, as before. */
  speakerLabels?: SpeakerLabels;
}

/**
 * "This is me" — per-speaker voice enrollment on a stored recording.
 *
 * Renders NOTHING until it confirms the server can actually do voice ID
 * (`available` + `storage_enabled` from GET /voice/profile) — no dead button on
 * a server without the optional embedding deps. For each distinct diarized
 * speaker it offers a "This is me" tap that POSTs /voice/enroll and shows an
 * honest confirmation ("Voice saved — you'll be labeled 'You' from now on").
 *
 * Biometric transparency is stated up front and again in the confirmation: what
 * is stored is a numeric voice signature, not the audio.
 */
export default function SpeakerEnrollment({
  recordingId,
  turns,
  speakerLabels,
}: SpeakerEnrollmentProps) {
  const [profile, setProfile] = useState<VoiceProfile | null>(null);
  const [enrollingSpeaker, setEnrollingSpeaker] = useState<string | null>(null);
  const [enrolledSpeaker, setEnrolledSpeaker] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Distinct speakers in first-appearance order (stable, matches the transcript).
  const speakers = useMemo(() => {
    const seen: string[] = [];
    for (const t of turns) {
      if (t.speaker && !seen.includes(t.speaker)) seen.push(t.speaker);
    }
    return seen;
  }, [turns]);

  useEffect(() => {
    let cancelled = false;
    getVoiceProfile()
      .then((p) => {
        if (!cancelled) setProfile(p);
      })
      .catch(() => {
        // A failure here just means we don't offer enrollment — never a crash.
        if (!cancelled) setProfile(null);
      });
    return () => {
      cancelled = true;
    };
  }, [recordingId]);

  const handleEnroll = useCallback(
    async (speaker: string) => {
      setEnrollingSpeaker(speaker);
      setError(null);
      try {
        const res = await enrollVoice(recordingId, speaker);
        setEnrolledSpeaker(speaker);
        setProfile((prev) =>
          prev
            ? { ...prev, enrolled: true, enroll_count: res.enroll_count }
            : prev,
        );
      } catch (e) {
        const err = e as Error & { status?: number };
        if (err.status === 503) {
          setError("Voice enrollment isn’t available on this server yet.");
        } else if (err.status === 422) {
          setError(
            err.message ||
              "There isn’t enough of that speaker’s voice in this recording to enroll.",
          );
        } else {
          setError("Couldn’t save your voice. Please try again.");
        }
      } finally {
        setEnrollingSpeaker(null);
      }
    },
    [recordingId],
  );

  // Only offer enrollment when the server can actually do it.
  if (!profile || !profile.available || !profile.storage_enabled) {
    return null;
  }

  if (enrolledSpeaker) {
    return (
      <View style={styles.card} testID="speaker-enrollment">
        <Text style={styles.confirmTitle}>Voice saved</Text>
        <Text style={styles.confirmBody}>
          You’ll be labeled “You” in your recordings from now on. We stored a
          numeric voice signature — not your audio.
        </Text>
        <Text style={styles.manageHint}>
          You can remove it anytime under Advanced → “Forget my voice”.
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.card} testID="speaker-enrollment">
      <Text style={styles.sectionTitle}>Which voice is you?</Text>
      <Text style={styles.subtitle}>
        Tap yourself once and MindShift will label you “You” in every recording
        from now on. It stores a numeric voice signature, not your audio.
      </Text>
      {profile.enrolled ? (
        <Text style={styles.alreadyNote}>
          You’re already enrolled ({profile.enroll_count}{" "}
          {profile.enroll_count === 1 ? "sample" : "samples"}). Tapping again
          refines your voiceprint.
        </Text>
      ) : null}

      {speakers.map((speaker) => (
        <View key={speaker} style={styles.speakerRow}>
          <View style={styles.speakerLabelWrap}>
            <View
              style={[styles.dot, { backgroundColor: getSpeakerColor(speaker) }]}
            />
            <Text style={styles.speakerLabel}>
              {speakerLabel(speaker, speakerLabels)}
            </Text>
          </View>
          <TouchableOpacity
            testID={`enroll-${speaker}`}
            accessibilityRole="button"
            accessibilityLabel={`Enroll ${speakerLabel(speaker, speakerLabels)} as me`}
            style={styles.enrollButton}
            disabled={enrollingSpeaker !== null}
            onPress={() => void handleEnroll(speaker)}
          >
            {enrollingSpeaker === speaker ? (
              <ActivityIndicator size="small" color={PRIMARY} />
            ) : (
              <Text style={styles.enrollButtonText}>This is me</Text>
            )}
          </TouchableOpacity>
        </View>
      ))}

      {error ? (
        <Text style={styles.error} testID="enroll-error">
          {error}
        </Text>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: INK,
    marginBottom: 6,
  },
  subtitle: {
    fontSize: 13,
    lineHeight: 19,
    color: MUTED,
    marginBottom: 12,
  },
  alreadyNote: {
    fontSize: 12.5,
    lineHeight: 18,
    color: PRIMARY,
    marginBottom: 10,
  },
  speakerRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: "#F3F4F6",
  },
  speakerLabelWrap: {
    flexDirection: "row",
    alignItems: "center",
    flexShrink: 1,
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 10,
  },
  speakerLabel: {
    fontSize: 15,
    fontWeight: "600",
    color: INK,
  },
  enrollButton: {
    minHeight: 40,
    minWidth: 96,
    paddingHorizontal: 16,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: PRIMARY,
    alignItems: "center",
    justifyContent: "center",
  },
  enrollButtonText: {
    fontSize: 14,
    fontWeight: "700",
    color: PRIMARY,
  },
  confirmTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: GOOD,
    marginBottom: 6,
  },
  confirmBody: {
    fontSize: 13.5,
    lineHeight: 20,
    color: INK,
  },
  manageHint: {
    marginTop: 8,
    fontSize: 12.5,
    lineHeight: 18,
    color: MUTED,
    fontStyle: "italic",
  },
  error: {
    marginTop: 10,
    fontSize: 13,
    lineHeight: 18,
    color: "#DC2626",
  },
});
