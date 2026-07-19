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

import { forgetVoice, getVoiceProfile } from "../api/client";

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

export default function AdvancedScreen({
  onBack,
  onOpenDashboard,
  onSignOut,
}: AdvancedScreenProps) {
  // "Forget my voice" — only shown once we confirm a voiceprint exists to delete.
  const [enrolled, setEnrolled] = useState(false);
  const [forgetting, setForgetting] = useState(false);

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
  signOutRow: {
    marginTop: 16,
  },
  signOutText: {
    color: "#DC2626",
  },
});
