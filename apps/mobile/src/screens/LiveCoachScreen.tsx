import React, { useState, useCallback, useEffect } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Switch,
  StyleSheet,
} from "react-native";
import EmpathySlider from "../components/EmpathySlider";
import InterjectSlider from "../components/InterjectSlider";
import SuggestionCard from "../components/SuggestionCard";
import LiveTranscript from "../components/LiveTranscript";
import { useAudioStream } from "../hooks/useAudioStream";

const STATUS_COLORS: Record<string, string> = {
  idle: "#9CA3AF",
  connecting: "#F59E0B",
  live: "#10B981",
  disconnected: "#EF4444",
};

interface LiveCoachScreenProps {
  /** Hand the finished live transcript off to the async-review Session screen.
   *  Optional so the screen still renders standalone (e.g. in isolation tests);
   *  the review button only calls it when present. */
  onReviewTranscript?: (turns: { speaker: string; text: string }[]) => void;
}

export default function LiveCoachScreen({
  onReviewTranscript,
}: LiveCoachScreenProps = {}) {
  const {
    isRecording,
    sessionActive,
    transcript,
    suggestions,
    selfSpeaker,
    setSelfSpeaker,
    connectionStatus,
    transcriptionMessage,
    micError,
    speechAvailable,
    setSpeechEnabled,
    startSession,
    stopSession,
    sendEmpathyUpdate,
    sendInterjectUpdate,
  } = useAudioStream();

  const [empathyLevel, setEmpathyLevel] = useState(50);
  const [interjectLevel, setInterjectLevel] = useState(0);
  const [coachMode, setCoachMode] = useState<"earpiece" | "visual">(
    "visual",
  );

  // Earpiece mode speaks the top suggestion aloud (free on-device TTS);
  // visual mode stays silent. The hook stops any in-flight utterance when
  // this flips to false.
  useEffect(() => {
    setSpeechEnabled(coachMode === "earpiece");
  }, [coachMode, setSpeechEnabled]);

  const handleToggle = useCallback(async () => {
    // Toggle on sessionActive, not isRecording: a session can be live with
    // mic capture unavailable (e.g. web) and must still be stoppable.
    if (sessionActive) {
      await stopSession();
    } else {
      const sessionId = `live-${Date.now()}`;
      await startSession(sessionId, empathyLevel, interjectLevel);
    }
  }, [sessionActive, stopSession, startSession, empathyLevel, interjectLevel]);

  // Flip the coached user's identity between the two diarized speakers. The
  // server labels the first voice it hears "Speaker A", so that's the default.
  const handleToggleSelfSpeaker = useCallback(() => {
    setSelfSpeaker(selfSpeaker === "Speaker B" ? "Speaker A" : "Speaker B");
  }, [selfSpeaker, setSelfSpeaker]);

  const handleReview = useCallback(() => {
    onReviewTranscript?.(
      transcript.map((t) => ({ speaker: t.speaker, text: t.text })),
    );
  }, [onReviewTranscript, transcript]);

  const handleEmpathyChange = useCallback(
    (value: number) => {
      setEmpathyLevel(value);
      sendEmpathyUpdate(value);
    },
    [sendEmpathyUpdate],
  );

  const handleInterjectChange = useCallback(
    (value: number) => {
      // Round at the source: the server takes an int, and this state also
      // feeds startSession's initial config — keep both paths in sync.
      const rounded = Math.round(value);
      setInterjectLevel(rounded);
      sendInterjectUpdate(rounded);
    },
    [sendInterjectUpdate],
  );

  const statusColor = STATUS_COLORS[connectionStatus] || STATUS_COLORS.idle;

  return (
    <View style={styles.container}>
      {/* Header with connection status. The heading takes the flexible space
          and the status pins to the right at a fixed width, so a long status
          word ("disconnected") can never overlap the title (a real Pixel bug). */}
      <View style={styles.header}>
        <Text style={styles.heading} numberOfLines={1}>
          Live Coach
        </Text>
        <View style={styles.statusRow}>
          <View
            style={[styles.statusDot, { backgroundColor: statusColor }]}
            testID="connection-status"
          />
          <Text
            style={[styles.statusText, { color: statusColor }]}
            numberOfLines={1}
          >
            {connectionStatus}
          </Text>
        </View>
      </View>

      {/* Identity chip: which diarized voice is the user's. Shown once there's
          a session or a first transcript line — before that the toggle would
          be meaningless. Tapping flips A↔B; the hint reminds the "you speak
          first" convention while idle. */}
      {(sessionActive || transcript.length > 0) && (
        <View style={styles.identityRow}>
          <TouchableOpacity
            testID="self-speaker-chip"
            style={styles.identityChip}
            onPress={handleToggleSelfSpeaker}
          >
            <Text style={styles.identityChipText}>
              You: {selfSpeaker ?? "Speaker A"} ⇄
            </Text>
          </TouchableOpacity>
          {connectionStatus === "idle" && (
            <Text style={styles.identityHint}>you speak first</Text>
          )}
        </View>
      )}

      {/* Microphone error banner — honest failure state, never a fake session */}
      {micError ? (
        <View style={styles.errorBanner} testID="mic-error-banner">
          <Text style={styles.errorBannerText}>{micError}</Text>
        </View>
      ) : null}

      {/* Transcription availability banner */}
      {transcriptionMessage ? (
        <View style={styles.banner} testID="transcription-banner">
          <Text style={styles.bannerText}>
            Transcription unavailable: {transcriptionMessage}
          </Text>
        </View>
      ) : null}

      {/* Coach mode toggle */}
      <View style={styles.modeRow}>
        <Text style={styles.modeLabel}>Coach mode:</Text>
        <TouchableOpacity
          testID="mode-earpiece"
          style={[
            styles.modeButton,
            coachMode === "earpiece" && styles.modeButtonActive,
          ]}
          onPress={() => setCoachMode("earpiece")}
        >
          <Text
            style={[
              styles.modeButtonText,
              coachMode === "earpiece" && styles.modeButtonTextActive,
            ]}
          >
            Earpiece
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          testID="mode-visual"
          style={[
            styles.modeButton,
            coachMode === "visual" && styles.modeButtonActive,
          ]}
          onPress={() => setCoachMode("visual")}
        >
          <Text
            style={[
              styles.modeButtonText,
              coachMode === "visual" && styles.modeButtonTextActive,
            ]}
          >
            Visual
          </Text>
        </TouchableOpacity>
      </View>

      {/* Honest state: earpiece selected but this platform has no TTS —
          suggestions stay visual-only instead of silently pretending. */}
      {coachMode === "earpiece" && !speechAvailable ? (
        <Text style={styles.speechUnavailableText} testID="speech-unavailable-note">
          Spoken suggestions aren&apos;t available on this platform — showing
          them on screen only.
        </Text>
      ) : null}

      {/* Empathy slider */}
      <EmpathySlider
        value={empathyLevel}
        onValueChange={handleEmpathyChange}
      />

      {/* How often the coach should interject */}
      <InterjectSlider
        value={interjectLevel}
        onValueChange={handleInterjectChange}
      />

      {/* Idle explainer: before the first session, spell out how the flow
          works. Disappears the moment a session starts or any transcript
          arrives — no persistence needed. */}
      {connectionStatus === "idle" &&
      transcript.length === 0 &&
      !sessionActive ? (
        <View style={styles.explainerCard} testID="idle-explainer">
          <Text style={styles.explainerLine}>
            Place the phone between you.
          </Text>
          <Text style={styles.explainerLine}>
            Tap Start, then speak first — the coach learns which voice is yours.
          </Text>
          <Text style={styles.explainerLine}>
            Earpiece = private coaching in your ear; Visual = on-screen only.
          </Text>
        </View>
      ) : null}

      {/* Live transcript */}
      <LiveTranscript entries={transcript} />

      {/* Suggestion feed: newest first, older entries faded so the eye lands
          on the latest. Nudges (about the user's OWN turn) render as a compact
          banner; responses render the usual SuggestionCard stack. */}
      {suggestions.length > 0 && (
        <ScrollView
          style={styles.suggestionsContainer}
          horizontal={false}
          testID="suggestions-list"
        >
          <Text style={styles.suggestionsTitle}>Suggestions</Text>
          {suggestions.map((entry, i) => {
            // Newest at full strength; older entries fade uniformly. Muted
            // entries keep their own extra dimming (in SuggestionCard / the
            // banner style) on top of this.
            const ageStyle =
              i === 0 ? styles.feedEntryNewest : styles.feedEntryOlder;
            if (entry.kind === "nudge") {
              return (
                <View
                  key={entry.id}
                  testID="nudge-banner"
                  style={[
                    styles.nudgeBanner,
                    ageStyle,
                    entry.muted && styles.nudgeBannerMuted,
                  ]}
                >
                  <View style={styles.nudgeBadge}>
                    <Text style={styles.nudgeBadgeText}>NUDGE</Text>
                  </View>
                  <Text style={styles.nudgeText} numberOfLines={1}>
                    {entry.texts[0]}
                  </Text>
                </View>
              );
            }
            return (
              <View key={entry.id} style={ageStyle}>
                {entry.texts.map((text, j) => (
                  <SuggestionCard
                    key={j}
                    text={text}
                    tone={entry.tone}
                    muted={entry.muted}
                  />
                ))}
              </View>
            );
          })}
        </ScrollView>
      )}

      {/* Post-session review handoff: after a session ends with something to
          review, offer a prominent jump to the async-review Session screen. */}
      {!sessionActive && transcript.length > 0 && (
        <TouchableOpacity
          testID="review-transcript-button"
          style={styles.reviewButton}
          onPress={handleReview}
        >
          <Text style={styles.reviewButtonText}>
            Review this conversation →
          </Text>
        </TouchableOpacity>
      )}

      {/* Start/Stop button */}
      <TouchableOpacity
        testID="mic-toggle"
        style={[
          styles.micButton,
          isRecording && styles.micButtonRecording,
        ]}
        onPress={handleToggle}
      >
        <Text style={styles.micButtonText}>
          {sessionActive ? "Stop Listening" : "Start Listening"}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F9FAFB",
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: 8,
  },
  heading: {
    fontSize: 24,
    fontWeight: "700",
    color: "#111827",
    // Take the flexible space and shrink/ellipsize rather than shove the
    // status text off the right edge.
    flex: 1,
    flexShrink: 1,
    marginRight: 8,
  },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    // Never shrink: the status keeps its full width, the heading yields.
    flexShrink: 0,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  statusText: {
    fontSize: 13,
    fontWeight: "600",
    textTransform: "capitalize",
    // A fixed-ish width right-aligned so the dot doesn't jump as the word
    // length changes ("live" vs "disconnected").
    minWidth: 92,
    textAlign: "right",
  },
  identityRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingBottom: 4,
    gap: 8,
  },
  identityChip: {
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#4A90D9",
    backgroundColor: "#EFF6FF",
  },
  identityChipText: {
    fontSize: 13,
    fontWeight: "600",
    color: "#4A90D9",
  },
  identityHint: {
    fontSize: 12,
    color: "#6B7280",
    fontStyle: "italic",
  },
  explainerCard: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    marginHorizontal: 16,
    marginVertical: 8,
    padding: 14,
    gap: 6,
  },
  explainerLine: {
    fontSize: 13.5,
    lineHeight: 19,
    color: "#374151",
  },
  banner: {
    backgroundColor: "#FEF3C7",
    borderLeftWidth: 4,
    borderLeftColor: "#F59E0B",
    marginHorizontal: 16,
    marginBottom: 4,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
  },
  bannerText: {
    fontSize: 13,
    color: "#92400E",
  },
  errorBanner: {
    backgroundColor: "#FEE2E2",
    borderLeftWidth: 4,
    borderLeftColor: "#EF4444",
    marginHorizontal: 16,
    marginBottom: 4,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
  },
  errorBannerText: {
    fontSize: 13,
    color: "#991B1B",
  },
  modeRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 8,
    gap: 8,
  },
  modeLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: "#374151",
    marginRight: 4,
  },
  modeButton: {
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#FFFFFF",
  },
  modeButtonActive: {
    backgroundColor: "#4A90D9",
    borderColor: "#4A90D9",
  },
  modeButtonText: {
    fontSize: 13,
    fontWeight: "500",
    color: "#6B7280",
  },
  modeButtonTextActive: {
    color: "#FFFFFF",
  },
  speechUnavailableText: {
    fontSize: 12,
    color: "#6B7280",
    paddingHorizontal: 16,
    paddingBottom: 4,
  },
  suggestionsContainer: {
    maxHeight: 200,
    paddingBottom: 8,
  },
  suggestionsTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#1F2937",
    paddingHorizontal: 16,
    paddingTop: 8,
    marginBottom: 4,
  },
  feedEntryNewest: {
    opacity: 1,
  },
  feedEntryOlder: {
    // Faded so the eye lands on the newest advice, but still legible.
    opacity: 0.75,
  },
  nudgeBanner: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    backgroundColor: "#FFFBEB",
    borderLeftWidth: 4,
    borderLeftColor: "#F59E0B",
    marginHorizontal: 16,
    marginVertical: 6,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 8,
  },
  nudgeBannerMuted: {
    opacity: 0.5,
  },
  nudgeBadge: {
    backgroundColor: "#F59E0B",
    paddingVertical: 2,
    paddingHorizontal: 8,
    borderRadius: 6,
  },
  nudgeBadgeText: {
    fontSize: 10,
    fontWeight: "700",
    color: "#FFFFFF",
    letterSpacing: 0.5,
  },
  nudgeText: {
    flex: 1,
    fontSize: 14,
    fontWeight: "700",
    color: "#92400E",
  },
  reviewButton: {
    backgroundColor: "#EFF6FF",
    borderWidth: 1,
    borderColor: "#4A90D9",
    marginHorizontal: 16,
    marginTop: 8,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
  },
  reviewButtonText: {
    color: "#4A90D9",
    fontSize: 16,
    fontWeight: "700",
  },
  micButton: {
    backgroundColor: "#4A90D9",
    marginHorizontal: 16,
    marginVertical: 12,
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: "center",
  },
  micButtonRecording: {
    backgroundColor: "#EF4444",
  },
  micButtonText: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "700",
  },
});
