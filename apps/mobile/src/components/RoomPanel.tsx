import React, { useState } from "react";
import { View, Text, Switch, TouchableOpacity, StyleSheet, Platform } from "react-native";
import { IDLE_ROOM_STATE, ROOM_WAKE_PHRASE, type RoomViewState } from "../live/roomMode";

/** The consent banner's words — exported so tests and the screen agree. */
export const ROOM_BANNER_TITLE = "MindShift is listening to this room";
export const ROOM_BANNER_HINT = `Say “${ROOM_WAKE_PHRASE}, …” to ask a question out loud. Tap Stop to end.`;
export const ROOM_CONSENT_NOTE =
  "Tell the room before you start: everyone present is heard and transcribed while Room mode is on.";

interface TranscriptLine {
  speaker: string;
  text: string;
}

interface Props {
  state?: RoomViewState | null;
  sessionActive: boolean;
  /** Recent transcript lines (oldest first); shown only when the user opens
   *  "Show transcript". */
  transcript?: readonly TranscriptLine[];
  /** Whether answers are spoken aloud (the screen's speak-aloud switch). */
  speakAloud: boolean;
  onSpeakAloudChange: (on: boolean) => void;
  /** False when this platform has no TTS — answers stay on screen. */
  speechAvailable?: boolean;
  /** How many library items are selected (cards and answers need some). */
  librarySelected: number;
}

const TRANSCRIPT_LINES = 6;

/**
 * Room mode's screen: a large, readable layout meant to be screen-shared or
 * read across a table. A persistent consent banner while the session runs;
 * the latest spoken answer; the latest few library cards (never spoken);
 * an optional live transcript. No coaching UI at all.
 */
export default function RoomPanel({
  state,
  sessionActive,
  transcript = [],
  speakAloud,
  onSpeakAloudChange,
  speechAvailable = true,
  librarySelected,
}: Props) {
  const room = state ?? IDLE_ROOM_STATE;
  const [showTranscript, setShowTranscript] = useState(false);
  const answer = room.answers[0] ?? null;
  const lines = transcript.slice(-TRANSCRIPT_LINES);

  return (
    <View style={styles.wrap} testID="room-panel">
      {sessionActive ? (
        <View
          style={styles.banner}
          testID="room-consent-banner"
          accessibilityRole="alert"
          accessibilityLiveRegion="polite"
        >
          <Text style={styles.bannerTitle}>{"● "}{ROOM_BANNER_TITLE}</Text>
          <Text style={styles.bannerHint}>{ROOM_BANNER_HINT}</Text>
        </View>
      ) : (
        <View style={styles.explainer} testID="room-explainer">
          <Text style={styles.explainerTitle}>Room assistant</Text>
          <Text style={styles.explainerLine}>
            Put the phone on the table (or share this screen from a laptop). When someone says
            {` “${ROOM_WAKE_PHRASE}, …”`} it answers out loud in a sentence or two, from your
            library and the conversation. When the conversation mentions something your library
            covers, the fact shows here.
          </Text>
          <Text style={styles.explainerLine}>No personal coaching in this mode.</Text>
          <Text style={[styles.explainerLine, styles.consentNote]} testID="room-consent-note">
            {ROOM_CONSENT_NOTE}
          </Text>
        </View>
      )}

      {librarySelected === 0 ? (
        <Text style={styles.warn} testID="room-no-library">
          No library items selected — pick some below so the assistant has facts to show and answer from.
        </Text>
      ) : null}

      <View style={styles.toggleRow} testID="room-speak-row">
        <Text style={styles.toggleLabel}>Speak answers aloud</Text>
        <Switch testID="room-speak-switch" value={speakAloud} onValueChange={onSpeakAloudChange} />
      </View>
      {!speechAvailable ? (
        <Text style={styles.warn} testID="room-speech-unavailable">
          Spoken answers aren&apos;t available on this platform — they show on screen only.
        </Text>
      ) : null}

      {room.listening ? (
        <Text style={styles.listening} testID="room-listening">
          Listening for your question…
        </Text>
      ) : null}

      {answer ? (
        <View style={[styles.answer, !answer.known && styles.answerUnknown]} testID="room-answer">
          {answer.question ? (
            <Text style={styles.answerQuestion} numberOfLines={2}>
              {`“${answer.question}”`}
            </Text>
          ) : null}
          <Text style={styles.answerText} testID="room-answer-text">
            {answer.text}
          </Text>
          {!answer.known ? (
            <Text style={styles.answerTag} testID="room-answer-unknown">
              Not in the library or the conversation
            </Text>
          ) : null}
        </View>
      ) : null}
      {room.answerError ? (
        <Text style={styles.warn} testID="room-answer-error">
          Couldn&apos;t answer that one ({room.answerError}).
        </Text>
      ) : null}

      {room.cards.length > 0 ? (
        <View style={styles.cards} testID="room-cards">
          {room.cards.map((card, i) => (
            <View
              key={card.id}
              style={[styles.card, i > 0 && styles.cardOlder]}
              testID={`room-card-${card.id}`}
            >
              {card.title ? <Text style={styles.cardTitle}>{card.title}</Text> : null}
              <Text style={styles.cardFact}>{card.fact}</Text>
              <Text style={styles.cardSource}>From: {card.sourceTitle || "your library"}</Text>
            </View>
          ))}
        </View>
      ) : sessionActive ? (
        <Text style={styles.empty} testID="room-cards-empty">
          Library facts appear here when the conversation mentions them.
        </Text>
      ) : null}

      {sessionActive || lines.length > 0 ? (
        <TouchableOpacity
          testID="room-transcript-toggle"
          accessibilityRole="button"
          accessibilityState={{ expanded: showTranscript }}
          onPress={() => setShowTranscript((v) => !v)}
        >
          <Text style={styles.transcriptToggle}>
            {showTranscript ? "Hide transcript ▾" : "Show transcript ▸"}
          </Text>
        </TouchableOpacity>
      ) : null}
      {showTranscript ? (
        <View testID="room-transcript">
          {lines.map((l, i) => (
            <Text key={i} style={styles.transcriptLine}>
              <Text style={styles.transcriptSpeaker}>{l.speaker}: </Text>
              {l.text}
            </Text>
          ))}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    gap: 12,
    width: "100%",
    // Screen-shared from a laptop browser: a readable column, not a
    // full-width line of text across a projector.
    maxWidth: Platform.OS === "web" ? 1100 : undefined,
    alignSelf: "center",
  },
  banner: {
    backgroundColor: "#B91C1C",
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 16,
    gap: 4,
  },
  bannerTitle: { color: "#FFFFFF", fontSize: 20, fontWeight: "700" },
  bannerHint: { color: "#FEE2E2", fontSize: 15, lineHeight: 21 },
  explainer: {
    backgroundColor: "#F3F4F6",
    borderRadius: 12,
    padding: 16,
    gap: 8,
  },
  explainerTitle: { fontSize: 18, fontWeight: "700", color: "#111827" },
  explainerLine: { fontSize: 15, lineHeight: 21, color: "#374151" },
  consentNote: { fontWeight: "600", color: "#92400E" },
  warn: { fontSize: 14, lineHeight: 20, color: "#92400E" },
  toggleRow: { flexDirection: "row", alignItems: "center", gap: 12 },
  toggleLabel: { fontSize: 16, fontWeight: "600", color: "#374151" },
  listening: { fontSize: 20, fontWeight: "600", color: "#1D4ED8" },
  answer: {
    backgroundColor: "#EFF6FF",
    borderLeftWidth: 6,
    borderLeftColor: "#2563EB",
    borderRadius: 12,
    padding: 16,
    gap: 6,
  },
  answerUnknown: { backgroundColor: "#F9FAFB", borderLeftColor: "#9CA3AF" },
  answerQuestion: { fontSize: 16, fontStyle: "italic", color: "#4B5563" },
  answerText: { fontSize: 28, lineHeight: 36, fontWeight: "600", color: "#111827" },
  answerTag: { fontSize: 14, color: "#6B7280" },
  cards: { gap: 12 },
  card: {
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 12,
    padding: 16,
    gap: 6,
  },
  cardOlder: { opacity: 0.7 },
  cardTitle: { fontSize: 22, fontWeight: "700", color: "#111827" },
  cardFact: { fontSize: 24, lineHeight: 32, color: "#111827" },
  cardSource: { fontSize: 15, color: "#6B7280" },
  empty: { fontSize: 16, color: "#6B7280" },
  transcriptToggle: { fontSize: 15, fontWeight: "600", color: "#4A90D9" },
  transcriptLine: { fontSize: 18, lineHeight: 26, color: "#374151" },
  transcriptSpeaker: { fontWeight: "700" },
});
