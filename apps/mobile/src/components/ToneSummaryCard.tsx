import React from "react";
import { View, Text, StyleSheet } from "react-native";

import type { ToneSummary } from "../api/client";
import {
  describeBucket,
  meanLine,
  personName,
  toneChipColors,
  topLabels,
} from "../screens/toneTrends";

const INK = "#1F2937";
const MUTED = "#6B7280";

interface ToneSummaryCardProps {
  /** A live session's tone summary (analysis.live.tone_summary). */
  summary: ToneSummary;
  /** Optional heading override (defaults to "Your tone"). */
  title?: string;
  testID?: string;
}

/**
 * "Your tone" — the Track 2 tone/identity view shared by Replay, Dynamics,
 * the therapist SessionDetail, and (in miniature) YourDay: the user's OWN
 * tone label distribution over the session, their escalations, and the
 * same split per conversation partner ("with Mom · mostly warm · 1
 * escalation"). Reads straight from the server's buckets — nothing is
 * inferred here.
 *
 * Renders nothing when the summary has no self bucket (no turn in the
 * session is the user's) — there is no honest "your tone" to show then.
 * Audio-tone counts appear ONLY when the server said they may be surfaced.
 */
export default function ToneSummaryCard({
  summary,
  title = "Your tone",
  testID = "tone-summary",
}: ToneSummaryCardProps) {
  const me = summary.self;
  if (!me || me.turns === 0) return null;

  const chips = topLabels(me.labels, 4);
  const line = describeBucket(me.labels, me.escalation_count, me.scored_turns);
  const means = meanLine(me);
  const unscored = me.turns - me.scored_turns;
  const audio = summary.audio_tone_surfaced ? summary.audio : null;
  const people = summary.people.filter((p) => p.self_turns > 0);

  return (
    <View style={styles.card} testID={testID}>
      <Text style={styles.title}>{title}</Text>
      {chips.length > 0 ? (
        <View style={styles.chipRow}>
          {chips.map((c) => {
            const colors = toneChipColors(c.label);
            return (
              <View
                key={c.label}
                style={[styles.chip, { backgroundColor: colors.bg }]}
                testID={`${testID}-chip-${c.label}`}
              >
                <Text style={[styles.chipText, { color: colors.fg }]}>
                  {c.label} ×{c.count}
                </Text>
              </View>
            );
          })}
        </View>
      ) : (
        <Text style={styles.muted} testID={`${testID}-unscored`}>
          Your turns carried no tone reading in this session.
        </Text>
      )}
      {line && (
        <Text style={styles.line} testID={`${testID}-line`}>
          {line}
          {unscored > 0
            ? ` · ${unscored} turn${unscored === 1 ? "" : "s"} unscored`
            : ""}
        </Text>
      )}
      {means && <Text style={styles.means}>{means}</Text>}

      {audio && audio.turns > 0 && (
        <Text style={styles.audio} testID={`${testID}-audio`}>
          From your voice:{" "}
          {topLabels(audio.labels, 3)
            .map((c) => `${c.label} ×${c.count}`)
            .join(", ")}
          {audio.escalation_count > 0
            ? ` · ${audio.escalation_count} heated`
            : ""}
        </Text>
      )}

      {people.length > 0 && (
        <View style={styles.people}>
          {people.map((p) => {
            const summaryLine = describeBucket(
              p.labels,
              p.escalation_count,
              p.scored_turns,
            );
            const name = personName(p);
            return (
              <View
                key={p.speaker}
                style={styles.personRow}
                testID={`${testID}-person-${p.speaker}`}
              >
                <Text style={styles.personName} numberOfLines={1}>
                  with {name}
                </Text>
                <Text style={styles.personLine} numberOfLines={1}>
                  {summaryLine ?? `${p.self_turns} turn${p.self_turns === 1 ? "" : "s"} · no tone reading`}
                </Text>
              </View>
            );
          })}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    padding: 14,
    marginBottom: 12,
  },
  title: {
    fontSize: 15,
    fontWeight: "700",
    color: INK,
    marginBottom: 8,
  },
  chipRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
    marginBottom: 6,
  },
  chip: {
    borderRadius: 12,
    paddingVertical: 3,
    paddingHorizontal: 10,
  },
  chipText: {
    fontSize: 12.5,
    fontWeight: "600",
  },
  line: {
    fontSize: 13,
    color: INK,
  },
  means: {
    fontSize: 12,
    color: MUTED,
    marginTop: 2,
  },
  muted: {
    fontSize: 13,
    color: MUTED,
  },
  audio: {
    fontSize: 12.5,
    color: MUTED,
    marginTop: 6,
  },
  people: {
    marginTop: 10,
    borderTopWidth: 1,
    borderTopColor: "#F3F4F6",
    paddingTop: 8,
    gap: 6,
  },
  personRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 8,
  },
  personName: {
    fontSize: 13.5,
    fontWeight: "600",
    color: INK,
    flexShrink: 1,
  },
  personLine: {
    fontSize: 12.5,
    color: MUTED,
    flexShrink: 1,
    textAlign: "right",
  },
});
