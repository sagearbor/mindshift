import React, { useMemo } from "react";
import { View, Text, StyleSheet, useWindowDimensions } from "react-native";
import ToneSparkline from "./ToneSparkline";
import type { PersonScore, Scoreboard } from "../live/pleasantness";
import { resolveSpeakerColors } from "../utils/speakerColors";

export interface ScoreboardPanelProps {
  /** Keyed by raw speaker label; `nameOf` turns a key into what to show. */
  board: Scoreboard | null;
  nameOf?: (speaker: string) => string;
  title?: string;
  /** Shown when nobody has a score yet (the honest reason, e.g. "needs
   *  on-device coaching"). */
  emptyText?: string;
  testID?: string;
}

/** The lead line's copy — kind on purpose: this is a race to be nicer. */
export function leadCopy(board: Scoreboard | null, nameOf: (s: string) => string): string {
  const scored = (board?.people ?? []).filter((p) => typeof p.current === "number");
  if (scored.length === 0) return "Waiting for the first scored turn…";
  if (scored.length === 1) return `${nameOf(scored[0].speaker)} is warming up the room.`;
  if (!board?.lead) return "Neck and neck — you're both bringing it.";
  return `${nameOf(board.lead.speaker)} +${board.lead.margin} — leading with kindness.`;
}

function scoreColor(score: number | null): string {
  if (score === null) return "#9CA3AF";
  if (score >= 65) return "#10B981";
  if (score >= 40) return "#F59E0B";
  return "#EF4444";
}

/**
 * The "who's being nicer" scoreboard: one row per person (current score,
 * a sparkline of their last turns) and a playful lead line. Opt-in on the
 * Live Coach screen; the same panel renders a stored session's board on
 * SessionDetail / Replay so the post-session view matches what the couple
 * watched. Copy stays kind: higher is warmer/calmer/more constructive, and
 * both lines climbing is the win.
 */
export default function ScoreboardPanel({
  board,
  nameOf = (s) => s,
  title = "Kindness scoreboard",
  emptyText,
  testID = "scoreboard",
}: ScoreboardPanelProps) {
  const { width } = useWindowDimensions();
  // Phone-width friendly: the sparkline takes what's left after the name
  // column and the big number, never forcing the row past the screen.
  const sparkWidth = Math.max(80, Math.min(220, width - 32 - 24 - 110 - 64));
  const people: PersonScore[] = board?.people ?? [];
  const colors = useMemo(
    () => resolveSpeakerColors(people.map((p) => nameOf(p.speaker))),
    [people, nameOf],
  );
  const anyScored = people.some((p) => p.scoredTurns > 0);

  return (
    <View style={styles.card} testID={testID}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>{title}</Text>
        <Text style={styles.subtitle}>higher = warmer, calmer, more constructive</Text>
      </View>
      {!anyScored ? (
        <Text style={styles.empty} testID={`${testID}-empty`}>
          {emptyText ?? "Scores appear as people talk — the first turn with a tone read starts the board."}
        </Text>
      ) : null}
      {people.map((p) => {
        const name = nameOf(p.speaker);
        const color = colors.get(name) ?? "#4A90D9";
        return (
          <View key={p.speaker} style={styles.row} testID={`${testID}-row-${p.speaker}`}>
            <View style={styles.nameCol}>
              <Text style={[styles.name, { color }]} numberOfLines={1}>
                {name}
              </Text>
              <Text style={styles.turns}>
                {p.scoredTurns === 0 ? "no score yet" : `${p.scoredTurns} turn${p.scoredTurns === 1 ? "" : "s"}`}
              </Text>
            </View>
            <ToneSparkline scores={p.series} width={sparkWidth} height={36} color={color} />
            <Text
              style={[styles.score, { color: scoreColor(p.current) }]}
              testID={`${testID}-score-${p.speaker}`}
            >
              {p.current === null ? "—" : String(p.current)}
            </Text>
          </View>
        );
      })}
      {anyScored ? (
        <Text style={styles.lead} testID={`${testID}-lead`}>
          {leadCopy(board, nameOf)}
        </Text>
      ) : null}
      <Text style={styles.footer}>Everyone wins when both lines climb.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    marginHorizontal: 16,
    marginVertical: 6,
    paddingVertical: 10,
    paddingHorizontal: 12,
    gap: 6,
  },
  headerRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    alignItems: "baseline",
    justifyContent: "space-between",
    gap: 6,
  },
  title: {
    fontSize: 15,
    fontWeight: "700",
    color: "#1F2937",
  },
  subtitle: {
    fontSize: 11,
    color: "#6B7280",
  },
  empty: {
    fontSize: 12.5,
    color: "#6B7280",
    fontStyle: "italic",
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingVertical: 2,
  },
  nameCol: {
    width: 110,
    flexShrink: 1,
  },
  name: {
    fontSize: 14,
    fontWeight: "700",
  },
  turns: {
    fontSize: 11,
    color: "#9CA3AF",
  },
  score: {
    width: 56,
    textAlign: "right",
    fontSize: 28,
    fontWeight: "800",
    fontVariant: ["tabular-nums"],
  },
  lead: {
    fontSize: 13,
    fontWeight: "600",
    color: "#374151",
    marginTop: 2,
  },
  footer: {
    fontSize: 11,
    color: "#9CA3AF",
  },
});
