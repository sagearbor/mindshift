import React, { useMemo } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Alert,
  Share,
  StyleSheet,
} from "react-native";
import { useDashboardStore, ToneScores } from "../store/dashboardStore";
import ToneSparkline from "../components/ToneSparkline";
import ToneSummaryCard from "../components/ToneSummaryCard";
import CouldHaveSaidList from "../components/CouldHaveSaidList";
import { modeLabel, toneChipColors } from "./toneTrends";

interface SessionDetailProps {
  sessionId: string;
  onBack: () => void;
}

export default function SessionDetail({
  sessionId,
  onBack,
}: SessionDetailProps) {
  const { sessions, exportSession } = useDashboardStore();
  const session = sessions.find((s) => s.id === sessionId);

  const aggregateStats = useMemo(() => {
    if (!session || session.turns.length === 0) return null;
    // Average each dimension over the turns that actually CARRY it: a
    // server-projected live session scores only what was measured
    // (Track 2), so a missing key is skipped, never counted as zero. A
    // dimension no turn carries is omitted from the grid entirely.
    const keys: (keyof ToneScores)[] = [
      "warmth",
      "constructiveness",
      "calmness",
      "respect",
      "engagement",
      "pleasantness",
    ];
    const totals: Partial<ToneScores> = {};
    for (const key of keys) {
      let sum = 0;
      let n = 0;
      for (const turn of session.turns) {
        const v = turn.toneScores[key];
        if (typeof v === "number") {
          sum += v;
          n += 1;
        }
      }
      if (n > 0) totals[key] = Math.round(sum / n);
    }
    return Object.keys(totals).length > 0 ? totals : null;
  }, [session]);

  // Track 2: the reflections keyed by turn index so each self turn can show
  // its "what you could have said" card right beneath the words.
  const reflectionsByTurn = useMemo(() => {
    const map = new Map<number, NonNullable<typeof session>["couldHaveSaid"]>();
    for (const item of session?.couldHaveSaid ?? []) {
      map.set(item.turn_index, [item]);
    }
    return map;
  }, [session]);
  const pleasantnessSeries = session
    ? session.turns
        .map((t) => t.toneScores.pleasantness)
        .filter((v): v is number => typeof v === "number")
    : [];
  const modeText = modeLabel(session?.mode);

  const handleExport = async () => {
    try {
      const text = await exportSession(sessionId);
      await Share.share({ message: text, title: "MindShift Session Export" });
    } catch {
      Alert.alert("Export Failed", "Could not export session data.");
    }
  };

  if (!session) {
    return (
      <View style={styles.centered} testID="session-detail-empty">
        <Text style={styles.emptyText}>Session not found.</Text>
        <TouchableOpacity style={styles.backButton} onPress={onBack}>
          <Text style={styles.backButtonText}>Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.flex}
      contentContainerStyle={styles.content}
      testID="session-detail"
    >
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          testID="back-button"
          onPress={onBack}
          style={styles.backButton}
        >
          <Text style={styles.backButtonText}>Back</Text>
        </TouchableOpacity>
        <TouchableOpacity
          testID="export-button"
          onPress={handleExport}
          style={styles.exportButton}
        >
          <Text style={styles.exportButtonText}>Export</Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.heading}>Session Detail</Text>
      <Text style={styles.meta}>
        {new Date(session.date).toLocaleDateString()} — {session.role}
      </Text>
      {/* Track 2: a live session says so, with its coaching mode and the
          honest state of its analysis (heats arrive after the batch pass). */}
      {session.source === "live" && (
        <Text style={styles.liveMeta} testID="session-live-meta">
          {session.title ? `${session.title} · ` : ""}Live session
          {modeText ? ` · ${modeText}` : ""}
          {session.analysisStatus === "lite"
            ? " · heat analysis pending"
            : session.analysisStatus === "failed"
              ? " · heat analysis unavailable"
              : ""}
        </Text>
      )}

      {/* Tone timeline — only over turns that carry a pleasantness score. */}
      {pleasantnessSeries.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Tone Timeline</Text>
          <ToneSparkline
            scores={pleasantnessSeries}
            width={320}
            height={60}
            color="#4A90D9"
          />
        </View>
      )}

      {/* Track 2: the user's own tone + per-person split, same card as
          Replay/Dynamics so a therapist and a patient read the same thing. */}
      {session.toneSummary && (
        <View style={styles.section}>
          <ToneSummaryCard
            summary={session.toneSummary}
            title="Patient's tone"
            testID="session-tone-summary"
          />
        </View>
      )}

      {/* Aggregate stats */}
      {aggregateStats && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Average Scores</Text>
          <View style={styles.statsGrid}>
            {(
              Object.entries(aggregateStats) as [keyof ToneScores, number][]
            ).map(([key, value]) => (
              <View key={key} style={styles.statItem}>
                <Text style={styles.statValue}>{value}</Text>
                <Text style={styles.statLabel}>{key}</Text>
              </View>
            ))}
          </View>
        </View>
      )}

      {/* Transcript */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Transcript</Text>
        {session.turns.map((turn, i) => {
          const pleasantness = turn.toneScores.pleasantness;
          const toneColor = getTurnColor(pleasantness);
          const chip = turn.toneLabel ? toneChipColors(turn.toneLabel) : null;
          const reflections = reflectionsByTurn.get(i);
          const scoreBadge =
            typeof pleasantness === "number" ? (
              <View style={styles.turnScoreBadge}>
                <Text style={[styles.turnScoreText, { color: toneColor }]}>
                  {Math.round(pleasantness)}
                </Text>
              </View>
            ) : null;
          return (
            <View
              key={i}
              style={[styles.turnCard, { borderLeftColor: toneColor }]}
              testID={`turn-${i}`}
            >
              <View style={styles.turnHeader}>
                <Text style={styles.turnSpeaker}>{turn.speaker}</Text>
                {/* Track 2: the phone's tone read for this turn (+ an
                    escalation marker) sits beside the score. The row wrapper
                    exists only when there IS a chip, so a legacy session
                    renders byte-for-byte as before. */}
                {chip && turn.toneLabel ? (
                  <View style={styles.turnBadges}>
                    <View
                      style={[styles.toneChip, { backgroundColor: chip.bg }]}
                      testID={`turn-${i}-tone`}
                    >
                      <Text style={[styles.toneChipText, { color: chip.fg }]}>
                        {turn.toneLabel}
                        {turn.escalated ? " ↑" : ""}
                      </Text>
                    </View>
                    {scoreBadge}
                  </View>
                ) : (
                  scoreBadge
                )}
              </View>
              <Text style={styles.turnText}>{turn.text}</Text>
              {typeof turn.empathyLevel === "number" ? (
                <Text style={styles.empathyLabel}>
                  Empathy: {turn.empathyLevel}
                </Text>
              ) : null}
              {reflections && reflections.length > 0 ? (
                <View style={styles.reflection}>
                  <Text style={styles.reflectionTitle}>Could have said</Text>
                  <CouldHaveSaidList
                    items={reflections}
                    turns={null}
                    testID={`turn-${i}-could-have-said`}
                  />
                </View>
              ) : null}
            </View>
          );
        })}
      </View>
    </ScrollView>
  );
}

function getTurnColor(pleasantness: number | undefined): string {
  // No pleasantness yet (a live session before its batch analysis) → the
  // neutral gray the rest of the app uses for "heat unknown", never a color
  // that would read as a verdict.
  if (typeof pleasantness !== "number") return "#9CA3AF";
  if (pleasantness >= 65) return "#10B981"; // green = warm
  if (pleasantness >= 40) return "#F59E0B"; // amber = neutral
  return "#EF4444"; // red = defensive
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  content: {
    paddingTop: 60,
    paddingBottom: 40,
    paddingHorizontal: 16,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  backButton: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: "#F3F4F6",
  },
  backButtonText: {
    fontSize: 14,
    fontWeight: "600",
    color: "#4A90D9",
  },
  exportButton: {
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 8,
    backgroundColor: "#4A90D9",
  },
  exportButtonText: {
    fontSize: 14,
    fontWeight: "600",
    color: "#FFFFFF",
  },
  heading: {
    fontSize: 22,
    fontWeight: "700",
    color: "#111827",
    marginBottom: 4,
  },
  meta: {
    fontSize: 14,
    color: "#6B7280",
    marginBottom: 20,
  },
  liveMeta: {
    fontSize: 13,
    color: "#4A90D9",
    fontWeight: "600",
    marginTop: -14,
    marginBottom: 20,
  },
  turnBadges: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  toneChip: {
    borderRadius: 8,
    paddingVertical: 2,
    paddingHorizontal: 8,
  },
  toneChipText: {
    fontSize: 11.5,
    fontWeight: "600",
  },
  reflection: {
    marginTop: 8,
  },
  reflectionTitle: {
    fontSize: 11.5,
    fontWeight: "700",
    color: "#6B7280",
    textTransform: "uppercase",
    letterSpacing: 0.4,
    marginBottom: 4,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#1F2937",
    marginBottom: 10,
  },
  statsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  statItem: {
    backgroundColor: "#F9FAFB",
    borderRadius: 10,
    padding: 10,
    alignItems: "center",
    minWidth: 90,
  },
  statValue: {
    fontSize: 20,
    fontWeight: "700",
    color: "#4A90D9",
  },
  statLabel: {
    fontSize: 11,
    color: "#6B7280",
    textTransform: "capitalize",
    marginTop: 2,
  },
  turnCard: {
    backgroundColor: "#FFFFFF",
    borderRadius: 10,
    padding: 12,
    marginBottom: 8,
    borderLeftWidth: 4,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  turnHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 4,
  },
  turnSpeaker: {
    fontSize: 13,
    fontWeight: "700",
    color: "#374151",
  },
  turnScoreBadge: {
    backgroundColor: "#F9FAFB",
    paddingVertical: 2,
    paddingHorizontal: 8,
    borderRadius: 6,
  },
  turnScoreText: {
    fontSize: 13,
    fontWeight: "700",
  },
  turnText: {
    fontSize: 14,
    lineHeight: 20,
    color: "#1F2937",
    marginBottom: 4,
  },
  empathyLabel: {
    fontSize: 11,
    color: "#9CA3AF",
  },
  emptyText: {
    fontSize: 15,
    color: "#9CA3AF",
    marginBottom: 16,
  },
});
