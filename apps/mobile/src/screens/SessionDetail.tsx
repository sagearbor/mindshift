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
    const totals: ToneScores = {
      warmth: 0,
      constructiveness: 0,
      calmness: 0,
      respect: 0,
      engagement: 0,
      pleasantness: 0,
    };
    for (const turn of session.turns) {
      for (const key of Object.keys(totals) as (keyof ToneScores)[]) {
        totals[key] += turn.toneScores[key];
      }
    }
    const count = session.turns.length;
    for (const key of Object.keys(totals) as (keyof ToneScores)[]) {
      totals[key] = Math.round(totals[key] / count);
    }
    return totals;
  }, [session]);

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

      {/* Tone timeline */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Tone Timeline</Text>
        <ToneSparkline
          scores={session.turns.map((t) => t.toneScores.pleasantness)}
          width={320}
          height={60}
          color="#4A90D9"
        />
      </View>

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
          const toneColor = getTurnColor(turn.toneScores.pleasantness);
          return (
            <View
              key={i}
              style={[styles.turnCard, { borderLeftColor: toneColor }]}
              testID={`turn-${i}`}
            >
              <View style={styles.turnHeader}>
                <Text style={styles.turnSpeaker}>{turn.speaker}</Text>
                <View style={styles.turnScoreBadge}>
                  <Text style={[styles.turnScoreText, { color: toneColor }]}>
                    {Math.round(turn.toneScores.pleasantness)}
                  </Text>
                </View>
              </View>
              <Text style={styles.turnText}>{turn.text}</Text>
              <Text style={styles.empathyLabel}>
                Empathy: {turn.empathyLevel}
              </Text>
            </View>
          );
        })}
      </View>
    </ScrollView>
  );
}

function getTurnColor(pleasantness: number): string {
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
