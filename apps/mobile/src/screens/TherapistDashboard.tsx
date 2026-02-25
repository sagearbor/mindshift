import React, { useEffect, useMemo } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import { useDashboardStore, SavedSession } from "../store/dashboardStore";
import ToneSparkline from "../components/ToneSparkline";

interface TherapistDashboardProps {
  onSelectSession: (id: string) => void;
}

export default function TherapistDashboard({
  onSelectSession,
}: TherapistDashboardProps) {
  const { sessions, roleFilter, loading, fetchSessions, setRoleFilter } =
    useDashboardStore();

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  const roles = useMemo(() => {
    const set = new Set(sessions.map((s) => s.role));
    return Array.from(set).sort();
  }, [sessions]);

  const filteredSessions = useMemo(() => {
    if (!roleFilter) return sessions;
    return sessions.filter((s) => s.role === roleFilter);
  }, [sessions, roleFilter]);

  const grouped = useMemo(() => {
    const map = new Map<string, SavedSession[]>();
    for (const session of filteredSessions) {
      const group = map.get(session.role) || [];
      group.push(session);
      map.set(session.role, group);
    }
    return map;
  }, [filteredSessions]);

  return (
    <ScrollView
      style={styles.flex}
      contentContainerStyle={styles.content}
      testID="therapist-dashboard"
    >
      <Text style={styles.heading}>Therapist Dashboard</Text>

      {/* Role filter chips */}
      <View style={styles.filterRow}>
        <TouchableOpacity
          testID="filter-all"
          style={[styles.filterChip, !roleFilter && styles.filterChipActive]}
          onPress={() => setRoleFilter(null)}
        >
          <Text
            style={[
              styles.filterChipText,
              !roleFilter && styles.filterChipTextActive,
            ]}
          >
            All
          </Text>
        </TouchableOpacity>
        {roles.map((role) => (
          <TouchableOpacity
            key={role}
            testID={`filter-${role}`}
            style={[
              styles.filterChip,
              roleFilter === role && styles.filterChipActive,
            ]}
            onPress={() => setRoleFilter(roleFilter === role ? null : role)}
          >
            <Text
              style={[
                styles.filterChipText,
                roleFilter === role && styles.filterChipTextActive,
              ]}
            >
              {role}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {loading && (
        <ActivityIndicator
          testID="dashboard-loading"
          size="large"
          color="#4A90D9"
          style={styles.loader}
        />
      )}

      {!loading && filteredSessions.length === 0 && (
        <Text style={styles.emptyText}>No sessions found.</Text>
      )}

      {/* Session list grouped by role */}
      {Array.from(grouped.entries()).map(([role, groupSessions]) => (
        <View key={role} style={styles.group}>
          <Text style={styles.groupTitle}>{role}</Text>
          {groupSessions.map((session) => (
            <TouchableOpacity
              key={session.id}
              testID={`session-${session.id}`}
              style={styles.sessionCard}
              onPress={() => onSelectSession(session.id)}
            >
              <View style={styles.sessionHeader}>
                <Text style={styles.sessionDate}>
                  {new Date(session.date).toLocaleDateString()}
                </Text>
                <View style={styles.scoreBadge}>
                  <Text style={styles.scoreText}>
                    {Math.round(session.avgPleasantness)}
                  </Text>
                </View>
              </View>
              <Text style={styles.sessionMeta}>
                {session.turns.length} turns
              </Text>
              <ToneSparkline
                scores={session.turns.map((t) => t.toneScores.pleasantness)}
                width={200}
                height={36}
                color={getScoreColor(session.avgPleasantness)}
              />
            </TouchableOpacity>
          ))}
        </View>
      ))}
    </ScrollView>
  );
}

function getScoreColor(score: number): string {
  if (score >= 70) return "#10B981";
  if (score >= 40) return "#F59E0B";
  return "#EF4444";
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  content: {
    paddingTop: 60,
    paddingBottom: 40,
    paddingHorizontal: 16,
  },
  heading: {
    fontSize: 24,
    fontWeight: "700",
    textAlign: "center",
    marginBottom: 16,
    color: "#111827",
  },
  filterRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginBottom: 16,
  },
  filterChip: {
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#F9FAFB",
  },
  filterChipActive: {
    backgroundColor: "#4A90D9",
    borderColor: "#4A90D9",
  },
  filterChipText: {
    fontSize: 13,
    color: "#374151",
  },
  filterChipTextActive: {
    color: "#FFFFFF",
    fontWeight: "600",
  },
  loader: {
    marginTop: 40,
  },
  emptyText: {
    textAlign: "center",
    color: "#9CA3AF",
    fontSize: 15,
    marginTop: 40,
  },
  group: {
    marginBottom: 20,
  },
  groupTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#1F2937",
    marginBottom: 8,
  },
  sessionCard: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.08,
    shadowRadius: 3,
    elevation: 2,
  },
  sessionHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 4,
  },
  sessionDate: {
    fontSize: 14,
    fontWeight: "600",
    color: "#1F2937",
  },
  scoreBadge: {
    backgroundColor: "#EFF6FF",
    paddingVertical: 2,
    paddingHorizontal: 8,
    borderRadius: 8,
  },
  scoreText: {
    fontSize: 14,
    fontWeight: "700",
    color: "#4A90D9",
  },
  sessionMeta: {
    fontSize: 12,
    color: "#6B7280",
    marginBottom: 8,
  },
});
