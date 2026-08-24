import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  RefreshControl,
  StyleSheet,
  useWindowDimensions,
} from "react-native";
import { useDashboardStore, SavedSession } from "../store/dashboardStore";
import ToneSparkline from "../components/ToneSparkline";
import { describeBucket, modeLabel } from "./toneTrends";
import {
  acceptPatient,
  declinePatient,
  listPatients,
  type PatientLink,
} from "../api/therapist";

/** The patient list: every account that named this one as therapist
 *  (accepted — pending ones are requests, shown separately) plus every
 *  patient label present in the session list, so a patient who shared by
 *  hand without linking still appears. "You" (own sessions) stays first. */
export function patientRows(
  sessions: SavedSession[],
  patients: PatientLink[],
): { label: string; sessions: number; linked: boolean }[] {
  const counts = new Map<string, number>();
  for (const s of sessions) counts.set(s.role, (counts.get(s.role) ?? 0) + 1);
  const linked = new Set(
    patients
      .filter((p) => p.status === "accepted" && p.patient_email)
      .map((p) => p.patient_email as string),
  );
  const labels = new Set<string>([...counts.keys(), ...linked]);
  const rows = [...labels].map((label) => ({
    label,
    sessions: counts.get(label) ?? 0,
    linked: linked.has(label),
  }));
  rows.sort((a, b) => {
    if (a.label === "You") return -1;
    if (b.label === "You") return 1;
    return a.label.localeCompare(b.label);
  });
  return rows;
}

interface TherapistDashboardProps {
  onSelectSession: (id: string) => void;
  /** Return to the Settings screen (wired by App). Optional so the screen
   *  still renders standalone in tests; no back affordance without it. */
  onBack?: () => void;
}

export default function TherapistDashboard({
  onSelectSession,
  onBack,
}: TherapistDashboardProps) {
  const { sessions, roleFilter, loading, fetchSessions, setRoleFilter } =
    useDashboardStore();
  // Linked patients (two-sided setup): pending requests to accept/decline
  // and accepted patients for the list. null = not loaded / unavailable
  // (older server, offline) — the dashboard then shows sessions only.
  const [patients, setPatients] = useState<PatientLink[] | null>(null);
  const [patientBusy, setPatientBusy] = useState<string | null>(null);
  const [patientError, setPatientError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadPatients = useCallback(async () => {
    try {
      setPatients(await listPatients());
    } catch {
      setPatients(null);
    }
  }, []);
  // The sparkline must fit a phone-width Safari viewport (iPhone SE: 375 pt
  // minus the page + card padding) — never wider than the card.
  const { width: windowWidth } = useWindowDimensions();
  const sparkWidth = Math.max(120, Math.min(200, windowWidth - 16 * 2 - 14 * 2));

  useEffect(() => {
    fetchSessions();
    void loadPatients();
  }, [fetchSessions, loadPatients]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await Promise.all([fetchSessions(), loadPatients()]);
    } finally {
      setRefreshing(false);
    }
  }, [fetchSessions, loadPatients]);

  const handleAccept = useCallback(
    async (p: PatientLink) => {
      setPatientBusy(p.patient_uid);
      setPatientError(null);
      try {
        const updated = await acceptPatient(p.patient_uid);
        setPatients((prev) =>
          (prev ?? []).map((x) => (x.patient_uid === p.patient_uid ? updated : x)),
        );
      } catch {
        setPatientError("Couldn’t accept right now — please try again.");
      } finally {
        setPatientBusy(null);
      }
    },
    [],
  );

  const handleDecline = useCallback(
    async (p: PatientLink) => {
      setPatientBusy(p.patient_uid);
      setPatientError(null);
      try {
        await declinePatient(p.patient_uid);
        setPatients((prev) => (prev ?? []).filter((x) => x.patient_uid !== p.patient_uid));
      } catch {
        setPatientError("Couldn’t decline right now — please try again.");
      } finally {
        setPatientBusy(null);
      }
    },
    [],
  );

  const pending = useMemo(
    () => (patients ?? []).filter((p) => p.status === "pending"),
    [patients],
  );
  const rows = useMemo(() => patientRows(sessions, patients ?? []), [sessions, patients]);
  const roles = useMemo(() => rows.map((r) => r.label), [rows]);

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
      refreshControl={
        <RefreshControl
          testID="dashboard-refresh"
          refreshing={refreshing}
          onRefresh={handleRefresh}
        />
      }
    >
      {onBack && (
        <TouchableOpacity
          testID="dashboard-back"
          accessibilityRole="button"
          style={styles.backButton}
          onPress={onBack}
        >
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>
      )}
      <Text style={styles.heading}>Therapist Dashboard</Text>

      {/* Patients who named this account as their therapist and are waiting
          for an acknowledgement. Their sessions are already shared (the
          patient chose to); Accept lists them as a patient, Decline removes
          the link so nothing further is shared. */}
      {pending.length > 0 ? (
        <View style={styles.pendingCard} testID="pending-patients">
          <Text style={styles.pendingTitle}>Wants to share sessions with you</Text>
          {pending.map((p) => (
            <View key={p.patient_uid} style={styles.pendingRow} testID={`pending-${p.patient_uid}`}>
              <Text style={styles.pendingEmail} numberOfLines={1}>
                {p.patient_email ?? p.patient_uid}
              </Text>
              <TouchableOpacity
                testID={`accept-${p.patient_uid}`}
                accessibilityRole="button"
                style={styles.acceptButton}
                disabled={patientBusy === p.patient_uid}
                onPress={() => handleAccept(p)}
              >
                <Text style={styles.acceptText}>Accept</Text>
              </TouchableOpacity>
              <TouchableOpacity
                testID={`decline-${p.patient_uid}`}
                accessibilityRole="button"
                style={styles.declineButton}
                disabled={patientBusy === p.patient_uid}
                onPress={() => handleDecline(p)}
              >
                <Text style={styles.declineText}>Decline</Text>
              </TouchableOpacity>
            </View>
          ))}
          {patientError ? (
            <Text style={styles.patientError} testID="patient-error">
              {patientError}
            </Text>
          ) : null}
        </View>
      ) : null}

      {/* Patient list — "You" first, then every linked or sharing patient;
          tapping one filters the sessions below (the existing role filter). */}
      <Text style={styles.patientsTitle}>Patients</Text>
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
        {rows.map((row) => (
          <TouchableOpacity
            key={row.label}
            testID={`filter-${row.label}`}
            style={[
              styles.filterChip,
              roleFilter === row.label && styles.filterChipActive,
            ]}
            onPress={() => setRoleFilter(roleFilter === row.label ? null : row.label)}
          >
            <Text
              style={[
                styles.filterChipText,
                roleFilter === row.label && styles.filterChipTextActive,
              ]}
            >
              {row.label}
              {row.linked ? " ✓" : ""}
              {` · ${row.sessions}`}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
      {roles.length === 0 && !loading ? (
        <Text style={styles.patientsHint} testID="patients-empty">
          No patients yet. A patient links you from their Settings → My
          therapist; their sessions then appear here.
        </Text>
      ) : null}

      {loading && (
        <ActivityIndicator
          testID="dashboard-loading"
          size="large"
          color="#4A90D9"
          style={styles.loader}
        />
      )}

      {!loading && filteredSessions.length === 0 && (
        <Text style={styles.emptyText}>
          {roleFilter && rows.some((r) => r.label === roleFilter && r.linked)
            ? "No sessions from this patient yet — their next live session or recording will appear here."
            : "No sessions found."}
        </Text>
      )}

      {/* Session list grouped by role */}
      {Array.from(grouped.entries()).map(([role, groupSessions]) => (
        <View key={role} style={styles.group}>
          <Text style={styles.groupTitle}>{role}</Text>
          {groupSessions.map((session) => {
            // Track 2: a live session's self-tone one-liner ("mostly warm ·
            // 2 escalations") from the same server bucket SessionDetail shows.
            const me = session.toneSummary?.self ?? null;
            const toneLine = me
              ? describeBucket(me.labels, me.escalation_count, me.scored_turns)
              : null;
            const mode = modeLabel(session.mode);
            const scores = session.turns
              .map((t) => t.toneScores.pleasantness)
              .filter((v): v is number => typeof v === "number");
            return (
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
                    {/* "—" until the batch analysis has scored the turns —
                        never a fabricated 0. */}
                    {typeof session.avgPleasantness === "number"
                      ? Math.round(session.avgPleasantness)
                      : "—"}
                  </Text>
                </View>
              </View>
              {session.source === "live" && (
                <Text style={styles.liveBadge} testID={`session-${session.id}-live`}>
                  Live{mode ? ` · ${mode}` : ""}
                  {session.title ? ` · ${session.title}` : ""}
                </Text>
              )}
              <Text style={styles.sessionMeta}>
                {session.turns.length} turns
                {toneLine ? ` · ${toneLine}` : ""}
              </Text>
              {/* People labeling (name display only): the patient's own
                  names for who they spoke with — a person the server
                  identified or the patient labeled, never a raw
                  "Speaker B". */}
              {(() => {
                const named = (session.speakers ?? [])
                  .filter(
                    (s) =>
                      s.display !== "You" &&
                      s.labelSource &&
                      s.labelSource !== "generic" &&
                      s.labelSource !== "voice",
                  )
                  .map((s) => s.display);
                return named.length > 0 ? (
                  <Text style={styles.sessionPeople} testID={`session-${session.id}-people`}>
                    {`with ${named.join(", ")}`}
                  </Text>
                ) : null;
              })()}
              <ToneSparkline
                scores={scores}
                width={sparkWidth}
                height={36}
                color={getScoreColor(session.avgPleasantness)}
              />
            </TouchableOpacity>
            );
          })}
        </View>
      ))}
    </ScrollView>
  );
}

function getScoreColor(score: number | null): string {
  if (typeof score !== "number") return "#9CA3AF"; // unscored → neutral gray
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
  backButton: {
    alignSelf: "flex-start",
    minHeight: 44,
    justifyContent: "center",
    paddingRight: 12,
  },
  backButtonText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#4A90D9",
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
  sessionPeople: {
    fontSize: 12,
    color: "#374151",
    fontWeight: "600",
    marginTop: -4,
    marginBottom: 8,
  },
  liveBadge: {
    fontSize: 12,
    fontWeight: "600",
    color: "#4A90D9",
    marginBottom: 2,
  },
  pendingCard: {
    backgroundColor: "#FFFBEB",
    borderWidth: 1,
    borderColor: "#FCD34D",
    borderRadius: 12,
    padding: 12,
    marginBottom: 14,
    gap: 8,
  },
  pendingTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: "#92400E",
    textTransform: "uppercase",
    letterSpacing: 0.4,
  },
  pendingRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  pendingEmail: {
    flex: 1,
    fontSize: 14,
    fontWeight: "600",
    color: "#1F2937",
  },
  acceptButton: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: "#4A90D9",
  },
  acceptText: {
    color: "#FFFFFF",
    fontSize: 13,
    fontWeight: "700",
  },
  declineButton: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#D1D5DB",
  },
  declineText: {
    color: "#6B7280",
    fontSize: 13,
    fontWeight: "600",
  },
  patientError: {
    fontSize: 12.5,
    color: "#DC2626",
  },
  patientsTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: "#6B7280",
    textTransform: "uppercase",
    letterSpacing: 0.4,
    marginBottom: 8,
  },
  patientsHint: {
    fontSize: 13,
    lineHeight: 18,
    color: "#6B7280",
    marginBottom: 12,
  },
});
