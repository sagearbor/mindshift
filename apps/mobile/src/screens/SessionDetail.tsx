import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Alert,
  Share,
  StyleSheet,
} from "react-native";
import { useDashboardStore, ToneScores, type SavedSession } from "../store/dashboardStore";
import ToneSparkline from "../components/ToneSparkline";
import ToneSummaryCard from "../components/ToneSummaryCard";
import CouldHaveSaidList from "../components/CouldHaveSaidList";
import WhoIsThisSheet from "../components/WhoIsThisSheet";
import * as apiClient from "../api/client";
import type { PatchSpeakerLabelsResult, VoicePerson } from "../api/client";
import { isEnrolledPersonLabel } from "../utils/people";
import { modeLabel, toneChipColors } from "./toneTrends";

/**
 * Apply a speaker-labels save to a stored session row: every turn (and the
 * `speakers` list) whose raw speaker id is in the server's resolved map
 * takes the new display name + provenance. Pure; exported for tests.
 */
export function applyLabelsToSession(
  session: SavedSession,
  labels: PatchSpeakerLabelsResult["speaker_labels"],
): SavedSession {
  const turns = session.turns.map((t) => {
    const entry = t.speakerId ? labels[t.speakerId] : undefined;
    if (!entry) return t;
    return {
      ...t,
      speaker: entry.display_label,
      labelSource: entry.label_source,
      personId: entry.person_id ?? null,
    };
  });
  const speakers = (session.speakers ?? []).map((s) => {
    const entry = labels[s.id];
    if (!entry) return s;
    return {
      ...s,
      display: entry.display_label,
      labelSource: entry.label_source,
      personId: entry.person_id ?? null,
    };
  });
  return { ...session, turns, speakers };
}

interface SessionDetailProps {
  sessionId: string;
  onBack: () => void;
}

export default function SessionDetail({
  sessionId,
  onBack,
}: SessionDetailProps) {
  const { sessions, exportSession, setSessions } = useDashboardStore();
  const session = sessions.find((s) => s.id === sessionId);

  // People labeling: "Who is this?" on a speaker row — only for the caller's
  // OWN sessions (a therapist can't relabel a patient's recording) whose
  // server rows carry raw speaker ids (newer servers).
  const canLabel =
    !!session &&
    session.shared !== true &&
    typeof session.recordingId === "string" &&
    session.turns.some((t) => typeof t.speakerId === "string" && t.speakerId);
  const [people, setPeople] = useState<VoicePerson[]>([]);
  const [who, setWho] = useState<{ speaker: string; label: string; personId: string | null } | null>(null);
  const refreshPeople = useCallback(async () => {
    try {
      const res = await apiClient.listVoicePeople();
      setPeople(res.people);
    } catch {
      // No people list (older server / offline) — the sheet still offers
      // "New person…"; naming this recording never depends on it.
    }
  }, []);
  useEffect(() => {
    if (canLabel) void refreshPeople();
  }, [canLabel, refreshPeople]);

  const handleLabeled = useCallback(
    (result: PatchSpeakerLabelsResult) => {
      setSessions(
        useDashboardStore
          .getState()
          .sessions.map((s) => (s.id === sessionId ? applyLabelsToSession(s, result.speaker_labels) : s)),
      );
    },
    [sessionId, setSessions],
  );

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
                {canLabel && turn.speakerId ? (
                  <TouchableOpacity
                    testID={`turn-${i}-who`}
                    accessibilityRole="button"
                    accessibilityLabel={`Who is ${turn.speaker}?`}
                    style={styles.turnSpeakerRow}
                    onPress={() =>
                      setWho({
                        speaker: turn.speakerId as string,
                        label: turn.speaker,
                        personId: turn.personId ?? null,
                      })
                    }
                  >
                    <Text style={[styles.turnSpeaker, styles.turnSpeakerLink]}>
                      {turn.speaker}
                    </Text>
                    {isEnrolledPersonLabel(
                      {
                        display_label: turn.speaker,
                        label_source: turn.labelSource ?? "generic",
                        person_id: turn.personId ?? null,
                      },
                      people,
                    ) ? (
                      <View style={styles.enrolledBadge} testID={`turn-${i}-enrolled`}>
                        <Text style={styles.enrolledBadgeText}>enrolled</Text>
                      </View>
                    ) : (
                      <Text style={styles.whoHint}>who?</Text>
                    )}
                  </TouchableOpacity>
                ) : (
                  <Text style={styles.turnSpeaker}>{turn.speaker}</Text>
                )}
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

      {canLabel && who && session.recordingId ? (
        <WhoIsThisSheet
          visible
          recordingId={session.recordingId}
          speaker={who.speaker}
          currentLabel={who.label}
          currentPersonId={who.personId}
          people={people}
          hasAudio={session.hasAudio === true}
          onClose={() => setWho(null)}
          onLabeled={handleLabeled}
          onEnrolled={() => void refreshPeople()}
        />
      ) : null}
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
  turnSpeakerRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    flexShrink: 1,
  },
  turnSpeakerLink: {
    color: "#4A90D9",
  },
  whoHint: {
    fontSize: 11,
    color: "#9CA3AF",
    fontStyle: "italic",
  },
  enrolledBadge: {
    backgroundColor: "#ECFDF5",
    borderRadius: 6,
    paddingHorizontal: 6,
    paddingVertical: 1,
  },
  enrolledBadgeText: {
    fontSize: 10,
    fontWeight: "700",
    color: "#047857",
    textTransform: "uppercase",
    letterSpacing: 0.3,
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
