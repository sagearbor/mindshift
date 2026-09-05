import React, { useCallback, useState } from "react";
import { View, Text, TouchableOpacity, ActivityIndicator, StyleSheet } from "react-native";
import type { SessionSummary } from "../live/sessionSummary";
import { formatDuration, formatLatency } from "../live/sessionSummary";
import { CANDOR_MEDIAN_GAP_S } from "../live/conversationDynamics";
import type { LastEpisode } from "../hooks/useAudioStream";
import type { TherapistLink } from "../api/therapist";
import { postShare } from "../api/client";
import { useDevModeStore } from "../store/devModeStore";

interface Props {
  summary: SessionSummary;
  /** The server's record of this session (null on the legacy path). */
  episode: LastEpisode | null;
  /** The patient's therapist link (null while unknown / not linked). */
  therapist: TherapistLink | null;
  /** Injected for tests; defaults to the real per-episode share call. */
  share?: (episodeId: string, email: string) => Promise<unknown>;
}

/** "0.4 s" / "—" — dynamics gaps/overlap are always seconds, unlike the ms
 *  latency stat above. */
function formatSeconds(s: number | null): string {
  if (s === null) return "—";
  return `${s.toFixed(s < 10 ? 1 : 0)} s`;
}

function humanizeShareError(err: unknown): string {
  const e = err as { detail?: string; status?: number };
  if (typeof e?.detail === "string" && e.detail) return e.detail;
  if (e?.status === 401) return "Please sign in again to share.";
  return "Couldn’t share right now — please try again.";
}

/**
 * The end-of-session card: duration, turns per person, escalations, and the
 * measured first-words latency (median / best — from the fast loop's per-
 * turn log; "—" when the phone never spoke), plus "Share with my therapist"
 * when a therapist is linked and the episode wasn't auto-shared already.
 */
export default function SessionSummaryCard({ summary, episode, therapist, share }: Props) {
  const [sharing, setSharing] = useState(false);
  const [sharedTo, setSharedTo] = useState<string | null>(null);
  const [shareError, setShareError] = useState<string | null>(null);

  const email = therapist?.linked ? therapist.therapist_email ?? null : null;
  const episodeId = episode?.postStatus === "created" ? episode.episodeId : null;
  const autoShared =
    Boolean(email) && (episode?.sharedWith ?? []).some((e) => e.toLowerCase() === email!.toLowerCase());

  const handleShare = useCallback(async () => {
    if (!episodeId || !email || sharing) return;
    setSharing(true);
    setShareError(null);
    try {
      await (share ?? postShare)(episodeId, email);
      setSharedTo(email);
    } catch (e) {
      setShareError(humanizeShareError(e));
    } finally {
      setSharing(false);
    }
  }, [episodeId, email, sharing, share]);

  // Developer mode off: no latency stat, no provider tag — a tester reads
  // duration/turns/escalations and the share button, nothing else.
  const devMode = useDevModeStore((s) => s.devMode);
  return (
    <View style={styles.card} testID="session-summary">
      <Text style={styles.title}>Session summary</Text>
      <View style={styles.grid}>
        <Stat testID="summary-duration" label="Duration" value={formatDuration(summary.durationMs)} />
        <Stat testID="summary-turns" label="Turns" value={String(summary.totalTurns)} />
        <Stat
          testID="summary-escalations"
          label="Escalations"
          value={String(summary.escalations)}
          warn={summary.escalations > 0}
        />
        {devMode ? (
          <Stat
            testID="summary-latency"
            label="First words"
            value={formatLatency(summary.firstWordsMedianMs)}
            sub={
              summary.firstWordsMedianMs === null
                ? "nothing spoken"
                : `best ${formatLatency(summary.firstWordsBestMs)} · ${summary.spokenTurns} spoken`
            }
          />
        ) : null}
      </View>
      {summary.turnsBySpeaker.length > 0 ? (
        <Text style={styles.people} testID="summary-people">
          {summary.turnsBySpeaker.map((s) => `${s.speaker}: ${s.turns}`).join(" · ")}
          {devMode && summary.topProvider ? ` · via ${summary.topProvider}` : ""}
        </Text>
      ) : null}

      {devMode && summary.dynamics ? (
        <View style={styles.dynamics} testID="summary-dynamics">
          <Text style={styles.dynamicsTitle}>Dynamics (dev)</Text>
          <Text style={styles.dynamicsLine} testID="summary-dynamics-gap">
            Your response gap: median {formatSeconds(summary.dynamics.selfResponseGaps.medianS)}
            {" "}(CANDOR norm {CANDOR_MEDIAN_GAP_S.toFixed(2)} s)
          </Text>
          <Text style={styles.dynamicsLine} testID="summary-dynamics-slow">
            Slow responses (&gt;2s): {summary.dynamics.selfResponseGaps.slowCount}
          </Text>
          <Text style={styles.dynamicsLine} testID="summary-dynamics-overlap">
            Overlap: {formatSeconds(summary.dynamics.overlapSecondsTotal)} total ·{" "}
            {summary.dynamics.sustainedOverlapCountOver1s} sustained episode
            {summary.dynamics.sustainedOverlapCountOver1s === 1 ? "" : "s"}
          </Text>
        </View>
      ) : null}

      {episode?.postStatus === "failed" ? (
        <Text style={styles.note} testID="summary-post-failed">
          Couldn’t save this session to your account — the transcript is still here.
        </Text>
      ) : null}

      {email && episodeId ? (
        autoShared || sharedTo ? (
          <Text style={styles.shared} testID="summary-shared">
            ✓ Shared with {sharedTo ?? email}
            {autoShared && !sharedTo ? " automatically" : ""}
          </Text>
        ) : (
          <TouchableOpacity
            testID="summary-share-therapist"
            accessibilityRole="button"
            style={styles.shareButton}
            onPress={handleShare}
            disabled={sharing}
          >
            {sharing ? (
              <ActivityIndicator size="small" color="#4A90D9" />
            ) : (
              <Text style={styles.shareText}>Share with my therapist ({email})</Text>
            )}
          </TouchableOpacity>
        )
      ) : null}
      {shareError ? (
        <Text style={styles.error} testID="summary-share-error">
          {shareError}
        </Text>
      ) : null}
    </View>
  );
}

function Stat({
  testID,
  label,
  value,
  sub,
  warn,
}: {
  testID: string;
  label: string;
  value: string;
  sub?: string;
  warn?: boolean;
}) {
  return (
    <View style={styles.stat} testID={testID}>
      <Text style={[styles.statValue, warn && styles.statWarn]}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
      {sub ? <Text style={styles.statSub}>{sub}</Text> : null}
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
    marginVertical: 8,
    padding: 14,
    gap: 8,
  },
  title: {
    fontSize: 16,
    fontWeight: "700",
    color: "#111827",
  },
  grid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  stat: {
    minWidth: 92,
    flexGrow: 1,
    backgroundColor: "#F9FAFB",
    borderRadius: 10,
    padding: 10,
    alignItems: "center",
  },
  statValue: {
    fontSize: 20,
    fontWeight: "700",
    color: "#4A90D9",
  },
  statWarn: {
    color: "#B45309",
  },
  statLabel: {
    fontSize: 11,
    color: "#6B7280",
    marginTop: 2,
  },
  statSub: {
    fontSize: 10.5,
    color: "#9CA3AF",
    marginTop: 2,
    textAlign: "center",
  },
  people: {
    fontSize: 12.5,
    color: "#374151",
  },
  dynamics: {
    backgroundColor: "#F9FAFB",
    borderRadius: 10,
    padding: 10,
    gap: 2,
  },
  dynamicsTitle: {
    fontSize: 11,
    fontWeight: "700",
    color: "#6B7280",
    marginBottom: 2,
  },
  dynamicsLine: {
    fontSize: 12,
    color: "#374151",
  },
  note: {
    fontSize: 12.5,
    color: "#B45309",
  },
  shared: {
    fontSize: 13.5,
    fontWeight: "600",
    color: "#15803D",
  },
  shareButton: {
    backgroundColor: "#EFF6FF",
    borderWidth: 1,
    borderColor: "#4A90D9",
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: "center",
  },
  shareText: {
    color: "#4A90D9",
    fontSize: 14.5,
    fontWeight: "700",
  },
  error: {
    fontSize: 12.5,
    color: "#DC2626",
  },
});
