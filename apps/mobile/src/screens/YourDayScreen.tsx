import React, { useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import { listRecordings, getRecordingEpisodes } from "../api/client";
import type { Episode, RecordingSummary } from "../api/client";
import type { DayEntry } from "./dayTimeline";
import {
  addDays,
  dateKey,
  dayTitle,
  episodeTimeRange,
  heatColor,
  participantsLine,
  recordingsForDay,
} from "./dayTimeline";
import { formatDateTime } from "../utils/dateDisplay";

// House colors.
const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";

interface YourDayScreenProps {
  /** Open the existing replay/detail view for a recording. */
  onOpenReplay: (recordingId: string) => void;
  onBack: () => void;
}

/** Honest message for list-level failures (same mapping as RecordingsScreen). */
function humanizeError(message: string): string {
  if (message.includes("503")) return "Replay storage isn’t enabled yet.";
  if (message.includes("401")) return "Please sign in again to see your day.";
  return message;
}

/**
 * "Your Day" — a vertical day timeline of the conversations you recorded
 * (Companion P1). Each recording's EPISODES (the transcript split on silence
 * gaps server-side) render as blocks down a time-of-day axis: a heat ribbon
 * whose color tracks the episode's mean heat, the participants, and the
 * derived one-line summary. Tapping an episode opens the existing replay for
 * its recording. ‹ / › steps through past days; nothing here fabricates data —
 * days without recordings say so plainly.
 */
export default function YourDayScreen({
  onOpenReplay,
  onBack,
}: YourDayScreenProps) {
  const [day, setDay] = useState<Date>(() => new Date());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [entries, setEntries] = useState<DayEntry[]>([]);
  // Bumped by "Try again" to force a full refetch.
  const [reloadNonce, setReloadNonce] = useState(0);

  const mountedRef = useRef(true);
  // The full recordings list, fetched once (cleared on retry) — stepping
  // between days re-buckets locally instead of re-listing.
  const recordingsRef = useRef<RecordingSummary[] | null>(null);
  // Per-recording episode cache so stepping between days never refetches a
  // recording it already resolved.
  const episodeCacheRef = useRef<Map<string, Episode[] | null>>(new Map());

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    let stale = false;
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        if (recordingsRef.current === null) {
          recordingsRef.current = await listRecordings();
        }
        const dayRecs = recordingsForDay(recordingsRef.current, day);
        const resolved: DayEntry[] = await Promise.all(
          dayRecs.map(async (rec) => {
            const cache = episodeCacheRef.current;
            if (!cache.has(rec.id)) {
              // A recording without an analysis has no episodes — skip the
              // fetch and render it honestly as not analyzed.
              cache.set(
                rec.id,
                rec.has_analysis ? await getRecordingEpisodes(rec.id) : null,
              );
            }
            return { recording: rec, episodes: cache.get(rec.id) ?? null };
          }),
        );
        if (mountedRef.current && !stale) setEntries(resolved);
      } catch (e) {
        if (mountedRef.current && !stale) {
          setError(
            humanizeError(
              e instanceof Error ? e.message : "Something went wrong.",
            ),
          );
        }
      } finally {
        if (mountedRef.current && !stale) setLoading(false);
      }
    };
    void run();
    return () => {
      stale = true;
    };
  }, [day, reloadNonce]);

  const todayKey = dateKey(new Date());
  const isToday = dateKey(day) === todayKey;

  return (
    <View style={styles.flex} testID="your-day-screen">
      <View style={styles.header}>
        <TouchableOpacity
          testID="your-day-back"
          onPress={onBack}
          hitSlop={{ top: 10, bottom: 10, left: 8, right: 16 }}
        >
          <Text style={styles.backText}>‹ Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Your Day</Text>
        <View style={styles.headerSpacer} />
      </View>

      {/* Simple date picker: step through past days; can't step into the
          future (there is nothing honest to show there). */}
      <View style={styles.dateRow}>
        <TouchableOpacity
          testID="your-day-prev"
          accessibilityRole="button"
          accessibilityLabel="Previous day"
          style={styles.dateArrow}
          onPress={() => setDay((d) => addDays(d, -1))}
        >
          <Text style={styles.dateArrowText}>‹</Text>
        </TouchableOpacity>
        <View style={styles.dateCenter}>
          <Text style={styles.dateTitle} testID="your-day-title">
            {dayTitle(day)}
          </Text>
          {!isToday && (
            <TouchableOpacity
              testID="your-day-today"
              onPress={() => setDay(new Date())}
            >
              <Text style={styles.todayLink}>Jump to today</Text>
            </TouchableOpacity>
          )}
        </View>
        <TouchableOpacity
          testID="your-day-next"
          accessibilityRole="button"
          accessibilityLabel="Next day"
          style={[styles.dateArrow, isToday && styles.dateArrowDisabled]}
          disabled={isToday}
          onPress={() => setDay((d) => addDays(d, 1))}
        >
          <Text
            style={[styles.dateArrowText, isToday && styles.dateArrowTextDisabled]}
          >
            ›
          </Text>
        </TouchableOpacity>
      </View>

      {loading && (
        <View style={styles.centered} testID="your-day-loading">
          <ActivityIndicator size="large" color={PRIMARY} />
        </View>
      )}

      {!loading && error && (
        <View style={styles.centered} testID="your-day-error">
          <Text style={styles.errorTitle}>Couldn’t load your day</Text>
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity
            testID="your-day-retry"
            style={styles.retryButton}
            onPress={() => {
              // Drop caches so retry truly refetches.
              episodeCacheRef.current.clear();
              recordingsRef.current = null;
              setReloadNonce((n) => n + 1);
            }}
          >
            <Text style={styles.retryText}>Try again</Text>
          </TouchableOpacity>
        </View>
      )}

      {!loading && !error && entries.length === 0 && (
        <View style={styles.centered} testID="your-day-empty">
          <Text style={styles.emptyText}>
            {isToday
              ? "No conversations recorded today."
              : "No conversations recorded on this day."}
          </Text>
        </View>
      )}

      {!loading && !error && entries.length > 0 && (
        <ScrollView
          style={styles.flex}
          contentContainerStyle={styles.content}
          testID="your-day-timeline"
        >
          {entries.map(({ recording, episodes }) => (
            <View key={recording.id} testID={`day-recording-${recording.id}`}>
              <Text style={styles.recordingHeader} numberOfLines={1}>
                {recording.title || recording.filename}
              </Text>
              {/* Full date + wall-clock start, so the day a row belongs to is
                  never in doubt. From the recording's real created_at; omitted
                  (never guessed) when that's missing. */}
              {formatDateTime(recording.created_at) && (
                <Text
                  style={styles.recordingWhen}
                  testID={`day-recording-${recording.id}-when`}
                >
                  {formatDateTime(recording.created_at)}
                </Text>
              )}

              {episodes === null && (
                <Text
                  style={styles.notAnalyzed}
                  testID={`day-recording-${recording.id}-unanalyzed`}
                >
                  Not analyzed — no episode data.
                </Text>
              )}

              {episodes !== null &&
                episodes.map((ep) => (
                  <View style={styles.episodeRow} key={ep.index}>
                    {/* Time-of-day axis: clock label + rail. */}
                    <View style={styles.axisCol}>
                      <Text style={styles.axisTime}>
                        {episodeTimeRange(recording.created_at, ep)}
                      </Text>
                      <View style={styles.axisRail} />
                    </View>

                    <TouchableOpacity
                      style={styles.episodeCard}
                      testID={`episode-${recording.id}-${ep.index}`}
                      accessibilityRole="button"
                      onPress={() => onOpenReplay(recording.id)}
                      activeOpacity={0.85}
                    >
                      {/* Heat ribbon: color intensity = mean heat; gray when
                          the stored analysis carried no heats. */}
                      <View
                        style={[
                          styles.heatRibbon,
                          { backgroundColor: heatColor(ep.mean_heat) },
                        ]}
                        testID={`episode-heat-${recording.id}-${ep.index}`}
                      />
                      <View style={styles.episodeBody}>
                        {participantsLine(ep) !== "" && (
                          <Text style={styles.participants} numberOfLines={1}>
                            {participantsLine(ep)}
                          </Text>
                        )}
                        {ep.summary !== null && (
                          <Text style={styles.summary} numberOfLines={2}>
                            {ep.summary_source === "excerpt"
                              ? `“${ep.summary}”`
                              : ep.summary}
                          </Text>
                        )}
                        <Text style={styles.episodeMeta}>
                          {ep.turn_count}{" "}
                          {ep.turn_count === 1 ? "turn" : "turns"}
                          {ep.peak_heat !== null
                            ? ` · peak heat ${ep.peak_heat}`
                            : " · heat unknown"}
                        </Text>
                      </View>
                      <Text style={styles.chevron}>›</Text>
                    </TouchableOpacity>
                  </View>
                ))}
            </View>
          ))}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: "#F9FAFB" },
  header: {
    flexDirection: "row",
    alignItems: "center",
    // App's SafeAreaView already applies the notch inset; this base pad matches
    // the hub screens (~20-24) instead of the old hardcoded 56 that double-padded
    // on notched devices.
    paddingTop: 24,
    paddingBottom: 12,
    paddingHorizontal: 16,
    backgroundColor: "#FFFFFF",
    borderBottomWidth: 1,
    borderBottomColor: "#E5E7EB",
  },
  backText: { fontSize: 16, color: PRIMARY, fontWeight: "600", width: 64 },
  headerTitle: {
    flex: 1,
    textAlign: "center",
    fontSize: 17,
    fontWeight: "700",
    color: INK,
  },
  headerSpacer: { width: 64 },
  dateRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingVertical: 10,
  },
  dateArrow: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: "#E5E7EB",
    backgroundColor: "#FFFFFF",
  },
  dateArrowDisabled: { opacity: 0.4 },
  dateArrowText: { fontSize: 22, color: PRIMARY, fontWeight: "700" },
  dateArrowTextDisabled: { color: MUTED },
  dateCenter: { alignItems: "center" },
  dateTitle: { fontSize: 17, fontWeight: "700", color: INK },
  todayLink: { fontSize: 12.5, color: PRIMARY, marginTop: 2 },
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  emptyText: { color: MUTED, fontSize: 15, textAlign: "center" },
  errorTitle: { fontSize: 18, fontWeight: "700", color: INK, marginBottom: 6 },
  errorText: {
    fontSize: 14,
    color: MUTED,
    textAlign: "center",
    marginBottom: 16,
  },
  retryButton: {
    backgroundColor: PRIMARY,
    paddingVertical: 12,
    paddingHorizontal: 28,
    borderRadius: 10,
  },
  retryText: { color: "#FFFFFF", fontSize: 15, fontWeight: "600" },
  content: { padding: 16, paddingBottom: 40 },
  recordingHeader: {
    fontSize: 13,
    fontWeight: "700",
    color: MUTED,
    textTransform: "uppercase",
    letterSpacing: 0.5,
    marginTop: 10,
    marginBottom: 6,
  },
  recordingWhen: {
    fontSize: 12.5,
    color: MUTED,
    fontWeight: "600",
    marginTop: -2,
    marginBottom: 6,
  },
  notAnalyzed: { fontSize: 13, color: MUTED, marginBottom: 8 },
  episodeRow: { flexDirection: "row", marginBottom: 10 },
  axisCol: { width: 92, alignItems: "flex-start", paddingTop: 4 },
  axisTime: { fontSize: 11.5, color: MUTED, fontWeight: "600" },
  axisRail: {
    flex: 1,
    width: 2,
    backgroundColor: "#E5E7EB",
    marginLeft: 6,
    marginTop: 4,
    borderRadius: 1,
  },
  episodeCard: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    overflow: "hidden",
  },
  heatRibbon: { width: 6, alignSelf: "stretch" },
  episodeBody: { flex: 1, paddingVertical: 10, paddingHorizontal: 12 },
  participants: { fontSize: 14.5, fontWeight: "700", color: INK },
  summary: { fontSize: 13, color: "#374151", marginTop: 2 },
  episodeMeta: { fontSize: 12, color: MUTED, marginTop: 4 },
  chevron: { fontSize: 20, color: MUTED, paddingHorizontal: 10 },
});
