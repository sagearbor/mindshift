import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  AppState,
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  RefreshControl,
  StyleSheet,
  ActivityIndicator,
  useWindowDimensions,
} from "react-native";

import {
  catchUpVoice,
  getGrowth,
  getVoiceProfile,
  type CatchUpResult,
  type GrowthResult,
} from "../api/client";
import GrowthChart from "../components/GrowthChart";
import ToneSparkline from "../components/ToneSparkline";
import {
  filterPoints,
  hasUnidentifiedPartner,
  partnerNames,
  scoredPoints,
  type PartnerFilter,
} from "../components/growthSeries";
import {
  bucketToneByDay,
  calmShare,
  dayLabel,
  describeBucket,
  peopleRows,
  toneChipColors,
  topLabels,
} from "./toneTrends";
import { catchUpErrorMessage } from "./catchUpError";

const PRIMARY = "#4A90D9";
const INK = "#111827";
const MUTED = "#6B7280";

interface GrowthScreenProps {
  /** Optional (Task N3): omitted when Growth is rendered as a primary
   *  screen inside AppChrome, which already provides a way back to Home —
   *  a dedicated back button here would just duplicate it. Still supported
   *  for any other caller (e.g. this screen's own unit tests). */
  onBack?: () => void;
  /** Tap a dot → that recording's replay/detail. */
  onOpenRecording: (recordingId: string) => void;
  /** Empty-state CTA → the recordings list, where "This is me" (the existing
   *  SpeakerEnrollment card on a recording's replay) starts voice tracking. */
  onOpenRecordings: () => void;
}

function filterKey(f: PartnerFilter): string {
  switch (f.kind) {
    case "all":
      return "growth-filter-all";
    case "partner":
      return `growth-filter-${f.name}`;
    case "unidentified":
      return "growth-filter-unidentified";
  }
}

function filterLabel(f: PartnerFilter): string {
  switch (f.kind) {
    case "all":
      return "All";
    case "partner":
      return `with ${f.name}`;
    case "unidentified":
      return "Unidentified partner";
  }
}

/**
 * The full "Your growth" chart: every stored recording where the user's own
 * voice was confidently identified, as a score-over-time dot chart with a
 * moving-average trend (≥5 scored points), filterable by conversation partner.
 *
 * Honesty rules, straight from the API:
 * * the footer always states "N of M recordings identified your voice" — the
 *   chart never pretends to cover conversations it can't attribute;
 * * identified recordings without a usable score are gaps, never zeros;
 * * partners are only named when a real cross-recording name exists (manual
 *   tag / transcript name); the rest live under "Unidentified partner".
 */
export default function GrowthScreen({
  onBack,
  onOpenRecording,
  onOpenRecordings,
}: GrowthScreenProps) {
  const [result, setResult] = useState<GrowthResult | null>(null);
  const [error, setError] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [filter, setFilter] = useState<PartnerFilter>({ kind: "all" });
  const { width: windowWidth } = useWindowDimensions();
  const [chartWidth, setChartWidth] = useState(windowWidth - 40);

  // "Catch up my past recordings" — offered only once we know the caller has
  // actually enrolled a voiceprint (GET /voice/profile). A failure here just
  // means the affordance stays hidden, same "never crash on a status check"
  // rule SpeakerEnrollment already follows.
  const [voiceEnrolled, setVoiceEnrolled] = useState(false);
  const [catchingUp, setCatchingUp] = useState(false);
  const [catchUpResult, setCatchUpResult] = useState<CatchUpResult | null>(null);
  const [catchUpError, setCatchUpError] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(false);
    setResult(null);
    getGrowth()
      .then(setResult)
      .catch(() => setError(true));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Pull-to-refresh keeps the current chart on screen while re-reading —
  // a live session's batch analysis lands a few seconds after it ends.
  const handleRefresh = useCallback(() => {
    setRefreshing(true);
    getGrowth()
      .then((r) => {
        setResult(r);
        setError(false);
      })
      .catch(() => setError(true))
      .finally(() => setRefreshing(false));
  }, []);

  useEffect(() => {
    let cancelled = false;
    getVoiceProfile()
      .then((p) => {
        if (!cancelled) setVoiceEnrolled(p.enrolled);
      })
      .catch(() => {
        if (!cancelled) setVoiceEnrolled(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Catch-up runs for a long time (the server re-embeds every speaker of up
  // to 25 recordings). If the user backgrounds the app meanwhile, the OS drops
  // the socket and the fetch rejects with a transport error — but the server
  // keeps going and persists matches per recording. So: remember that we
  // came back to the foreground with a catch-up in flight, and once the
  // request settles (either way) re-read growth so whatever DID get
  // identified shows up instead of a stale chart under a scary banner.
  const catchUpInFlight = useRef(false);
  const reloadAfterCatchUp = useRef(false);
  useEffect(() => {
    const sub = AppState.addEventListener("change", (state) => {
      if (state === "active" && catchUpInFlight.current) {
        reloadAfterCatchUp.current = true;
      }
    });
    return () => sub.remove();
  }, []);

  const handleCatchUp = useCallback(() => {
    setCatchingUp(true);
    setCatchUpError(null);
    catchUpInFlight.current = true;
    reloadAfterCatchUp.current = false;
    catchUpVoice()
      .then((res) => {
        setCatchUpResult(res);
        load(); // pull the newly identified points in immediately
      })
      .catch((err: unknown) => {
        setCatchUpError(catchUpErrorMessage(err));
        if (reloadAfterCatchUp.current) load(); // partial progress is real
      })
      .finally(() => {
        catchUpInFlight.current = false;
        reloadAfterCatchUp.current = false;
        setCatchingUp(false);
      });
  }, [load]);

  const filters = useMemo<PartnerFilter[]>(() => {
    if (!result) return [];
    const names = partnerNames(result.points);
    if (names.length === 0) return []; // nothing to filter by — no chip row
    const out: PartnerFilter[] = [{ kind: "all" }];
    for (const name of names) out.push({ kind: "partner", name });
    if (hasUnidentifiedPartner(result.points)) {
      out.push({ kind: "unidentified" });
    }
    return out;
  }, [result]);

  const visible = result ? filterPoints(result.points, filter) : [];

  // Track 2 — "How you sound": the user's OWN tone per local day, summed
  // over the live sessions that carried per-turn tone (respecting the
  // partner filter, so "with Mom" narrows the days too), plus the
  // cross-session per-person rows the server aggregated. Both are empty —
  // and the section hidden — when no live session has tone yet.
  const toneDays = useMemo(() => bucketToneByDay(visible), [visible]);
  const people = useMemo(() => peopleRows(result?.people), [result]);
  const calmSeries = toneDays
    .map((d) => calmShare(d.scored_turns, d.escalation_count))
    .filter((v): v is number => typeof v === "number");

  const renderBody = () => {
    if (error) {
      return (
        <View style={styles.centerBox} testID="growth-error">
          <Text style={styles.emptyTitle}>Couldn’t load your growth</Text>
          <Text style={styles.emptyBody}>
            Something went wrong talking to the server.
          </Text>
          <TouchableOpacity
            testID="growth-retry"
            accessibilityRole="button"
            style={styles.ctaButton}
            onPress={load}
          >
            <Text style={styles.ctaText}>Try again</Text>
          </TouchableOpacity>
        </View>
      );
    }
    if (result === null) {
      return (
        <View style={styles.centerBox} testID="growth-loading">
          <ActivityIndicator size="large" color={PRIMARY} />
        </View>
      );
    }
    // Offered whenever there's at least one enrolled-but-not-yet-identified
    // recording left — NOT gated to the empty state. Part A's first "This is
    // me" tap flips identified_recordings from 0 to 1 immediately, which
    // would otherwise permanently strand every other unidentified recording
    // behind a button that only ever existed in the now-gone empty state.
    const canCatchUp =
      voiceEnrolled && result.identified_recordings < result.total_recordings;

    const catchUpButton = (
      <TouchableOpacity
        testID="growth-catchup-cta"
        accessibilityRole="button"
        style={[styles.ctaButton, styles.catchUpButton]}
        disabled={catchingUp}
        onPress={handleCatchUp}
      >
        {catchingUp ? (
          <ActivityIndicator
            size="small"
            color="#FFFFFF"
            testID="growth-catchup-pending"
          />
        ) : (
          <Text style={styles.ctaText}>Catch up my past recordings</Text>
        )}
      </TouchableOpacity>
    );

    if (result.identified_recordings === 0) {
      return (
        <View style={styles.centerBox} testID="growth-empty">
          <Text style={styles.emptyTitle}>No growth data yet</Text>
          <Text style={styles.emptyBody}>
            {result.total_recordings === 0
              ? "Analyze and store a conversation first — then teach MindShift " +
                "your voice by tapping “This is me” on your speaker."
              : `You have ${result.total_recordings} stored recording` +
                `${result.total_recordings === 1 ? "" : "s"}, but none has ` +
                "identified your voice yet. Tap “This is me” on a recording " +
                "you're confident about" +
                (canCatchUp
                  ? ", or use “Catch up my past recordings” below to " +
                    "auto-match your enrolled voice against everything " +
                    "you’ve already stored."
                  : " to start tracking your scores.")}
          </Text>
          {canCatchUp ? catchUpButton : null}
          <TouchableOpacity
            testID="growth-enroll-cta"
            accessibilityRole="button"
            style={styles.ctaButton}
            onPress={onOpenRecordings}
          >
            <Text style={styles.ctaText}>Open past recordings</Text>
          </TouchableOpacity>
        </View>
      );
    }
    return (
      <>
        {filters.length > 0 ? (
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            style={styles.chipRow}
            contentContainerStyle={styles.chipRowContent}
          >
            {filters.map((f) => {
              const key = filterKey(f);
              const active = filterKey(filter) === key;
              return (
                <TouchableOpacity
                  key={key}
                  testID={key}
                  accessibilityRole="button"
                  style={[styles.chip, active ? styles.chipActive : null]}
                  onPress={() => setFilter(f)}
                >
                  <Text
                    style={[
                      styles.chipText,
                      active ? styles.chipTextActive : null,
                    ]}
                  >
                    {filterLabel(f)}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </ScrollView>
        ) : null}

        <View
          style={styles.chartCard}
          onLayout={(e) =>
            setChartWidth(Math.max(0, e.nativeEvent.layout.width - 24))
          }
        >
          {scoredPoints(visible).length === 0 ? (
            <Text style={styles.filterEmpty} testID="growth-filter-empty">
              No scored recordings match this filter.
            </Text>
          ) : (
            <GrowthChart
              points={visible}
              width={chartWidth}
              height={220}
              dotRadius={4}
              axes
              onPressPoint={(p) => onOpenRecording(p.recording_id)}
            />
          )}
          <Text style={styles.axisHint} testID="growth-axis-hint">
            tap a dot to open it
          </Text>
        </View>

        <Text style={styles.footer} testID="growth-footer">
          {`${result.identified_recordings} of ${result.total_recordings} ` +
            `recording${result.total_recordings === 1 ? "" : "s"} identified ` +
            "your voice"}
        </Text>

        {(toneDays.length > 0 || people.length > 0) && (
          <View style={styles.toneCard} testID="growth-tone-section">
            <Text style={styles.toneTitle}>How you sound</Text>
            <Text style={styles.toneHint}>
              Your own tone in live sessions, by day — from the words you said.
            </Text>
            {calmSeries.length > 1 && (
              <View style={styles.toneSpark}>
                <ToneSparkline
                  scores={calmSeries}
                  width={chartWidth}
                  height={44}
                  color="#1B7A4B"
                />
                <Text style={styles.toneSparkCaption}>
                  share of your turns that stayed calm, day by day
                </Text>
              </View>
            )}
            {toneDays.map((day) => {
              const chips = topLabels(day.labels, 3);
              const line = describeBucket(
                day.labels,
                day.escalation_count,
                day.scored_turns,
              );
              return (
                <View
                  key={day.key}
                  style={styles.toneDayRow}
                  testID={`growth-tone-day-${day.key}`}
                >
                  <Text style={styles.toneDayLabel}>{dayLabel(day)}</Text>
                  <View style={styles.toneDayBody}>
                    <View style={styles.toneChipRow}>
                      {chips.map((c) => {
                        const colors = toneChipColors(c.label);
                        return (
                          <View
                            key={c.label}
                            style={[styles.toneChip, { backgroundColor: colors.bg }]}
                          >
                            <Text style={[styles.toneChipText, { color: colors.fg }]}>
                              {c.label} ×{c.count}
                            </Text>
                          </View>
                        );
                      })}
                    </View>
                    <Text style={styles.toneDayLine} numberOfLines={1}>
                      {line}
                      {day.sessions > 1 ? ` · ${day.sessions} sessions` : ""}
                    </Text>
                  </View>
                </View>
              );
            })}
            {people.length > 0 && (
              <View style={styles.tonePeople}>
                <Text style={styles.tonePeopleTitle}>With people</Text>
                {people.map((p) => (
                  <View
                    key={p.person_id ?? p.name}
                    style={styles.tonePersonRow}
                    testID={`growth-tone-person-${p.person_id ?? p.name}`}
                  >
                    <Text style={styles.tonePersonName} numberOfLines={1}>
                      with {p.name}
                    </Text>
                    <Text style={styles.tonePersonLine} numberOfLines={1}>
                      {p.summary}
                      {` · ${p.sessions} session${p.sessions === 1 ? "" : "s"}`}
                    </Text>
                  </View>
                ))}
              </View>
            )}
          </View>
        )}
        {/* Stays reachable after the first identification — see canCatchUp's
         *  comment: the empty state (where this button used to live
         *  exclusively) is gone the moment even one recording is identified,
         *  but there can still be plenty left to catch up. */}
        {canCatchUp ? (
          <View style={styles.footerCatchUp}>{catchUpButton}</View>
        ) : null}
      </>
    );
  };

  return (
    <ScrollView
      style={styles.flex}
      contentContainerStyle={styles.content}
      testID="growth-screen"
      refreshControl={
        <RefreshControl
          testID="growth-refresh"
          refreshing={refreshing}
          onRefresh={handleRefresh}
        />
      }
    >
      {onBack && (
        <TouchableOpacity
          testID="growth-back"
          accessibilityRole="button"
          style={styles.backButton}
          onPress={onBack}
        >
          <Text style={styles.backText}>← Back</Text>
        </TouchableOpacity>
      )}
      <Text style={styles.heading}>Your growth</Text>
      {/* Rendered OUTSIDE the empty/chart branches on purpose: a successful
       *  catch-up refetches growth immediately, which can flip the screen
       *  straight from the empty state to the chart — the result banner must
       *  survive that transition instead of vanishing with the empty state. */}
      {catchUpResult ? (
        <Text style={styles.catchUpResult} testID="growth-catchup-result">
          {catchUpResult.newly_identified > 0
            ? `Found you in ${catchUpResult.newly_identified} of ` +
              `${catchUpResult.checked} recordings`
            : "No match found in your past recordings — try “This is me” " +
              "on one you’re sure about"}
        </Text>
      ) : null}
      {catchUpError ? (
        <Text style={styles.error} testID="growth-catchup-error">
          {catchUpError}
        </Text>
      ) : null}
      {renderBody()}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  flex: {
    flex: 1,
  },
  content: {
    paddingTop: 24,
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  backButton: {
    alignSelf: "flex-start",
    minHeight: 44,
    justifyContent: "center",
    paddingRight: 12,
    marginBottom: 4,
  },
  backText: {
    fontSize: 16,
    fontWeight: "600",
    color: PRIMARY,
  },
  heading: {
    fontSize: 24,
    fontWeight: "700",
    color: INK,
    marginBottom: 16,
  },
  centerBox: {
    alignItems: "center",
    paddingVertical: 48,
    paddingHorizontal: 12,
  },
  emptyTitle: {
    fontSize: 17,
    fontWeight: "700",
    color: INK,
    marginBottom: 8,
  },
  emptyBody: {
    fontSize: 14,
    lineHeight: 20,
    color: MUTED,
    textAlign: "center",
    marginBottom: 20,
  },
  ctaButton: {
    minHeight: 48,
    paddingHorizontal: 24,
    borderRadius: 12,
    backgroundColor: PRIMARY,
    alignItems: "center",
    justifyContent: "center",
  },
  ctaText: {
    fontSize: 15,
    fontWeight: "700",
    color: "#FFFFFF",
  },
  catchUpButton: {
    marginBottom: 12,
    backgroundColor: "#0F9D58",
  },
  catchUpResult: {
    fontSize: 13.5,
    lineHeight: 19,
    color: "#0F9D58",
    textAlign: "center",
    marginBottom: 12,
  },
  error: {
    fontSize: 13.5,
    lineHeight: 19,
    color: "#DC2626",
    textAlign: "center",
    marginBottom: 12,
  },
  chipRow: {
    marginBottom: 12,
    flexGrow: 0,
  },
  chipRowContent: {
    gap: 8,
  },
  chip: {
    minHeight: 36,
    paddingHorizontal: 14,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#FFFFFF",
    alignItems: "center",
    justifyContent: "center",
  },
  chipActive: {
    borderColor: PRIMARY,
    backgroundColor: "#EAF2FB",
  },
  chipText: {
    fontSize: 13.5,
    fontWeight: "600",
    color: MUTED,
  },
  chipTextActive: {
    color: PRIMARY,
  },
  chartCard: {
    borderWidth: 1,
    borderColor: "#E5E7EB",
    borderRadius: 14,
    backgroundColor: "#FFFFFF",
    padding: 12,
  },
  filterEmpty: {
    fontSize: 14,
    color: MUTED,
    textAlign: "center",
    paddingVertical: 48,
  },
  axisHint: {
    marginTop: 6,
    fontSize: 11,
    color: "#9CA3AF",
    textAlign: "right",
  },
  footer: {
    marginTop: 12,
    fontSize: 13,
    color: MUTED,
    textAlign: "center",
  },
  footerCatchUp: {
    marginTop: 16,
    alignItems: "center",
  },
  toneCard: {
    marginTop: 20,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    borderRadius: 14,
    backgroundColor: "#FFFFFF",
    padding: 12,
  },
  toneTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: INK,
  },
  toneHint: {
    fontSize: 12.5,
    color: MUTED,
    marginTop: 2,
    marginBottom: 10,
  },
  toneSpark: {
    marginBottom: 10,
  },
  toneSparkCaption: {
    fontSize: 11,
    color: "#9CA3AF",
    marginTop: 2,
  },
  toneDayRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 10,
    paddingVertical: 6,
    borderTopWidth: 1,
    borderTopColor: "#F3F4F6",
  },
  toneDayLabel: {
    width: 56,
    fontSize: 12.5,
    fontWeight: "700",
    color: MUTED,
    paddingTop: 3,
  },
  toneDayBody: {
    flex: 1,
  },
  toneChipRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
  },
  toneChip: {
    borderRadius: 12,
    paddingVertical: 3,
    paddingHorizontal: 10,
  },
  toneChipText: {
    fontSize: 12,
    fontWeight: "600",
  },
  toneDayLine: {
    fontSize: 12.5,
    color: INK,
    marginTop: 4,
  },
  tonePeople: {
    marginTop: 10,
    borderTopWidth: 1,
    borderTopColor: "#F3F4F6",
    paddingTop: 8,
  },
  tonePeopleTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: MUTED,
    marginBottom: 6,
  },
  tonePersonRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 8,
    paddingVertical: 4,
  },
  tonePersonName: {
    fontSize: 13.5,
    fontWeight: "600",
    color: INK,
    flexShrink: 1,
  },
  tonePersonLine: {
    fontSize: 12.5,
    color: MUTED,
    flexShrink: 1,
    textAlign: "right",
  },
});
