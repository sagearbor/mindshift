import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  useWindowDimensions,
} from "react-native";

import { getGrowth, type GrowthResult } from "../api/client";
import GrowthChart from "../components/GrowthChart";
import {
  filterPoints,
  hasUnidentifiedPartner,
  partnerNames,
  scoredPoints,
  type PartnerFilter,
} from "../components/growthSeries";

const PRIMARY = "#4A90D9";
const INK = "#111827";
const MUTED = "#6B7280";

interface GrowthScreenProps {
  onBack: () => void;
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
  const [filter, setFilter] = useState<PartnerFilter>({ kind: "all" });
  const { width: windowWidth } = useWindowDimensions();
  const [chartWidth, setChartWidth] = useState(windowWidth - 40);

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
                "identified your voice yet. Open one and tap “This is me” on " +
                "your speaker to start tracking your scores."}
          </Text>
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
              onPressPoint={(p) => onOpenRecording(p.recording_id)}
            />
          )}
          <View style={styles.axisRow}>
            <Text style={styles.axisText}>older</Text>
            <Text style={styles.axisText}>score 0–100 · tap a dot to open it</Text>
            <Text style={styles.axisText}>newer</Text>
          </View>
        </View>

        <Text style={styles.footer} testID="growth-footer">
          {`${result.identified_recordings} of ${result.total_recordings} ` +
            `recording${result.total_recordings === 1 ? "" : "s"} identified ` +
            "your voice"}
        </Text>
      </>
    );
  };

  return (
    <ScrollView
      style={styles.flex}
      contentContainerStyle={styles.content}
      testID="growth-screen"
    >
      <TouchableOpacity
        testID="growth-back"
        accessibilityRole="button"
        style={styles.backButton}
        onPress={onBack}
      >
        <Text style={styles.backText}>← Back</Text>
      </TouchableOpacity>
      <Text style={styles.heading}>Your growth</Text>
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
  axisRow: {
    marginTop: 6,
    flexDirection: "row",
    justifyContent: "space-between",
  },
  axisText: {
    fontSize: 11,
    color: "#9CA3AF",
  },
  footer: {
    marginTop: 12,
    fontSize: 13,
    color: MUTED,
    textAlign: "center",
  },
});
