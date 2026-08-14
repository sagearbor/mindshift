import React, { useEffect, useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

import { getGrowth, type GrowthResult } from "../api/client";
import GrowthChart from "./GrowthChart";

/**
 * The narrow "Your growth" strip on the home screen — a glanceable sparkline
 * of the user's own per-recording scores over time. Tapping it opens the full
 * GrowthScreen.
 *
 * Self-fetching (like AdvancedScreen's voice check) so HomeScreen stays a dumb
 * two-button surface. Honest states:
 * * while loading, or when the fetch fails (signed out, storage disabled,
 *   pre-growth server), it renders NOTHING — home never shows a broken chart;
 * * loaded but no identified recordings → a quiet "not tracked yet" row that
 *   still opens the screen (which explains how to enroll).
 */
export default function GrowthStrip({ onPress }: { onPress: () => void }) {
  const [result, setResult] = useState<GrowthResult | null>(null);
  const [chartWidth, setChartWidth] = useState(0);

  useEffect(() => {
    let cancelled = false;
    getGrowth()
      .then((r) => {
        if (!cancelled) setResult(r);
      })
      .catch(() => {
        // No strip is better than a wrong strip — the full screen has the
        // honest error surface.
        if (!cancelled) setResult(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (result === null) return null;

  const tracked = result.identified_recordings > 0;
  return (
    <TouchableOpacity
      testID="growth-strip"
      accessibilityRole="button"
      accessibilityLabel="Your growth"
      style={styles.strip}
      onPress={onPress}
      activeOpacity={0.85}
    >
      <View style={styles.labelColumn}>
        <Text style={styles.title}>Your growth</Text>
        <Text style={styles.sub} testID="growth-strip-sub">
          {tracked
            ? `${result.identified_recordings} of ${result.total_recordings} ` +
              `recording${result.total_recordings === 1 ? "" : "s"}`
            : "not tracked yet"}
        </Text>
      </View>
      {tracked ? (
        <View
          style={styles.chartArea}
          onLayout={(e) => setChartWidth(e.nativeEvent.layout.width)}
        >
          <GrowthChart
            points={result.points}
            width={chartWidth}
            height={40}
          />
        </View>
      ) : (
        <Text style={styles.chevron}>›</Text>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  strip: {
    marginTop: 16,
    minHeight: 56,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 16,
    paddingVertical: 8,
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  labelColumn: {
    flexShrink: 0,
  },
  title: {
    fontSize: 14,
    fontWeight: "700",
    color: "#1F2937",
  },
  sub: {
    marginTop: 2,
    fontSize: 12,
    color: "#6B7280",
  },
  chartArea: {
    flex: 1,
    height: 40,
  },
  chevron: {
    flex: 1,
    textAlign: "right",
    fontSize: 20,
    color: "#9CA3AF",
  },
});
