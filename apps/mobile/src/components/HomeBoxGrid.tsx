import React, { useState } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";

import {
  getDestination,
  type DestId,
  type DestScreen,
  type IconId,
} from "../nav/destinations";
import { getIcon } from "./icons";
import GrowthChart from "./GrowthChart";
import { useGrowthPreview } from "../hooks/useGrowthPreview";

interface HomeBoxGridProps {
  boxes: readonly DestId[];
  /** Hand a tapped box's destination straight to App.tsx's setScreen — the
   *  exact same callback AppChrome's tab bar and hamburger catalog use
   *  (Task N3's `handleNavigate`), so a box behaves identically to its tab
   *  or catalog counterpart. */
  onNavigate: (screen: DestScreen) => void;
}

/**
 * Task N4 of P3-10: the home screen's main area — up to 4 user-arranged
 * shortcut boxes (Settings → "Home screen design", Task N5), replacing the
 * old fixed two-huge-mode-cards layout. EVERY box is an icon + label (owner:
 * "bare colored boxes look horrible — users learn icons", P3-9 RESOLVED) —
 * never a bare color block.
 *
 * Layout scales with count, not a fixed grid — a single box deserves a full
 * banner (it IS the whole home area), two split a row, three or four wrap
 * into a 2-column grid (3 renders as 2 + 1, still grid-sized cards):
 *  - 0: no boxes at all — an explanatory hint, never a blank/broken-looking
 *    screen (P3-10 N4 scope: "honest, not empty-broken").
 *  - 1: one full-width banner card.
 *  - 2: two half-width cards, side by side.
 *  - 3-4: a wrapping 2-column grid.
 *
 * When a box's destination id isn't recognized (a stale persisted value from
 * a prior app version), it's silently dropped — the exact same fail-open
 * rule AppChrome's tab bar applies to `tabSlots`.
 */
export default function HomeBoxGrid({ boxes, onNavigate }: HomeBoxGridProps) {
  if (boxes.length === 0) {
    return (
      <View style={styles.empty} testID="home-boxes-empty">
        <Text style={styles.emptyText}>
          Add shortcuts in Settings → Home screen design
        </Text>
      </View>
    );
  }

  const sizeStyle =
    boxes.length === 1 ? styles.full : boxes.length === 2 ? styles.half : styles.quarter;

  return (
    <View style={styles.grid} testID="home-boxes-grid">
      {boxes.map((id) => {
        const dest = getDestination(id);
        if (!dest) return null; // stale persisted id — drop it silently
        return (
          <HomeBox
            key={id}
            id={id}
            title={dest.title}
            iconId={dest.iconId}
            large={boxes.length === 1}
            boxStyle={sizeStyle}
            onPress={() => onNavigate(dest.screen)}
          />
        );
      })}
    </View>
  );
}

function HomeBox({
  id,
  title,
  iconId,
  large,
  boxStyle,
  onPress,
}: {
  id: DestId;
  title: string;
  iconId: IconId;
  large: boolean;
  boxStyle: object;
  onPress: () => void;
}) {
  const Icon = getIcon(iconId);
  return (
    <TouchableOpacity
      testID={`home-box-${id}`}
      accessibilityRole="button"
      accessibilityLabel={title}
      style={[styles.box, boxStyle]}
      onPress={onPress}
      activeOpacity={0.85}
    >
      <Icon size={large ? 30 : 26} color="#4A90D9" />
      <Text style={[styles.boxTitle, large && styles.boxTitleLarge]} numberOfLines={2}>
        {title}
      </Text>
      {id === "growth" ? <GrowthBoxPreview /> : null}
    </TouchableOpacity>
  );
}

/**
 * The growth box's "cheaply feasible" mini trend preview (Task N4 scope
 * item 4): reuses `useGrowthPreview` (the same fetch-once, fail-open hook
 * GrowthStrip uses) and `GrowthChart` (the same sparkline renderer) — no new
 * fetch plumbing, just the existing growth data shown a second, smaller way.
 * Honest states: while loading, on fetch failure, or with zero identified
 * recordings, it renders nothing extra — the icon + title above is already
 * a complete, non-broken card on its own.
 */
function GrowthBoxPreview() {
  const { result } = useGrowthPreview();
  const [width, setWidth] = useState(0);

  if (!result || result.identified_recordings === 0) return null;

  return (
    <View
      style={styles.growthPreview}
      testID="home-box-growth-preview"
      onLayout={(e) => setWidth(e.nativeEvent.layout.width)}
    >
      <GrowthChart points={result.points} width={width} height={28} />
    </View>
  );
}

const styles = StyleSheet.create({
  grid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
  },
  full: {
    width: "100%",
    minHeight: 168,
  },
  half: {
    width: "48%",
    minHeight: 140,
  },
  quarter: {
    width: "48%",
    minHeight: 120,
  },
  box: {
    borderRadius: 20,
    backgroundColor: "#FFFFFF",
    borderWidth: 1.5,
    borderColor: "#D1D5DB",
    padding: 20,
    justifyContent: "flex-end",
    gap: 8,
  },
  boxTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: "#111827",
  },
  boxTitleLarge: {
    fontSize: 22,
  },
  growthPreview: {
    height: 28,
    marginTop: 2,
  },
  empty: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 32,
  },
  emptyText: {
    fontSize: 15,
    lineHeight: 22,
    color: "#6B7280",
    textAlign: "center",
  },
});
