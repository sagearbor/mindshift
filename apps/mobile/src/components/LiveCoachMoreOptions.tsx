/**
 * The Live Coach screen's simple-by-default layout: the default view carries
 * only Start/Stop, status, nudges and (while running) Mark this moment; every
 * other control lives under one collapsed "More options" expander at the
 * bottom. The body stays MOUNTED while collapsed (display: none) so state,
 * refs and effects inside it behave exactly as before this was a re-layout.
 * Open/closed is remembered for the session only (module variable), never
 * persisted.
 *
 * {@link DevOptionsBanner} is the other half: a one-line notice in the default
 * view whenever a developer option that records or listens is on, so nothing
 * runs invisibly with the options folded away.
 */
import React, { useEffect, useState } from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import type { DevStatus } from "../emotion/devHub";
import { emotionDevHub } from "../emotion/hub";

// Session-only memory (resets when the app restarts).
let sessionOpen = false;

export function LiveCoachMoreOptions({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(sessionOpen);
  const toggle = () => {
    sessionOpen = !open;
    setOpen(!open);
  };
  return (
    <View style={styles.wrap}>
      <TouchableOpacity
        testID="live-more-options-toggle"
        accessibilityRole="button"
        accessibilityState={{ expanded: open }}
        style={styles.header}
        onPress={toggle}
      >
        <Text style={styles.title}>{open ? "More options ▾" : "More options ▸"}</Text>
      </TouchableOpacity>
      <View style={open ? undefined : styles.hidden} testID="live-more-options-body">
        {children}
      </View>
    </View>
  );
}

/** A short plain heading inside More options. */
export function MoreOptionsGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <View style={styles.group}>
      <Text style={styles.groupTitle} accessibilityRole="header">
        {title}
      </Text>
      {children}
    </View>
  );
}

/** Names of the developer options currently ON (empty = none). */
export function activeDevOptions(status: DevStatus): string[] {
  const out: string[] = [];
  if (status.state === "off") return out;
  if (emotionDevHub.userSettings.saveAudio) out.push("Save audio");
  out.push("Emotion tester");
  return out;
}

export function DevOptionsBanner() {
  const [status, setStatus] = useState<DevStatus>(() => emotionDevHub.status());
  useEffect(() => emotionDevHub.subscribe(setStatus), []);
  const active = activeDevOptions(status);
  if (active.length === 0) return null;
  return (
    <View style={styles.banner} testID="live-dev-banner" accessibilityLiveRegion="polite">
      <Text style={styles.bannerText}>Developer options on: {active.join(", ")}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginTop: 8 },
  header: {
    minHeight: 48,
    justifyContent: "center",
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  title: { fontSize: 15, fontWeight: "700", color: "#374151" },
  hidden: { display: "none" },
  group: { marginTop: 4 },
  groupTitle: {
    paddingHorizontal: 16,
    paddingTop: 10,
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 0.5,
    textTransform: "uppercase",
    color: "#6B7280",
  },
  banner: {
    marginHorizontal: 16,
    marginBottom: 8,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: "#FEF3C7",
  },
  bannerText: { fontSize: 13, fontWeight: "600", color: "#92400E" },
});
