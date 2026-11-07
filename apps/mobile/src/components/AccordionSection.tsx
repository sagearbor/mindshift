/**
 * Collapsible Settings section. Children stay MOUNTED while collapsed (hidden
 * with display:none) so in-progress state (delete-account confirmation text,
 * voice-sample lists, dev panels) is never lost by closing a section, and so
 * every inner testID still exists. Open/closed per section is remembered per
 * user (see utils/accordionPrefs.ts) via useAccordionState.
 */
import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  LayoutChangeEvent,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { loadOpenSections, saveOpenSections, type OpenSections } from "../utils/accordionPrefs";

export function useAccordionState(userId: string | null | undefined) {
  const [open, setOpen] = useState<OpenSections>({});
  const touched = useRef<OpenSections>({});
  useEffect(() => {
    let live = true;
    loadOpenSections(userId).then((saved) => {
      // Anything the user toggled before the load finished wins over disk.
      if (live) setOpen({ ...saved, ...touched.current });
    });
    return () => {
      live = false;
    };
  }, [userId]);
  const toggle = useCallback(
    (id: string) => {
      setOpen((prev) => {
        const next = { ...prev, [id]: !prev[id] };
        touched.current = { ...touched.current, [id]: next[id] };
        void saveOpenSections(userId, next);
        return next;
      });
    },
    [userId],
  );
  return { open, toggle };
}

export interface AccordionSectionProps {
  /** testID of the title Text (kept equal to the old section heading's testID); the pressable is `<testID>-toggle`. */
  sectionId: string;
  title: string;
  subtitle?: string;
  expanded: boolean;
  onToggle: () => void;
  onLayout?: (e: LayoutChangeEvent) => void;
  children: React.ReactNode;
}

export default function AccordionSection({
  sectionId,
  title,
  subtitle,
  expanded,
  onToggle,
  onLayout,
  children,
}: AccordionSectionProps) {
  return (
    <View style={styles.wrap} onLayout={onLayout}>
      <TouchableOpacity
        testID={`${sectionId}-toggle`}
        accessibilityRole="button"
        accessibilityLabel={title}
        accessibilityState={{ expanded }}
        style={styles.header}
        onPress={onToggle}
      >
        <View style={styles.headerText}>
          <Text style={styles.title} testID={sectionId}>
            {title}
          </Text>
          {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
        </View>
        <Text style={styles.chevron}>{expanded ? "▾" : "▸"}</Text>
      </TouchableOpacity>
      <View testID={`${sectionId}-body`} style={expanded ? styles.body : styles.hidden}>
        {children}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    marginTop: 8,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "#E5E7EB",
  },
  header: {
    minHeight: 48,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 8,
  },
  headerText: { flex: 1, paddingRight: 8 },
  title: {
    fontSize: 13,
    fontWeight: "700",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    color: "#6B7280",
  },
  subtitle: { fontSize: 12, color: "#9CA3AF", marginTop: 2 },
  chevron: { fontSize: 16, color: "#6B7280" },
  body: { paddingBottom: 4 },
  hidden: { display: "none" },
});
