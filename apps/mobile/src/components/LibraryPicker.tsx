import React, { useCallback, useEffect, useRef, useState } from "react";
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { listLibrary, RETRIEVAL_UNAVAILABLE_TEXT, type LibraryList } from "../api/library";
import { MAX_LIBRARY_SELECTION } from "../live/librarySelection";


/**
 * "Use from library" on Live Coach, next to the session note: tick which
 * saved notes/documents the coach may draw on this session (ids only reach
 * the hook, which sends `library_item_ids`). Collapsed by default to one
 * row saying how many are picked. Once the list loads, ids that no longer
 * exist (deleted elsewhere) are dropped from the selection.
 */
export interface LibraryPickerProps {
  selectedIds: string[];
  onChange: (ids: string[]) => void;
  /** Open the full Library screen (add / rename / delete). Optional. */
  onManage?: () => void;
  /** How many selected ids the server said it ignored (config_ack). */
  ignoredCount?: number;
}

export default function LibraryPicker({ selectedIds, onChange, onManage, ignoredCount = 0 }: LibraryPickerProps) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<LibraryList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const load = useCallback(async () => {
    try {
      const res = await listLibrary();
      if (!mounted.current) return;
      setData(res);
      setError(null);
    } catch (e) {
      if (mounted.current) setError((e as Error).message || "Couldn't load your library.");
    }
  }, []);

  useEffect(() => {
    if (open) void load();
  }, [open, load]);

  // Drop ids that are gone from the library (deleted on another device).
  useEffect(() => {
    if (!data) return;
    const known = new Set(data.items.map((i) => i.id));
    const kept = selectedIds.filter((id) => known.has(id));
    if (kept.length !== selectedIds.length) onChange(kept);
  }, [data, selectedIds, onChange]);

  const toggle = (id: string) => {
    if (selectedIds.includes(id)) onChange(selectedIds.filter((x) => x !== id));
    else if (selectedIds.length < MAX_LIBRARY_SELECTION) onChange([...selectedIds, id]);
  };

  const count = selectedIds.length;
  return (
    <View style={styles.container} testID="library-picker">
      <TouchableOpacity
        onPress={() => setOpen((v) => !v)}
        accessibilityRole="button"
        accessibilityState={{ expanded: open }}
        testID="library-picker-toggle"
      >
        <Text style={styles.label}>
          {open ? "▾" : "▸"} Use from library{count ? ` · ${count} selected` : ""}
        </Text>
      </TouchableOpacity>
      {ignoredCount > 0 ? (
        <Text style={styles.note} testID="library-picker-ignored">
          {ignoredCount === 1 ? "1 selected item isn't available" : `${ignoredCount} selected items aren't available`}{" "}
          and won&apos;t be used.
        </Text>
      ) : null}
      {open ? (
        <View>
          {!data && !error ? <ActivityIndicator /> : null}
          {error ? (
            <Text style={styles.error} testID="library-picker-error">
              {error}
            </Text>
          ) : null}
          {data && !data.retrieval_available ? (
            <Text style={styles.note} testID="library-picker-retrieval-unavailable">
              {RETRIEVAL_UNAVAILABLE_TEXT}
            </Text>
          ) : null}
          {data && data.items.length === 0 ? (
            <Text style={styles.note} testID="library-picker-empty">
              Your library is empty.
            </Text>
          ) : null}
          {data?.items.map((item) => {
            const selected = selectedIds.includes(item.id);
            const usable = item.status === "ready";
            return (
              <TouchableOpacity
                key={item.id}
                testID={`library-pick-${item.id}`}
                style={styles.row}
                onPress={() => toggle(item.id)}
                disabled={!usable && !selected}
                accessibilityRole="checkbox"
                accessibilityState={{ checked: selected, disabled: !usable && !selected }}
              >
                <Text style={[styles.box, selected && styles.boxOn]}>{selected ? "☑" : "☐"}</Text>
                <Text style={[styles.itemTitle, !usable && styles.dim]} numberOfLines={1}>
                  {item.title}
                  {item.status === "processing" ? " (processing…)" : item.status === "failed" ? " (failed)" : ""}
                </Text>
              </TouchableOpacity>
            );
          })}
          {count >= MAX_LIBRARY_SELECTION ? (
            <Text style={styles.note}>Up to {MAX_LIBRARY_SELECTION} items per session.</Text>
          ) : null}
          {onManage ? (
            <TouchableOpacity onPress={onManage} accessibilityRole="button" testID="library-picker-manage">
              <Text style={styles.link}>Manage library</Text>
            </TouchableOpacity>
          ) : null}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { paddingHorizontal: 16, paddingVertical: 6 },
  label: { fontSize: 13, fontWeight: "600", color: "#374151", marginBottom: 6 },
  row: { flexDirection: "row", alignItems: "center", paddingVertical: 6, gap: 8 },
  box: { fontSize: 18, color: "#9CA3AF" },
  boxOn: { color: "#4A90D9" },
  itemTitle: { flex: 1, fontSize: 14, color: "#111827" },
  dim: { color: "#9CA3AF" },
  note: { fontSize: 12, color: "#92400E", marginBottom: 6 },
  error: { fontSize: 12, color: "#B91C1C", marginBottom: 6 },
  link: { fontSize: 13, color: "#2563EB", fontWeight: "600", marginTop: 6 },
});
