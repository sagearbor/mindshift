import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { listLibrary, RETRIEVAL_UNAVAILABLE_TEXT, type LibraryItem, type LibraryList } from "../api/library";
import { MAX_LIBRARY_SELECTION } from "../live/librarySelection";
import {
  buildTree,
  projectCheckState,
  toggleProject,
  type ProjectNode,
} from "../live/libraryTree";

/**
 * "Use from library" on Live Coach, next to the session note: tick which
 * saved notes/documents the coach may draw on this session (ids only reach
 * the hook, which sends `library_item_ids`). Collapsed by default to one
 * row saying how many are picked. Once the list loads, ids that no longer
 * exist (deleted elsewhere) are dropped from the selection.
 *
 * Ticking a project selects every ready file in it and its sub-projects
 * (resolved to item ids here; the server only ever sees ids).
 *
 * The list loads on mount (not only on expand) so the row can warn, even
 * collapsed and while a session runs, when the library has files but none
 * are selected: the coach then uses nothing.
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
  const [limitMsg, setLimitMsg] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
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
      if (!res || !Array.isArray(res.items)) throw new Error("Couldn't load your library.");
      setData(res);
      setError(null);
    } catch (e) {
      if (mounted.current) setError((e as Error).message || "Couldn't load your library.");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

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

  const tree = useMemo(() => buildTree(data?.projects ?? [], data?.items ?? []), [data]);

  const toggle = (id: string) => {
    setLimitMsg(null);
    if (selectedIds.includes(id)) onChange(selectedIds.filter((x) => x !== id));
    else if (selectedIds.length < MAX_LIBRARY_SELECTION) onChange([...selectedIds, id]);
  };

  const toggleNode = (node: ProjectNode) => {
    const res = toggleProject(node, selectedIds);
    if (res.ok) {
      setLimitMsg(null);
      onChange(res.ids);
    } else {
      setLimitMsg(res.message);
    }
  };

  const count = selectedIds.length;
  const hasFiles = (data?.items.length ?? 0) > 0;
  const noneSelected = hasFiles && count === 0;

  const renderItem = (item: LibraryItem, depth: number) => {
    const selected = selectedIds.includes(item.id);
    const usable = item.status === "ready";
    return (
      <TouchableOpacity
        key={item.id}
        testID={`library-pick-${item.id}`}
        style={[styles.row, { paddingLeft: depth * 16 }]}
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
  };

  const renderNode = (node: ProjectNode): React.ReactNode => {
    const id = node.project.id;
    const state = projectCheckState(node, selectedIds);
    const isCollapsed = !!collapsed[id];
    return (
      <View key={id}>
        <View style={[styles.row, { paddingLeft: node.depth * 16 }]}>
          <TouchableOpacity
            testID={`library-pick-project-fold-${id}`}
            onPress={() => setCollapsed((c) => ({ ...c, [id]: !c[id] }))}
            accessibilityRole="button"
            accessibilityState={{ expanded: !isCollapsed }}
            hitSlop={{ top: 6, bottom: 6, left: 6, right: 6 }}
          >
            <Text style={styles.fold}>{isCollapsed ? "▸" : "▾"}</Text>
          </TouchableOpacity>
          <TouchableOpacity
            testID={`library-pick-project-${id}`}
            style={styles.projectMain}
            onPress={() => toggleNode(node)}
            accessibilityRole="checkbox"
            accessibilityState={{ checked: state === "all" ? true : state === "some" ? "mixed" : false }}
          >
            <Text style={[styles.box, state !== "none" && styles.boxOn]}>
              {state === "all" ? "☑" : state === "some" ? "◪" : "☐"}
            </Text>
            <Text style={styles.projectTitle} numberOfLines={1}>
              📁 {node.project.name}
            </Text>
          </TouchableOpacity>
        </View>
        {isCollapsed ? null : (
          <View>
            {node.children.map(renderNode)}
            {node.items.map((it) => renderItem(it, node.depth + 1))}
          </View>
        )}
      </View>
    );
  };

  return (
    <View style={styles.container} testID="library-picker">
      <TouchableOpacity
        onPress={() => setOpen((v) => !v)}
        accessibilityRole="button"
        accessibilityState={{ expanded: open }}
        testID="library-picker-toggle"
      >
        <Text style={noneSelected ? [styles.label, styles.labelWarn] : styles.label}>
          {open ? "▾" : "▸"} Use from library{count || noneSelected ? ` · ${count} selected` : ""}
        </Text>
      </TouchableOpacity>
      {noneSelected ? (
        <Text style={styles.warn} testID="library-picker-none-selected">
          No files in use — tick the ones you want
        </Text>
      ) : null}
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
          {tree.roots.map(renderNode)}
          {tree.rootItems.map((it) => renderItem(it, 0))}
          {limitMsg ? (
            <Text style={styles.error} testID="library-picker-limit">
              {limitMsg}
            </Text>
          ) : null}
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
  labelWarn: { color: "#B91C1C" },
  warn: {
    fontSize: 12,
    color: "#B91C1C",
    backgroundColor: "#FEE2E2",
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 4,
    marginBottom: 6,
  },
  row: { flexDirection: "row", alignItems: "center", paddingVertical: 6, gap: 8 },
  projectMain: { flex: 1, flexDirection: "row", alignItems: "center", gap: 8 },
  fold: { fontSize: 14, color: "#6B7280", width: 14 },
  projectTitle: { flex: 1, fontSize: 14, fontWeight: "600", color: "#111827" },
  box: { fontSize: 18, color: "#9CA3AF" },
  boxOn: { color: "#4A90D9" },
  itemTitle: { flex: 1, fontSize: 14, color: "#111827" },
  dim: { color: "#9CA3AF" },
  note: { fontSize: 12, color: "#92400E", marginBottom: 6 },
  error: { fontSize: 12, color: "#B91C1C", marginBottom: 6 },
  link: { fontSize: 13, color: "#2563EB", fontWeight: "600", marginTop: 6 },
});
