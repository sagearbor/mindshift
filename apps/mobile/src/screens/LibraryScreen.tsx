import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import * as DocumentPicker from "expo-document-picker";
import {
  createLibraryNote,
  deleteLibraryItem,
  getLibraryItem,
  listLibrary,
  RETRIEVAL_UNAVAILABLE_TEXT,
  updateLibraryItem,
  uploadLibraryFile,
  type LibraryItem,
  type LibraryList,
} from "../api/library";
import { showAlert } from "../utils/showAlert";

const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";
const DANGER = "#DC2626";

/** What the server can read (server/library/extract.py). */
export const LIBRARY_PICKER_TYPES = [
  "application/pdf",
  "text/plain",
  "text/markdown",
  "text/csv",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
];

/** While any item is still being processed, re-list this often. */
export const LIBRARY_POLL_MS = 3000;

interface LibraryScreenProps {
  onBack: () => void;
}

function describe(item: LibraryItem): string {
  const kind = item.kind === "note" ? "Note" : item.kind === "table" ? "Table" : "Document";
  const size = item.chars >= 1000 ? `${Math.round(item.chars / 1000)}k chars` : `${item.chars} chars`;
  const status =
    item.status === "processing" ? "processing…" : item.status === "failed" ? "failed" : "ready";
  return `${kind} · ${size} · ${status}`;
}

/**
 * Coach library: material the earpiece coach can draw on in Live Coach.
 * Add a note (title + text), upload a file, rename, edit a note's text,
 * delete (with confirm). The list is exactly GET /library/items; every error
 * shown is the server's own words.
 */
export default function LibraryScreen({ onBack }: LibraryScreenProps) {
  const [data, setData] = useState<LibraryList | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const [noteOpen, setNoteOpen] = useState(false);
  const [noteTitle, setNoteTitle] = useState("");
  const [noteText, setNoteText] = useState("");

  const [renaming, setRenaming] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState("");
  const [editBlocked, setEditBlocked] = useState<string | null>(null);

  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const res = await listLibrary();
      if (!mounted.current) return;
      setData(res);
      setLoadError(null);
    } catch (e) {
      if (!mounted.current) return;
      setLoadError((e as Error).message || "Couldn't load your library.");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const processing = data?.items.some((i) => i.status === "processing") ?? false;
  useEffect(() => {
    if (!processing) return;
    const t = setTimeout(() => void refresh(), LIBRARY_POLL_MS);
    return () => clearTimeout(t);
  }, [processing, data, refresh]);

  const run = useCallback(
    async (key: string, fn: () => Promise<unknown>) => {
      setBusy(key);
      setActionError(null);
      try {
        await fn();
        await refresh();
        return true;
      } catch (e) {
        if (mounted.current) setActionError((e as Error).message || "Something went wrong.");
        return false;
      } finally {
        if (mounted.current) setBusy(null);
      }
    },
    [refresh],
  );

  const saveNote = useCallback(async () => {
    const ok = await run("note", () => createLibraryNote(noteTitle.trim(), noteText));
    if (ok && mounted.current) {
      setNoteOpen(false);
      setNoteTitle("");
      setNoteText("");
    }
  }, [noteTitle, noteText, run]);

  const upload = useCallback(async () => {
    const result = await DocumentPicker.getDocumentAsync({
      type: LIBRARY_PICKER_TYPES,
      copyToCacheDirectory: true,
      multiple: false,
    });
    if (result.canceled) return;
    const asset = result.assets?.[0];
    if (!asset) return;
    await run("upload", () =>
      uploadLibraryFile({
        uri: asset.uri,
        name: asset.name,
        mimeType: asset.mimeType,
        file: (asset as { file?: File }).file ?? null,
      }),
    );
  }, [run]);

  const saveRename = useCallback(
    async (id: string) => {
      const ok = await run(`rename-${id}`, () => updateLibraryItem(id, { title: renameDraft.trim() }));
      if (ok && mounted.current) setRenaming(null);
    },
    [renameDraft, run],
  );

  const startEdit = useCallback(async (item: LibraryItem) => {
    setActionError(null);
    setEditBlocked(null);
    setBusy(`load-${item.id}`);
    try {
      const detail = await getLibraryItem(item.id);
      if (!mounted.current) return;
      if (detail.preview_truncated) {
        setEditBlocked(item.id);
        return;
      }
      setEditing(item.id);
      setEditDraft(detail.preview);
    } catch (e) {
      if (mounted.current) setActionError((e as Error).message || "Couldn't open that note.");
    } finally {
      if (mounted.current) setBusy(null);
    }
  }, []);

  const saveEdit = useCallback(
    async (id: string) => {
      const ok = await run(`edit-${id}`, () => updateLibraryItem(id, { text: editDraft }));
      if (ok && mounted.current) setEditing(null);
    },
    [editDraft, run],
  );

  const confirmDelete = useCallback(
    (item: LibraryItem) => {
      showAlert(
        "Delete from library?",
        `"${item.title}" will be removed for good, including from any session that uses it.`,
        [
          { text: "Cancel", style: "cancel" },
          { text: "Delete", style: "destructive", onPress: () => void run(`delete-${item.id}`, () => deleteLibraryItem(item.id)) },
        ],
      );
    },
    [run],
  );

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content} testID="library-screen">
      <View style={styles.header}>
        <TouchableOpacity onPress={onBack} accessibilityRole="button" testID="library-back">
          <Text style={styles.back}>‹ Back</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Coach library</Text>
      </View>
      <Text style={styles.intro}>
        Notes and documents the coach can draw on. Pick which ones to use on the Live Coach screen.
      </Text>

      {data && !data.retrieval_available ? (
        <Text style={styles.notice} testID="library-retrieval-unavailable">
          {RETRIEVAL_UNAVAILABLE_TEXT}
        </Text>
      ) : null}

      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.button}
          onPress={() => setNoteOpen((v) => !v)}
          accessibilityRole="button"
          testID="library-add-note"
        >
          <Text style={styles.buttonText}>Add note</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() => void upload()}
          disabled={busy === "upload"}
          accessibilityRole="button"
          testID="library-upload"
        >
          <Text style={styles.buttonText}>{busy === "upload" ? "Uploading…" : "Upload file"}</Text>
        </TouchableOpacity>
      </View>
      <Text style={styles.hint}>PDF, Word, text, Markdown, CSV or Excel, up to 10 MB.</Text>

      {noteOpen ? (
        <View style={styles.card} testID="library-note-form">
          <TextInput
            testID="library-note-title"
            style={styles.input}
            value={noteTitle}
            onChangeText={setNoteTitle}
            placeholder="Title"
            placeholderTextColor="#9CA3AF"
            maxLength={200}
          />
          <TextInput
            testID="library-note-text"
            style={[styles.input, styles.multiline]}
            value={noteText}
            onChangeText={setNoteText}
            placeholder="What should the coach know?"
            placeholderTextColor="#9CA3AF"
            multiline
          />
          <View style={styles.rowActions}>
            <TouchableOpacity
              testID="library-note-save"
              onPress={() => void saveNote()}
              disabled={busy === "note" || !noteTitle.trim() || !noteText.trim()}
              accessibilityRole="button"
            >
              <Text style={styles.link}>{busy === "note" ? "Saving…" : "Save"}</Text>
            </TouchableOpacity>
            <TouchableOpacity testID="library-note-cancel" onPress={() => setNoteOpen(false)} accessibilityRole="button">
              <Text style={styles.linkMuted}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      ) : null}

      {actionError ? (
        <Text style={styles.error} testID="library-action-error">
          {actionError}
        </Text>
      ) : null}

      {loadError && !data ? (
        <View testID="library-load-error">
          <Text style={styles.error}>{loadError}</Text>
          <TouchableOpacity onPress={() => void refresh()} accessibilityRole="button">
            <Text style={styles.link}>Try again</Text>
          </TouchableOpacity>
        </View>
      ) : null}

      {!data && !loadError ? <ActivityIndicator style={styles.spinner} /> : null}

      {data && data.items.length === 0 ? (
        <Text style={styles.empty} testID="library-empty">
          Nothing saved yet. Add a note or upload a file.
        </Text>
      ) : null}

      {data?.items.map((item) => (
        <View key={item.id} style={styles.card} testID={`library-item-${item.id}`}>
          {renaming === item.id ? (
            <View>
              <TextInput
                testID={`library-rename-input-${item.id}`}
                style={styles.input}
                value={renameDraft}
                onChangeText={setRenameDraft}
                maxLength={200}
              />
              <View style={styles.rowActions}>
                <TouchableOpacity
                  testID={`library-rename-save-${item.id}`}
                  onPress={() => void saveRename(item.id)}
                  disabled={!renameDraft.trim()}
                  accessibilityRole="button"
                >
                  <Text style={styles.link}>Save</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={() => setRenaming(null)} accessibilityRole="button">
                  <Text style={styles.linkMuted}>Cancel</Text>
                </TouchableOpacity>
              </View>
            </View>
          ) : (
            <Text style={styles.itemTitle}>{item.title}</Text>
          )}
          <Text style={styles.itemSub} testID={`library-status-${item.id}`}>
            {describe(item)}
          </Text>
          {item.status === "failed" && item.error ? <Text style={styles.error}>{item.error}</Text> : null}

          {editing === item.id ? (
            <View>
              <TextInput
                testID={`library-edit-input-${item.id}`}
                style={[styles.input, styles.multiline]}
                value={editDraft}
                onChangeText={setEditDraft}
                multiline
              />
              <View style={styles.rowActions}>
                <TouchableOpacity
                  testID={`library-edit-save-${item.id}`}
                  onPress={() => void saveEdit(item.id)}
                  disabled={!editDraft.trim()}
                  accessibilityRole="button"
                >
                  <Text style={styles.link}>Save text</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={() => setEditing(null)} accessibilityRole="button">
                  <Text style={styles.linkMuted}>Cancel</Text>
                </TouchableOpacity>
              </View>
            </View>
          ) : null}
          {editBlocked === item.id ? (
            <Text style={styles.itemSub} testID={`library-edit-blocked-${item.id}`}>
              This note is too long to edit on the phone. Delete it and add a new one instead.
            </Text>
          ) : null}

          {renaming !== item.id && editing !== item.id ? (
            <View style={styles.rowActions}>
              <TouchableOpacity
                testID={`library-rename-${item.id}`}
                onPress={() => {
                  setRenaming(item.id);
                  setRenameDraft(item.title);
                }}
                accessibilityRole="button"
              >
                <Text style={styles.link}>Rename</Text>
              </TouchableOpacity>
              {item.kind === "note" ? (
                <TouchableOpacity
                  testID={`library-edit-${item.id}`}
                  onPress={() => void startEdit(item)}
                  accessibilityRole="button"
                >
                  <Text style={styles.link}>Edit text</Text>
                </TouchableOpacity>
              ) : null}
              <TouchableOpacity
                testID={`library-delete-${item.id}`}
                onPress={() => confirmDelete(item)}
                accessibilityRole="button"
              >
                <Text style={styles.danger}>Delete</Text>
              </TouchableOpacity>
            </View>
          ) : null}
        </View>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#FFFFFF" },
  content: { padding: 16, paddingBottom: 48 },
  header: { flexDirection: "row", alignItems: "center", gap: 12, marginBottom: 8 },
  back: { fontSize: 16, color: PRIMARY, fontWeight: "600" },
  title: { fontSize: 20, fontWeight: "700", color: INK },
  intro: { fontSize: 14, color: MUTED, marginBottom: 12 },
  notice: {
    fontSize: 13,
    color: "#92400E",
    backgroundColor: "#FEF3C7",
    borderRadius: 8,
    padding: 10,
    marginBottom: 12,
  },
  actions: { flexDirection: "row", gap: 10 },
  button: { backgroundColor: PRIMARY, borderRadius: 8, paddingVertical: 10, paddingHorizontal: 16 },
  buttonText: { color: "#FFFFFF", fontWeight: "600", fontSize: 14 },
  hint: { fontSize: 12, color: MUTED, marginTop: 6, marginBottom: 12 },
  card: {
    borderWidth: 1,
    borderColor: "#E5E7EB",
    borderRadius: 10,
    padding: 12,
    marginBottom: 10,
  },
  input: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    padding: 8,
    fontSize: 14,
    color: "#111827",
    backgroundColor: "#FFFFFF",
    marginBottom: 8,
  },
  multiline: { minHeight: 96, maxHeight: 240, textAlignVertical: "top" },
  rowActions: { flexDirection: "row", gap: 18, marginTop: 6 },
  link: { color: PRIMARY, fontWeight: "600", fontSize: 14 },
  linkMuted: { color: MUTED, fontWeight: "600", fontSize: 14 },
  danger: { color: DANGER, fontWeight: "600", fontSize: 14 },
  error: { color: DANGER, fontSize: 13, marginBottom: 8 },
  spinner: { marginTop: 24 },
  empty: { fontSize: 14, color: MUTED, marginTop: 12 },
  itemTitle: { fontSize: 15, fontWeight: "600", color: INK },
  itemSub: { fontSize: 12, color: MUTED, marginTop: 2 },
});
