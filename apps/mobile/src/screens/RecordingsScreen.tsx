import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import { listRecordings, deleteRecording } from "../api/client";
import type { RecordingSummary } from "../api/client";
import { formatTime } from "../components/MediaPlayer";

// House colors.
const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";
const DANGER = "#DC2626";

interface RecordingsScreenProps {
  onSelectRecording: (id: string) => void;
  onBack: () => void;
}

/** Honest message for the list-level failures (same mapping spirit as
 *  ReplayScreen): 503 = storage not configured. */
function humanizeError(message: string): string {
  if (message.includes("503")) return "Replay storage isn’t enabled yet.";
  if (message.includes("401")) return "Please sign in again to see recordings.";
  return message;
}

/**
 * The stored-recordings list: each row shows filename, date, duration, and a
 * type icon; tapping opens the replay. A per-row delete uses an inline confirm
 * (no native Alert, so the flow is deterministic and testable) before calling
 * DELETE. Honest empty/error/loading states throughout.
 */
export default function RecordingsScreen({
  onSelectRecording,
  onBack,
}: RecordingsScreenProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [recordings, setRecordings] = useState<RecordingSummary[]>([]);
  // The row currently awaiting delete confirmation (id), and any inline delete
  // error keyed by id.
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const mountedRef = useRef(true);
  const inFlightRef = useRef(false);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const load = useCallback(async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    setLoading(true);
    setError(null);
    try {
      const list = await listRecordings();
      if (mountedRef.current) setRecordings(list);
    } catch (e) {
      if (mountedRef.current) {
        setError(
          humanizeError(e instanceof Error ? e.message : "Something went wrong."),
        );
      }
    } finally {
      inFlightRef.current = false;
      if (mountedRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const confirmDelete = useCallback(async (id: string) => {
    setDeletingId(id);
    setDeleteError(null);
    try {
      await deleteRecording(id);
      if (mountedRef.current) {
        setRecordings((prev) => prev.filter((r) => r.id !== id));
        setConfirmingId(null);
      }
    } catch (e) {
      if (mountedRef.current) {
        setDeleteError(
          humanizeError(e instanceof Error ? e.message : "Couldn’t delete."),
        );
      }
    } finally {
      if (mountedRef.current) setDeletingId(null);
    }
  }, []);

  return (
    <View style={styles.flex}>
      <View style={styles.header}>
        <TouchableOpacity testID="recordings-back" onPress={onBack}>
          <Text style={styles.backText}>‹ Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Recordings</Text>
        <View style={styles.headerSpacer} />
      </View>

      {loading && (
        <View style={styles.centered} testID="recordings-loading">
          <ActivityIndicator size="large" color={PRIMARY} />
        </View>
      )}

      {!loading && error && (
        <View style={styles.centered} testID="recordings-error">
          <Text style={styles.errorTitle}>Couldn’t load recordings</Text>
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity
            testID="recordings-retry"
            style={styles.retryButton}
            onPress={() => void load()}
          >
            <Text style={styles.retryText}>Try again</Text>
          </TouchableOpacity>
        </View>
      )}

      {!loading && !error && recordings.length === 0 && (
        <View style={styles.centered} testID="recordings-empty">
          <Text style={styles.emptyText}>No stored recordings yet.</Text>
        </View>
      )}

      {!loading && !error && recordings.length > 0 && (
        <ScrollView
          style={styles.flex}
          contentContainerStyle={styles.content}
          testID="recordings-list"
        >
          {recordings.map((rec) => (
            <View
              key={rec.id}
              style={styles.card}
              testID={`recording-${rec.id}`}
            >
              <TouchableOpacity
                style={styles.cardMain}
                testID={`recording-open-${rec.id}`}
                onPress={() => onSelectRecording(rec.id)}
              >
                <Text style={styles.typeIcon}>
                  {rec.media_type === "video" ? "🎬" : "🎧"}
                </Text>
                <View style={styles.cardBody}>
                  <Text style={styles.filename} numberOfLines={1}>
                    {rec.filename}
                  </Text>
                  <Text style={styles.meta}>
                    {new Date(rec.created_at).toLocaleDateString()}
                    {/* duration can be null (decode degraded, no transcript end
                        time) — omit it rather than render a fake 0:00 */}
                    {rec.duration_seconds !== null
                      ? ` · ${formatTime(rec.duration_seconds)}`
                      : ""}
                    {rec.has_analysis ? " · analyzed" : ""}
                  </Text>
                </View>
              </TouchableOpacity>

              {/* Inline delete confirm (id-scoped). */}
              {confirmingId === rec.id ? (
                <View style={styles.confirmRow} testID={`confirm-${rec.id}`}>
                  <Text style={styles.confirmText}>Delete?</Text>
                  <TouchableOpacity
                    testID={`confirm-yes-${rec.id}`}
                    disabled={deletingId === rec.id}
                    onPress={() => void confirmDelete(rec.id)}
                  >
                    {deletingId === rec.id ? (
                      <ActivityIndicator size="small" color={DANGER} />
                    ) : (
                      <Text style={styles.confirmYes}>Delete</Text>
                    )}
                  </TouchableOpacity>
                  <TouchableOpacity
                    testID={`confirm-no-${rec.id}`}
                    onPress={() => {
                      setConfirmingId(null);
                      setDeleteError(null);
                    }}
                  >
                    <Text style={styles.confirmNo}>Cancel</Text>
                  </TouchableOpacity>
                </View>
              ) : (
                <TouchableOpacity
                  testID={`recording-delete-${rec.id}`}
                  style={styles.deleteButton}
                  onPress={() => {
                    setConfirmingId(rec.id);
                    setDeleteError(null);
                  }}
                >
                  <Text style={styles.deleteButtonText}>Delete</Text>
                </TouchableOpacity>
              )}

              {confirmingId === rec.id && deleteError && (
                <Text style={styles.deleteError} testID={`delete-error-${rec.id}`}>
                  {deleteError}
                </Text>
              )}
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
    paddingTop: 56,
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
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  emptyText: { color: MUTED, fontSize: 15 },
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
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 12,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  cardMain: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  typeIcon: { fontSize: 24 },
  cardBody: { flex: 1 },
  filename: { fontSize: 15, fontWeight: "700", color: INK },
  meta: { fontSize: 12.5, color: MUTED, marginTop: 2 },
  deleteButton: {
    alignSelf: "flex-start",
    marginTop: 10,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  deleteButtonText: { fontSize: 13, color: DANGER, fontWeight: "600" },
  confirmRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 16,
    marginTop: 10,
  },
  confirmText: { fontSize: 13, color: INK, fontWeight: "600" },
  confirmYes: { fontSize: 13, color: DANGER, fontWeight: "700" },
  confirmNo: { fontSize: 13, color: MUTED, fontWeight: "600" },
  deleteError: { marginTop: 6, fontSize: 12.5, color: DANGER },
});
