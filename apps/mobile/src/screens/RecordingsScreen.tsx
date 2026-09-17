import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  AppState,
  RefreshControl,
  StyleSheet,
} from "react-native";
import { listRecordingsAndShared, deleteRecording } from "../api/client";
import type { RecordingSummary, SharedRecordingSummary } from "../api/client";
import { formatTime } from "../components/MediaPlayer";
import { formatDateTime } from "../utils/dateDisplay";
import { deleteCachedMedia } from "../utils/mediaCache";
import {
  isRecordingsListStale,
  mergeRecordingsList,
  readRecordingsCache,
  writeRecordingsCache,
} from "../utils/recordingsListCache";
import { useAuthStore } from "../store/authStore";

/** A cached list older than this is refreshed when the app returns to the
 *  foreground (a fresh mount ALWAYS refreshes in the background). */
export const FOREGROUND_REFRESH_MAX_AGE_MS = 60_000;

// House colors.
const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";
const DANGER = "#DC2626";

interface RecordingsScreenProps {
  onSelectRecording: (id: string) => void;
  /** Optional (Task N3 fix round 1): omitted when Recordings is reached with
   *  `returnTo: "home"` — that shape makes it a PRIMARY screen (see
   *  App.tsx's `isPrimary`), rendered inside AppChrome, which already
   *  provides a way back — a dedicated back button here would duplicate it.
   *  Still passed (and rendered) when pushed from Analyze (`returnTo:
   *  "analyze"`), same as before this fix. */
  onBack?: () => void;
}

/** Honest message for the list-level failures (same mapping spirit as
 *  ReplayScreen): 503 = storage not configured. */
function humanizeError(message: string): string {
  if (message.includes("503")) return "Replay storage isn’t enabled yet.";
  if (message.includes("401")) return "Please sign in again to see recordings.";
  return message;
}

/**
 * Honest participant line for a list row from the raw `manual_speaker_labels`
 * the list returns (a {speaker_id: name} map of the user's OWN names). We only
 * name people the user actually named — the list carries no diarized roster or
 * inferred labels, so we never invent "Speaker A & B". Returns null when nobody
 * has been named yet (the row simply shows no participant line). Names join as
 * "A", "A & B", "A, B & C"; a long roster is trimmed to "A, B & 2 more".
 */
export function formatParticipants(
  manualLabels: Record<string, string> | undefined,
): string | null {
  if (!manualLabels) return null;
  const names = Object.values(manualLabels)
    .map((n) => (typeof n === "string" ? n.trim() : ""))
    .filter((n) => n.length > 0);
  if (names.length === 0) return null;
  if (names.length === 1) return names[0];
  if (names.length === 2) return `${names[0]} & ${names[1]}`;
  if (names.length === 3) return `${names[0]}, ${names[1]} & ${names[2]}`;
  return `${names[0]}, ${names[1]} & ${names.length - 2} more`;
}

/**
 * The stored-recordings list: each row shows filename, date, duration, and a
 * type icon; tapping opens the replay. A per-row delete uses an inline confirm
 * (no native Alert, so the flow is deterministic and testable) before calling
 * DELETE. Honest empty/error/loading states throughout.
 *
 * Cache-first: the last list this account saw renders immediately from the
 * on-device cache (recordingsListCache.ts) while a background fetch updates
 * it in place — spinner and error screen only when there's nothing cached.
 * Refreshes on mount, on pull-to-refresh, and on foreground return when the
 * list is older than FOREGROUND_REFRESH_MAX_AGE_MS or was marked stale by a
 * list-changing action (delete / rename / share / new recording).
 */
export default function RecordingsScreen({
  onSelectRecording,
  onBack,
}: RecordingsScreenProps) {
  const userId = useAuthStore((s) => s.user?.uid ?? null);

  // Cache-first (2026-08-30, "Recordings takes a while to load"): the list
  // the phone already has renders on the FIRST frame, straight from the
  // per-account cache (a synchronous read — see recordingsListCache.ts), and
  // the network only *updates* it in the background. The spinner is reserved
  // for the genuinely-nothing-to-show case (no cache yet), exactly as before.
  // Read ONCE, on the first render (a lazy initializer, so re-renders don't
  // re-read); the ref just lets the other initializers share the result.
  const [initialCacheValue] = useState(() => readRecordingsCache(userId));
  const initialCache = useRef(initialCacheValue);
  const [recordings, setRecordings] = useState<RecordingSummary[]>(
    () => initialCache.current?.recordings ?? [],
  );
  // Recordings other accounts have shared with the user (read-only). Additive —
  // an older server omits the section, leaving this empty (nothing rendered).
  const [sharedWithMe, setSharedWithMe] = useState<SharedRecordingSummary[]>(
    () => initialCache.current?.sharedWithMe ?? [],
  );
  // True while we have something cached on screen (drives the quiet
  // "updating…" / "couldn't refresh" notes instead of spinner / error screen).
  const [hasCache, setHasCache] = useState(initialCache.current !== null);
  // Full-screen spinner: ONLY when there's no cache to show.
  const [loading, setLoading] = useState(initialCache.current === null);
  // Background refresh in flight over a cached list ("Updating…").
  const [updating, setUpdating] = useState(false);
  // Pull-to-refresh in flight (the RefreshControl's own spinner).
  const [refreshing, setRefreshing] = useState(false);
  // Full-screen error: only when there's no cache to fall back on.
  const [error, setError] = useState<string | null>(null);
  // Quiet note when a background refresh failed but the cache is still up.
  const [refreshNote, setRefreshNote] = useState<string | null>(null);
  // The row currently awaiting delete confirmation (id), and any inline delete
  // error keyed by id.
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const mountedRef = useRef(true);
  const inFlightRef = useRef(false);
  // When the on-screen list was last confirmed against the server (epoch ms):
  // the cache's own fetched_at until the first refresh lands. Drives the
  // foreground-return staleness check.
  const lastFetchedAtRef = useRef(initialCache.current?.fetched_at ?? 0);
  // Mirrors of state the async paths need without re-creating `load`.
  const hasCacheRef = useRef(initialCache.current !== null);
  const recordingsRef = useRef(recordings);
  useEffect(() => {
    recordingsRef.current = recordings;
  }, [recordings]);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  /**
   * Fetch the list. `mode` decides what the user sees meanwhile:
   *  - "initial": spinner if there's nothing cached, else a quiet
   *    "Updating…" over the cached rows;
   *  - "pull": the RefreshControl spinner (always a real network fetch).
   * On success the rows update IN PLACE (stable keys keep the scroll position;
   * unchanged rows aren't even re-rendered) and the cache is rewritten. On
   * failure with a cache present the cache stays up with a quiet note; with
   * no cache the honest full-screen error renders, as before.
   */
  const load = useCallback(
    async (mode: "initial" | "pull" = "initial") => {
      if (inFlightRef.current) return;
      inFlightRef.current = true;
      const showingCache = hasCacheRef.current;
      if (mode === "pull") setRefreshing(true);
      else if (showingCache) setUpdating(true);
      else setLoading(true);
      setError(null);
      setRefreshNote(null);
      try {
        const result = await listRecordingsAndShared();
        const fetchedAt = Date.now();
        writeRecordingsCache(userId, result, fetchedAt);
        lastFetchedAtRef.current = fetchedAt;
        hasCacheRef.current = true;
        if (mountedRef.current) {
          setRecordings((prev) => mergeRecordingsList(prev, result.recordings));
          setSharedWithMe((prev) => mergeRecordingsList(prev, result.sharedWithMe));
          setHasCache(true);
        }
      } catch (e) {
        if (mountedRef.current) {
          const msg = humanizeError(
            e instanceof Error ? e.message : "Something went wrong.",
          );
          if (showingCache) {
            // Keep the cached list; never replace it with an error screen.
            setRefreshNote(`Couldn’t refresh — showing your saved list. ${msg}`);
          } else {
            setError(msg);
          }
        }
      } finally {
        inFlightRef.current = false;
        if (mountedRef.current) {
          setLoading(false);
          setUpdating(false);
          setRefreshing(false);
        }
      }
    },
    [userId],
  );

  // Every mount refreshes in the background (a fresh mount is the only way
  // in; the cached rows are already on screen by the time this runs).
  useEffect(() => {
    void load("initial");
  }, [load]);

  // Returning to the foreground: refresh if the on-screen list is older than
  // the max age OR a client-side mutation (delete / rename / share / a new
  // recording) marked it stale since it was fetched.
  useEffect(() => {
    const sub = AppState.addEventListener("change", (state) => {
      if (state !== "active") return;
      const age = Date.now() - lastFetchedAtRef.current;
      if (
        age > FOREGROUND_REFRESH_MAX_AGE_MS ||
        isRecordingsListStale(lastFetchedAtRef.current)
      ) {
        void load("initial");
      }
    });
    return () => sub.remove();
  }, [load]);

  const handlePullRefresh = useCallback(() => {
    void load("pull");
  }, [load]);

  const confirmDelete = useCallback(async (id: string) => {
    setDeletingId(id);
    setDeleteError(null);
    try {
      await deleteRecording(id);
      // Best-effort local-cache cleanup (2026-08-18): only after the server
      // confirms the delete — a failed deleteRecording must leave any cached
      // copy alone, since the recording still exists and would otherwise be
      // forced to re-fetch from the network on its next replay for no reason.
      // Fire-and-forget, fail-open — mirrors avatarStore.ts's deleteAvatarFile;
      // never blocks or fails this action over a cache-cleanup miss.
      void deleteCachedMedia(id);
      const next = recordingsRef.current.filter((r) => r.id !== id);
      // Keep the cache honest so the next open doesn't briefly resurrect the
      // deleted row before its background refresh lands.
      const cached = readRecordingsCache(userId);
      if (cached) {
        writeRecordingsCache(
          userId,
          { recordings: next, sharedWithMe: cached.sharedWithMe },
          cached.fetched_at,
        );
      }
      if (mountedRef.current) {
        setRecordings(next);
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
  }, [userId]);

  return (
    <View style={styles.flex}>
      <View style={styles.header}>
        {onBack ? (
          <TouchableOpacity
            testID="recordings-back"
            onPress={onBack}
            hitSlop={{ top: 10, bottom: 10, left: 8, right: 16 }}
          >
            <Text style={styles.backText}>‹ Back</Text>
          </TouchableOpacity>
        ) : (
          <View style={styles.headerSpacer} />
        )}
        <Text style={styles.headerTitle}>Recordings</Text>
        <View style={styles.headerSpacer} />
      </View>

      {/* Quiet status over a cached list — never a spinner, never an error
          screen, while there's something real to show. */}
      {hasCache && updating && !refreshing && (
        <View style={styles.statusRow} testID="recordings-updating">
          <ActivityIndicator size="small" color={MUTED} />
          <Text style={styles.statusText}>Updating…</Text>
        </View>
      )}
      {hasCache && refreshNote && (
        <View style={styles.statusRow} testID="recordings-refresh-note">
          <Text style={styles.statusText} numberOfLines={2}>
            {refreshNote}
          </Text>
        </View>
      )}

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

      {!loading &&
        !error &&
        recordings.length === 0 &&
        sharedWithMe.length === 0 && (
          <ScrollView
            style={styles.flex}
            contentContainerStyle={styles.centered}
            testID="recordings-empty"
            refreshControl={
              <RefreshControl
                testID="recordings-refresh"
                refreshing={refreshing}
                onRefresh={handlePullRefresh}
              />
            }
          >
            <Text style={styles.emptyText}>No stored recordings yet.</Text>
          </ScrollView>
        )}

      {!loading &&
        !error &&
        (recordings.length > 0 || sharedWithMe.length > 0) && (
        <ScrollView
          style={styles.flex}
          contentContainerStyle={styles.content}
          testID="recordings-list"
          refreshControl={
            <RefreshControl
              testID="recordings-refresh"
              refreshing={refreshing}
              onRefresh={handlePullRefresh}
            />
          }
        >
          {/* "Shared with me" — recordings other accounts shared with the user
              (read-only). Each opens the normal replay, which renders in read-only
              mode. No delete affordance (the recipient can't delete). */}
          {sharedWithMe.length > 0 && (
            <View testID="shared-with-me-section">
              <Text style={styles.sectionHeader}>Shared with me</Text>
              {sharedWithMe.map((rec) => (
                <TouchableOpacity
                  key={`shared-${rec.id}`}
                  style={styles.card}
                  testID={`shared-open-${rec.id}`}
                  onPress={() => onSelectRecording(rec.id)}
                >
                  <View style={styles.cardMain}>
                    <Text style={styles.typeIcon}>
                      {rec.media_type === "video" ? "🎬" : "🎧"}
                    </Text>
                    <View style={styles.cardBody}>
                      <Text style={styles.filename} numberOfLines={1}>
                        {rec.title || rec.filename}
                      </Text>
                      <Text style={styles.meta}>
                        {formatDateTime(rec.created_at) ?? ""}
                        {rec.duration_seconds !== null
                          ? ` · ${formatTime(rec.duration_seconds)}`
                          : ""}
                        {rec.has_analysis ? " · analyzed" : ""}
                      </Text>
                      {/* Who it's from — honest, never fabricated. */}
                      {rec.owner_email ? (
                        <Text
                          style={styles.fromLine}
                          numberOfLines={1}
                          testID={`shared-from-${rec.id}`}
                        >
                          from {rec.owner_email}
                        </Text>
                      ) : null}
                    </View>
                  </View>
                </TouchableOpacity>
              ))}
              {recordings.length > 0 && (
                <Text style={styles.sectionHeader}>Your recordings</Text>
              )}
            </View>
          )}

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
                    {rec.title || rec.filename}
                  </Text>
                  <Text style={styles.meta}>
                    {/* Full, unambiguous date + wall-clock time (this year omits
                        the year). Never fabricated: a missing/invalid created_at
                        renders nothing rather than a guessed date. */}
                    {formatDateTime(rec.created_at) ?? ""}
                    {/* duration can be null (decode degraded, no transcript end
                        time) — omit it rather than render a fake 0:00 */}
                    {rec.duration_seconds !== null
                      ? ` · ${formatTime(rec.duration_seconds)}`
                      : ""}
                    {rec.has_analysis ? " · analyzed" : ""}
                  </Text>
                  {/* Named participants — only the people the user actually named
                      (from the list's manual_speaker_labels). Omitted when none,
                      so we never fabricate a roster. */}
                  {formatParticipants(rec.manual_speaker_labels) && (
                    <Text
                      style={styles.participants}
                      numberOfLines={1}
                      testID={`recording-participants-${rec.id}`}
                    >
                      {formatParticipants(rec.manual_speaker_labels)}
                    </Text>
                  )}
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
    // App's SafeAreaView already applies the notch inset; this base pad matches
    // the hub screens (~20-24) instead of the old hardcoded 56 that double-padded
    // on notched devices.
    paddingTop: 24,
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
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 6,
    backgroundColor: "#F3F4F6",
  },
  statusText: { fontSize: 12.5, color: MUTED, flexShrink: 1 },
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
  participants: {
    fontSize: 12.5,
    color: PRIMARY,
    fontWeight: "600",
    marginTop: 2,
  },
  sectionHeader: {
    fontSize: 12,
    fontWeight: "700",
    color: MUTED,
    textTransform: "uppercase",
    letterSpacing: 0.5,
    marginBottom: 8,
    marginTop: 4,
  },
  fromLine: {
    fontSize: 12.5,
    color: PRIMARY,
    fontWeight: "600",
    marginTop: 2,
  },
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
