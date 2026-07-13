import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import {
  getRecording,
  getRecordingMediaUrl,
  getRecordingSourceUrl,
  patchRecordingSource,
  patchRecordingTitle,
} from "../api/client";
import type { RecordingDetail } from "../api/client";
import HeatChart from "../components/HeatChart";
import MediaPlayer, { MediaPlayerHandle } from "../components/MediaPlayer";
import SpeakerEnrollment from "../components/SpeakerEnrollment";
import { setPlaybackMode } from "../utils/audioMode";

/** How long we give a source to report a real duration (moov parsed /
 *  readyToPlay) before treating it as stuck. A moov-at-end HD MP4 served
 *  without HTTP Range support buffers forever at 0:00 with no error event, so a
 *  wall-clock watchdog is the only way to recover. */
const LOAD_TIMEOUT_MS = 8000;

// House colors.
const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";

interface ReplayScreenProps {
  recordingId: string;
  onBack: () => void;
  /** Open the attach-HD-source input immediately on mount (from the Dynamics
   *  "Attach link now" popup). Otherwise the affordance starts collapsed. */
  initialAttachOpen?: boolean;
}

/** True when a source URL is a Google Photos share link. The Photos `=dv` HD
 *  stream is moov-at-end with NO HTTP Range support, so ExoPlayer can never play
 *  it (device-proven: buffers forever at 0:00, watchdog reverts). For these we
 *  suppress the "Try HD" offer entirely and show an honest note instead — HD
 *  replay is genuinely impossible from a Photos link. Other link types (direct
 *  file, Drive) may stream fine, so they keep the Try-HD affordance. */
function isGooglePhotosUrl(url: string | null | undefined): boolean {
  if (!url) return false;
  const u = url.toLowerCase();
  return u.includes("photos.app.goo.gl") || u.includes("photos.google.com");
}

/** Read the HTTP status off a client error (the `.status` the client attaches,
 *  else parsed from the `API error: <status>` message). */
function statusOf(err: unknown): number {
  const withStatus = err as { status?: number };
  if (typeof withStatus?.status === "number") return withStatus.status;
  const msg = err instanceof Error ? err.message : "";
  return Number(msg.match(/API error: (\d+)/)?.[1] ?? 0);
}

/** Turn a client error into an honest, human message. The pinned contract uses
 *  503 for "storage not configured" — we name that case plainly instead of
 *  showing a raw status. */
function humanizeError(message: string): string {
  if (message.includes("503")) {
    return "Replay storage isn’t enabled yet.";
  }
  if (message.includes("404")) {
    return "This recording is no longer available.";
  }
  if (message.includes("401")) {
    return "Please sign in again to view this recording.";
  }
  return message;
}

/**
 * "Watch yourselves at the spike" — plays a stored recording with the heat chart
 * synced beneath it. A moving playhead tracks playback; tapping the chart seeks
 * the media. Everything is fetched against the pinned recordings contract with
 * honest loading/error/retry states (never a fabricated recording).
 */
export default function ReplayScreen({
  recordingId,
  onBack,
  initialAttachOpen = false,
}: ReplayScreenProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<RecordingDetail | null>(null);
  const [mediaUrl, setMediaUrl] = useState<string | null>(null);
  // Replay plays our STORED DERIVATIVE first for every source — the /media
  // endpoint serves it with proper HTTP Range support, so it loads reliably.
  // For a link-sourced recording the user's OWN hosted original is an opt-in
  // "Try HD" (hdMode): that remote stream can be a moov-at-end MP4 with no Range
  // support, which ExoPlayer buffers forever at 0:00, so it must never be the
  // default. `fellBack` drives an honest "playing stored copy" note.
  const [hdMode, setHdMode] = useState(false);
  const [fellBack, setFellBack] = useState(false);
  // The current media source never reported a real duration within
  // LOAD_TIMEOUT_MS and there's nothing better to switch to (we're already on
  // the derivative) — drives an honest "media isn't loading" note.
  const [mediaStuck, setMediaStuck] = useState(false);
  // Mirrors hdMode for the async player-error / watchdog handlers (avoids a
  // stale closure) and guards the fallback so a burst of errors triggers it once.
  const hdRef = useRef(false);
  // Set true once the current source reports a real duration; the load-timeout
  // watchdog reads it to tell "loaded" from "stuck buffering". A ref (not state)
  // so updating it from the duration callback never re-renders.
  const durationKnownRef = useRef(false);

  // Playback position (seconds), pushed up from MediaPlayer at ~4Hz to drive the
  // heat chart playhead.
  const [playheadSeconds, setPlayheadSeconds] = useState(0);
  // Stretch: for video, overlay the chart on the bottom third of the frame.
  const [overlayMode, setOverlayMode] = useState(false);

  // --- Attach / replace HD source ---
  // The user can attach a durable share/direct link to their own hosted
  // original so replay streams it in HD. `attachOpen` reveals the input;
  // `attachError` renders a 422's user-facing detail verbatim.
  const [attachOpen, setAttachOpen] = useState(initialAttachOpen);
  const [attachUrl, setAttachUrl] = useState("");
  const [attaching, setAttaching] = useState(false);
  const [attachError, setAttachError] = useState<string | null>(null);

  // --- Rename (tap the title to edit) ---
  // `renamedTitle` is a local override applied optimistically on a successful
  // PATCH; `renameNote` carries an honest status line (e.g. the capability-gated
  // "not supported yet" when the backend has no title field). Nothing is
  // fabricated — a failed rename never changes the displayed name.
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const [savingTitle, setSavingTitle] = useState(false);
  const [renamedTitle, setRenamedTitle] = useState<string | null>(null);
  const [renameNote, setRenameNote] = useState<string | null>(null);

  const playerRef = useRef<MediaPlayerHandle>(null);

  // In-flight + mount guards, mirroring DynamicsScreen: one fetch at a time, and
  // no state writes after the user backs out mid-load.
  const inFlightRef = useRef(false);
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Defensively put the device audio session into a PLAYBACK configuration on
  // mount. The Live Coach flow configures a recording session (allowsRecording),
  // which on Android leaves media playback silent until it's reset — and its
  // teardown could be missed. Resetting here guarantees replay is audible no
  // matter how the user got to this screen. Fire-and-forget; web no-ops.
  useEffect(() => {
    void setPlaybackMode().catch(() => {});
  }, []);

  const setHd = useCallback((on: boolean) => {
    hdRef.current = on;
    setHdMode(on);
  }, []);

  // Resolve the stored derivative URL (our own copy). Shared by the upload path,
  // the link fallback, and the player-error fallback.
  const loadDerivative = useCallback(async () => {
    const media = await getRecordingMediaUrl(recordingId);
    if (mountedRef.current) {
      setMediaUrl(media.url);
      setHd(false);
    }
  }, [recordingId, setHd]);

  const load = useCallback(async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    setLoading(true);
    setError(null);
    setFellBack(false);
    setMediaStuck(false);
    setHd(false);
    try {
      // Detail first — its `source` decides whether a "Try HD" opt-in is offered.
      const rec = await getRecording(recordingId);
      if (mountedRef.current) setDetail(rec);

      // Derivative-FIRST for every source. The stored copy streams from our
      // /media endpoint with Range support and always loads; the linked HD
      // original is opt-in (see handleTryHd) precisely because it can buffer
      // forever. Uploads never had an HD option and are unaffected.
      await loadDerivative();
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
  }, [recordingId, loadDerivative, setHd]);

  useEffect(() => {
    void load();
  }, [load]);

  // The player errored on the remote HD stream (expired/blocked/unseekable) —
  // fall back to the stored derivative once. Ignored when we're already on the
  // derivative (there's nothing better to switch to).
  const handlePlayerError = useCallback(async () => {
    if (!hdRef.current) return;
    hdRef.current = false; // guard re-entrancy from a burst of error events
    try {
      await loadDerivative();
      if (mountedRef.current) setFellBack(true);
    } catch {
      if (mountedRef.current) {
        setError(humanizeError("Something went wrong."));
      }
    }
  }, [loadDerivative]);

  // Opt-in HD: resolve and switch to the user's own hosted original. On any
  // failure to resolve it, stay on the derivative and note it — never leave the
  // player pointed at nothing.
  const handleTryHd = useCallback(async () => {
    try {
      const src = await getRecordingSourceUrl(recordingId);
      if (mountedRef.current) {
        setFellBack(false);
        setMediaStuck(false);
        setMediaUrl(src.url);
        setHd(true);
      }
    } catch {
      if (mountedRef.current) setFellBack(true);
    }
  }, [recordingId, setHd]);

  // Manual return to the stored copy (from the HD stream). Also the owner-facing
  // escape hatch when a remote stream misbehaves in ways we can't detect.
  const handleBackToDerivative = useCallback(async () => {
    try {
      setFellBack(false);
      setMediaStuck(false);
      await loadDerivative(); // sets the derivative URL and clears hdMode
    } catch {
      if (mountedRef.current) {
        setError(humanizeError("Something went wrong."));
      }
    }
  }, [loadDerivative]);

  // Marks the current source as loaded once it reports a real duration; the
  // watchdog reads this ref to distinguish "loaded" from "stuck buffering".
  const handleDurationChange = useCallback((seconds: number) => {
    if (seconds > 0) durationKnownRef.current = true;
  }, []);

  // Load-timeout watchdog. Each time the source URL changes, give it
  // LOAD_TIMEOUT_MS to report a real duration. If it never does — the classic
  // moov-at-end-without-Range HD stream buffers forever at 0:00 and emits NO
  // error event, so handlePlayerError never fires — auto-recover: drop from HD
  // to the stored derivative, or (if already on the derivative) surface an
  // honest "isn't loading" note rather than spin at 0:00 indefinitely.
  useEffect(() => {
    if (!mediaUrl) return;
    durationKnownRef.current = false;
    const id = setTimeout(() => {
      if (durationKnownRef.current) return; // Loaded fine — nothing to do.
      if (hdRef.current) {
        hdRef.current = false; // guard against a double-fire
        void loadDerivative().then(() => {
          if (mountedRef.current) setFellBack(true);
        });
      } else if (mountedRef.current) {
        setMediaStuck(true);
      }
    }, LOAD_TIMEOUT_MS);
    return () => clearTimeout(id);
  }, [mediaUrl, loadDerivative]);

  // Tap-to-seek from the chart: drive the player's position directly.
  const handleSeekToTurn = useCallback((startTime: number) => {
    playerRef.current?.seek(startTime);
  }, []);

  // Submit the attach/replace link: PATCH the source, then refetch the recording
  // so the "Try HD" opt-in becomes available for the now-linked source. A 422's
  // user-facing detail is shown verbatim; nothing is fabricated on failure.
  const handleAttachSubmit = useCallback(async () => {
    const url = attachUrl.trim();
    if (!url || attaching) return;
    setAttaching(true);
    setAttachError(null);
    try {
      await patchRecordingSource(recordingId, url);
      if (mountedRef.current) {
        setAttachOpen(false);
        setAttachUrl("");
      }
      // Refetch: detail.source is now a link, so `load` takes the HD-first path.
      await load();
    } catch (e) {
      if (mountedRef.current) {
        const detail = (e as { detail?: string })?.detail;
        setAttachError(
          detail ??
            humanizeError(e instanceof Error ? e.message : "Something went wrong."),
        );
      }
    } finally {
      if (mountedRef.current) setAttaching(false);
    }
  }, [attachUrl, attaching, recordingId, load]);

  // The name shown in the header: a just-applied rename wins, then any
  // server-provided title, then the raw filename.
  const currentTitle =
    renamedTitle ?? detail?.title ?? detail?.filename ?? "Replay";

  const openRename = useCallback(() => {
    setTitleDraft(currentTitle === "Replay" ? "" : currentTitle);
    setRenameNote(null);
    setEditingTitle(true);
  }, [currentTitle]);

  // Attempt the rename. The backend may not support titling yet (no field / no
  // PATCH /recordings/{id} route → a 4xx): in that case we DON'T change the name
  // and show an honest "not supported yet" note rather than pretend it worked. A
  // transient 5xx/network failure gets a distinct "try again" note.
  const handleRenameSubmit = useCallback(async () => {
    const next = titleDraft.trim();
    if (!next || savingTitle) return;
    setSavingTitle(true);
    setRenameNote(null);
    try {
      await patchRecordingTitle(recordingId, next);
      if (mountedRef.current) {
        setRenamedTitle(next);
        setEditingTitle(false);
      }
    } catch (e) {
      if (mountedRef.current) {
        const status = statusOf(e);
        setRenameNote(
          status >= 400 && status < 500
            ? "Renaming isn’t supported yet — coming soon."
            : "Couldn’t rename right now — please try again.",
        );
        setEditingTitle(false);
      }
    } finally {
      if (mountedRef.current) setSavingTitle(false);
    }
  }, [titleDraft, savingTitle, recordingId]);

  const perTurn = detail?.analysis?.per_turn ?? [];
  const turns = detail?.turns ?? [];
  const hasChart = perTurn.length > 0 && turns.length > 0;
  const isVideo = detail?.media_type === "video";
  // Only a link-sourced recording has a user-hosted original to stream in HD.
  const hdAvailable = detail?.source?.type === "link";

  const chart = hasChart ? (
    <HeatChart
      perTurn={perTurn}
      turns={turns.map((t) => ({ speaker: t.speaker, text: t.text }))}
      turnsTiming={turns.map((t) => ({
        start_time: t.start_time,
        end_time: t.end_time,
      }))}
      durationSeconds={detail?.duration_seconds ?? null}
      playheadSeconds={playheadSeconds}
      onSeekToTurn={handleSeekToTurn}
    />
  ) : null;

  return (
    <View style={styles.flex}>
      <View style={styles.header}>
        <TouchableOpacity testID="replay-back" onPress={onBack}>
          <Text style={styles.backText}>‹ Back</Text>
        </TouchableOpacity>
        {editingTitle ? (
          <View style={styles.titleEditRow}>
            <TextInput
              testID="rename-input"
              style={styles.renameInput}
              value={titleDraft}
              onChangeText={setTitleDraft}
              placeholder="Name this conversation"
              placeholderTextColor="#9CA3AF"
              autoFocus
              maxLength={120}
              editable={!savingTitle}
              onSubmitEditing={() => void handleRenameSubmit()}
            />
            <TouchableOpacity
              testID="rename-save"
              onPress={() => void handleRenameSubmit()}
              disabled={savingTitle || !titleDraft.trim()}
            >
              {savingTitle ? (
                <ActivityIndicator size="small" color={PRIMARY} />
              ) : (
                <Text style={styles.renameSave}>Save</Text>
              )}
            </TouchableOpacity>
            <TouchableOpacity
              testID="rename-cancel"
              onPress={() => {
                setEditingTitle(false);
                setRenameNote(null);
              }}
            >
              <Text style={styles.renameCancel}>Cancel</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <TouchableOpacity
            testID="replay-title"
            style={styles.titleWrap}
            onPress={openRename}
            accessibilityRole="button"
            accessibilityLabel="Rename this recording"
          >
            <Text style={styles.headerTitle} numberOfLines={1}>
              {currentTitle}
            </Text>
          </TouchableOpacity>
        )}
        <View style={styles.headerSpacer} />
      </View>

      {renameNote && (
        <Text style={styles.renameNote} testID="rename-note">
          {renameNote}
        </Text>
      )}

      {loading && (
        <View style={styles.centered} testID="replay-loading">
          <ActivityIndicator size="large" color={PRIMARY} />
          <Text style={styles.centeredText}>Loading the recording…</Text>
        </View>
      )}

      {!loading && error && (
        <View style={styles.centered} testID="replay-error">
          <Text style={styles.errorTitle}>Couldn’t open this recording</Text>
          <Text style={styles.errorText} testID="replay-error-message">
            {error}
          </Text>
          <TouchableOpacity
            testID="replay-retry"
            style={styles.retryButton}
            onPress={() => void load()}
          >
            <Text style={styles.retryText}>Try again</Text>
          </TouchableOpacity>
        </View>
      )}

      {!loading && !error && detail && mediaUrl && (
        <ScrollView
          style={styles.flex}
          contentContainerStyle={styles.content}
          testID="replay-content"
        >
          {/* Attach / replace HD source. When the recording isn't yet
              link-sourced we offer "Attach HD source"; when it already is, a
              quieter "Replace source link". Both reveal the same input. */}
          <View style={styles.attachSection}>
            {!attachOpen ? (
              detail.source?.type === "link" ? (
                <TouchableOpacity
                  testID="replace-source-button"
                  style={styles.replaceSourceButton}
                  onPress={() => {
                    setAttachError(null);
                    setAttachOpen(true);
                  }}
                >
                  <Text style={styles.replaceSourceText}>Replace source link</Text>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity
                  testID="attach-source-button"
                  style={styles.attachSourceButton}
                  onPress={() => {
                    setAttachError(null);
                    setAttachOpen(true);
                  }}
                >
                  <Text style={styles.attachSourceButtonText}>
                    Attach HD source
                  </Text>
                </TouchableOpacity>
              )
            ) : (
              <View>
                <Text style={styles.attachHelp}>
                  Paste a share link to your own hosted original (e.g. a Google
                  Photos single-item link). We’ll stream it in HD for replay.
                </Text>
                <TextInput
                  testID="attach-source-input"
                  style={styles.attachInput}
                  placeholder="https://…"
                  value={attachUrl}
                  onChangeText={setAttachUrl}
                  autoCapitalize="none"
                  autoCorrect={false}
                  keyboardType="url"
                  placeholderTextColor="#9CA3AF"
                  editable={!attaching}
                />
                <View style={styles.attachButtonRow}>
                  <TouchableOpacity
                    testID="attach-source-cancel"
                    style={styles.attachCancel}
                    onPress={() => {
                      setAttachOpen(false);
                      setAttachError(null);
                    }}
                    disabled={attaching}
                  >
                    <Text style={styles.attachCancelText}>Cancel</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    testID="attach-source-submit"
                    style={[
                      styles.attachSubmit,
                      (attaching || !attachUrl.trim()) && styles.attachDisabled,
                    ]}
                    onPress={() => void handleAttachSubmit()}
                    disabled={attaching || !attachUrl.trim()}
                  >
                    {attaching ? (
                      <ActivityIndicator color="#FFFFFF" />
                    ) : (
                      <Text style={styles.attachSubmitText}>Attach</Text>
                    )}
                  </TouchableOpacity>
                </View>
                {attachError && (
                  <Text style={styles.attachError} testID="attach-source-error">
                    {attachError}
                  </Text>
                )}
              </View>
            )}
          </View>

          {/* HD badge: streaming the user's own linked source. */}
          {hdMode && (
            <View style={styles.hdBadge} testID="hd-badge">
              <Text style={styles.hdBadgeText}>
                HD · streaming from your linked source
              </Text>
            </View>
          )}

          {/* HD is opt-in: the stored copy plays by default (it loads reliably);
              streaming the user's own original is offered only for a linked
              source. Hidden once HD is active (then we show "Back to stored
              copy" instead).

              Google Photos links are the ONE case where HD is impossible: the
              Photos `=dv` stream is moov-at-end with no Range support, so
              ExoPlayer buffers forever at 0:00. Offering "Try HD" there was
              dishonest — replace it with a plain note. Other link types keep the
              opt-in (they may stream fine; the watchdog stays as a backstop). */}
          {hdAvailable &&
            !hdMode &&
            (isGooglePhotosUrl(detail?.source?.url) ? (
              <Text style={styles.photosHdNote} testID="photos-hd-note">
                HD replay isn’t possible from Google Photos links — playing the
                stored copy.
              </Text>
            ) : (
              <TouchableOpacity
                testID="try-hd-button"
                style={styles.tryHdButton}
                onPress={() => void handleTryHd()}
              >
                <Text style={styles.tryHdButtonText}>
                  ▶ Try HD from your linked source
                </Text>
              </TouchableOpacity>
            ))}

          {/* Escape hatch: manually return to the stored copy when the HD
              stream misbehaves in ways we can't auto-detect. */}
          {hdMode && (
            <TouchableOpacity
              testID="force-derivative"
              style={styles.backToStoredButton}
              onPress={() => void handleBackToDerivative()}
            >
              <Text style={styles.backToStoredText}>Back to stored copy</Text>
            </TouchableOpacity>
          )}

          {/* Honest note when the linked source was unavailable and we fell
              back to (or stayed on) the stored derivative. */}
          {fellBack && (
            <Text style={styles.fallbackNote} testID="source-fallback-note">
              original source unavailable — playing stored copy
            </Text>
          )}

          {/* Honest note when even the stored copy never reported a duration —
              nothing better to switch to. */}
          {mediaStuck && (
            <Text style={styles.fallbackNote} testID="media-stuck-note">
              this recording’s media isn’t loading
            </Text>
          )}

          {/* Overlay toggle — only meaningful for video with a chart. */}
          {isVideo && hasChart && (
            <TouchableOpacity
              testID="overlay-toggle"
              style={styles.overlayToggle}
              onPress={() => setOverlayMode((v) => !v)}
            >
              <Text style={styles.overlayToggleText}>
                {overlayMode ? "Stacked view" : "Overlay chart on video"}
              </Text>
            </TouchableOpacity>
          )}

          {overlayMode && isVideo && hasChart ? (
            // Overlay mode: chart floats over the bottom third of the video.
            <View style={styles.overlayWrap} testID="replay-overlay">
              <MediaPlayer
                ref={playerRef}
                uri={mediaUrl}
                mediaType={detail.media_type}
                onPositionChange={setPlayheadSeconds}
                onDurationChange={handleDurationChange}
                onError={handlePlayerError}
              />
              <View style={styles.overlayChart} pointerEvents="box-none">
                {chart}
              </View>
            </View>
          ) : (
            // Default stacked layout: player on top, chart beneath.
            <>
              <MediaPlayer
                ref={playerRef}
                uri={mediaUrl}
                mediaType={detail.media_type}
                onPositionChange={setPlayheadSeconds}
                onDurationChange={handleDurationChange}
                onError={handlePlayerError}
              />
              <View style={styles.chartCard}>
                {hasChart ? (
                  <>
                    <Text style={styles.sectionTitle}>
                      Heat over the conversation
                    </Text>
                    {chart}
                    <Text style={styles.hint}>
                      Each dash is one turn across the recording — length shows
                      how long they spoke. Tap a dash to jump there.
                    </Text>
                  </>
                ) : (
                  <Text style={styles.noAnalysis} testID="replay-no-analysis">
                    This recording hasn’t been analyzed, so there’s no heat graph
                    to sync.
                  </Text>
                )}
              </View>
            </>
          )}

          {/* "This is me" — enroll a speaker's voice so they're labeled "You"
              in future recordings. Self-hides when the server can't do voice ID
              or there are no diarized turns to choose from. */}
          {turns.length > 0 ? (
            <SpeakerEnrollment recordingId={recordingId} turns={turns} />
          ) : null}
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
  titleWrap: { flex: 1, alignItems: "center" },
  headerTitle: {
    flex: 1,
    textAlign: "center",
    fontSize: 17,
    fontWeight: "700",
    color: INK,
  },
  headerSpacer: { width: 64 },
  titleEditRow: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  renameInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
    fontSize: 15,
    color: INK,
    backgroundColor: "#FFFFFF",
  },
  renameSave: { fontSize: 15, fontWeight: "700", color: PRIMARY },
  renameCancel: { fontSize: 15, fontWeight: "600", color: MUTED },
  renameNote: {
    fontSize: 12.5,
    color: MUTED,
    fontStyle: "italic",
    textAlign: "center",
    paddingHorizontal: 16,
    paddingTop: 8,
  },
  photosHdNote: {
    fontSize: 12.5,
    lineHeight: 18,
    color: MUTED,
    fontStyle: "italic",
    marginBottom: 10,
  },
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  centeredText: { marginTop: 12, color: MUTED, fontSize: 14 },
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
  chartCard: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: INK,
    marginBottom: 12,
  },
  hint: {
    fontSize: 12.5,
    color: MUTED,
    fontStyle: "italic",
    marginTop: 8,
  },
  noAnalysis: {
    fontSize: 14,
    lineHeight: 20,
    color: MUTED,
  },
  hdBadge: {
    alignSelf: "flex-start",
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#EAF3FC",
    borderColor: PRIMARY,
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 4,
    marginBottom: 10,
  },
  hdBadgeText: { fontSize: 12, fontWeight: "700", color: PRIMARY },
  tryHdButton: {
    alignSelf: "flex-start",
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: PRIMARY,
    backgroundColor: "#EEF2FF",
    marginBottom: 10,
  },
  tryHdButtonText: { fontSize: 14, fontWeight: "600", color: PRIMARY },
  backToStoredButton: {
    alignSelf: "flex-start",
    paddingHorizontal: 12,
    paddingVertical: 6,
    marginBottom: 10,
  },
  backToStoredText: {
    fontSize: 13,
    fontWeight: "600",
    color: MUTED,
    textDecorationLine: "underline",
  },
  fallbackNote: {
    fontSize: 12,
    color: MUTED,
    fontStyle: "italic",
    marginBottom: 10,
  },
  attachSection: { marginBottom: 12 },
  attachSourceButton: {
    alignSelf: "flex-start",
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: PRIMARY,
    backgroundColor: "#EEF2FF",
  },
  attachSourceButtonText: { fontSize: 14, fontWeight: "600", color: PRIMARY },
  replaceSourceButton: {
    alignSelf: "flex-start",
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  replaceSourceText: {
    fontSize: 13,
    fontWeight: "600",
    color: MUTED,
    textDecorationLine: "underline",
  },
  attachHelp: {
    fontSize: 12.5,
    lineHeight: 18,
    color: MUTED,
    marginBottom: 8,
  },
  attachInput: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
    color: INK,
    backgroundColor: "#FFFFFF",
  },
  attachButtonRow: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 10,
    marginTop: 10,
  },
  attachCancel: {
    paddingVertical: 10,
    paddingHorizontal: 18,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#D1D5DB",
  },
  attachCancelText: { color: MUTED, fontSize: 14, fontWeight: "600" },
  attachSubmit: {
    paddingVertical: 10,
    paddingHorizontal: 22,
    borderRadius: 8,
    backgroundColor: PRIMARY,
    alignItems: "center",
    justifyContent: "center",
    minWidth: 90,
  },
  attachDisabled: { opacity: 0.6 },
  attachSubmitText: { color: "#FFFFFF", fontSize: 14, fontWeight: "600" },
  attachError: {
    marginTop: 10,
    fontSize: 13,
    lineHeight: 19,
    color: "#DC2626",
  },
  overlayToggle: {
    alignSelf: "flex-end",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#FFFFFF",
    marginBottom: 10,
  },
  overlayToggleText: { fontSize: 12, fontWeight: "600", color: MUTED },
  overlayWrap: {
    position: "relative",
  },
  overlayChart: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    padding: 8,
    backgroundColor: "rgba(255,255,255,0.75)",
    borderTopLeftRadius: 12,
    borderTopRightRadius: 12,
  },
});
