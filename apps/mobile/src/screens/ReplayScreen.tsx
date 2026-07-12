import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import {
  getRecording,
  getRecordingMediaUrl,
  getRecordingSourceUrl,
} from "../api/client";
import type { RecordingDetail } from "../api/client";
import HeatChart from "../components/HeatChart";
import MediaPlayer, { MediaPlayerHandle } from "../components/MediaPlayer";

// House colors.
const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";

interface ReplayScreenProps {
  recordingId: string;
  onBack: () => void;
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
export default function ReplayScreen({ recordingId, onBack }: ReplayScreenProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<RecordingDetail | null>(null);
  const [mediaUrl, setMediaUrl] = useState<string | null>(null);
  // HD replay: for a link-sourced recording we stream the user's OWN hosted
  // original (hdMode) and fall back to our stored derivative on any failure
  // (fellBack drives an honest note). Uploads use neither.
  const [hdMode, setHdMode] = useState(false);
  const [fellBack, setFellBack] = useState(false);
  // Mirrors hdMode for the async player-error handler (avoids a stale closure)
  // and guards the fallback so a burst of player errors triggers it only once.
  const hdRef = useRef(false);

  // Playback position (seconds), pushed up from MediaPlayer at ~4Hz to drive the
  // heat chart playhead.
  const [playheadSeconds, setPlayheadSeconds] = useState(0);
  // Stretch: for video, overlay the chart on the bottom third of the frame.
  const [overlayMode, setOverlayMode] = useState(false);

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
    setHd(false);
    try {
      // Detail first — its `source` decides which playback URL to resolve.
      const rec = await getRecording(recordingId);
      if (mountedRef.current) setDetail(rec);

      if (rec.source?.type === "link") {
        // HD-first: stream the user's own hosted original. On ANY failure to
        // resolve it, fall back to the stored derivative with an honest note.
        try {
          const src = await getRecordingSourceUrl(recordingId);
          if (mountedRef.current) {
            setMediaUrl(src.url);
            setHd(true);
          }
        } catch {
          await loadDerivative();
          if (mountedRef.current) setFellBack(true);
        }
      } else {
        // Upload (or a server that omits `source`): derivative-only, untouched.
        await loadDerivative();
      }
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

  // Tap-to-seek from the chart: drive the player's position directly.
  const handleSeekToTurn = useCallback((startTime: number) => {
    playerRef.current?.seek(startTime);
  }, []);

  const perTurn = detail?.analysis?.per_turn ?? [];
  const turns = detail?.turns ?? [];
  const hasChart = perTurn.length > 0 && turns.length > 0;
  const isVideo = detail?.media_type === "video";

  const chart = hasChart ? (
    <HeatChart
      perTurn={perTurn}
      turns={turns.map((t) => ({ speaker: t.speaker, text: t.text }))}
      turnsTiming={turns.map((t) => ({
        start_time: t.start_time,
        end_time: t.end_time,
      }))}
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
        <Text style={styles.headerTitle} numberOfLines={1}>
          {detail?.filename ?? "Replay"}
        </Text>
        <View style={styles.headerSpacer} />
      </View>

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
          {/* HD badge: streaming the user's own linked source. */}
          {hdMode && (
            <View style={styles.hdBadge} testID="hd-badge">
              <Text style={styles.hdBadgeText}>
                HD · streaming from your linked source
              </Text>
            </View>
          )}

          {/* Honest note when the linked source was unavailable and we fell
              back to the stored derivative. */}
          {fellBack && (
            <Text style={styles.fallbackNote} testID="source-fallback-note">
              original source unavailable — playing stored copy
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
                      Tap a point to jump the recording there.
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
  fallbackNote: {
    fontSize: 12,
    color: MUTED,
    fontStyle: "italic",
    marginBottom: 10,
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
