import React, {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { useVideoPlayer, VideoView } from "expo-video";
import type { MediaType } from "../api/client";

// House colors.
const PRIMARY = "#4A90D9";

/** ~4Hz position polling — cheap enough to keep the playhead smooth without
 *  churning React on every frame. */
const POLL_MS = 250;

/** Imperative handle so the parent (and the heat chart's tap-to-seek) can drive
 *  playback position without re-rendering the player on every seek. */
export interface MediaPlayerHandle {
  seek: (seconds: number) => void;
}

interface MediaPlayerProps {
  /** Short-lived signed media URL. expo-video plays both audio and video from a
   *  plain URL string. */
  uri: string;
  /** Drives the layout: `video` renders the frame; `audio` renders a compact
   *  card with just transport + time. */
  mediaType: MediaType;
  /** Fired at ~4Hz with the current position (seconds) so the parent can sync
   *  the heat chart playhead. */
  onPositionChange?: (seconds: number) => void;
  /** Fired once the media's duration becomes known. */
  onDurationChange?: (seconds: number) => void;
  /** Fired when the player fails to load/decode the source (e.g. a remote HD
   *  stream that expired or blocked). ReplayScreen uses this to fall back from
   *  the linked source to the stored derivative. */
  onError?: (message?: string) => void;
}

/** mm:ss for the transport readout. */
export function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) seconds = 0;
  const total = Math.floor(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * A thin wrapper over expo-video's `useVideoPlayer` (which plays audio files
 * too). Chosen over a DOM `<video>` branch because expo-video ships web
 * entrypoints (`*.web.js`) — one code path across native and web, and tests
 * mock this component wholesale so no player module is exercised there.
 *
 * We drive our OWN transport (play/pause + time) rather than native controls so
 * the position we poll is the single source of truth for the synced playhead,
 * and so seeks issued from the chart stay consistent with the button state.
 */
const MediaPlayer = forwardRef<MediaPlayerHandle, MediaPlayerProps>(
  function MediaPlayer(
    { uri, mediaType, onPositionChange, onDurationChange, onError },
    ref,
  ) {
    const player = useVideoPlayer(uri, (p) => {
      p.loop = false;
    });

    // Surface a load/decode failure so the parent can fall back to the stored
    // derivative. expo-video reports it via a `statusChange` event carrying an
    // `error`; guarded so the wholesale test mock (a plain host View) is unaffected.
    useEffect(() => {
      const listener = (payload: { status?: string; error?: unknown }) => {
        if (payload?.status === "error" || payload?.error) {
          const err = payload?.error as { message?: string } | undefined;
          onError?.(err?.message);
        }
      };
      const sub = player?.addListener?.("statusChange", listener);
      return () => sub?.remove?.();
    }, [player, onError]);

    const [playing, setPlaying] = useState(false);
    const [position, setPosition] = useState(0);
    const [duration, setDuration] = useState(0);
    // Emit onDurationChange only once, when duration first resolves.
    const durationSentRef = useRef(false);

    useImperativeHandle(
      ref,
      () => ({
        seek: (seconds: number) => {
          // Setting currentTime seeks in expo-video. Reflect it immediately so
          // the playhead doesn't wait for the next poll tick.
          try {
            player.currentTime = seconds;
          } catch {
            // A not-yet-loaded player can reject a seek; the next poll recovers.
          }
          setPosition(seconds);
          onPositionChange?.(seconds);
        },
      }),
      [player, onPositionChange],
    );

    // Poll the player at ~4Hz for position / play-state / duration.
    useEffect(() => {
      const id = setInterval(() => {
        const cur = player.currentTime ?? 0;
        setPosition(cur);
        onPositionChange?.(cur);
        setPlaying(player.playing ?? false);

        const dur = player.duration ?? 0;
        if (dur > 0 && !durationSentRef.current) {
          durationSentRef.current = true;
          setDuration(dur);
          onDurationChange?.(dur);
        }
      }, POLL_MS);
      return () => clearInterval(id);
    }, [player, onPositionChange, onDurationChange]);

    const toggle = () => {
      if (player.playing) {
        player.pause();
        setPlaying(false);
      } else {
        player.play();
        setPlaying(true);
      }
    };

    const timeLabel = `${formatTime(position)} / ${formatTime(duration)}`;

    return (
      <View testID="media-player" style={styles.container}>
        {mediaType === "video" && (
          <VideoView
            testID="video-view"
            style={styles.video}
            player={player}
            nativeControls={false}
            contentFit="contain"
          />
        )}

        {mediaType === "audio" && (
          <View testID="audio-card" style={styles.audioCard}>
            <Text style={styles.audioGlyph}>♪</Text>
            <Text style={styles.audioLabel}>Audio recording</Text>
          </View>
        )}

        {/* Shared transport row — one play/pause control and a time readout for
            both media types. */}
        <View style={styles.transport}>
          <TouchableOpacity
            testID="play-pause-button"
            style={styles.playButton}
            onPress={toggle}
            accessibilityLabel={playing ? "Pause" : "Play"}
          >
            <Text style={styles.playButtonText}>{playing ? "❚❚" : "▶"}</Text>
          </TouchableOpacity>
          <Text style={styles.time} testID="media-time">
            {timeLabel}
          </Text>
        </View>
      </View>
    );
  },
);

export default MediaPlayer;

const styles = StyleSheet.create({
  container: {
    backgroundColor: "#000000",
    borderRadius: 12,
    overflow: "hidden",
  },
  video: {
    width: "100%",
    aspectRatio: 16 / 9,
    backgroundColor: "#000000",
  },
  audioCard: {
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 32,
    backgroundColor: "#111827",
  },
  audioGlyph: {
    fontSize: 40,
    color: PRIMARY,
    marginBottom: 6,
  },
  audioLabel: {
    fontSize: 14,
    color: "#E5E7EB",
    fontWeight: "600",
  },
  transport: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: "#1F2937",
  },
  playButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: PRIMARY,
    alignItems: "center",
    justifyContent: "center",
  },
  playButtonText: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "700",
  },
  time: {
    color: "#E5E7EB",
    fontSize: 14,
    fontWeight: "600",
    fontVariant: ["tabular-nums"],
  },
});
