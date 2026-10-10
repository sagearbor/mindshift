/**
 * Listening-stage knobs a REPLAY may override from the environment, so one
 * tuning sweep (scripts/recording_replay.py -> recordingReplay.ts) can A/B
 * the VAD / segmenter / turn splitter without editing code. Unset = what the
 * phone ships (the production constants), so a plain replay is the app.
 *
 *   MINDSHIFT_VAD_AGC=0|1         gain stage in front of Silero (vad.ts VAD_AGC_ENABLED)
 *   MINDSHIFT_VAD_ON=0.4          Silero speech-on threshold (off = on - 0.15 unless MINDSHIFT_VAD_OFF)
 *   MINDSHIFT_VAD_OFF=0.25
 *   MINDSHIFT_SEG_MERGE_GAP=0.3   segmenter merge gap / silence-close (s)
 *   MINDSHIFT_SEG_MIN=0.6         segmenter minimum span (s)
 *   MINDSHIFT_TURN_SPLIT=0|1      speaker-change split inside a VAD span (turnSplit.ts)
 *   MINDSHIFT_TURN_SPLIT_THR=0.3  its cosine threshold (implies the split on)
 *   MINDSHIFT_TURN_SPLIT_WIN=1.5 / _HOP=0.5 / _MIN_SPAN=3
 */
import { TURN_SPLIT_DEFAULTS, type TurnSplitConfig } from "../turnSplit";
import { DEFAULT_SEGMENTER_CONFIG, type SegmenterConfig } from "../segmenter";
import { SILERO_SPEECH_OFF, SILERO_SPEECH_ON, VAD_AGC_ENABLED } from "../vad";

export interface ListeningTuning {
  agc: boolean;
  vadOn: number;
  vadOff: number;
  segmenter: SegmenterConfig;
  turnSplit: boolean | TurnSplitConfig | null; // null = the loop's own default
}

function num(env: Record<string, string | undefined>, key: string): number | null {
  const v = env[key];
  if (v === undefined || v.trim() === "") return null;
  const n = Number(v);
  if (!Number.isFinite(n)) throw new Error(`${key}=${v}: not a number`);
  return n;
}

function flag(env: Record<string, string | undefined>, key: string): boolean | null {
  const v = env[key];
  if (v === undefined || v.trim() === "") return null;
  return !["0", "false", "off", "no"].includes(v.trim().toLowerCase());
}

export function tuningFromEnv(env: Record<string, string | undefined> = process.env): ListeningTuning {
  const on = num(env, "MINDSHIFT_VAD_ON");
  const off = num(env, "MINDSHIFT_VAD_OFF");
  const vadOn = on ?? SILERO_SPEECH_ON;
  const vadOff = off ?? (on !== null ? Math.max(0.01, on - 0.15) : SILERO_SPEECH_OFF);
  return {
    agc: flag(env, "MINDSHIFT_VAD_AGC") ?? VAD_AGC_ENABLED,
    vadOn,
    vadOff,
    segmenter: {
      mergeGapSeconds: num(env, "MINDSHIFT_SEG_MERGE_GAP") ?? DEFAULT_SEGMENTER_CONFIG.mergeGapSeconds,
      minSeconds: num(env, "MINDSHIFT_SEG_MIN") ?? DEFAULT_SEGMENTER_CONFIG.minSeconds,
    },
    turnSplit: splitFromEnv(env),
  };
}

function splitFromEnv(env: Record<string, string | undefined>): boolean | TurnSplitConfig | null {
  const on = flag(env, "MINDSHIFT_TURN_SPLIT");
  const thr = num(env, "MINDSHIFT_TURN_SPLIT_THR");
  const win = num(env, "MINDSHIFT_TURN_SPLIT_WIN");
  const hop = num(env, "MINDSHIFT_TURN_SPLIT_HOP");
  const minSpan = num(env, "MINDSHIFT_TURN_SPLIT_MIN_SPAN");
  if (on === false) return false;
  if (thr === null && win === null && hop === null && minSpan === null) return on;
  const w = win ?? TURN_SPLIT_DEFAULTS.windowSec;
  return {
    ...TURN_SPLIT_DEFAULTS,
    threshold: thr ?? TURN_SPLIT_DEFAULTS.threshold,
    windowSec: w,
    minPieceSec: w,
    hopSec: hop ?? TURN_SPLIT_DEFAULTS.hopSec,
    minSpanSec: minSpan ?? TURN_SPLIT_DEFAULTS.minSpanSec,
  };
}

/** Short description for logs / the replay output. */
export function describeTuning(t: ListeningTuning): string {
  return (
    `agc=${t.agc ? "on" : "off"} vad=${t.vadOn}/${t.vadOff} ` +
    `seg gap=${t.segmenter.mergeGapSeconds}s min=${t.segmenter.minSeconds}s split=${
      typeof t.turnSplit === "object" && t.turnSplit !== null
        ? `thr${t.turnSplit.threshold}/win${t.turnSplit.windowSec}/hop${t.turnSplit.hopSec}`
        : (t.turnSplit ?? "default")
    }`
  );
}
