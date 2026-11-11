/**
 * Emotion dev tester — every tunable in one file (docs/plans/2026-11-07-emotion-dev-mode.md).
 *
 * This is a measuring instrument for ONE owner judging two experimental
 * models; none of these numbers is a product claim and none is tuned on
 * real data yet. Change them here, not in the logic.
 */

/** Model output order (input_contract.json "labels"). */
export const EMOTION_LABELS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"] as const;
export type EmotionLabel = (typeof EMOTION_LABELS)[number];
export type Probs = readonly number[]; // length 6, EMOTION_LABELS order

/**
 * Per-class logit offsets added to log-probabilities before re-normalising.
 * OFF (all zero): the offsets that helped on acted data only did not carry to
 * real speech, so we ship raw and LOG raw. Kept as a constant so the owner
 * can experiment; `applyOffsets` is a no-op while this is all zeros.
 */
export const LOGIT_OFFSETS: readonly number[] = [0, 0, 0, 0, 0, 0];

export const EMOTION_DEV_DEFAULTS = {
  /** Window fed to the models: the last 6 s (the models' fixed input). */
  windowS: 6,
  /** A new window is classified at most this often. */
  hopS: 3,
  /** Skip windows with no speech-like energy (saves battery, avoids noise logs). */
  gateFrameMs: 100,
  gateFrameDbfs: -45,
  gateMinActiveFraction: 0.3,

  /** --- Nudge (TINY only) --- */
  /** Top class must not be neutral and must reach this probability. */
  nudgeMinProb: 0.5,
  /** Classes that may buzz (neutral never does). */
  nudgeLabels: ["anger", "disgust", "fear", "happiness", "sadness"] as EmotionLabel[],
  /** The same top class on this many consecutive windows before a buzz. */
  nudgeConfirmWindows: 2,
  /** Buzz budget: at most this many per rolling hour… */
  maxNudgesPerHour: 4,
  /** …and never closer together than this. */
  minNudgeGapS: 120,

  /** --- Nudge rule "both" (default): TINY and SMALL must name the SAME
   *  negative class, each at >= this probability. TINY's disgust alone never
   *  buzzes (scripted test 2026-11: TINY says disgust on almost anything). --- */
  nudgeBothMinProb: 0.6,

  /** --- Save-audio mode (default OFF) --- */
  saveAudioClipS: 6,
  saveAudioDefaultMinProb: 0.5,
  saveAudioDefaultCapMb: 2048,
  /** Storage warning fraction: the panel and Live Coach status warn from here. */
  capWarnFraction: 0.8,
  /** A user mark/flag may still save this far past the cap (their explicit act). */
  capUserOverageFraction: 0.1,

  /** --- Episode clips: consecutive triggered windows merge into ONE clip --- */
  episodeLeadS: 5,
  episodeTailS: 3,
  episodeMaxS: 45,

  /** --- Neutral sampling (measure MISSED emotions): one clip per N min of speech --- */
  neutralSampleClipS: 10,
  neutralSampleDefaultEveryMin: 5,

  /** --- Flag capture --- */
  flagBeforeS: 45,
  flagAfterS: 15,
  /** Ring buffer holds this much audio (>= flagBeforeS; headroom for a late tap). */
  ringS: 60,

  /** --- Disagreement auto-clip --- */
  disagreeMinProb: 0.6,
  maxDisagreeClipsPerDay: 20,
  disagreeClipS: 8,

  /** --- Storage (on device only) --- */
  maxClipBytes: 150 * 1024 * 1024,
  maxLogBytes: 8 * 1024 * 1024,
  /** A nudge with no flag after this long is final as "unrated". */
  unratedAfterS: 600,
};

export type EmotionDevConfig = typeof EMOTION_DEV_DEFAULTS;

/** A single tester-set override layer (persisted by devPrefs.ts). */
export interface EmotionDevUserSettings {
  enabled: boolean;
  maxNudgesPerHour: number;
  nudgeSuggestion: boolean; // show the (optional) suggestion text on the watch
  /** "both" = both models must agree (default); "tiny" = the old TINY-only rule. */
  nudgeRule: NudgeRule;
  /** Save-audio mode: keep a 6 s clip whenever either model hears a non-neutral emotion. */
  saveAudio: boolean;
  saveAudioMinProb: number;
  /** Storage cap for saved clips, in MB. At the cap NEW clips stop saving (unrated clips are never deleted). */
  storageCapMb: number;
  /** Save a ~10 s clip of both-models-neutral speech about once per this many minutes of speech (0 = off). Needs saveAudio. */
  neutralSampleEveryMin: number;
}

export type NudgeRule = "both" | "tiny";

export const DEFAULT_USER_SETTINGS: EmotionDevUserSettings = {
  enabled: false,
  maxNudgesPerHour: EMOTION_DEV_DEFAULTS.maxNudgesPerHour,
  nudgeSuggestion: false,
  nudgeRule: "both",
  saveAudio: false,
  saveAudioMinProb: EMOTION_DEV_DEFAULTS.saveAudioDefaultMinProb,
  storageCapMb: EMOTION_DEV_DEFAULTS.saveAudioDefaultCapMb,
  neutralSampleEveryMin: EMOTION_DEV_DEFAULTS.neutralSampleDefaultEveryMin,
};

export const MODEL_FILES = {
  tiny: "emotion-tiny-int8.onnx",
  small: "emotion-small-int8.onnx",
} as const;
/** Sanity bounds on a sideloaded file (tiny ~10 MB, small ~95 MB). */
export const MODEL_MIN_BYTES = { tiny: 5_000_000, small: 50_000_000 } as const;
