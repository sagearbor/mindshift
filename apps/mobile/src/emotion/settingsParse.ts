import { DEFAULT_USER_SETTINGS, type EmotionDevUserSettings } from "./config";

/** Defensive parse: anything unreadable or out of range falls back to the default. */
function inRange(v: unknown, lo: number, hi: number, dflt: number): number {
  const n = Number(v);
  return v !== undefined && v !== null && Number.isFinite(n) && n >= lo && n <= hi ? n : dflt;
}

export function parseSettings(raw: string | null): EmotionDevUserSettings {
  if (!raw) return DEFAULT_USER_SETTINGS;
  try {
    const o = JSON.parse(raw) as Partial<EmotionDevUserSettings>;
    const perHour = Number(o.maxNudgesPerHour);
    return {
      enabled: o.enabled === true,
      maxNudgesPerHour: Number.isFinite(perHour) && perHour >= 0 && perHour <= 60 ? Math.floor(perHour) : DEFAULT_USER_SETTINGS.maxNudgesPerHour,
      nudgeSuggestion: o.nudgeSuggestion === true,
      nudgeRule: o.nudgeRule === "tiny" ? "tiny" : "both",
      saveAudio: o.saveAudio === true,
      saveAudioMinProb: inRange(o.saveAudioMinProb, 0.1, 0.95, DEFAULT_USER_SETTINGS.saveAudioMinProb),
      // 500 was the old default (it silently deleted unrated clips); treat a saved 500 as "never chose" and use the new default.
      storageCapMb: Number(o.storageCapMb) === 500 ? DEFAULT_USER_SETTINGS.storageCapMb : inRange(o.storageCapMb, 50, 20000, DEFAULT_USER_SETTINGS.storageCapMb),
      neutralSampleEveryMin: inRange(o.neutralSampleEveryMin, 0, 120, DEFAULT_USER_SETTINGS.neutralSampleEveryMin),
    };
  } catch {
    return DEFAULT_USER_SETTINGS;
  }
}

