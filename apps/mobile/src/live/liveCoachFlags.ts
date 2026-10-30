/**
 * Live Coach behaviour flags for the PRODUCTION fast loop ("quiet coach",
 * 2026-10-29). One place, plain constants, so any of them can be rolled back
 * with a JS-only OTA update (scripts/ota_publish.sh) — no native rebuild.
 * The FastLoop class itself keeps every one of these OFF by default (tests
 * and replays opt in explicitly); createDefaultFastLoop / createWebFastLoop
 * and useAudioStream read them from here.
 *
 * Server counterparts are env flags on Cloud Run (server/coach_trigger.py:
 * MINDSHIFT_TRIGGER_GATE and friends) and roll back with an env change.
 */
export interface LiveCoachFlags {
  /** Mid-turn wearer floor cue at >= 45 s (live/floorTick.ts). Replayed
   *  offline: 6 of 7 cues inside a real >= 45 s wearer floor run. */
  floorTick: boolean;
  /** Heated-words buzz (live/heatLexicon.ts). Offline: 10/10 fires on units
   *  both blind judges rated heated, 0 on judge-calm material. */
  heatLexiconBuzz: boolean;
  /** Voice a local free-form line only when a trigger holds
   *  (live/coachTrigger.ts). */
  quietCoach: boolean;
  /** Send turn_local before the on-device LLM so the cloud line starts
   *  sooner (FastLoopDeps.earlyTurnLocal). */
  earlyTurnLocal: boolean;
}

export const LIVE_COACH_FLAGS: Readonly<LiveCoachFlags> = Object.freeze({
  floorTick: true,
  heatLexiconBuzz: true,
  quietCoach: true,
  earlyTurnLocal: true,
});

/** The FastLoopDeps fields these flags drive (spread BEFORE the hook's
 *  handlers, so an explicit handler value, e.g. earlyTurnLocal, wins). */
export function liveCoachDeps(flags: Readonly<LiveCoachFlags> = LIVE_COACH_FLAGS): LiveCoachFlags {
  return {
    floorTick: flags.floorTick,
    heatLexiconBuzz: flags.heatLexiconBuzz,
    quietCoach: flags.quietCoach,
    earlyTurnLocal: flags.earlyTurnLocal,
  };
}
