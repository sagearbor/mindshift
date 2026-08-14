/**
 * Pre-recording reality check: is there room for the session, and is the
 * battery likely to survive it? Pure math over facts the caller gathered —
 * unknown inputs are reported honestly, never guessed.
 */

/** Below this much recordable time free, starting is refused outright — a
 *  session that dies of a full disk in minutes helps no one. */
export const MIN_STORAGE_SECONDS = 10 * 60;

/** Battery fraction below which we warn (never block — the user may be
 *  plugged in, or know better). */
export const BATTERY_WARN_LEVEL = 0.3;

export interface PreflightInput {
  /** Free disk space in bytes; null when the platform can't say. */
  freeBytes: number | null;
  /** Battery level 0..1; null when unknown (no battery API / simulator). */
  batteryLevel: number | null;
  /** On-disk cost of one second of audio for the active format. */
  bytesPerSecond: number;
  /** How long a session we plan for (the honest worst case, not the average). */
  plannedSeconds: number;
}

export interface PreflightReport {
  storage: {
    /** True = do not start (less than MIN_STORAGE_SECONDS of audio fits). */
    blocking: boolean;
    /** Human message for a warning or block; null when storage is fine. */
    message: string | null;
    freeBytes: number | null;
    neededBytes: number;
  };
  battery: {
    warn: boolean;
    message: string | null;
    level: number | null;
  };
  canStart: boolean;
}

export function estimateSessionBytes(
  bytesPerSecond: number,
  seconds: number,
): number {
  return bytesPerSecond * seconds;
}

export function runPreflight(input: PreflightInput): PreflightReport {
  const neededBytes = estimateSessionBytes(
    input.bytesPerSecond,
    input.plannedSeconds,
  );

  let blocking = false;
  let storageMessage: string | null = null;
  if (input.freeBytes === null) {
    storageMessage =
      "Couldn’t estimate free storage — make sure the device has room for a long recording.";
  } else {
    const fitSeconds = input.freeBytes / input.bytesPerSecond;
    if (fitSeconds < MIN_STORAGE_SECONDS) {
      blocking = true;
      storageMessage = `Not enough storage — only about ${Math.max(
        0,
        Math.floor(fitSeconds / 60),
      )} min of audio fits. Free up space before recording.`;
    } else if (input.freeBytes < neededBytes) {
      storageMessage = `Storage is tight — about ${Math.floor(
        fitSeconds / 60,
      )} min of audio fits (planning for ${Math.round(
        input.plannedSeconds / 60,
      )} min).`;
    }
  }

  let batteryWarn = false;
  let batteryMessage: string | null = null;
  if (input.batteryLevel !== null && input.batteryLevel < BATTERY_WARN_LEVEL) {
    batteryWarn = true;
    batteryMessage = `Battery at ${Math.round(
      input.batteryLevel * 100,
    )}% — plug in for a long session.`;
  }

  return {
    storage: {
      blocking,
      message: storageMessage,
      freeBytes: input.freeBytes,
      neededBytes,
    },
    battery: {
      warn: batteryWarn,
      message: batteryMessage,
      level: input.batteryLevel,
    },
    canStart: !blocking,
  };
}
