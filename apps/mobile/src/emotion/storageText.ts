/** Plain-text storage lines for the Emotion tester panel (kept free of React Native so tests import it directly). */
import type { EmotionDevStore } from "./devStore";

type Usage = ReturnType<EmotionDevStore["usage"]> | null;
const mbOf = (b: number) => (b / (1024 * 1024)).toFixed(1);

/** "Storage used: 12.3 MB of 2048 MB (40 clips, 12 unrated)" */
export function storageLine(usage: Usage, capMb: number): string {
  const cap = usage ? Math.round(usage.capBytes / (1024 * 1024)) : capMb;
  return `Storage used: ${mbOf(usage?.clipBytes ?? 0)} MB of ${cap} MB (${usage?.clips ?? 0} clips, ${usage?.unrated ?? 0} unrated)`;
}

/** A persistent warning once storage is 80% full, and a loud one when new clips are being skipped. */
export function storageWarning(usage: Usage): string | null {
  if (!usage) return null;
  if (usage.state === "full" || usage.skipped > 0) {
    return `Storage is FULL: new clips are NOT being saved${usage.skipped ? ` (${usage.skipped} skipped so far)` : ""}. Nothing unrated is deleted. Rate clips in Review (rated clips make room) or raise the cap.`;
  }
  if (usage.state === "warn") return "Storage is over 80% full. At 100% new clips stop saving (nothing unrated is ever deleted).";
  return null;
}
