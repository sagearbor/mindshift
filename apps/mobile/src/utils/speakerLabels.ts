/**
 * Single rendering path for a speaker's DISPLAY name (§2/§3). The server resolves
 * a per-speaker `display_label` (a real name inferred from the transcript, a
 * relative "Deeper/Higher voice" pitch label, or the raw id) plus the `label_source`
 * rung that produced it. Every screen that shows a speaker — the heat chart legend
 * and talk-share line, the replay/turn inspector, report cards, dynamics, and
 * transcript views — funnels through `speakerLabel` so the label is consistent
 * everywhere and OLD recordings (whose stored analysis has no `speaker_labels`)
 * render exactly as before: the raw speaker id.
 *
 * A concurrent effort adds an "enrolled" rung ABOVE these on the server; nothing
 * here needs to change for that — the map simply carries a new `label_source`.
 */
import type { SpeakerLabel } from "../api/client";

export type SpeakerLabels = Record<string, SpeakerLabel> | null | undefined;

/**
 * The human-facing label for `speaker`, given the analysis's `speaker_labels`
 * map. Falls back to the raw speaker id whenever the map is absent (old
 * recording / pre-labels server) or the entry is missing/blank — so a speaker is
 * NEVER rendered as an empty string.
 */
export function speakerLabel(speaker: string, labels?: SpeakerLabels): string {
  const entry = labels?.[speaker];
  if (entry && typeof entry.display_label === "string") {
    const trimmed = entry.display_label.trim();
    if (trimmed) return trimmed;
  }
  return speaker;
}
