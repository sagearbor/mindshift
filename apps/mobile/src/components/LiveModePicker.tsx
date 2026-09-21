import React from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import type { LiveMode } from "../live/localLlm";

/** The five session shapes, with the one line each needs to be picked
 *  correctly. Exported so the screen's explainer and tests share the copy.
 *  "In person" keeps its original wire value `speaker` (see LiveMode);
 *  "Journal" is the all-day listen-for-my-voice mode (no coaching). */
export const LIVE_MODE_OPTIONS: readonly {
  mode: LiveMode;
  label: string;
  hint: string;
}[] = [
  {
    mode: "earpiece",
    label: "Earpiece",
    hint: "Phone to your ear or on a call — the coach whispers to you privately.",
  },
  {
    mode: "speaker",
    label: "In person",
    hint: "Both of you in the room, one mic — the coach speaks only while you're both silent.",
  },
  {
    mode: "therapist",
    label: "Therapist",
    hint: "You're observing two people — on-screen only, nothing is ever spoken.",
  },
  {
    mode: "call",
    label: "Call",
    hint: "MindShift places the call itself — each phone coaches its own person.",
  },
  {
    mode: "journal",
    label: "Journal",
    hint: "Listen for my voice all day — keeps only what you say; no coaching, no transcription until later.",
  },
];

interface Props {
  value: LiveMode;
  onChange: (mode: LiveMode) => void;
  /** Locked while a session runs — the loop reads the mode at start. */
  disabled?: boolean;
  /** Modes to leave OUT of the row entirely (not greyed out — absent).
   *  Today's only user is guest mode hiding "Call": an in-app call needs a
   *  second real account on the other end, so offering it to a guest would be
   *  offering something that cannot work. Defaults to none, so every existing
   *  caller renders exactly the five chips it always did. */
  hiddenModes?: readonly LiveMode[];
}

/**
 * The explicit mode picker for a live session. Replaces the old
 * earpiece/visual toggle + on-device shape row with ONE choice that decides
 * both who is on the mic and whether the coach speaks (therapist never
 * does). Persisted per account by the screen (src/live/modePrefs.ts).
 */
export default function LiveModePicker({
  value,
  onChange,
  disabled,
  hiddenModes,
}: Props) {
  const options = hiddenModes?.length
    ? LIVE_MODE_OPTIONS.filter((o) => !hiddenModes.includes(o.mode))
    : LIVE_MODE_OPTIONS;
  // The hint must describe a mode that is actually on screen: if the selected
  // mode was just hidden, fall back to the first visible one rather than
  // explaining a chip nobody can see.
  const current =
    options.find((o) => o.mode === value) ?? options[0] ?? LIVE_MODE_OPTIONS[0];
  return (
    <View style={styles.wrap} testID="live-mode-picker">
      <Text style={styles.label}>Session mode</Text>
      <View style={styles.row}>
        {options.map((o) => {
          const active = o.mode === value;
          return (
            <TouchableOpacity
              key={o.mode}
              testID={`session-mode-${o.mode}`}
              accessibilityRole="button"
              accessibilityState={{ selected: active, disabled: Boolean(disabled) }}
              style={[styles.chip, active && styles.chipActive, disabled && styles.chipDisabled]}
              disabled={disabled}
              onPress={() => onChange(o.mode)}
            >
              <Text style={[styles.chipText, active && styles.chipTextActive]}>{o.label}</Text>
            </TouchableOpacity>
          );
        })}
      </View>
      <Text style={styles.hint} testID="session-mode-hint">
        {current.hint}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    paddingHorizontal: 16,
    paddingVertical: 6,
    gap: 6,
  },
  label: {
    fontSize: 14,
    fontWeight: "600",
    color: "#374151",
  },
  row: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
  },
  chip: {
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#FFFFFF",
  },
  chipActive: {
    backgroundColor: "#4A90D9",
    borderColor: "#4A90D9",
  },
  chipDisabled: {
    opacity: 0.6,
  },
  chipText: {
    fontSize: 13,
    fontWeight: "500",
    color: "#6B7280",
  },
  chipTextActive: {
    color: "#FFFFFF",
  },
  hint: {
    fontSize: 12.5,
    lineHeight: 17,
    color: "#6B7280",
  },
});
