import React from "react";
import { View, Text, Pressable, StyleSheet } from "react-native";

/**
 * The relationship-context picker for the Analyze flow: one OPTIONAL tap
 * answers "who's in this conversation?" so the analysis can be framed
 * correctly (partners vs. coworkers read very differently). Deliberately a
 * single chip row — never a wall of form fields — and never required: with
 * nothing selected, no relationship context is sent and the analysis infers
 * it from the conversation. Tapping the selected chip deselects it.
 */

export type RelationshipId =
  | "partners"
  | "parent_child"
  | "coworkers"
  | "friends"
  | "just_me";

export interface RelationshipOption {
  id: RelationshipId;
  label: string;
  /** The plain-English sentence sent to the analyze API's free-text `context`
   *  field. Written as a fact the user asserted (they tapped it), never a
   *  guess. */
  context: string;
}

export const RELATIONSHIP_OPTIONS: RelationshipOption[] = [
  {
    id: "partners",
    label: "Partners",
    context: "The people in this conversation are romantic partners.",
  },
  {
    id: "parent_child",
    label: "Parent & child",
    context: "This conversation is between a parent and their child.",
  },
  {
    id: "coworkers",
    label: "Coworkers",
    context: "The people in this conversation are coworkers.",
  },
  {
    id: "friends",
    label: "Friends",
    context: "The people in this conversation are friends.",
  },
  {
    id: "just_me",
    label: "Just me",
    context: "This recording is one person speaking alone, reflecting by themselves.",
  },
];

/** The context sentence for a relationship id (used when composing the
 *  analyze request's `context` field). Null — nothing selected — yields ""
 *  so no relationship sentence is ever sent in infer mode. */
export function relationshipContext(id: RelationshipId | null): string {
  if (id === null) return "";
  const opt = RELATIONSHIP_OPTIONS.find((o) => o.id === id);
  // The union type makes a miss impossible at compile time; the fallback keeps
  // runtime honest if the list ever drifts.
  return opt ? opt.context : "";
}

interface RelationshipPickerProps {
  /** The selected relationship, or null for "not chosen — infer it". */
  value: RelationshipId | null;
  /** Called with the tapped id, or null when the user taps the selected chip
   *  to deselect it (back to infer mode). */
  onSelect: (id: RelationshipId | null) => void;
  /** Disable taps while an upload/analysis is in flight. */
  disabled?: boolean;
}

export default function RelationshipPicker({
  value,
  onSelect,
  disabled,
}: RelationshipPickerProps) {
  return (
    <View style={styles.container} testID="relationship-picker">
      <Text style={styles.label}>Who’s in this conversation?</Text>
      <Text style={styles.optionalHint} testID="relationship-optional-hint">
        Optional — we’ll figure it out if you skip.
      </Text>
      <View style={styles.chipRow}>
        {RELATIONSHIP_OPTIONS.map((opt) => {
          const selected = opt.id === value;
          return (
            <Pressable
              key={opt.id}
              testID={`relationship-${opt.id}`}
              accessibilityRole="button"
              accessibilityState={{ selected }}
              style={[styles.chip, selected && styles.chipSelected]}
              // Tapping the selected chip deselects (null = infer mode).
              onPress={() => onSelect(selected ? null : opt.id)}
              disabled={disabled}
            >
              <Text
                style={[styles.chipText, selected && styles.chipTextSelected]}
              >
                {opt.label}
              </Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginBottom: 14,
  },
  label: {
    fontSize: 14,
    fontWeight: "600",
    color: "#1F2937",
    marginBottom: 2,
  },
  optionalHint: {
    fontSize: 12.5,
    lineHeight: 18,
    color: "#6B7280",
    marginBottom: 8,
  },
  chipRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  chip: {
    // Big tap target: users may be mid-conflict and stressed.
    minHeight: 44,
    justifyContent: "center",
    paddingHorizontal: 16,
    borderRadius: 22,
    borderWidth: 1.5,
    borderColor: "#D1D5DB",
    backgroundColor: "#FFFFFF",
  },
  chipSelected: {
    borderColor: "#4A90D9",
    backgroundColor: "#EEF2FF",
  },
  chipText: {
    fontSize: 14,
    fontWeight: "600",
    color: "#6B7280",
  },
  chipTextSelected: {
    color: "#4A90D9",
  },
});
