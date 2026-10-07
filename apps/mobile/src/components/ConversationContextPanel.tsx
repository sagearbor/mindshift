import React from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import { RELATIONSHIPS, type Relationship } from "../live/sessionContext";

/**
 * "About this conversation" — optional, generic by default. Who the user is
 * talking with (one tap; tap again to clear). Nothing is preselected: the
 * coach never assumes a relationship.
 */
export interface ConversationContextPanelProps {
  relationship: Relationship | null;
  onRelationshipChange: (relationship: Relationship | null) => void;
}

export default function ConversationContextPanel({
  relationship,
  onRelationshipChange,
}: ConversationContextPanelProps) {
  return (
    <View style={styles.container} testID="conversation-context-panel">
      <Text style={styles.label}>Talking with (optional)</Text>
      <View style={styles.chips}>
        {RELATIONSHIPS.map((r) => {
          const selected = relationship === r;
          return (
            <TouchableOpacity
              key={r}
              testID={`relationship-${r}`}
              style={[styles.chip, selected && styles.chipSelected]}
              onPress={() => onRelationshipChange(selected ? null : r)}
              accessibilityRole="button"
              accessibilityState={{ selected }}
            >
              <Text style={[styles.chipText, selected && styles.chipTextSelected]}>{r}</Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { paddingHorizontal: 16, paddingVertical: 6 },
  label: { fontSize: 13, fontWeight: "600", color: "#374151", marginBottom: 6 },
  chips: { flexDirection: "row", flexWrap: "wrap", gap: 6 },
  chip: {
    paddingVertical: 5,
    paddingHorizontal: 12,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#F9FAFB",
  },
  chipSelected: { backgroundColor: "#4A90D9", borderColor: "#4A90D9" },
  chipText: { fontSize: 13, color: "#374151" },
  chipTextSelected: { color: "#FFFFFF" },
});
