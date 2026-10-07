import React from "react";
import { View, Text, TextInput, TouchableOpacity, StyleSheet } from "react-native";
import { RELATIONSHIPS, SESSION_CONTEXT_MAX_CHARS, clampSessionContext, type Relationship } from "../live/sessionContext";

/**
 * "About this conversation" — optional, generic by default. Who the user is
 * talking with (one tap; tap again to clear). Nothing is preselected: the
 * coach never assumes a relationship. And what the conversation is about,
 * in the user's own words (capped at SESSION_CONTEXT_MAX_CHARS with a
 * visible counter; remembered on this device; one-tap clear).
 */
export interface ConversationContextPanelProps {
  relationship: Relationship | null;
  onRelationshipChange: (relationship: Relationship | null) => void;
  sessionContext: string;
  onSessionContextChange: (text: string) => void;
  onClearSessionContext: () => void;
}

export default function ConversationContextPanel({
  relationship,
  onRelationshipChange,
  sessionContext,
  onSessionContextChange,
  onClearSessionContext,
}: ConversationContextPanelProps) {
  const used = sessionContext.length;
  return (
    <View style={styles.container} testID="conversation-context-panel">
      <Text style={styles.label}>What&apos;s this conversation about? (optional)</Text>
      <TextInput
        testID="session-context-input"
        style={styles.input}
        multiline
        maxLength={SESSION_CONTEXT_MAX_CHARS}
        value={sessionContext}
        onChangeText={(t) => onSessionContextChange(clampSessionContext(t))}
        placeholder="e.g. 'Meeting with my boss to ask for a raise; this year I shipped…'"
        placeholderTextColor="#9CA3AF"
        accessibilityLabel="What's this conversation about? Optional."
      />
      <View style={styles.contextFooter}>
        <Text
          testID="session-context-counter"
          style={[styles.counter, used >= SESSION_CONTEXT_MAX_CHARS && styles.counterFull]}
        >
          {used} / {SESSION_CONTEXT_MAX_CHARS}
        </Text>
        {used > 0 ? (
          <TouchableOpacity testID="session-context-clear" onPress={onClearSessionContext} accessibilityRole="button">
            <Text style={styles.clear}>Clear</Text>
          </TouchableOpacity>
        ) : null}
      </View>
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
  input: {
    minHeight: 64,
    maxHeight: 160,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    padding: 8,
    fontSize: 14,
    color: "#111827",
    backgroundColor: "#FFFFFF",
    textAlignVertical: "top",
  },
  contextFooter: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginTop: 4, marginBottom: 8 },
  counter: { fontSize: 12, color: "#6B7280" },
  counterFull: { color: "#B91C1C", fontWeight: "600" },
  clear: { fontSize: 13, color: "#2563EB", fontWeight: "600" },
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
