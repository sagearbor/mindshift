import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import { useSessionStore } from "../store/sessionStore";
import RoleSelector from "../components/RoleSelector";
import EmpathySlider from "../components/EmpathySlider";
import SuggestionCard from "../components/SuggestionCard";

export default function SessionScreen() {
  const {
    role,
    empathyLevel,
    turns,
    suggestions,
    loading,
    setRole,
    setEmpathyLevel,
    addTurn,
    fetchSuggestions,
  } = useSessionStore();

  const [speaker, setSpeaker] = useState("");
  const [text, setText] = useState("");

  const handleAddTurn = () => {
    const trimmedText = text.trim();
    const trimmedSpeaker = speaker.trim() || "Speaker";
    if (!trimmedText) return;
    addTurn({ speaker: trimmedSpeaker, text: trimmedText });
    setText("");
  };

  const handleGetSuggestions = () => {
    fetchSuggestions();
  };

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ScrollView
        style={styles.flex}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={styles.heading}>MindShift Session</Text>

        <RoleSelector selectedRole={role} onSelect={setRole} />

        <EmpathySlider value={empathyLevel} onValueChange={setEmpathyLevel} />

        {/* Transcript input */}
        <View style={styles.inputSection}>
          <Text style={styles.sectionTitle}>Transcript</Text>

          <TextInput
            testID="speaker-input"
            style={styles.speakerInput}
            placeholder="Speaker name"
            value={speaker}
            onChangeText={setSpeaker}
            placeholderTextColor="#9CA3AF"
          />

          <View style={styles.turnRow}>
            <TextInput
              testID="text-input"
              style={styles.textInput}
              placeholder="What did they say?"
              value={text}
              onChangeText={setText}
              multiline
              placeholderTextColor="#9CA3AF"
            />
            <TouchableOpacity
              testID="add-turn-button"
              style={styles.addButton}
              onPress={handleAddTurn}
            >
              <Text style={styles.addButtonText}>+</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* Turns list */}
        {turns.length > 0 && (
          <View style={styles.turnsSection}>
            {turns.map((turn, i) => (
              <View key={i} style={styles.turnBubble}>
                <Text style={styles.turnSpeaker}>{turn.speaker}</Text>
                <Text style={styles.turnText}>{turn.text}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Get Suggestions button */}
        <TouchableOpacity
          testID="get-suggestions-button"
          style={[styles.suggestButton, loading && styles.suggestButtonDisabled]}
          onPress={handleGetSuggestions}
          disabled={loading || turns.length === 0}
        >
          {loading ? (
            <ActivityIndicator color="#FFFFFF" />
          ) : (
            <Text style={styles.suggestButtonText}>Get Suggestions</Text>
          )}
        </TouchableOpacity>

        {/* Suggestions */}
        {suggestions.length > 0 && (
          <View style={styles.suggestionsSection}>
            <Text style={styles.sectionTitle}>Suggestions</Text>
            {suggestions.map((s, i) => (
              <SuggestionCard key={i} text={s.text} tone={s.tone} />
            ))}
          </View>
        )}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: {
    flex: 1,
  },
  content: {
    paddingTop: 60,
    paddingBottom: 40,
  },
  heading: {
    fontSize: 24,
    fontWeight: "700",
    textAlign: "center",
    marginBottom: 16,
    color: "#111827",
  },
  inputSection: {
    paddingHorizontal: 16,
    marginTop: 8,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "600",
    marginBottom: 8,
    color: "#1F2937",
  },
  speakerInput: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
    marginBottom: 8,
    color: "#1F2937",
  },
  turnRow: {
    flexDirection: "row",
    alignItems: "flex-end",
    gap: 8,
  },
  textInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
    minHeight: 44,
    color: "#1F2937",
  },
  addButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "#4A90D9",
    alignItems: "center",
    justifyContent: "center",
  },
  addButtonText: {
    color: "#FFFFFF",
    fontSize: 22,
    fontWeight: "600",
  },
  turnsSection: {
    marginTop: 16,
    paddingHorizontal: 16,
  },
  turnBubble: {
    backgroundColor: "#F3F4F6",
    borderRadius: 10,
    padding: 10,
    marginBottom: 6,
  },
  turnSpeaker: {
    fontSize: 12,
    fontWeight: "700",
    color: "#6B7280",
    marginBottom: 2,
  },
  turnText: {
    fontSize: 14,
    color: "#1F2937",
  },
  suggestButton: {
    backgroundColor: "#4A90D9",
    marginHorizontal: 16,
    marginTop: 16,
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
  },
  suggestButtonDisabled: {
    opacity: 0.6,
  },
  suggestButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
  },
  suggestionsSection: {
    marginTop: 20,
    paddingHorizontal: 16,
  },
});
