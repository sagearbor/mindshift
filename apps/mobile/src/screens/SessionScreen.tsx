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

interface SessionScreenProps {
  /** Navigate to the post-session Conversation Dynamics analysis. Provided by
   *  App; optional so the screen still renders standalone (and in tests that
   *  don't exercise navigation). */
  onAnalyzeDynamics?: () => void;
  /** Open the stored-recordings list (media replay). Optional for the same
   *  standalone-render reason. */
  onOpenRecordings?: () => void;
}

export default function SessionScreen({
  onAnalyzeDynamics,
  onOpenRecordings,
}: SessionScreenProps = {}) {
  const {
    role,
    empathyLevel,
    turns,
    suggestions,
    loading,
    setRole,
    setEmpathyLevel,
    addTurn,
    loadTranscript,
    clearTurns,
    fetchSuggestions,
  } = useSessionStore();

  const [speaker, setSpeaker] = useState("");
  const [text, setText] = useState("");
  const [pasted, setPasted] = useState("");

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

        {/* Small link to the stored-recordings list (media replay). */}
        {onOpenRecordings && (
          <TouchableOpacity
            testID="open-recordings-link"
            style={styles.recordingsLink}
            onPress={onOpenRecordings}
          >
            <Text style={styles.recordingsLinkText}>▶ Recordings</Text>
          </TouchableOpacity>
        )}

        <RoleSelector selectedRole={role} onSelect={setRole} />

        <EmpathySlider value={empathyLevel} onValueChange={setEmpathyLevel} />

        {/* Async review: paste or type a whole conversation, then Load it. */}
        <View style={styles.inputSection}>
          <Text style={styles.sectionTitle}>Review a conversation</Text>
          <Text style={styles.hint}>
            Paste or type a conversation and Load it — one line per turn. Start a
            line with a name and colon (e.g. “Me:” / “Her:”) to label who spoke.
            Then set the empathy dial and tap Get Suggestions.
          </Text>
          <TextInput
            testID="paste-transcript-input"
            style={styles.pasteInput}
            placeholder={"Me: I do all the cooking around here...\nHer: I've been buried at work all week..."}
            value={pasted}
            onChangeText={setPasted}
            multiline
            placeholderTextColor="#9CA3AF"
          />
          <View style={styles.pasteRow}>
            <TouchableOpacity
              testID="load-transcript-button"
              style={[
                styles.loadButton,
                !pasted.trim() && styles.suggestButtonDisabled,
              ]}
              onPress={() => {
                if (pasted.trim()) loadTranscript(pasted);
              }}
              disabled={!pasted.trim()}
            >
              <Text style={styles.loadButtonText}>Load conversation</Text>
            </TouchableOpacity>
            {turns.length > 0 && (
              <TouchableOpacity
                testID="clear-turns-button"
                style={styles.clearButton}
                onPress={() => {
                  clearTurns();
                  setPasted("");
                }}
              >
                <Text style={styles.clearButtonText}>Clear</Text>
              </TouchableOpacity>
            )}
          </View>
        </View>

        {/* Or build the transcript one turn at a time */}
        <View style={styles.inputSection}>
          <Text style={styles.sectionTitle}>Or add turns one at a time</Text>

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

        {/* Analyze dynamics: only meaningful once there's enough back-and-forth
            to find patterns (>= 4 turns). Pushes the post-session analysis
            screen; no-op if App didn't wire a handler. */}
        {turns.length >= 4 && onAnalyzeDynamics && (
          <TouchableOpacity
            testID="analyze-dynamics-button"
            style={styles.analyzeButton}
            onPress={onAnalyzeDynamics}
          >
            <Text style={styles.analyzeButtonText}>Analyze dynamics →</Text>
          </TouchableOpacity>
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
  recordingsLink: {
    alignSelf: "center",
    marginBottom: 12,
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#FFFFFF",
  },
  recordingsLinkText: {
    fontSize: 13,
    fontWeight: "600",
    color: "#4A90D9",
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
  hint: {
    fontSize: 12.5,
    lineHeight: 18,
    color: "#6B7280",
    marginBottom: 10,
  },
  pasteInput: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
    minHeight: 110,
    textAlignVertical: "top",
    color: "#1F2937",
    backgroundColor: "#FFFFFF",
  },
  pasteRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginTop: 10,
  },
  loadButton: {
    flex: 1,
    backgroundColor: "#4A90D9",
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: "center",
  },
  loadButtonText: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "600",
  },
  clearButton: {
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#D1D5DB",
  },
  clearButtonText: {
    color: "#6B7280",
    fontSize: 14,
    fontWeight: "600",
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
  analyzeButton: {
    marginHorizontal: 16,
    marginTop: 16,
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
    borderWidth: 1.5,
    borderColor: "#4A90D9",
    backgroundColor: "#EEF2FF",
  },
  analyzeButtonText: {
    color: "#4A90D9",
    fontSize: 16,
    fontWeight: "600",
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
