import React, { useCallback, useMemo, useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
} from "react-native";

import {
  patchSpeakerLabels,
  type RecordingTurn,
  type PatchSpeakerLabelsResult,
} from "../api/client";
import { getSpeakerColor } from "../utils/speakerColors";
import {
  speakerLabel,
  labelProvenanceNote,
  type SpeakerLabels,
} from "../utils/speakerLabels";

const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";
const DANGER = "#DC2626";

interface SpeakerNamingProps {
  recordingId: string;
  turns: RecordingTurn[];
  /** Effective per-speaker labels (name → voice → id), already reflecting any
   *  manual overrides — what every surface renders. Used here for the CURRENT
   *  display name + provenance note per row. */
  speakerLabels?: SpeakerLabels;
  /** Raw manual name map ({id: name}) — prefills a row's editor with the name
   *  the user typed (not the inferred label), and marks which rows are manual. */
  manualLabels: Record<string, string>;
  /** Lift a successful save so the parent can update EVERY label surface (legend,
   *  talk-share, inspector, enrollment) from the server's resolved response. */
  onSaved: (result: PatchSpeakerLabelsResult) => void;
  /** A save 404'd — the server has no manual-labels route (older build). The
   *  parent hides the whole affordance rather than leave a button that can't work. */
  onUnsupported?: () => void;
}

/**
 * "Name the speakers" — manually label who each diarized speaker is, for THIS
 * recording. One listen (tap a dash to hear a voice) then two taps: ✎ name → type
 * → Save. The save PATCHes /recordings/{id}/speaker-labels (append-only merge) and
 * the resolved response re-labels the whole screen through `onSaved`.
 *
 * Deliberately distinct from the "This is me" voice-enrollment card that sits
 * beside it: naming labels this ONE recording; enrollment teaches your voice for
 * every future one. The helper copy states that difference plainly.
 *
 * Clearing: editing a name to empty and saving clears the manual label, so the
 * inferred one (a detected name, a voice label, or the raw id) comes back — the
 * server confirms this in its response, which flows through `onSaved`.
 */
export default function SpeakerNaming({
  recordingId,
  turns,
  speakerLabels,
  manualLabels,
  onSaved,
  onUnsupported,
}: SpeakerNamingProps) {
  // The speaker whose inline editor is open (canonical id), the draft text, and
  // per-save busy/error state (scoped to the open row).
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Distinct speakers in first-appearance order (matches the transcript/legend).
  const speakers = useMemo(() => {
    const seen: string[] = [];
    for (const t of turns) {
      if (t.speaker && !seen.includes(t.speaker)) seen.push(t.speaker);
    }
    return seen;
  }, [turns]);

  const openEditor = useCallback(
    (speaker: string) => {
      setError(null);
      // Prefill with the raw manual name if one exists; otherwise leave it blank
      // with the "Who is this?" placeholder (we never seed the inferred guess —
      // the field is for the user's own name, and blank keeps clearing obvious).
      setDraft(manualLabels[speaker] ?? "");
      setEditing(speaker);
    },
    [manualLabels],
  );

  const cancelEditor = useCallback(() => {
    setEditing(null);
    setDraft("");
    setError(null);
  }, []);

  const handleSave = useCallback(
    async (speaker: string) => {
      if (saving) return;
      const name = draft.trim();
      setSaving(true);
      setError(null);
      try {
        // Append-only merge: send just this speaker. An empty string clears it
        // (restoring the inferred label) — the server echoes the result.
        const result = await patchSpeakerLabels(recordingId, {
          [speaker]: name,
        });
        onSaved(result);
        setEditing(null);
        setDraft("");
      } catch (e) {
        const err = e as Error & { status?: number; detail?: string };
        if (err.status === 404) {
          // No manual-labels route on this (older) server — hand off to the
          // parent to hide the affordance entirely rather than show a dead button.
          onUnsupported?.();
          return;
        }
        if (err.status === 422) {
          // Unknown speaker id — the server writes this for the user.
          setError(
            err.detail || err.message || "That speaker isn’t in this recording.",
          );
        } else {
          setError("Couldn’t save that name. Please try again.");
        }
      } finally {
        setSaving(false);
      }
    },
    [draft, saving, recordingId, onSaved, onUnsupported],
  );

  if (speakers.length === 0) return null;

  return (
    <View style={styles.card} testID="speaker-naming">
      <Text style={styles.sectionTitle}>Name the speakers</Text>
      <Text style={styles.subtitle}>
        Not sure who’s who? Tap a dash on the chart to hear that voice, then name
        them here. This labels just this recording — the “This is me” card below
        teaches MindShift your own voice for every recording.
      </Text>

      {speakers.map((speaker) => {
        const source = speakerLabels?.[speaker]?.label_source;
        const provenance = labelProvenanceNote(source);
        const display = speakerLabel(speaker, speakerLabels);
        const isEditing = editing === speaker;
        return (
          <View key={speaker} style={styles.speakerRow} testID={`name-row-${speaker}`}>
            {isEditing ? (
              <View style={styles.editWrap}>
                <View style={styles.labelWrap}>
                  <View
                    style={[styles.dot, { backgroundColor: getSpeakerColor(speaker) }]}
                  />
                  <TextInput
                    testID={`name-input-${speaker}`}
                    style={styles.input}
                    value={draft}
                    onChangeText={setDraft}
                    placeholder="Who is this?"
                    placeholderTextColor="#9CA3AF"
                    autoFocus
                    maxLength={40}
                    editable={!saving}
                    onSubmitEditing={() => void handleSave(speaker)}
                    returnKeyType="done"
                  />
                </View>
                <View style={styles.editButtons}>
                  <TouchableOpacity
                    testID={`name-save-${speaker}`}
                    onPress={() => void handleSave(speaker)}
                    disabled={saving}
                    accessibilityRole="button"
                  >
                    {saving ? (
                      <ActivityIndicator size="small" color={PRIMARY} />
                    ) : (
                      <Text style={styles.saveText}>Save</Text>
                    )}
                  </TouchableOpacity>
                  <TouchableOpacity
                    testID={`name-cancel-${speaker}`}
                    onPress={cancelEditor}
                    disabled={saving}
                    accessibilityRole="button"
                  >
                    <Text style={styles.cancelText}>Cancel</Text>
                  </TouchableOpacity>
                </View>
              </View>
            ) : (
              <>
                <View style={styles.labelWrap}>
                  <View
                    style={[styles.dot, { backgroundColor: getSpeakerColor(speaker) }]}
                  />
                  <View style={styles.nameCol}>
                    <Text style={styles.speakerName} numberOfLines={1}>
                      {display}
                    </Text>
                    {provenance ? (
                      <Text
                        style={styles.provenance}
                        testID={`name-provenance-${speaker}`}
                      >
                        {provenance}
                      </Text>
                    ) : null}
                  </View>
                </View>
                <TouchableOpacity
                  testID={`name-edit-${speaker}`}
                  style={styles.editButton}
                  onPress={() => openEditor(speaker)}
                  accessibilityRole="button"
                  accessibilityLabel={`Name ${display}`}
                >
                  <Text style={styles.editButtonText}>✎ Name</Text>
                </TouchableOpacity>
              </>
            )}
          </View>
        );
      })}

      {error ? (
        <Text style={styles.error} testID="name-error">
          {error}
        </Text>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: INK,
    marginBottom: 6,
  },
  subtitle: {
    fontSize: 13,
    lineHeight: 19,
    color: MUTED,
    marginBottom: 12,
  },
  speakerRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: "#F3F4F6",
  },
  labelWrap: {
    flexDirection: "row",
    alignItems: "center",
    flexShrink: 1,
    flex: 1,
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 10,
  },
  nameCol: {
    flexShrink: 1,
  },
  speakerName: {
    fontSize: 15,
    fontWeight: "600",
    color: INK,
  },
  provenance: {
    fontSize: 11.5,
    color: MUTED,
    fontStyle: "italic",
    marginTop: 1,
  },
  editButton: {
    minHeight: 40,
    paddingHorizontal: 14,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: PRIMARY,
    alignItems: "center",
    justifyContent: "center",
    marginLeft: 8,
  },
  editButtonText: {
    fontSize: 14,
    fontWeight: "700",
    color: PRIMARY,
  },
  editWrap: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
    fontSize: 15,
    color: INK,
    backgroundColor: "#FFFFFF",
  },
  editButtons: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  saveText: { fontSize: 15, fontWeight: "700", color: PRIMARY },
  cancelText: { fontSize: 15, fontWeight: "600", color: MUTED },
  error: {
    marginTop: 10,
    fontSize: 13,
    lineHeight: 18,
    color: DANGER,
  },
});
