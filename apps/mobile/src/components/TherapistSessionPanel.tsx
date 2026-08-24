import React, { useCallback, useEffect, useRef, useState } from "react";
import { View, Text, TextInput, TouchableOpacity, StyleSheet } from "react-native";
import type { SavedSession } from "../store/dashboardStore";
import { getSessionNote, putSessionNote } from "../api/therapist";

interface Props {
  session: SavedSession;
  /** Injected for tests; default to the real note calls. */
  loadNote?: (episodeId: string) => Promise<{ text: string }>;
  saveNote?: (episodeId: string, text: string) => Promise<{ text: string }>;
}

/** Turn indexes the phone flagged as escalations on the patient's own turns. */
export function escalationTurns(session: SavedSession): number[] {
  const out: number[] = [];
  session.turns.forEach((t, i) => {
    if (t.escalated) out.push(i);
  });
  return out;
}

/** Named people in the session (from the tone summary — real identities
 *  only), with how the patient sounded with each. */
export function namedPeople(session: SavedSession): {
  name: string;
  theirTurns: number;
  selfTurns: number;
  escalations: number;
}[] {
  const people = session.toneSummary?.people ?? [];
  return people
    .map((p) => ({
      name: p.display_name || p.speaker,
      theirTurns: p.their_turns,
      selfTurns: p.self_turns,
      escalations: p.escalation_count,
    }))
    .filter((p) => Boolean(p.name));
}

/**
 * The therapist-specific view of one shared session, rendered inside
 * SessionDetail beneath the tone summary: escalation markers across the
 * turn timeline, the named people the patient spoke with, and a private
 * notes field (stored server-side, visible only to whoever wrote it).
 */
export default function TherapistSessionPanel({ session, loadNote, saveNote }: Props) {
  const episodeId = session.recordingId ?? session.id;
  const [note, setNote] = useState("");
  const [savedText, setSavedText] = useState("");
  const [status, setStatus] = useState<"loading" | "idle" | "saving" | "saved" | "error">("loading");
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (loadNote ?? getSessionNote)(episodeId)
      .then((n) => {
        if (cancelled) return;
        setNote(n.text);
        setSavedText(n.text);
        setStatus("idle");
      })
      .catch(() => {
        if (!cancelled) setStatus("idle");
      });
    return () => {
      cancelled = true;
    };
  }, [episodeId, loadNote]);

  const save = useCallback(async () => {
    if (note === savedText) return;
    setStatus("saving");
    try {
      const saved = await (saveNote ?? putSessionNote)(episodeId, note);
      if (!mountedRef.current) return;
      setSavedText(saved.text);
      setNote(saved.text);
      setStatus("saved");
    } catch {
      if (mountedRef.current) setStatus("error");
    }
  }, [episodeId, note, savedText, saveNote]);

  const escalations = escalationTurns(session);
  const people = namedPeople(session);
  const dirty = note !== savedText;

  return (
    <View style={styles.wrap} testID="therapist-session-panel">
      <Text style={styles.sectionTitle}>Escalations</Text>
      {session.turns.length > 0 ? (
        <View style={styles.markerRow} testID="escalation-markers">
          {session.turns.map((t, i) => (
            <View
              key={i}
              testID={`escalation-marker-${i}`}
              style={[
                styles.marker,
                t.isSelf ? styles.markerSelf : styles.markerOther,
                t.escalated && styles.markerEscalated,
              ]}
            />
          ))}
        </View>
      ) : null}
      <Text style={styles.meta} testID="escalation-summary">
        {escalations.length === 0
          ? "No escalations flagged on the patient's turns."
          : `${escalations.length} escalation${escalations.length === 1 ? "" : "s"} on the patient's turns (turn${
              escalations.length === 1 ? "" : "s"
            } ${escalations.map((i) => i + 1).join(", ")}).`}
      </Text>

      {people.length > 0 ? (
        <>
          <Text style={styles.sectionTitle}>People in this session</Text>
          {people.map((p) => (
            <Text key={p.name} style={styles.meta} testID={`named-person-${p.name}`}>
              {p.name}: {p.theirTurns} turn{p.theirTurns === 1 ? "" : "s"} · patient spoke to them{" "}
              {p.selfTurns}× · {p.escalations} escalation{p.escalations === 1 ? "" : "s"}
            </Text>
          ))}
        </>
      ) : null}

      <Text style={styles.sectionTitle}>Your notes</Text>
      <Text style={styles.hint}>Private to you — the patient never sees these.</Text>
      <TextInput
        testID="therapist-note-input"
        style={styles.input}
        multiline
        placeholder="Observations, themes to revisit, homework…"
        placeholderTextColor="#9CA3AF"
        value={note}
        onChangeText={(t) => {
          setNote(t);
          if (status === "saved" || status === "error") setStatus("idle");
        }}
        editable={status !== "loading"}
      />
      <View style={styles.noteFooter}>
        <Text style={styles.noteStatus} testID="therapist-note-status">
          {status === "loading"
            ? "Loading…"
            : status === "saving"
              ? "Saving…"
              : status === "saved"
                ? "Saved"
                : status === "error"
                  ? "Couldn’t save — try again"
                  : dirty
                    ? "Unsaved changes"
                    : ""}
        </Text>
        <TouchableOpacity
          testID="therapist-note-save"
          accessibilityRole="button"
          style={[styles.saveButton, (!dirty || status === "saving") && styles.saveDisabled]}
          onPress={save}
          disabled={!dirty || status === "saving"}
        >
          <Text style={styles.saveText}>Save note</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    marginBottom: 24,
    gap: 6,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#1F2937",
    marginTop: 8,
  },
  markerRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 3,
  },
  marker: {
    width: 10,
    height: 10,
    borderRadius: 2,
  },
  markerSelf: {
    backgroundColor: "#BFDBFE",
  },
  markerOther: {
    backgroundColor: "#E5E7EB",
  },
  markerEscalated: {
    backgroundColor: "#EF4444",
  },
  meta: {
    fontSize: 13,
    lineHeight: 18,
    color: "#374151",
  },
  hint: {
    fontSize: 12,
    color: "#6B7280",
    fontStyle: "italic",
  },
  input: {
    minHeight: 96,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 10,
    padding: 10,
    fontSize: 14,
    color: "#1F2937",
    backgroundColor: "#FFFFFF",
    textAlignVertical: "top",
  },
  noteFooter: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  noteStatus: {
    fontSize: 12,
    color: "#6B7280",
  },
  saveButton: {
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 8,
    backgroundColor: "#4A90D9",
  },
  saveDisabled: {
    opacity: 0.5,
  },
  saveText: {
    color: "#FFFFFF",
    fontSize: 13.5,
    fontWeight: "700",
  },
});
