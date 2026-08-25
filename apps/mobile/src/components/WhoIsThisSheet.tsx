import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";

import {
  enrollPersonFromRecording,
  patchSpeakerLabels,
  type EnrollFromRecordingResult,
  type PatchSpeakerLabelsResult,
  type VoicePerson,
} from "../api/client";
import {
  SELF_PERSON_ID,
  enrollRefusalReason,
  enrollRefusalTitle,
  personDisplayName,
  slugifyPersonId,
  sortPeople,
} from "../utils/people";

const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";
const DANGER = "#DC2626";
const GOOD = "#0F9D58";

/** What the mid-call flow is told when a person is picked or typed. */
export interface LiveLabelChoice {
  personId: string;
  displayName: string;
  isSelf: boolean;
  /** True for "New person…" (the flow may create + enroll them). */
  isNew: boolean;
}

export interface WhoIsThisSheetProps {
  visible: boolean;
  /** The stored recording being relabeled. Not needed in live mode. */
  recordingId?: string;
  /**
   * LIVE / mid-call mode: instead of PATCHing a stored recording, hand the
   * choice to the running session (the hook binds the cluster to the
   * person, learns the voice from the session's audio, tells the server)
   * and show the returned outcome text in the "done" stage. The "Remember
   * this voice" stage is skipped — the live flow decides that itself.
   */
  onLiveLabel?: (choice: LiveLabelChoice) => Promise<{ text: string }>;
  /** The raw diarized speaker id being labeled ("Speaker B"). */
  speaker: string;
  /** What the row currently shows for this speaker (for the header). */
  currentLabel: string;
  /** The person this speaker is already attached to, if any. */
  currentPersonId?: string | null;
  /** The account's enrolled people (GET /voice/people). */
  people: VoicePerson[];
  /** Whether the server kept audio for this recording — "Remember this
   *  voice" is offered only then (a live session has none). */
  hasAudio: boolean;
  onClose: () => void;
  /** A label was saved — the parent re-labels every surface from the
   *  server's resolved response. Not called in live mode. */
  onLabeled?: (result: PatchSpeakerLabelsResult) => void;
  /** A voice was learned — the parent refreshes its people list. */
  onEnrolled?: (result: EnrollFromRecordingResult) => void;
}

type Stage =
  | { kind: "pick" }
  | { kind: "new" }
  | { kind: "remember"; personId: string; name: string; isNew: boolean }
  | { kind: "done"; text: string };

/**
 * "Who is this?" — name a recording's speaker as a PERSON, once.
 *
 * Lists the account's enrolled people ("You" first) plus "New person…".
 * Picking one relabels this recording through the existing manual-label
 * endpoint (with the person id → the `manual-person` rung, so Growth and
 * the therapist rows follow the person across sessions) and then offers
 * "Remember this voice" — learn the voice from this recording so the app
 * recognizes them on the phone and in every later analysis. The server's
 * three honest refusals (too little speech / sounds like someone else /
 * no audio on a live session) are shown inline, verbatim, with the label
 * kept either way: naming this recording never depends on the voiceprint.
 */
export default function WhoIsThisSheet({
  visible,
  recordingId = "",
  onLiveLabel,
  speaker,
  currentLabel,
  currentPersonId,
  people,
  hasAudio,
  onClose,
  onLabeled,
  onEnrolled,
}: WhoIsThisSheetProps) {
  const [stage, setStage] = useState<Stage>({ kind: "pick" });
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ title: string; message: string } | null>(null);

  // Reset whenever the sheet is (re)opened for a speaker.
  useEffect(() => {
    if (visible) {
      setStage({ kind: "pick" });
      setDraft("");
      setBusy(false);
      setError(null);
    }
  }, [visible, speaker, recordingId]);

  const sorted = useMemo(() => sortPeople(people), [people]);

  const label = useCallback(
    async (name: string, personId: string | null) => {
      const result = await patchSpeakerLabels(
        recordingId,
        { [speaker]: name },
        personId ? { [speaker]: personId } : undefined,
      );
      onLabeled?.(result);
      return result;
    },
    [recordingId, speaker, onLabeled],
  );

  const pickPerson = useCallback(
    async (p: VoicePerson) => {
      if (busy) return;
      setBusy(true);
      setError(null);
      const name = personDisplayName(p);
      try {
        if (onLiveLabel) {
          // Mid-call: the session binds the voice to this person right away.
          const outcome = await onLiveLabel({
            personId: p.person_id,
            displayName: name,
            isSelf: p.is_self === true || p.person_id === SELF_PERSON_ID,
            isNew: false,
          });
          setStage({ kind: "done", text: outcome.text });
          return;
        }
        await label(name, p.person_id);
        if (hasAudio) {
          setStage({ kind: "remember", personId: p.person_id, name, isNew: false });
        } else {
          setStage({
            kind: "done",
            text: `${name} is labeled in this session. The app already knows ${name}’s voice from earlier — no audio was kept for this live session to learn more from.`,
          });
        }
      } catch (e) {
        const err = e as Error & { status?: number; detail?: string };
        setError({
          title: "Couldn’t save that name",
          message:
            err.status === 422
              ? err.detail || err.message
              : "Please check your connection and try again.",
        });
      } finally {
        setBusy(false);
      }
    },
    [busy, hasAudio, label, onLiveLabel],
  );

  const saveNewName = useCallback(async () => {
    const name = draft.trim();
    if (!name || busy) return;
    setBusy(true);
    setError(null);
    try {
      if (onLiveLabel) {
        // Mid-call: the session creates the person (and learns the voice
        // from what it has heard so far) — the outcome text says how far.
        const outcome = await onLiveLabel({
          personId: slugifyPersonId(name, people.map((p) => p.person_id)),
          displayName: name,
          isSelf: false,
          isNew: true,
        });
        setStage({ kind: "done", text: outcome.text });
        return;
      }
      // A free-text name labels THIS recording now; the person is created
      // only when a voice is remembered (a person is a voiceprint).
      await label(name, null);
      const newId = slugifyPersonId(name, people.map((p) => p.person_id));
      if (hasAudio) {
        setStage({ kind: "remember", personId: newId, name, isNew: true });
      } else {
        setStage({
          kind: "done",
          text: `${name} is labeled in this session. Live sessions keep no audio, so to recognize ${name} next time add them under People with a 20-second recording.`,
        });
      }
    } catch (e) {
      const err = e as Error & { status?: number; detail?: string };
      setError({
        title: "Couldn’t save that name",
        message:
          err.status === 422
            ? err.detail || err.message
            : "Please check your connection and try again.",
      });
    } finally {
      setBusy(false);
    }
  }, [draft, busy, label, people, hasAudio, onLiveLabel]);

  const remember = useCallback(async () => {
    if (stage.kind !== "remember" || busy) return;
    setBusy(true);
    setError(null);
    try {
      const result = await enrollPersonFromRecording(
        stage.personId,
        recordingId,
        speaker,
        stage.isNew ? stage.name : undefined,
      );
      // Attach the (possibly brand-new) person to the manual label so the
      // recording carries the person id, not just the name.
      await label(stage.name, stage.personId);
      onEnrolled?.(result);
      const secs = Math.round(result.seconds);
      setStage({
        kind: "done",
        text:
          `Learned ${secs} s of ${stage.name}’s voice (${result.enroll_count} sample` +
          `${result.enroll_count === 1 ? "" : "s"} now). ${stage.name} will be recognized ` +
          "in future sessions — on the phone and in analysis. Stored as a numeric " +
          "signature, not the audio.",
      });
    } catch (e) {
      const reason = enrollRefusalReason(e);
      setError({ title: enrollRefusalTitle(reason.kind), message: reason.message });
    } finally {
      setBusy(false);
    }
  }, [stage, busy, recordingId, speaker, label, onEnrolled]);

  const clearName = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await label("", null);
      onClose();
    } catch {
      setError({
        title: "Couldn’t clear that name",
        message: "Please check your connection and try again.",
      });
    } finally {
      setBusy(false);
    }
  }, [busy, label, onClose]);

  const errorBlock = error ? (
    <View style={styles.errorBox} testID="who-error">
      <Text style={styles.errorTitle}>{error.title}</Text>
      <Text style={styles.errorText}>{error.message}</Text>
    </View>
  ) : null;

  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onClose}
    >
      <View style={styles.backdrop}>
        <View style={styles.sheet} testID="who-sheet">
          <Text style={styles.title}>Who is this?</Text>
          <Text style={styles.subtitle} testID="who-subtitle">
            {`Currently “${currentLabel}”`}
          </Text>

          {stage.kind === "pick" ? (
            <ScrollView style={styles.list} keyboardShouldPersistTaps="handled">
              {sorted.map((p) => {
                const name = personDisplayName(p);
                const isCurrent = currentPersonId === p.person_id;
                return (
                  <TouchableOpacity
                    key={p.person_id}
                    testID={`who-person-${p.person_id}`}
                    accessibilityRole="button"
                    style={[styles.personRow, isCurrent ? styles.personRowCurrent : null]}
                    disabled={busy}
                    onPress={() => void pickPerson(p)}
                  >
                    <Text style={styles.personName}>{name}</Text>
                    <Text style={styles.personMeta}>
                      {p.is_self ? "your enrolled voice" : `${p.enroll_count} voice sample${p.enroll_count === 1 ? "" : "s"}`}
                      {isCurrent ? " · current" : ""}
                    </Text>
                  </TouchableOpacity>
                );
              })}
              {sorted.length === 0 ? (
                <Text style={styles.note} testID="who-no-people">
                  No one is enrolled yet — name this person below and the app can
                  learn their voice from this recording.
                </Text>
              ) : null}
              <TouchableOpacity
                testID="who-new-person"
                accessibilityRole="button"
                style={styles.personRow}
                disabled={busy}
                onPress={() => {
                  setError(null);
                  setStage({ kind: "new" });
                }}
              >
                <Text style={[styles.personName, { color: PRIMARY }]}>New person…</Text>
                <Text style={styles.personMeta}>type a name</Text>
              </TouchableOpacity>
              {currentLabel !== speaker && !onLiveLabel ? (
                <TouchableOpacity
                  testID="who-clear"
                  accessibilityRole="button"
                  style={styles.personRow}
                  disabled={busy}
                  onPress={() => void clearName()}
                >
                  <Text style={[styles.personName, { color: MUTED }]}>Not sure — clear the name</Text>
                </TouchableOpacity>
              ) : null}
              {errorBlock}
            </ScrollView>
          ) : null}

          {stage.kind === "new" ? (
            <View>
              <TextInput
                testID="who-name-input"
                style={styles.input}
                value={draft}
                onChangeText={setDraft}
                placeholder="Their name (e.g. Mom)"
                placeholderTextColor="#9CA3AF"
                autoFocus
                maxLength={40}
                editable={!busy}
                returnKeyType="done"
                onSubmitEditing={() => void saveNewName()}
              />
              {errorBlock}
              <TouchableOpacity
                testID="who-save-name"
                accessibilityRole="button"
                style={[styles.primaryButton, !draft.trim() ? styles.disabled : null]}
                disabled={busy || !draft.trim()}
                onPress={() => void saveNewName()}
              >
                {busy ? (
                  <ActivityIndicator size="small" color="#FFFFFF" />
                ) : (
                  <Text style={styles.primaryButtonText}>Save name</Text>
                )}
              </TouchableOpacity>
            </View>
          ) : null}

          {stage.kind === "remember" ? (
            <View testID="who-remember-stage">
              <Text style={styles.note}>
                {`This speaker is now “${stage.name}” in this recording.`}
              </Text>
              <Text style={styles.note}>
                {`Remember ${stage.name}’s voice from this recording so the app recognizes ` +
                  "them next time — on the phone during a session and in every later " +
                  "analysis? It stores a numeric voice signature, never the audio."}
              </Text>
              {errorBlock}
              <TouchableOpacity
                testID="who-remember"
                accessibilityRole="button"
                style={styles.primaryButton}
                disabled={busy}
                onPress={() => void remember()}
              >
                {busy ? (
                  <ActivityIndicator size="small" color="#FFFFFF" />
                ) : (
                  <Text style={styles.primaryButtonText}>
                    {error ? "Try again" : "Remember this voice"}
                  </Text>
                )}
              </TouchableOpacity>
              <TouchableOpacity
                testID="who-skip-remember"
                accessibilityRole="button"
                style={styles.secondaryButton}
                disabled={busy}
                onPress={onClose}
              >
                <Text style={styles.secondaryButtonText}>
                  {error ? "Keep the name anyway" : "Just this recording"}
                </Text>
              </TouchableOpacity>
            </View>
          ) : null}

          {stage.kind === "done" ? (
            <View testID="who-done-stage">
              <Text style={styles.doneText} testID="who-done-text">
                {stage.text}
              </Text>
              <TouchableOpacity
                testID="who-done"
                accessibilityRole="button"
                style={styles.primaryButton}
                onPress={onClose}
              >
                <Text style={styles.primaryButtonText}>Done</Text>
              </TouchableOpacity>
            </View>
          ) : null}

          {stage.kind === "pick" || stage.kind === "new" ? (
            <TouchableOpacity
              testID="who-cancel"
              accessibilityRole="button"
              style={styles.cancelButton}
              disabled={busy}
              onPress={stage.kind === "new" ? () => setStage({ kind: "pick" }) : onClose}
            >
              <Text style={styles.cancelText}>{stage.kind === "new" ? "Back" : "Cancel"}</Text>
            </TouchableOpacity>
          ) : null}
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    justifyContent: "flex-end",
    backgroundColor: "rgba(17, 24, 39, 0.45)",
  },
  sheet: {
    backgroundColor: "#FFFFFF",
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    padding: 20,
    paddingBottom: 28,
    maxHeight: "85%",
  },
  title: { fontSize: 18, fontWeight: "700", color: INK, marginBottom: 2 },
  subtitle: { fontSize: 13, color: MUTED, marginBottom: 12 },
  list: { flexGrow: 0 },
  personRow: {
    paddingVertical: 12,
    borderTopWidth: 1,
    borderTopColor: "#F3F4F6",
  },
  personRowCurrent: { backgroundColor: "#F0F7FF" },
  personName: { fontSize: 16, fontWeight: "600", color: INK },
  personMeta: { fontSize: 12, color: MUTED, marginTop: 2 },
  note: { fontSize: 13.5, lineHeight: 19, color: MUTED, marginBottom: 10 },
  doneText: { fontSize: 14, lineHeight: 20, color: GOOD, fontWeight: "600", marginBottom: 12 },
  input: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
    color: INK,
    marginBottom: 12,
  },
  primaryButton: {
    backgroundColor: PRIMARY,
    borderRadius: 10,
    minHeight: 44,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 6,
  },
  primaryButtonText: { color: "#FFFFFF", fontSize: 15, fontWeight: "700" },
  secondaryButton: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 10,
    minHeight: 44,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 8,
  },
  secondaryButtonText: { color: INK, fontSize: 14, fontWeight: "600" },
  cancelButton: { minHeight: 40, alignItems: "center", justifyContent: "center", marginTop: 8 },
  cancelText: { color: MUTED, fontSize: 14, fontWeight: "600" },
  disabled: { opacity: 0.5 },
  errorBox: {
    backgroundColor: "#FEF2F2",
    borderRadius: 8,
    padding: 10,
    marginVertical: 8,
  },
  errorTitle: { fontSize: 13.5, fontWeight: "700", color: DANGER, marginBottom: 2 },
  errorText: { fontSize: 13, lineHeight: 18, color: INK },
});
