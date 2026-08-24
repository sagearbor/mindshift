import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";

import {
  deleteVoicePerson,
  enrollPersonFromRecording,
  getRecording,
  listRecordings,
  listVoicePeople,
  renameVoicePerson,
  type RecordingSummary,
  type VoicePerson,
} from "../api/client";
import VoiceTrainingFlow from "../components/VoiceTrainingFlow";
import {
  SELF_PERSON_ID,
  enrollRefusalReason,
  enrollRefusalTitle,
  personDisplayName,
  personSummary,
  slugifyPersonId,
  sortPeople,
} from "../utils/people";

const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";
const DANGER = "#DC2626";
const GOOD = "#0F9D58";

interface PeopleScreenProps {
  onBack: () => void;
  /** Open a recording's replay (a sample's source). Optional. */
  onOpenReplay?: (recordingId: string) => void;
}

/** Which person an add-sample flow is for, and how the voice is captured. */
interface AddFlow {
  personId: string;
  displayName: string;
  isNew: boolean;
  method: "choose" | "record" | "pick-recording" | "pick-speaker";
  recording?: { id: string; title: string; speakers: string[] };
}

/**
 * People — everyone whose voice the app knows, in one place.
 *
 * Lists enrolled people ("You" pinned first) with sample count + seconds of
 * learned speech; rename (partners), delete (confirm), and "Add person":
 * type a name, then either read the four guided phrases (VoiceTrainingFlow
 * for that person) or pick a stored recording + the speaker who is them
 * (enroll-from-recording, with the server's honest refusals shown inline).
 *
 * Nothing here fabricates state: the list is exactly GET /voice/people, and
 * a server without voice ID says so instead of showing an empty "add" flow.
 */
export default function PeopleScreen({ onBack, onOpenReplay }: PeopleScreenProps) {
  const [people, setPeople] = useState<VoicePerson[] | null>(null);
  const [available, setAvailable] = useState<boolean | null>(null);
  const [storageEnabled, setStorageEnabled] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [renaming, setRenaming] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [rowError, setRowError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const [add, setAdd] = useState<AddFlow | null>(null);
  const [newName, setNewName] = useState("");
  const [recordings, setRecordings] = useState<RecordingSummary[] | null>(null);
  const [addBusy, setAddBusy] = useState(false);
  const [addError, setAddError] = useState<{ title: string; message: string } | null>(null);
  const [addSuccess, setAddSuccess] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await listVoicePeople();
      setPeople(res.people);
      setAvailable(res.available);
      setStorageEnabled(res.storage_enabled);
      setLoadError(null);
    } catch {
      setPeople((prev) => prev ?? []);
      setLoadError("Couldn’t load your people. Check your connection and pull to retry.");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const sorted = useMemo(() => sortPeople(people ?? []), [people]);
  const takenIds = useMemo(() => (people ?? []).map((p) => p.person_id), [people]);

  // --- rename ---------------------------------------------------------------
  const startRename = useCallback((p: VoicePerson) => {
    setRowError(null);
    setRenaming(p.person_id);
    setRenameDraft(p.display_name ?? "");
  }, []);

  const saveRename = useCallback(async () => {
    const id = renaming;
    const name = renameDraft.trim();
    if (!id || !name) return;
    setBusyId(id);
    setRowError(null);
    try {
      const updated = await renameVoicePerson(id, name);
      setPeople((prev) =>
        (prev ?? []).map((p) => (p.person_id === id ? { ...p, ...updated } : p)),
      );
      setRenaming(null);
    } catch (e) {
      const err = e as Error & { detail?: string };
      setRowError(err.detail || "Couldn’t rename. Please try again.");
    } finally {
      setBusyId(null);
    }
  }, [renaming, renameDraft]);

  // --- delete ---------------------------------------------------------------
  const confirmDelete = useCallback((p: VoicePerson) => {
    const name = personDisplayName(p);
    Alert.alert(
      p.is_self ? "Forget my voice?" : `Forget ${name}’s voice?`,
      p.is_self
        ? "This permanently deletes the numeric voice signature used to label you “You”. Your recordings are not affected."
        : `This permanently deletes the numeric voice signature the app uses to recognize ${name}. Recordings already labeled “${name}” keep that name.`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Forget",
          style: "destructive",
          onPress: () => {
            setBusyId(p.person_id);
            setRowError(null);
            deleteVoicePerson(p.person_id)
              .then(() => {
                setPeople((prev) => (prev ?? []).filter((x) => x.person_id !== p.person_id));
              })
              .catch(() => setRowError(`Couldn’t forget ${name}. Please try again.`))
              .finally(() => setBusyId(null));
          },
        },
      ],
    );
  }, []);

  // --- add person / add sample ---------------------------------------------
  const beginAddSample = useCallback((p: VoicePerson) => {
    setAddError(null);
    setAddSuccess(null);
    setAdd({
      personId: p.person_id,
      displayName: personDisplayName(p),
      isNew: false,
      method: "choose",
    });
  }, []);

  const beginAddPerson = useCallback(() => {
    setAddError(null);
    setAddSuccess(null);
    setNewName("");
    setAdd({ personId: "", displayName: "", isNew: true, method: "choose" });
  }, []);

  const chooseMethod = useCallback(
    async (method: "record" | "pick-recording") => {
      if (!add) return;
      let flow = add;
      if (add.isNew) {
        const name = newName.trim();
        if (!name) return;
        flow = { ...add, displayName: name, personId: slugifyPersonId(name, takenIds) };
      }
      setAddError(null);
      setAdd({ ...flow, method });
      if (method === "pick-recording" && recordings === null) {
        try {
          const list = await listRecordings();
          setRecordings(list.filter((r) => r.has_analysis && r.media_type !== "none"));
        } catch {
          setRecordings([]);
          setAddError({
            title: "Couldn’t load recordings",
            message: "Check your connection and try again.",
          });
        }
      }
    },
    [add, newName, takenIds, recordings],
  );

  const pickRecording = useCallback(
    async (r: RecordingSummary) => {
      if (!add) return;
      setAddBusy(true);
      setAddError(null);
      try {
        const detail = await getRecording(r.id);
        const speakers: string[] = [];
        for (const t of detail.turns) {
          if (t.speaker && !speakers.includes(t.speaker)) speakers.push(t.speaker);
        }
        setAdd({
          ...add,
          method: "pick-speaker",
          recording: { id: r.id, title: r.title || r.filename, speakers },
        });
      } catch {
        setAddError({
          title: "Couldn’t open that recording",
          message: "Check your connection and try again.",
        });
      } finally {
        setAddBusy(false);
      }
    },
    [add],
  );

  const enrollFromSpeaker = useCallback(
    async (speaker: string) => {
      if (!add || !add.recording) return;
      setAddBusy(true);
      setAddError(null);
      try {
        const result = await enrollPersonFromRecording(
          add.personId,
          add.recording.id,
          speaker,
          add.isNew ? add.displayName : undefined,
        );
        setAddSuccess(
          `Learned ${Math.round(result.seconds)} s of ${add.displayName}’s voice from “${add.recording.title}” ` +
            `(${result.enroll_count} sample${result.enroll_count === 1 ? "" : "s"} now).`,
        );
        setAdd(null);
        await refresh();
      } catch (e) {
        const reason = enrollRefusalReason(e);
        setAddError({ title: enrollRefusalTitle(reason.kind), message: reason.message });
      } finally {
        setAddBusy(false);
      }
    },
    [add, refresh],
  );

  const handleTrained = useCallback(
    (count: number) => {
      if (add) {
        setAddSuccess(
          `${add.displayName}’s voice profile now blends ${count} sample${count === 1 ? "" : "s"}.`,
        );
      }
      setAdd(null);
      void refresh();
    },
    [add, refresh],
  );

  // --- render ---------------------------------------------------------------
  const renderAdd = () => {
    if (!add) return null;
    const nameOk = add.isNew ? newName.trim().length > 0 : true;
    return (
      <View style={styles.card} testID="people-add-flow">
        <Text style={styles.cardTitle}>
          {add.isNew ? "Add a person" : `Add a sample for ${add.displayName}`}
        </Text>

        {add.method === "choose" ? (
          <>
            {add.isNew ? (
              <TextInput
                testID="people-new-name"
                style={styles.input}
                value={newName}
                onChangeText={setNewName}
                placeholder="Their name (e.g. Mom)"
                placeholderTextColor="#9CA3AF"
                maxLength={40}
                autoFocus
              />
            ) : null}
            <Text style={styles.note}>
              How should the app learn this voice? It stores a numeric voice
              signature, never the audio.
            </Text>
            <TouchableOpacity
              testID="people-method-record"
              accessibilityRole="button"
              style={[styles.primaryButton, !nameOk ? styles.disabled : null]}
              disabled={!nameOk}
              onPress={() => void chooseMethod("record")}
            >
              <Text style={styles.primaryButtonText}>Record 20 seconds now</Text>
            </TouchableOpacity>
            <TouchableOpacity
              testID="people-method-recording"
              accessibilityRole="button"
              style={[styles.secondaryButton, !nameOk ? styles.disabled : null]}
              disabled={!nameOk}
              onPress={() => void chooseMethod("pick-recording")}
            >
              <Text style={styles.secondaryButtonText}>From a recording I’ve stored</Text>
            </TouchableOpacity>
          </>
        ) : null}

        {add.method === "record" ? (
          <VoiceTrainingFlow
            person={
              add.personId === SELF_PERSON_ID
                ? undefined
                : { personId: add.personId, displayName: add.displayName }
            }
            onDone={handleTrained}
            onCancel={() => setAdd({ ...add, method: "choose" })}
          />
        ) : null}

        {add.method === "pick-recording" ? (
          <View testID="people-pick-recording">
            <Text style={styles.note}>
              {`Which recording has ${add.displayName} in it? Live sessions keep no audio, so they aren’t listed.`}
            </Text>
            {recordings === null ? (
              <ActivityIndicator size="small" color={PRIMARY} />
            ) : recordings.length === 0 ? (
              <Text style={styles.note} testID="people-no-recordings">
                No analyzed recordings with audio yet.
              </Text>
            ) : (
              recordings.map((r) => (
                <TouchableOpacity
                  key={r.id}
                  testID={`people-recording-${r.id}`}
                  accessibilityRole="button"
                  style={styles.listRow}
                  disabled={addBusy}
                  onPress={() => void pickRecording(r)}
                >
                  <Text style={styles.listRowTitle} numberOfLines={1}>
                    {r.title || r.filename}
                  </Text>
                  <Text style={styles.listRowMeta}>
                    {new Date(r.created_at).toLocaleDateString()}
                  </Text>
                </TouchableOpacity>
              ))
            )}
          </View>
        ) : null}

        {add.method === "pick-speaker" && add.recording ? (
          <View testID="people-pick-speaker">
            <Text style={styles.note}>
              {`Which speaker in “${add.recording.title}” is ${add.displayName}?`}
            </Text>
            {add.recording.speakers.map((sp) => (
              <TouchableOpacity
                key={sp}
                testID={`people-speaker-${sp}`}
                accessibilityRole="button"
                style={styles.listRow}
                disabled={addBusy}
                onPress={() => void enrollFromSpeaker(sp)}
              >
                <Text style={styles.listRowTitle}>{sp}</Text>
              </TouchableOpacity>
            ))}
            {onOpenReplay ? (
              <TouchableOpacity
                testID="people-open-replay"
                accessibilityRole="button"
                onPress={() => onOpenReplay(add.recording!.id)}
              >
                <Text style={styles.link}>Not sure? Listen to the recording first</Text>
              </TouchableOpacity>
            ) : null}
          </View>
        ) : null}

        {addBusy ? <ActivityIndicator size="small" color={PRIMARY} style={styles.spinner} /> : null}
        {addError ? (
          <View style={styles.errorBox} testID="people-add-error">
            <Text style={styles.errorTitle}>{addError.title}</Text>
            <Text style={styles.errorText}>{addError.message}</Text>
          </View>
        ) : null}

        {add.method !== "record" ? (
          <TouchableOpacity
            testID="people-add-cancel"
            accessibilityRole="button"
            style={styles.cancelButton}
            disabled={addBusy}
            onPress={() =>
              add.method === "choose" ? setAdd(null) : setAdd({ ...add, method: "choose" })
            }
          >
            <Text style={styles.cancelText}>{add.method === "choose" ? "Cancel" : "Back"}</Text>
          </TouchableOpacity>
        ) : null}
      </View>
    );
  };

  return (
    <ScrollView
      style={styles.flex}
      contentContainerStyle={styles.content}
      testID="people-screen"
      keyboardShouldPersistTaps="handled"
    >
      <TouchableOpacity testID="people-back" onPress={onBack} style={styles.backButton}>
        <Text style={styles.backButtonText}>Back</Text>
      </TouchableOpacity>
      <Text style={styles.heading}>People</Text>
      <Text style={styles.subheading}>
        Everyone whose voice MindShift recognizes. Name someone once — from a
        recording or by recording them — and they’re labeled in every later
        session, on the phone and in analysis.
      </Text>

      {people === null && !loadError ? (
        <ActivityIndicator size="small" color={PRIMARY} testID="people-loading" />
      ) : null}
      {loadError ? (
        <Text style={styles.errorText} testID="people-load-error">
          {loadError}
        </Text>
      ) : null}
      {available === false ? (
        <View style={styles.card} testID="people-unavailable">
          <Text style={styles.note}>
            {storageEnabled
              ? "Voice recognition isn’t available on this server (its voice model isn’t installed), so people can’t be enrolled yet."
              : "Recording storage is off on this server, so there is nowhere to keep voiceprints."}
          </Text>
        </View>
      ) : null}

      {sorted.map((p) => {
        const name = personDisplayName(p);
        const isRenaming = renaming === p.person_id;
        return (
          <View key={p.person_id} style={styles.card} testID={`people-row-${p.person_id}`}>
            {isRenaming ? (
              <View style={styles.renameRow}>
                <TextInput
                  testID={`people-rename-input-${p.person_id}`}
                  style={[styles.input, styles.renameInput]}
                  value={renameDraft}
                  onChangeText={setRenameDraft}
                  maxLength={60}
                  autoFocus
                  onSubmitEditing={() => void saveRename()}
                  returnKeyType="done"
                />
                <TouchableOpacity
                  testID={`people-rename-save-${p.person_id}`}
                  accessibilityRole="button"
                  disabled={busyId === p.person_id || !renameDraft.trim()}
                  onPress={() => void saveRename()}
                >
                  <Text style={styles.link}>Save</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  testID={`people-rename-cancel-${p.person_id}`}
                  accessibilityRole="button"
                  onPress={() => setRenaming(null)}
                >
                  <Text style={styles.cancelText}>Cancel</Text>
                </TouchableOpacity>
              </View>
            ) : (
              <View style={styles.rowHeader}>
                <View style={styles.rowText}>
                  <Text style={styles.rowTitle} testID={`people-name-${p.person_id}`}>
                    {name}
                  </Text>
                  <Text style={styles.rowSub} testID={`people-summary-${p.person_id}`}>
                    {personSummary(p)}
                    {p.is_self ? " · that’s you" : ""}
                  </Text>
                </View>
                {busyId === p.person_id ? (
                  <ActivityIndicator size="small" color={MUTED} />
                ) : null}
              </View>
            )}
            <View style={styles.actions}>
              <TouchableOpacity
                testID={`people-add-sample-${p.person_id}`}
                accessibilityRole="button"
                style={styles.actionButton}
                onPress={() => beginAddSample(p)}
              >
                <Text style={styles.actionText}>Add sample</Text>
              </TouchableOpacity>
              {!p.is_self && !isRenaming ? (
                <TouchableOpacity
                  testID={`people-rename-${p.person_id}`}
                  accessibilityRole="button"
                  style={styles.actionButton}
                  onPress={() => startRename(p)}
                >
                  <Text style={styles.actionText}>Rename</Text>
                </TouchableOpacity>
              ) : null}
              <TouchableOpacity
                testID={`people-delete-${p.person_id}`}
                accessibilityRole="button"
                style={styles.actionButton}
                onPress={() => confirmDelete(p)}
              >
                <Text style={[styles.actionText, { color: DANGER }]}>
                  {p.is_self ? "Forget my voice" : "Forget"}
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        );
      })}
      {rowError ? (
        <Text style={styles.errorText} testID="people-row-error">
          {rowError}
        </Text>
      ) : null}

      {people !== null && sorted.length === 0 && available !== false ? (
        <Text style={styles.note} testID="people-empty">
          Nobody yet. Add yourself first (so the app knows which voice is you),
          then the people you talk with most.
        </Text>
      ) : null}

      {addSuccess ? (
        <Text style={styles.success} testID="people-add-success">
          {addSuccess}
        </Text>
      ) : null}

      {add ? (
        renderAdd()
      ) : available !== false ? (
        <TouchableOpacity
          testID="people-add-person"
          accessibilityRole="button"
          style={styles.primaryButton}
          onPress={beginAddPerson}
        >
          <Text style={styles.primaryButtonText}>Add person</Text>
        </TouchableOpacity>
      ) : null}

      <Text style={styles.footnote}>
        Renaming applies to new labels; recordings already labeled keep the name
        they were given. Forgetting someone deletes their voice signature for
        real — nothing is kept.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  content: { paddingTop: 60, paddingBottom: 40, paddingHorizontal: 16 },
  backButton: {
    alignSelf: "flex-start",
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: "#F3F4F6",
    marginBottom: 12,
  },
  backButtonText: { fontSize: 14, fontWeight: "600", color: PRIMARY },
  heading: { fontSize: 22, fontWeight: "700", color: "#111827", marginBottom: 4 },
  subheading: { fontSize: 13.5, lineHeight: 19, color: MUTED, marginBottom: 16 },
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  cardTitle: { fontSize: 16, fontWeight: "700", color: INK, marginBottom: 8 },
  rowHeader: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  rowText: { flex: 1 },
  rowTitle: { fontSize: 16, fontWeight: "700", color: INK },
  rowSub: { fontSize: 12.5, color: MUTED, marginTop: 2 },
  actions: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 10 },
  actionButton: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#D1D5DB",
  },
  actionText: { fontSize: 13, fontWeight: "600", color: PRIMARY },
  renameRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  renameInput: { flex: 1, marginBottom: 0 },
  input: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    fontSize: 15,
    color: INK,
    marginBottom: 10,
    backgroundColor: "#FFFFFF",
  },
  note: { fontSize: 13.5, lineHeight: 19, color: MUTED, marginBottom: 10 },
  footnote: { fontSize: 12, lineHeight: 17, color: "#9CA3AF", marginTop: 16 },
  success: { fontSize: 13.5, lineHeight: 19, color: GOOD, fontWeight: "600", marginBottom: 10 },
  link: { fontSize: 14, fontWeight: "700", color: PRIMARY },
  listRow: {
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: "#F3F4F6",
  },
  listRowTitle: { fontSize: 15, fontWeight: "600", color: INK },
  listRowMeta: { fontSize: 12, color: MUTED, marginTop: 2 },
  spinner: { marginVertical: 8 },
  primaryButton: {
    backgroundColor: PRIMARY,
    borderRadius: 10,
    minHeight: 44,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 6,
    marginBottom: 8,
  },
  primaryButtonText: { color: "#FFFFFF", fontSize: 15, fontWeight: "700" },
  secondaryButton: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 10,
    minHeight: 44,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 8,
  },
  secondaryButtonText: { color: INK, fontSize: 14, fontWeight: "600" },
  cancelButton: { minHeight: 40, alignItems: "center", justifyContent: "center" },
  cancelText: { color: MUTED, fontSize: 14, fontWeight: "600" },
  disabled: { opacity: 0.5 },
  errorBox: { backgroundColor: "#FEF2F2", borderRadius: 8, padding: 10, marginVertical: 8 },
  errorTitle: { fontSize: 13.5, fontWeight: "700", color: DANGER, marginBottom: 2 },
  errorText: { fontSize: 13, lineHeight: 18, color: DANGER },
});
