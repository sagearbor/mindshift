import React, { useCallback, useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import { postShare, deleteShare } from "../api/client";
import type { RecordingShare } from "../api/client";

// House colors (match ReplayScreen).
const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";
const DANGER = "#DC2626";
const GOOD = "#1B7A4B";

interface Props {
  recordingId: string;
  /** The recording's current shares (owner view), from the detail read. */
  initialShares: RecordingShare[];
}

/** Honest message for a share failure. The server writes user-facing details for
 *  the expected cases (no account / self-share / bad email) — surface those
 *  verbatim; only fall back to a mapped message for the generic statuses. */
function humanizeShareError(err: unknown): string {
  const e = err as { detail?: string; status?: number; message?: string };
  if (typeof e?.detail === "string" && e.detail) return e.detail;
  const status = e?.status;
  if (status === 401) return "Please sign in again to share.";
  if (status === 503) return "Sharing isn’t available right now.";
  return "Couldn’t share right now — please try again.";
}

/**
 * Owner-only "Share with…" affordance for a stored recording: enter an account
 * email to grant READ-ONLY access, see a success confirmation, and manage the
 * list of current shares (each removable with ✕). Rendered by ReplayScreen only
 * when the caller OWNS the recording (never in read-only shared mode). Every
 * error is honest — the server's no-account/self-share/bad-email details are
 * shown verbatim; nothing is faked on failure.
 */
export default function RecordingShareManager({
  recordingId,
  initialShares,
}: Props) {
  const [shares, setShares] = useState<RecordingShare[]>(initialShares);
  const [open, setOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successEmail, setSuccessEmail] = useState<string | null>(null);
  // uid currently being removed (drives its row spinner) + any remove error.
  const [removingUid, setRemovingUid] = useState<string | null>(null);
  const [removeError, setRemoveError] = useState<string | null>(null);

  const submit = useCallback(async () => {
    const trimmed = email.trim();
    if (!trimmed || submitting) return;
    setSubmitting(true);
    setError(null);
    setSuccessEmail(null);
    try {
      const res = await postShare(recordingId, trimmed);
      setShares(res.shares);
      setSuccessEmail(trimmed);
      setEmail("");
    } catch (e) {
      setError(humanizeShareError(e));
    } finally {
      setSubmitting(false);
    }
  }, [email, submitting, recordingId]);

  const remove = useCallback(
    async (uid: string) => {
      setRemovingUid(uid);
      setRemoveError(null);
      try {
        await deleteShare(recordingId, uid);
        setShares((prev) => prev.filter((s) => s.uid !== uid));
      } catch {
        setRemoveError("Couldn’t remove that share — please try again.");
      } finally {
        setRemovingUid(null);
      }
    },
    [recordingId],
  );

  return (
    <View style={styles.section} testID="share-section">
      {!open ? (
        <TouchableOpacity
          testID="share-open-button"
          style={styles.openButton}
          onPress={() => {
            setError(null);
            setOpen(true);
          }}
        >
          <Text style={styles.openButtonText}>
            {shares.length > 0 ? "Manage sharing" : "Share with…"}
          </Text>
        </TouchableOpacity>
      ) : (
        <View>
          <Text style={styles.help}>
            Enter the email of a MindShift account to give them read-only access to
            this recording.
          </Text>
          <TextInput
            testID="share-email-input"
            style={styles.input}
            placeholder="name@example.com"
            value={email}
            onChangeText={setEmail}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="email-address"
            placeholderTextColor="#9CA3AF"
            editable={!submitting}
            onSubmitEditing={() => void submit()}
          />
          <View style={styles.buttonRow}>
            <TouchableOpacity
              testID="share-cancel"
              style={styles.cancel}
              onPress={() => {
                setOpen(false);
                setError(null);
              }}
              disabled={submitting}
            >
              <Text style={styles.cancelText}>Done</Text>
            </TouchableOpacity>
            <TouchableOpacity
              testID="share-submit"
              style={[
                styles.submit,
                (submitting || !email.trim()) && styles.disabled,
              ]}
              onPress={() => void submit()}
              disabled={submitting || !email.trim()}
            >
              {submitting ? (
                <ActivityIndicator color="#FFFFFF" />
              ) : (
                <Text style={styles.submitText}>Share</Text>
              )}
            </TouchableOpacity>
          </View>
          {error && (
            <Text style={styles.error} testID="share-error">
              {error}
            </Text>
          )}
          {successEmail && (
            <Text style={styles.success} testID="share-success">
              Shared with {successEmail}
            </Text>
          )}
        </View>
      )}

      {/* Current shares — always visible so the owner sees who has access. */}
      {shares.length > 0 && (
        <View style={styles.list} testID="share-list">
          <Text style={styles.listTitle}>Shared with</Text>
          {shares.map((s) => (
            <View key={s.uid} style={styles.row} testID={`share-row-${s.uid}`}>
              <Text style={styles.rowEmail} numberOfLines={1}>
                {s.email}
              </Text>
              <TouchableOpacity
                testID={`share-remove-${s.uid}`}
                onPress={() => void remove(s.uid)}
                disabled={removingUid === s.uid}
                hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
              >
                {removingUid === s.uid ? (
                  <ActivityIndicator size="small" color={DANGER} />
                ) : (
                  <Text style={styles.remove}>✕</Text>
                )}
              </TouchableOpacity>
            </View>
          ))}
          {removeError && (
            <Text style={styles.error} testID="share-remove-error">
              {removeError}
            </Text>
          )}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  section: { marginTop: 16 },
  openButton: {
    alignSelf: "flex-start",
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: PRIMARY,
    backgroundColor: "#EEF2FF",
  },
  openButtonText: { fontSize: 14, fontWeight: "700", color: PRIMARY },
  help: { fontSize: 12.5, lineHeight: 18, color: MUTED, marginBottom: 8 },
  input: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
    color: INK,
    backgroundColor: "#FFFFFF",
  },
  buttonRow: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 10,
    marginTop: 10,
  },
  cancel: {
    paddingVertical: 10,
    paddingHorizontal: 18,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#D1D5DB",
  },
  cancelText: { color: MUTED, fontSize: 14, fontWeight: "600" },
  submit: {
    paddingVertical: 10,
    paddingHorizontal: 22,
    borderRadius: 8,
    backgroundColor: PRIMARY,
    alignItems: "center",
    justifyContent: "center",
    minWidth: 90,
  },
  disabled: { opacity: 0.6 },
  submitText: { color: "#FFFFFF", fontSize: 14, fontWeight: "600" },
  error: { marginTop: 10, fontSize: 13, lineHeight: 19, color: DANGER },
  success: { marginTop: 10, fontSize: 13, fontWeight: "600", color: GOOD },
  list: {
    marginTop: 14,
    padding: 12,
    borderRadius: 10,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  listTitle: {
    fontSize: 11,
    fontWeight: "700",
    color: MUTED,
    textTransform: "uppercase",
    letterSpacing: 0.5,
    marginBottom: 8,
  },
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 6,
  },
  rowEmail: { flex: 1, fontSize: 14, color: INK },
  remove: { fontSize: 16, color: DANGER, fontWeight: "700", paddingHorizontal: 6 },
});
