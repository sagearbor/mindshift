import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  Pressable,
  Switch,
  ScrollView,
  ActivityIndicator,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import * as DocumentPicker from "expo-document-picker";
import { useSessionStore } from "../store/sessionStore";
import { useRecorderStore } from "../store/recorderStore";
import {
  postAnalyzeUpload,
  postAnalyzeUploadChunked,
  postAnalyzeLink,
} from "../api/client";
import type { AnalyzeResult } from "../api/client";
import RoleSelector from "../components/RoleSelector";
import EmpathySlider from "../components/EmpathySlider";
import SuggestionCard from "../components/SuggestionCard";

interface SessionScreenProps {
  /** Navigate to the post-session Conversation Dynamics analysis. Provided by
   *  App; optional so the screen still renders standalone (and in tests that
   *  don't exercise navigation).
   *
   *  `initialData` carries a ready-made analysis (from the recording-upload
   *  flow) so DynamicsScreen can render it without re-fetching; omitted for the
   *  plain "Analyze dynamics" button, which analyzes the store transcript.
   *
   *  `recordingId` is the server-assigned id of a *stored* recording (only set
   *  when the upload flow's consent+store both landed as true); null/omitted
   *  otherwise. Threaded through so DynamicsScreen can offer its Replay
   *  affordance for the recording. */
  onAnalyzeDynamics?: (
    initialData?: AnalyzeResult,
    recordingId?: string | null,
    cameFromRecorder?: boolean,
  ) => void;
  /** Open the stored-recordings list (media replay). Optional for the same
   *  standalone-render reason. */
  onOpenRecordings?: () => void;
  /** Open the in-app video recorder. Optional for the same standalone-render
   *  reason. */
  onRecordVideo?: () => void;
}

/** A file the user picked but hasn't uploaded yet. `file` (web File) is set only
 *  on web; native carries just the `uri`. */
interface PickedRecording {
  uri: string;
  name: string;
  mimeType: string;
  size?: number;
  file?: File;
}

// Files at/under this size take the plain multipart /analyze/upload path; larger
// ones are streamed in chunks (see postAnalyzeUploadChunked) so no single request
// trips the platform's ~32MB body ceiling. Above the hard cap we refuse up front,
// before touching the network, with an honest size message.
const DIRECT_UPLOAD_MAX_BYTES = 20 * 1024 * 1024; // 20 MB
const MAX_UPLOAD_BYTES = 200 * 1024 * 1024; // 200 MB

/** Honest "too big" message naming the actual size and the hard limit. Used both
 *  for the pre-flight refusal (>200MB, no network) and a server 413. */
function sizeLimitMessage(bytes?: number): string {
  const size = formatSize(bytes);
  return `That file is ${size ?? "too large"} — the limit is 200 MB. Try a shorter clip.`;
}

/** Read the HTTP status off an upload/link error — from the `.status` property
 *  the link client attaches, else parsed from the `API error: <status>` message. */
function errorStatus(err: unknown): number {
  const withStatus = err as { status?: number };
  if (typeof withStatus?.status === "number") return withStatus.status;
  const msg = err instanceof Error ? err.message : "";
  return Number(msg.match(/API error: (\d+)/)?.[1] ?? 0);
}

/** Map an upload/link error to an honest, human message. Never invents a result
 *  — every branch tells the user what happened. When the file went up the chunked
 *  path, 413/503 mean something different (the size cap and recording-storage
 *  being off, respectively). For the link path, the server's 422/413 `detail` is
 *  written for users, so it's rendered verbatim. */
function uploadErrorMessage(
  err: unknown,
  opts?: { chunked?: boolean; link?: boolean; sizeBytes?: number },
): string {
  const status = errorStatus(err);
  if (opts?.link) {
    // The server's 422 (not a direct link / blocked URL / Google Photos) and
    // 413 (too large) messages are user-facing — show them verbatim.
    const detail = (err as { detail?: string })?.detail;
    if ((status === 422 || status === 413) && detail) return detail;
  }
  if (opts?.chunked) {
    // The chunked path only exists for large files that need recording storage;
    // a 503 there means that storage isn't enabled, not that analysis is missing.
    if (status === 503)
      return "Large uploads need recording storage enabled on the server.";
    if (status === 413) return sizeLimitMessage(opts.sizeBytes);
  }
  switch (status) {
    case 401:
      return "Please sign in again to analyze a recording.";
    case 413:
      return "That file is too large or too long. Try a shorter clip.";
    case 422:
      return "We couldn’t read that file — no clear speech found. Try another recording.";
    case 429:
      return "Too many requests right now. Give it a moment and try again.";
    case 503:
      return "Recording analysis isn’t configured on the server yet.";
    case 502:
      return "The analysis service is unavailable right now. Please try again.";
    default:
      return "Something went wrong analyzing that recording. Please try again.";
  }
}

/** Human-readable file size for the picked-file line. */
function formatSize(bytes?: number): string | null {
  if (bytes === undefined || bytes <= 0) return null;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function SessionScreen({
  onAnalyzeDynamics,
  onOpenRecordings,
  onRecordVideo,
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
    loadTurns,
    clearTurns,
    fetchSuggestions,
  } = useSessionStore();

  const [speaker, setSpeaker] = useState("");
  const [text, setText] = useState("");
  const [pasted, setPasted] = useState("");

  // --- Analyze-a-recording flow ---
  // Two entry modes share the consent/store/context controls: upload a local
  // file, or hand the server a link to fetch.
  const [mode, setMode] = useState<"file" | "link">("file");
  const [linkUrl, setLinkUrl] = useState("");
  const [picked, setPicked] = useState<PickedRecording | null>(null);
  // True when `picked` is a clip just recorded in-app (handed over via the
  // recorder store). Threaded to onAnalyzeDynamics so the post-analysis
  // "attach HD later" popup only appears for recorder-origin analyses. Reset the
  // moment the user picks a different file or switches to link mode.
  const [cameFromRecorder, setCameFromRecorder] = useState(false);
  const [uploadContext, setUploadContext] = useState("");
  const [uploading, setUploading] = useState(false);
  // Chunked-upload progress as a 0→1 fraction; null on the direct path (which
  // shows a plain spinner instead of a bar).
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  // Consent to have this recording analyzed & (optionally) stored. Unchecked
  // by default — nothing is ever stored without an explicit opt-in.
  const [consent, setConsent] = useState(false);
  // Whether a consenting upload should also be retained for replay. Defaults
  // on so a consenting user gets storage unless they turn it off; irrelevant
  // (and disabled) while consent is unchecked.
  const [storeRecording, setStoreRecording] = useState(true);
  // What the server actually did with the last uploaded file — set only after
  // a successful upload, so the UI reports fact rather than intent.
  const [uploadStored, setUploadStored] = useState<boolean | null>(null);
  const [uploadStorageNote, setUploadStorageNote] = useState<string | null>(null);

  // Consume a freshly-recorded clip handed over from RecordScreen (one-shot):
  // preselect it into the normal upload flow, flag it as recorder-origin, and
  // clear the store so a later remount doesn't re-pick a stale file.
  useEffect(() => {
    const pending = useRecorderStore.getState().pendingFile;
    if (pending) {
      setPicked({
        uri: pending.uri,
        name: pending.name,
        mimeType: pending.mimeType,
        size: pending.size,
      });
      setCameFromRecorder(true);
      setMode("file");
      useRecorderStore.getState().setPendingFile(null);
    }
  }, []);

  const handlePickRecording = async () => {
    setUploadError(null);
    setUploadStored(null);
    setUploadStorageNote(null);
    const result = await DocumentPicker.getDocumentAsync({
      type: ["audio/*", "video/*"],
      copyToCacheDirectory: true,
      multiple: false,
    });
    if (result.canceled) return;
    const asset = result.assets[0];
    if (!asset) return;
    // A manually-picked file is not the in-app recording — drop the flag so the
    // HD-later popup doesn't misfire for it.
    setCameFromRecorder(false);
    setPicked({
      uri: asset.uri,
      name: asset.name,
      mimeType: asset.mimeType ?? "application/octet-stream",
      size: asset.size,
      file: asset.file,
    });
  };

  const handleUploadAnalyze = async () => {
    if (!picked || uploading) return;
    const size = picked.size;
    // Refuse an over-cap file up front — no network call, an honest size message.
    if (size !== undefined && size > MAX_UPLOAD_BYTES) {
      setUploadError(sizeLimitMessage(size));
      return;
    }
    // Anything above the direct-upload ceiling streams in chunks so the platform
    // doesn't reject the body before the server can respond. Size must be known
    // to chunk (we slice against it); an unknown size falls back to direct.
    const useChunked = size !== undefined && size > DIRECT_UPLOAD_MAX_BYTES;
    setUploading(true);
    setUploadProgress(useChunked ? 0 : null);
    setUploadError(null);
    setUploadStored(null);
    setUploadStorageNote(null);
    try {
      // Web hands us a File; native hands us the local URI string.
      const fileArg = Platform.OS === "web" && picked.file ? picked.file : picked.uri;
      const trimmedContext = uploadContext.trim();
      const result = useChunked
        ? await postAnalyzeUploadChunked(
            fileArg,
            picked.name,
            picked.mimeType,
            size as number,
            {
              consent,
              store: storeRecording,
              context: trimmedContext ? trimmedContext : undefined,
              onProgress: setUploadProgress,
            },
          )
        : await postAnalyzeUpload(
            fileArg,
            picked.name,
            picked.mimeType,
            trimmedContext ? trimmedContext : undefined,
            { consent, store: storeRecording },
          );
      // Load the server-produced transcript so the what-if flow (and inspector
      // text) works off the store, then jump straight to the ready-made analysis
      // — no second /analyze round-trip.
      loadTurns(result.turns);
      setUploadStored(result.stored);
      setUploadStorageNote(result.stored ? null : result.storage_note);
      onAnalyzeDynamics?.(result, result.recording_id ?? null, cameFromRecorder);
    } catch (e) {
      setUploadError(uploadErrorMessage(e, { chunked: useChunked, sizeBytes: size }));
    } finally {
      setUploading(false);
      setUploadProgress(null);
    }
  };

  const handleAnalyzeLink = async () => {
    const url = linkUrl.trim();
    if (!url || uploading) return;
    setUploading(true);
    // The server does the fetching/transcription; there's no client-side chunk
    // progress to show, so the button falls back to a spinner (progress null).
    setUploadProgress(null);
    setUploadError(null);
    setUploadStored(null);
    setUploadStorageNote(null);
    try {
      const trimmedContext = uploadContext.trim();
      const result = await postAnalyzeLink(url, {
        consent,
        store: storeRecording,
        context: trimmedContext ? trimmedContext : undefined,
      });
      // Identical handoff to the upload path: hydrate the store transcript, then
      // jump to the ready-made analysis (and thread the recording id if stored).
      loadTurns(result.turns);
      setUploadStored(result.stored);
      setUploadStorageNote(result.stored ? null : result.storage_note);
      onAnalyzeDynamics?.(result, result.recording_id ?? null);
    } catch (e) {
      setUploadError(uploadErrorMessage(e, { link: true }));
    } finally {
      setUploading(false);
      setUploadProgress(null);
    }
  };

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

        {/* Analyze a recording: pick an audio/video file, upload it, and land on
            the ready-made Conversation Dynamics analysis. */}
        <View style={styles.inputSection}>
          <View style={styles.recordingCard}>
            <Text style={styles.sectionTitle}>Analyze a recording</Text>
            <Text style={styles.recordingNote}>
              Without consent to store, we analyze the sound and discard the file.
            </Text>

            {/* Record a fresh video in-app (480p, 10-min cap). Saves to the
                camera roll, then drops straight into this upload flow. */}
            {onRecordVideo && (
              <TouchableOpacity
                testID="record-video-button"
                style={styles.recordVideoButton}
                onPress={onRecordVideo}
                disabled={uploading}
              >
                <Text style={styles.recordVideoButtonText}>⏺ Record video</Text>
              </TouchableOpacity>
            )}

            <Pressable
              testID="consent-checkbox"
              style={styles.consentRow}
              onPress={() => setConsent((v) => !v)}
            >
              <Text style={styles.consentCheckbox}>{consent ? "☑" : "☐"}</Text>
              <Text style={styles.consentLabel}>
                Everyone in this recording knows it was recorded and agrees to
                it being analyzed and stored.
              </Text>
            </Pressable>

            <View style={[styles.storeRow, !consent && styles.storeRowDisabled]}>
              <Text style={styles.storeLabel}>Store for replay</Text>
              <Switch
                testID="store-toggle"
                value={storeRecording}
                onValueChange={setStoreRecording}
                disabled={!consent}
              />
            </View>

            {/* Entry mode: upload a local file, or paste a link the server fetches. */}
            <View style={styles.modeToggle} testID="link-mode-toggle">
              <Pressable
                testID="mode-file-tab"
                style={[styles.modeTab, mode === "file" && styles.modeTabActive]}
                onPress={() => setMode("file")}
                disabled={uploading}
              >
                <Text
                  style={[
                    styles.modeTabText,
                    mode === "file" && styles.modeTabTextActive,
                  ]}
                >
                  Upload file
                </Text>
              </Pressable>
              <Pressable
                testID="mode-link-tab"
                style={[styles.modeTab, mode === "link" && styles.modeTabActive]}
                onPress={() => setMode("link")}
                disabled={uploading}
              >
                <Text
                  style={[
                    styles.modeTabText,
                    mode === "link" && styles.modeTabTextActive,
                  ]}
                >
                  Paste link
                </Text>
              </Pressable>
            </View>

            {mode === "file" && (
              <>
                <TouchableOpacity
                  testID="pick-recording-button"
                  style={styles.pickButton}
                  onPress={() => void handlePickRecording()}
                  disabled={uploading}
                >
                  <Text style={styles.pickButtonText}>
                    {picked ? "Choose a different file" : "Choose a recording"}
                  </Text>
                </TouchableOpacity>

                {picked && (
                  <Text style={styles.pickedFile} testID="picked-file">
                    {picked.name}
                    {formatSize(picked.size) ? `  ·  ${formatSize(picked.size)}` : ""}
                  </Text>
                )}

                {picked && (
                  <>
                    <TextInput
                      testID="recording-context-input"
                      style={styles.recordingContextInput}
                      placeholder="Optional: any context about this conversation"
                      value={uploadContext}
                      onChangeText={setUploadContext}
                      multiline
                      placeholderTextColor="#9CA3AF"
                      editable={!uploading}
                    />
                    <TouchableOpacity
                      testID="upload-analyze-button"
                      style={[
                        styles.uploadButton,
                        uploading && styles.suggestButtonDisabled,
                      ]}
                      onPress={() => void handleUploadAnalyze()}
                      disabled={uploading}
                    >
                      {uploading ? (
                        uploadProgress !== null ? (
                          // Chunked upload: an honest progress bar + percentage.
                          <View style={styles.uploadProgress} testID="upload-progress">
                            <View style={styles.progressTrack}>
                              <View
                                style={[
                                  styles.progressFill,
                                  { width: `${Math.round(uploadProgress * 100)}%` },
                                ]}
                              />
                            </View>
                            <Text style={styles.uploadButtonText}>
                              {`Uploading… ${Math.round(uploadProgress * 100)}%`}
                            </Text>
                          </View>
                        ) : (
                          <View style={styles.uploadingRow}>
                            <ActivityIndicator color="#FFFFFF" />
                            <Text style={styles.uploadButtonText}>Analyzing…</Text>
                          </View>
                        )
                      ) : (
                        <Text style={styles.uploadButtonText}>Upload &amp; analyze</Text>
                      )}
                    </TouchableOpacity>
                  </>
                )}
              </>
            )}

            {mode === "link" && (
              <>
                <TextInput
                  testID="link-input"
                  style={styles.linkInput}
                  placeholder="https://…"
                  value={linkUrl}
                  onChangeText={setLinkUrl}
                  autoCapitalize="none"
                  autoCorrect={false}
                  keyboardType="url"
                  placeholderTextColor="#9CA3AF"
                  editable={!uploading}
                />
                <Text style={styles.linkHelp}>
                  Direct file links, Google Drive share links, and Google Photos
                  share links (single video) all work.
                </Text>
                <TextInput
                  testID="link-context-input"
                  style={styles.recordingContextInput}
                  placeholder="Optional: any context about this conversation"
                  value={uploadContext}
                  onChangeText={setUploadContext}
                  multiline
                  placeholderTextColor="#9CA3AF"
                  editable={!uploading}
                />
                <TouchableOpacity
                  testID="analyze-link-button"
                  style={[
                    styles.uploadButton,
                    (uploading || !linkUrl.trim()) && styles.suggestButtonDisabled,
                  ]}
                  onPress={() => void handleAnalyzeLink()}
                  disabled={uploading || !linkUrl.trim()}
                >
                  {uploading ? (
                    <View style={styles.uploadingRow}>
                      <ActivityIndicator color="#FFFFFF" />
                      <Text style={styles.uploadButtonText}>Analyzing…</Text>
                    </View>
                  ) : (
                    <Text style={styles.uploadButtonText}>Analyze link</Text>
                  )}
                </TouchableOpacity>
              </>
            )}

            {uploadError && (
              <Text style={styles.uploadError} testID="upload-error">
                {uploadError}
              </Text>
            )}

            {uploadStored === true && (
              <Text style={styles.storedNote} testID="stored-note">
                Saved for replay ✓
              </Text>
            )}
            {uploadStored === false && uploadStorageNote && (
              <Text style={styles.storageNote} testID="storage-note">
                {uploadStorageNote}
              </Text>
            )}
          </View>
        </View>

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
            // No arg: the plain button analyzes the store transcript on mount.
            onPress={() => onAnalyzeDynamics()}
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
  recordingCard: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 12,
    padding: 14,
    backgroundColor: "#FFFFFF",
  },
  recordingNote: {
    fontSize: 12.5,
    lineHeight: 18,
    color: "#6B7280",
    marginBottom: 12,
  },
  recordVideoButton: {
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: "center",
    backgroundColor: "#111827",
    marginBottom: 12,
  },
  recordVideoButtonText: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "600",
  },
  consentRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 8,
    marginBottom: 10,
  },
  consentCheckbox: {
    fontSize: 18,
    lineHeight: 20,
    color: "#4A90D9",
  },
  consentLabel: {
    flex: 1,
    fontSize: 12.5,
    lineHeight: 18,
    color: "#374151",
  },
  storeRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 12,
  },
  storeRowDisabled: {
    opacity: 0.5,
  },
  storeLabel: {
    fontSize: 14,
    color: "#1F2937",
    fontWeight: "500",
  },
  storedNote: {
    marginTop: 10,
    fontSize: 13,
    lineHeight: 19,
    color: "#059669",
    fontWeight: "600",
  },
  storageNote: {
    marginTop: 10,
    fontSize: 13,
    lineHeight: 19,
    color: "#6B7280",
  },
  modeToggle: {
    flexDirection: "row",
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 10,
    overflow: "hidden",
    marginBottom: 12,
  },
  modeTab: {
    flex: 1,
    paddingVertical: 10,
    alignItems: "center",
    backgroundColor: "#FFFFFF",
  },
  modeTabActive: {
    backgroundColor: "#EEF2FF",
  },
  modeTabText: {
    fontSize: 14,
    fontWeight: "600",
    color: "#6B7280",
  },
  modeTabTextActive: {
    color: "#4A90D9",
  },
  linkInput: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
    color: "#1F2937",
    backgroundColor: "#FFFFFF",
  },
  linkHelp: {
    fontSize: 12.5,
    lineHeight: 18,
    color: "#6B7280",
    marginTop: 8,
    marginBottom: 4,
  },
  pickButton: {
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: "center",
    borderWidth: 1.5,
    borderColor: "#4A90D9",
    backgroundColor: "#EEF2FF",
  },
  pickButtonText: {
    color: "#4A90D9",
    fontSize: 15,
    fontWeight: "600",
  },
  pickedFile: {
    marginTop: 10,
    fontSize: 13,
    color: "#1F2937",
    fontWeight: "600",
  },
  recordingContextInput: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
    minHeight: 60,
    textAlignVertical: "top",
    color: "#1F2937",
    backgroundColor: "#FFFFFF",
    marginTop: 10,
  },
  uploadButton: {
    backgroundColor: "#4A90D9",
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: "center",
    marginTop: 10,
  },
  uploadingRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  uploadProgress: {
    width: "100%",
    alignItems: "center",
    gap: 6,
  },
  progressTrack: {
    width: "100%",
    height: 6,
    borderRadius: 3,
    backgroundColor: "rgba(255,255,255,0.35)",
    overflow: "hidden",
  },
  progressFill: {
    height: 6,
    borderRadius: 3,
    backgroundColor: "#FFFFFF",
  },
  uploadButtonText: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "600",
  },
  uploadError: {
    marginTop: 10,
    fontSize: 13,
    lineHeight: 19,
    color: "#DC2626",
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
