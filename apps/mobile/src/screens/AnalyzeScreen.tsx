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
import { useAnalyzeStore } from "../store/analyzeStore";
import {
  postAnalyzeUpload,
  postAnalyzeUploadChunkedJob,
  postAnalyzeLink,
  postAnalyzeLinkJob,
  getAnalyzeJob,
} from "../api/client";
import type {
  AnalyzeResult,
  AnalyzeJobState,
  JobStatus,
  UploadAnalyzeResult,
} from "../api/client";
import RelationshipPicker, {
  relationshipContext,
} from "../components/RelationshipPicker";

/**
 * The "Analyze a Conversation" mode — everything after-the-fact, in one place:
 * record a video in-app, upload a file, or paste a link, with the relationship
 * picker framing the analysis and the past-recordings list one tap away. The
 * upload/link/job machinery here moved verbatim from the old SessionScreen
 * (which is now the text-only review tools, reachable via "Work with text").
 */
interface AnalyzeScreenProps {
  /** Return to Home. */
  onBack?: () => void;
  /** Navigate to the post-session Conversation Dynamics analysis. Same contract
   *  as the old SessionScreen prop: `initialData` carries a ready-made analysis
   *  so DynamicsScreen renders without re-fetching; `recordingId` is the id of
   *  a *stored* recording (consent+store both true) or null; `cameFromRecorder`
   *  gates the attach-HD-later popup for in-app recordings. */
  onAnalyzeDynamics?: (
    initialData?: AnalyzeResult,
    recordingId?: string | null,
    cameFromRecorder?: boolean,
  ) => void;
  /** Open the stored-recordings list (media replay). */
  onOpenRecordings?: () => void;
  /** Open the in-app video recorder. */
  onRecordVideo?: () => void;
  /** Open the text tools (paste/type a transcript, get suggestions). */
  onOpenTextTools?: () => void;
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

/** Pick the right message for an upload/link failure: a job failure/stall
 *  already carries a user-written message (surface it verbatim), otherwise map
 *  the HTTP error. */
function jobOrUploadErrorMessage(
  err: unknown,
  opts?: { chunked?: boolean; link?: boolean; sizeBytes?: number },
): string {
  if (err instanceof JobFailedError) return err.message;
  return uploadErrorMessage(err, opts);
}

/** Human-readable file size for the picked-file line. */
function formatSize(bytes?: number): string | null {
  if (bytes === undefined || bytes <= 0) return null;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// --- Async analysis jobs (submit-and-poll) ----------------------------------
// A link download or a chunked-upload completion runs server-side as a job we
// poll; the multi-minute synchronous request it replaces was routinely killed by
// Android backgrounding. The result survives because it's decoupled from a
// single long-lived connection.

const JOB_POLL_INTERVAL_MS = 3000;

/** A job that ended in an honest failure/stall — its message is written for the
 *  user (the server's own detail, or the stalled note), so it's surfaced
 *  verbatim rather than mapped through {@link uploadErrorMessage}. */
class JobFailedError extends Error {
  readonly jobError = true;
}

/** Human stage label for the progress card. */
function jobStageLabel(status: JobStatus): string {
  switch (status) {
    case "queued":
      return "Queued…";
    case "downloading":
      return "Downloading…";
    case "transcribing":
      return "Transcribing…";
    case "analyzing":
      return "Analyzing…";
    case "storing":
      return "Saving…";
    case "done":
      return "Done";
    default:
      return "Working…";
  }
}

/**
 * Rough remaining-seconds estimate from the current stage + the (audio)
 * duration, or null when we can't estimate yet (duration unknown, which is the
 * case until decode/transcribe finishes). Deliberately approximate — the UI
 * labels it an estimate. The factors are the fraction of the audio-length's
 * worth of processing still ahead at each stage, plus a small fixed floor.
 */
function estimateRemainingSeconds(
  status: JobStatus,
  durationSeconds: number | null,
): number | null {
  if (durationSeconds == null || durationSeconds <= 0) return null;
  const factor: Record<string, number> = {
    queued: 0.9,
    downloading: 0.9,
    transcribing: 0.7,
    analyzing: 0.4,
    storing: 0.15,
  };
  const f = factor[status];
  if (f == null) return null;
  return Math.max(5, Math.round(durationSeconds * f + 5));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Poll a job to a terminal state, calling `onState` after each poll so the UI
 * can render staged progress. Returns the result on `done`; throws a
 * {@link JobFailedError} (message written for the user) on `failed`/`stalled`.
 * Stateless polling — it survives brief app backgrounding naturally (on resume
 * the next poll simply catches up). A transient poll error is retried a few
 * times before giving up, so one dropped request doesn't abort a live job.
 */
async function pollJobToDone(
  jobId: string,
  onState: (state: AnalyzeJobState) => void,
): Promise<UploadAnalyzeResult> {
  let transientErrors = 0;
  for (;;) {
    let state: AnalyzeJobState;
    try {
      state = await getAnalyzeJob(jobId);
    } catch (e) {
      // A 404 means the job is truly gone (or never existed) — stop. Any other
      // transport hiccup is retried a handful of times.
      const status = errorStatus(e);
      if (status === 404 || transientErrors >= 3) throw e;
      transientErrors += 1;
      await sleep(JOB_POLL_INTERVAL_MS);
      continue;
    }
    transientErrors = 0;
    onState(state);
    if (state.status === "done") {
      if (!state.result) {
        throw new JobFailedError(
          "The analysis finished but returned no result. Please try again.",
        );
      }
      return state.result;
    }
    if (state.status === "failed") {
      throw new JobFailedError(
        state.error ?? "The analysis failed. Please try again.",
      );
    }
    if (state.status === "stalled") {
      throw new JobFailedError(
        state.progress_note ??
          "The analysis appears to have stalled — please try again.",
      );
    }
    await sleep(JOB_POLL_INTERVAL_MS);
  }
}

export default function AnalyzeScreen({
  onBack,
  onAnalyzeDynamics,
  onOpenRecordings,
  onRecordVideo,
  onOpenTextTools,
}: AnalyzeScreenProps = {}) {
  const { loadTurns } = useSessionStore();

  // Relationship context: one-tap picker, remembered across mounts (the smart
  // default is simply the last choice; first run defaults to partners).
  const relationship = useAnalyzeStore((s) => s.relationship);
  const setRelationship = useAnalyzeStore((s) => s.setRelationship);

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
  // The most recent poll of an in-flight analysis job — drives the staged
  // progress card. Null when no job is running (small direct uploads, or before
  // a job is created / after it finishes).
  const [jobState, setJobState] = useState<AnalyzeJobState | null>(null);

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

  /** The `context` string sent to the analyze API: the tapped relationship as
   *  a plain sentence, then any free text the user typed. The relationship is
   *  a fact the user asserted with the picker — never a guess. */
  const composedContext = (): string | undefined => {
    const parts = [relationshipContext(relationship), uploadContext.trim()].filter(
      Boolean,
    );
    return parts.length > 0 ? parts.join(" ") : undefined;
  };

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
    setJobState(null);
    try {
      // Web hands us a File; native hands us the local URI string.
      const fileArg = Platform.OS === "web" && picked.file ? picked.file : picked.uri;
      const context = composedContext();
      let result: UploadAnalyzeResult;
      if (useChunked) {
        // Stream the parts (byte-progress bar), then complete as a JOB we poll —
        // the multi-minute synchronous /complete it replaces was routinely
        // killed by Android backgrounding. On an old server / storage off the
        // client transparently falls back to synchronous complete.
        const outcome = await postAnalyzeUploadChunkedJob(
          fileArg,
          picked.name,
          picked.mimeType,
          size as number,
          {
            consent,
            store: storeRecording,
            context,
            onProgress: setUploadProgress,
          },
        );
        if ("result" in outcome) {
          result = outcome.result; // synchronous fallback already produced it
        } else {
          // Upload finished → swap the byte-progress bar for the staged job card.
          setUploadProgress(null);
          result = await pollJobToDone(outcome.jobId, setJobState);
        }
      } else {
        // Small files (<=20MB) stay a single fast synchronous request.
        result = await postAnalyzeUpload(
          fileArg,
          picked.name,
          picked.mimeType,
          context,
          { consent, store: storeRecording },
        );
      }
      // Load the server-produced transcript so the what-if flow (and inspector
      // text) works off the store, then jump straight to the ready-made analysis
      // — no second /analyze round-trip.
      loadTurns(result.turns);
      setUploadStored(result.stored);
      setUploadStorageNote(result.stored ? null : result.storage_note);
      onAnalyzeDynamics?.(result, result.recording_id ?? null, cameFromRecorder);
    } catch (e) {
      setUploadError(
        jobOrUploadErrorMessage(e, { chunked: useChunked, sizeBytes: size }),
      );
    } finally {
      setUploading(false);
      setUploadProgress(null);
      setJobState(null);
    }
  };

  const handleAnalyzeLink = async () => {
    const url = linkUrl.trim();
    if (!url || uploading) return;
    setUploading(true);
    // The server does the fetching/transcription; there's no client-side chunk
    // progress to show — the staged job card (or a spinner) covers it instead.
    setUploadProgress(null);
    setUploadError(null);
    setUploadStored(null);
    setUploadStorageNote(null);
    setJobState(null);
    try {
      const opts = {
        consent,
        store: storeRecording,
        context: composedContext(),
      };
      // Submit as a background JOB and poll it — the synchronous /analyze/link
      // it replaces is a multi-minute request Android backgrounding routinely
      // kills. ONLY a failed SUBMIT (old server / storage off → 404/503) falls
      // back to the synchronous call; a poll failure is a real error (never a
      // silent re-analysis).
      const result = await (async (): Promise<UploadAnalyzeResult> => {
        let created;
        try {
          created = await postAnalyzeLinkJob(url, opts);
        } catch (e) {
          const status = errorStatus(e);
          if (status === 404 || status === 503) {
            return await postAnalyzeLink(url, opts);
          }
          throw e;
        }
        return pollJobToDone(created.job_id, setJobState);
      })();
      // Identical handoff to the upload path: hydrate the store transcript, then
      // jump to the ready-made analysis (and thread the recording id if stored).
      loadTurns(result.turns);
      setUploadStored(result.stored);
      setUploadStorageNote(result.stored ? null : result.storage_note);
      onAnalyzeDynamics?.(result, result.recording_id ?? null);
    } catch (e) {
      setUploadError(jobOrUploadErrorMessage(e, { link: true }));
    } finally {
      setUploading(false);
      setUploadProgress(null);
      setJobState(null);
    }
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
        testID="analyze-screen"
      >
        {/* Header: back to Home + the past-recordings entry point. */}
        <View style={styles.headerRow}>
          {onBack && (
            <TouchableOpacity
              testID="analyze-back"
              accessibilityRole="button"
              style={styles.backButton}
              onPress={onBack}
            >
              <Text style={styles.backText}>← Home</Text>
            </TouchableOpacity>
          )}
          {onOpenRecordings && (
            <TouchableOpacity
              testID="open-recordings-link"
              style={styles.recordingsLink}
              onPress={onOpenRecordings}
            >
              <Text style={styles.recordingsLinkText}>▶ Recordings</Text>
            </TouchableOpacity>
          )}
        </View>

        <Text style={styles.heading}>Analyze a Conversation</Text>

        <View style={styles.body}>
          {/* One-tap relationship context — frames the analysis honestly
              without a wall of form fields. */}
          <RelationshipPicker
            value={relationship}
            onSelect={setRelationship}
            disabled={uploading}
          />

          <View style={styles.recordingCard}>
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
                        uploading && styles.uploadButtonDisabled,
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
                    (uploading || !linkUrl.trim()) && styles.uploadButtonDisabled,
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

            {/* Staged progress for an in-flight analysis job: stage label +
                spinner, the server's honest progress note, and a rough ETA once
                the recording's duration is known (labeled an estimate). */}
            {jobState &&
              jobState.status !== "done" &&
              jobState.status !== "failed" &&
              jobState.status !== "stalled" && (
                <View style={styles.jobProgress} testID="job-progress">
                  <View style={styles.uploadingRow}>
                    <ActivityIndicator color="#4A90D9" />
                    <Text style={styles.jobStageLabel} testID="job-stage-label">
                      {jobStageLabel(jobState.status)}
                    </Text>
                  </View>
                  {jobState.progress_note && (
                    <Text style={styles.jobNote} testID="job-progress-note">
                      {jobState.progress_note}
                    </Text>
                  )}
                  {(() => {
                    const eta = estimateRemainingSeconds(
                      jobState.status,
                      jobState.duration_seconds,
                    );
                    return eta == null ? null : (
                      <Text style={styles.jobEta} testID="job-eta">
                        {`~${eta}s remaining (estimate)`}
                      </Text>
                    );
                  })()}
                </View>
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

          {/* Text tools: the paste/type-a-transcript review lives one tap away
              so the primary flow stays uncluttered. */}
          {onOpenTextTools && (
            <TouchableOpacity
              testID="open-text-tools"
              accessibilityRole="button"
              style={styles.textToolsLink}
              onPress={onOpenTextTools}
              disabled={uploading}
            >
              <Text style={styles.textToolsLinkText}>
                Have a written transcript? Work with text →
              </Text>
            </TouchableOpacity>
          )}
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: {
    flex: 1,
  },
  content: {
    paddingTop: 20,
    paddingBottom: 40,
  },
  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    marginBottom: 8,
  },
  backButton: {
    minHeight: 44,
    justifyContent: "center",
    paddingRight: 12,
  },
  backText: {
    fontSize: 16,
    fontWeight: "600",
    color: "#4A90D9",
  },
  heading: {
    fontSize: 24,
    fontWeight: "700",
    textAlign: "center",
    marginBottom: 14,
    color: "#111827",
  },
  body: {
    paddingHorizontal: 16,
  },
  recordingsLink: {
    paddingHorizontal: 14,
    paddingVertical: 8,
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
    paddingVertical: 14,
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
    paddingVertical: 12,
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
    paddingVertical: 14,
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
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
    marginTop: 10,
  },
  uploadButtonDisabled: {
    opacity: 0.6,
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
  jobProgress: {
    marginTop: 12,
    padding: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#F9FAFB",
    gap: 6,
  },
  jobStageLabel: {
    fontSize: 15,
    fontWeight: "600",
    color: "#1F2937",
  },
  jobNote: {
    fontSize: 13,
    lineHeight: 18,
    color: "#6B7280",
  },
  jobEta: {
    fontSize: 12.5,
    lineHeight: 17,
    color: "#4A90D9",
    fontWeight: "500",
  },
  textToolsLink: {
    marginTop: 16,
    minHeight: 48,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    backgroundColor: "#FFFFFF",
  },
  textToolsLinkText: {
    fontSize: 14,
    fontWeight: "600",
    color: "#4A90D9",
  },
});
