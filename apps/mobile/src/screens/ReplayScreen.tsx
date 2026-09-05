import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import {
  getRecording,
  getRecordingMediaUrl,
  getRecordingSourceUrl,
  patchRecordingSource,
  patchRecordingTitle,
  postReanalyze,
  getAnalyzeJob,
  postReflect,
} from "../api/client";
import type {
  RecordingDetail,
  AnalyzeResult,
  AnalyzeJobState,
  JobStatus,
  SpeakerLabel,
  PatchSpeakerLabelsResult,
  MediaType,
  CouldHaveSaid,
  VoicePerson,
} from "../api/client";
import * as apiClient from "../api/client";
import ToneSummaryCard from "../components/ToneSummaryCard";
import CouldHaveSaidList from "../components/CouldHaveSaidList";
import ScoreboardPanel from "../components/ScoreboardPanel";
import { scoreSession } from "../live/pleasantness";
import { modeLabel } from "./toneTrends";
import HeatChart from "../components/HeatChart";
import MediaPlayer, { MediaPlayerHandle } from "../components/MediaPlayer";
import {
  speakerSegments,
  createSegmentAudition,
  type SegmentAudition,
} from "../components/auditionPlayback";
import { speakerLabel } from "../utils/speakerLabels";
import { getCachedMediaUri, cacheMediaInBackground } from "../utils/mediaCache";
import RecordingShareManager from "../components/RecordingShareManager";
import SpeakerNaming from "../components/SpeakerNaming";
import SpeakerEnrollment, { useSpeakerEnrollment } from "../components/SpeakerEnrollment";
import PulseDot from "../components/PulseDot";
import DeviceDiarizationRow from "../components/DeviceDiarizationRow";
import { summarizeReanalyze, type ReanalyzeSummary } from "./reanalyzeDelta";
import { setPlaybackMode } from "../utils/audioMode";
import { formatDateTime } from "../utils/dateDisplay";

/** How long we give a source to report a real duration (moov parsed /
 *  readyToPlay) before treating it as stuck. A moov-at-end HD MP4 served
 *  without HTTP Range support buffers forever at 0:00 with no error event, so a
 *  wall-clock watchdog is the only way to recover. */
const LOAD_TIMEOUT_MS = 8000;

// --- Re-analyze job polling (mirrors AnalyzeScreen's submit-and-poll pattern) --
const REANALYZE_POLL_MS = 3000;
// Keep polling through a computed "stalled" for a while — a re-analysis can
// legitimately sit before advancing — and only give up (honestly) after this.
const REANALYZE_STALL_GRACE_MS = 3 * 60 * 1000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Human stage label for the re-analyze progress card. */
function reanalyzeStageLabel(status: JobStatus | undefined): string {
  switch (status) {
    case "queued":
      return "Queued…";
    case "transcribing":
      return "Transcribing…";
    case "analyzing":
      return "Analyzing…";
    case "storing":
      return "Saving…";
    case "stalled":
      return "Still working…";
    default:
      return "Re-analyzing…";
  }
}

// House colors.
/** Live-session reflection poll: every 5 s, at most ~1 min. */
const REFLECTION_POLL_MS = 5000;
const REFLECTION_POLL_MAX_ATTEMPTS = 12;

const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";
const GOOD = "#1B7A4B"; // improved conduct score (house green)
const DANGER = "#DC2626"; // dropped conduct score

interface ReplayScreenProps {
  recordingId: string;
  onBack: () => void;
  /** Open the attach-HD-source input immediately on mount (from the Dynamics
   *  "Attach link now" popup). Otherwise the affordance starts collapsed. */
  initialAttachOpen?: boolean;
}

/** True when a source URL is a Google Photos share link. The Photos `=dv` HD
 *  stream is moov-at-end with NO HTTP Range support, so ExoPlayer can never play
 *  it (device-proven: buffers forever at 0:00, watchdog reverts). For these we
 *  suppress the "Try HD" offer entirely and show an honest note instead — HD
 *  replay is genuinely impossible from a Photos link. Other link types (direct
 *  file, Drive) may stream fine, so they keep the Try-HD affordance. */
function isGooglePhotosUrl(url: string | null | undefined): boolean {
  if (!url) return false;
  const u = url.toLowerCase();
  return u.includes("photos.app.goo.gl") || u.includes("photos.google.com");
}

/** Read the HTTP status off a client error (the `.status` the client attaches,
 *  else parsed from the `API error: <status>` message). */
function statusOf(err: unknown): number {
  const withStatus = err as { status?: number };
  if (typeof withStatus?.status === "number") return withStatus.status;
  const msg = err instanceof Error ? err.message : "";
  return Number(msg.match(/API error: (\d+)/)?.[1] ?? 0);
}

/** Turn a client error into an honest, human message. The pinned contract uses
 *  503 for "storage not configured" — we name that case plainly instead of
 *  showing a raw status. */
function humanizeError(message: string): string {
  if (message.includes("503")) {
    return "Replay storage isn’t enabled yet.";
  }
  if (message.includes("404")) {
    return "This recording is no longer available.";
  }
  if (message.includes("401")) {
    return "Please sign in again to view this recording.";
  }
  return message;
}

/**
 * "Watch yourselves at the spike" — plays a stored recording with the heat chart
 * synced beneath it. A moving playhead tracks playback; tapping the chart seeks
 * the media. Everything is fetched against the pinned recordings contract with
 * honest loading/error/retry states (never a fabricated recording).
 */
export default function ReplayScreen({
  recordingId,
  onBack,
  initialAttachOpen = false,
}: ReplayScreenProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<RecordingDetail | null>(null);
  const [mediaUrl, setMediaUrl] = useState<string | null>(null);
  // Track 2: a LIVE coaching session kept no audio on the server
  // (media_type "none") — the screen renders transcript + analysis without
  // a player, and never asks for a media URL it knows doesn't exist.
  const [noMedia, setNoMedia] = useState(false);
  // "What you could have said": the server's cached reflection (seeded from
  // the detail read), or fetched on demand. `reflectError` is honest copy
  // for the failure modes (no identified "you", shared, model failure).
  const [reflections, setReflections] = useState<CouldHaveSaid[] | null>(null);
  const [reflecting, setReflecting] = useState(false);
  const [reflectError, setReflectError] = useState<string | null>(null);
  const reflectInFlightRef = useRef(false);
  // Replay plays our STORED DERIVATIVE first for every source — the /media
  // endpoint serves it with proper HTTP Range support, so it loads reliably.
  // For a link-sourced recording the user's OWN hosted original is an opt-in
  // "Try HD" (hdMode): that remote stream can be a moov-at-end MP4 with no Range
  // support, which ExoPlayer buffers forever at 0:00, so it must never be the
  // default. `fellBack` drives an honest "playing stored copy" note.
  const [hdMode, setHdMode] = useState(false);
  const [fellBack, setFellBack] = useState(false);
  // The current media source never reported a real duration within
  // LOAD_TIMEOUT_MS and there's nothing better to switch to (we're already on
  // the derivative) — drives an honest "media isn't loading" note.
  const [mediaStuck, setMediaStuck] = useState(false);
  // Mirrors hdMode for the async player-error / watchdog handlers (avoids a
  // stale closure) and guards the fallback so a burst of errors triggers it once.
  const hdRef = useRef(false);
  // Set true once the current source reports a real duration; the load-timeout
  // watchdog reads it to tell "loaded" from "stuck buffering". A ref (not state)
  // so updating it from the duration callback never re-renders.
  const durationKnownRef = useRef(false);
  // The current recording's media type, set alongside `setDetail` in `load()`
  // — read (not a dependency) by `loadDerivative`, which is also called from
  // callbacks created before `detail` state existed (handlePlayerError, the
  // load-timeout watchdog), so a ref avoids any stale-closure risk. Only used
  // to pick the local cache filename's extension (see ../utils/mediaCache).
  const mediaTypeRef = useRef<MediaType>("audio");

  // Playback position (seconds), pushed up from MediaPlayer at ~4Hz to drive the
  // heat chart playhead.
  const [playheadSeconds, setPlayheadSeconds] = useState(0);

  // --- Speaker isolation + "Play this voice" audition ---
  // `isolatedSpeaker` mirrors the chart legend's isolation (controlled here so
  // the audition button and — later — the naming flow share one seam).
  // `auditioning` gates Play vs Stop; the live chain lives in a ref because the
  // player's position callbacks must reach it without re-renders.
  const [isolatedSpeaker, setIsolatedSpeaker] = useState<string | null>(null);
  const [auditioning, setAuditioning] = useState(false);
  const auditionRef = useRef<SegmentAudition | null>(null);
  // Stretch: for video, overlay the chart on the bottom third of the frame.
  const [overlayMode, setOverlayMode] = useState(false);

  // --- Attach / replace HD source ---
  // The user can attach a durable share/direct link to their own hosted
  // original so replay streams it in HD. `attachOpen` reveals the input;
  // `attachError` renders a 422's user-facing detail verbatim.
  const [attachOpen, setAttachOpen] = useState(initialAttachOpen);
  const [attachUrl, setAttachUrl] = useState("");
  const [attaching, setAttaching] = useState(false);
  const [attachError, setAttachError] = useState<string | null>(null);

  // --- Rename (tap the title to edit) ---
  // `renamedTitle` is a local override applied optimistically on a successful
  // PATCH; `renameNote` carries an honest status line (e.g. the capability-gated
  // "not supported yet" when the backend has no title field). Nothing is
  // fabricated — a failed rename never changes the displayed name.
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const [savingTitle, setSavingTitle] = useState(false);
  const [renamedTitle, setRenamedTitle] = useState<string | null>(null);
  const [renameNote, setRenameNote] = useState<string | null>(null);

  // --- Manual speaker naming ("Name the speakers") ---
  // A successful PATCH …/speaker-labels returns the RESOLVED effective labels;
  // we hold them in `labelOverride` so every surface (legend, talk-share,
  // inspector, enrollment, the naming panel) re-labels at once without a refetch.
  // Null until the first save — the base labels then come straight from the
  // stored analysis (which already reflects any manual labels on read).
  // `manualLabels` is the raw {id: name} map the editor prefills from.
  // `namingUnsupported` latches true only if a save 404s (older server with no
  // route) so the affordance hides itself gracefully.
  const [labelOverride, setLabelOverride] = useState<Record<
    string,
    SpeakerLabel
  > | null>(null);
  const [manualLabels, setManualLabels] = useState<Record<string, string>>({});
  const [namingUnsupported, setNamingUnsupported] = useState(false);
  // People labeling: the account's enrolled people for the "Who is this?"
  // sheet. Null until fetched (or when the server has no people endpoint —
  // the sheet then simply isn't offered, and the plain name editor remains).
  const [people, setPeople] = useState<VoicePerson[] | null>(null);
  const refreshPeople = useCallback(async () => {
    try {
      const res = await apiClient.listVoicePeople();
      setPeople(res.available && res.storage_enabled ? res.people : null);
    } catch {
      setPeople(null);
    }
  }, []);
  useEffect(() => {
    void refreshPeople();
  }, [refreshPeople, recordingId]);

  // --- Re-analyze with the latest engine ---
  // `reanalyzing` gates the button vs the progress card; `reanalyzeJob` is the
  // latest poll (drives the staged progress); `reanalyzeError` carries an honest
  // message (e.g. a 422 when the recording kept no audio). On completion we
  // refetch the recording so the fresh analysis renders.
  const [reanalyzing, setReanalyzing] = useState(false);
  const [reanalyzeJob, setReanalyzeJob] = useState<AnalyzeJobState | null>(null);
  const [reanalyzeError, setReanalyzeError] = useState<string | null>(null);
  // The honest before/after read shown once a re-analysis completes (what the
  // latest engine actually changed). Null until then; cleared when a new re-run
  // starts.
  const [reanalyzeSummary, setReanalyzeSummary] =
    useState<ReanalyzeSummary | null>(null);
  const reanalyzeInFlightRef = useRef(false);
  // The analysis on screen when re-analyze was tapped, kept to diff against the
  // fresh one for the "what changed" summary.
  const preReanalyzeRef = useRef<AnalyzeResult | null>(null);

  const playerRef = useRef<MediaPlayerHandle>(null);

  // In-flight + mount guards, mirroring DynamicsScreen: one fetch at a time, and
  // no state writes after the user backs out mid-load.
  const inFlightRef = useRef(false);
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Defensively put the device audio session into a PLAYBACK configuration on
  // mount. The Live Coach flow configures a recording session (allowsRecording),
  // which on Android leaves media playback silent until it's reset — and its
  // teardown could be missed. Resetting here guarantees replay is audible no
  // matter how the user got to this screen. Fire-and-forget; web no-ops.
  useEffect(() => {
    void setPlaybackMode().catch(() => {});
  }, []);

  const setHd = useCallback((on: boolean) => {
    hdRef.current = on;
    setHdMode(on);
  }, []);

  // Resolve the stored derivative URL (our own copy). Shared by the upload path,
  // the link fallback, and the player-error fallback.
  //
  // Local-cache-first (2026-08-18): before touching the network at all, check
  // whether this recording's media is already cached on disk from a PRIOR
  // play (native only — see ../utils/mediaCache's doc comment). A hit hands
  // the player a local `file://` uri with zero network calls and an instant
  // start; nothing below this branch runs. A miss falls through to EXACTLY
  // today's behavior — fetch the signed media_url and stream it — and only
  // ADDS a fire-and-forget background download of that same url so the NEXT
  // play/replay is a cache hit. First-play start time for a large video is
  // unaffected either way: streaming never waits on the download.
  const loadDerivative = useCallback(async () => {
    const mediaType = mediaTypeRef.current;
    const cachedUri = getCachedMediaUri(recordingId, mediaType);
    if (cachedUri) {
      if (mountedRef.current) {
        setMediaUrl(cachedUri);
        setHd(false);
      }
      return;
    }
    const media = await getRecordingMediaUrl(recordingId);
    if (mountedRef.current) {
      setMediaUrl(media.url);
      setHd(false);
    }
    cacheMediaInBackground(recordingId, mediaType, media.url);
  }, [recordingId, setHd]);

  const load = useCallback(async (): Promise<RecordingDetail | null> => {
    if (inFlightRef.current) return null;
    inFlightRef.current = true;
    setLoading(true);
    setError(null);
    setFellBack(false);
    setMediaStuck(false);
    setHd(false);
    let fetched: RecordingDetail | null = null;
    try {
      // Detail first — its `source` decides whether a "Try HD" opt-in is offered.
      const rec = await getRecording(recordingId);
      fetched = rec;
      mediaTypeRef.current = rec.media_type;
      const liveNoMedia = rec.media_type === "none";
      if (mountedRef.current) {
        setDetail(rec);
        // Fresh read — the stored analysis already carries any manual overrides
        // (label_source "manual"), so drop the local override and re-seed the raw
        // manual map for the editor's prefill.
        setLabelOverride(null);
        setManualLabels(rec.manual_speaker_labels ?? {});
        setNoMedia(liveNoMedia);
        // The cached "what you could have said", if the server has one.
        setReflections(rec.analysis?.live?.could_have_said ?? null);
        setReflectError(null);
      }

      if (liveNoMedia) {
        // Nothing to stream — a live session's audio never reached the
        // server. Leave mediaUrl null; the content gate below accepts that.
        if (mountedRef.current) setMediaUrl(null);
        return fetched;
      }

      // Derivative-FIRST for every source. The stored copy streams from our
      // /media endpoint with Range support and always loads; the linked HD
      // original is opt-in (see handleTryHd) precisely because it can buffer
      // forever. Uploads never had an HD option and are unaffected.
      await loadDerivative();
    } catch (e) {
      if (mountedRef.current) {
        setError(
          humanizeError(e instanceof Error ? e.message : "Something went wrong."),
        );
      }
    } finally {
      inFlightRef.current = false;
      if (mountedRef.current) setLoading(false);
    }
    return fetched;
  }, [recordingId, loadDerivative, setHd]);

  useEffect(() => {
    void load();
  }, [load]);

  // A live session's batch analysis + "what you could have said" land a few
  // seconds after ingest (server/routers/sessions.py runs them in the
  // background). While the stored analysis is still "lite" — or the
  // reflection hasn't been written yet for a session with an identified
  // self voice — re-read the recording every few seconds, bounded, so the
  // reflection appears without a manual refresh. Silent on failure.
  useEffect(() => {
    const live = detail?.analysis?.live;
    if (!detail || detail.media_type !== "none" || !live) return;
    const pending =
      live.analysis_status === "lite" ||
      (live.could_have_said == null && Boolean(live.self_speaker) && !live.reflection);
    if (!pending) return;
    let attempts = 0;
    const timer = setInterval(async () => {
      attempts += 1;
      if (attempts > REFLECTION_POLL_MAX_ATTEMPTS) {
        clearInterval(timer);
        return;
      }
      try {
        const fresh = await getRecording(recordingId);
        if (!mountedRef.current) return;
        setDetail(fresh);
        setReflections(fresh.analysis?.live?.could_have_said ?? null);
      } catch {
        // Keep polling until the cap; the manual Reflect button still works.
      }
    }, REFLECTION_POLL_MS);
    return () => clearInterval(timer);
  }, [detail, recordingId]);

  // The player errored on the remote HD stream (expired/blocked/unseekable) —
  // fall back to the stored derivative once. Ignored when we're already on the
  // derivative (there's nothing better to switch to).
  const handlePlayerError = useCallback(async () => {
    if (!hdRef.current) return;
    hdRef.current = false; // guard re-entrancy from a burst of error events
    try {
      await loadDerivative();
      if (mountedRef.current) setFellBack(true);
    } catch {
      if (mountedRef.current) {
        setError(humanizeError("Something went wrong."));
      }
    }
  }, [loadDerivative]);

  // Opt-in HD: resolve and switch to the user's own hosted original. On any
  // failure to resolve it, stay on the derivative and note it — never leave the
  // player pointed at nothing.
  const handleTryHd = useCallback(async () => {
    try {
      const src = await getRecordingSourceUrl(recordingId);
      if (mountedRef.current) {
        setFellBack(false);
        setMediaStuck(false);
        setMediaUrl(src.url);
        setHd(true);
      }
    } catch {
      if (mountedRef.current) setFellBack(true);
    }
  }, [recordingId, setHd]);

  // Manual return to the stored copy (from the HD stream). Also the owner-facing
  // escape hatch when a remote stream misbehaves in ways we can't detect.
  const handleBackToDerivative = useCallback(async () => {
    try {
      setFellBack(false);
      setMediaStuck(false);
      await loadDerivative(); // sets the derivative URL and clears hdMode
    } catch {
      if (mountedRef.current) {
        setError(humanizeError("Something went wrong."));
      }
    }
  }, [loadDerivative]);

  // Marks the current source as loaded once it reports a real duration; the
  // watchdog reads this ref to distinguish "loaded" from "stuck buffering".
  const handleDurationChange = useCallback((seconds: number) => {
    if (seconds > 0) {
      durationKnownRef.current = true;
      // A source that reports a real duration is NOT stuck — clear any latch
      // left by an EARLIER source (the Google-Photos-link flow stalls first,
      // then falls back to a stored copy that loads fine; field bug #11 was
      // this latch outliving the failure and locking out audition playback
      // while the transport played happily).
      setMediaStuck(false);
    }
  }, []);

  // Load-timeout watchdog. Each time the source URL changes, give it
  // LOAD_TIMEOUT_MS to report a real duration. If it never does — the classic
  // moov-at-end-without-Range HD stream buffers forever at 0:00 and emits NO
  // error event, so handlePlayerError never fires — auto-recover: drop from HD
  // to the stored derivative, or (if already on the derivative) surface an
  // honest "isn't loading" note rather than spin at 0:00 indefinitely.
  useEffect(() => {
    if (!mediaUrl) return;
    durationKnownRef.current = false;
    const id = setTimeout(() => {
      if (durationKnownRef.current) return; // Loaded fine — nothing to do.
      if (hdRef.current) {
        hdRef.current = false; // guard against a double-fire
        void loadDerivative().then(() => {
          if (mountedRef.current) setFellBack(true);
        });
      } else if (mountedRef.current) {
        setMediaStuck(true);
      }
    }, LOAD_TIMEOUT_MS);
    return () => clearTimeout(id);
  }, [mediaUrl, loadDerivative]);

  // Position updates feed the chart playhead AND the audition chain (which
  // uses them to jump between the isolated speaker's segments).
  const handlePositionChange = useCallback((seconds: number) => {
    setPlayheadSeconds(seconds);
    auditionRef.current?.handlePosition(seconds);
  }, []);

  // Stop any running audition (pauses playback via the chain's onEnded).
  const stopAudition = useCallback(() => {
    auditionRef.current?.stop();
    auditionRef.current = null;
  }, []);

  // Tap-to-seek from the chart: drive the player's position directly. A manual
  // seek is a user override, so it also ends any running audition rather than
  // letting the chain fight the new position.
  const handleSeekToTurn = useCallback(
    (startTime: number) => {
      stopAudition();
      playerRef.current?.seek(startTime);
    },
    [stopAudition],
  );

  // Legend isolation changes come through here (the chart is controlled).
  // Switching voices — or going back to All — always kills a running audition
  // so one speaker's playback never bleeds into another's isolation.
  const handleIsolateSpeaker = useCallback(
    (speaker: string | null) => {
      stopAudition();
      setIsolatedSpeaker(speaker);
    },
    [stopAudition],
  );

  // "▶ Play this voice": play ONLY the isolated speaker's segments in order.
  const handleAuditionPlay = useCallback(() => {
    const player = playerRef.current;
    if (!isolatedSpeaker || !player || !detail) return;
    stopAudition();
    const segments = speakerSegments(detail.turns ?? [], isolatedSpeaker);
    const audition = createSegmentAudition(player, segments, () => {
      auditionRef.current = null;
      if (mountedRef.current) setAuditioning(false);
    });
    auditionRef.current = audition;
    setAuditioning(true);
    audition.start();
  }, [isolatedSpeaker, detail, stopAudition]);

  // Submit the attach/replace link: PATCH the source, then refetch the recording
  // so the "Try HD" opt-in becomes available for the now-linked source. A 422's
  // user-facing detail is shown verbatim; nothing is fabricated on failure.
  const handleAttachSubmit = useCallback(async () => {
    const url = attachUrl.trim();
    if (!url || attaching) return;
    setAttaching(true);
    setAttachError(null);
    try {
      await patchRecordingSource(recordingId, url);
      if (mountedRef.current) {
        setAttachOpen(false);
        setAttachUrl("");
      }
      // Refetch: detail.source is now a link, so `load` takes the HD-first path.
      await load();
    } catch (e) {
      if (mountedRef.current) {
        const detail = (e as { detail?: string })?.detail;
        setAttachError(
          detail ??
            humanizeError(e instanceof Error ? e.message : "Something went wrong."),
        );
      }
    } finally {
      if (mountedRef.current) setAttaching(false);
    }
  }, [attachUrl, attaching, recordingId, load]);

  // The name shown in the header: a just-applied rename wins, then any
  // server-provided title, then the raw filename.
  const currentTitle =
    renamedTitle ?? detail?.title ?? detail?.filename ?? "Replay";

  const openRename = useCallback(() => {
    setTitleDraft(currentTitle === "Replay" ? "" : currentTitle);
    setRenameNote(null);
    setEditingTitle(true);
  }, [currentTitle]);

  // Attempt the rename. The backend may not support titling yet (no field / no
  // PATCH /recordings/{id} route → a 4xx): in that case we DON'T change the name
  // and show an honest "not supported yet" note rather than pretend it worked. A
  // transient 5xx/network failure gets a distinct "try again" note.
  const handleRenameSubmit = useCallback(async () => {
    const next = titleDraft.trim();
    if (!next || savingTitle) return;
    setSavingTitle(true);
    setRenameNote(null);
    try {
      await patchRecordingTitle(recordingId, next);
      if (mountedRef.current) {
        setRenamedTitle(next);
        setEditingTitle(false);
      }
    } catch (e) {
      if (mountedRef.current) {
        const status = statusOf(e);
        setRenameNote(
          status >= 400 && status < 500
            ? "Renaming isn’t supported yet — coming soon."
            : "Couldn’t rename right now — please try again.",
        );
        setEditingTitle(false);
      }
    } finally {
      if (mountedRef.current) setSavingTitle(false);
    }
  }, [titleDraft, savingTitle, recordingId]);

  // Re-analyze this recording. `submit` returns the job id to poll — the
  // plain "latest engine" re-run POSTs here (handleReanalyze); the device
  // voice row POSTs its own segments and hands the job id over
  // (handleReanalyzeJob). Either way the job is polled to completion (reusing
  // the AnalyzeScreen job pattern) and the recording refetched so the fresh
  // analysis + chart render. Errors are honest: a 422 means the server kept no
  // audio to re-run, so it plainly can't be redone. Resolves `{ ok }` (with
  // the message shown) so a caller can mirror the outcome where it started.
  const runReanalyzeJob = useCallback(async (
    submit: () => Promise<string>,
  ): Promise<{ ok: boolean; message?: string }> => {
    if (reanalyzeInFlightRef.current) {
      return { ok: false, message: "A re-analysis is already running." }; // One costed re-run at a time.
    }
    reanalyzeInFlightRef.current = true;
    setReanalyzing(true);
    setReanalyzeError(null);
    setReanalyzeJob(null);
    setReanalyzeSummary(null);
    // Snapshot the analysis on screen now, to diff against the fresh one.
    preReanalyzeRef.current = detail?.analysis ?? null;
    try {
      const job_id = await submit();
      let transientErrors = 0;
      let stalledSince: number | null = null;
      for (;;) {
        let state: AnalyzeJobState;
        try {
          state = await getAnalyzeJob(job_id);
        } catch (e) {
          // A 404 means the job is truly gone; other hiccups get a few retries so
          // one dropped poll doesn't abort a live re-run.
          if (statusOf(e) === 404 || transientErrors >= 3) throw e;
          transientErrors += 1;
          await sleep(REANALYZE_POLL_MS);
          continue;
        }
        transientErrors = 0;
        if (mountedRef.current) setReanalyzeJob(state);
        if (state.status === "done") break;
        if (state.status === "failed") {
          throw new Error(state.error ?? "The re-analysis failed. Please try again.");
        }
        if (state.status === "stalled") {
          if (stalledSince === null) stalledSince = Date.now();
          if (Date.now() - stalledSince > REANALYZE_STALL_GRACE_MS) {
            throw new Error(
              "This is taking much longer than expected. If the update doesn’t " +
                "appear shortly, please try again.",
            );
          }
        } else {
          stalledSince = null;
        }
        await sleep(REANALYZE_POLL_MS);
      }
      // Fresh analysis is stored server-side — refetch so it renders, then diff
      // it against what was on screen so we can honestly say what changed.
      const fresh = await load();
      if (fresh && mountedRef.current) {
        setReanalyzeSummary(
          summarizeReanalyze(
            preReanalyzeRef.current,
            fresh.analysis,
            fresh.analysis?.speaker_labels,
          ),
        );
      }
      return { ok: true };
    } catch (e) {
      const status = statusOf(e);
      let message: string;
      if (status === 422) {
        message =
          "This recording didn’t keep its audio, so it can’t be re-analyzed.";
      } else if (status === 404) {
        message = "This recording is no longer available.";
      } else if (status === 503) {
        message = "Re-analysis isn’t available right now.";
      } else if (status === 401) {
        message = "Please sign in again to re-analyze.";
      } else {
        message =
          e instanceof Error && e.message
            ? e.message
            : "Couldn’t re-analyze right now — please try again.";
      }
      if (mountedRef.current) setReanalyzeError(message);
      return { ok: false, message };
    } finally {
      reanalyzeInFlightRef.current = false;
      if (mountedRef.current) {
        setReanalyzing(false);
        setReanalyzeJob(null);
      }
    }
  }, [load, detail]);

  const handleReanalyze = useCallback(
    () => runReanalyzeJob(async () => (await postReanalyze(recordingId)).job_id),
    [runReanalyzeJob, recordingId],
  );

  // The device voice row already POSTed its segments; poll + refresh from
  // its job id so the chart, legend, Speakers card and report cards follow.
  const handleReanalyzeJob = useCallback(
    (jobId: string) => runReanalyzeJob(async () => jobId),
    [runReanalyzeJob],
  );

  // Track 2 — "What you could have said": ask the server for the reflection
  // over the user's own turns. The server caches it on the episode, so a
  // repeat tap is free; a real LLM call happens at most once per transcript.
  // Failure copy is honest per status: 422 = nobody in the recording is
  // identified as the user (point at "This is me"), 403 = a shared recording
  // (the owner's spend), 502 = the model gave nothing usable.
  const handleReflect = useCallback(async () => {
    if (reflectInFlightRef.current) return; // One costed call at a time.
    reflectInFlightRef.current = true;
    setReflecting(true);
    setReflectError(null);
    try {
      const result = await postReflect(recordingId);
      if (mountedRef.current) setReflections(result.could_have_said);
    } catch (e) {
      if (mountedRef.current) {
        const status = statusOf(e);
        if (status === 422) {
          setReflectError(
            "No speaker in this recording is identified as you yet — tap " +
              "“This is me” on your voice below, then try again.",
          );
        } else if (status === 403) {
          setReflectError("Only the recording’s owner can ask for reflections.");
        } else if (status === 502) {
          setReflectError(
            "The coach couldn’t produce a reflection just now — please try again.",
          );
        } else if (status === 401) {
          setReflectError("Please sign in again to reflect on this recording.");
        } else if (status === 503) {
          setReflectError("Reflections aren’t available right now.");
        } else {
          setReflectError(
            humanizeError(e instanceof Error ? e.message : "Something went wrong."),
          );
        }
      }
    } finally {
      reflectInFlightRef.current = false;
      if (mountedRef.current) setReflecting(false);
    }
  }, [recordingId]);

  // A save's resolved labels win; otherwise the stored analysis labels (which
  // already reflect any manual overrides applied server-side on read). Shared by
  // the chart legend/inspector, the enrollment card, and the naming panel so a
  // rename lands everywhere at once.
  const handleLabelsSaved = useCallback((result: PatchSpeakerLabelsResult) => {
    setLabelOverride(result.speaker_labels);
    setManualLabels(result.manual_speaker_labels);
  }, []);

  const perTurn = detail?.analysis?.per_turn ?? [];
  const turns = detail?.turns ?? [];
  const hasChart = perTurn.length > 0 && turns.length > 0;
  const isVideo = detail?.media_type === "video";
  // Only a link-sourced recording has a user-hosted original to stream in HD.
  const hdAvailable = detail?.source?.type === "link";
  // Read-only mode: a recording SHARED with the caller by another account. Every
  // owner-only affordance (rename, share, re-analyze, attach source, name
  // speakers, "This is me" enrollment) is hidden — playback, charts, and word
  // patterns remain. An older server omits `shared`, so this is false ⇒ owned.
  const isShared = detail?.shared === true;

  // The labels every surface renders. `labelOverride` (from the latest save) wins;
  // otherwise the stored analysis labels.
  const effectiveLabels = labelOverride ?? detail?.analysis?.speaker_labels;
  // Manual naming is offered only when the server actually supports it: the detail
  // must carry a `manual_speaker_labels` map (older servers omit it), and no save
  // has 404'd this session.
  const namingSupported =
    detail?.manual_speaker_labels !== undefined && !namingUnsupported;
  // "This is me" state, shared by the inline rows (current servers) and the
  // standalone fallback card (older servers without speaker naming).
  const enrollment = useSpeakerEnrollment(
    recordingId, !isShared && turns.length > 0 && !noMedia,
  );
  // PRD §6 scoreboard for a live session, re-scored on the phone from the
  // stored per-turn tone + prosody with the SAME arithmetic the live board
  // used (live/pleasantness.ts ≡ server/pleasantness.py). Null for an
  // upload — no per-turn tone, nothing honest to show.
  const liveScoreboard = React.useMemo(() => {
    if (!turns.some((t) => t.text_tone || t.prosody)) return null;
    const { board } = scoreSession(turns);
    return board.people.some((p) => p.scoredTurns > 0) ? board : null;
  }, [turns]);

  const chart = hasChart ? (
    <HeatChart
      perTurn={perTurn}
      turns={turns.map((t) => ({ speaker: t.speaker, text: t.text }))}
      speakerLabels={effectiveLabels}
      turnsTiming={turns.map((t) => ({
        start_time: t.start_time,
        end_time: t.end_time,
      }))}
      durationSeconds={detail?.duration_seconds ?? null}
      playheadSeconds={playheadSeconds}
      onSeekToTurn={handleSeekToTurn}
      isolatedSpeaker={isolatedSpeaker}
      onIsolateSpeaker={handleIsolateSpeaker}
    />
  ) : null;

  // The audition affordance: only while a speaker is isolated. When the media
  // never loaded there is nothing to play — say so instead of a dead button.
  const auditionBlock =
    hasChart && isolatedSpeaker != null && !noMedia ? (
      <View style={styles.auditionRow} testID="audition-block">
        {mediaStuck ? (
          <Text style={styles.auditionUnavailable} testID="audition-unavailable">
            This recording’s audio isn’t available, so this voice can’t be
            played.
          </Text>
        ) : auditioning ? (
          <TouchableOpacity
            testID="audition-stop"
            style={styles.auditionStopButton}
            onPress={stopAudition}
            accessibilityRole="button"
          >
            <Text style={styles.auditionStopText}>◼ Stop</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            testID="audition-play"
            style={styles.auditionPlayButton}
            onPress={handleAuditionPlay}
            accessibilityRole="button"
            accessibilityLabel={`Play only ${speakerLabel(
              isolatedSpeaker,
              effectiveLabels,
            )}'s turns`}
          >
            <Text style={styles.auditionPlayText}>
              ▶ Play this voice —{" "}
              {speakerLabel(isolatedSpeaker, effectiveLabels)}
            </Text>
          </TouchableOpacity>
        )}
      </View>
    ) : null;

  return (
    <View style={styles.flex}>
      <View style={styles.header}>
        <TouchableOpacity
          testID="replay-back"
          onPress={onBack}
          hitSlop={{ top: 10, bottom: 10, left: 8, right: 16 }}
        >
          <Text style={styles.backText}>‹ Back</Text>
        </TouchableOpacity>
        {editingTitle ? (
          <View style={styles.titleEditRow}>
            <TextInput
              testID="rename-input"
              style={styles.renameInput}
              value={titleDraft}
              onChangeText={setTitleDraft}
              placeholder="Name this conversation"
              placeholderTextColor="#9CA3AF"
              autoFocus
              maxLength={120}
              editable={!savingTitle}
              onSubmitEditing={() => void handleRenameSubmit()}
            />
            <TouchableOpacity
              testID="rename-save"
              onPress={() => void handleRenameSubmit()}
              disabled={savingTitle || !titleDraft.trim()}
            >
              {savingTitle ? (
                <ActivityIndicator size="small" color={PRIMARY} />
              ) : (
                <Text style={styles.renameSave}>Save</Text>
              )}
            </TouchableOpacity>
            <TouchableOpacity
              testID="rename-cancel"
              onPress={() => {
                setEditingTitle(false);
                setRenameNote(null);
              }}
            >
              <Text style={styles.renameCancel}>Cancel</Text>
            </TouchableOpacity>
          </View>
        ) : isShared ? (
          // Read-only: the title is not editable for a shared recording.
          <View style={styles.titleWrap}>
            <Text
              style={styles.headerTitle}
              numberOfLines={1}
              testID="replay-title-readonly"
            >
              {currentTitle}
            </Text>
          </View>
        ) : (
          <TouchableOpacity
            testID="replay-title"
            style={styles.titleWrap}
            onPress={openRename}
            accessibilityRole="button"
            accessibilityLabel="Rename this recording"
          >
            <Text style={styles.headerTitle} numberOfLines={1}>
              {currentTitle}
            </Text>
          </TouchableOpacity>
        )}
        <View style={styles.headerSpacer} />
      </View>

      {/* When this conversation was recorded — absolute date + wall-clock time,
          from the recording's real created_at. Omitted entirely (never guessed)
          when the timestamp is missing/unparseable. */}
      {detail && formatDateTime(detail.created_at) && (
        <Text style={styles.recordedAt} testID="replay-recorded-at">
          {formatDateTime(detail.created_at)}
        </Text>
      )}

      {/* Track 2: a live coaching session says so (with its mode) and, while
          its heat analysis is still pending or failed, says that too. */}
      {detail?.source?.type === "live" && (
        <Text style={styles.liveBadge} testID="replay-live-badge">
          Live session
          {modeLabel(detail.mode) ? ` · ${modeLabel(detail.mode)}` : ""}
          {detail.analysis?.live?.analysis_status === "lite"
            ? " · heat analysis pending"
            : detail.analysis?.live?.analysis_status === "failed"
              ? " · heat analysis unavailable"
              : ""}
        </Text>
      )}

      {/* Read-only shared recording — say who it's from, plainly. */}
      {isShared && (
        <Text style={styles.sharedFrom} testID="replay-shared-from">
          {detail?.owner_email
            ? `Shared with you by ${detail.owner_email} · read-only`
            : "Shared with you · read-only"}
        </Text>
      )}

      {renameNote && (
        <Text style={styles.renameNote} testID="rename-note">
          {renameNote}
        </Text>
      )}

      {loading && (
        <View style={styles.centered} testID="replay-loading">
          <ActivityIndicator size="large" color={PRIMARY} />
          <Text style={styles.centeredText}>Loading the recording…</Text>
        </View>
      )}

      {!loading && error && (
        <View style={styles.centered} testID="replay-error">
          <Text style={styles.errorTitle}>Couldn’t open this recording</Text>
          <Text style={styles.errorText} testID="replay-error-message">
            {error}
          </Text>
          <TouchableOpacity
            testID="replay-retry"
            style={styles.retryButton}
            onPress={() => void load()}
          >
            <Text style={styles.retryText}>Try again</Text>
          </TouchableOpacity>
        </View>
      )}

      {!loading && !error && detail && (mediaUrl || noMedia) && (
        <ScrollView
          style={styles.flex}
          contentContainerStyle={styles.content}
          testID="replay-content"
        >
          {/* Attach / replace HD source. When the recording isn't yet
              link-sourced we offer "Attach HD source"; when it already is, a
              quieter "Replace source link". Both reveal the same input.
              Owner-only — hidden in read-only shared mode, and for a live
              session (there is no stored copy to replace). */}
          {!isShared && !noMedia && (
          <View style={styles.attachSection}>
            {!attachOpen ? (
              detail.source?.type === "link" ? (
                <TouchableOpacity
                  testID="replace-source-button"
                  style={styles.replaceSourceButton}
                  onPress={() => {
                    setAttachError(null);
                    setAttachOpen(true);
                  }}
                >
                  <Text style={styles.replaceSourceText}>Replace source link</Text>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity
                  testID="attach-source-button"
                  style={styles.attachSourceButton}
                  onPress={() => {
                    setAttachError(null);
                    setAttachOpen(true);
                  }}
                >
                  <Text style={styles.attachSourceButtonText}>
                    Attach HD source
                  </Text>
                </TouchableOpacity>
              )
            ) : (
              <View>
                <Text style={styles.attachHelp}>
                  Paste a share link to your own hosted original (e.g. a Google
                  Photos single-item link). We’ll stream it in HD for replay.
                </Text>
                <TextInput
                  testID="attach-source-input"
                  style={styles.attachInput}
                  placeholder="https://…"
                  value={attachUrl}
                  onChangeText={setAttachUrl}
                  autoCapitalize="none"
                  autoCorrect={false}
                  keyboardType="url"
                  placeholderTextColor="#9CA3AF"
                  editable={!attaching}
                />
                <View style={styles.attachButtonRow}>
                  <TouchableOpacity
                    testID="attach-source-cancel"
                    style={styles.attachCancel}
                    onPress={() => {
                      setAttachOpen(false);
                      setAttachError(null);
                    }}
                    disabled={attaching}
                  >
                    <Text style={styles.attachCancelText}>Cancel</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    testID="attach-source-submit"
                    style={[
                      styles.attachSubmit,
                      (attaching || !attachUrl.trim()) && styles.attachDisabled,
                    ]}
                    onPress={() => void handleAttachSubmit()}
                    disabled={attaching || !attachUrl.trim()}
                  >
                    {attaching ? (
                      <ActivityIndicator color="#FFFFFF" />
                    ) : (
                      <Text style={styles.attachSubmitText}>Attach</Text>
                    )}
                  </TouchableOpacity>
                </View>
                {attachError && (
                  <Text style={styles.attachError} testID="attach-source-error">
                    {attachError}
                  </Text>
                )}
              </View>
            )}
          </View>
          )}

          {/* HD badge: streaming the user's own linked source. */}
          {hdMode && (
            <View style={styles.hdBadge} testID="hd-badge">
              <Text style={styles.hdBadgeText}>
                HD · streaming from your linked source
              </Text>
            </View>
          )}

          {/* HD is opt-in: the stored copy plays by default (it loads reliably);
              streaming the user's own original is offered only for a linked
              source. Hidden once HD is active (then we show "Back to stored
              copy" instead).

              Google Photos links are the ONE case where HD is impossible: the
              Photos `=dv` stream is moov-at-end with no Range support, so
              ExoPlayer buffers forever at 0:00. Offering "Try HD" there was
              dishonest — replace it with a plain note. Other link types keep the
              opt-in (they may stream fine; the watchdog stays as a backstop). */}
          {hdAvailable &&
            !hdMode &&
            (isGooglePhotosUrl(detail?.source?.url) ? (
              <Text style={styles.photosHdNote} testID="photos-hd-note">
                HD replay isn’t possible from Google Photos links — playing the
                stored copy.
              </Text>
            ) : (
              <TouchableOpacity
                testID="try-hd-button"
                style={styles.tryHdButton}
                onPress={() => void handleTryHd()}
              >
                <Text style={styles.tryHdButtonText}>
                  ▶ Try HD from your linked source
                </Text>
              </TouchableOpacity>
            ))}

          {/* Escape hatch: manually return to the stored copy when the HD
              stream misbehaves in ways we can't auto-detect. */}
          {hdMode && (
            <TouchableOpacity
              testID="force-derivative"
              style={styles.backToStoredButton}
              onPress={() => void handleBackToDerivative()}
            >
              <Text style={styles.backToStoredText}>Back to stored copy</Text>
            </TouchableOpacity>
          )}

          {/* Honest note when the linked source was unavailable and we fell
              back to (or stayed on) the stored derivative. */}
          {fellBack && (
            <Text style={styles.fallbackNote} testID="source-fallback-note">
              original source unavailable — playing stored copy
            </Text>
          )}

          {/* Honest note when even the stored copy never reported a duration —
              nothing better to switch to. */}
          {mediaStuck && (
            <Text style={styles.fallbackNote} testID="media-stuck-note">
              this recording’s media isn’t loading
            </Text>
          )}

          {/* Overlay toggle — only meaningful for video with a chart. */}
          {isVideo && hasChart && (
            <TouchableOpacity
              testID="overlay-toggle"
              style={styles.overlayToggle}
              onPress={() => setOverlayMode((v) => !v)}
            >
              <Text style={styles.overlayToggleText}>
                {overlayMode ? "Stacked view" : "Overlay chart on video"}
              </Text>
            </TouchableOpacity>
          )}

          {overlayMode && isVideo && hasChart && mediaUrl ? (
            // Overlay mode: chart floats over the bottom third of the video.
            // (`mediaUrl` is always set for video; the guard just narrows the
            // type now that a live session may reach this branch's sibling
            // with no media at all.)
            <>
            <View style={styles.overlayWrap} testID="replay-overlay">
              <MediaPlayer
                ref={playerRef}
                uri={mediaUrl}
                mediaType={detail.media_type}
                onPositionChange={handlePositionChange}
                onDurationChange={handleDurationChange}
                onError={handlePlayerError}
              />
              <View style={styles.overlayChart} pointerEvents="box-none">
                {chart}
              </View>
            </View>
            {auditionBlock}
            </>
          ) : (
            // Default stacked layout: player on top, chart beneath.
            <>
              {noMedia || !mediaUrl ? (
                <View style={styles.noMediaCard} testID="replay-no-media">
                  <Text style={styles.noMediaText}>
                    Live session — no audio was kept. The transcript and
                    analysis are below.
                  </Text>
                </View>
              ) : (
                <MediaPlayer
                  ref={playerRef}
                  uri={mediaUrl}
                  mediaType={detail.media_type}
                  onPositionChange={handlePositionChange}
                  onDurationChange={handleDurationChange}
                  onError={handlePlayerError}
                />
              )}
              <View style={styles.chartCard}>
                {hasChart ? (
                  <>
                    <Text style={styles.sectionTitle}>
                      Heat over the conversation
                    </Text>
                    {chart}
                    {auditionBlock}
                    <Text style={styles.hint}>
                      {noMedia
                        ? "Each dash is one turn across the session — length shows how long they spoke."
                        : "Each dash is one turn across the recording — length shows how long they spoke. Tap a dash to jump there."}
                    </Text>
                  </>
                ) : (
                  <Text style={styles.noAnalysis} testID="replay-no-analysis">
                    {detail.analysis?.live?.analysis_status === "lite"
                      ? "The heat analysis for this live session is still running — check back in a moment."
                      : "This recording hasn’t been analyzed, so there’s no heat graph to sync."}
                  </Text>
                )}
              </View>
            </>
          )}

          {/* Experimental (Advanced → "Experimental voice engine"): run the
              bake-off's window engine over THIS recording's audio on the
              phone and show its own separation under the chart. Owner-only
              and only where audio is stored; the row hides itself when the
              switch is off. */}
          {!isShared && !noMedia ? (
            <DeviceDiarizationRow recordingId={recordingId} onReanalyzeJob={handleReanalyzeJob} />
          ) : null}

          {/* Track 2 — "Your tone": the live session's self / per-person tone
              summary from the phone's per-turn tone + identity. Only live
              sessions carry it; uploads render nothing here. */}
          {liveScoreboard ? (
            <ScoreboardPanel
              board={liveScoreboard}
              nameOf={(raw) => speakerLabel(raw, effectiveLabels)}
              testID="replay-scoreboard"
            />
          ) : null}

          {detail.analysis?.live?.tone_summary ? (
            <ToneSummaryCard
              summary={detail.analysis.live.tone_summary}
              testID="replay-tone-summary"
            />
          ) : null}

          {/* Track 2 — "What you could have said": the LLM reflection over
              the user's OWN turns. Served from the episode's cache when the
              server already ran it (ingest schedules one); otherwise the
              owner can ask for it here. Never runs twice for the same words. */}
          <View style={styles.reflectSection} testID="replay-reflect-section">
            <Text style={styles.sectionTitle}>What you could have said</Text>
            {reflections && reflections.length > 0 ? (
              <CouldHaveSaidList
                items={reflections}
                turns={turns}
                testID="replay-could-have-said"
              />
            ) : reflections && reflections.length === 0 ? (
              <Text style={styles.reflectEmpty} testID="replay-reflect-empty">
                Nothing to add — your turns already landed well.
              </Text>
            ) : null}
            {!isShared && !(reflections && reflections.length > 0) && (
              reflecting ? (
                <View style={styles.reflectRow} testID="replay-reflect-pending">
                  <ActivityIndicator color={PRIMARY} />
                  <Text style={styles.reflectPendingText}>
                    Reading your turns…
                  </Text>
                </View>
              ) : (
                <TouchableOpacity
                  testID="replay-reflect-button"
                  style={styles.reflectButton}
                  onPress={() => void handleReflect()}
                  accessibilityRole="button"
                >
                  <Text style={styles.reflectButtonText}>
                    {reflections ? "Reflect again" : "Reflect on my turns"}
                  </Text>
                </TouchableOpacity>
              )
            )}
            {isShared && !(reflections && reflections.length > 0) && (
              <Text style={styles.reflectEmpty} testID="replay-reflect-shared">
                The owner hasn’t reflected on this recording yet.
              </Text>
            )}
            {reflectError && (
              <Text style={styles.reflectError} testID="replay-reflect-error">
                {reflectError}
              </Text>
            )}
          </View>

          {/* Re-analyze with the latest engine. Reuses the job-progress card
              pattern; on completion the recording is refetched so the fresh
              analysis renders. Honest errors (e.g. a 422 when no audio was
              kept) surface below instead of a silent no-op. Owner-only — a
              recipient can't spend a re-analysis on someone else's recording,
              and a live session kept no audio to re-run. */}
          {!isShared && !noMedia && (
          <View style={styles.reanalyzeSection}>
            {!reanalyzing ? (
              <TouchableOpacity
                testID="reanalyze-button"
                style={styles.reanalyzeButton}
                onPress={() => void handleReanalyze()}
                accessibilityRole="button"
              >
                <Text style={styles.reanalyzeButtonText}>
                  ↻ Re-analyze with the latest engine
                </Text>
              </TouchableOpacity>
            ) : (
              <View style={styles.jobProgress} testID="reanalyze-progress">
                <View style={styles.reanalyzeRow}>
                  <ActivityIndicator color={PRIMARY} />
                  <Text
                    style={styles.jobStageLabel}
                    testID="reanalyze-stage-label"
                  >
                    {reanalyzeStageLabel(reanalyzeJob?.status)}
                  </Text>
                </View>
                {reanalyzeJob?.status === "stalled" ? (
                  <Text style={styles.jobNote} testID="reanalyze-progress-note">
                    Still working — this is taking longer than expected.
                  </Text>
                ) : (
                  reanalyzeJob?.progress_note && (
                    <Text style={styles.jobNote} testID="reanalyze-progress-note">
                      {reanalyzeJob.progress_note}
                    </Text>
                  )
                )}
              </View>
            )}
            {reanalyzeError && (
              <Text style={styles.reanalyzeError} testID="reanalyze-error">
                {reanalyzeError}
              </Text>
            )}

            {/* Honest "what changed" read after a completed re-analysis. The
                pulse marker only appears when a comparable number actually
                moved — it points at real signal, never decorates a no-op. */}
            {reanalyzeSummary && (
              <View style={styles.reanalyzeSummary} testID="reanalyze-summary">
                <View style={styles.reanalyzeSummaryHead}>
                  {reanalyzeSummary.changed && (
                    <PulseDot color={PRIMARY} size={9} testID="reanalyze-pulse" />
                  )}
                  <Text style={styles.reanalyzeSummaryTitle}>
                    {reanalyzeSummary.changed
                      ? "Updated — here’s what changed"
                      : "No change — the latest engine read this the same"}
                  </Text>
                </View>

                {reanalyzeSummary.scoreDeltas.some((d) => d.delta !== 0) && (
                  <Text style={styles.deltaCaption}>Conduct score</Text>
                )}
                {reanalyzeSummary.scoreDeltas
                  .filter((d) => d.delta !== 0)
                  .map((d) => (
                    <View
                      key={d.id}
                      style={styles.deltaRow}
                      testID={`reanalyze-delta-${d.id}`}
                    >
                      <Text style={styles.deltaLabel} numberOfLines={1}>
                        {d.label}
                      </Text>
                      <View style={styles.deltaValues}>
                        <Text style={styles.deltaOld}>{d.before}</Text>
                        <Text style={styles.deltaArrow}>→</Text>
                        <Text
                          style={[
                            styles.deltaNew,
                            { color: d.delta > 0 ? GOOD : DANGER },
                          ]}
                        >
                          {d.after}
                        </Text>
                        <Text
                          style={[
                            styles.deltaBadge,
                            d.delta > 0
                              ? styles.deltaBadgeUp
                              : styles.deltaBadgeDown,
                          ]}
                        >
                          {d.delta > 0 ? `+${d.delta}` : `${d.delta}`}
                        </Text>
                      </View>
                    </View>
                  ))}

                {reanalyzeSummary.changed &&
                  reanalyzeSummary.peakDelta != null &&
                  reanalyzeSummary.peakDelta !== 0 && (
                    <Text style={styles.deltaPeak} testID="reanalyze-peak-delta">
                      Peak heat {reanalyzeSummary.peakBefore} →{" "}
                      {reanalyzeSummary.peakAfter}
                    </Text>
                  )}
              </View>
            )}
          </View>
          )}

          {/* Share this recording with another account (owner-only, read-only
              grant). Hidden in read-only shared mode. Seeded with the owner's
              current shares from the detail read. */}
          {!isShared && (
            <RecordingShareManager
              recordingId={recordingId}
              initialShares={detail.shares ?? []}
            />
          )}

          {/* "Name the speakers" — manually label who each diarized speaker is,
              for THIS recording (the human override, top of the label ladder).
              Distinct from "This is me" below: naming labels this one recording;
              enrollment teaches your voice for every recording. Self-hides on an
              older server (no `manual_speaker_labels` field / a save 404s), and
              is owner-only — a recipient can't relabel someone else's recording. */}
          {!isShared && turns.length > 0 && namingSupported ? (
            <SpeakerNaming
              recordingId={recordingId}
              turns={turns}
              speakerLabels={effectiveLabels}
              manualLabels={manualLabels}
              onSaved={handleLabelsSaved}
              onUnsupported={() => setNamingUnsupported(true)}
              // People labeling: "Who is this?" per speaker (enrolled people +
              // "New person…" + "Remember this voice"). Only when the server
              // has people; a live session kept no audio to learn from.
              people={people ?? undefined}
              hasAudio={!noMedia}
              onPeopleChanged={() => void refreshPeople()}
              enrollment={noMedia ? undefined : enrollment}
            />
          ) : null}

          {/* "This is me" standalone card — ONLY the fallback for an older
              server without speaker naming; otherwise it lives inline in the
              rows above. Self-hides when the server can't do voice ID or there
              are no diarized turns. Owner-only. */}
          {!isShared && turns.length > 0 && !noMedia && !namingSupported ? (
            <SpeakerEnrollment
              recordingId={recordingId}
              turns={turns}
              speakerLabels={effectiveLabels}
            />
          ) : null}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  liveBadge: {
    fontSize: 12.5,
    color: PRIMARY,
    fontWeight: "600",
    textAlign: "center",
    paddingHorizontal: 16,
    paddingTop: 2,
  },
  noMediaCard: {
    backgroundColor: "#EAF2FB",
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
  },
  noMediaText: {
    fontSize: 13,
    color: "#2F5F9E",
    textAlign: "center",
  },
  reflectSection: {
    marginBottom: 16,
  },
  reflectEmpty: {
    fontSize: 13,
    color: MUTED,
    marginBottom: 8,
  },
  reflectRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingVertical: 8,
  },
  reflectPendingText: {
    fontSize: 13.5,
    color: MUTED,
  },
  reflectButton: {
    alignSelf: "flex-start",
    backgroundColor: PRIMARY,
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 10,
  },
  reflectButtonText: {
    color: "#FFFFFF",
    fontSize: 14,
    fontWeight: "700",
  },
  reflectError: {
    fontSize: 13,
    color: DANGER,
    marginTop: 8,
  },
  flex: { flex: 1, backgroundColor: "#F9FAFB" },
  header: {
    flexDirection: "row",
    alignItems: "center",
    // App's SafeAreaView already applies the notch inset; this base pad matches
    // the hub screens (~20-24) instead of the old hardcoded 56 that double-padded
    // on notched devices.
    paddingTop: 24,
    paddingBottom: 12,
    paddingHorizontal: 16,
    backgroundColor: "#FFFFFF",
    borderBottomWidth: 1,
    borderBottomColor: "#E5E7EB",
  },
  backText: { fontSize: 16, color: PRIMARY, fontWeight: "600", width: 64 },
  titleWrap: { flex: 1, alignItems: "center" },
  headerTitle: {
    flex: 1,
    textAlign: "center",
    fontSize: 17,
    fontWeight: "700",
    color: INK,
  },
  headerSpacer: { width: 64 },
  titleEditRow: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  renameInput: {
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
  renameSave: { fontSize: 15, fontWeight: "700", color: PRIMARY },
  renameCancel: { fontSize: 15, fontWeight: "600", color: MUTED },
  renameNote: {
    fontSize: 12.5,
    color: MUTED,
    fontStyle: "italic",
    textAlign: "center",
    paddingHorizontal: 16,
    paddingTop: 8,
  },
  recordedAt: {
    fontSize: 13,
    color: MUTED,
    fontWeight: "600",
    textAlign: "center",
    paddingHorizontal: 16,
    paddingTop: 8,
  },
  sharedFrom: {
    fontSize: 12.5,
    color: PRIMARY,
    fontWeight: "600",
    textAlign: "center",
    paddingHorizontal: 16,
    paddingTop: 6,
  },
  photosHdNote: {
    fontSize: 12.5,
    lineHeight: 18,
    color: MUTED,
    fontStyle: "italic",
    marginBottom: 10,
  },
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  centeredText: { marginTop: 12, color: MUTED, fontSize: 14 },
  errorTitle: { fontSize: 18, fontWeight: "700", color: INK, marginBottom: 6 },
  errorText: {
    fontSize: 14,
    color: MUTED,
    textAlign: "center",
    marginBottom: 16,
  },
  retryButton: {
    backgroundColor: PRIMARY,
    paddingVertical: 12,
    paddingHorizontal: 28,
    borderRadius: 10,
  },
  retryText: { color: "#FFFFFF", fontSize: 15, fontWeight: "600" },
  content: { padding: 16, paddingBottom: 40 },
  chartCard: {
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
    marginBottom: 12,
  },
  hint: {
    fontSize: 12.5,
    color: MUTED,
    fontStyle: "italic",
    marginTop: 8,
  },
  noAnalysis: {
    fontSize: 14,
    lineHeight: 20,
    color: MUTED,
  },
  hdBadge: {
    alignSelf: "flex-start",
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#EAF3FC",
    borderColor: PRIMARY,
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 4,
    marginBottom: 10,
  },
  hdBadgeText: { fontSize: 12, fontWeight: "700", color: PRIMARY },
  tryHdButton: {
    alignSelf: "flex-start",
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: PRIMARY,
    backgroundColor: "#EEF2FF",
    marginBottom: 10,
  },
  tryHdButtonText: { fontSize: 14, fontWeight: "600", color: PRIMARY },
  backToStoredButton: {
    alignSelf: "flex-start",
    paddingHorizontal: 12,
    paddingVertical: 6,
    marginBottom: 10,
  },
  backToStoredText: {
    fontSize: 13,
    fontWeight: "600",
    color: MUTED,
    textDecorationLine: "underline",
  },
  fallbackNote: {
    fontSize: 12,
    color: MUTED,
    fontStyle: "italic",
    marginBottom: 10,
  },
  attachSection: { marginBottom: 12 },
  attachSourceButton: {
    alignSelf: "flex-start",
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: PRIMARY,
    backgroundColor: "#EEF2FF",
  },
  attachSourceButtonText: { fontSize: 14, fontWeight: "600", color: PRIMARY },
  replaceSourceButton: {
    alignSelf: "flex-start",
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  replaceSourceText: {
    fontSize: 13,
    fontWeight: "600",
    color: MUTED,
    textDecorationLine: "underline",
  },
  attachHelp: {
    fontSize: 12.5,
    lineHeight: 18,
    color: MUTED,
    marginBottom: 8,
  },
  attachInput: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
    color: INK,
    backgroundColor: "#FFFFFF",
  },
  attachButtonRow: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 10,
    marginTop: 10,
  },
  attachCancel: {
    paddingVertical: 10,
    paddingHorizontal: 18,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#D1D5DB",
  },
  attachCancelText: { color: MUTED, fontSize: 14, fontWeight: "600" },
  attachSubmit: {
    paddingVertical: 10,
    paddingHorizontal: 22,
    borderRadius: 8,
    backgroundColor: PRIMARY,
    alignItems: "center",
    justifyContent: "center",
    minWidth: 90,
  },
  attachDisabled: { opacity: 0.6 },
  attachSubmitText: { color: "#FFFFFF", fontSize: 14, fontWeight: "600" },
  attachError: {
    marginTop: 10,
    fontSize: 13,
    lineHeight: 19,
    color: "#DC2626",
  },
  overlayToggle: {
    alignSelf: "flex-end",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#FFFFFF",
    marginBottom: 10,
  },
  overlayToggleText: { fontSize: 12, fontWeight: "600", color: MUTED },
  overlayWrap: {
    position: "relative",
  },
  overlayChart: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    padding: 8,
    backgroundColor: "rgba(255,255,255,0.75)",
    borderTopLeftRadius: 12,
    borderTopRightRadius: 12,
  },
  // Audition ("Play this voice") — sits directly under the heat chart while a
  // speaker is isolated via the legend.
  auditionRow: {
    marginTop: 10,
    flexDirection: "row",
    alignItems: "center",
  },
  auditionPlayButton: {
    alignSelf: "flex-start",
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: PRIMARY,
    backgroundColor: "#EEF2FF",
  },
  auditionPlayText: { fontSize: 14, fontWeight: "700", color: PRIMARY },
  auditionStopButton: {
    alignSelf: "flex-start",
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: MUTED,
    backgroundColor: "#F3F4F6",
  },
  auditionStopText: { fontSize: 14, fontWeight: "700", color: INK },
  auditionUnavailable: {
    fontSize: 12.5,
    lineHeight: 18,
    color: MUTED,
    fontStyle: "italic",
  },
  reanalyzeSection: { marginTop: 16 },
  reanalyzeButton: {
    alignSelf: "flex-start",
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: PRIMARY,
    backgroundColor: "#EEF2FF",
  },
  reanalyzeButtonText: { fontSize: 14, fontWeight: "700", color: PRIMARY },
  jobProgress: {
    padding: 14,
    borderRadius: 10,
    backgroundColor: "#F9FAFB",
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  reanalyzeRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  jobStageLabel: { fontSize: 15, fontWeight: "700", color: INK },
  jobNote: { marginTop: 8, fontSize: 13, color: MUTED },
  reanalyzeError: {
    marginTop: 10,
    fontSize: 13,
    lineHeight: 19,
    color: "#DC2626",
  },
  reanalyzeSummary: {
    marginTop: 12,
    padding: 14,
    borderRadius: 10,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  reanalyzeSummaryHead: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  reanalyzeSummaryTitle: {
    flex: 1,
    fontSize: 14,
    fontWeight: "700",
    color: INK,
  },
  deltaCaption: {
    marginTop: 10,
    fontSize: 11,
    fontWeight: "700",
    color: MUTED,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  deltaRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginTop: 6,
  },
  deltaLabel: { flex: 1, fontSize: 14, fontWeight: "600", color: INK },
  deltaValues: { flexDirection: "row", alignItems: "center", gap: 6 },
  deltaOld: {
    fontSize: 14,
    color: MUTED,
    textDecorationLine: "line-through",
  },
  deltaArrow: { fontSize: 13, color: MUTED },
  deltaNew: { fontSize: 15, fontWeight: "800" },
  deltaBadge: {
    fontSize: 12,
    fontWeight: "800",
    overflow: "hidden",
    borderRadius: 8,
    paddingHorizontal: 6,
    paddingVertical: 1,
  },
  deltaBadgeUp: { color: GOOD, backgroundColor: "#E7F6EE" },
  deltaBadgeDown: { color: DANGER, backgroundColor: "#FEECEC" },
  deltaPeak: {
    marginTop: 10,
    fontSize: 13,
    color: MUTED,
  },
});
