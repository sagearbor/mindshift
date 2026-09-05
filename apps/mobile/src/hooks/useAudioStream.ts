import { useState, useRef, useCallback, useEffect } from "react";
import { Platform } from "react-native";
import {
  useAudioStream as useMicrophoneStream,
  requestRecordingPermissionsAsync,
} from "expo-audio";
import type { AudioStreamBuffer } from "expo-audio";
import { setRecordingMode, setPlaybackMode } from "../utils/audioMode";
import * as Speech from "expo-speech";
import {
  concatInt16,
  downmixToMono,
  float32ToInt16,
  StreamingResampler,
} from "../utils/audio";
import {
  WebAudioCapture,
  WebCaptureError,
  isWebAudioCaptureSupported,
} from "../utils/webAudioCapture";
import { unlockWebSpeechSynthesis, webSpeechSynthesisAvailable } from "../utils/webSpeech";
import { getCachedToken } from "../auth/authToken";
import type { FastLoop, SpeakerBinding } from "../live/fastLoop";
import { formatLatencyLog } from "../live/fastLoop";
import { PleasantnessTracker, type Scoreboard } from "../live/pleasantness";
import {
  enrollSpeakerAudio,
  MIN_ENROLL_SECONDS,
  type EnrollFromSessionResult,
} from "../live/enrollFromSession";
import { patchSpeakerLabels } from "../api/client";
import { SELF_PERSON_ID } from "../utils/people";
import type {
  FastLoopBuild,
  FastLoopCapabilities,
  FastLoopHandlers,
} from "../live/defaultDeps";
import { createDefaultFastLoop, expoHaptics, probeFastLoopCapabilities } from "../live/defaultDeps";
import { createWebFastLoop, primeWebRecognizer, probeWebFastLoopCapabilities } from "../live/webDeps";
import type { SpeechRecognizer } from "../live/stt";
import type { TurnLatency } from "../live/fastLoop";
import { summarizeSession, type SessionSummary } from "../live/sessionSummary";
import { useLiveEpisodeStore } from "../store/liveEpisodeStore";
import { useMoodStore } from "../store/moodStore";
import { detectLiveCapability, type LiveCapability } from "../live/capability";
import type { LiveMode } from "../live/localLlm";
import type { NudgeEvent } from "../live/nudgePolicy";
import type {
  SpeakerIdentityEvent,
  ToneFlagEvent,
  TurnLocalEvent,
} from "../live/types";
import {
  postLiveSession,
  postLiveSessionAudio,
  type LiveSessionAudioUploader,
  type LiveSessionBody,
  type LiveSpeakerLabel,
  type PostLiveSessionResult,
} from "../api/liveSessions";
import {
  openDefaultLiveAudioKeeper,
  type LiveAudioKeeper,
  type LiveAudioKeeperFactory,
} from "../live/liveAudioKeeper";
import type { LiveSessionMode } from "../api/liveSessions";
import { IDLE_JOURNAL_STATE, type JournalRecorder, type JournalState } from "../live/journalRecorder";
import {
  createBackgroundMicHold,
  createDefaultJournalRecorder,
  prepareJournalAudioSession,
  type BackgroundMicHold,
  type JournalHandlers,
} from "../live/journalDeps";
import { CallSession } from "../live/call/callSession";
import { callApi as defaultCallApi, type CallApi } from "../live/call/callApi";
import { createNativeRtcAdapter } from "../live/call/rtcNative";
import { createWebRtcAdapter } from "../live/call/callWeb";
import type { AudioRoute, RtcAdapter } from "../live/call/rtc";
import { IDLE_CALL_VIEW, type CallClientMessage, type CallRole, type CallView } from "../live/call/types";
import { summarizeLatency, summarizeSpeakerId, useDiagnosticsStore, type SessionDiagnostics } from "../diagnostics/diagnostics";
import { useAuthStore } from "../store/authStore";

const API_URL =
  process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

const WS_BASE = API_URL.replace(/^http/, "ws");

export interface TranscriptEntry {
  /** What the line SHOWS: the person's name once known, else the raw label. */
  speaker: string;
  /** The raw wire label behind `speaker` ("Speaker B") — the key mid-call
   *  naming binds. Absent on legacy-path lines that were never relabeled. */
  speakerId?: string;
  /** NaturalTurn tag (live/naturalTurn.ts): "backchannel" lines are listener
   *  noises ("yeah") — shown in the transcript but not counted as turns.
   *  Absent on server/legacy lines (treated as primary). */
  kind?: "primary" | "backchannel";
  /** Vocal-activation probability of the user's own line (live/activation.ts);
   *  Developer-mode tag only. Absent on others' lines / server lines. */
  activation?: number;
  /** true = the phone's owner, false = a partner, null/undefined = unknown.
   *  Feeds the dev-mode conversation-dynamics block (live/sessionSummary.ts,
   *  live/conversationDynamics.ts) — absent on lines where speaker identity
   *  was never resolved. */
  isSelf?: boolean | null;
  /** In a CALL: the row's position in the server's merged transcript
   *  (`transcript.seq`). The identity of the line — a later frame carrying
   *  this seq (directly, or as its `replaces_seq`) corrects it in place
   *  rather than appending a second copy. Absent outside a call. */
  callSeq?: number;
  text: string;
  timestamp: number;
  /** Utterance boundaries in seconds (from the server's transcript events).
   *  Undefined on the legacy suggestion-event fallback path, which carries no
   *  timing — downstream consumers (e.g. /analyze interruption stats) must
   *  treat them as genuinely unknown there, never as 0. */
  startTime?: number;
  endTime?: number;
}

/** "response" = coaching lines about the OTHER person's turn (the normal
 *  case). "nudge" = a single ≤6-word delivery cue about the user's OWN
 *  just-finished turn (e.g. "ease up"). Absent on old servers → "response". */
export type SuggestionKind = "response" | "nudge";

export interface SuggestionEntry {
  /** Monotonic, unique per event: a stable React key and a strict ordering
   *  even for two events landing in the same millisecond. */
  id: number;
  kind: SuggestionKind;
  /** The suggestion strings (a single element for a nudge). */
  texts: string[];
  tone: string;
  /** True when the server said not to voice this suggestion (speak: false).
   *  Rendered dimmed in the UI and never passed to speakSuggestion. */
  muted: boolean;
  timestamp: number;
  /** Which runtime produced it: the phone's fast loop or the server. Absent
   *  on the legacy path (every legacy suggestion is the server's). */
  source?: "on-device" | "cloud";
  /** A streaming preview from the server (SuggestionEvent.partial): the
   *  first suggestion string while the model is still writing. Never voiced;
   *  replaced by the final event for the same turn. */
  partial?: boolean;
  /** In-app call, therapist view: this is a read-only copy of coaching meant
   *  for another participant (`for_uid`), shown with their name. Never
   *  voiced, never counted as one of the viewer's own escalations. */
  forName?: string;
}

type ConnectionStatus = "idle" | "connecting" | "live" | "disconnected";

/**
 * The web build's gesture-bound start, done synchronously inside the tap
 * (see startWebSession): the capture graph, the primed recognizer and the
 * capture's start promise (already caught, so a permission refusal that
 * lands before it's awaited is never an unhandled rejection).
 */
interface PreparedWebCapture {
  capture: WebAudioCapture;
  primed: SpeechRecognizer | null;
  started: Promise<{ error: unknown } | null>;
}

/**
 * Seams for the on-device fast loop (Track 3). Production wires the native
 * stack (src/live/defaultDeps.ts); tests inject fakes here and drive the
 * real orchestrator with synthetic PCM.
 */
export interface UseAudioStreamOptions {
  /** Override the device capability probe. */
  capability?: LiveCapability;
  /** Build the fast loop for a session. */
  makeFastLoop?: (handlers: FastLoopHandlers, mode: LiveMode) => Promise<FastLoopBuild>;
  /** POST the finished session (Track 2's /sessions/live). */
  postSession?: (body: LiveSessionBody) => Promise<PostLiveSessionResult>;
  /** Pre-flight capability probe (what the loop would load right now). */
  probeCapabilities?: () => Promise<FastLoopCapabilities>;
  /** Mid-call naming: upload a speaker's pooled session audio as a new
   *  person's voiceprint (production: live/enrollFromSession.ts). */
  enrollSpeaker?: (
    pcm: Float32Array,
    person: { personId: string; displayName: string },
  ) => Promise<EnrollFromSessionResult>;
  /** Post-session naming once the episode exists on the server: the same
   *  PATCH "Who is this?" makes on a stored recording. */
  patchLabels?: (
    recordingId: string,
    labels: Record<string, string>,
    people?: Record<string, string>,
  ) => Promise<unknown>;
  /** In-app calls: the REST client (create/join/end) and the WebRTC
   *  adapter (production: react-native-webrtc on the phone, the browser's
   *  RTCPeerConnection on the web; tests: fakes). */
  callApi?: CallApi;
  makeRtcAdapter?: (getCaptureStream: () => MediaStream | null) => RtcAdapter;
  /** Keep the session's mic audio on the phone as a WAV and attach it to
   *  the saved episode once `POST /sessions/live` has stored it (default
   *  true; the Live Coach "keep audio" switch). False writes nothing. */
  keepAudio?: boolean;
  /** Seam: open the per-session WAV keeper (production: one file per
   *  session under Paths.cache/live-audio; tests: MemoryFs). */
  openAudioKeeper?: LiveAudioKeeperFactory;
  /** Seam: upload the kept WAV to the saved episode
   *  (production: api/liveSessions.ts postLiveSessionAudio). */
  postSessionAudio?: LiveSessionAudioUploader;
  /** Journal mode: build the recorder (production: live/journalDeps.ts —
   *  Silero + ECAPA + the enrolled prints, the cache store, the chunked
   *  upload job; tests: fakes over MemoryFs). Rejects with the reason when
   *  the models or the store are unavailable. */
  makeJournal?: (handlers: JournalHandlers) => Promise<JournalRecorder>;
  /** Journal mode: configure the audio session for an all-day mic
   *  (production: background flags + the Android notification permission). */
  prepareJournalAudio?: () => Promise<unknown>;
  /** Journal mode: the best-effort Android foreground-service hold (see
   *  journalDeps.createBackgroundMicHold). Null = none. */
  holdBackgroundMic?: () => Promise<BackgroundMicHold | null>;
}

/** The kept session audio, as the screen shows it.
 *  - off: not keeping (switch off, web, disk full, or the session was never
 *    saved on the server — `error` says which when there is a reason);
 *  - keeping: the WAV is growing on the phone;
 *  - uploading / kept: attaching it to the saved episode / done (the local
 *    file is deleted once the server has it);
 *  - failed: the upload failed — the file stays on the phone and `retry`
 *    sends it again. The session record itself is never affected. */
export interface AudioKeepState {
  status: "off" | "keeping" | "uploading" | "kept" | "failed";
  bytes?: number;
  seconds?: number;
  error?: string;
  retry?: () => Promise<void>;
}

/** What the sheet is told after a mid-call "that's Mom". */
export interface LabelSpeakerOutcome {
  /** Human sentence for the sheet's done stage. */
  text: string;
  /** A voiceprint sample was stored for a new person. */
  enrolled: boolean;
  /** Seconds of the speaker's pooled audio available at the time. */
  seconds: number | null;
}

export interface LabelSpeakerChoice {
  personId: string;
  displayName: string;
  isSelf: boolean;
  isNew: boolean;
}

/** The pre-session capability check, as the screen shows it. */
export type PreflightState =
  | { status: "probing" }
  | { status: "ready"; capabilities: FastLoopCapabilities }
  | { status: "failed"; reason: string };

/** The server's record of the session that just ended (Track 2 ingest). */
export interface LastEpisode {
  episodeId: string | null;
  /** "created" = stored; "unsupported" = server predates /sessions/live;
   *  "failed" = the POST failed (the transcript is still on screen). */
  postStatus: "created" | "unsupported" | "failed";
  /** Therapist emails the server auto-shared it with at ingest. */
  sharedWith: string[];
}

interface UseAudioStreamReturn {
  isRecording: boolean;
  /** True while a session is running, even when mic capture is unavailable
   *  (e.g. web) and no audio is being recorded. Drives the start/stop toggle. */
  sessionActive: boolean;
  transcript: TranscriptEntry[];
  /** Accumulating suggestion feed, newest FIRST, capped at MAX_SUGGESTION_FEED.
   *  A live conversation moves fast — replacing on every event means a glance a
   *  second late finds the advice already gone. */
  suggestions: SuggestionEntry[];
  speakerLabel: string;
  /** Which diarized speaker is the coached user ("Speaker A" | "Speaker B" |
   *  null). Diarization labels are assigned PER SESSION by speaking order —
   *  "Speaker A" is whoever speaks first in THAT session, not a stable
   *  identity — so this resets to "Speaker A" (the "you speak first"
   *  convention) at every session start. It toggles freely within a session. */
  selfSpeaker: string | null;
  setSelfSpeaker: (label: string) => void;
  connectionStatus: ConnectionStatus;
  transcriptionAvailable: boolean;
  transcriptionMessage: string;
  micError: string;
  /** True when on-device text-to-speech can actually produce sound here.
   *  False (e.g. a browser without the Web Speech API) means suggestions are
   *  visual-only — an honest state, never a fake "spoken" claim. */
  speechAvailable: boolean;
  /** True when new top suggestions should be spoken aloud (earpiece mode). */
  speechEnabled: boolean;
  setSpeechEnabled: (enabled: boolean) => void;
  startSession: (
    sessionId: string,
    empathyLevel: number,
    interjectLevel?: number,
  ) => Promise<void>;
  stopSession: () => Promise<void>;
  sendEmpathyUpdate: (level: number) => void;
  sendInterjectUpdate: (value: number) => void;
  /** On-device fast loop: can this device run it (on-device STT present)? */
  liveCapable: boolean;
  liveCapabilityReason: string;
  /** Whether the next session runs the fast loop (default: on when capable).
   *  Off = the legacy server path, unchanged. */
  liveMode: boolean;
  setLiveMode: (on: boolean) => void;
  /** Session shape for the fast loop: earpiece (speak), in person (`speaker`:
   *  both voices on one mic; speak only in silences), therapist (on-screen
   *  only), call (an in-app call; only the user's voice on this mic). */
  sessionMode: LiveMode;
  setSessionMode: (mode: LiveMode) => void;
  /** What the fast loop actually loaded, or why it isn't running. Empty on
   *  the legacy path. */
  liveStatus: string;
  /** Latest haptic nudge on the user's own delivery (level 1–3); the screen
   *  shows it briefly and clears it. */
  nudgeFlash: NudgeEvent | null;
  clearNudgeFlash: () => void;
  /** One-line latency report after a live session ends. */
  latencySummary: string;
  /** Server tone flags (newest first), rendered additively in live mode. */
  toneFlags: ToneFlagEvent[];
  /** Tier B: true while a paired watch has a live/companion socket open on
   *  the server (`watch_connected` frames) — nudges are reaching the wrist. */
  watchConnected: boolean;
  /** Pre-session capability check; null until `runPreflight` is called. */
  preflight: PreflightState | null;
  /** Probe what the on-device loop would load (no session started). No-op
   *  on a device that can't run the loop at all. */
  runPreflight: () => Promise<void>;
  /** Nudges (level ≥ 1) raised on the user's own delivery this session. */
  escalationCount: number;
  /** Numbers for the end-of-session card; null until a session has ended. */
  sessionSummary: SessionSummary | null;
  /** The server's record of the last finished session (null until then, and
   *  null on the legacy path where nothing is POSTed). */
  lastEpisode: LastEpisode | null;
  /** The session's kept audio (see AudioKeepState). */
  audioKeep: AudioKeepState;
  /** Journal mode ("listen for my voice"): what the journal is doing —
   *  listening time, kept stretches, file size, uploads. IDLE otherwise. */
  journal: JournalState;
  /** Journal mode: send every journal file still on the phone now. */
  retryJournalUploads: () => Promise<void>;
  /** Mid-call naming: raw wire label → the person the user (or a voiceprint
   *  match) says it is. Reset per session. */
  speakerNames: Record<string, SpeakerBinding>;
  /** What to show for a raw label: its bound name, else the label itself. */
  displayNameOf: (speaker: string) => string;
  /** Name a speaker for the rest of the session (and, when the session
   *  is already stored, on its episode). Never throws — the outcome text
   *  says what happened. */
  labelSpeaker: (speaker: string, choice: LabelSpeakerChoice) => Promise<LabelSpeakerOutcome>;
  /** The pleasantness scoreboard over this session's on-device turns,
   *  keyed by raw label (see live/pleasantness.ts). Null before any turn. */
  scoreboard: Scoreboard | null;
  /** In-app call (Call mode): what the call is doing right now. */
  call: CallView;
  /** Create a call on the server, start the session and wait for the other
   *  person; the invite (code + link) lands in `call`. Forces Call mode. */
  startCall: (empathyLevel: number, interjectLevel?: number) => Promise<void>;
  /** Join a call by its code (typed, or from an invite link) and start the
   *  session. On the web this MUST be called from the Answer tap. */
  joinCall: (code: string, empathyLevel: number, interjectLevel?: number, role?: CallRole) => Promise<void>;
  /** Hang up: ends the call for both sides and stops the session. */
  hangUp: () => Promise<void>;
  setCallMuted: (muted: boolean) => void;
  /** Where the other person's voice comes out on the phone (native only;
   *  the browser decides for itself). Speaker by default. */
  callRoute: AudioRoute;
  setCallRoute: (route: AudioRoute) => void;
}

const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECT_ATTEMPTS = 5;
/** The suggestion feed keeps at most this many entries; older ones drop off
 *  the bottom. Enough to glance back a few turns without unbounded growth. */
const MAX_SUGGESTION_FEED = 20;
/**
 * After a manual stop we keep the socket open so the server can deliver the
 * final utterance's suggestion before `session_complete`. This is an
 * INACTIVITY window, not a fixed deadline: any frame received while draining
 * proves the server is alive and still working (the Whisper path can spend
 * several seconds transcribing the final utterance before emitting the last
 * suggestion), so each frame re-arms the window instead of racing it.
 */
const STOP_DRAIN_TIMEOUT_MS = 4000;
/**
 * Absolute upper bound on the whole drain — however chatty the server is, a
 * manual stop must never leave the UI hanging in a half-stopped state.
 */
const STOP_DRAIN_MAX_MS = 15000;

/**
 * Wire contract with the backend: binary WS frames carry raw PCM,
 * int16 little-endian, 16 kHz, mono, no header. Text frames stay JSON.
 */
const TARGET_SAMPLE_RATE = 16000;
/** ~100 ms of audio per binary frame: 1600 samples = 3200 bytes. */
const SAMPLES_PER_FRAME = 1600;
/**
 * At most ~5 s of audio is buffered while the socket is down (e.g. during a
 * reconnect). Beyond that we drop the oldest audio rather than grow forever.
 */
const MAX_PENDING_SAMPLES = TARGET_SAMPLE_RATE * 5;

/**
 * Maps the empathy slider to the coaching stance label shown on each
 * suggestion. This describes how the suggestion was generated — it is not a
 * claim about detected tone (the server's suggestion event carries no tone).
 */
function empathyTone(slider: number): string {
  if (slider <= 20) return "assertive";
  if (slider <= 50) return "balanced";
  if (slider <= 80) return "empathetic";
  return "validating";
}

/**
 * Whether expo-speech (free, on-device TTS: iOS AVSpeechSynthesizer, Android
 * TextToSpeech, web SpeechSynthesis) can produce sound on this platform.
 * expo-speech's web build calls `window.speechSynthesis` without guarding, so
 * on browsers lacking the Web Speech API we must never call it — detect that
 * up front and degrade honestly (visual suggestions keep working, no crash).
 */
function detectSpeechSupport(): boolean {
  if (Platform.OS !== "web") return true; // iOS/Android ship a TTS engine.
  return webSpeechSynthesisAvailable();
}

/**
 * Stop any in-flight utterance without ever throwing. `Speech.stop()` is a
 * no-op when nothing is speaking, but on a platform with no TTS backend it
 * can reject — swallow that (there was nothing speaking to stop anyway).
 */
function stopSpeechSafely() {
  try {
    void Promise.resolve(Speech.stop()).catch(() => {});
  } catch {
    // No TTS backend — nothing could have been speaking.
  }
}

/** The mode a session RECORD carries. Journal never posts a record (it
 *  uploads files instead), so it never reaches the callers of this; the
 *  narrowing keeps the wire type honest. */
function liveSessionModeOf(mode: LiveMode): LiveSessionMode {
  return mode === "journal" ? "speaker" : mode;
}

/** Production fast-loop factory: the native stack, or the browser stack on
 *  the web build (onnxruntime-web + Web Speech API — src/live/webDeps.ts). */
const defaultMakeFastLoop = (handlers: FastLoopHandlers) =>
  Platform.OS === "web" ? createWebFastLoop(handlers) : createDefaultFastLoop(handlers);
/** Production pre-flight probe (same builders, no loop): the native stack,
 *  or the browser stack on the web build. */
const defaultProbeCapabilities = () =>
  Platform.OS === "web" ? probeWebFastLoopCapabilities() : probeFastLoopCapabilities();

/**
 * While the on-device loop runs, should a CLOUD suggestion be voiced?
 * `recent` is the phone's own turns (oldest first) with whether the phone
 * answered each itself; `utteranceText` is the words the server coached
 * (it echoes the phone's text back verbatim for a turn_local). Exported for
 * tests.
 *
 * - answers the LATEST local turn: only if the phone had nothing to say
 *   for it (its providers fell through) — never two answers to one moment;
 * - answers an EARLIER local turn: never — that moment has passed;
 * - answers words the phone never reported (its VAD missed the span and
 *   the server's transcriber caught it), or carries no text: the cloud is
 *   the only voice there is, unless the latest local turn was already
 *   answered.
 */
export function cloudAnswersOpenMoment(
  recent: readonly { text: string; hadSuggestion: boolean }[],
  utteranceText: string | null,
): boolean {
  const latest = recent.length > 0 ? recent[recent.length - 1] : null;
  if (utteranceText === null) return latest ? !latest.hadSuggestion : true;
  let idx = -1;
  for (let i = recent.length - 1; i >= 0; i--) {
    if (recent[i].text === utteranceText) {
      idx = i;
      break;
    }
  }
  if (idx === -1) return latest ? !latest.hadSuggestion : true;
  if (idx === recent.length - 1) return !recent[idx].hadSuggestion;
  return false;
}

export function useAudioStream(
  options: UseAudioStreamOptions = {},
): UseAudioStreamReturn {
  const [isRecording, setIsRecording] = useState(false);
  const [sessionActive, setSessionActive] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [suggestions, setSuggestions] = useState<SuggestionEntry[]>([]);
  const [speakerLabel, setSpeakerLabel] = useState("");
  // Default "Speaker A" encodes the "you speak first" convention — the server
  // labels the first voice it hears "Speaker A". Reset to this default at
  // every session start (see startSession): diarization labels are assigned
  // per session, so a previous session's toggle would mis-type every turn.
  const [selfSpeaker, setSelfSpeakerState] = useState<string | null>(
    "Speaker A",
  );
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("idle");
  const [transcriptionAvailable, setTranscriptionAvailable] = useState(true);
  const [transcriptionMessage, setTranscriptionMessage] = useState("");
  const [micError, setMicError] = useState("");
  const [speechAvailable, setSpeechAvailable] = useState(detectSpeechSupport);
  const [speechEnabled, setSpeechEnabledState] = useState(false);

  // --- On-device fast loop (Track 3) ---------------------------------------
  // Capability is probed once (synchronously — it's a native module query,
  // no I/O). Live mode defaults to ON when the device can do it; the user can
  // switch it off to get the legacy server path exactly as before.
  const [liveCapability] = useState<LiveCapability>(
    () => options.capability ?? detectLiveCapability(),
  );
  const [liveMode, setLiveModeState] = useState(liveCapability.capable);
  const [sessionMode, setSessionModeState] = useState<LiveMode>("earpiece");
  const [liveStatus, setLiveStatus] = useState("");
  const [nudgeFlash, setNudgeFlash] = useState<NudgeEvent | null>(null);
  const [latencySummary, setLatencySummary] = useState("");
  const [toneFlags, setToneFlags] = useState<ToneFlagEvent[]>([]);
  // Tier B: the server says a paired watch socket is registered on its relay
  // (a `watch_connected` frame per change) — Live Coach shows a wrist note.
  const [watchConnected, setWatchConnected] = useState(false);
  const [preflight, setPreflight] = useState<PreflightState | null>(null);
  const [escalationCount, setEscalationCount] = useState(0);
  const [sessionSummary, setSessionSummary] = useState<SessionSummary | null>(null);
  const [lastEpisode, setLastEpisode] = useState<LastEpisode | null>(null);
  const [speakerNames, setSpeakerNames] = useState<Record<string, SpeakerBinding>>({});
  const [scoreboard, setScoreboard] = useState<Scoreboard | null>(null);
  // --- In-app call (Call mode) ----------------------------------------------
  const [callView, setCallView] = useState<CallView>(IDLE_CALL_VIEW);
  const callViewRef = useRef<CallView>(IDLE_CALL_VIEW);
  const callRef = useRef<CallSession | null>(null);
  const [callRoute, setCallRouteState] = useState<AudioRoute>("speaker");
  const callRouteRef = useRef<AudioRoute>("speaker");
  const callApiRef = useRef(options.callApi ?? defaultCallApi);
  callApiRef.current = options.callApi ?? defaultCallApi;
  const makeRtcAdapterRef = useRef(options.makeRtcAdapter ?? null);
  makeRtcAdapterRef.current = options.makeRtcAdapter ?? null;
  /** The active WebRTC adapter (route changes go through it). */
  const rtcAdapterRef = useRef<RtcAdapter | null>(null);
  /** Web: capture + recognizer primed inside the Answer/Start tap, ahead of
   *  the REST call that creates/joins the call (see beginWebCapture). */
  const preparedWebRef = useRef<PreparedWebCapture | null>(null);
  // --- Diagnostics counters (src/diagnostics) -------------------------------
  const wsReconnectsRef = useRef(0);
  const sttRestartsRef = useRef<number | null>(null);
  const sttFailureRef = useRef<string | null>(null);
  const micErrorRef = useRef("");
  const transcriptionMessageRef = useRef("");
  const liveStatusRef = useRef("");
  /** Mirrors speakerNames for the long-lived callbacks (onTurn, onmessage). */
  const speakerNamesRef = useRef<Record<string, SpeakerBinding>>({});
  /** personId -> the raw label the loop's CONTRAST rule currently puts that
   *  person on. A contrast identity can move to a later cluster (a person is
   *  one voice), and the screen's name must follow it — unlike an absolute
   *  match or a user's own label, which stay where they were put. */
  const contrastLabelsRef = useRef<Record<string, string>>({});
  /** Labels the USER gave this session (not voiceprint matches) — what
   *  POST /sessions/live carries as `speaker_labels`. */
  const speakerLabelsRef = useRef<Record<string, LiveSpeakerLabel>>({});
  const trackerRef = useRef(new PleasantnessTracker());
  /** The loop of the session that just ended, kept for post-session naming
   *  (its pooled speaker audio + embedder outlive the session). */
  const lastLoopRef = useRef<FastLoop | null>(null);
  const lastEpisodeRef = useRef<LastEpisode | null>(null);
  const enrollRef = useRef(options.enrollSpeaker ?? enrollSpeakerAudio);
  enrollRef.current = options.enrollSpeaker ?? enrollSpeakerAudio;
  const patchLabelsRef = useRef(options.patchLabels ?? patchSpeakerLabels);
  patchLabelsRef.current = options.patchLabels ?? patchSpeakerLabels;
  /** Session-end inputs kept outside React state so finishDrain (which runs
   *  from timers / socket callbacks) reads the final values. */
  const transcriptRef = useRef<TranscriptEntry[]>([]);
  const latencyLogRef = useRef<TurnLatency[]>([]);
  const escalationRef = useRef(0);
  const preflightInFlightRef = useRef(false);
  const liveModeRef = useRef(liveCapability.capable);
  const sessionModeRef = useRef<LiveMode>("earpiece");
  /** The mode handed to the FAST LOOP, which can differ from the recorded /
   *  UI mode: a therapist-role call runs the loop in "therapist" (STT + a
   *  merged turn_local, but no TTS and no coaching) while the session record
   *  and the Call UI stay "call". Mirrors sessionModeRef otherwise. */
  const loopModeRef = useRef<LiveMode>("earpiece");
  /** The running loop for this session (null on the legacy path). */
  const fastLoopRef = useRef<FastLoop | null>(null);
  /** True from the loop's start until it has stopped — gates which server
   *  events are rendered (the phone owns the transcript while it runs). */
  const liveActiveRef = useRef(false);
  /** On-device STT died mid-session: accept the server's transcript again. */
  const liveSttFailedRef = useRef(false);
  /** The recent local turns' words and whether the phone answered them
   *  itself (oldest first, capped). A cloud suggestion always renders, but
   *  is only VOICED when it is the answer to the LATEST turn (the server
   *  echoes the phone's own text back as `utterance_text`) and the phone had
   *  nothing to say for it — a late cloud answer to an earlier, already
   *  coached turn must not be spoken; a suggestion for words the phone never
   *  reported (a span its VAD missed, caught by the server) is the only
   *  voice there is and is spoken. */
  const recentLocalTurnsRef = useRef<{ text: string; hadSuggestion: boolean }[]>([]);
  /** Everything the phone told the server this session, for POST /sessions/live. */
  const localTurnsRef = useRef<TurnLocalEvent[]>([]);
  const toneFlagsRef = useRef<ToneFlagEvent[]>([]);
  const identitiesRef = useRef<SpeakerIdentityEvent[]>([]);
  const sessionStartedAtRef = useRef("");
  /** Seams read at call time so a re-render with new options is honoured. */
  const makeFastLoopRef = useRef(options.makeFastLoop ?? defaultMakeFastLoop);
  makeFastLoopRef.current = options.makeFastLoop ?? defaultMakeFastLoop;
  const postSessionRef = useRef(options.postSession ?? postLiveSession);
  postSessionRef.current = options.postSession ?? postLiveSession;
  const probeRef = useRef(options.probeCapabilities ?? defaultProbeCapabilities);
  probeRef.current = options.probeCapabilities ?? defaultProbeCapabilities;

  // --- Kept session audio (live/liveAudioKeeper.ts) ------------------------
  const keepAudioRef = useRef(options.keepAudio ?? true);
  keepAudioRef.current = options.keepAudio ?? true;
  const openKeeperRef = useRef(options.openAudioKeeper ?? openDefaultLiveAudioKeeper);
  openKeeperRef.current = options.openAudioKeeper ?? openDefaultLiveAudioKeeper;
  const postAudioRef = useRef(options.postSessionAudio ?? postLiveSessionAudio);
  postAudioRef.current = options.postSessionAudio ?? postLiveSessionAudio;
  /** The running session's keeper; null when not keeping. */
  const keeperRef = useRef<LiveAudioKeeper | null>(null);
  const [audioKeep, setAudioKeep] = useState<AudioKeepState>({ status: "off" });

  // --- Journal mode (live/journalRecorder.ts) --------------------------------
  const makeJournalRef = useRef(options.makeJournal ?? createDefaultJournalRecorder);
  makeJournalRef.current = options.makeJournal ?? createDefaultJournalRecorder;
  const prepareJournalAudioRef = useRef(options.prepareJournalAudio ?? prepareJournalAudioSession);
  prepareJournalAudioRef.current = options.prepareJournalAudio ?? prepareJournalAudioSession;
  const holdBackgroundMicRef = useRef(options.holdBackgroundMic ?? createBackgroundMicHold);
  holdBackgroundMicRef.current = options.holdBackgroundMic ?? createBackgroundMicHold;
  /** The running journal; null outside Journal mode. Mic frames go here
   *  INSTEAD of the fast loop / keeper / WebSocket while it is set. */
  const journalRef = useRef<JournalRecorder | null>(null);
  /** The journal that last ran (its uploads may still be settling). */
  const lastJournalRef = useRef<JournalRecorder | null>(null);
  const micHoldRef = useRef<BackgroundMicHold | null>(null);
  const [journal, setJournal] = useState<JournalState>(IDLE_JOURNAL_STATE);

  const releaseMicHold = useCallback(() => {
    const hold = micHoldRef.current;
    micHoldRef.current = null;
    if (hold) void hold.release().catch(() => {});
  }, []);

  /** End a journal run: mic off, loop flushed, file closed, uploads kicked
   *  off (they continue detached; the state reports them). Never throws. */
  const stopJournalSession = useCallback(async () => {
    const recorder = journalRef.current;
    journalRef.current = null;
    streamingRef.current = false;
    shouldReconnect.current = false;
    try {
      micStreamRef.current?.stop();
    } catch {
      // Stream may already be stopped.
    }
    releaseMicHold();
    setIsRecording(false);
    setSessionActive(false);
    if (recorder) {
      lastJournalRef.current = recorder;
      try {
        const final = await recorder.stop();
        console.log(
          `[journal] stopped: listened ${final.listeningSeconds}s, kept ${final.selfCount} stretch(es) / ${final.selfSeconds}s, ` +
            `${final.filesClosed} file(s), uploads sent ${final.uploads.sent} pending ${final.uploads.pending}`,
        );
      } catch (err) {
        console.warn("[journal] stop failed:", err);
      }
    }
    sessionActiveRef.current = false;
    // Hand the audio session back to playback (see finishDrain).
    void setPlaybackMode().catch(() => {});
  }, [releaseMicHold]);

  const retryJournalUploads = useCallback(async () => {
    const recorder = journalRef.current ?? lastJournalRef.current;
    if (recorder) await recorder.retryUploads();
  }, []);

  /** Open the keeper for a starting session (or say why not). */
  const beginAudioKeep = useCallback((sessionId: string) => {
    keeperRef.current?.discard();
    keeperRef.current = null;
    if (!keepAudioRef.current) {
      setAudioKeep({ status: "off" });
      return;
    }
    if (sessionModeRef.current === "call") {
      // The server records an in-app call itself (one episode per side).
      setAudioKeep({ status: "off", error: "in-app calls are recorded by the server" });
      return;
    }
    const opened = openKeeperRef.current(sessionId);
    if (!opened.keeper) {
      setAudioKeep({ status: "off", error: opened.reason });
      return;
    }
    keeperRef.current = opened.keeper;
    setAudioKeep({ status: "keeping" });
  }, []);

  /** Attach a finished file to its episode. A failure keeps the file on the
   *  phone and offers a retry; success deletes it. Never throws. */
  const uploadKeptAudio = useCallback(
    async (kept: {
      episodeId: string;
      uri: string;
      bytes: number;
      seconds: number;
      keeper: LiveAudioKeeper;
    }): Promise<void> => {
      const { bytes, seconds } = kept;
      setAudioKeep({ status: "uploading", bytes, seconds });
      try {
        await postAudioRef.current(kept.episodeId, kept.uri, kept.bytes);
      } catch (err) {
        const error = err instanceof Error ? err.message : String(err);
        console.warn("[useAudioStream] session audio upload failed:", error);
        setAudioKeep({ status: "failed", bytes, seconds, error, retry: () => uploadKeptAudio(kept) });
        return;
      }
      kept.keeper.discard(); // the server has it — free the cache
      setAudioKeep({ status: "kept", bytes, seconds });
    },
    [],
  );

  /** Session over: upload the kept file when the server stored the session,
   *  otherwise discard it (never upload audio for a session with no episode). */
  const settleAudioKeep = useCallback(
    async (keeper: LiveAudioKeeper | null, episodeId: string | null) => {
      if (!keeper) return;
      if (!episodeId) {
        keeper.discard();
        setAudioKeep({ status: "off", error: "not kept — the session wasn't saved on the server" });
        return;
      }
      let kept: Awaited<ReturnType<LiveAudioKeeper["finish"]>>;
      try {
        kept = await keeper.finish();
      } catch (err) {
        const error = err instanceof Error ? err.message : String(err);
        console.warn("[useAudioStream] keeping session audio failed:", error);
        keeper.discard();
        setAudioKeep({ status: "failed", error });
        return;
      }
      if (!kept) {
        setAudioKeep({ status: "off", error: "no audio was captured" });
        return;
      }
      await uploadKeptAudio({ ...kept, episodeId, keeper });
    },
    [uploadKeptAudio],
  );

  useEffect(() => {
    transcriptRef.current = transcript;
  }, [transcript]);
  useEffect(() => {
    micErrorRef.current = micError;
  }, [micError]);
  useEffect(() => {
    transcriptionMessageRef.current = transcriptionMessage;
  }, [transcriptionMessage]);
  useEffect(() => {
    liveStatusRef.current = liveStatus;
  }, [liveStatus]);

  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string>("");
  const reconnectAttempts = useRef(0);
  const shouldReconnect = useRef(false);
  const empathyRef = useRef(50);
  /** How often the coach should interject (0 = every turn / old default, 100
   *  = only the most critical moments). Mirrors empathyRef: read at config-
   *  send time, not captured stale in the onopen closure. */
  const interjectRef = useRef(0);
  /** Mirrors selfSpeaker so the long-lived onopen closure reads the current
   *  choice at config-send time, not a stale render's value. */
  const selfSpeakerRef = useRef<string | null>("Speaker A");
  /** Monotonic source of suggestion feed entry ids (see SuggestionEntry.id).
   *  Not reset per session — keeping it strictly increasing avoids key reuse. */
  const suggestionIdRef = useRef(0);
  /** Sticky per-session flag: true once the server has sent any "transcript"
   *  event. A new server owns the transcript entirely via those events, so
   *  suggestion.utterance_text must then never be appended: suggestions lag
   *  their utterance by seconds of LLM+TTS work while newer utterances keep
   *  finalizing (transcript A, transcript B, THEN suggestion for A), so a
   *  last-entry dedupe would miss the interleaving and re-append A out of
   *  order. Reset per session in startSession alongside the transcript. */
  const sawTranscriptEventRef = useRef(false);
  /** Synchronous re-entry guard: true from the first line of startSession
   *  until the session fully ends (stop drain finished / failure). A ref, not
   *  state, so a double-tap can never open two WebSockets (state flips too
   *  late — only after the async permission/audio-mode/start chain). */
  const sessionActiveRef = useRef(false);
  /** True while a graceful stop is waiting for the server's final events. */
  const drainingRef = useRef(false);
  const drainTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** Wall-clock time (ms epoch) at which the drain must end no matter what —
   *  the absolute cap the re-armed inactivity window can never exceed. */
  const drainDeadlineRef = useRef(0);
  /** Gates the onBuffer callback — the native stream can deliver a trailing
   *  buffer after stop() has been requested. */
  const streamingRef = useRef(false);
  /** Int16 samples captured but not yet sent (accumulates to frame size). */
  const pendingRef = useRef<Int16Array<ArrayBuffer>>(new Int16Array(0));
  /** One stateful resampler per capture session (created lazily from the
   *  actual hardware rate, reset on start/stop). Statelessly resampling each
   *  ~100 ms buffer would restart the read phase every call and drop
   *  fractional samples at non-integer ratios like 44.1k -> 16k. */
  const resamplerRef = useRef<StreamingResampler | null>(null);
  /** Refs mirroring the speech states: the WS onmessage handler is a
   *  long-lived closure, so it must read these at event time, not capture a
   *  stale render's value. */
  const speechEnabledRef = useRef(false);
  const speechAvailableRef = useRef(speechAvailable);

  const setSpeechEnabled = useCallback((enabled: boolean) => {
    const wasEnabled = speechEnabledRef.current;
    speechEnabledRef.current = enabled;
    setSpeechEnabledState(enabled);
    if (wasEnabled && !enabled) {
      // Switching to visual mid-utterance: go silent immediately.
      stopSpeechSafely();
    }
  }, []);

  /**
   * TTS failed to actually produce sound — synchronously (speak() threw) or
   * asynchronously (the utterance's onError fired, e.g. Android with no
   * installed voice data even though detectSpeechSupport() said true).
   * Degrade honestly: flip speechAvailable so LiveCoachScreen shows its
   * "spoken suggestions aren't available" note instead of the user hearing
   * silence while the UI implies audio coaching works. Idempotent — flips
   * and logs only once, however many late onError callbacks arrive.
   */
  const markSpeechUnavailable = useCallback((reason: unknown) => {
    if (!speechAvailableRef.current) return; // Already known — don't spam.
    speechAvailableRef.current = false;
    setSpeechAvailable(false);
    console.warn(
      "[useAudioStream] On-device TTS failed — suggestions are visual-only from here on:",
      reason,
    );
  }, []);

  /**
   * Speak one suggestion via free on-device TTS (expo-speech) — the free
   * analog of the server's Deepgram Aura audio. Most-recent-wins: any
   * utterance still in flight is stopped first, and nothing is ever queued —
   * in a live conversation, stale advice is worse than interrupted advice.
   */
  const speakSuggestion = useCallback(
    (text: string) => {
      if (!speechEnabledRef.current) return; // Visual mode: stay silent.
      if (!speechAvailableRef.current) return; // No TTS here: honest silence.
      if (drainingRef.current) return; // User pressed stop: don't keep talking.
      // Therapist mode is on-screen only, by contract — the fast loop never
      // asks to speak in it, and neither may the cloud's suggestion event.
      if (liveActiveRef.current && loopModeRef.current === "therapist") return;
      // A therapist observing a call is never spoken to either.
      if (callRef.current?.selfRole === "therapist") return;
      try {
        // Unconditional stop guarantees most-recent-wins without tracking
        // speaking state (Speech.stop() is a no-op when nothing is speaking,
        // and it clears expo-speech's internal utterance queue).
        stopSpeechSafely();
        Speech.speak(text, {
          // Nothing was actually spoken — never pretend otherwise. The
          // suggestion is already on screen; surface the degraded state.
          onError: (error) => markSpeechUnavailable(error),
        });
      } catch (err) {
        // speak() itself threw: no usable TTS backend on this platform.
        markSpeechUnavailable(err);
      }
    },
    [markSpeechUnavailable],
  );

  /**
   * Stop the fast loop (if one is running), print its latency log, and hand
   * the session record to the server. Idempotent; never throws. Awaiting
   * loop.stop() lets an in-flight final turn finish so its turn_local goes
   * out while the socket is still open — callers that can't wait (unmount,
   * reconnect exhaustion) fire-and-forget it.
   */
  const stopFastLoop = useCallback(async () => {
    const loop = fastLoopRef.current;
    // The kept audio is settled by the SAME call that posts the session (the
    // first one — later calls find no loop): taken here so a re-entrant stop
    // can't upload or discard it twice.
    const keeper = keeperRef.current;
    keeperRef.current = null;
    if (!loop) {
      // Legacy path (no on-device loop): nothing is POSTed, so there is no
      // episode to attach the audio to.
      void settleAudioKeep(keeper, null);
      return;
    }
    fastLoopRef.current = null;
    lastLoopRef.current = loop;
    liveActiveRef.current = false;
    let summary: Awaited<ReturnType<FastLoop["stop"]>> | null = null;
    try {
      summary = await loop.stop();
    } catch (err) {
      console.warn("[useAudioStream] fast loop stop failed:", err);
    }
    if (summary) {
      const report = formatLatencyLog(summary.latencyLog);
      console.log(report);
      setLatencySummary(report.split("\n")[0]);
      latencyLogRef.current = summary.latencyLog;
      sttRestartsRef.current = summary.sttRestarts ?? 0;
    }
    if (sessionModeRef.current === "call") {
      // An in-app call is persisted by the SERVER (one episode per
      // participant at call end — see call_ended below); a phone that POSTed
      // its half again would store it twice. The record was already handed
      // to lastEpisode from the call_ended frame.
      localTurnsRef.current = [];
      toneFlagsRef.current = [];
      identitiesRef.current = [];
      void settleAudioKeep(keeper, null);
      return;
    }
    const body: LiveSessionBody = {
      session_id: sessionIdRef.current,
      started_at: sessionStartedAtRef.current,
      ended_at: new Date().toISOString(),
      mode: liveSessionModeOf(sessionModeRef.current),
      turns: localTurnsRef.current,
      tone_flags: toneFlagsRef.current,
      speaker_identities: identitiesRef.current,
      // Names the user gave mid-call ride along so the stored episode (and
      // the therapist's auto-shared view of it) shows them at once.
      ...(Object.keys(speakerLabelsRef.current).length > 0
        ? { speaker_labels: { ...speakerLabelsRef.current } }
        : {}),
      // Outcome engine (Workstream 4): the BEFORE mood check, answered
      // before Start — already in the store by the time a session stops.
      // The AFTER half isn't answered yet at this point (it shows once
      // sessionSummary lands), so it goes out later via patchSessionMood.
      ...(useMoodStore.getState().before !== null
        ? { mood_before: useMoodStore.getState().before! }
        : {}),
    };
    localTurnsRef.current = [];
    toneFlagsRef.current = [];
    identitiesRef.current = [];
    // 404 (endpoint not deployed yet) is "unsupported", not a failure — the
    // transcript is already on screen; the record is a bonus.
    const turnCount = body.turns.length;
    const result = await postSessionRef.current(body);
    if (result.status === "failed") {
      console.warn("[useAudioStream] POST /sessions/live failed:", result.error);
      lastEpisodeRef.current = { episodeId: null, postStatus: "failed", sharedWith: [] };
      setLastEpisode(lastEpisodeRef.current);
    } else if (result.status === "unsupported") {
      lastEpisodeRef.current = { episodeId: null, postStatus: "unsupported", sharedWith: [] };
      setLastEpisode(lastEpisodeRef.current);
    } else {
      lastEpisodeRef.current = {
        episodeId: result.episodeId || null,
        postStatus: "created",
        sharedWith: result.sharedWith ?? [],
      };
      setLastEpisode(lastEpisodeRef.current);
      if (result.episodeId) {
        // Confirmed by the server: Your Day can show it right away.
        useLiveEpisodeStore.getState().remember({
          episodeId: result.episodeId,
          sessionId: body.session_id,
          startedAt: body.started_at,
          mode: body.mode,
          title: `Live session · ${body.mode}`,
          turnCount,
          sharedWith: result.sharedWith ?? [],
        });
      }
    }
    // Attach the kept audio to the stored episode (or drop it when there is
    // none). Awaited so a stop that completes has settled the file; an
    // upload failure never fails the session — it leaves a retry.
    await settleAudioKeep(
      keeper,
      result.status === "created" && result.episodeId ? result.episodeId : null,
    );
  }, [settleAudioKeep]);

  /**
   * Start the fast loop for this session. Any failure (native module absent,
   * model download failed, STT permission denied) leaves the legacy server
   * path running and says so in liveStatus — never a broken session.
   */
  const startFastLoop = useCallback(
    async (sessionId: string, empathy: number, primedRecognizer: SpeechRecognizer | null = null) => {
      // Set by onSttError during loop.start() (a recognizer that fails to
      // start) so the status line below can say so instead of claiming a
      // working loop.
      let sttFailure: string | null = null;
      const handlers: FastLoopHandlers = {
        ...(primedRecognizer ? { recognizer: primedRecognizer } : {}),
        // Build progress (the web build's one-time voice-model download).
        onStatus: (line) => {
          if (sessionActiveRef.current && !drainingRef.current) setLiveStatus(line);
        },
        speak: (text) => speakSuggestion(text),
        send: (event) => {
          localTurnsRef.current.push(event);
          const ws = wsRef.current;
          if (ws && ws.readyState === WebSocket.OPEN) {
            try {
              ws.send(JSON.stringify(event));
            } catch {
              // Socket mid-close: the session record still has the turn.
            }
          }
        },
        onTurn: (turn) => {
          // A voiceprint match (pre-enrolled or learned mid-call) names the
          // raw label for every later line; the wire label stays raw.
          if (turn.displayName && turn.personId) {
            let names = speakerNamesRef.current;
            if (turn.matchBasis === "contrast") {
              // The contrast rule moved this person off an earlier cluster:
              // the old label's name goes with them, unless the USER gave it.
              const previous = contrastLabelsRef.current[turn.personId];
              if (previous !== undefined && previous !== turn.speaker && !speakerLabelsRef.current[previous]) {
                const { [previous]: _moved, ...rest } = names;
                names = rest;
              }
              contrastLabelsRef.current = { ...contrastLabelsRef.current, [turn.personId]: turn.speaker };
            }
            if (!names[turn.speaker]) {
              names = {
                ...names,
                [turn.speaker]: {
                  personId: turn.personId,
                  displayName: turn.displayName,
                  isSelf: turn.isSelf === true,
                },
              };
            }
            if (names !== speakerNamesRef.current) {
              speakerNamesRef.current = names;
              setSpeakerNames(names);
            }
          }
          // In a call this phone hears only its owner: every local turn is
          // "You", keyed by the slot label the server relabels it with
          // (call_state.self_label) so naming/scoreboard/episodes line up.
          const inCall = callRef.current !== null;
          const rawLabel = inCall ? (callViewRef.current.selfLabel ?? turn.speaker) : turn.speaker;
          const display = inCall
            ? "You"
            : (speakerNamesRef.current[turn.speaker]?.displayName ?? turn.speaker);
          setSpeakerLabel(display);
          if (turn.text) {
            setTranscript((prev) => [
              ...prev,
              {
                speaker: display,
                speakerId: rawLabel,
                kind: turn.kind,
                isSelf: turn.isSelf,
                text: turn.text,
                timestamp: Date.now(),
                startTime: turn.startTime,
                endTime: turn.endTime,
              },
            ]);
          }
          const recent = recentLocalTurnsRef.current;
          recent.push({ text: turn.text, hadSuggestion: turn.suggestion !== null });
          if (recent.length > MAX_SUGGESTION_FEED) recent.splice(0, recent.length - MAX_SUGGESTION_FEED);
          // Scoreboard: every on-device turn scores from what it carries
          // (tone, prosody, balance) — null inputs are honest gaps.
          trackerRef.current.observe(rawLabel, turn.textTone, turn.prosody);
          setScoreboard(trackerRef.current.board());
          // A therapist observing a call is never coached: the loop runs in
          // "therapist" (no local LLM speech) and a stray local suggestion
          // is dropped here rather than shown as advice to her.
          if (turn.suggestion && callRef.current?.selfRole !== "therapist") {
            const id = (suggestionIdRef.current += 1);
            const text = turn.suggestion;
            setSuggestions((prev) => {
              const entry: SuggestionEntry = {
                id,
                kind: turn.suggestionKind ?? "response",
                texts: [text],
                tone: empathyTone(empathyRef.current),
                muted: false,
                timestamp: Date.now(),
                source: "on-device",
              };
              const next = [entry, ...prev];
              return next.length > MAX_SUGGESTION_FEED
                ? next.slice(0, MAX_SUGGESTION_FEED)
                : next;
            });
          }
        },
        onNudge: (nudge) => {
          if (nudge.level > 0) {
            setNudgeFlash(nudge);
            escalationRef.current += 1;
            setEscalationCount(escalationRef.current);
          }
        },
        onSttError: (code, message) => {
          liveSttFailedRef.current = true;
          sttFailure = `speech recognition failed (${code}${message ? `: ${message}` : ""}) — transcript from the server`;
          sttFailureRef.current = sttFailure;
          setLiveStatus(`On-device ${sttFailure}.`);
        },
        onDegrade: (stage, reason) => {
          // Say what is actually running now, not what loaded at start.
          if (stage === "vad") {
            setLiveStatus((s) =>
              s.replace(/^On-device: [^·]*VAD/, "On-device: energy VAD (Silero failed)") +
              ` — ${reason}`,
            );
          }
        },
      };
      try {
        const build = await makeFastLoopRef.current(handlers, loopModeRef.current);
        if (!sessionActiveRef.current || drainingRef.current) {
          // The user stopped while models were loading: don't start now.
          void build.loop.stop().catch(() => {});
          primedRecognizer?.stop();
          return;
        }
        // Reset BEFORE start: onSttError may fire inside start() and must
        // not be undone afterwards.
        liveSttFailedRef.current = false;
        recentLocalTurnsRef.current = [];
        build.loop.setSelfSpeakerFallback(selfSpeakerRef.current);
        await build.loop.start({
          sessionId,
          mode: loopModeRef.current,
          empathy,
        });
        fastLoopRef.current = build.loop;
        lastLoopRef.current = null;
        liveActiveRef.current = true;
        sessionStartedAtRef.current = new Date().toISOString();
        setLiveStatus(
          sttFailure
            ? `On-device: ${build.status} · ${sttFailure}.`
            : `On-device: ${build.status}`,
        );
        // The phone speaks for itself now: tell the server to skip its TTS
        // (and to report its own latency at session end).
        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(
            JSON.stringify({ type: "config", tts: "on-device", report_latency: true }),
          );
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        primedRecognizer?.stop();
        setLiveStatus(`On-device coaching unavailable (${msg}) — using the server.`);
      }
    },
    [speakSuggestion],
  );

  /**
   * Send accumulated audio as ~100 ms binary frames. Reads wsRef.current at
   * call time so after a reconnect frames go to the NEW socket, never a stale
   * one captured in a closure.
   */
  const flushAudioFrames = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      // Socket down (e.g. mid-reconnect): keep buffering, but bounded.
      if (pendingRef.current.length > MAX_PENDING_SAMPLES) {
        pendingRef.current = pendingRef.current.slice(
          pendingRef.current.length - MAX_PENDING_SAMPLES,
        );
      }
      return;
    }
    while (pendingRef.current.length >= SAMPLES_PER_FRAME) {
      const frame = pendingRef.current.slice(0, SAMPLES_PER_FRAME);
      pendingRef.current = pendingRef.current.slice(SAMPLES_PER_FRAME);
      // frame.buffer is exactly the frame's bytes (slice() allocates fresh),
      // int16 little-endian on every RN/browser platform (ARM/x86).
      ws.send(frame.buffer);
    }
  }, []);

  /**
   * Receives raw PCM from expo-audio. We request 16 kHz mono float32, but
   * normalise against the ACTUAL rate/channels the buffer reports — sending
   * audio whose real rate isn't 16 kHz would silently break transcription.
   */
  const handleAudioBuffer = useCallback(
    (buffer: AudioStreamBuffer) => {
      if (!streamingRef.current) return;
      let samples: Float32Array = new Float32Array(buffer.data);
      if (buffer.channels > 1) {
        samples = downmixToMono(samples, buffer.channels);
      }
      if (buffer.sampleRate !== TARGET_SAMPLE_RATE) {
        let resampler = resamplerRef.current;
        if (!resampler || resampler.inputRate !== buffer.sampleRate) {
          resampler = new StreamingResampler(
            buffer.sampleRate,
            TARGET_SAMPLE_RATE,
          );
          resamplerRef.current = resampler;
        }
        samples = resampler.process(samples);
      }
      const int16 = float32ToInt16(samples);
      // Journal mode: the recorder is the ONLY consumer — no fast loop, no
      // kept-audio WAV, no WebSocket frames (nothing leaves the phone while
      // it listens).
      const journalRecorder = journalRef.current;
      if (journalRecorder) {
        journalRecorder.pushSamples(int16);
        return;
      }
      // The on-device fast loop (when running) hears exactly what the server
      // hears — one 16 kHz mono int16 conversion, two consumers.
      fastLoopRef.current?.pushSamples(int16);
      // …and the kept-audio WAV is the third consumer of the same frames.
      keeperRef.current?.append(int16);
      pendingRef.current = concatInt16(pendingRef.current, int16);
      flushAudioFrames();
    },
    [flushAudioFrames],
  );

  // expo-audio's realtime PCM capture (SDK 56+). On web this returns
  // { stream: null } — expo-audio has no web capture implementation yet.
  const { stream: micStream } = useMicrophoneStream({
    sampleRate: TARGET_SAMPLE_RATE,
    channels: 1,
    encoding: "float32",
    onBuffer: handleAudioBuffer,
  });
  const micStreamRef = useRef(micStream);
  micStreamRef.current = micStream;

  /** Web-only microphone capture (expo-audio has no web recorder). Null on
   *  native and until a web session actually starts capturing. */
  const webCaptureRef = useRef<WebAudioCapture | null>(null);

  /**
   * Release whichever capture backend is active — native (expo-audio) or web
   * (getUserMedia + AudioWorklet) — without ever throwing. Called from every
   * teardown path (manual stop, reconnect exhaustion, unmount). On native the
   * web ref is always null (and vice versa), so this is exactly the old
   * `micStreamRef.current?.stop()` on those platforms.
   */
  const releaseCapture = useCallback(() => {
    try {
      micStreamRef.current?.stop();
    } catch {
      // Stream may already be stopped.
    }
    const web = webCaptureRef.current;
    if (web) {
      webCaptureRef.current = null;
      // stop() releases the mic tracks synchronously; the async context close
      // is fire-and-forget (nothing to await on an unmount path).
      void web.stop().catch(() => {});
    }
  }, []);

  /** Detach handlers BEFORE closing so the deliberate close() can't fire
   *  onclose and stomp the status we set afterwards (manual stop must end
   *  at "idle", not "disconnected"). */
  const teardownWebSocket = useCallback(() => {
    const ws = wsRef.current;
    if (!ws) return;
    ws.onopen = null;
    ws.onmessage = null;
    ws.onerror = null;
    ws.onclose = null;
    try {
      ws.close();
    } catch {
      // Socket may already be closed.
    }
    wsRef.current = null;
  }, []);

  /**
   * The session's diagnostics record (src/diagnostics): what ran, what
   * broke, how fast. Written to the diagnostics store for Settings' "Send
   * diagnostics"; sent automatically when the session had errors, so a
   * failed demo is diagnosable without the owner doing anything.
   */
  const recordSessionDiagnostics = useCallback(() => {
    if (!sessionIdRef.current) return;
    const call = callViewRef.current;
    const errors: string[] = [];
    if (micErrorRef.current) errors.push(`mic: ${micErrorRef.current}`);
    if (sttFailureRef.current) errors.push(`stt: ${sttFailureRef.current}`);
    if (transcriptionMessageRef.current) errors.push(`transcription: ${transcriptionMessageRef.current}`);
    if (wsReconnectsRef.current > 0) errors.push(`ws reconnects: ${wsReconnectsRef.current}`);
    if (/unavailable|failed/i.test(liveStatusRef.current)) errors.push(`live: ${liveStatusRef.current}`);
    if (lastEpisodeRef.current?.postStatus === "failed") errors.push("POST /sessions/live failed");
    if (call.status === "failed" && call.error) errors.push(`call: ${call.error}`);
    if (call.iceRestarts > 0) errors.push(`call: ${call.iceRestarts} ICE restart(s)`);
    const record: SessionDiagnostics = {
      sessionId: sessionIdRef.current,
      mode: sessionModeRef.current,
      startedAt: sessionStartedAtRef.current || null,
      endedAt: new Date().toISOString(),
      turns: transcriptRef.current.length,
      latency: summarizeLatency(latencyLogRef.current),
      // stopFastLoop parks the loop in lastLoopRef before this runs; its
      // turns carry any identity revision the contrast rule made.
      speakerId: (() => {
        const loop = fastLoopRef.current ?? lastLoopRef.current;
        return loop ? summarizeSpeakerId(loop.turnsSoFar) : null;
      })(),
      liveStatus: liveStatusRef.current,
      onDevice: liveModeRef.current && liveCapability.capable,
      sttRestarts: sttRestartsRef.current,
      sttFailure: sttFailureRef.current,
      wsReconnects: wsReconnectsRef.current,
      micError: micErrorRef.current || null,
      transcriptionMessage: transcriptionMessageRef.current || null,
      postStatus: lastEpisodeRef.current?.postStatus ?? "none",
      call:
        call.status === "idle"
          ? null
          : {
              status: call.status,
              iceRestarts: call.iceRestarts,
              error: call.error,
              connectedSeconds: call.connectedAt ? Math.round((Date.now() - call.connectedAt) / 1000) : null,
            },
      errors,
    };
    const store = useDiagnosticsStore.getState();
    store.recordSession(record);
    if (errors.length > 0) {
      const user = useAuthStore.getState().user;
      void store.send("auto", { uid: user?.uid ?? null, email: user?.email ?? null });
    }
  }, [liveCapability.capable]);

  /**
   * Final cleanup shared by every way a session ends after a manual stop:
   * server `session_complete`, server-side close, or the drain timeout.
   * Idempotent — safe to call from any of those paths.
   */
  const finishDrain = useCallback(() => {
    if (drainTimerRef.current !== null) {
      clearTimeout(drainTimerRef.current);
      drainTimerRef.current = null;
    }
    drainingRef.current = false;
    sessionActiveRef.current = false;
    // A call outlives nothing: if the session ends for any reason (server
    // close, reconnect exhaustion) the WebRTC side goes down with it.
    const call = callRef.current;
    callRef.current = null;
    call?.hangUp();
    // Normally already stopped by stopSession; this covers a session that
    // ends from the server side while the loop is still up. The diagnostics
    // record waits for the POST /sessions/live outcome it reports.
    void stopFastLoop().finally(() => recordSessionDiagnostics());
    if (sessionStartedAtRef.current) {
      setSessionSummary(
        summarizeSession({
          startedAt: sessionStartedAtRef.current,
          transcript: transcriptRef.current,
          latencyLog: latencyLogRef.current,
          escalations: escalationRef.current,
        }),
      );
    }
    teardownWebSocket();
    // The session is over: put the audio session back into a playback config so
    // a subsequent replay is audible (the record-oriented mode we set on start
    // silences media playback on Android). Fire-and-forget; web no-ops.
    void setPlaybackMode().catch(() => {});
    setIsRecording(false);
    setSessionActive(false);
    setConnectionStatus("idle");
  }, [teardownWebSocket, stopFastLoop, recordSessionDiagnostics]);

  /**
   * (Re)arm the drain inactivity timer: STOP_DRAIN_TIMEOUT_MS of server
   * silence ends the drain, but never later than the absolute deadline set
   * when the drain started. Called once from stopSession and again on every
   * frame received while draining (a frame = the server is alive and still
   * working on the final utterance).
   */
  const armDrainTimer = useCallback(() => {
    if (drainTimerRef.current !== null) {
      clearTimeout(drainTimerRef.current);
    }
    const untilDeadline = drainDeadlineRef.current - Date.now();
    drainTimerRef.current = setTimeout(
      () => {
        drainTimerRef.current = null;
        finishDrain();
      },
      Math.max(0, Math.min(STOP_DRAIN_TIMEOUT_MS, untilDeadline)),
    );
  }, [finishDrain]);

  const stopSession = useCallback(async () => {
    if (drainingRef.current) return; // Stop already in progress.
    if (journalRef.current) {
      // Journal mode has no socket to drain and no record to post.
      await stopJournalSession();
      return;
    }
    shouldReconnect.current = false;
    streamingRef.current = false;
    // Call mode: hang up first (releases WebRTC's mic, stops remote audio)
    // and tell the server — detached from callRef BEFORE hangUp so its
    // "ended" callback can't re-enter this stop.
    const call = callRef.current;
    callRef.current = null;
    if (call) {
      const callId = call.callId;
      call.hangUp();
      if (callId) void callApiRef.current.end(callId);
    }
    // The session is over: never keep coaching aloud after the user stops.
    // (drainingRef gates speakSuggestion, so late drain-window suggestions
    // still render visually but are not spoken.)
    stopSpeechSafely();
    releaseCapture();
    const resampler = resamplerRef.current;
    resamplerRef.current = null;
    // Let the fast loop finish its last turn (its turn_local rides the
    // still-open socket) before the stop handshake below.
    await stopFastLoop();

    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      // Flush the resampler's held-back tail and any sub-frame remainder
      // while the socket is still open.
      if (resampler) {
        const tail = resampler.flush();
        if (tail.length > 0) {
          pendingRef.current = concatInt16(
            pendingRef.current,
            float32ToInt16(tail),
          );
        }
      }
      if (pendingRef.current.length > 0) {
        ws.send(pendingRef.current.buffer);
      }
      pendingRef.current = new Int16Array(0);
      // Graceful stop: tell the server we're done, then keep the socket open
      // for a short drain window so the final utterance's suggestion (which
      // arrives from transcription a few hundred ms later) is not lost. The
      // server replies with any remaining events, then `session_complete`.
      ws.send(JSON.stringify({ type: "stop" }));
      setIsRecording(false);
      setSessionActive(false);
      drainingRef.current = true;
      drainDeadlineRef.current = Date.now() + STOP_DRAIN_MAX_MS;
      armDrainTimer();
      return;
    }

    // Socket already closed / mid-reconnect: nothing to hand-shake with —
    // clean up immediately, exactly as before.
    pendingRef.current = new Int16Array(0);
    finishDrain();
  }, [finishDrain, armDrainTimer, releaseCapture, stopFastLoop, stopJournalSession]);

  useEffect(() => {
    return () => {
      // Unmount: tear everything down synchronously — no drain window, no
      // dangling timers, no setState on an unmounted component.
      if (drainTimerRef.current !== null) {
        clearTimeout(drainTimerRef.current);
        drainTimerRef.current = null;
      }
      drainingRef.current = false;
      sessionActiveRef.current = false;
      shouldReconnect.current = false;
      streamingRef.current = false;
      releaseCapture();
      const call = callRef.current;
      callRef.current = null;
      call?.hangUp();
      stopSpeechSafely(); // Never keep talking after the screen is gone.
      // A session abandoned mid-way was never saved: drop its audio rather
      // than upload it (taken before stopFastLoop so it can't).
      keeperRef.current?.discard();
      keeperRef.current = null;
      // A journal stops cleanly on unmount: its file is closed (valid WAV)
      // and its upload kicked off, detached — nothing kept is lost.
      if (journalRef.current) void stopJournalSession();
      void stopFastLoop();
      teardownWebSocket();
      // Leaving the live screen: hand the audio session back to playback so a
      // replay elsewhere in the app isn't left silent by our record mode.
      void setPlaybackMode().catch(() => {});
    };
  }, [teardownWebSocket, releaseCapture, stopFastLoop, stopJournalSession]);

  const connectWebSocket = useCallback(
    (sessionId: string) => {
      // Defensive: never let a previous socket keep live handlers (they would
      // stomp connectionStatus and schedule reconnects to a stale session).
      teardownWebSocket();

      const url = `${WS_BASE}/ws/session/${sessionId}`;
      setConnectionStatus("connecting");

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionStatus("live");
        reconnectAttempts.current = 0;
        // The server learns the empathy setting (and role) via a config
        // message — there is no query-param channel. The WebSocket handshake
        // can't carry an Authorization header, so the Firebase ID token rides
        // in this FIRST config frame as `id_token` (the exact field the
        // backend verifies before accepting the session). Read synchronously
        // from the cache — onopen can't await. Empathy updates reuse the
        // config shape but deliberately omit the token: the server verifies
        // only the first config frame.
        const idToken = getCachedToken();
        ws.send(
          JSON.stringify({
            type: "config",
            empathy_slider: empathyRef.current,
            interject_level: interjectRef.current,
            // Which diarized voice is the coached user's. Read from the ref so
            // a toggle made before the socket opened is still honoured here.
            self_speaker: selfSpeakerRef.current,
            // On-device TTS: the server must not synthesize audio for us;
            // and report its per-stage latency with session_complete.
            ...(liveActiveRef.current
              ? { tts: "on-device", report_latency: true }
              : {}),
            ...(idToken ? { id_token: idToken } : {}),
          }),
        );
        // Call mode: (re)announce ourselves in the call on every (re)open.
        callRef.current?.onSocketOpen();
      };

      ws.onmessage = (event) => {
        // Any frame while draining — transcript, suggestion, ack, anything —
        // means the server is alive and still finishing the session (e.g.
        // Whisper transcribing the final utterance): re-arm the inactivity
        // window (bounded by the absolute cap) instead of racing a fixed
        // timeout and losing the final suggestion + session_complete.
        if (drainingRef.current) {
          armDrainTimer();
        }
        try {
          const data = JSON.parse(event.data);

          // Call mode: call_state / rtc_signal / call_ended belong to the
          // call state machine (src/live/call/callSession.ts). call_ended
          // also carries THIS participant's episode (the server persisted
          // it) — the record a solo session would have POSTed itself.
          if (callRef.current && data.type === "call_ended") {
            const episodeId = typeof data.episode_id === "string" ? data.episode_id : null;
            const sharedWith: string[] = Array.isArray(data.shared_with)
              ? data.shared_with.filter((e: unknown): e is string => typeof e === "string")
              : [];
            if (episodeId) {
              lastEpisodeRef.current = { episodeId, postStatus: "created", sharedWith };
              setLastEpisode(lastEpisodeRef.current);
              useLiveEpisodeStore.getState().remember({
                episodeId,
                sessionId: sessionIdRef.current,
                startedAt: sessionStartedAtRef.current,
                mode: "call",
                title: "Live session · call",
                turnCount: typeof data.turn_count === "number" ? data.turn_count : transcriptRef.current.length,
                sharedWith,
              });
            }
          }
          if (callRef.current?.handleServerMessage(data)) return;

          // In a call, every OTHER member's turns arrive as transcript events
          // (their phone sent turn_local; the server relabels them with their
          // slot label and a display_name relative to us) and our own turns
          // are never echoed — so in a call every transcript event is remote.
          const inCall = callRef.current !== null;
          const remoteTurn = inCall && data.type === "transcript";
          if (
            data.type === "transcript" &&
            (!liveActiveRef.current || liveSttFailedRef.current || remoteTurn)
          ) {
            // New protocol: the finalized utterance arrives on its own,
            // ahead of the suggestion event for the same turn. From the
            // first one, the transcript belongs to these events alone.
            // (While the on-device loop runs, the phone's turn_local turns
            // ARE the transcript; the server's copy is only used again if
            // on-device STT died mid-session.)
            sawTranscriptEventRef.current = true;
            // In a call: `speaker` is the sender's slot label ("Speaker A")
            // and `display_name` their name relative to us ("Dad", "Mom
            // (therapist)"); the slot label stays the key (naming, scoreboard,
            // the episode's label ladder all use it).
            const peers = callViewRef.current.peers;
            const senderUid = typeof data.participant_uid === "string" ? data.participant_uid : null;
            const peer = senderUid ? peers.find((pr) => pr.uid === senderUid) : undefined;
            const rawLabel: string =
              (typeof data.speaker === "string" && data.speaker) || peer?.label || "Unknown";
            const speaker: string = remoteTurn
              ? speakerNamesRef.current[rawLabel]?.displayName ??
                ((typeof data.display_name === "string" && data.display_name) || peer?.displayName || rawLabel)
              : rawLabel;
            setSpeakerLabel(speaker);
            if (remoteTurn) {
              // The sender's own on-device measurements ride along so an
              // observer can score everyone (honest nulls otherwise). A
              // corrected row (`replaces_seq`) is scored here and not before:
              // the cloud copy it replaces carried neither tone nor prosody,
              // so it contributed no score — only a slot in the rolling
              // who-spoke balance window.
              trackerRef.current.observe(
                rawLabel,
                data.text_tone && typeof data.text_tone === "object" ? data.text_tone : null,
                data.prosody && typeof data.prosody === "object" ? data.prosody : null,
              );
              setScoreboard(trackerRef.current.board());
            }
            // In a call every relayed row is keyed by its merged-transcript
            // `seq`. A row whose seq (or whose `replaces_seq`) we already
            // rendered REPLACES that line in place — the server corrects the
            // first turn of each member, whose words the cloud transcriber
            // finalized before that phone's own turn_local latched the
            // session local-first, and a naive append would show it twice.
            const rowSeq = typeof data.seq === "number" ? data.seq : null;
            const replacesSeq =
              typeof data.replaces_seq === "number" ? data.replaces_seq : null;
            const targetSeq = remoteTurn ? replacesSeq ?? rowSeq : null;
            setTranscript((prev) => {
              const entry: TranscriptEntry = {
                speaker,
                ...(remoteTurn ? { speakerId: rawLabel } : {}),
                ...(rowSeq !== null && remoteTurn ? { callSeq: rowSeq } : {}),
                text: data.text,
                timestamp: Date.now(),
                // Utterance timing (seconds) when the server provides it —
                // guarded by type so a malformed frame can't smuggle in a
                // string. This is what makes post-session interruption stats
                // computable for live conversations.
                ...(typeof data.start_time === "number"
                  ? { startTime: data.start_time }
                  : {}),
                ...(typeof data.end_time === "number"
                  ? { endTime: data.end_time }
                  : {}),
              };
              const at =
                targetSeq === null
                  ? -1
                  : prev.findIndex((t) => t.callSeq === targetSeq);
              if (at < 0) return [...prev, entry];
              const next = prev.slice();
              // Keep the original arrival timestamp: the line stays where the
              // reader saw it, only its words/timing are corrected.
              next[at] = { ...entry, timestamp: prev[at].timestamp };
              return next;
            });
          } else if (data.type === "suggestion") {
            // The server bundles the transcribed utterance and its coaching
            // suggestions in one event (see server SuggestionEvent).
            const speaker = data.speaker || "Unknown";
            setSpeakerLabel(speaker);
            // Where it came from: the server's own LLM ("cloud", the default
            // on every legacy event) or our own turn echoed back.
            const source: "on-device" | "cloud" =
              data.suggestion_source === "on-device" ? "on-device" : "cloud";
            // Therapist sockets in a call get every participant's coaching as
            // a read-only copy tagged for_uid: shown with that person's name,
            // never voiced, never one of the viewer's own escalations.
            const forUid = typeof data.for_uid === "string" ? data.for_uid : null;
            const forName: string | undefined = forUid
              ? (callViewRef.current.peers.find((pr) => pr.uid === forUid)?.displayName ?? forUid)
              : undefined;
            if (
              data.utterance_text &&
              !sawTranscriptEventRef.current &&
              !liveActiveRef.current
            ) {
              // Legacy fallback ONLY: an old server never sends "transcript"
              // events, so its suggestion event is the sole transcript
              // source. On a new server this append must never run — its
              // suggestions lag behind newer transcript events (LLM+TTS take
              // seconds), so appending here would duplicate the utterance
              // out of order.
              setTranscript((prev) => [
                ...prev,
                { speaker, text: data.utterance_text, timestamp: Date.now() },
              ]);
            }
            const tone = empathyTone(
              typeof data.empathy_slider === "number"
                ? data.empathy_slider
                : empathyRef.current,
            );
            const items: string[] = Array.isArray(data.suggestions)
              ? data.suggestions
              : [];
            if (items.length > 0) {
              // speak === false means the server judged this turn not worth
              // interjecting on: stay silent and dim it in the UI instead of
              // voicing every suggestion regardless of importance.
              const muted = data.speak === false;
              // kind may be absent on older servers → a normal "response".
              const kind: SuggestionKind =
                data.kind === "nudge" ? "nudge" : "response";
              if (kind === "nudge" && data.speak !== false && !liveActiveRef.current && !forName) {
                // Legacy path: the server's delivery nudge is the only
                // escalation signal (the fast loop counts its own).
                escalationRef.current += 1;
                setEscalationCount(escalationRef.current);
              }
              // Streaming preview (local-first sessions only): shown dimmed,
              // superseded by the final event — which drops every preview
              // still in the feed, so a turn never shows twice.
              const partial = data.partial === true;
              const id = (suggestionIdRef.current += 1);
              // Accumulate instead of replace: newest first, capped so the
              // feed never grows without bound. A glance a second late still
              // finds the last few turns of advice.
              setSuggestions((prev) => {
                const entry: SuggestionEntry = {
                  id,
                  kind,
                  texts: items,
                  tone,
                  muted,
                  timestamp: Date.now(),
                  source,
                  ...(partial ? { partial: true } : {}),
                  ...(forName ? { forName } : {}),
                };
                const kept = partial ? prev : prev.filter((e) => !e.partial);
                const next = [entry, ...kept];
                return next.length > MAX_SUGGESTION_FEED
                  ? next.slice(0, MAX_SUGGESTION_FEED)
                  : next;
              });
              // While the on-device loop runs, a cloud suggestion AUGMENTS
              // the local one on screen but is only voiced when it answers
              // the LATEST local turn (the server echoes the phone's text
              // back as utterance_text) and the phone had nothing to say
              // for it (its providers fell through to "cloud") — otherwise
              // the user would hear two answers to one moment, or a late
              // answer to a moment that has passed. With on-device STT
              // dead the server's transcript is the only one, so its
              // suggestions are voiced exactly as on the legacy path.
              const loop = fastLoopRef.current;
              const voiceIt =
                !muted &&
                !forName &&
                (!liveActiveRef.current ||
                  liveSttFailedRef.current ||
                  (source === "cloud" &&
                    cloudAnswersOpenMoment(
                      recentLocalTurnsRef.current,
                      typeof data.utterance_text === "string" ? data.utterance_text : null,
                    )));
              if (voiceIt) {
                // Earpiece mode: speak the newest TOP suggestion with free
                // on-device TTS — nudges too, they're short. (The event also
                // carries data.audio_b64 — Deepgram Aura mp3, paid key
                // required — deliberately ignored; a future premium option.)
                // While the loop runs it applies its own rules first: never
                // over live speech, never in therapist mode.
                if (!loop || !liveActiveRef.current) {
                  speakSuggestion(items[0]);
                } else {
                  loop.offerSpeech(items[0]);
                }
              }
            }
          } else if (data.type === "nudge") {
            // A nudge the SERVER computed about this user — today: call-mode
            // "interrupting" (sustained talking-over on the merged timeline,
            // server/calls.py; exact there, unobservable on a single mic).
            // Same shape/semantics as the on-device policy's NudgeEvent:
            // screen always, haptic on escalation only.
            const level = typeof data.level === "number" ? data.level : 0;
            const vectors = Array.isArray(data.vectors) ? (data.vectors as string[]) : [];
            const nudge: NudgeEvent = {
              channel: data.channel === "B" ? "B" : "A",
              level,
              t: typeof data.t === "number" ? data.t : 0,
              vectors,
            };
            if (level > 0) {
              setNudgeFlash(nudge);
              escalationRef.current += 1;
              setEscalationCount(escalationRef.current);
              if (vectors.length > 0) void expoHaptics.nudge(level).catch(() => {});
            }
          } else if (data.type === "tone_flag") {
            // Server-side tone analysis over a turn: rendered additively.
            const flag = data as ToneFlagEvent;
            toneFlagsRef.current.push(flag);
            setToneFlags((prev) => [flag, ...prev].slice(0, MAX_SUGGESTION_FEED));
          } else if (data.type === "watch_connected") {
            // Tier B: a paired watch's (companion) socket came or went on the
            // server — the screen shows "nudges on your wrist" while present.
            setWatchConnected(data.connected === true);
          } else if (data.type === "speaker_identity") {
            // The server's (possibly revised) identity for a label: relabel
            // every transcript line that carries it. A null display_name is
            // "unknown" and changes nothing.
            const identity = data as SpeakerIdentityEvent;
            identitiesRef.current.push(identity);
            if (
              typeof identity.speaker === "string" &&
              typeof identity.display_name === "string" &&
              identity.display_name &&
              // A name the user gave mid-call beats the server's guess.
              !speakerLabelsRef.current[identity.speaker]
            ) {
              const from = identity.speaker;
              const to = identity.display_name;
              speakerNamesRef.current = {
                ...speakerNamesRef.current,
                [from]: {
                  personId: identity.person_id ?? "",
                  displayName: to,
                  isSelf: identity.is_self === true,
                },
              };
              setSpeakerNames(speakerNamesRef.current);
              setTranscript((prev) =>
                prev.map((t) =>
                  t.speaker === from || t.speakerId === from
                    ? { ...t, speaker: to, speakerId: t.speakerId ?? from }
                    : t,
                ),
              );
              setSpeakerLabel((current) => (current === from ? to : current));
            }
          } else if (data.type === "transcription_unavailable") {
            // Be explicit instead of silently showing an empty live screen.
            setTranscriptionAvailable(false);
            setTranscriptionMessage(data.reason || "Transcription unavailable");
          } else if (data.type === "session_complete") {
            // Local-first sessions asked for the server's own per-stage
            // timings (config.report_latency); log them next to ours.
            if (data.latency_summary !== undefined) {
              console.log("[useAudioStream] server latency:", data.latency_summary);
            }
            // Server has flushed everything after our `stop` — finish now
            // instead of waiting out the drain timer.
            if (drainingRef.current) {
              finishDrain();
            }
          }
          // config_ack and other control frames need no UI action.
        } catch {
          // Ignore malformed messages
        }
      };

      ws.onerror = () => {
        // During a stop drain the session is ending anyway — the close/timer
        // path finishes cleanup; don't flash "disconnected" on the way out.
        if (drainingRef.current) return;
        setConnectionStatus("disconnected");
      };

      ws.onclose = () => {
        // Only involuntary closes reach here — deliberate teardown detaches
        // this handler first.
        if (drainingRef.current) {
          // Server closed after (or instead of) session_complete: the stop
          // handshake is over.
          finishDrain();
          return;
        }
        setConnectionStatus("disconnected");
        if (
          shouldReconnect.current &&
          reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS
        ) {
          reconnectAttempts.current += 1;
          wsReconnectsRef.current += 1;
          setTimeout(() => {
            if (shouldReconnect.current) {
              connectWebSocket(sessionId);
            }
          }, RECONNECT_DELAY_MS);
        } else {
          // Out of retries: stop capturing rather than pretend the session
          // is still live.
          shouldReconnect.current = false;
          streamingRef.current = false;
          sessionActiveRef.current = false;
          releaseCapture();
          const call = callRef.current;
          callRef.current = null;
          call?.hangUp();
          pendingRef.current = new Int16Array(0);
          resamplerRef.current = null;
          stopSpeechSafely(); // Session is dead — stop coaching aloud too.
          void stopFastLoop().finally(() => recordSessionDiagnostics());
          // Restore a playback audio session so later replay is audible.
          void setPlaybackMode().catch(() => {});
          setIsRecording(false);
          setSessionActive(false);
        }
      };
    },
    [
      teardownWebSocket,
      finishDrain,
      armDrainTimer,
      speakSuggestion,
      releaseCapture,
      stopFastLoop,
      recordSessionDiagnostics,
    ],
  );

  /**
   * Web capture path (Platform.OS === "web"). expo-audio ships no web
   * recorder, so we capture the mic ourselves with getUserMedia + an
   * AudioWorklet (see utils/webAudioCapture) and feed the SAME resample /
   * int16 / batching / WebSocket pipeline the native path uses — the backend
   * cannot tell the two apart. The on-device fast loop (src/live/webDeps.ts:
   * Silero + ECAPA over onnxruntime-web, the Web Speech API for words) then
   * hears exactly those frames too, from handleAudioBuffer.
   *
   * Ordering matters for iOS Safari, which gates three things on the Start
   * tap's user gesture: the AudioContext (created + resumed at the top of
   * `capture.start()`), speech recognition (`primeWebRecognizer` starts it
   * synchronously — its permission prompt needs the gesture) and speech
   * synthesis (`unlockWebSpeechSynthesis` speaks a silent utterance so the
   * first real suggestion, seconds later, isn't dropped). Everything before
   * the first `await` below runs inside that gesture. getUserMedia inside
   * start() is the mic prompt — requested BEFORE the session opens,
   * mirroring the native path.
   *
   * Known limit (documented for the therapist): the page must stay in the
   * foreground — locking the screen or switching apps stops the microphone
   * (iOS releases it), and the session must be restarted.
   */
  /**
   * The synchronous, gesture-bound half of a web start: unlock TTS, start
   * speech recognition, create + resume the AudioContext and request the
   * mic — all before the first `await`, so it can also run at the top of
   * an in-app call's Answer tap, ahead of the REST join. Null when the
   * browser can't capture at all.
   */
  const beginWebCapture = useCallback((): PreparedWebCapture | null => {
    if (!isWebAudioCaptureSupported()) return null;
    const wantLive = liveModeRef.current && liveCapability.capable;
    // Still inside the Start gesture: unlock TTS, start speech recognition.
    if (speechAvailableRef.current) unlockWebSpeechSynthesis();
    const primed = wantLive ? primeWebRecognizer() : null;
    const capture = new WebAudioCapture({
      onBuffer: handleAudioBuffer,
      onTrackEnded: (reason) => {
        if (!sessionActiveRef.current || drainingRef.current) return;
        setMicError(
          reason === "ended"
            ? "The browser released the microphone (screen locked or app switched?) — stop and start the session again."
            : "The microphone was muted by the browser — check for another app using it, then restart the session.",
        );
      },
    });
    const started = capture.start().then(
      () => null,
      (error: unknown) => ({ error }),
    );
    return { capture, primed, started };
  }, [handleAudioBuffer, liveCapability.capable]);

  const startWebSession = useCallback(
    async (sessionId: string, empathyLevel: number) => {
      // A call's Answer tap prepared the capture already; otherwise this IS
      // the tap (startSession is reached synchronously from onPress).
      const prepared = preparedWebRef.current ?? beginWebCapture();
      preparedWebRef.current = null;
      if (!prepared) {
        // Honest unsupported-browser state. Still run the session (no audio):
        // the coaching UI works and the server reports its own state (e.g.
        // transcription_unavailable) rather than us faking capture.
        setMicError(
          "Your browser can't capture audio — live coaching needs microphone support (use a recent Chrome, Safari, Firefox, or Edge over HTTPS).",
        );
        shouldReconnect.current = true;
        setSessionActive(true);
        connectWebSocket(sessionId);
        return;
      }

      const wantLive = liveModeRef.current && liveCapability.capable;
      const { capture, primed } = prepared;
      const failure = await prepared.started;
      if (failure) {
        const err = failure.error;
        primed?.stop();
        // Permission denied / no mic / unsupported: surface the honest reason
        // and open no session (nothing to record).
        const kind = err instanceof WebCaptureError ? err.kind : "unavailable";
        if (kind === "permission-denied") {
          setMicError(
            "Microphone permission denied — enable microphone access to start a live session.",
          );
        } else if (kind === "no-microphone") {
          setMicError(
            "No microphone found — connect a microphone to start a live session.",
          );
        } else {
          setMicError(
            err instanceof Error && err.message
              ? `Microphone unavailable: ${err.message}`
              : "Microphone unavailable — could not start audio capture.",
          );
        }
        await capture.stop();
        sessionActiveRef.current = false;
        return;
      }

      webCaptureRef.current = capture;
      // Frames start flowing from the worklet now, so open the gate before the
      // socket connects (frames buffer in pendingRef until the WS is OPEN).
      streamingRef.current = true;
      beginAudioKeep(sessionId);
      shouldReconnect.current = true;
      setSessionActive(true);
      connectWebSocket(sessionId);
      setIsRecording(true);

      if (wantLive) {
        // Mic is flowing: bring up the browser fast loop alongside the server
        // stream. Failure degrades to the server path (see startFastLoop).
        await startFastLoop(sessionId, empathyLevel, primed);
      }
    },
    [connectWebSocket, beginWebCapture, startFastLoop, beginAudioKeep, liveCapability.capable],
  );

  /**
   * Journal mode ("listen for my voice"): mic + the journal recorder, and
   * NOTHING else — no WebSocket, no STT, no LLM, no TTS, no kept-audio WAV.
   * Honest gates, in order: the phone app (not the browser), microphone
   * permission, the models + a place to write, and an enrolled owner
   * voiceprint (without one nothing could ever be kept). Every refusal
   * lands in `journal.error` and opens no session.
   */
  const startJournalSession = useCallback(async (sessionId: string) => {
    const fail = (error: string) => {
      setJournal({ ...IDLE_JOURNAL_STATE, error });
      sessionActiveRef.current = false;
    };
    const stream = micStreamRef.current;
    if (Platform.OS === "web" || !stream) {
      fail("The journal needs the phone app's microphone — it isn't available in the browser.");
      return;
    }
    setJournal({ ...IDLE_JOURNAL_STATE, status: "starting" });
    let granted = false;
    try {
      granted = (await requestRecordingPermissionsAsync()).granted;
    } catch {
      granted = false;
    }
    if (!granted) {
      fail("Microphone permission denied — enable microphone access to start the journal.");
      return;
    }
    let recorder: JournalRecorder;
    try {
      recorder = await makeJournalRef.current({
        onState: (state) => {
          // Only the running (or just-stopped) journal drives the screen.
          if (journalRef.current === recorder || lastJournalRef.current === recorder) setJournal(state);
        },
      });
    } catch (err) {
      fail(`Journal unavailable: ${err instanceof Error ? err.message : String(err)}`);
      return;
    }
    if (!recorder.hasSelfPrint) {
      fail("Enroll your voice first — the journal keeps only the stretches that match your voiceprint.");
      return;
    }
    if (!sessionActiveRef.current) return; // Stopped while the models loaded.
    try {
      await prepareJournalAudioRef.current();
      await stream.start();
    } catch (err) {
      fail(
        err instanceof Error && err.message
          ? `Microphone unavailable: ${err.message}`
          : "Microphone unavailable — could not start audio capture.",
      );
      return;
    }
    sessionIdRef.current = sessionId;
    sessionStartedAtRef.current = new Date().toISOString();
    journalRef.current = recorder;
    lastJournalRef.current = null;
    streamingRef.current = true;
    await recorder.start();
    setJournal(recorder.stateSnapshot);
    setSessionActive(true);
    setIsRecording(true);
    setConnectionStatus("idle"); // no server in this loop
    // Best-effort foreground-service hold (Android); the journal runs
    // without it. Taken AFTER the stream is up so it can never block it.
    void holdBackgroundMicRef.current()
      .then((hold) => {
        if (journalRef.current === recorder) micHoldRef.current = hold;
        else void hold?.release();
      })
      .catch(() => {});
  }, []);

  const startSession = useCallback(
    async (
      sessionId: string,
      empathyLevel: number,
      interjectLevel: number = 0,
    ) => {
      // Synchronous double-start guard (a ref: isRecording flips only after
      // the async permission/audio-mode/start chain, far too late to stop a
      // double-tap from opening two WebSockets).
      if (sessionActiveRef.current) {
        if (!drainingRef.current) return; // Starting or active: no-op.
        // Previous session is only draining after a stop — finish it now so
        // the new session starts clean.
        finishDrain();
      }
      sessionActiveRef.current = true;

      sessionIdRef.current = sessionId;
      empathyRef.current = empathyLevel;
      interjectRef.current = interjectLevel;
      reconnectAttempts.current = 0;

      setTranscript([]);
      setSuggestions([]);
      setSpeakerLabel("");
      setTranscriptionAvailable(true);
      setTranscriptionMessage("");
      setMicError("");
      setLiveStatus("");
      setLatencySummary("");
      setToneFlags([]);
      setWatchConnected(false);
      setNudgeFlash(null);
      setSessionSummary(null);
      setLastEpisode(null);
      lastEpisodeRef.current = null;
      lastLoopRef.current = null;
      speakerNamesRef.current = {};
      contrastLabelsRef.current = {};
      setSpeakerNames({});
      speakerLabelsRef.current = {};
      trackerRef.current = new PleasantnessTracker();
      setScoreboard(null);
      setEscalationCount(0);
      escalationRef.current = 0;
      latencyLogRef.current = [];
      // The legacy path never starts the fast loop, so stamp the start here
      // too (startFastLoop re-stamps when the loop actually comes up).
      sessionStartedAtRef.current = new Date().toISOString();
      localTurnsRef.current = [];
      toneFlagsRef.current = [];
      identitiesRef.current = [];
      liveSttFailedRef.current = false;
      pendingRef.current = new Int16Array(0);
      resamplerRef.current = null;
      // Fresh session, fresh protocol detection: don't let the previous
      // server's transcript events silence a legacy server's fallback.
      sawTranscriptEventRef.current = false;
      // Fresh session, fresh diarization: "Speaker A" is whoever speaks first
      // in THIS session, so a previous session's toggle must never leak into
      // the new initial config frame — it could invert coaching entirely
      // (nudges for the other person, response cards for the user). Reset
      // BEFORE the socket opens so onopen always sends the per-session
      // "you speak first" default.
      selfSpeakerRef.current = "Speaker A";
      setSelfSpeakerState("Speaker A");

      if (sessionModeRef.current === "journal") {
        // Mic + journal recorder only: no socket, no loop, no keeper.
        await startJournalSession(sessionId);
        return;
      }
      setJournal(IDLE_JOURNAL_STATE);

      if (Platform.OS === "web") {
        await startWebSession(sessionId, empathyLevel);
        return;
      }

      const stream = micStreamRef.current;
      if (stream) {
        // Ask for the microphone BEFORE opening the session — a denied
        // permission is a user choice to respect, not something to route
        // around with an audio-less session.
        let granted = false;
        try {
          const permission = await requestRecordingPermissionsAsync();
          granted = permission.granted;
        } catch {
          granted = false;
        }
        if (!granted) {
          setMicError(
            "Microphone permission denied — enable microphone access to start a live session.",
          );
          sessionActiveRef.current = false;
          return;
        }
      } else {
        // expo-audio's web build has no realtime capture (its useAudioStream
        // returns a null stream). Say so honestly, but still run the session:
        // config/empathy flow both ways and the server reports its own state
        // (e.g. transcription_unavailable) instead of us faking anything.
        setMicError(
          "Live microphone capture is not supported on this platform yet — running the session without audio.",
        );
      }

      shouldReconnect.current = true;
      setSessionActive(true);
      connectWebSocket(sessionId);

      if (!stream) {
        // No capture backend: session runs, but nothing records and no
        // binary frames are ever sent. isRecording stays honestly false.
        return;
      }

      try {
        // Configure the shared audio session for mic capture. This leaves the
        // session record-oriented, which on Android silences later media
        // playback — so every path that ends the session resets it to a
        // playback mode (see finishDrain and the teardown paths below).
        await setRecordingMode();
        await stream.start();
      } catch (err) {
        // Mic capture failed after the socket opened: close the session
        // cleanly and surface the real reason — never stream silence.
        shouldReconnect.current = false;
        sessionActiveRef.current = false;
        teardownWebSocket();
        setSessionActive(false);
        setConnectionStatus("idle");
        setMicError(
          err instanceof Error && err.message
            ? `Microphone unavailable: ${err.message}`
            : "Microphone unavailable — could not start audio capture.",
        );
        return;
      }

      streamingRef.current = true;
      beginAudioKeep(sessionId);
      setIsRecording(true);

      if (liveModeRef.current && liveCapability.capable) {
        // Native mic is flowing: bring up the on-device loop alongside the
        // server stream. Failure degrades to the server path (see
        // startFastLoop) — the session is already live either way.
        await startFastLoop(sessionId, empathyLevel);
      }
    },
    [
      connectWebSocket,
      teardownWebSocket,
      finishDrain,
      startWebSession,
      startJournalSession,
      startFastLoop,
      beginAudioKeep,
      liveCapability.capable,
    ],
  );

  const sendEmpathyUpdate = useCallback((level: number) => {
    empathyRef.current = level;
    fastLoopRef.current?.setEmpathy(level);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      // Empathy changes go through the same `config` channel the server
      // understands (it rejects unknown message types).
      wsRef.current.send(
        JSON.stringify({ type: "config", empathy_slider: level }),
      );
    }
  }, []);

  const sendInterjectUpdate = useCallback((value: number) => {
    const rounded = Math.round(value);
    interjectRef.current = rounded;
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      // Same `config` channel as empathy updates — the server rejects
      // unknown message types.
      wsRef.current.send(
        JSON.stringify({ type: "config", interject_level: rounded }),
      );
    }
  }, []);

  /**
   * Set which diarized speaker is the coached user and, if a session is live,
   * tell the server immediately via the same `config` channel empathy/interject
   * use. Mirrors sendInterjectUpdate. Scoped to the CURRENT session: startSession
   * resets the choice to "Speaker A" because diarization labels are re-assigned
   * per session by speaking order (a stale toggle would invert the coaching).
   */
  const setSelfSpeaker = useCallback((label: string) => {
    selfSpeakerRef.current = label;
    setSelfSpeakerState(label);
    // The on-device loop applies the same convention to its own unknown
    // clusters (nudge vs response, haptics on self turns only).
    fastLoopRef.current?.setSelfSpeakerFallback(label);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ type: "config", self_speaker: label }),
      );
    }
  }, []);

  const setLiveMode = useCallback((on: boolean) => {
    liveModeRef.current = on;
    setLiveModeState(on);
  }, []);

  const setSessionMode = useCallback((mode: LiveMode) => {
    sessionModeRef.current = mode;
    loopModeRef.current = mode;
    setSessionModeState(mode);
  }, []);

  const clearNudgeFlash = useCallback(() => setNudgeFlash(null), []);

  const displayNameOf = useCallback(
    (speaker: string) => speakerNames[speaker]?.displayName ?? speaker,
    [speakerNames],
  );

  /**
   * Mid-call naming: "that Speaker B is Mom".
   *
   * Immediately (synchronously, before any network): the transcript
   * relabels, the fast loop binds the cluster to the person (later turns
   * carry `speaker_person_id`/`is_self`; the prompt says "Mom"), the
   * session record rewrites earlier turns on that label, and the server is
   * told (`speaker_label`) so its coach uses the name and its side-aware
   * coaching knows who "me" is. Then, best-effort: a NEW person is created
   * on the server and their voice learned from the session's pooled audio
   * (≥ 3 s) through the existing enroll endpoint, and the print is added to
   * the on-device labeler so the rest of the call matches by voice too.
   * After the session, once the episode exists, the name is PATCHed onto
   * it exactly as "Who is this?" does on a stored recording.
   */
  const labelSpeaker = useCallback(
    async (speaker: string, choice: LabelSpeakerChoice): Promise<LabelSpeakerOutcome> => {
      const binding: SpeakerBinding = {
        personId: choice.personId,
        displayName: choice.displayName,
        isSelf: choice.isSelf,
      };
      // A person can be one voice: naming a second label as the same person
      // releases the first (the user corrected themselves).
      const names: Record<string, SpeakerBinding> = {};
      for (const [label, b] of Object.entries(speakerNamesRef.current)) {
        if (b.personId !== binding.personId || label === speaker) names[label] = b;
      }
      names[speaker] = binding;
      speakerNamesRef.current = names;
      setSpeakerNames(names);
      speakerLabelsRef.current = {
        ...speakerLabelsRef.current,
        [speaker]: {
          display_name: choice.displayName,
          person_id: choice.isNew ? null : choice.personId,
          is_self: choice.isSelf,
        },
      };
      setTranscript((prev) =>
        prev.map((t) =>
          (t.speakerId ?? t.speaker) === speaker
            ? { ...t, speaker: choice.displayName, speakerId: speaker }
            : t,
        ),
      );
      setSpeakerLabel((current) => (current === speaker ? choice.displayName : current));
      if (choice.isSelf) {
        selfSpeakerRef.current = speaker;
        setSelfSpeakerState(speaker);
      }
      // The record carries the person id the user chose (the loop's later
      // turns do too); the stored episode attaches it only once that person
      // exists on the server (see manual_labels_from_live) — the wire is
      // honest about what the USER said, the store about what exists.
      for (const ev of localTurnsRef.current) {
        if (ev.speaker !== speaker) continue;
        ev.speaker_person_id = choice.personId;
        ev.is_self = choice.isSelf;
      }
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(
            JSON.stringify({
              type: "speaker_label",
              session_id: sessionIdRef.current,
              speaker,
              person_id: choice.isNew ? null : choice.personId,
              display_name: choice.displayName,
              is_self: choice.isSelf,
            }),
          );
        } catch {
          // Socket mid-close: the session record still carries the label.
        }
      }

      const loop = fastLoopRef.current ?? lastLoopRef.current;
      const seconds = loop ? Math.round(loop.speakerAudioSeconds(speaker) * 10) / 10 : null;
      let enrolled = false;
      let text = `${choice.displayName} is labeled for the rest of this call.`;
      if (loop) {
        // Bind first (instant), then learn the voice when there is one.
        loop.bindSpeaker(speaker, binding);
        if (choice.isNew) {
          const pcm = loop.speakerAudio(speaker);
          if (pcm.length / 16000 >= MIN_ENROLL_SECONDS) {
            try {
              const result = await enrollRef.current(pcm, {
                personId: choice.personId,
                displayName: choice.displayName,
              });
              enrolled = true;
              // Now a real person on the server: the stored label may
              // attach the id (the manual-person rung).
              speakerLabelsRef.current = {
                ...speakerLabelsRef.current,
                [speaker]: { ...speakerLabelsRef.current[speaker], person_id: choice.personId },
              };
              const embedding = await loop.embedSpeaker(speaker);
              if (embedding) {
                loop.bindSpeaker(speaker, binding, {
                  personId: choice.personId,
                  displayName: choice.displayName,
                  isSelf: choice.isSelf,
                  embedding,
                });
              }
              text +=
                ` Learned ${result.seconds} s of ${choice.displayName}’s voice — they’ll be ` +
                "recognized next time. Stored as a numeric signature, not the audio.";
            } catch (err) {
              const msg = err instanceof Error ? err.message : String(err);
              text += ` Couldn’t remember the voice yet (${msg}). You can add ${choice.displayName} under People later.`;
            }
          } else {
            text +=
              ` Only ${seconds ?? 0} s of ${choice.displayName}’s voice so far — ` +
              `after ${MIN_ENROLL_SECONDS} s you can name them again to remember the voice.`;
          }
        } else if (choice.personId === SELF_PERSON_ID || choice.isSelf) {
          text += " The coach now knows this voice is you.";
        } else {
          text += ` The app already knows ${choice.displayName}’s voice.`;
        }
      } else if (choice.isNew) {
        text += " No on-device audio was kept, so the voice can’t be learned from this session.";
      }

      // Post-session: the episode already exists — put the name on it.
      const episodeId = lastEpisodeRef.current?.episodeId;
      if (!sessionActiveRef.current && episodeId) {
        const personId = speakerLabelsRef.current[speaker]?.person_id ?? null;
        try {
          await patchLabelsRef.current(
            episodeId,
            { [speaker]: choice.displayName },
            personId ? { [speaker]: personId } : undefined,
          );
          text += " Saved to the session record.";
        } catch {
          text += " (Couldn’t save the name to the stored session — check your connection.)";
        }
      }
      return { text, enrolled, seconds };
    },
    [],
  );

  const runPreflight = useCallback(async () => {
    if (!liveCapability.capable || preflightInFlightRef.current) return;
    preflightInFlightRef.current = true;
    setPreflight({ status: "probing" });
    try {
      const capabilities = await probeRef.current();
      setPreflight({ status: "ready", capabilities });
      useDiagnosticsStore.getState().setCapability(capabilities, liveCapability.reason);
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      setPreflight({ status: "failed", reason });
      useDiagnosticsStore.getState().setCapability(null, reason);
    } finally {
      preflightInFlightRef.current = false;
    }
  }, [liveCapability.capable, liveCapability.reason]);

  // --- In-app call (Call mode) ------------------------------------------------

  /** The WebRTC adapter for this platform (or the test seam). */
  const makeRtcAdapter = useCallback((): RtcAdapter => {
    const getCaptureStream = () => webCaptureRef.current?.mediaStream ?? null;
    if (makeRtcAdapterRef.current) return makeRtcAdapterRef.current(getCaptureStream);
    return Platform.OS === "web" ? createWebRtcAdapter({ getCaptureStream }) : createNativeRtcAdapter();
  }, []);

  /**
   * Create or join a call, then start the session in Call mode. On the web
   * everything Safari gates on the tap (AudioContext, speech recognition,
   * getUserMedia, the remote <audio> element) happens synchronously at the
   * top, BEFORE the REST round-trip — startWebSession then consumes it.
   */
  const beginCall = useCallback(
    async (
      how: { kind: "create" } | { kind: "join"; code: string; role: CallRole },
      empathyLevel: number,
      interjectLevel: number,
    ) => {
      const role: CallRole = how.kind === "join" ? how.role : "participant";
      if (sessionActiveRef.current && !drainingRef.current) return;
      // How the others see us: the account's name, else the email's local
      // part (the server falls back to the email itself, then the slot).
      const me = useAuthStore.getState().user;
      const displayName =
        (me?.displayName && me.displayName.trim()) || (me?.email ? me.email.split("@")[0] : "") || undefined;
      if (callRef.current) {
        callRef.current.hangUp();
        callRef.current = null;
      }
      const setView = (v: CallView) => {
        callViewRef.current = v;
        setCallView(v);
      };
      setView({ ...IDLE_CALL_VIEW, status: "creating" });
      let adapter: RtcAdapter;
      try {
        adapter = makeRtcAdapter();
      } catch (err) {
        setView({ ...IDLE_CALL_VIEW, status: "failed", error: err instanceof Error ? err.message : String(err) });
        return;
      }
      rtcAdapterRef.current = adapter;
      const session = new CallSession({
        adapter,
        role,
        displayName,
        send: (message: CallClientMessage) => {
          const ws = wsRef.current;
          if (!ws || ws.readyState !== WebSocket.OPEN) return false;
          try {
            ws.send(JSON.stringify(message));
            return true;
          } catch {
            return false;
          }
        },
        onChange: (v) => {
          setView(v);
          // The other side hung up / the server ended it: the session is
          // over too. (A local hangUp detaches callRef first, so this never
          // re-enters stopSession.)
          if ((v.status === "ended" || v.status === "failed") && callRef.current === session) {
            callRef.current = null;
            if (sessionActiveRef.current && !drainingRef.current) void stopSession();
          }
        },
      });
      callRef.current = session;
      if (Platform.OS === "web") {
        session.prime();
        preparedWebRef.current = beginWebCapture();
      }
      let created;
      try {
        created =
          how.kind === "create"
            ? await callApiRef.current.create({ displayName, maxParticipants: 3 })
            : await callApiRef.current.join(how.code, how.role, displayName);
      } catch (err) {
        const prepared = preparedWebRef.current;
        preparedWebRef.current = null;
        prepared?.primed?.stop();
        void prepared?.capture.stop().catch(() => {});
        callRef.current = null;
        setView({ ...IDLE_CALL_VIEW, status: "failed", error: err instanceof Error ? err.message : String(err) });
        return;
      }
      session.begin(created);
      // Call mode is implied by starting a call: the record + UI say "call".
      // A therapist observer runs the loop in "therapist" so their own speech
      // is transcribed and merged (turn_local) but never spoken to or coached.
      sessionModeRef.current = "call";
      loopModeRef.current = role === "therapist" ? "therapist" : "call";
      setSessionModeState("call");
      await startSession(`call-${created.callId.replace(/[^A-Za-z0-9_-]/g, "")}`, empathyLevel, interjectLevel);
      if (!sessionActiveRef.current) {
        // The session never opened (mic denied …): no call either.
        const c = callRef.current;
        callRef.current = null;
        c?.hangUp();
        void callApiRef.current.end(created.callId);
        return;
      }
      if (adapter.setRoute) void adapter.setRoute(callRouteRef.current).catch(() => {});
    },
    [makeRtcAdapter, beginWebCapture, startSession, stopSession],
  );

  const startCall = useCallback(
    (empathyLevel: number, interjectLevel: number = 0) => beginCall({ kind: "create" }, empathyLevel, interjectLevel),
    [beginCall],
  );

  const joinCall = useCallback(
    (code: string, empathyLevel: number, interjectLevel: number = 0, role: CallRole = "participant") =>
      beginCall({ kind: "join", code, role }, empathyLevel, interjectLevel),
    [beginCall],
  );

  const hangUp = useCallback(async () => {
    if (sessionActiveRef.current || drainingRef.current) {
      await stopSession();
      return;
    }
    const call = callRef.current;
    callRef.current = null;
    if (call) {
      const callId = call.callId;
      call.hangUp();
      if (callId) void callApiRef.current.end(callId);
    }
  }, [stopSession]);

  const setCallMuted = useCallback((muted: boolean) => {
    const call = callRef.current;
    if (call) {
      call.setMuted(muted);
    } else {
      const v = { ...callViewRef.current, muted };
      callViewRef.current = v;
      setCallView(v);
    }
  }, []);

  const setCallRoute = useCallback((route: AudioRoute) => {
    callRouteRef.current = route;
    setCallRouteState(route);
    const adapter = rtcAdapterRef.current;
    if (adapter?.setRoute && callRef.current) void adapter.setRoute(route).catch(() => {});
  }, []);

  return {
    isRecording,
    sessionActive,
    transcript,
    suggestions,
    speakerLabel,
    selfSpeaker,
    setSelfSpeaker,
    connectionStatus,
    transcriptionAvailable,
    transcriptionMessage,
    micError,
    speechAvailable,
    speechEnabled,
    setSpeechEnabled,
    startSession,
    stopSession,
    sendEmpathyUpdate,
    sendInterjectUpdate,
    liveCapable: liveCapability.capable,
    liveCapabilityReason: liveCapability.reason,
    liveMode,
    setLiveMode,
    sessionMode,
    setSessionMode,
    liveStatus,
    nudgeFlash,
    clearNudgeFlash,
    latencySummary,
    toneFlags,
    watchConnected,
    preflight,
    runPreflight,
    escalationCount,
    sessionSummary,
    lastEpisode,
    audioKeep,
    journal,
    retryJournalUploads,
    speakerNames,
    displayNameOf,
    labelSpeaker,
    scoreboard,
    call: callView,
    startCall,
    joinCall,
    hangUp,
    setCallMuted,
    callRoute,
    setCallRoute,
  };
}
