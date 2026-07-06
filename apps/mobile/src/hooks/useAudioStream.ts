import { useState, useRef, useCallback, useEffect } from "react";
import { Platform } from "react-native";
import {
  useAudioStream as useMicrophoneStream,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
} from "expo-audio";
import type { AudioStreamBuffer } from "expo-audio";
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
import { getCachedToken } from "../auth/authToken";

const API_URL =
  process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

const WS_BASE = API_URL.replace(/^http/, "ws");

export interface TranscriptEntry {
  speaker: string;
  text: string;
  timestamp: number;
}

export interface SuggestionEntry {
  text: string;
  tone: string;
}

type ConnectionStatus = "idle" | "connecting" | "live" | "disconnected";

interface UseAudioStreamReturn {
  isRecording: boolean;
  /** True while a session is running, even when mic capture is unavailable
   *  (e.g. web) and no audio is being recorded. Drives the start/stop toggle. */
  sessionActive: boolean;
  transcript: TranscriptEntry[];
  suggestions: SuggestionEntry[];
  speakerLabel: string;
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
  startSession: (sessionId: string, empathyLevel: number) => Promise<void>;
  stopSession: () => Promise<void>;
  sendEmpathyUpdate: (level: number) => void;
}

const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECT_ATTEMPTS = 5;
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
  return (
    typeof window !== "undefined" &&
    typeof window.speechSynthesis !== "undefined" &&
    typeof SpeechSynthesisUtterance !== "undefined"
  );
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

export function useAudioStream(): UseAudioStreamReturn {
  const [isRecording, setIsRecording] = useState(false);
  const [sessionActive, setSessionActive] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [suggestions, setSuggestions] = useState<SuggestionEntry[]>([]);
  const [speakerLabel, setSpeakerLabel] = useState("");
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("idle");
  const [transcriptionAvailable, setTranscriptionAvailable] = useState(true);
  const [transcriptionMessage, setTranscriptionMessage] = useState("");
  const [micError, setMicError] = useState("");
  const [speechAvailable, setSpeechAvailable] = useState(detectSpeechSupport);
  const [speechEnabled, setSpeechEnabledState] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string>("");
  const reconnectAttempts = useRef(0);
  const shouldReconnect = useRef(false);
  const empathyRef = useRef(50);
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
      pendingRef.current = concatInt16(
        pendingRef.current,
        float32ToInt16(samples),
      );
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
    teardownWebSocket();
    setIsRecording(false);
    setSessionActive(false);
    setConnectionStatus("idle");
  }, [teardownWebSocket]);

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
    shouldReconnect.current = false;
    streamingRef.current = false;
    // The session is over: never keep coaching aloud after the user stops.
    // (drainingRef gates speakSuggestion, so late drain-window suggestions
    // still render visually but are not spoken.)
    stopSpeechSafely();
    releaseCapture();
    const resampler = resamplerRef.current;
    resamplerRef.current = null;

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
  }, [finishDrain, armDrainTimer, releaseCapture]);

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
      stopSpeechSafely(); // Never keep talking after the screen is gone.
      teardownWebSocket();
    };
  }, [teardownWebSocket, releaseCapture]);

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
            ...(idToken ? { id_token: idToken } : {}),
          }),
        );
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

          if (data.type === "suggestion") {
            // The server bundles the transcribed utterance and its coaching
            // suggestions in one event (see server SuggestionEvent).
            const speaker = data.speaker || "Unknown";
            setSpeakerLabel(speaker);
            if (data.utterance_text) {
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
              setSuggestions(items.map((text) => ({ text, tone })));
              // Earpiece mode: speak the newest TOP suggestion with free
              // on-device TTS. (The event also carries data.audio_b64 —
              // Deepgram Aura mp3, paid key required — deliberately ignored;
              // kept as a future premium-playback option.)
              speakSuggestion(items[0]);
            }
          } else if (data.type === "transcription_unavailable") {
            // Be explicit instead of silently showing an empty live screen.
            setTranscriptionAvailable(false);
            setTranscriptionMessage(data.reason || "Transcription unavailable");
          } else if (data.type === "session_complete") {
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
          pendingRef.current = new Int16Array(0);
          resamplerRef.current = null;
          stopSpeechSafely(); // Session is dead — stop coaching aloud too.
          setIsRecording(false);
          setSessionActive(false);
        }
      };
    },
    [teardownWebSocket, finishDrain, armDrainTimer, speakSuggestion, releaseCapture],
  );

  /**
   * Web capture path (Platform.OS === "web"). expo-audio ships no web
   * recorder, so we capture the mic ourselves with getUserMedia + an
   * AudioWorklet (see utils/webAudioCapture) and feed the SAME resample /
   * int16 / batching / WebSocket pipeline the native path uses — the backend
   * cannot tell the two apart.
   *
   * Ordering matters for iOS Safari: `capture.start()` is reached with only
   * synchronous setState calls before it, so the AudioContext is created and
   * resumed inside the Start-button gesture (Safari refuses to resume it
   * later). getUserMedia inside start() is the permission prompt — we request
   * the mic BEFORE opening the session, mirroring the native path.
   */
  const startWebSession = useCallback(
    async (sessionId: string) => {
      if (!isWebAudioCaptureSupported()) {
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

      const capture = new WebAudioCapture({ onBuffer: handleAudioBuffer });
      try {
        await capture.start();
      } catch (err) {
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
      shouldReconnect.current = true;
      setSessionActive(true);
      connectWebSocket(sessionId);
      setIsRecording(true);
    },
    [connectWebSocket, handleAudioBuffer],
  );

  const startSession = useCallback(
    async (sessionId: string, empathyLevel: number) => {
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
      reconnectAttempts.current = 0;

      setTranscript([]);
      setSuggestions([]);
      setSpeakerLabel("");
      setTranscriptionAvailable(true);
      setTranscriptionMessage("");
      setMicError("");
      pendingRef.current = new Int16Array(0);
      resamplerRef.current = null;

      if (Platform.OS === "web") {
        await startWebSession(sessionId);
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
        await setAudioModeAsync({
          allowsRecording: true,
          playsInSilentMode: true,
        });
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
      setIsRecording(true);
    },
    [connectWebSocket, teardownWebSocket, finishDrain, startWebSession],
  );

  const sendEmpathyUpdate = useCallback((level: number) => {
    empathyRef.current = level;
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      // Empathy changes go through the same `config` channel the server
      // understands (it rejects unknown message types).
      wsRef.current.send(
        JSON.stringify({ type: "config", empathy_slider: level }),
      );
    }
  }, []);

  return {
    isRecording,
    sessionActive,
    transcript,
    suggestions,
    speakerLabel,
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
  };
}
