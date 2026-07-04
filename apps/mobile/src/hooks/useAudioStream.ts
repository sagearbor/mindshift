import { useState, useRef, useCallback, useEffect } from "react";
import {
  useAudioStream as useMicrophoneStream,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
} from "expo-audio";
import type { AudioStreamBuffer } from "expo-audio";
import {
  concatInt16,
  downmixToMono,
  float32ToInt16,
  StreamingResampler,
} from "../utils/audio";

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
  startSession: (sessionId: string, empathyLevel: number) => Promise<void>;
  stopSession: () => Promise<void>;
  sendEmpathyUpdate: (level: number) => void;
}

const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECT_ATTEMPTS = 5;
/**
 * After a manual stop we keep the socket open this long so the server can
 * deliver the final utterance's suggestion (Deepgram finishes a few hundred
 * ms after the last audio frame) before it sends `session_complete`.
 */
const STOP_DRAIN_TIMEOUT_MS = 4000;

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

  const stopSession = useCallback(async () => {
    if (drainingRef.current) return; // Stop already in progress.
    shouldReconnect.current = false;
    streamingRef.current = false;
    try {
      micStreamRef.current?.stop();
    } catch {
      // Stream may already be stopped.
    }
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
      drainTimerRef.current = setTimeout(() => {
        finishDrain();
      }, STOP_DRAIN_TIMEOUT_MS);
      return;
    }

    // Socket already closed / mid-reconnect: nothing to hand-shake with —
    // clean up immediately, exactly as before.
    pendingRef.current = new Int16Array(0);
    finishDrain();
  }, [finishDrain]);

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
      try {
        micStreamRef.current?.stop();
      } catch {
        // Stream may already be stopped.
      }
      teardownWebSocket();
    };
  }, [teardownWebSocket]);

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
        // message — there is no query-param channel.
        ws.send(
          JSON.stringify({
            type: "config",
            empathy_slider: empathyRef.current,
          }),
        );
      };

      ws.onmessage = (event) => {
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
          try {
            micStreamRef.current?.stop();
          } catch {
            // Stream may already be stopped.
          }
          pendingRef.current = new Int16Array(0);
          resamplerRef.current = null;
          setIsRecording(false);
          setSessionActive(false);
        }
      };
    },
    [teardownWebSocket, finishDrain],
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
    [connectWebSocket, teardownWebSocket, finishDrain],
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
    startSession,
    stopSession,
    sendEmpathyUpdate,
  };
}
