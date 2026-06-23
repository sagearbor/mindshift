import { useState, useRef, useCallback, useEffect } from "react";
import { Audio } from "expo-av";
import { Platform } from "react-native";

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
  transcript: TranscriptEntry[];
  suggestions: SuggestionEntry[];
  speakerLabel: string;
  connectionStatus: ConnectionStatus;
  transcriptionAvailable: boolean;
  transcriptionMessage: string;
  startSession: (sessionId: string, empathyLevel: number) => Promise<void>;
  stopSession: () => Promise<void>;
  sendEmpathyUpdate: (level: number) => void;
}

const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECT_ATTEMPTS = 5;

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
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [suggestions, setSuggestions] = useState<SuggestionEntry[]>([]);
  const [speakerLabel, setSpeakerLabel] = useState("");
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("idle");
  const [transcriptionAvailable, setTranscriptionAvailable] = useState(true);
  const [transcriptionMessage, setTranscriptionMessage] = useState("");

  const wsRef = useRef<WebSocket | null>(null);
  const recordingRef = useRef<Audio.Recording | null>(null);
  const sessionIdRef = useRef<string>("");
  const reconnectAttempts = useRef(0);
  const shouldReconnect = useRef(false);
  const empathyRef = useRef(50);

  const cleanup = useCallback(async () => {
    shouldReconnect.current = false;
    if (recordingRef.current) {
      try {
        await recordingRef.current.stopAndUnloadAsync();
      } catch {
        // Recording may already be stopped
      }
      recordingRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsRecording(false);
    setConnectionStatus("idle");
  }, []);

  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  const connectWebSocket = useCallback(
    (sessionId: string, empathyLevel: number) => {
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
          }
          // config_ack and other control frames need no UI action.
        } catch {
          // Ignore malformed messages
        }
      };

      ws.onerror = () => {
        setConnectionStatus("disconnected");
      };

      ws.onclose = () => {
        setConnectionStatus("disconnected");
        if (
          shouldReconnect.current &&
          reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS
        ) {
          reconnectAttempts.current += 1;
          setTimeout(() => {
            if (shouldReconnect.current) {
              connectWebSocket(sessionId, empathyRef.current);
            }
          }, RECONNECT_DELAY_MS);
        }
      };
    },
    [],
  );

  const startRecording = useCallback(async () => {
    const { status } = await Audio.requestPermissionsAsync();
    if (status !== "granted") {
      throw new Error("Microphone permission not granted");
    }

    await Audio.setAudioModeAsync({
      allowsRecordingIOS: true,
      playsInSilentModeIOS: true,
    });

    const recording = new Audio.Recording();
    await recording.prepareToRecordAsync(
      Audio.RecordingOptionsPresets.HIGH_QUALITY,
    );

    // NOTE: streaming raw audio chunks to the server is not yet wired up.
    // expo-av doesn't expose raw PCM streaming directly, so this needs a
    // chunked-upload or native-module approach. We intentionally do NOT send
    // a placeholder message here — the server only understands binary audio
    // and `config`, and would reject anything else. Until real chunk
    // streaming lands, the server reports `transcription_unavailable`.

    await recording.startAsync();
    recordingRef.current = recording;
    setIsRecording(true);
  }, []);

  const startSession = useCallback(
    async (sessionId: string, empathyLevel: number) => {
      sessionIdRef.current = sessionId;
      empathyRef.current = empathyLevel;
      shouldReconnect.current = true;
      reconnectAttempts.current = 0;

      setTranscript([]);
      setSuggestions([]);
      setSpeakerLabel("");
      setTranscriptionAvailable(true);
      setTranscriptionMessage("");

      connectWebSocket(sessionId, empathyLevel);

      if (Platform.OS !== "web") {
        await startRecording();
      }
    },
    [connectWebSocket, startRecording],
  );

  const stopSession = useCallback(async () => {
    await cleanup();
  }, [cleanup]);

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
    transcript,
    suggestions,
    speakerLabel,
    connectionStatus,
    transcriptionAvailable,
    transcriptionMessage,
    startSession,
    stopSession,
    sendEmpathyUpdate,
  };
}
