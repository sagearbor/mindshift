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
  startSession: (sessionId: string, empathyLevel: number) => Promise<void>;
  stopSession: () => Promise<void>;
  sendEmpathyUpdate: (level: number) => void;
}

const RECONNECT_DELAY_MS = 2000;
const MAX_RECONNECT_ATTEMPTS = 5;

export function useAudioStream(): UseAudioStreamReturn {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [suggestions, setSuggestions] = useState<SuggestionEntry[]>([]);
  const [speakerLabel, setSpeakerLabel] = useState("");
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("idle");

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
      const url = `${WS_BASE}/ws/session/${sessionId}?empathy=${empathyLevel}`;
      setConnectionStatus("connecting");

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionStatus("live");
        reconnectAttempts.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === "transcript") {
            const entry: TranscriptEntry = {
              speaker: data.speaker || "Unknown",
              text: data.text || "",
              timestamp: data.timestamp || Date.now(),
            };
            setSpeakerLabel(entry.speaker);
            setTranscript((prev) => [...prev, entry]);
          } else if (data.type === "suggestion") {
            const suggestion: SuggestionEntry = {
              text: data.text || "",
              tone: data.tone || "neutral",
            };
            setSuggestions((prev) => [...prev, suggestion]);
          }
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

    recording.setOnRecordingStatusUpdate((status) => {
      if (
        status.isRecording &&
        status.durationMillis > 0 &&
        wsRef.current?.readyState === WebSocket.OPEN
      ) {
        // In a production implementation, we'd read audio chunks from
        // the recording buffer and send them over WebSocket. Expo-av
        // doesn't expose raw PCM streaming directly, so the backend
        // would use the recording URI or a chunked upload approach.
        // For now, we signal the backend that audio is being captured.
        wsRef.current.send(
          JSON.stringify({ type: "audio_status", recording: true }),
        );
      }
    });

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
      wsRef.current.send(
        JSON.stringify({ type: "empathy_update", level }),
      );
    }
  }, []);

  return {
    isRecording,
    transcript,
    suggestions,
    speakerLabel,
    connectionStatus,
    startSession,
    stopSession,
    sendEmpathyUpdate,
  };
}
