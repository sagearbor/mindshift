import { renderHook, act } from "@testing-library/react-native";

/**
 * Controllable expo-audio mock (overrides the default one in jest-setup).
 * Captures the onBuffer callback the hook registers so tests can push
 * synthetic PCM buffers through the real conversion/batching pipeline.
 */
const mockMic = {
  onBuffer: null as
    | ((buffer: {
        data: ArrayBuffer;
        sampleRate: number;
        channels: number;
        timestamp: number;
      }) => void)
    | null,
  streamAvailable: true,
  start: jest.fn<Promise<void>, []>(),
  stop: jest.fn(),
  requestPermissions: jest.fn<
    Promise<{ status: string; granted: boolean }>,
    []
  >(),
  setAudioMode: jest.fn<Promise<void>, [unknown]>(),
};

jest.mock("expo-audio", () => ({
  __esModule: true,
  requestRecordingPermissionsAsync: () => mockMic.requestPermissions(),
  setAudioModeAsync: (mode: unknown) => mockMic.setAudioMode(mode),
  useAudioStream: (options?: { onBuffer?: (buffer: never) => void }) => {
    mockMic.onBuffer = (options?.onBuffer ?? null) as typeof mockMic.onBuffer;
    return {
      stream: mockMic.streamAvailable
        ? {
            id: "mock-stream",
            sampleRate: 16000,
            channels: 1,
            isStreaming: false,
            start: mockMic.start,
            stop: mockMic.stop,
          }
        : null,
      isStreaming: false,
    };
  },
}));

import * as Speech from "expo-speech";
import { useAudioStream } from "../src/hooks/useAudioStream";

const speakMock = Speech.speak as jest.Mock;
const speechStopMock = Speech.stop as jest.Mock;

/**
 * Fake WebSocket capturing sent frames and letting tests drive server events.
 * Mirrors the subset of the WebSocket API the hook uses. Text frames (JSON)
 * and binary frames (PCM audio) are recorded separately.
 */
class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];

  url: string;
  readyState = FakeWebSocket.OPEN;
  sent: string[] = [];
  sentBinary: ArrayBuffer[] = [];
  onopen: ((e: unknown) => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;
  onclose: ((e: unknown) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  send(data: string | ArrayBuffer | ArrayBufferView) {
    if (typeof data === "string") {
      this.sent.push(data);
    } else if (ArrayBuffer.isView(data)) {
      this.sentBinary.push(
        data.buffer.slice(
          data.byteOffset,
          data.byteOffset + data.byteLength,
        ) as ArrayBuffer,
      );
    } else {
      this.sentBinary.push(data);
    }
  }

  close() {
    this.readyState = 3;
    this.onclose?.({});
  }

  // --- test helpers ---
  emitOpen() {
    this.onopen?.({});
  }
  emitServer(obj: unknown) {
    this.onmessage?.({ data: JSON.stringify(obj) });
  }
  sentJson() {
    return this.sent.map((s) => JSON.parse(s));
  }
}

/** Build a fake PCM capture buffer (constant-valued float32 samples). */
function makePcmBuffer(
  sampleCount: number,
  { sampleRate = 16000, channels = 1, value = 0.25 } = {},
) {
  const samples = new Float32Array(sampleCount).fill(value);
  return { data: samples.buffer, sampleRate, channels, timestamp: 0 };
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  // @ts-expect-error — install fake WebSocket for the hook under test
  global.WebSocket = FakeWebSocket;

  mockMic.onBuffer = null;
  mockMic.streamAvailable = true;
  mockMic.start.mockReset().mockResolvedValue(undefined);
  mockMic.stop.mockReset();
  mockMic.requestPermissions
    .mockReset()
    .mockResolvedValue({ status: "granted", granted: true });
  mockMic.setAudioMode.mockReset().mockResolvedValue(undefined);

  speakMock.mockReset();
  speechStopMock.mockReset().mockResolvedValue(undefined);
});

async function startLiveSession(empathy = 50) {
  const hook = await renderHook(() => useAudioStream());
  await act(async () => {
    await hook.result.current.startSession("sess-1", empathy);
  });
  const ws = FakeWebSocket.instances.at(-1)!;
  await act(() => ws.emitOpen());
  return { hook, ws };
}

describe("useAudioStream — WebSocket protocol", () => {
  it("maps a suggestion event into transcript + suggestions", async () => {
    const { hook, ws } = await startLiveSession(75);

    await act(() =>
      ws.emitServer({
        type: "suggestion",
        session_id: "sess-1",
        utterance_text: "You never listen to me.",
        speaker: "Speaker A",
        suggestions: ["I hear you.", "Tell me more.", "That sounds hard."],
        empathy_slider: 75,
        audio_b64: null,
      }),
    );

    const { transcript, suggestions } = hook.result.current;
    expect(transcript).toHaveLength(1);
    expect(transcript[0].speaker).toBe("Speaker A");
    expect(transcript[0].text).toBe("You never listen to me.");

    expect(suggestions).toHaveLength(3);
    expect(suggestions[0].text).toBe("I hear you.");
    // Tone reflects the empathy stance (75 -> empathetic), not a fabricated label.
    expect(suggestions[0].tone).toBe("empathetic");
  });

  it("surfaces transcription_unavailable instead of silently ignoring it", async () => {
    const { hook, ws } = await startLiveSession();

    await act(() =>
      ws.emitServer({
        type: "transcription_unavailable",
        reason: "DEEPGRAM_API_KEY not set",
      }),
    );

    expect(hook.result.current.transcriptionAvailable).toBe(false);
    expect(hook.result.current.transcriptionMessage).toContain("DEEPGRAM");
  });

  it("sends empathy changes as a config message the server understands", async () => {
    const { hook, ws } = await startLiveSession(50);

    await act(() => hook.result.current.sendEmpathyUpdate(90));

    const configMsgs = ws
      .sentJson()
      .filter((m) => m.type === "config" && m.empathy_slider === 90);
    expect(configMsgs).toHaveLength(1);
    // The old, server-rejected message types must not be sent.
    expect(ws.sentJson().some((m) => m.type === "empathy_update")).toBe(false);
  });

  it("sends initial empathy as config on connect", async () => {
    const { ws } = await startLiveSession(30);
    const configMsgs = ws
      .sentJson()
      .filter((m) => m.type === "config" && m.empathy_slider === 30);
    expect(configMsgs.length).toBeGreaterThanOrEqual(1);
  });
});

describe("useAudioStream — live PCM streaming", () => {
  it("batches captured PCM into ~100ms int16 binary frames (3200 bytes)", async () => {
    const { hook, ws } = await startLiveSession();
    expect(hook.result.current.isRecording).toBe(true);
    expect(mockMic.start).toHaveBeenCalledTimes(1);

    // 800 samples (50ms @ 16kHz) — below the 1600-sample frame threshold.
    await act(() => mockMic.onBuffer!(makePcmBuffer(800)));
    expect(ws.sentBinary).toHaveLength(0);

    // Another 800 samples completes one 1600-sample frame.
    await act(() => mockMic.onBuffer!(makePcmBuffer(800)));
    expect(ws.sentBinary).toHaveLength(1);
    expect(ws.sentBinary[0].byteLength).toBe(3200);

    // int16 little-endian content: 0.25 * 32767 = 8191.75 -> 8192.
    const samples = new Int16Array(ws.sentBinary[0]);
    expect(samples).toHaveLength(1600);
    expect(samples[0]).toBe(8192);
    expect(samples[1599]).toBe(8192);
  });

  it("downsamples to 16kHz when the hardware reports a different actual rate", async () => {
    const { ws } = await startLiveSession();

    // 4800 samples @ 48kHz = 100ms of audio -> 1600 samples @ 16kHz.
    await act(() =>
      mockMic.onBuffer!(makePcmBuffer(4800, { sampleRate: 48000 })),
    );
    expect(ws.sentBinary).toHaveLength(1);
    expect(ws.sentBinary[0].byteLength).toBe(3200);
  });

  it("permission denied: honest error state, no WebSocket, no audio sent", async () => {
    mockMic.requestPermissions.mockResolvedValue({
      status: "denied",
      granted: false,
    });

    const hook = await renderHook(() => useAudioStream());
    await act(async () => {
      await hook.result.current.startSession("sess-1", 50);
    });

    expect(hook.result.current.micError).toMatch(/permission denied/i);
    expect(hook.result.current.isRecording).toBe(false);
    expect(hook.result.current.connectionStatus).toBe("idle");
    // No session is opened and no audio can have been sent.
    expect(FakeWebSocket.instances).toHaveLength(0);
    expect(mockMic.start).not.toHaveBeenCalled();
  });

  it("mic start failure after connect closes the session cleanly", async () => {
    mockMic.start.mockRejectedValue(new Error("mic already in use"));

    const hook = await renderHook(() => useAudioStream());
    await act(async () => {
      await hook.result.current.startSession("sess-1", 50);
    });

    expect(hook.result.current.micError).toContain("mic already in use");
    expect(hook.result.current.isRecording).toBe(false);
    expect(hook.result.current.connectionStatus).toBe("idle");
    // The WebSocket that was opened must have been closed again.
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(FakeWebSocket.instances[0].readyState).not.toBe(FakeWebSocket.OPEN);
  });

  it("unavailable capture (e.g. web): the session still runs, with an honest banner and zero audio frames", async () => {
    mockMic.streamAvailable = false;

    const hook = await renderHook(() => useAudioStream());
    await act(async () => {
      await hook.result.current.startSession("sess-1", 50);
    });

    // The session WebSocket opens anyway — no capture backend must not cost
    // web users the live session (config/empathy/server events still flow).
    expect(FakeWebSocket.instances).toHaveLength(1);
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    expect(hook.result.current.connectionStatus).toBe("live");
    expect(ws.sentJson().filter((m) => m.type === "config")).toHaveLength(1);

    // Honest states: banner up, session active, but NOT recording.
    expect(hook.result.current.micError).toMatch(/not supported/i);
    expect(hook.result.current.isRecording).toBe(false);
    expect(hook.result.current.sessionActive).toBe(true);
    expect(mockMic.start).not.toHaveBeenCalled();
    expect(ws.sentBinary).toHaveLength(0);

    // The server's honest transcription_unavailable still surfaces.
    await act(() =>
      ws.emitServer({
        type: "transcription_unavailable",
        reason: "no audio received",
      }),
    );
    expect(hook.result.current.transcriptionAvailable).toBe(false);

    // And the audio-less session is stoppable via the same handshake.
    await act(async () => {
      await hook.result.current.stopSession();
    });
    expect(ws.sentJson().some((m) => m.type === "stop")).toBe(true);
    await act(() => ws.emitServer({ type: "session_complete" }));
    expect(hook.result.current.connectionStatus).toBe("idle");
    expect(hook.result.current.sessionActive).toBe(false);
  });

  it("double-tap start opens exactly ONE WebSocket and sends one config", async () => {
    const hook = await renderHook(() => useAudioStream());

    // Two rapid taps: the second lands while the first is still awaiting the
    // async permission/audio-mode/start chain (isRecording hasn't flipped).
    await act(async () => {
      await Promise.all([
        hook.result.current.startSession("sess-1", 50),
        hook.result.current.startSession("sess-1", 50),
      ]);
    });

    expect(FakeWebSocket.instances).toHaveLength(1);
    const ws = FakeWebSocket.instances[0];
    await act(() => ws.emitOpen());
    expect(ws.sentJson().filter((m) => m.type === "config")).toHaveLength(1);
    expect(mockMic.start).toHaveBeenCalledTimes(1);
    expect(hook.result.current.isRecording).toBe(true);

    // Starting again while a session is active is also a no-op.
    await act(async () => {
      await hook.result.current.startSession("sess-2", 50);
    });
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(mockMic.start).toHaveBeenCalledTimes(1);
  });

  it("rebinds the audio sender to the NEW socket after a reconnect", async () => {
    jest.useFakeTimers();
    try {
      const { ws } = await startLiveSession();

      await act(() => mockMic.onBuffer!(makePcmBuffer(1600)));
      expect(ws.sentBinary).toHaveLength(1);

      // Involuntary drop -> the hook schedules a reconnect.
      await act(() => ws.close());
      await act(() => {
        jest.advanceTimersByTime(2000);
      });

      const ws2 = FakeWebSocket.instances.at(-1)!;
      expect(ws2).not.toBe(ws);
      await act(() => ws2.emitOpen());

      await act(() => mockMic.onBuffer!(makePcmBuffer(1600)));
      // The frame lands on the NEW socket, not the stale one.
      expect(ws2.sentBinary).toHaveLength(1);
      expect(ws.sentBinary).toHaveLength(1);
    } finally {
      jest.useRealTimers();
    }
  });
});

/** A server suggestion event whose top suggestion is `top`. */
function suggestionEvent(top: string, rest: string[] = []) {
  return {
    type: "suggestion",
    session_id: "sess-1",
    utterance_text: "some utterance",
    speaker: "Speaker A",
    suggestions: [top, ...rest],
    empathy_slider: 50,
  };
}

describe("useAudioStream — on-device speech (expo-speech, free)", () => {
  it("earpiece mode: speaks the TOP suggestion exactly once", async () => {
    const { hook, ws } = await startLiveSession();
    await act(() => hook.result.current.setSpeechEnabled(true));

    await act(() =>
      ws.emitServer(suggestionEvent("I hear you.", ["Tell me more."])),
    );

    expect(speakMock).toHaveBeenCalledTimes(1);
    expect(speakMock.mock.calls[0][0]).toBe("I hear you.");
    expect(hook.result.current.speechEnabled).toBe(true);
    expect(hook.result.current.speechAvailable).toBe(true);
  });

  it("visual mode (default): suggestions render but are NOT spoken", async () => {
    const { hook, ws } = await startLiveSession();
    expect(hook.result.current.speechEnabled).toBe(false);

    await act(() => ws.emitServer(suggestionEvent("Stay silent about this.")));

    expect(hook.result.current.suggestions[0].text).toBe(
      "Stay silent about this.",
    );
    expect(speakMock).not.toHaveBeenCalled();
  });

  it("most-recent-wins: a new suggestion stops the current utterance and speaks the new one", async () => {
    const { hook, ws } = await startLiveSession();
    await act(() => hook.result.current.setSpeechEnabled(true));

    await act(() => ws.emitServer(suggestionEvent("First advice.")));
    expect(speakMock).toHaveBeenCalledTimes(1);

    // The mock never fires onDone — the first utterance is still "speaking"
    // when the second suggestion lands.
    await act(() => ws.emitServer(suggestionEvent("Newer advice.")));

    expect(speakMock).toHaveBeenCalledTimes(2);
    expect(speakMock.mock.calls[1][0]).toBe("Newer advice.");
    // stop() ran before the second speak() — interrupt, never queue.
    const stopOrders = speechStopMock.mock.invocationCallOrder;
    const speakOrders = speakMock.mock.invocationCallOrder;
    expect(Math.max(...stopOrders)).toBeLessThan(speakOrders[1]);
  });

  it("stopSession stops speech and nothing is spoken afterward (drain included)", async () => {
    const { hook, ws } = await startLiveSession();
    await act(() => hook.result.current.setSpeechEnabled(true));

    await act(() => ws.emitServer(suggestionEvent("Mid-session advice.")));
    expect(speakMock).toHaveBeenCalledTimes(1);
    speechStopMock.mockClear();

    await act(async () => {
      await hook.result.current.stopSession();
    });
    expect(speechStopMock).toHaveBeenCalled();

    // A late suggestion during the drain window still renders visually but
    // is NOT spoken — the user pressed stop.
    await act(() => ws.emitServer(suggestionEvent("Too late to say aloud.")));
    expect(hook.result.current.suggestions[0].text).toBe(
      "Too late to say aloud.",
    );
    expect(speakMock).toHaveBeenCalledTimes(1);
  });

  it("unmount stops any in-flight speech", async () => {
    const { hook, ws } = await startLiveSession();
    await act(() => hook.result.current.setSpeechEnabled(true));
    await act(() => ws.emitServer(suggestionEvent("Still talking...")));
    speechStopMock.mockClear();

    await act(() => hook.unmount());

    expect(speechStopMock).toHaveBeenCalled();
  });

  it("switching back to visual mid-utterance silences immediately", async () => {
    const { hook, ws } = await startLiveSession();
    await act(() => hook.result.current.setSpeechEnabled(true));
    await act(() => ws.emitServer(suggestionEvent("Long spoken advice...")));
    speechStopMock.mockClear();

    await act(() => hook.result.current.setSpeechEnabled(false));
    expect(speechStopMock).toHaveBeenCalled();

    await act(() => ws.emitServer(suggestionEvent("Visual-only now.")));
    expect(speakMock).toHaveBeenCalledTimes(1); // no new speech
  });

  it("degrades honestly when the platform has no TTS: no crash, visual suggestions intact", async () => {
    const warnSpy = jest.spyOn(console, "warn").mockImplementation(() => {});
    try {
      speakMock.mockImplementation(() => {
        throw new Error("Speech synthesis is unavailable on this platform");
      });

      const { hook, ws } = await startLiveSession();
      await act(() => hook.result.current.setSpeechEnabled(true));

      await act(() => ws.emitServer(suggestionEvent("Try saying this.")));

      // The failure is surfaced, not hidden — and the visual path still works.
      expect(hook.result.current.speechAvailable).toBe(false);
      expect(warnSpy).toHaveBeenCalledTimes(1);
      expect(hook.result.current.suggestions[0].text).toBe("Try saying this.");

      // Once known-unavailable, we stop attempting to speak at all.
      speakMock.mockClear();
      await act(() => ws.emitServer(suggestionEvent("Another idea.")));
      expect(speakMock).not.toHaveBeenCalled();
      expect(hook.result.current.suggestions[0].text).toBe("Another idea.");
      expect(warnSpy).toHaveBeenCalledTimes(1); // logged once, never spammed
    } finally {
      warnSpy.mockRestore();
    }
  });

  it("async TTS failure (onError) flips speechAvailable to false — the silent earpiece is never presented as working", async () => {
    const warnSpy = jest.spyOn(console, "warn").mockImplementation(() => {});
    try {
      // e.g. Android where detectSpeechSupport() is true but the TTS engine
      // has no installed voice data: speak() returns normally, then the
      // utterance fails asynchronously via its onError callback.
      speakMock.mockImplementation(
        (_text: string, opts?: { onError?: (error: Error) => void }) => {
          opts?.onError?.(new Error("TTS engine has no voice data"));
        },
      );

      const { hook, ws } = await startLiveSession();
      await act(() => hook.result.current.setSpeechEnabled(true));
      expect(hook.result.current.speechAvailable).toBe(true);

      await act(() => ws.emitServer(suggestionEvent("Say this aloud.")));

      // Nothing was actually spoken, so the hook must say so: the flag flips
      // and LiveCoachScreen's "spoken suggestions aren't available" note
      // renders (pinned by the LiveCoachScreen speechAvailable=false test).
      expect(hook.result.current.speechAvailable).toBe(false);
      expect(warnSpy).toHaveBeenCalledTimes(1);

      // Visual transcript + suggestions keep working untouched.
      expect(hook.result.current.suggestions[0].text).toBe("Say this aloud.");
      expect(hook.result.current.transcript.at(-1)?.text).toBe(
        "some utterance",
      );

      // Known-unavailable: no further speak attempts, no log spam.
      speakMock.mockClear();
      await act(() => ws.emitServer(suggestionEvent("Another idea.")));
      expect(speakMock).not.toHaveBeenCalled();
      expect(hook.result.current.suggestions[0].text).toBe("Another idea.");
      expect(warnSpy).toHaveBeenCalledTimes(1);
    } finally {
      warnSpy.mockRestore();
    }
  });
});

describe("useAudioStream — graceful stop handshake", () => {
  it("stop flushes the remainder, sends a stop message, and drains late suggestions before closing", async () => {
    const { hook, ws } = await startLiveSession();

    // A sub-frame remainder is pending when the user stops.
    await act(() => mockMic.onBuffer!(makePcmBuffer(400)));
    expect(ws.sentBinary).toHaveLength(0);

    await act(async () => {
      await hook.result.current.stopSession();
    });

    expect(mockMic.stop).toHaveBeenCalled();
    expect(hook.result.current.isRecording).toBe(false);
    // The 400-sample remainder (800 bytes) was flushed...
    expect(ws.sentBinary).toHaveLength(1);
    expect(ws.sentBinary[0].byteLength).toBe(800);
    // ...followed by the stop message — and the socket stays OPEN so the
    // server can deliver the final utterance's suggestion.
    expect(ws.sentJson().some((m) => m.type === "stop")).toBe(true);
    expect(ws.readyState).toBe(FakeWebSocket.OPEN);

    // Buffers delivered after stop are ignored — nothing new is sent.
    await act(() => mockMic.onBuffer!(makePcmBuffer(1600)));
    expect(ws.sentBinary).toHaveLength(1);

    // A suggestion arriving during the drain window IS applied.
    await act(() =>
      ws.emitServer({
        type: "suggestion",
        session_id: "sess-1",
        utterance_text: "one final thought",
        speaker: "Speaker A",
        suggestions: ["Thanks for talking this through."],
        empathy_slider: 50,
      }),
    );
    expect(hook.result.current.transcript.at(-1)?.text).toBe(
      "one final thought",
    );
    expect(hook.result.current.suggestions[0].text).toBe(
      "Thanks for talking this through.",
    );

    // session_complete ends the handshake: socket closed, status idle (the
    // deliberate close must never leave "disconnected" behind).
    await act(() => ws.emitServer({ type: "session_complete" }));
    expect(ws.readyState).not.toBe(FakeWebSocket.OPEN);
    expect(hook.result.current.connectionStatus).toBe("idle");
    expect(hook.result.current.sessionActive).toBe(false);
  });

  it("stop with no server response cleans up after the 4s drain timeout", async () => {
    jest.useFakeTimers();
    try {
      const { hook, ws } = await startLiveSession();

      await act(async () => {
        await hook.result.current.stopSession();
      });
      expect(ws.sentJson().some((m) => m.type === "stop")).toBe(true);
      expect(ws.readyState).toBe(FakeWebSocket.OPEN);
      expect(hook.result.current.isRecording).toBe(false);

      // Server never answers: the drain window times out.
      await act(() => {
        jest.advanceTimersByTime(4000);
      });
      expect(ws.readyState).not.toBe(FakeWebSocket.OPEN);
      expect(hook.result.current.connectionStatus).toBe("idle");
      expect(hook.result.current.sessionActive).toBe(false);
    } finally {
      jest.useRealTimers();
    }
  });

  it("drain window resets on server activity: a slow (e.g. Whisper) final suggestion at t=6s still lands", async () => {
    jest.useFakeTimers();
    try {
      const { hook, ws } = await startLiveSession();

      await act(async () => {
        await hook.result.current.stopSession();
      });
      expect(ws.readyState).toBe(FakeWebSocket.OPEN);

      // t=3.5s: any frame from the server (here a config_ack control frame)
      // proves it is alive and still finalizing — the window must reset.
      await act(() => {
        jest.advanceTimersByTime(3500);
      });
      await act(() => ws.emitServer({ type: "config_ack" }));

      // t=6s: the OLD fixed 4s deadline would have killed the socket at t=4s.
      // With the reset the drain is still open, waiting for the server.
      await act(() => {
        jest.advanceTimersByTime(2500);
      });
      expect(ws.readyState).toBe(FakeWebSocket.OPEN);
      expect(hook.result.current.connectionStatus).toBe("live");

      // The slow-transcription final suggestion + session_complete arrive
      // at t=6s and must land in state before the session ends at "idle".
      await act(() =>
        ws.emitServer({
          type: "suggestion",
          session_id: "sess-1",
          utterance_text: "one last thing",
          speaker: "Speaker A",
          suggestions: ["Closing advice."],
          empathy_slider: 50,
        }),
      );
      expect(hook.result.current.transcript.at(-1)?.text).toBe(
        "one last thing",
      );
      expect(hook.result.current.suggestions[0].text).toBe("Closing advice.");

      await act(() => ws.emitServer({ type: "session_complete" }));
      expect(ws.readyState).not.toBe(FakeWebSocket.OPEN);
      expect(hook.result.current.connectionStatus).toBe("idle");
      expect(hook.result.current.sessionActive).toBe(false);
    } finally {
      jest.useRealTimers();
    }
  });

  it("a server that keeps sending but never completes is cut off at the absolute drain cap", async () => {
    jest.useFakeTimers();
    try {
      const { hook, ws } = await startLiveSession();

      await act(async () => {
        await hook.result.current.stopSession();
      });

      // The server emits a frame every 3s (each one inside the 4s inactivity
      // window, so the drain keeps extending) but never session_complete.
      for (let t = 3000; t <= 12000; t += 3000) {
        await act(() => {
          jest.advanceTimersByTime(3000);
        });
        expect(ws.readyState).toBe(FakeWebSocket.OPEN); // still draining
        await act(() => ws.emitServer({ type: "config_ack" }));
      }

      // The window re-armed at t=12s would run to t=16s, but the absolute
      // 15s cap wins: the stop can never hang forever, and it ends at idle.
      await act(() => {
        jest.advanceTimersByTime(3000);
      });
      expect(ws.readyState).not.toBe(FakeWebSocket.OPEN);
      expect(hook.result.current.connectionStatus).toBe("idle");
      expect(hook.result.current.sessionActive).toBe(false);
    } finally {
      jest.useRealTimers();
    }
  });

  it("unmount during the drain clears the drain timer — no leaks, no setState after unmount", async () => {
    jest.useFakeTimers();
    const setTimeoutSpy = jest.spyOn(global, "setTimeout");
    const clearTimeoutSpy = jest.spyOn(global, "clearTimeout");
    try {
      const { hook, ws } = await startLiveSession();

      await act(async () => {
        await hook.result.current.stopSession();
      });
      expect(ws.readyState).toBe(FakeWebSocket.OPEN); // draining

      // The drain timer is the only 4000ms timer armed here (reconnects use
      // 2000ms; the rest are React Native internals) — pin its exact id.
      const drainCalls = setTimeoutSpy.mock.calls
        .map((call, i) => ({ delay: call[1], i }))
        .filter(({ delay }) => delay === 4000);
      expect(drainCalls).toHaveLength(1);
      const drainTimerId = setTimeoutSpy.mock.results[drainCalls[0].i].value;

      await act(() => hook.unmount());

      // Unmount cleared that exact timer, so finishDrain can never fire and
      // call setState on the unmounted hook.
      expect(clearTimeoutSpy.mock.calls.some(([id]) => id === drainTimerId)).toBe(
        true,
      );
      await act(() => {
        jest.advanceTimersByTime(20000); // belt-and-braces: nothing throws
      });
    } finally {
      setTimeoutSpy.mockRestore();
      clearTimeoutSpy.mockRestore();
      jest.useRealTimers();
    }
  });

  it("stop while disconnected skips the handshake and cleans up immediately", async () => {
    jest.useFakeTimers();
    try {
      const { hook, ws } = await startLiveSession();

      // Involuntary drop: the hook schedules a reconnect.
      await act(() => ws.close());
      expect(hook.result.current.connectionStatus).toBe("disconnected");

      await act(async () => {
        await hook.result.current.stopSession();
      });

      // No stop message on a dead socket — immediate cleanup, ending idle.
      expect(ws.sentJson().some((m) => m.type === "stop")).toBe(false);
      expect(hook.result.current.connectionStatus).toBe("idle");
      expect(hook.result.current.isRecording).toBe(false);
      expect(hook.result.current.sessionActive).toBe(false);

      // The pending reconnect must not resurrect the session.
      await act(() => {
        jest.advanceTimersByTime(10000);
      });
      expect(FakeWebSocket.instances).toHaveLength(1);
      expect(hook.result.current.connectionStatus).toBe("idle");
    } finally {
      jest.useRealTimers();
    }
  });

  it("server closing the socket during the drain also ends the session at idle", async () => {
    const { hook, ws } = await startLiveSession();

    await act(async () => {
      await hook.result.current.stopSession();
    });
    expect(ws.readyState).toBe(FakeWebSocket.OPEN);

    // Server skips session_complete and just closes (e.g. code 1000).
    await act(() => ws.close());

    expect(hook.result.current.connectionStatus).toBe("idle");
    expect(hook.result.current.sessionActive).toBe(false);
    // No reconnect attempt: a manual stop is final.
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("starting a new session during a drain finishes the old one and connects fresh", async () => {
    const { hook, ws } = await startLiveSession();

    await act(async () => {
      await hook.result.current.stopSession();
    });
    expect(ws.readyState).toBe(FakeWebSocket.OPEN); // draining

    await act(async () => {
      await hook.result.current.startSession("sess-2", 60);
    });

    // Old socket was closed, a new one opened for the new session.
    expect(FakeWebSocket.instances).toHaveLength(2);
    expect(ws.readyState).not.toBe(FakeWebSocket.OPEN);
    const ws2 = FakeWebSocket.instances.at(-1)!;
    expect(ws2.url).toContain("sess-2");
    await act(() => ws2.emitOpen());
    expect(hook.result.current.connectionStatus).toBe("live");
    expect(hook.result.current.isRecording).toBe(true);
  });
});
