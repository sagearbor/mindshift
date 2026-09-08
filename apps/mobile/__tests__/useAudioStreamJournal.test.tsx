import { renderHook, act } from "@testing-library/react-native";

/**
 * useAudioStream in JOURNAL mode: the mic frames go to the injected journal
 * recorder and NOWHERE else — no WebSocket is opened, no fast loop is
 * built, no kept-audio keeper is opened — and the honest gates (no owner
 * voiceprint, models unavailable) open no session.
 */
const mockMic = {
  onBuffer: null as
    | ((buffer: { data: ArrayBuffer; sampleRate: number; channels: number; timestamp: number }) => void)
    | null,
  start: jest.fn<Promise<void>, []>(),
  stop: jest.fn(),
  requestPermissions: jest.fn<Promise<{ status: string; granted: boolean }>, []>(),
  setAudioMode: jest.fn<Promise<void>, [unknown]>(),
};

jest.mock("expo-audio", () => ({
  __esModule: true,
  requestRecordingPermissionsAsync: () => mockMic.requestPermissions(),
  requestNotificationPermissionsAsync: jest.fn().mockResolvedValue({ granted: true }),
  setAudioModeAsync: (mode: unknown) => mockMic.setAudioMode(mode),
  AudioModule: { AudioRecorder: jest.fn() },
  useAudioStream: (options?: { onBuffer?: (buffer: never) => void }) => {
    mockMic.onBuffer = (options?.onBuffer ?? null) as typeof mockMic.onBuffer;
    return {
      stream: { id: "mock-stream", sampleRate: 16000, channels: 1, isStreaming: false, start: mockMic.start, stop: mockMic.stop },
      isStreaming: false,
    };
  },
}));

import { useAudioStream } from "../src/hooks/useAudioStream";
import type { JournalRecorder, JournalState } from "../src/live/journalRecorder";
import { IDLE_JOURNAL_STATE } from "../src/live/journalRecorder";
import type { FastLoopHandlers } from "../src/live/defaultDeps";

class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];
  readyState = FakeWebSocket.OPEN;
  onopen: ((e: unknown) => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;
  onclose: ((e: unknown) => void) | null = null;
  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }
  send() {}
  close() {}
}

function fakeRecorder(opts: { hasSelfPrint?: boolean } = {}) {
  const pushed: Int16Array[] = [];
  let onState: ((s: JournalState) => void) | null = null;
  let state: JournalState = { ...IDLE_JOURNAL_STATE };
  const rec = {
    hasSelfPrint: opts.hasSelfPrint ?? true,
    pushed,
    stopped: 0,
    retried: 0,
    get stateSnapshot() {
      return state;
    },
    async start() {
      state = { ...state, status: "listening", startedAt: 1 };
      onState?.(state);
    },
    pushSamples(s: Int16Array) {
      pushed.push(s.slice());
      state = { ...state, listeningSeconds: state.listeningSeconds + s.length / 16000, selfCount: 1, lastSelfAt: 2 };
      onState?.(state);
    },
    async stop() {
      rec.stopped += 1;
      state = { ...state, status: "stopped", filesClosed: 1, uploads: { ...state.uploads, sent: 1 } };
      onState?.(state);
      return state;
    },
    async retryUploads() {
      rec.retried += 1;
    },
    bind(handlers: { onState?: (s: JournalState) => void }) {
      onState = handlers.onState ?? null;
    },
  };
  return rec;
}

function feed(seconds: number) {
  const n = Math.round(seconds * 16000);
  for (let off = 0; off < n; off += 1600) {
    const f32 = new Float32Array(Math.min(1600, n - off)).fill(0.2);
    mockMic.onBuffer?.({ data: f32.buffer, sampleRate: 16000, channels: 1, timestamp: 0 });
  }
}

let logSpy: jest.SpyInstance;

beforeEach(() => {
  FakeWebSocket.instances = [];
  (globalThis as Record<string, unknown>).WebSocket = FakeWebSocket;
  mockMic.onBuffer = null;
  mockMic.start.mockReset().mockResolvedValue(undefined);
  mockMic.stop.mockReset();
  mockMic.requestPermissions.mockReset().mockResolvedValue({ status: "granted", granted: true });
  mockMic.setAudioMode.mockReset().mockResolvedValue(undefined);
  logSpy = jest.spyOn(console, "log").mockImplementation(() => {});
});

afterEach(() => {
  logSpy.mockRestore();
});

describe("useAudioStream journal mode", () => {
  it("runs mic → journal only: no WebSocket, no fast loop, no keeper; Stop closes cleanly", async () => {
    const rec = fakeRecorder();
    const makeFastLoop = jest.fn<Promise<never>, [FastLoopHandlers]>();
    const openAudioKeeper = jest.fn();
    const prepareJournalAudio = jest.fn().mockResolvedValue({ notificationsGranted: true });
    const hold = { release: jest.fn().mockResolvedValue(undefined) };
    const holdBackgroundMic = jest.fn().mockResolvedValue(hold);
    const hook = await renderHook(() =>
      useAudioStream({
        capability: { capable: true, reason: "ok" },
        makeFastLoop: makeFastLoop as never,
        openAudioKeeper: openAudioKeeper as never,
        makeJournal: async (handlers) => {
          rec.bind(handlers);
          return rec as unknown as JournalRecorder;
        },
        prepareJournalAudio,
        holdBackgroundMic,
      }),
    );
    await act(() => {
      hook.result.current.setSessionMode("journal");
    });
    expect(hook.result.current.journal).toEqual(IDLE_JOURNAL_STATE);

    await act(async () => {
      await hook.result.current.startSession("journal-1", 50);
    });
    await act(async () => {
      await Promise.resolve();
    });
    expect(hook.result.current.sessionActive).toBe(true);
    expect(hook.result.current.isRecording).toBe(true);
    expect(hook.result.current.journal.status).toBe("listening");
    expect(FakeWebSocket.instances).toHaveLength(0);
    expect(makeFastLoop).not.toHaveBeenCalled();
    expect(openAudioKeeper).not.toHaveBeenCalled();
    expect(mockMic.start).toHaveBeenCalledTimes(1);
    expect(prepareJournalAudio).toHaveBeenCalledTimes(1);
    expect(holdBackgroundMic).toHaveBeenCalledTimes(1);
    expect(hook.result.current.connectionStatus).toBe("idle");

    await act(() => {
      feed(0.5);
    });
    expect(rec.pushed).toHaveLength(5);
    expect(rec.pushed[0]).toBeInstanceOf(Int16Array);
    expect(rec.pushed[0][0]).toBe(Math.round(0.2 * 32767));
    expect(hook.result.current.journal.selfCount).toBe(1);
    expect(hook.result.current.journal.listeningSeconds).toBeCloseTo(0.5, 3);

    await act(async () => {
      await hook.result.current.stopSession();
    });
    expect(rec.stopped).toBe(1);
    expect(mockMic.stop).toHaveBeenCalled();
    expect(hold.release).toHaveBeenCalledTimes(1);
    expect(hook.result.current.sessionActive).toBe(false);
    expect(hook.result.current.isRecording).toBe(false);
    expect(hook.result.current.journal.status).toBe("stopped");
    expect(hook.result.current.journal.uploads.sent).toBe(1);
    // Frames after Stop go nowhere.
    await act(() => {
      feed(0.1);
    });
    expect(rec.pushed).toHaveLength(5);
    // The playback audio mode is restored for the rest of the app.
    expect(mockMic.setAudioMode).toHaveBeenLastCalledWith(expect.objectContaining({ allowsRecording: false }));

    await act(async () => {
      await hook.result.current.retryJournalUploads();
    });
    expect(rec.retried).toBe(1);
    expect(FakeWebSocket.instances).toHaveLength(0);
  });

  it("refuses to start without an enrolled owner voiceprint — and says so", async () => {
    const rec = fakeRecorder({ hasSelfPrint: false });
    const hook = await renderHook(() =>
      useAudioStream({
        capability: { capable: true, reason: "ok" },
        makeJournal: async (handlers) => {
          rec.bind(handlers);
          return rec as unknown as JournalRecorder;
        },
        prepareJournalAudio: async () => ({}),
        holdBackgroundMic: async () => null,
      }),
    );
    await act(() => {
      hook.result.current.setSessionMode("journal");
    });
    await act(async () => {
      await hook.result.current.startSession("journal-2", 50);
    });
    expect(hook.result.current.sessionActive).toBe(false);
    expect(hook.result.current.journal.status).toBe("idle");
    expect(hook.result.current.journal.error).toMatch(/Enroll your voice first/);
    expect(mockMic.start).not.toHaveBeenCalled();
    expect(FakeWebSocket.instances).toHaveLength(0);
    // A later normal session still works (the journal branch left nothing behind).
    await act(() => {
      hook.result.current.setSessionMode("earpiece");
      hook.result.current.setLiveMode(false);
    });
    await act(async () => {
      await hook.result.current.startSession("live-3", 50);
    });
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(hook.result.current.journal).toEqual(IDLE_JOURNAL_STATE);
    await act(async () => {
      await hook.result.current.stopSession();
    });
  });

  it("surfaces an unavailable recorder (models / store) as the journal error", async () => {
    const hook = await renderHook(() =>
      useAudioStream({
        capability: { capable: true, reason: "ok" },
        makeJournal: async () => {
          throw new Error("speaker-ID unavailable (voice model not downloaded)");
        },
        prepareJournalAudio: async () => ({}),
        holdBackgroundMic: async () => null,
      }),
    );
    await act(() => {
      hook.result.current.setSessionMode("journal");
    });
    await act(async () => {
      await hook.result.current.startSession("journal-3", 50);
    });
    expect(hook.result.current.sessionActive).toBe(false);
    expect(hook.result.current.journal.error).toBe(
      "Journal unavailable: speaker-ID unavailable (voice model not downloaded)",
    );
    expect(mockMic.start).not.toHaveBeenCalled();
  });

  it("stops the journal on unmount (file closed, uploads kicked off)", async () => {
    const rec = fakeRecorder();
    const hook = await renderHook(() =>
      useAudioStream({
        capability: { capable: true, reason: "ok" },
        makeJournal: async (handlers) => {
          rec.bind(handlers);
          return rec as unknown as JournalRecorder;
        },
        prepareJournalAudio: async () => ({}),
        holdBackgroundMic: async () => null,
      }),
    );
    await act(() => {
      hook.result.current.setSessionMode("journal");
    });
    await act(async () => {
      await hook.result.current.startSession("journal-4", 50);
    });
    expect(hook.result.current.sessionActive).toBe(true);
    hook.unmount();
    await act(async () => {
      await Promise.resolve();
    });
    expect(rec.stopped).toBe(1);
    expect(mockMic.stop).toHaveBeenCalled();
  });
});
