/**
 * AudioRecordScreen driving the v2 gapless STREAM engine — plus the engine
 * selection rules that keep v1 available as an escape hatch:
 *
 * - deps carrying a PcmSource factory (production wiring, stream-capable
 *   test doubles) → v2 stream engine, gapless disclosure;
 * - deps with engine: "recorder" or with no PcmSource factory (every
 *   pre-existing v1 test and any device where the stream API fails) → v1
 *   segmented engine, unchanged behavior and honest 0.2 s-gap disclosure.
 */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import {
  getRecordingPermissionsAsync,
  requestRecordingPermissionsAsync,
} from "expo-audio";
import AudioRecordScreen from "../src/recorder/AudioRecordScreen";
import type { AudioRecorderDeps } from "../src/recorder/AudioRecordScreen";
import { MemoryFs } from "../src/recorder/memoryFs";
import { RecorderSessionStore } from "../src/recorder/sessionStore";
import type { PcmFrame, PcmSource } from "../src/recorder/pcmSource";
import type { RecordedAudioFile, RecorderPort } from "../src/recorder/types";

const RATE = 1000;
const SEGMENT_MS = 4000;
const FLUSH_MS = 2000;

class FakePcmSource implements PcmSource {
  onFrame: ((frame: PcmFrame) => void) | null = null;
  capturing = false;
  async start(onFrame: (frame: PcmFrame) => void): Promise<void> {
    this.onFrame = onFrame;
    this.capturing = true;
  }
  stop(): void {
    this.capturing = false;
  }
  isCapturing(): boolean {
    return this.capturing;
  }
  push(count: number, start = 0): void {
    const samples = new Int16Array(count);
    for (let i = 0; i < count; i++) samples[i] = (start + i) % 30000;
    this.onFrame?.({ samples, sampleRate: RATE });
  }
  kill(): void {
    this.capturing = false;
  }
}

/** Minimal v1 recorder fake for the escape-hatch tests. */
class FakeRecorder implements RecorderPort {
  uri: string | null = null;
  recording = false;
  constructor(
    private fs: MemoryFs,
    readonly tag: number,
  ) {}
  async prepare(): Promise<void> {
    this.uri = `file:///cache/v1-rec-${this.tag}.wav`;
  }
  start(): void {
    this.recording = true;
  }
  async stop(): Promise<{ uri: string | null }> {
    this.recording = false;
    if (this.uri) {
      // A tiny but valid one-sample wav.
      const buf = new ArrayBuffer(46);
      const v = new DataView(buf);
      const w = (o: number, s: string) => {
        for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
      };
      w(0, "RIFF");
      v.setUint32(4, 38, true);
      w(8, "WAVE");
      w(12, "fmt ");
      v.setUint32(16, 16, true);
      v.setUint16(20, 1, true);
      v.setUint16(22, 1, true);
      v.setUint32(24, 16000, true);
      v.setUint32(28, 32000, true);
      v.setUint16(32, 2, true);
      v.setUint16(34, 16, true);
      w(36, "data");
      v.setUint32(40, 2, true);
      v.setInt16(44, this.tag, true);
      this.fs.writeBytes(this.uri, new Uint8Array(buf));
    }
    return { uri: this.uri };
  }
  isRecording(): boolean {
    return this.recording;
  }
  release(): void {}
}

function makeStreamDeps() {
  const fs = new MemoryFs();
  const store = new RecorderSessionStore(fs);
  const sources: FakePcmSource[] = [];
  const deps: AudioRecorderDeps = {
    store,
    // v1 factory present but must NOT be used when a PcmSource exists.
    makeRecorder: () => {
      throw new Error("v1 recorder must not be constructed for stream deps");
    },
    makePcmSource: () => {
      const s = new FakePcmSource();
      sources.push(s);
      return s;
    },
    format: {
      format: "wav",
      extension: ".wav",
      mimeType: "audio/wav",
      bytesPerSecond: 32100,
    },
    getBatteryLevel: async () => 0.9,
    configureAudioSession: jest.fn().mockResolvedValue(undefined),
    segmentMs: SEGMENT_MS,
    flushMs: FLUSH_MS,
    resumeRetryMs: 2000,
    stallMs: 60_000,
  };
  return {
    fs,
    store,
    sources,
    deps,
    get source() {
      return sources[sources.length - 1];
    },
  };
}

function makeV1Deps(overrides: Partial<AudioRecorderDeps> = {}) {
  const fs = new MemoryFs();
  const store = new RecorderSessionStore(fs);
  const recorders: FakeRecorder[] = [];
  const deps: AudioRecorderDeps = {
    store,
    makeRecorder: () => {
      const r = new FakeRecorder(fs, recorders.length + 1);
      recorders.push(r);
      return r;
    },
    format: {
      format: "wav",
      extension: ".wav",
      mimeType: "audio/wav",
      bytesPerSecond: 32000,
    },
    getBatteryLevel: async () => 0.9,
    configureAudioSession: jest.fn().mockResolvedValue(undefined),
    segmentMs: SEGMENT_MS,
    resumeRetryMs: 2000,
    ...overrides,
  };
  return { fs, store, recorders, deps };
}

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

function textOf(node: ReactTestInstance | null): string {
  if (!node) return "";
  const parts: string[] = [];
  const walk = (children: unknown) => {
    if (Array.isArray(children)) {
      children.forEach(walk);
    } else if (typeof children === "string") {
      parts.push(children);
    } else if (children && typeof children === "object") {
      walk((children as { props?: { children?: unknown } }).props?.children);
    }
  };
  walk(node.props.children);
  return parts.join("");
}

async function mount(deps: AudioRecorderDeps, onComplete = jest.fn()) {
  const onBack = jest.fn();
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(
      <AudioRecordScreen onBack={onBack} onComplete={onComplete} deps={deps} />,
    );
  });
  return { comp, onBack, onComplete };
}

async function pressStart(comp: renderer.ReactTestRenderer) {
  await act(async () => {
    queryId(comp, "start-audio-recording")!.props.onPress();
  });
}

async function tickSeconds(comp: renderer.ReactTestRenderer, seconds: number) {
  for (let i = 0; i < seconds; i++) {
    await act(async () => {
      jest.advanceTimersByTime(1000);
    });
  }
}

beforeEach(() => {
  jest.useFakeTimers();
  (getRecordingPermissionsAsync as jest.Mock).mockResolvedValue({
    status: "granted",
    granted: true,
  });
  (requestRecordingPermissionsAsync as jest.Mock).mockResolvedValue({
    status: "granted",
    granted: true,
  });
});

afterEach(() => {
  jest.useRealTimers();
});

describe("AudioRecordScreen — engine selection", () => {
  it("uses the STREAM engine when deps provide a PcmSource factory", async () => {
    const h = makeStreamDeps();
    const { comp } = await mount(h.deps);
    await pressStart(comp);
    // The stream opened; the v1 recorder factory (which throws) was never hit.
    expect(h.sources).toHaveLength(1);
    expect(h.source.isCapturing()).toBe(true);
  });

  it("falls back to the v1 engine when deps carry no PcmSource factory", async () => {
    const h = makeV1Deps();
    const { comp } = await mount(h.deps);
    await pressStart(comp);
    expect(h.recorders).toHaveLength(1);
    expect(h.recorders[0].isRecording()).toBe(true);
  });

  it('honors an explicit engine: "recorder" escape hatch even with a PcmSource available', async () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs);
    const recorders: FakeRecorder[] = [];
    const sources: FakePcmSource[] = [];
    const deps: AudioRecorderDeps = {
      store,
      engine: "recorder",
      makeRecorder: () => {
        const r = new FakeRecorder(fs, recorders.length + 1);
        recorders.push(r);
        return r;
      },
      makePcmSource: () => {
        const s = new FakePcmSource();
        sources.push(s);
        return s;
      },
      format: {
        format: "wav",
        extension: ".wav",
        mimeType: "audio/wav",
        bytesPerSecond: 32000,
      },
      getBatteryLevel: async () => 0.9,
      configureAudioSession: jest.fn().mockResolvedValue(undefined),
      segmentMs: SEGMENT_MS,
      resumeRetryMs: 2000,
    };
    const { comp } = await mount(deps);
    await pressStart(comp);
    expect(recorders).toHaveLength(1);
    expect(sources).toHaveLength(0);
  });
});

describe("AudioRecordScreen — honest disclosure per engine", () => {
  it("stream engine: promises gapless capture and a seconds-level loss bound — no 0.2s-gap notice", async () => {
    const h = makeStreamDeps();
    const { comp } = await mount(h.deps);
    const line = textOf(queryId(comp, "gap-disclosure"));
    expect(line).toMatch(/gapless/i);
    expect(line).toMatch(/every few seconds/i);
    expect(line).not.toMatch(/0\.2/);
  });

  it("v1 engine: keeps the honest per-rotation gap disclosure", async () => {
    const h = makeV1Deps();
    const { comp } = await mount(h.deps);
    const line = textOf(queryId(comp, "gap-disclosure"));
    expect(line).toMatch(/0\.2/);
    expect(line).toMatch(/5 min/);
  });
});

describe("AudioRecordScreen — stream recording flow", () => {
  it("records, shows crash-safe progress after a flush, and hands one WAV to onComplete", async () => {
    const h = makeStreamDeps();
    const onComplete = jest.fn();
    const { comp } = await mount(h.deps, onComplete);
    await pressStart(comp);

    // ~3 s of audio, frames arriving between ticks.
    for (let s = 0; s < 3; s++) {
      h.source.push(RATE, s * RATE);
      await tickSeconds(comp, 1);
    }
    // The 2 s flush cadence has fired: audio is reported crash-safe.
    expect(textOf(queryId(comp, "segment-status"))).toMatch(/1/);
    expect(h.store.listRecoverable()[0]?.segmentCount).toBeGreaterThanOrEqual(1);

    await act(async () => {
      queryId(comp, "stop-audio-recording")!.props.onPress();
    });
    expect(onComplete).toHaveBeenCalledTimes(1);
    const file = onComplete.mock.calls[0][0] as RecordedAudioFile;
    expect(file.mimeType).toBe("audio/wav");
    expect(file.name).toMatch(/\.wav$/);
    expect(h.fs.exists(file.uri)).toBe(true);
    // Every pushed sample made it: 3 s at 1 kHz = 3000 samples = 6000 bytes.
    const bytes = h.fs.readBytes(file.uri);
    expect(bytes.byteLength).toBe(44 + 3000 * 2);
    // No recovery residue after a clean finish.
    expect(h.store.listRecoverable()).toEqual([]);
  });

  it("shows the interruption banner when the stream dies, then auto-resumes", async () => {
    const h = makeStreamDeps();
    const { comp } = await mount(h.deps);
    await pressStart(comp);
    h.source.push(RATE);
    await tickSeconds(comp, 1);
    h.source.kill();
    await tickSeconds(comp, 1);
    expect(queryId(comp, "interruption-banner")).not.toBeNull();
    // resumeRetryMs is 2 s — the next ticks bring a fresh stream back.
    await tickSeconds(comp, 3);
    expect(queryId(comp, "interruption-banner")).toBeNull();
    expect(h.sources).toHaveLength(2);
    expect(h.source.isCapturing()).toBe(true);
  });

  it("everything captured before an interruption is already crash-safe", async () => {
    const h = makeStreamDeps();
    const { comp } = await mount(h.deps);
    await pressStart(comp);
    h.source.push(RATE);
    await tickSeconds(comp, 1);
    h.source.kill();
    await tickSeconds(comp, 1);
    // Unlike v1 (which could lose the whole in-flight segment), the stream
    // engine salvages every captured sample at the moment of interruption.
    expect(h.store.listRecoverable()[0]?.totalDurationMs).toBe(1000);
  });
});
