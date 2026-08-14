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
import type { RecordedAudioFile, RecorderPort } from "../src/recorder/types";

const SEGMENT_MS = 4000;

function wavBytes(tag: number): Uint8Array {
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
  v.setInt16(44, tag, true);
  return new Uint8Array(buf);
}

class FakeRecorder implements RecorderPort {
  uri: string | null = null;
  recording = false;
  constructor(
    private fs: MemoryFs,
    readonly tag: number,
  ) {}
  async prepare(): Promise<void> {
    this.uri = `file:///cache/screen-rec-${this.tag}.wav`;
  }
  start(): void {
    this.recording = true;
  }
  async stop(): Promise<{ uri: string | null }> {
    this.recording = false;
    if (this.uri) this.fs.writeBytes(this.uri, wavBytes(this.tag));
    return { uri: this.uri };
  }
  isRecording(): boolean {
    return this.recording;
  }
  release(): void {}
  kill(): void {
    this.recording = false;
  }
}

function makeDeps(
  overrides: {
    freeBytes?: number | null;
    batteryLevel?: number | null;
    failResume?: boolean;
  } = {},
) {
  const fs = new MemoryFs({
    freeBytes: overrides.freeBytes === undefined ? 50e9 : overrides.freeBytes,
  });
  const store = new RecorderSessionStore(fs);
  const recorders: FakeRecorder[] = [];
  const deps: AudioRecorderDeps = {
    store,
    makeRecorder: () => {
      if (overrides.failResume && recorders.length >= 1) {
        throw new Error("mic busy");
      }
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
    getBatteryLevel: async () =>
      overrides.batteryLevel === undefined ? 0.9 : overrides.batteryLevel,
    configureAudioSession: jest.fn().mockResolvedValue(undefined),
    segmentMs: SEGMENT_MS,
    resumeRetryMs: 2000,
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

describe("AudioRecordScreen — honesty & preflight", () => {
  it("discloses the segment-gap tradeoff up front", async () => {
    const { deps } = makeDeps();
    const { comp } = await mount(deps);
    const line = queryId(comp, "gap-disclosure");
    expect(line).not.toBeNull();
    expect(textOf(line)).toMatch(/5 min/);
  });

  it("warns when the battery is below 30%", async () => {
    const { deps } = makeDeps({ batteryLevel: 0.2 });
    const { comp } = await mount(deps);
    expect(textOf(queryId(comp, "battery-warning"))).toContain("20");
  });

  it("shows no battery warning when the level is healthy", async () => {
    const { deps } = makeDeps({ batteryLevel: 0.8 });
    const { comp } = await mount(deps);
    expect(queryId(comp, "battery-warning")).toBeNull();
  });

  it("blocks starting when storage can't hold even a short session", async () => {
    const { deps } = makeDeps({ freeBytes: 1024 });
    const { comp } = await mount(deps);
    expect(textOf(queryId(comp, "storage-warning"))).toMatch(/storage/i);
    expect(queryId(comp, "start-audio-recording")!.props.disabled).toBe(true);
  });

  it("warns without blocking when storage is merely tight", async () => {
    // Fits ~30 min at 32 KB/s, planned session is longer.
    const { deps } = makeDeps({ freeBytes: 32000 * 30 * 60 });
    const { comp } = await mount(deps);
    expect(textOf(queryId(comp, "storage-warning"))).toContain("30");
    expect(queryId(comp, "start-audio-recording")!.props.disabled).toBe(false);
  });

  it("gates on microphone permission with a grant retry", async () => {
    (getRecordingPermissionsAsync as jest.Mock).mockResolvedValue({
      status: "denied",
      granted: false,
    });
    (requestRecordingPermissionsAsync as jest.Mock).mockResolvedValue({
      status: "denied",
      granted: false,
    });
    const { deps } = makeDeps();
    const { comp } = await mount(deps);
    expect(queryId(comp, "audio-permission-gate")).not.toBeNull();
    expect(queryId(comp, "start-audio-recording")).toBeNull();
  });
});

describe("AudioRecordScreen — recording", () => {
  it("start begins a session and shows the live timer", async () => {
    const { deps, recorders } = makeDeps();
    const { comp } = await mount(deps);
    await pressStart(comp);
    expect(recorders).toHaveLength(1);
    expect(recorders[0].isRecording()).toBe(true);
    expect(queryId(comp, "audio-timer")).not.toBeNull();
  });

  it("shows how much audio is already crash-safe after a rotation", async () => {
    const { deps, store } = makeDeps();
    const { comp } = await mount(deps);
    await pressStart(comp);
    await tickSeconds(comp, 5); // past the 4s test segment interval
    expect(textOf(queryId(comp, "segment-status"))).toMatch(/1/);
    expect(store.listRecoverable()[0]?.segmentCount).toBe(1);
  });

  it("stop hands one stitched AUDIO file (not video) to onComplete", async () => {
    const { deps, fs } = makeDeps();
    const onComplete = jest.fn();
    const { comp } = await mount(deps, onComplete);
    await pressStart(comp);
    await tickSeconds(comp, 5);
    await act(async () => {
      queryId(comp, "stop-audio-recording")!.props.onPress();
    });
    expect(onComplete).toHaveBeenCalledTimes(1);
    const file = onComplete.mock.calls[0][0] as RecordedAudioFile;
    expect(file.mimeType).toBe("audio/wav");
    expect(file.name).toMatch(/\.wav$/);
    expect(fs.exists(file.uri)).toBe(true);
    expect(file.size).toBeGreaterThan(0);
  });

  it("shows an interruption banner when the OS kills the mic, then auto-resumes", async () => {
    const { deps, recorders } = makeDeps();
    const { comp } = await mount(deps);
    await pressStart(comp);
    await tickSeconds(comp, 1);
    recorders[0].kill();
    await tickSeconds(comp, 1);
    expect(queryId(comp, "interruption-banner")).not.toBeNull();
    // resumeRetryMs is 2s — the next ticks bring a fresh recorder back.
    await tickSeconds(comp, 3);
    expect(queryId(comp, "interruption-banner")).toBeNull();
    expect(recorders).toHaveLength(2);
    expect(recorders[1].isRecording()).toBe(true);
  });

  it("stays honestly interrupted when resume keeps failing — and stop still saves what exists", async () => {
    const { deps, recorders, fs } = makeDeps({ failResume: true });
    const onComplete = jest.fn();
    const { comp } = await mount(deps, onComplete);
    await pressStart(comp);
    await tickSeconds(comp, 2);
    recorders[0].kill();
    await tickSeconds(comp, 4); // interruption + failed resume attempts
    expect(queryId(comp, "interruption-banner")).not.toBeNull();
    expect(recorders).toHaveLength(1);

    await act(async () => {
      queryId(comp, "stop-audio-recording")!.props.onPress();
    });
    expect(onComplete).toHaveBeenCalledTimes(1);
    const file = onComplete.mock.calls[0][0] as RecordedAudioFile;
    expect(file.mimeType).toBe("audio/wav");
    expect(fs.exists(file.uri)).toBe(true);
  });
});
