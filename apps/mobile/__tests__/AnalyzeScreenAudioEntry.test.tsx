import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import AnalyzeScreen from "../src/screens/AnalyzeScreen";
import type { AudioRecorderDeps } from "../src/recorder/AudioRecordScreen";
import { MemoryFs } from "../src/recorder/memoryFs";
import { RecorderSessionStore } from "../src/recorder/sessionStore";
import type { RecorderPort } from "../src/recorder/types";

// Keep the real client module shape; nothing here should hit the network.
jest.mock("../src/api/client", () => ({
  __esModule: true,
  ...jest.requireActual("../src/api/client"),
  postAnalyzeUpload: jest.fn(),
  postAnalyzeUploadChunkedJob: jest.fn(),
  postAnalyzeLink: jest.fn(),
  postAnalyzeLinkJob: jest.fn(),
  getAnalyzeJob: jest.fn(),
}));

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
    this.uri = `file:///cache/entry-rec-${this.tag}.wav`;
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
}

function makeDeps() {
  const fs = new MemoryFs();
  const store = new RecorderSessionStore(fs);
  let n = 0;
  const deps: AudioRecorderDeps = {
    store,
    makeRecorder: () => new FakeRecorder(fs, ++n),
    format: {
      format: "wav",
      extension: ".wav",
      mimeType: "audio/wav",
      bytesPerSecond: 32000,
    },
    getBatteryLevel: async () => 0.9,
    configureAudioSession: jest.fn().mockResolvedValue(undefined),
    segmentMs: 4000,
    resumeRetryMs: 2000,
  };
  return { fs, store, deps };
}

/** Seed the disk with a crashed session: two 5-minute segments, no finish. */
function seedOrphanedSession(fs: MemoryFs, store: RecorderSessionStore) {
  let m = store.createSession({
    format: "wav",
    extension: ".wav",
    mimeType: "audio/wav",
    segmentSeconds: 300,
  });
  for (const tag of [1, 2]) {
    const uri = `file:///cache/orphan-${tag}.wav`;
    fs.writeBytes(uri, wavBytes(tag));
    m = store.finalizeSegment(m, uri, 5 * 60 * 1000);
  }
  return m;
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
    } else if (typeof children === "string" || typeof children === "number") {
      parts.push(String(children));
    } else if (children && typeof children === "object") {
      walk((children as { props?: { children?: unknown } }).props?.children);
    }
  };
  walk(node.props.children);
  return parts.join("");
}

async function mountAnalyze(deps: AudioRecorderDeps) {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<AnalyzeScreen recorderDeps={deps} />);
  });
  return comp;
}

beforeEach(() => {
  jest.useFakeTimers();
});

afterEach(() => {
  jest.useRealTimers();
});

describe("AnalyzeScreen — audio recording entry", () => {
  it("offers a Record audio entry alongside Record video", async () => {
    const { deps } = makeDeps();
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <AnalyzeScreen onRecordVideo={jest.fn()} recorderDeps={deps} />,
      );
    });
    expect(queryId(comp, "record-audio-button")).not.toBeNull();
    expect(queryId(comp, "record-video-button")).not.toBeNull();
  });

  it("opens the audio recorder in place", async () => {
    const { deps } = makeDeps();
    const comp = await mountAnalyze(deps);
    await act(async () => {
      queryId(comp, "record-audio-button")!.props.onPress();
    });
    expect(queryId(comp, "gap-disclosure")).not.toBeNull();
    expect(queryId(comp, "analyze-screen")).toBeNull();
  });

  it("a finished recording lands preselected in the upload flow as AUDIO", async () => {
    const { deps } = makeDeps();
    const comp = await mountAnalyze(deps);
    await act(async () => {
      queryId(comp, "record-audio-button")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "start-audio-recording")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "stop-audio-recording")!.props.onPress();
    });
    // Back on the analyze screen with the stitched audio file preselected.
    expect(queryId(comp, "analyze-screen")).not.toBeNull();
    const picked = textOf(queryId(comp, "picked-file"));
    expect(picked).toMatch(/mindshift-audio-.*\.wav/);
    expect(queryId(comp, "upload-analyze-button")).not.toBeNull();
  });

  it("backing out of the recorder returns to the analyze screen", async () => {
    const { deps } = makeDeps();
    const comp = await mountAnalyze(deps);
    await act(async () => {
      queryId(comp, "record-audio-button")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "audio-record-back")!.props.onPress();
    });
    expect(queryId(comp, "analyze-screen")).not.toBeNull();
  });
});

describe("AnalyzeScreen — crash recovery prompt", () => {
  it("offers recovery when an interrupted session is found on disk", async () => {
    const { fs, store, deps } = makeDeps();
    seedOrphanedSession(fs, store);
    const comp = await mountAnalyze(deps);
    const prompt = queryId(comp, "recovery-prompt");
    expect(prompt).not.toBeNull();
    expect(textOf(prompt)).toMatch(/unfinished recording/i);
    expect(textOf(prompt)).toContain("10 min");
  });

  it("shows no prompt when the disk is clean", async () => {
    const { deps } = makeDeps();
    const comp = await mountAnalyze(deps);
    expect(queryId(comp, "recovery-prompt")).toBeNull();
  });

  it("recovering stitches the segments and preselects the audio file", async () => {
    const { fs, store, deps } = makeDeps();
    seedOrphanedSession(fs, store);
    const comp = await mountAnalyze(deps);
    await act(async () => {
      queryId(comp, "recovery-recover")!.props.onPress();
    });
    expect(queryId(comp, "recovery-prompt")).toBeNull();
    expect(textOf(queryId(comp, "picked-file"))).toMatch(
      /mindshift-audio-.*\.wav/,
    );
    // Recovered = consumed: nothing left on disk to re-prompt about.
    expect(store.listRecoverable()).toEqual([]);
  });

  it("discarding removes the session and keeps nothing selected", async () => {
    const { fs, store, deps } = makeDeps();
    seedOrphanedSession(fs, store);
    const comp = await mountAnalyze(deps);
    await act(async () => {
      queryId(comp, "recovery-discard")!.props.onPress();
    });
    expect(queryId(comp, "recovery-prompt")).toBeNull();
    expect(queryId(comp, "picked-file")).toBeNull();
    expect(store.listRecoverable()).toEqual([]);
  });
});
