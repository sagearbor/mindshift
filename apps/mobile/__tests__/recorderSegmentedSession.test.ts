import { MemoryFs } from "../src/recorder/memoryFs";
import { RecorderSessionStore } from "../src/recorder/sessionStore";
import { SegmentedAudioSession } from "../src/recorder/segmentedSession";
import type { RecorderPort } from "../src/recorder/types";

const WAV_FORMAT = {
  format: "wav" as const,
  extension: ".wav",
  mimeType: "audio/wav",
  bytesPerSecond: 32000,
};

const SEGMENT_MS = 5 * 60 * 1000;

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

function wavTags(bytes: Uint8Array): number[] {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const dataLen = v.getUint32(40, true);
  const tags: number[] = [];
  for (let i = 0; i < dataLen; i += 2) tags.push(v.getInt16(44 + i, true));
  return tags;
}

/**
 * A controllable stand-in for the expo-audio recorder. Each instance writes a
 * one-sample WAV (tagged with its creation number) to the fake filesystem when
 * stopped — mimicking the native recorder finalizing its file on stop().
 */
class FakeRecorder implements RecorderPort {
  uri: string | null = null;
  recording = false;
  released = false;
  stopThrows = false;
  constructor(
    private fs: MemoryFs,
    readonly tag: number,
  ) {}
  async prepare(): Promise<void> {
    this.uri = `file:///cache/fake-rec-${this.tag}.wav`;
  }
  start(): void {
    this.recording = true;
  }
  async stop(): Promise<{ uri: string | null }> {
    this.recording = false;
    if (this.stopThrows) throw new Error("recorder already dead");
    if (this.uri) this.fs.writeBytes(this.uri, wavBytes(this.tag));
    return { uri: this.uri };
  }
  isRecording(): boolean {
    return this.recording;
  }
  release(): void {
    this.released = true;
  }
  /** Simulate the OS killing the recording (phone call / focus loss). */
  kill(opts: { stopThrows?: boolean } = {}): void {
    this.recording = false;
    this.stopThrows = opts.stopThrows ?? false;
  }
}

function makeHarness(opts: { failPrepareAfter?: number } = {}) {
  const fs = new MemoryFs();
  const store = new RecorderSessionStore(fs);
  const recorders: FakeRecorder[] = [];
  let t = 1_000_000;
  const makeRecorder = () => {
    if (
      opts.failPrepareAfter !== undefined &&
      recorders.length >= opts.failPrepareAfter
    ) {
      throw new Error("mic unavailable");
    }
    const r = new FakeRecorder(fs, recorders.length + 1);
    recorders.push(r);
    return r;
  };
  const session = new SegmentedAudioSession({
    makeRecorder,
    store,
    format: WAV_FORMAT,
    segmentMs: SEGMENT_MS,
    resumeRetryMs: 2000,
    now: () => t,
  });
  return {
    fs,
    store,
    recorders,
    session,
    advance(ms: number) {
      t += ms;
    },
  };
}

describe("SegmentedAudioSession — rotation", () => {
  it("start() begins recording the first segment", async () => {
    const h = makeHarness();
    await h.session.start();
    expect(h.session.phase).toBe("recording");
    expect(h.recorders).toHaveLength(1);
    expect(h.recorders[0].isRecording()).toBe(true);
    expect(h.session.segmentsSaved).toBe(0);
  });

  it("does not rotate before the segment interval", async () => {
    const h = makeHarness();
    await h.session.start();
    h.advance(SEGMENT_MS - 1000);
    await h.session.tick();
    expect(h.recorders).toHaveLength(1);
    expect(h.session.segmentsSaved).toBe(0);
  });

  it("rotates at the interval: finalizes the segment to disk and keeps recording", async () => {
    const h = makeHarness();
    await h.session.start();
    h.advance(SEGMENT_MS);
    await h.session.tick();

    // Segment 1 is safely on disk — a crash now cannot lose it.
    expect(h.session.segmentsSaved).toBe(1);
    expect(h.session.savedDurationMs).toBe(SEGMENT_MS);
    const [rec] = h.store.listRecoverable();
    expect(rec.segmentCount).toBe(1);
    // A NEW recorder is capturing the next segment.
    expect(h.recorders).toHaveLength(2);
    expect(h.recorders[1].isRecording()).toBe(true);
    expect(h.session.phase).toBe("recording");
  });

  it("rotates repeatedly, keeping segments in order", async () => {
    const h = makeHarness();
    await h.session.start();
    for (let i = 0; i < 3; i++) {
      h.advance(SEGMENT_MS);
      await h.session.tick();
    }
    expect(h.session.segmentsSaved).toBe(3);
    const file = await h.session.stopAndFinish();
    expect(wavTags(h.fs.readBytes(file.uri))).toEqual([1, 2, 3, 4]);
  });
});

describe("SegmentedAudioSession — finishing", () => {
  it("stopAndFinish finalizes the live segment and returns one stitched audio file", async () => {
    const h = makeHarness();
    await h.session.start();
    h.advance(SEGMENT_MS);
    await h.session.tick();
    h.advance(60_000);
    const file = await h.session.stopAndFinish();

    expect(h.session.phase).toBe("done");
    expect(file.mimeType).toBe("audio/wav");
    expect(file.name).toMatch(/\.wav$/);
    expect(wavTags(h.fs.readBytes(file.uri))).toEqual([1, 2]);
    // Clean finish leaves nothing to recover.
    expect(h.store.listRecoverable()).toEqual([]);
  });

  it("a short session (no rotation yet) still produces a file", async () => {
    const h = makeHarness();
    await h.session.start();
    h.advance(30_000);
    const file = await h.session.stopAndFinish();
    expect(wavTags(h.fs.readBytes(file.uri))).toEqual([1]);
  });

  it("reports elapsed time across segments", async () => {
    const h = makeHarness();
    await h.session.start();
    h.advance(SEGMENT_MS);
    await h.session.tick();
    h.advance(90_000);
    expect(h.session.elapsedMs()).toBe(SEGMENT_MS + 90_000);
  });
});

describe("SegmentedAudioSession — interruption (call / audio-focus loss)", () => {
  it("finalizes the current segment immediately when the recorder dies", async () => {
    const h = makeHarness();
    await h.session.start();
    h.advance(120_000);
    h.recorders[0].kill();
    await h.session.tick();

    expect(h.session.phase).toBe("interrupted");
    // The partial segment was salvaged to disk (recoverable even if we die now).
    expect(h.session.segmentsSaved).toBe(1);
    expect(h.store.listRecoverable()[0].segmentCount).toBe(1);
  });

  it("survives a recorder whose stop() throws — no segment, but no crash", async () => {
    const h = makeHarness();
    await h.session.start();
    h.recorders[0].kill({ stopThrows: true });
    await h.session.tick();
    expect(h.session.phase).toBe("interrupted");
    expect(h.session.segmentsSaved).toBe(0);
  });

  it("auto-resumes with a fresh recorder once the retry delay passes", async () => {
    const h = makeHarness();
    await h.session.start();
    h.recorders[0].kill();
    await h.session.tick(); // -> interrupted
    h.advance(2000); // resumeRetryMs
    await h.session.tick();

    expect(h.session.phase).toBe("recording");
    expect(h.recorders).toHaveLength(2);
    expect(h.recorders[1].isRecording()).toBe(true);
  });

  it("does not retry before the resume delay", async () => {
    const h = makeHarness();
    await h.session.start();
    h.recorders[0].kill();
    await h.session.tick();
    h.advance(500);
    await h.session.tick();
    expect(h.session.phase).toBe("interrupted");
    expect(h.recorders).toHaveLength(1);
  });

  it("stays interrupted (recoverable) when resume keeps failing", async () => {
    const h = makeHarness({ failPrepareAfter: 1 });
    await h.session.start();
    h.advance(60_000);
    h.recorders[0].kill();
    await h.session.tick();
    h.advance(2000);
    await h.session.tick(); // resume attempt fails
    expect(h.session.phase).toBe("interrupted");
    // The salvaged audio is still on disk for the recovery flow.
    expect(h.store.listRecoverable()[0].segmentCount).toBe(1);
  });

  it("abandon() salvages the live segment and leaves a recoverable session", async () => {
    // The user backs out (or the screen unmounts) mid-recording: nothing may
    // be lost silently — the audio so far becomes a recoverable session.
    const h = makeHarness();
    await h.session.start();
    h.advance(90_000);
    await h.session.abandon();
    expect(h.session.phase).toBe("interrupted");
    const [rec] = h.store.listRecoverable();
    expect(rec.segmentCount).toBe(1);
    expect(rec.totalDurationMs).toBe(90_000);
  });

  it("stopAndFinish from interrupted returns what was saved", async () => {
    const h = makeHarness({ failPrepareAfter: 1 });
    await h.session.start();
    h.advance(60_000);
    h.recorders[0].kill();
    await h.session.tick();
    const file = await h.session.stopAndFinish();
    expect(wavTags(h.fs.readBytes(file.uri))).toEqual([1]);
    expect(h.store.listRecoverable()).toEqual([]);
  });
});
