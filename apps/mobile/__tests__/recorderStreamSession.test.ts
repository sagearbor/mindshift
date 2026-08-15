/**
 * StreamAudioSession — the gapless v2 recording engine.
 *
 * v1 (SegmentedAudioSession) stops/restarts the platform recorder every ~5
 * minutes: ~0.2 s of audio lost per rotation and up to a whole segment lost on
 * a crash. v2 never touches the mic after start: a continuous PCM stream is
 * appended to a WAV file, flushed + manifest-updated every ~5 seconds, and
 * rotated to a new file BETWEEN two buffer writes — so rotation loses zero
 * samples and a crash loses at most the unflushed tail (seconds, not minutes).
 *
 * The proofs here are sample-exact: a known ramp signal goes in through a fake
 * PCM source, and the reassembled output must contain every sample, in order,
 * with none lost at any boundary.
 */
import { MemoryFs } from "../src/recorder/memoryFs";
import {
  RecorderSessionStore,
  segmentFileName,
} from "../src/recorder/sessionStore";
import {
  StreamAudioSession,
  DEFAULT_FLUSH_MS,
  DEFAULT_STALL_MS,
} from "../src/recorder/streamSession";
import type { PcmFrame, PcmSource } from "../src/recorder/pcmSource";

/** Fake rate keeps test arrays small: 1000 samples = 1 s of audio. */
const RATE = 1000;
const SEGMENT_MS = 20_000; // rotate every 20 "seconds" (20k samples)
const FLUSH_MS = 5_000;
const RESUME_RETRY_MS = 2_000;

/** Deterministic ramp: sample i has value (i % 30000) — int16-safe, and any
 *  dropped/duplicated/reordered sample breaks the sequence detectably. */
function rampValue(i: number): number {
  return i % 30000;
}

function ramp(start: number, count: number): Int16Array {
  const out = new Int16Array(count);
  for (let i = 0; i < count; i++) out[i] = rampValue(start + i);
  return out;
}

/** A controllable stand-in for the expo-audio PCM stream. */
class FakePcmSource implements PcmSource {
  onFrame: ((frame: PcmFrame) => void) | null = null;
  capturing = false;
  startThrows = false;
  async start(onFrame: (frame: PcmFrame) => void): Promise<void> {
    if (this.startThrows) throw new Error("mic unavailable");
    this.onFrame = onFrame;
    this.capturing = true;
  }
  stop(): void {
    this.capturing = false;
  }
  isCapturing(): boolean {
    return this.capturing;
  }
  push(samples: Int16Array, sampleRate = RATE): void {
    this.onFrame?.({ samples, sampleRate });
  }
  /** Simulate the OS killing the stream (phone call / focus loss). */
  kill(): void {
    this.capturing = false;
  }
}

interface HarnessOpts {
  failResume?: boolean;
  stallMs?: number;
}

function makeHarness(opts: HarnessOpts = {}) {
  const fs = new MemoryFs();
  const store = new RecorderSessionStore(fs);
  const sources: FakePcmSource[] = [];
  let t = 1_000_000;
  const session = new StreamAudioSession({
    makeSource: () => {
      const s = new FakePcmSource();
      if (opts.failResume && sources.length >= 1) s.startThrows = true;
      sources.push(s);
      return s;
    },
    store,
    segmentMs: SEGMENT_MS,
    flushMs: FLUSH_MS,
    resumeRetryMs: RESUME_RETRY_MS,
    stallMs: opts.stallMs ?? 60_000,
    now: () => t,
  });
  return {
    fs,
    store,
    sources,
    session,
    get source() {
      return sources[sources.length - 1];
    },
    advance(ms: number) {
      t += ms;
    },
    now() {
      return t;
    },
    /** Push `seconds` of ramp audio in ~100 ms frames, starting at sample
     *  index `startSample`, ticking the engine once per pushed second. */
    async pushSeconds(startSample: number, seconds: number, rate = RATE) {
      let cursor = startSample;
      for (let s = 0; s < seconds; s++) {
        for (let f = 0; f < 10; f++) {
          const n = rate / 10;
          this.source.push(ramp(cursor, n), rate);
          cursor += n;
        }
        this.advance(1000);
        await session.tick();
      }
      return cursor;
    },
  };
}

/** Parse a WAV file: canonical 44-byte header + PCM s16le data. */
function parseWav(bytes: Uint8Array) {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const ascii = (o: number) =>
    String.fromCharCode(bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]);
  expect(ascii(0)).toBe("RIFF");
  expect(ascii(8)).toBe("WAVE");
  expect(ascii(12)).toBe("fmt ");
  expect(ascii(36)).toBe("data");
  const riffSize = v.getUint32(4, true);
  const fmtSize = v.getUint32(16, true);
  const audioFormat = v.getUint16(20, true);
  const channels = v.getUint16(22, true);
  const sampleRate = v.getUint32(24, true);
  const byteRate = v.getUint32(28, true);
  const blockAlign = v.getUint16(32, true);
  const bitsPerSample = v.getUint16(34, true);
  const dataSize = v.getUint32(40, true);
  const samples: number[] = [];
  for (let i = 0; i + 1 < dataSize && 44 + i + 1 < bytes.byteLength; i += 2) {
    samples.push(v.getInt16(44 + i, true));
  }
  return {
    riffSize,
    fmtSize,
    audioFormat,
    channels,
    sampleRate,
    byteRate,
    blockAlign,
    bitsPerSample,
    dataSize,
    samples,
    fileBytes: bytes.byteLength,
  };
}

function expectRamp(samples: number[], startSample: number, count: number) {
  expect(samples.length).toBe(count);
  // Sample-exact continuity: every position must hold exactly its ramp value.
  for (let i = 0; i < count; i++) {
    if (samples[i] !== rampValue(startSample + i)) {
      throw new Error(
        `ramp broken at sample ${i}: expected ${rampValue(
          startSample + i,
        )}, got ${samples[i]}`,
      );
    }
  }
}

describe("StreamAudioSession — capture & flush cadence", () => {
  it("start() opens the stream once and begins recording", async () => {
    const h = makeHarness();
    await h.session.start();
    expect(h.session.phase).toBe("recording");
    expect(h.sources).toHaveLength(1);
    expect(h.source.isCapturing()).toBe(true);
  });

  it("does not write audio to disk before the flush interval", async () => {
    const h = makeHarness();
    await h.session.start();
    h.source.push(ramp(0, 100));
    h.advance(1000);
    await h.session.tick();
    // Nothing flushed yet: no crash-safe audio, no manifest segments.
    expect(h.session.savedDurationMs).toBe(0);
    expect(h.store.listRecoverable()).toEqual([]);
  });

  it("flushes buffered frames to the segment file within the flush interval", async () => {
    const h = makeHarness();
    await h.session.start();
    await h.pushSeconds(0, 5); // 5 s of audio, tick each second
    // The 5 s tick crossed the flush interval: everything so far is on disk.
    expect(h.session.savedDurationMs).toBe(5000);
    const [rec] = h.store.listRecoverable();
    expect(rec.segmentCount).toBe(1);
    expect(rec.totalDurationMs).toBe(5000);
    const dir = `file:///doc/recorder-sessions/${rec.manifest.sessionId}`;
    const wav = parseWav(h.fs.readBytes(`${dir}/${rec.manifest.segments[0].file}`));
    expect(wav.sampleRate).toBe(RATE);
    expectRamp(wav.samples, 0, 5 * RATE);
  });

  it("keeps growing the same segment file on subsequent flushes", async () => {
    const h = makeHarness();
    await h.session.start();
    await h.pushSeconds(0, 5);
    const [before] = h.store.listRecoverable();
    const file = before.manifest.segments[0].file;
    const dir = `file:///doc/recorder-sessions/${before.manifest.sessionId}`;
    const sizeBefore = h.fs.readBytes(`${dir}/${file}`).byteLength;

    await h.pushSeconds(5 * RATE, 5); // next flush lands inside the same segment
    const sizeAfter = h.fs.readBytes(`${dir}/${file}`).byteLength;
    expect(sizeAfter).toBeGreaterThan(sizeBefore);
    expect(h.session.savedDurationMs).toBe(10_000);
    // Still ONE segment — no rotation yet.
    expect(h.store.listRecoverable()[0].segmentCount).toBe(1);
  });

  it("the open segment is playable after every flush (header sizes patched)", async () => {
    const h = makeHarness();
    await h.session.start();
    await h.pushSeconds(0, 5);
    const [rec] = h.store.listRecoverable();
    const dir = `file:///doc/recorder-sessions/${rec.manifest.sessionId}`;
    const bytes = h.fs.readBytes(`${dir}/${rec.manifest.segments[0].file}`);
    const wav = parseWav(bytes);
    // Header must describe exactly the bytes on disk — playable as-is.
    expect(wav.riffSize).toBe(wav.fileBytes - 8);
    expect(wav.dataSize).toBe(wav.fileBytes - 44);
    expect(wav.audioFormat).toBe(1); // PCM
    expect(wav.channels).toBe(1);
    expect(wav.bitsPerSample).toBe(16);
    expect(wav.byteRate).toBe(RATE * 2);
    expect(wav.blockAlign).toBe(2);
  });
});

describe("StreamAudioSession — gapless rotation", () => {
  it("rotates to a new file at the segment interval WITHOUT losing a sample", async () => {
    const h = makeHarness();
    await h.session.start();
    // 45 s of continuous ramp — crosses two 20 s rotation boundaries.
    const total = await h.pushSeconds(0, 45);
    expect(h.session.segmentsSaved).toBeGreaterThanOrEqual(3);
    // Only ONE stream was ever opened: the mic never stopped.
    expect(h.sources).toHaveLength(1);

    const file = await h.session.stopAndFinish();
    const wav = parseWav(h.fs.readBytes(file.uri));
    // THE gapless proof: every sample of the ramp present, in order.
    expectRamp(wav.samples, 0, total);
    expect(h.store.listRecoverable()).toEqual([]);
  });

  it("each closed segment is an independently playable WAV", async () => {
    const h = makeHarness();
    await h.session.start();
    await h.pushSeconds(0, 25); // one rotation (20 s) + 5 s into segment 2
    const [rec] = h.store.listRecoverable();
    expect(rec.segmentCount).toBe(2);
    const dir = `file:///doc/recorder-sessions/${rec.manifest.sessionId}`;
    const closed = parseWav(h.fs.readBytes(`${dir}/${segmentFileName(0, ".wav")}`));
    expect(closed.riffSize).toBe(closed.fileBytes - 8);
    expect(closed.dataSize).toBe(closed.fileBytes - 44);
    expectRamp(closed.samples, 0, 20 * RATE);
    // The open segment continues exactly where the closed one ended.
    const open = parseWav(h.fs.readBytes(`${dir}/${segmentFileName(1, ".wav")}`));
    expectRamp(open.samples, 20 * RATE, 5 * RATE);
  });

  it("a short session (no rotation) still finishes to a single valid file", async () => {
    const h = makeHarness();
    await h.session.start();
    const total = await h.pushSeconds(0, 3);
    const file = await h.session.stopAndFinish();
    expect(file.mimeType).toBe("audio/wav");
    expect(file.name).toMatch(/\.wav$/);
    const wav = parseWav(h.fs.readBytes(file.uri));
    expectRamp(wav.samples, 0, total);
    expect(h.session.phase).toBe("done");
  });

  it("stopAndFinish flushes the unflushed tail — stopping never loses audio", async () => {
    const h = makeHarness();
    await h.session.start();
    await h.pushSeconds(0, 5);
    // 2 more seconds of frames but NO tick — nothing flushed yet.
    for (let f = 0; f < 20; f++) {
      h.source.push(ramp(5 * RATE + f * (RATE / 10), RATE / 10));
    }
    const file = await h.session.stopAndFinish();
    const wav = parseWav(h.fs.readBytes(file.uri));
    expectRamp(wav.samples, 0, 7 * RATE);
  });
});

describe("StreamAudioSession — crash recovery (v2)", () => {
  it("a crash loses only the unflushed tail; flushed audio recovers sample-exact", async () => {
    const h = makeHarness();
    await h.session.start();
    await h.pushSeconds(0, 5); // flushed at the 5 s tick
    // 3 more seconds arrive but never flush (no tick) — then the app dies.
    for (let f = 0; f < 30; f++) {
      h.source.push(ramp(5 * RATE + f * (RATE / 10), RATE / 10));
    }

    // "Relaunch": a fresh store over the same disk.
    const relaunchStore = new RecorderSessionStore(h.fs);
    const recoverable = relaunchStore.listRecoverable();
    expect(recoverable).toHaveLength(1);
    expect(recoverable[0].totalDurationMs).toBe(5000);

    const file = relaunchStore.finishToFile(recoverable[0].manifest);
    const wav = parseWav(h.fs.readBytes(file.uri));
    // Exactly the flushed 5 s — the unflushed tail is honestly gone, and
    // nothing that WAS flushed is missing or corrupted.
    expectRamp(wav.samples, 0, 5 * RATE);
    expect(relaunchStore.listRecoverable()).toEqual([]);
  });

  it("recovery concatenates multiple segments sample-exact (pure byte math)", async () => {
    const h = makeHarness();
    await h.session.start();
    await h.pushSeconds(0, 45); // 2 closed segments + 5 s flushed in the third

    const relaunchStore = new RecorderSessionStore(h.fs);
    const [rec] = relaunchStore.listRecoverable();
    expect(rec.segmentCount).toBe(3);
    const file = relaunchStore.finishToFile(rec.manifest);
    const wav = parseWav(h.fs.readBytes(file.uri));
    expectRamp(wav.samples, 0, 45 * RATE);
  });

  it("a crash before the first flush leaves nothing recoverable (and no junk)", async () => {
    const h = makeHarness();
    await h.session.start();
    h.source.push(ramp(0, 300)); // 0.3 s, never flushed
    const relaunchStore = new RecorderSessionStore(h.fs);
    expect(relaunchStore.listRecoverable()).toEqual([]);
    // The empty session directory was cleaned up by the scan.
    expect(h.fs.listDirNames("file:///doc/recorder-sessions")).toEqual([]);
  });

  it("v1 sessions on the same disk still recover through the old path", async () => {
    const h = makeHarness();
    // Seed a v1-style crashed session (segments moved in via finalizeSegment).
    let m = h.store.createSession({
      format: "wav",
      extension: ".wav",
      mimeType: "audio/wav",
      segmentSeconds: 300,
    });
    const v1Wav = (() => {
      const data = ramp(0, 100);
      const buf = new Uint8Array(44 + 200);
      const v = new DataView(buf.buffer);
      const w = (o: number, s: string) => {
        for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
      };
      w(0, "RIFF");
      v.setUint32(4, 36 + 200, true);
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
      v.setUint32(40, 200, true);
      for (let i = 0; i < 100; i++) v.setInt16(44 + i * 2, data[i], true);
      return buf;
    })();
    h.fs.writeBytes("file:///cache/v1-seg.wav", v1Wav);
    m = h.store.finalizeSegment(m, "file:///cache/v1-seg.wav", 60_000);

    // And a crashed v2 session alongside it.
    await h.session.start();
    await h.pushSeconds(0, 5);

    const relaunchStore = new RecorderSessionStore(h.fs);
    const recoverable = relaunchStore.listRecoverable();
    expect(recoverable).toHaveLength(2);
    for (const rec of recoverable) {
      const file = relaunchStore.finishToFile(rec.manifest);
      expect(parseWav(h.fs.readBytes(file.uri)).samples.length).toBeGreaterThan(0);
    }
  });
});

describe("StreamAudioSession — honest sample rate", () => {
  it("writes the TRUE reported rate into the WAV header, never a mislabel", async () => {
    const h = makeHarness();
    await h.session.start();
    // Device reports 44.1 kHz despite our 16 kHz request.
    h.source.push(ramp(0, 4410), 44100);
    h.advance(FLUSH_MS);
    await h.session.tick();
    const [rec] = h.store.listRecoverable();
    const dir = `file:///doc/recorder-sessions/${rec.manifest.sessionId}`;
    const wav = parseWav(h.fs.readBytes(`${dir}/${rec.manifest.segments[0].file}`));
    expect(wav.sampleRate).toBe(44100);
    // Duration bookkeeping uses the true rate too: 4410 samples = 100 ms.
    expect(rec.manifest.segments[0].durationMs).toBe(100);
  });

  it("a mid-session rate change rotates so every file's header stays truthful", async () => {
    const h = makeHarness();
    await h.session.start();
    await h.pushSeconds(0, 2, RATE);
    h.source.push(ramp(0, 800), 8000); // the hardware switched rates
    h.advance(FLUSH_MS);
    await h.session.tick();
    const [rec] = h.store.listRecoverable();
    expect(rec.segmentCount).toBe(2);
    const dir = `file:///doc/recorder-sessions/${rec.manifest.sessionId}`;
    const first = parseWav(h.fs.readBytes(`${dir}/${rec.manifest.segments[0].file}`));
    const second = parseWav(h.fs.readBytes(`${dir}/${rec.manifest.segments[1].file}`));
    expect(first.sampleRate).toBe(RATE);
    expect(second.sampleRate).toBe(8000);
    expectRamp(second.samples, 0, 800);
  });
});

describe("StreamAudioSession — interruption & stall", () => {
  it("flushes everything and goes interrupted when the OS kills the stream", async () => {
    const h = makeHarness();
    await h.session.start();
    await h.pushSeconds(0, 2);
    // 0.5 s more, unflushed, then the OS takes the mic.
    for (let f = 0; f < 5; f++) {
      h.source.push(ramp(2 * RATE + f * (RATE / 10), RATE / 10));
    }
    h.source.kill();
    h.advance(1000);
    await h.session.tick();

    expect(h.session.phase).toBe("interrupted");
    // NOTHING was lost: even the unflushed tail was salvaged to disk.
    expect(h.session.savedDurationMs).toBe(2500);
    expect(h.store.listRecoverable()[0].totalDurationMs).toBe(2500);
  });

  it("auto-resumes with a fresh stream into a NEW segment after the retry delay", async () => {
    const h = makeHarness();
    await h.session.start();
    await h.pushSeconds(0, 2);
    h.source.kill();
    h.advance(1000);
    await h.session.tick(); // -> interrupted
    h.advance(RESUME_RETRY_MS);
    await h.session.tick(); // resume

    expect(h.session.phase).toBe("recording");
    expect(h.sources).toHaveLength(2);
    expect(h.source.isCapturing()).toBe(true);

    // Audio after resume lands in a new segment; stop stitches both runs.
    const more = ramp(2 * RATE, RATE);
    h.source.push(more);
    const file = await h.session.stopAndFinish();
    const wav = parseWav(h.fs.readBytes(file.uri));
    expectRamp(wav.samples, 0, 3 * RATE);
  });

  it("does not retry before the resume delay and stays interrupted on failure", async () => {
    const h = makeHarness({ failResume: true });
    await h.session.start();
    await h.pushSeconds(0, 2);
    h.source.kill();
    h.advance(1000);
    await h.session.tick();
    h.advance(500);
    await h.session.tick(); // too soon — no attempt
    expect(h.sources).toHaveLength(1);
    h.advance(RESUME_RETRY_MS);
    await h.session.tick(); // attempt fails (mic still unavailable)
    expect(h.session.phase).toBe("interrupted");
    // Everything captured so far stays recoverable.
    expect(h.store.listRecoverable()[0].totalDurationMs).toBe(2000);
  });

  it("treats a silent frame stall as an interruption (mic gone without telling us)", async () => {
    const h = makeHarness({ stallMs: 4000 });
    await h.session.start();
    await h.pushSeconds(0, 2);
    // The source still claims to be capturing, but frames stop arriving.
    h.advance(4000);
    await h.session.tick();
    expect(h.session.phase).toBe("interrupted");
    expect(h.session.savedDurationMs).toBe(2000);
  });

  it("abandon() salvages the unflushed tail and leaves a recoverable session", async () => {
    const h = makeHarness();
    await h.session.start();
    await h.pushSeconds(0, 2);
    for (let f = 0; f < 5; f++) {
      h.source.push(ramp(2 * RATE + f * (RATE / 10), RATE / 10));
    }
    await h.session.abandon();
    expect(h.session.phase).toBe("interrupted");
    expect(h.source.isCapturing()).toBe(false);
    expect(h.store.listRecoverable()[0].totalDurationMs).toBe(2500);
  });

  it("frames arriving after stop are ignored, not appended", async () => {
    const h = makeHarness();
    await h.session.start();
    const total = await h.pushSeconds(0, 3);
    const file = await h.session.stopAndFinish();
    // A trailing native buffer lands after stop — must not corrupt anything.
    h.source.push(ramp(9999, 100));
    const wav = parseWav(h.fs.readBytes(file.uri));
    expectRamp(wav.samples, 0, total);
  });
});

describe("StreamAudioSession — time accounting", () => {
  it("elapsedMs reflects captured audio; savedDurationMs only flushed audio", async () => {
    const h = makeHarness();
    await h.session.start();
    await h.pushSeconds(0, 5); // flushed
    for (let f = 0; f < 10; f++) {
      h.source.push(ramp(5 * RATE + f * (RATE / 10), RATE / 10)); // unflushed
    }
    expect(h.session.elapsedMs()).toBe(6000);
    expect(h.session.savedDurationMs).toBe(5000);
    expect(h.session.segmentsSaved).toBe(1);
  });

  it("exports sane defaults: flush every ~5 s, stall guard ~10 s", () => {
    expect(DEFAULT_FLUSH_MS).toBe(5000);
    expect(DEFAULT_STALL_MS).toBe(10_000);
  });
});
