/**
 * live/liveAudioKeeper.ts over MemoryFs: the per-session WAV the Live Coach
 * keeps on the phone is a valid file after every flush, sized exactly
 * 44 + 2 × samples, and leaves nothing behind when discarded or unused.
 */
import {
  createLiveAudioKeeper,
  keepAudioStorageProblem,
  liveAudioFileUri,
} from "../src/live/liveAudioKeeper";
import { MemoryFs } from "../src/recorder/memoryFs";
import { WAV_HEADER_BYTES } from "../src/recorder/wav";

const DIR = "file:///cache/live-audio";

function ascii(bytes: Uint8Array, offset: number, len: number): string {
  return String.fromCharCode(...bytes.subarray(offset, offset + len));
}

function u32(bytes: Uint8Array, offset: number): number {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(offset, true);
}

/** Assert a canonical 16 kHz mono s16 header whose sizes match the file. */
function expectValidWav(bytes: Uint8Array, samples: number) {
  expect(ascii(bytes, 0, 4)).toBe("RIFF");
  expect(ascii(bytes, 8, 4)).toBe("WAVE");
  expect(ascii(bytes, 12, 4)).toBe("fmt ");
  expect(ascii(bytes, 36, 4)).toBe("data");
  expect(u32(bytes, 24)).toBe(16000);
  expect(bytes.byteLength).toBe(WAV_HEADER_BYTES + samples * 2);
  expect(u32(bytes, 40)).toBe(samples * 2);
  expect(u32(bytes, 4)).toBe(bytes.byteLength - 8);
}

function frame(n: number, value = 1234): Int16Array {
  return new Int16Array(n).fill(value);
}

describe("createLiveAudioKeeper", () => {
  it("names the file after the session id, sanitised", () => {
    expect(liveAudioFileUri(DIR, "live-2")).toBe(`${DIR}/live-2.wav`);
    expect(liveAudioFileUri(`${DIR}/`, "a/b c")).toBe(`${DIR}/a_b_c.wav`);
  });

  it("writes nothing until audio arrives, then a valid WAV after every flush", async () => {
    const fs = new MemoryFs();
    let t = 0;
    const keeper = createLiveAudioKeeper({
      fs,
      dir: DIR,
      sessionId: "s1",
      flushMs: 1000,
      now: () => t,
    });
    expect(fs.exists(keeper.uri)).toBe(false);
    expect(fs.listFileNames(DIR)).toEqual([]);

    keeper.append(frame(1600));
    keeper.append(frame(1600));
    // Under the flush cadence: still only in memory.
    expect(fs.exists(keeper.uri)).toBe(false);
    expect(keeper.bytes).toBe(WAV_HEADER_BYTES + 3200 * 2);

    t = 1000;
    keeper.append(frame(800)); // crosses the cadence → flush
    expect(fs.exists(keeper.uri)).toBe(true);
    expectValidWav(fs.readBytes(keeper.uri), 4000);

    keeper.append(frame(1600));
    keeper.flush();
    expectValidWav(fs.readBytes(keeper.uri), 5600);

    const kept = await keeper.finish();
    expect(kept).toEqual({ uri: keeper.uri, bytes: WAV_HEADER_BYTES + 5600 * 2, seconds: 0.4 });
    expectValidWav(fs.readBytes(keeper.uri), 5600);
    // finish() is idempotent and appends after it are ignored.
    keeper.append(frame(1600));
    expect(await keeper.finish()).toEqual(kept);
    expect(fs.sizeOf(keeper.uri)).toBe(kept!.bytes);
  });

  it("finish flushes the unflushed tail and reports whole-file bytes and seconds", async () => {
    const fs = new MemoryFs();
    const keeper = createLiveAudioKeeper({ fs, dir: DIR, sessionId: "s2", flushMs: 60_000 });
    for (let i = 0; i < 10; i++) keeper.append(frame(1600)); // 1.0 s
    keeper.append(frame(8000)); // +0.5 s
    expect(fs.exists(keeper.uri)).toBe(false);
    const kept = await keeper.finish();
    expect(kept).toEqual({ uri: keeper.uri, bytes: 44 + 2 * 24000, seconds: 1.5 });
    expectValidWav(fs.readBytes(keeper.uri), 24000);
  });

  it("copies frames so a reused buffer can't rewrite what was appended", async () => {
    const fs = new MemoryFs();
    const keeper = createLiveAudioKeeper({ fs, dir: DIR, sessionId: "s3", flushMs: 0 });
    const buf = new Int16Array(4).fill(7);
    keeper.append(buf);
    buf.fill(9);
    keeper.append(buf);
    await keeper.finish();
    const data = new Int16Array(fs.readBytes(keeper.uri).slice(WAV_HEADER_BYTES).buffer);
    expect(Array.from(data)).toEqual([7, 7, 7, 7, 9, 9, 9, 9]);
  });

  it("finish returns null (and leaves no file) when nothing was ever appended", async () => {
    const fs = new MemoryFs();
    const keeper = createLiveAudioKeeper({ fs, dir: DIR, sessionId: "s4" });
    keeper.append(new Int16Array(0));
    expect(await keeper.finish()).toBeNull();
    expect(fs.exists(keeper.uri)).toBe(false);
    expect(fs.listFileNames(DIR)).toEqual([]);
  });

  it("discard removes the file and stops further appends", async () => {
    const fs = new MemoryFs();
    const keeper = createLiveAudioKeeper({ fs, dir: DIR, sessionId: "s5", flushMs: 0 });
    keeper.append(frame(1600));
    expect(fs.exists(keeper.uri)).toBe(true);
    keeper.discard();
    expect(fs.exists(keeper.uri)).toBe(false);
    keeper.append(frame(1600));
    expect(fs.exists(keeper.uri)).toBe(false);
    expect(await keeper.finish()).toBeNull();
    // Discard after finish (the "uploaded, clean up" path) is fine too.
    const k2 = createLiveAudioKeeper({ fs, dir: DIR, sessionId: "s6", flushMs: 0 });
    k2.append(frame(100));
    await k2.finish();
    expect(fs.exists(k2.uri)).toBe(true);
    k2.discard();
    expect(fs.exists(k2.uri)).toBe(false);
  });

  it("a disk failure mid-session closes the keeper and finish rejects with the reason", async () => {
    const fs = new MemoryFs();
    const keeper = createLiveAudioKeeper({ fs, dir: DIR, sessionId: "s7", flushMs: 0 });
    keeper.append(frame(100));
    fs.appendBytes = () => {
      throw new Error("ENOSPC");
    };
    keeper.append(frame(100)); // must not throw into the audio callback
    await expect(keeper.finish()).rejects.toThrow(/ENOSPC/);
  });
});

describe("keepAudioStorageProblem", () => {
  it("refuses only when less than the recorder's minimum fits; unknown space is allowed", () => {
    expect(keepAudioStorageProblem(new MemoryFs())).toBeNull();
    expect(keepAudioStorageProblem(new MemoryFs({ freeBytes: null }))).toBeNull();
    // 1 MB ≈ 33 s of 16 kHz audio — under the 10-minute floor.
    expect(keepAudioStorageProblem(new MemoryFs({ freeBytes: 1_000_000 }))).toMatch(/Not enough storage/);
  });
});
