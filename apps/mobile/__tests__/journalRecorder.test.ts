/**
 * live/journalRecorder.ts + live/journalStore.ts over MemoryFs: kept
 * stretches land in a valid WAV with a JSON sidecar of timings; the file
 * rotates after `rotateSeconds` of listening and on Stop; each closed file
 * is uploaded once (title "Journal — <date> <start–end>") and deleted only
 * when the upload resolves; a failed upload keeps the file and retries it
 * at the next boundary; leftovers from an earlier run go up at Start.
 */
import {
  IDLE_JOURNAL_STATE,
  JournalRecorder,
  journalContext,
  journalTitle,
  type JournalState,
} from "../src/live/journalRecorder";
import { createJournalStore, journalStartFromName, type ClosedJournalFile } from "../src/live/journalStore";
import { EnergyVad } from "../src/live/vad";
import { SpeakerLabeler, type Embedder } from "../src/live/speakerId";
import { MemoryFs } from "../src/recorder/memoryFs";
import { WAV_HEADER_BYTES, buildWavHeader } from "../src/recorder/wav";
import { silenceInt16, toneInt16, unitVector } from "../src/live/testing/synth";

const SR = 16000;
const DIR = "file:///cache/journal";
const DIM = 192;
const SELF_VEC = unitVector(DIM, 0);
const BASE_WALL = Date.UTC(2026, 7, 30, 9, 0, 0);

class LoudIsSelf implements Embedder {
  async embed(pcm: Float32Array): Promise<Float32Array> {
    let acc = 0;
    for (let i = 0; i < pcm.length; i++) acc += pcm[i] * pcm[i];
    return Math.sqrt(acc / pcm.length) > 0.1 ? Float32Array.from(SELF_VEC) : unitVector(DIM, 1);
  }
}

interface Harness {
  fs: MemoryFs;
  recorder: JournalRecorder;
  uploads: ClosedJournalFile[];
  states: JournalState[];
  wall: { ms: number };
  feed(pcm: Int16Array): Promise<void>;
  /** Make the next N uploads reject. */
  failNext(n: number): void;
}

function harness(opts: { rotateSeconds?: number; selfPrint?: boolean } = {}): Harness {
  const fs = new MemoryFs();
  const wall = { ms: BASE_WALL };
  const store = createJournalStore({ fs, dir: DIR, now: () => wall.ms, flushMs: 0 });
  const uploads: ClosedJournalFile[] = [];
  const states: JournalState[] = [];
  let failures = 0;
  const labeler = new SpeakerLabeler(
    opts.selfPrint === false
      ? []
      : [{ personId: "self", displayName: "You", isSelf: true, embedding: SELF_VEC, settings: 2 }],
  );
  const recorder = new JournalRecorder({
    vad: new EnergyVad(-45, 0.032),
    embedder: new LoudIsSelf(),
    labeler,
    store,
    upload: async (file) => {
      if (failures > 0) {
        failures -= 1;
        throw new Error("network down");
      }
      uploads.push(file);
    },
    onState: (s) => states.push(s),
    now: () => wall.ms,
    rotateSeconds: opts.rotateSeconds ?? 20,
  });
  let fed = 0;
  return {
    fs,
    recorder,
    uploads,
    states,
    wall,
    async feed(pcm) {
      for (let off = 0; off < pcm.length; off += 1600) {
        const chunk = pcm.subarray(off, Math.min(off + 1600, pcm.length));
        fed += chunk.length;
        wall.ms = BASE_WALL + (fed / SR) * 1000;
        recorder.pushSamples(chunk);
        await recorder.settle();
      }
    },
    failNext(n) {
      failures = n;
    },
  };
}

/** Self speech at t, then silence, as one contiguous feed. */
function selfThenSilence(speechSeconds: number, silenceSeconds: number): Int16Array {
  const a = toneInt16(speechSeconds, -10);
  const b = silenceInt16(silenceSeconds);
  const out = new Int16Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function u32(bytes: Uint8Array, offset: number): number {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(offset, true);
}

let warnSpy: jest.SpyInstance;
beforeEach(() => {
  // The failed-upload path warns on purpose; keep the test output quiet.
  warnSpy = jest.spyOn(console, "warn").mockImplementation(() => {});
});
afterEach(() => {
  warnSpy.mockRestore();
});

describe("JournalRecorder", () => {
  it("writes kept stretches to a valid WAV with a timed sidecar and uploads it on Stop", async () => {
    const h = harness();
    expect(h.recorder.hasSelfPrint).toBe(true);
    await h.recorder.start();
    await h.recorder.uploadsSettled();
    expect(h.recorder.stateSnapshot.status).toBe("listening");
    await h.feed(silenceInt16(1.0));
    await h.feed(selfThenSilence(2.0, 3.0)); // owner at 1.0–3.0
    await h.feed(selfThenSilence(1.5, 3.0)); // owner at 6.0–7.5
    await h.recorder.settle();

    const live = h.recorder.stateSnapshot;
    expect(live.selfCount).toBe(2);
    expect(live.selfSeconds).toBeCloseTo(3.5, 0);
    expect(live.lastSelfAt).not.toBeNull();
    expect(live.fileBytes).toBeGreaterThan(WAV_HEADER_BYTES);
    expect(live.fileStartedAt).toBe(BASE_WALL);
    const open = h.fs.listFileNames(DIR);
    expect(open).toEqual([`journal-${BASE_WALL}.json`, `journal-${BASE_WALL}.wav`]);

    const final = await h.recorder.stop();
    await h.recorder.uploadsSettled();
    expect(final.status).toBe("stopped");
    expect(final.filesClosed).toBe(1);
    expect(h.uploads).toHaveLength(1);
    const file = h.uploads[0];
    expect(file.uri).toBe(`${DIR}/journal-${BASE_WALL}.wav`);
    expect(file.startedAt).toBe(new Date(BASE_WALL).toISOString());
    // Two chunks: (1 + 2 + 1) s and (1 + 1.5 + 1) s ≈ 7.5 s of audio.
    expect(file.seconds).toBeCloseTo(7.5, 0);
    expect((file.bytes - WAV_HEADER_BYTES) / 2 / SR).toBeCloseTo(file.seconds, 0);
    expect(file.segments).toHaveLength(2);
    const [s0, s1] = file.segments!;
    expect(s0.offset_s).toBe(0);
    expect(s0.duration_s).toBeCloseTo(4.0, 0);
    expect(s0.lead_s).toBeCloseTo(1.0, 1);
    expect(s0.speech_s).toBeCloseTo(2.0, 1);
    expect(s0.basis).toBe("absolute");
    expect(s0.score).toBeGreaterThanOrEqual(0.65);
    expect(new Date(s0.start_wall_iso).getTime()).toBeGreaterThan(BASE_WALL + 800);
    expect(new Date(s0.start_wall_iso).getTime()).toBeLessThan(BASE_WALL + 1300);
    // The second chunk starts where the first ended in the file.
    expect(s1.offset_s).toBeCloseTo(s0.duration_s, 1);
    expect(s1.speech_s).toBeCloseTo(1.5, 1);
    expect(s1.start_wall_iso > s0.start_wall_iso).toBe(true);

    // The upload saw the file while it was still on disk; it is gone now.
    expect(h.recorder.stateSnapshot.uploads).toMatchObject({ sent: 1, pending: 0, failed: 0, inFlight: false });
    expect(h.fs.listFileNames(DIR)).toEqual([]);
  });

  it("the WAV on disk is valid at every instant and the sidecar mirrors it", async () => {
    const h = harness();
    await h.recorder.start();
    await h.feed(silenceInt16(0.5));
    await h.feed(selfThenSilence(1.2, 2.0));
    await h.recorder.settle();
    const wav = h.fs.readBytes(`${DIR}/journal-${BASE_WALL}.wav`);
    expect(String.fromCharCode(...wav.subarray(0, 4))).toBe("RIFF");
    expect(u32(wav, 40)).toBe(wav.byteLength - WAV_HEADER_BYTES);
    expect(u32(wav, 24)).toBe(SR);
    const sidecar = JSON.parse(h.fs.readText(`${DIR}/journal-${BASE_WALL}.json`));
    expect(sidecar).toMatchObject({ version: 1, started_at: new Date(BASE_WALL).toISOString(), ended_at: null, sample_rate: SR });
    expect(sidecar.segments).toHaveLength(1);
    expect(sidecar.segments[0].duration_s * SR * 2).toBeCloseTo(wav.byteLength - WAV_HEADER_BYTES, -2);
    await h.recorder.stop();
    await h.recorder.uploadsSettled();
  });

  it("rotates after rotateSeconds of listening: closes, uploads, opens a fresh file", async () => {
    const h = harness({ rotateSeconds: 20 });
    await h.recorder.start();
    await h.feed(silenceInt16(1.0));
    await h.feed(selfThenSilence(2.0, 16.0)); // owner at 1–3; 19 s in
    await h.recorder.settle();
    expect(h.uploads).toHaveLength(0);
    await h.feed(silenceInt16(2.0)); // crosses 20 s → rotation
    await h.recorder.settle();
    await h.recorder.uploadsSettled();
    expect(h.uploads).toHaveLength(1);
    expect(h.uploads[0].segments).toHaveLength(1);
    expect(h.uploads[0].endedAt).toBe(new Date(BASE_WALL + 20000).toISOString());
    expect(h.recorder.stateSnapshot.filesClosed).toBe(1);
    expect(h.recorder.stateSnapshot.fileStartedAt).toBe(BASE_WALL + 20000);
    expect(h.recorder.stateSnapshot.status).toBe("listening");
    // The first file is gone; the new one is the only thing on disk (its
    // header is written on the first kept stretch).
    expect(h.fs.listFileNames(DIR)).toEqual([]);

    await h.feed(silenceInt16(1.0));
    await h.feed(selfThenSilence(1.5, 2.0)); // owner in the SECOND file
    await h.recorder.settle();
    expect(h.recorder.stateSnapshot.selfCount).toBe(2); // counters span files
    expect(h.fs.listFileNames(DIR)).toEqual([`journal-${BASE_WALL + 20000}.json`, `journal-${BASE_WALL + 20000}.wav`]);
    const final = await h.recorder.stop();
    await h.recorder.uploadsSettled();
    expect(final.filesClosed).toBe(2);
    expect(h.uploads).toHaveLength(2);
    expect(h.uploads[1].uri).toBe(`${DIR}/journal-${BASE_WALL + 20000}.wav`);
    expect(h.uploads[1].segments).toHaveLength(1);
    expect(h.recorder.stateSnapshot.uploads.sent).toBe(2);
    expect(h.fs.listFileNames(DIR)).toEqual([]);
  });

  it("a file with nothing kept is not uploaded (no empty recordings)", async () => {
    const h = harness({ rotateSeconds: 5 });
    await h.recorder.start();
    await h.feed(silenceInt16(6.0)); // rotation with an empty file
    await h.recorder.settle();
    await h.recorder.uploadsSettled();
    expect(h.uploads).toHaveLength(0);
    expect(h.fs.listFileNames(DIR)).toEqual([]);
    const final = await h.recorder.stop();
    await h.recorder.uploadsSettled();
    expect(final.filesClosed).toBe(2);
    expect(final.uploads).toMatchObject({ sent: 0, pending: 0, failed: 0 });
    expect(h.uploads).toHaveLength(0);
  });

  it("keeps a file whose upload failed and retries it at the next boundary", async () => {
    const h = harness({ rotateSeconds: 10 });
    await h.recorder.start();
    h.failNext(1);
    await h.feed(silenceInt16(1.0));
    await h.feed(selfThenSilence(2.0, 8.0)); // rotation at 11 s → upload fails
    await h.recorder.settle();
    await h.recorder.uploadsSettled();
    expect(h.uploads).toHaveLength(0);
    expect(h.recorder.stateSnapshot.uploads).toMatchObject({ sent: 0, pending: 1, failed: 1, lastError: "network down" });
    // Still on disk, both halves.
    expect(h.fs.listFileNames(DIR)).toEqual([`journal-${BASE_WALL}.json`, `journal-${BASE_WALL}.wav`]);

    await h.feed(silenceInt16(1.0));
    await h.feed(selfThenSilence(1.5, 2.0));
    await h.recorder.settle();
    await h.recorder.stop(); // next boundary: both files go up, oldest first
    await h.recorder.uploadsSettled();
    expect(h.uploads.map((f) => f.uri)).toEqual([
      `${DIR}/journal-${BASE_WALL}.wav`,
      `${DIR}/journal-${BASE_WALL + 10000}.wav`,
    ]);
    expect(h.recorder.stateSnapshot.uploads).toMatchObject({ sent: 2, pending: 0, failed: 1 });
    expect(h.fs.listFileNames(DIR)).toEqual([]);
  });

  it("uploads leftovers from an earlier run at Start (a crash loses nothing)", async () => {
    const h = harness();
    // A file left behind: a valid header + 2 s of PCM, no sidecar.
    const pcm = new Uint8Array(2 * SR * 2);
    const header = buildWavHeader(SR, pcm.length);
    h.fs.ensureDir(DIR);
    h.fs.writeBytes(`${DIR}/journal-1788170000000.wav`, header);
    h.fs.appendBytes(`${DIR}/journal-1788170000000.wav`, pcm);
    // …and one that is just a header (nothing kept): swept, never uploaded.
    h.fs.writeBytes(`${DIR}/journal-1788169000000.wav`, buildWavHeader(SR, 0));
    await h.recorder.start();
    await h.recorder.uploadsSettled();
    expect(h.uploads).toHaveLength(1);
    expect(h.uploads[0]).toMatchObject({
      uri: `${DIR}/journal-1788170000000.wav`,
      seconds: 2,
      segments: null,
      startedAt: new Date(1788170000000).toISOString(),
    });
    expect(h.fs.listFileNames(DIR)).toEqual([]);
    await h.recorder.stop();
  });

  it("reports no self print honestly (the hook refuses to start on it)", () => {
    const h = harness({ selfPrint: false });
    expect(h.recorder.hasSelfPrint).toBe(false);
    expect(h.recorder.stateSnapshot).toEqual(IDLE_JOURNAL_STATE);
  });

  it("names the upload after the day and the window, and puts the timings in the context", () => {
    const file: ClosedJournalFile = {
      uri: `${DIR}/journal-1.wav`,
      sidecarUri: `${DIR}/journal-1.json`,
      startedAt: new Date(2026, 7, 30, 9, 0, 0).toISOString(),
      endedAt: new Date(2026, 7, 30, 9, 30, 0).toISOString(),
      bytes: 1000,
      seconds: 0.5,
      segments: [
        { start_wall_iso: "2026-08-30T09:03:55.000Z", offset_s: 0, duration_s: 4, lead_s: 1, speech_s: 2, score: 0.7, basis: "absolute" },
        { start_wall_iso: "2026-08-30T09:12:01.000Z", offset_s: 4, duration_s: 3.5, lead_s: 1, speech_s: 1.5, score: 0.5, basis: "contrast" },
      ],
    };
    expect(journalTitle(file)).toBe("Journal — 2026-08-30 09:00–09:30");
    const context = journalContext(file);
    expect(context).toContain("source: journal");
    expect(context).toContain("2 stretches");
    expect(context).toContain("Offsets: 0s@09:03:55, 4s@09:12:01");
    expect(context.length).toBeLessThanOrEqual(500);
    // Many segments: the context never exceeds the server's 500-char cap.
    const many = { ...file, segments: Array.from({ length: 200 }, (_, i) => ({ ...file.segments![0], offset_s: i * 5 })) };
    expect(journalContext(many).length).toBeLessThanOrEqual(500);
    expect(journalStartFromName(`journal-${BASE_WALL}.wav`)).toBe(BASE_WALL);
    expect(journalStartFromName("session.wav")).toBeNull();
  });
});
