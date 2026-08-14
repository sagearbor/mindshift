import { MemoryFs } from "../src/recorder/memoryFs";
import { RecorderSessionStore } from "../src/recorder/sessionStore";

const WAV_SESSION = {
  format: "wav" as const,
  extension: ".wav",
  mimeType: "audio/wav",
  segmentSeconds: 300,
};

/** Minimal valid WAV whose single sample tags which segment it was. */
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

/** Read the tag samples back out of a stitched WAV. */
function wavTags(bytes: Uint8Array): number[] {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const dataLen = v.getUint32(40, true);
  const tags: number[] = [];
  for (let i = 0; i < dataLen; i += 2) tags.push(v.getInt16(44 + i, true));
  return tags;
}

/** Simulate the recorder having produced a finished segment file in cache. */
function fakeRecorderOutput(fs: MemoryFs, n: number): string {
  const uri = `file:///cache/recording-${n}.wav`;
  fs.writeBytes(uri, wavBytes(n));
  return uri;
}

describe("RecorderSessionStore — segment persistence", () => {
  it("createSession creates a session directory with a manifest", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs);
    const manifest = store.createSession(WAV_SESSION);
    expect(manifest.sessionId).toBeTruthy();
    expect(manifest.segments).toEqual([]);
    // A fresh store over the same fs can see it (it's on disk, not in memory).
    const again = new RecorderSessionStore(fs);
    // No segments yet — not recoverable, but the dir exists until cleaned.
    expect(again.listRecoverable()).toEqual([]);
  });

  it("finalizeSegment moves the recorder file into the session and records it", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs);
    let manifest = store.createSession(WAV_SESSION);
    const src = fakeRecorderOutput(fs, 1);

    manifest = store.finalizeSegment(manifest, src, 300_000);

    expect(manifest.segments).toHaveLength(1);
    expect(manifest.segments[0].index).toBe(0);
    expect(manifest.segments[0].durationMs).toBe(300_000);
    expect(manifest.segments[0].bytes).toBe(wavBytes(1).byteLength);
    // The source file was MOVED, not copied — no stray cache files.
    expect(fs.exists(src)).toBe(false);
  });

  it("assigns increasing segment indexes", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs);
    let m = store.createSession(WAV_SESSION);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, 1), 1000);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, 2), 2000);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, 3), 3000);
    expect(m.segments.map((s) => s.index)).toEqual([0, 1, 2]);
  });
});

describe("RecorderSessionStore — crash recovery", () => {
  it("a crashed session (segments on disk, never finished) is recoverable by a fresh store", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs);
    let m = store.createSession(WAV_SESSION);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, 1), 300_000);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, 2), 300_000);
    // CRASH: no finish, no cleanup. A new launch scans the same disk.

    const relaunched = new RecorderSessionStore(fs);
    const found = relaunched.listRecoverable();
    expect(found).toHaveLength(1);
    expect(found[0].segmentCount).toBe(2);
    expect(found[0].totalDurationMs).toBe(600_000);
    expect(found[0].manifest.sessionId).toBe(m.sessionId);
  });

  it("recovering stitches the segments in order into one audio file and cleans up", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs);
    let m = store.createSession(WAV_SESSION);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, 11), 1000);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, 22), 1000);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, 33), 1000);

    const relaunched = new RecorderSessionStore(fs);
    const [found] = relaunched.listRecoverable();
    const file = relaunched.finishToFile(found.manifest);

    expect(file.mimeType).toBe("audio/wav");
    expect(file.name).toMatch(/\.wav$/);
    expect(file.size).toBeGreaterThan(0);
    expect(wavTags(fs.readBytes(file.uri))).toEqual([11, 22, 33]);
    // The session directory is gone — nothing left to re-prompt about.
    expect(relaunched.listRecoverable()).toEqual([]);
  });

  it("ignores and cleans up sessions with zero finished segments", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs);
    store.createSession(WAV_SESSION); // crash before the first rotation

    const relaunched = new RecorderSessionStore(fs);
    expect(relaunched.listRecoverable()).toEqual([]);
    // Scanning twice stays clean (the empty dir was removed).
    expect(relaunched.listRecoverable()).toEqual([]);
  });

  it("returns nothing when no sessions exist", () => {
    const fs = new MemoryFs();
    expect(new RecorderSessionStore(fs).listRecoverable()).toEqual([]);
  });

  it("discard removes the session without producing a file", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs);
    let m = store.createSession(WAV_SESSION);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, 1), 1000);

    store.discard(m.sessionId);
    expect(store.listRecoverable()).toEqual([]);
  });

  it("survives a corrupt manifest without blocking other sessions", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs);
    let good = store.createSession(WAV_SESSION);
    good = store.finalizeSegment(good, fakeRecorderOutput(fs, 5), 1000);
    // A second session whose manifest got mangled mid-write.
    const bad = store.createSession(WAV_SESSION);
    fs.writeText(
      `${fs.documentDirUri()}/recorder-sessions/${bad.sessionId}/manifest.json`,
      "{ not json",
    );

    const found = new RecorderSessionStore(fs).listRecoverable();
    expect(found).toHaveLength(1);
    expect(found[0].manifest.sessionId).toBe(good.sessionId);
  });
});

describe("RecorderSessionStore — storage facts", () => {
  it("reports the filesystem's free bytes", () => {
    const fs = new MemoryFs({ freeBytes: 123456 });
    expect(new RecorderSessionStore(fs).freeBytes()).toBe(123456);
  });

  it("reports null when free space is unknown", () => {
    const fs = new MemoryFs({ freeBytes: null });
    expect(new RecorderSessionStore(fs).freeBytes()).toBeNull();
  });
});

describe("RecorderSessionStore — orphaned stitched outputs", () => {
  function seedFinished(fs: MemoryFs, store: RecorderSessionStore) {
    let m = store.createSession(WAV_SESSION);
    const uri = fakeRecorderOutput(fs, 9);
    m = store.finalizeSegment(m, uri, 60_000);
    return store.finishToFile(m);
  }

  it("listOrphanStitched finds a finished file a later launch never consumed", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs);
    const file = seedFinished(fs, store);
    // A FRESH store over the same disk (app restart) must still see it.
    const again = new RecorderSessionStore(fs);
    const orphans = again.listOrphanStitched();
    expect(orphans).toHaveLength(1);
    expect(orphans[0].uri).toBe(file.uri);
    expect(orphans[0].mimeType).toBe("audio/wav");
    expect(orphans[0].size).toBeGreaterThan(0);
  });

  it("returns empty when nothing was ever stitched", () => {
    const store = new RecorderSessionStore(new MemoryFs());
    expect(store.listOrphanStitched()).toEqual([]);
  });

it("a new stitch KEEPS recent unconsumed outputs (no more destroy-on-stitch)", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs);
    const first = seedFinished(fs, store);
    const second = seedFinished(fs, store);
    const uris = store.listOrphanStitched().map((o) => o.uri);
    expect(uris).toContain(first.uri);
    expect(uris).toContain(second.uri);
  });

  it("discardOrphan deletes exactly that file", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs);
    const file = seedFinished(fs, store);
    store.discardOrphan(file.uri);
    expect(store.listOrphanStitched()).toEqual([]);
    expect(fs.exists(file.uri)).toBe(false);
  });
});
