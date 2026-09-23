import { MemoryFs } from "../src/recorder/memoryFs";
import { RecorderSessionStore } from "../src/recorder/sessionStore";

/** The signed-in uid these fixtures record as. Recorder files are OWNED:
 *  a store built without an owner records nothing claimable and offers
 *  nothing back (see RecorderSessionStore). */
const OWNER = "uid-owner";

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
    const store = new RecorderSessionStore(fs, OWNER);
    const manifest = store.createSession(WAV_SESSION);
    expect(manifest.sessionId).toBeTruthy();
    expect(manifest.segments).toEqual([]);
    // A fresh store over the same fs can see it (it's on disk, not in memory).
    const again = new RecorderSessionStore(fs, OWNER);
    // No segments yet — not recoverable, but the dir exists until cleaned.
    expect(again.listRecoverable()).toEqual([]);
  });

  it("finalizeSegment moves the recorder file into the session and records it", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs, OWNER);
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
    const store = new RecorderSessionStore(fs, OWNER);
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
    const store = new RecorderSessionStore(fs, OWNER);
    let m = store.createSession(WAV_SESSION);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, 1), 300_000);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, 2), 300_000);
    // CRASH: no finish, no cleanup. A new launch scans the same disk.

    const relaunched = new RecorderSessionStore(fs, OWNER);
    const found = relaunched.listRecoverable();
    expect(found).toHaveLength(1);
    expect(found[0].segmentCount).toBe(2);
    expect(found[0].totalDurationMs).toBe(600_000);
    expect(found[0].manifest.sessionId).toBe(m.sessionId);
  });

  it("recovering stitches the segments in order into one audio file and cleans up", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs, OWNER);
    let m = store.createSession(WAV_SESSION);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, 11), 1000);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, 22), 1000);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, 33), 1000);

    const relaunched = new RecorderSessionStore(fs, OWNER);
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
    const store = new RecorderSessionStore(fs, OWNER);
    store.createSession(WAV_SESSION); // crash before the first rotation

    const relaunched = new RecorderSessionStore(fs, OWNER);
    expect(relaunched.listRecoverable()).toEqual([]);
    // Scanning twice stays clean (the empty dir was removed).
    expect(relaunched.listRecoverable()).toEqual([]);
  });

  it("returns nothing when no sessions exist", () => {
    const fs = new MemoryFs();
    expect(new RecorderSessionStore(fs, OWNER).listRecoverable()).toEqual([]);
  });

  it("discard removes the session without producing a file", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs, OWNER);
    let m = store.createSession(WAV_SESSION);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, 1), 1000);

    store.discard(m.sessionId);
    expect(store.listRecoverable()).toEqual([]);
  });

  it("survives a corrupt manifest without blocking other sessions", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs, OWNER);
    let good = store.createSession(WAV_SESSION);
    good = store.finalizeSegment(good, fakeRecorderOutput(fs, 5), 1000);
    // A second session whose manifest got mangled mid-write.
    const bad = store.createSession(WAV_SESSION);
    fs.writeText(
      `${fs.documentDirUri()}/recorder-sessions/${bad.sessionId}/manifest.json`,
      "{ not json",
    );

    const found = new RecorderSessionStore(fs, OWNER).listRecoverable();
    expect(found).toHaveLength(1);
    expect(found[0].manifest.sessionId).toBe(good.sessionId);
  });
});

describe("RecorderSessionStore — storage facts", () => {
  it("reports the filesystem's free bytes", () => {
    const fs = new MemoryFs({ freeBytes: 123456 });
    expect(new RecorderSessionStore(fs, OWNER).freeBytes()).toBe(123456);
  });

  it("reports null when free space is unknown", () => {
    const fs = new MemoryFs({ freeBytes: null });
    expect(new RecorderSessionStore(fs, OWNER).freeBytes()).toBeNull();
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
    const store = new RecorderSessionStore(fs, OWNER);
    const file = seedFinished(fs, store);
    // A FRESH store over the same disk (app restart) must still see it.
    const again = new RecorderSessionStore(fs, OWNER);
    const orphans = again.listOrphanStitched();
    expect(orphans).toHaveLength(1);
    expect(orphans[0].uri).toBe(file.uri);
    expect(orphans[0].mimeType).toBe("audio/wav");
    expect(orphans[0].size).toBeGreaterThan(0);
  });

  it("returns empty when nothing was ever stitched", () => {
    const store = new RecorderSessionStore(new MemoryFs(), OWNER);
    expect(store.listOrphanStitched()).toEqual([]);
  });

it("a new stitch KEEPS recent unconsumed outputs (no more destroy-on-stitch)", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs, OWNER);
    const first = seedFinished(fs, store);
    const second = seedFinished(fs, store);
    const uris = store.listOrphanStitched().map((o) => o.uri);
    expect(uris).toContain(first.uri);
    expect(uris).toContain(second.uri);
  });

  it("discardOrphan deletes exactly that file", () => {
    const fs = new MemoryFs();
    const store = new RecorderSessionStore(fs, OWNER);
    const file = seedFinished(fs, store);
    store.discardOrphan(file.uri);
    expect(store.listOrphanStitched()).toEqual([]);
    expect(fs.exists(file.uri)).toBe(false);
  });
});

/**
 * Owner isolation — the 2026-09-23 privacy bug.
 *
 * Record as guest A → Settings → Log out → "Continue as guest" (a NEW
 * anonymous uid, guest B) → Analyze a Conversation, and guest B was offered
 * guest A's recording ("Found a recovered recording that was never analyzed
 * (1 MB) — use it?") and could upload it under their own account. Logging out
 * clears the Firebase session; it does not clear the document directory.
 *
 * Guest accounts have a real uid and `email === null`, so ownership is keyed
 * on uid throughout. Nothing here is fixed by deleting files: A must still
 * find their recording after logging out and back into the same account.
 */
describe("RecorderSessionStore — owner isolation", () => {
  const A = "uid-guest-a";
  const B = "uid-guest-b";

  /** An interrupted session (one finished segment) recorded by `owner`. */
  function seedSession(fs: MemoryFs, owner: string | null, tag = 1) {
    const store = new RecorderSessionStore(fs, owner);
    let m = store.createSession(WAV_SESSION);
    m = store.finalizeSegment(m, fakeRecorderOutput(fs, tag), 60_000);
    return { store, manifest: m };
  }

  /** A stitched-but-never-uploaded output (the exact thing the bug offered). */
  function seedStitched(fs: MemoryFs, owner: string | null, tag = 1) {
    const { store, manifest } = seedSession(fs, owner, tag);
    return store.finishToFile(manifest);
  }

  it("stamps the recording account onto the manifest, and keeps it there", () => {
    const fs = new MemoryFs();
    const { manifest } = seedSession(fs, A);
    expect(manifest.ownerUid).toBe(A);
    // The manifest is rewritten after every finalized segment — the owner has
    // to survive those rewrites as durably as the segment list does.
    const onDisk = JSON.parse(
      fs.readText(
        `file:///doc/recorder-sessions/${manifest.sessionId}/manifest.json`,
      ),
    );
    expect(onDisk.ownerUid).toBe(A);
  });

  it("does not offer guest A's unfinished session to guest B", () => {
    const fs = new MemoryFs();
    seedSession(fs, A);
    expect(new RecorderSessionStore(fs, B).listRecoverable()).toEqual([]);
  });

  it("still offers guest A's session back to guest A after a sign-out", () => {
    const fs = new MemoryFs();
    seedSession(fs, A);
    // Signed out and back into the SAME account: the recording is waiting.
    const asA = new RecorderSessionStore(fs, A);
    expect(asA.listRecoverable()).toHaveLength(1);
    expect(asA.listRecoverable()[0].segmentCount).toBe(1);
  });

  it("leaves another account's session untouched on disk", () => {
    const fs = new MemoryFs();
    const { manifest } = seedSession(fs, A);
    const dir = `file:///doc/recorder-sessions/${manifest.sessionId}`;
    new RecorderSessionStore(fs, B).listRecoverable();
    // Isolation is about visibility, never deletion — B's scan must not have
    // swept A's audio away.
    expect(fs.exists(`${dir}/manifest.json`)).toBe(true);
    expect(fs.exists(`${dir}/seg-000.wav`)).toBe(true);
  });

  it("offers an UNOWNED legacy session to nobody", () => {
    const fs = new MemoryFs();
    // Written before ownership existed: there is no way to know whose it is,
    // so it is unclaimable rather than adopted by whoever opens the app next.
    seedSession(fs, null);
    expect(new RecorderSessionStore(fs, A).listRecoverable()).toEqual([]);
    expect(new RecorderSessionStore(fs, B).listRecoverable()).toEqual([]);
    expect(new RecorderSessionStore(fs).listRecoverable()).toEqual([]);
  });

  it("does not offer guest A's stitched, never-uploaded file to guest B", () => {
    // The screenshot from the report: "Found a recovered recording that was
    // never analyzed (1 MB) — use it?" shown to the NEXT guest.
    const fs = new MemoryFs();
    seedStitched(fs, A);
    expect(new RecorderSessionStore(fs, B).listOrphanStitched()).toEqual([]);
  });

  it("still offers guest A's stitched file back to guest A", () => {
    const fs = new MemoryFs();
    const file = seedStitched(fs, A);
    const mine = new RecorderSessionStore(fs, A).listOrphanStitched();
    expect(mine.map((o) => o.uri)).toEqual([file.uri]);
  });

  it("offers an UNOWNED legacy stitched file to nobody", () => {
    const fs = new MemoryFs();
    seedStitched(fs, null);
    expect(new RecorderSessionStore(fs, A).listOrphanStitched()).toEqual([]);
    expect(new RecorderSessionStore(fs, B).listOrphanStitched()).toEqual([]);
    expect(new RecorderSessionStore(fs).listOrphanStitched()).toEqual([]);
  });

  it("keeps two accounts' stitched files apart on one phone", () => {
    const fs = new MemoryFs();
    const a = seedStitched(fs, A, 1);
    const b = seedStitched(fs, B, 2);
    expect(
      new RecorderSessionStore(fs, A).listOrphanStitched().map((o) => o.uri),
    ).toEqual([a.uri]);
    expect(
      new RecorderSessionStore(fs, B).listOrphanStitched().map((o) => o.uri),
    ).toEqual([b.uri]);
    // Neither scan deleted the other's audio.
    expect(fs.exists(a.uri)).toBe(true);
    expect(fs.exists(b.uri)).toBe(true);
  });

  it("never mistakes an owner sidecar for a recording of its own", () => {
    const fs = new MemoryFs();
    const file = seedStitched(fs, A);
    const asA = new RecorderSessionStore(fs, A);
    expect(asA.listOrphanStitched()).toHaveLength(1);
    // Discarding takes the sidecar with it — no stranded owner records.
    asA.discardOrphan(file.uri);
    expect(fs.exists(`${file.uri}.owner`)).toBe(false);
    expect(asA.listOrphanStitched()).toEqual([]);
  });

  it("reads the live uid, so a sign-out under a long-lived store is seen", () => {
    // Production passes a thunk over the auth store: the store object outlives
    // the sign-out that changes what it may offer.
    const fs = new MemoryFs();
    let uid: string | null = A;
    const store = new RecorderSessionStore(fs, () => uid);
    const m = store.createSession(WAV_SESSION);
    store.finalizeSegment(m, fakeRecorderOutput(fs, 3), 60_000);
    expect(store.listRecoverable()).toHaveLength(1);
    uid = B; // logged out, continued as a new guest — same store object
    expect(store.listRecoverable()).toEqual([]);
    uid = A;
    expect(store.listRecoverable()).toHaveLength(1);
  });
});
