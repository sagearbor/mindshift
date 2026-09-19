import { Platform } from "react-native";
import {
  RECORDINGS_CACHE_KEY_PREFIX,
  clearRecordingsCache,
  isRecordingsListStale,
  markRecordingsListStale,
  mergeRecordingsList,
  readRecordingsCache,
  recordingsCacheKey,
  resetRecordingsCacheMemory,
  writeRecordingsCache,
} from "../src/utils/recordingsListCache";
import type { RecordingSummary } from "../src/api/client";

// Local override of the shared expo-file-system mock (same pattern
// mediaCache.test.ts uses): this module needs Directory/Paths.cache plus the
// SDK-57 SYNCHRONOUS File.textSync()/write()/delete(), which the jest-setup
// wholesale mock doesn't provide. `__mockFsFiles` is a uri → text map the
// tests inspect/seed to stand in for the on-disk cache files.
jest.mock("expo-file-system", () => {
  const files = new Map<string, string>();
  (globalThis as Record<string, unknown>).__mockFsFiles = files;
  class Directory {
    uri: string;
    constructor(base: { uri: string } | string, name?: string) {
      const baseUri = typeof base === "string" ? base : base.uri;
      this.uri = name ? `${baseUri}/${name}` : baseUri;
    }
    get exists() {
      return true;
    }
    create() {}
  }
  class File {
    uri: string;
    constructor(source: { uri: string } | string, name?: string) {
      const baseUri = typeof source === "string" ? source : source.uri;
      this.uri = name ? `${baseUri}/${name}` : baseUri;
    }
    get exists() {
      return files.has(this.uri);
    }
    textSync() {
      const t = files.get(this.uri);
      if (t === undefined) throw new Error("ENOENT");
      return t;
    }
    write(text: string) {
      files.set(this.uri, text);
    }
    delete() {
      files.delete(this.uri);
    }
  }
  return {
    __esModule: true,
    Directory,
    File,
    Paths: { cache: { uri: "file:///cache" } },
  };
});
require("expo-file-system");
const files = (globalThis as Record<string, unknown>).__mockFsFiles as Map<
  string,
  string
>;

const originalOS = Platform.OS;
function setPlatform(os: string) {
  Object.defineProperty(Platform, "OS", { value: os, configurable: true });
}

const rec = (id: string, extra: Partial<RecordingSummary> = {}): RecordingSummary => ({
  id,
  created_at: "2026-07-01T10:00:00Z",
  filename: `${id}.m4a`,
  media_type: "audio",
  duration_seconds: 10,
  has_analysis: true,
  ...extra,
});

beforeEach(() => {
  resetRecordingsCacheMemory();
  files.clear();
  setPlatform("ios");
});
afterAll(() => setPlatform(originalOS));

describe("recordingsCacheKey", () => {
  it("is per-account, prefixed, and sanitized", () => {
    expect(recordingsCacheKey("uid123")).toBe(`${RECORDINGS_CACHE_KEY_PREFIX}.uid123`);
    expect(recordingsCacheKey(null)).toBe(`${RECORDINGS_CACHE_KEY_PREFIX}.anon`);
    expect(recordingsCacheKey("a@b.c/d")).toBe(`${RECORDINGS_CACHE_KEY_PREFIX}.a_b.c_d`);
  });
});

describe("recordingsListCache (native)", () => {
  it("round-trips a list through disk, with fetched_at", () => {
    const written = writeRecordingsCache(
      "u1",
      { recordings: [rec("r1")], sharedWithMe: [] },
      1_000,
    );
    expect(written.fetched_at).toBe(1_000);
    // On disk under Paths.cache/recordings-list/<key>.json
    const uri = `file:///cache/recordings-list/${recordingsCacheKey("u1")}.json`;
    expect(files.has(uri)).toBe(true);

    // Drop the memory copy so the read genuinely comes from "disk".
    resetRecordingsCacheMemory();
    const read = readRecordingsCache("u1");
    expect(read).toEqual({
      recordings: [rec("r1")],
      sharedWithMe: [],
      fetched_at: 1_000,
    });
  });

  it("keeps accounts separate", () => {
    writeRecordingsCache("u1", { recordings: [rec("mine")], sharedWithMe: [] }, 1);
    writeRecordingsCache("u2", { recordings: [rec("theirs")], sharedWithMe: [] }, 2);
    resetRecordingsCacheMemory();
    expect(readRecordingsCache("u1")?.recordings.map((r) => r.id)).toEqual(["mine"]);
    expect(readRecordingsCache("u2")?.recordings.map((r) => r.id)).toEqual(["theirs"]);
    expect(readRecordingsCache("u3")).toBeNull();
    expect(readRecordingsCache(null)).toBeNull();
  });

  it("reads a corrupt blob as empty and drops it", () => {
    const uri = `file:///cache/recordings-list/${recordingsCacheKey("u1")}.json`;
    files.set(uri, "{not json");
    expect(readRecordingsCache("u1")).toBeNull();
    expect(files.has(uri)).toBe(false);

    // Valid JSON of the wrong shape is corrupt too.
    files.set(uri, JSON.stringify({ recordings: "nope", fetched_at: 1 }));
    expect(readRecordingsCache("u1")).toBeNull();
    expect(files.has(uri)).toBe(false);
  });

  it("a storage failure reads as no cache and never throws on write", () => {
    // Simulate a broken file system: File constructor throws.
    const fs = require("expo-file-system");
    const RealFile = fs.File;
    fs.File = class {
      constructor() {
        throw new Error("disk on fire");
      }
    };
    try {
      expect(() =>
        writeRecordingsCache("u1", { recordings: [rec("r1")], sharedWithMe: [] }, 5),
      ).not.toThrow();
      // The memory copy still serves this process…
      expect(readRecordingsCache("u1")?.recordings[0].id).toBe("r1");
      // …but a cold read (no memory) is an honest miss.
      resetRecordingsCacheMemory();
      expect(readRecordingsCache("u1")).toBeNull();
    } finally {
      fs.File = RealFile;
    }
  });

  it("clearRecordingsCache drops memory + disk", () => {
    writeRecordingsCache("u1", { recordings: [rec("r1")], sharedWithMe: [] }, 1);
    clearRecordingsCache("u1");
    expect(readRecordingsCache("u1")).toBeNull();
    expect(files.size).toBe(0);
  });
});

describe("recordingsListCache (web)", () => {
  it("uses localStorage on web", () => {
    setPlatform("web");
    const store = new Map<string, string>();
    const ls = {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
      removeItem: (k: string) => void store.delete(k),
    };
    Object.defineProperty(globalThis, "localStorage", {
      value: ls,
      configurable: true,
      writable: true,
    });
    try {
      writeRecordingsCache("u1", { recordings: [rec("w1")], sharedWithMe: [] }, 7);
      expect(store.has(recordingsCacheKey("u1"))).toBe(true);
      expect(files.size).toBe(0);
      resetRecordingsCacheMemory();
      expect(readRecordingsCache("u1")?.recordings[0].id).toBe("w1");
      store.set(recordingsCacheKey("u1"), "garbage");
      resetRecordingsCacheMemory();
      expect(readRecordingsCache("u1")).toBeNull();
      expect(store.has(recordingsCacheKey("u1"))).toBe(false);
    } finally {
      delete (globalThis as { localStorage?: unknown }).localStorage;
    }
  });
});

describe("stale marking", () => {
  it("a list fetched before the last mutation is stale; after it is not", () => {
    const before = Date.now() - 10;
    expect(isRecordingsListStale(before)).toBe(false);
    markRecordingsListStale();
    expect(isRecordingsListStale(before)).toBe(true);
    expect(isRecordingsListStale(Date.now() + 1)).toBe(false);
  });
});

describe("mergeRecordingsList", () => {
  it("returns the SAME array when nothing changed (no re-render)", () => {
    const prev = [rec("a"), rec("b")];
    const next = [rec("a"), rec("b")];
    expect(mergeRecordingsList(prev, next)).toBe(prev);
  });

  it("returns the fresh list when a row was added, removed, or edited", () => {
    const prev = [rec("a"), rec("b")];
    expect(mergeRecordingsList(prev, [rec("new"), rec("a"), rec("b")])).toEqual([
      rec("new"),
      rec("a"),
      rec("b"),
    ]);
    expect(mergeRecordingsList(prev, [rec("b")])).toEqual([rec("b")]);
    const renamed = [rec("a", { title: "Kitchen" }), rec("b")];
    expect(mergeRecordingsList(prev, renamed)).toBe(renamed);
  });
});
