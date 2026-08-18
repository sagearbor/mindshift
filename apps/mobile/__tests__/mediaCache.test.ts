import { Platform } from "react-native";
import {
  getCachedMediaUri,
  cacheMediaInBackground,
  deleteCachedMedia,
} from "../src/utils/mediaCache";

// Local override of the shared expo-file-system mock (same pattern
// AvatarCaptureScreen.test.tsx uses): this module needs Directory/Paths.cache
// plus the SDK-57 static `File.downloadFileAsync`, which the jest-setup
// wholesale mock doesn't provide (it only covers the chunked-upload
// File.open()/readBytes() path). `__mockFsFiles` is a Set of file uris the
// test pre-seeds to represent "already cached"; `downloadFileAsync` is a spy
// so tests can assert what it was called with without touching the network.
const mockDownloadFileAsync = jest.fn();
jest.mock("expo-file-system", () => {
  const existingUris = new Set<string>();
  (globalThis as Record<string, unknown>).__mockFsExistingUris = existingUris;

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
      return existingUris.has(this.uri);
    }
    delete() {
      existingUris.delete(this.uri);
    }
    static downloadFileAsync = (...args: unknown[]) =>
      mockDownloadFileAsync(...args);
  }
  return {
    __esModule: true,
    Directory,
    File,
    Paths: { cache: { uri: "file:///cache" } },
  };
});
// mediaCache.ts lazy-requires expo-file-system (never touches native code at
// import time), so force the mock factory above to run now rather than on
// first call — otherwise `__mockFsExistingUris` isn't set until the first
// test actually exercises a native code path.
require("expo-file-system");

const originalOS = Platform.OS;
function setPlatform(os: string) {
  Object.defineProperty(Platform, "OS", { value: os, configurable: true });
}
function existingUris(): Set<string> {
  return (globalThis as Record<string, unknown>).__mockFsExistingUris as Set<
    string
  >;
}

beforeEach(() => {
  existingUris().clear();
  mockDownloadFileAsync.mockReset();
  mockDownloadFileAsync.mockResolvedValue({ uri: "file:///cache/media/x.mp4" });
});

afterEach(() => {
  setPlatform(originalOS);
});

describe("getCachedMediaUri", () => {
  it("returns null when nothing is cached for this recording", () => {
    setPlatform("ios");
    expect(getCachedMediaUri("rec-1", "video")).toBeNull();
  });

  it("returns the local file uri once that recording's media is cached", () => {
    setPlatform("ios");
    existingUris().add("file:///cache/media/rec-1.mp4");
    expect(getCachedMediaUri("rec-1", "video")).toBe(
      "file:///cache/media/rec-1.mp4",
    );
  });

  it("keys strictly by recording_id — a DIFFERENT recording_id is a cache miss even though some other file is cached", () => {
    setPlatform("ios");
    existingUris().add("file:///cache/media/rec-1.mp4");
    expect(getCachedMediaUri("rec-2", "video")).toBeNull();
  });

  it("uses the audio extension for audio recordings, so a video cache entry doesn't accidentally satisfy an audio lookup", () => {
    setPlatform("ios");
    existingUris().add("file:///cache/media/rec-1.mp4"); // cached as video
    expect(getCachedMediaUri("rec-1", "audio")).toBeNull();
  });

  it("always returns null on web — this is a native-only optimization", () => {
    setPlatform("web");
    existingUris().add("file:///cache/media/rec-1.mp4");
    expect(getCachedMediaUri("rec-1", "video")).toBeNull();
  });
});

describe("cacheMediaInBackground", () => {
  it("kicks off a fire-and-forget download to the cache path keyed by recording_id", async () => {
    setPlatform("ios");
    cacheMediaInBackground("rec-1", "video", "https://example.com/media?tk=abc");
    // Fire-and-forget: the call returns synchronously, before the download
    // resolves.
    await Promise.resolve();
    await Promise.resolve();
    expect(mockDownloadFileAsync).toHaveBeenCalledWith(
      "https://example.com/media?tk=abc",
      expect.objectContaining({ uri: "file:///cache/media/rec-1.mp4" }),
      expect.objectContaining({ idempotent: true }),
    );
  });

  it("never throws (best-effort) when the download fails", async () => {
    setPlatform("ios");
    mockDownloadFileAsync.mockRejectedValueOnce(new Error("network down"));
    expect(() =>
      cacheMediaInBackground("rec-1", "audio", "https://example.com/media"),
    ).not.toThrow();
    await Promise.resolve();
    await Promise.resolve();
  });

  it("does nothing on web — never calls the download API", async () => {
    setPlatform("web");
    cacheMediaInBackground("rec-1", "video", "https://example.com/media");
    await Promise.resolve();
    await Promise.resolve();
    expect(mockDownloadFileAsync).not.toHaveBeenCalled();
  });
});

describe("deleteCachedMedia", () => {
  it("best-effort deletes a cached file for the given recording_id", async () => {
    setPlatform("ios");
    existingUris().add("file:///cache/media/rec-1.mp4");
    await deleteCachedMedia("rec-1");
    expect(existingUris().has("file:///cache/media/rec-1.mp4")).toBe(false);
  });

  it("is a silent no-op when nothing is cached for that recording", async () => {
    setPlatform("ios");
    await expect(deleteCachedMedia("rec-never-cached")).resolves.not.toThrow();
  });

  it("does nothing on web", async () => {
    setPlatform("web");
    existingUris().add("file:///cache/media/rec-1.mp4");
    await deleteCachedMedia("rec-1");
    // Untouched — the web path never runs any file-system code.
    expect(existingUris().has("file:///cache/media/rec-1.mp4")).toBe(true);
  });
});
