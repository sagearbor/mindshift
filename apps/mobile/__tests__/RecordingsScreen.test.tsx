import React from "react";
import { AppState } from "react-native";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import RecordingsScreen, {
  FOREGROUND_REFRESH_MAX_AGE_MS,
  formatParticipants,
} from "../src/screens/RecordingsScreen";
import { listRecordingsAndShared, deleteRecording } from "../src/api/client";
import { deleteCachedMedia } from "../src/utils/mediaCache";
import {
  markRecordingsListStale,
  readRecordingsCache,
  resetRecordingsCacheMemory,
  writeRecordingsCache,
} from "../src/utils/recordingsListCache";
import { useAuthStore } from "../src/store/authStore";
import type {
  RecordingSummary,
  SharedRecordingSummary,
} from "../src/api/client";

// Local override of the shared expo-file-system mock (mediaCache.test.ts
// pattern): the recordings-list cache (2026-08-30) keeps one JSON file per
// account under Paths.cache and reads it SYNCHRONOUSLY (textSync) on the
// screen's first render. `__mockFsFiles` is the in-memory "disk".
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
const mockFsFiles = (globalThis as Record<string, unknown>).__mockFsFiles as Map<
  string,
  string
>;

jest.mock("../src/api/client", () => ({
  listRecordingsAndShared: jest.fn(),
  deleteRecording: jest.fn(),
}));
const mockList = listRecordingsAndShared as jest.Mock;
const mockDelete = deleteRecording as jest.Mock;

// Mock the local media-disk-cache util (2026-08-18): confirmDelete hooks
// into it for best-effort cleanup. Its own file-system behavior is covered
// independently in __tests__/mediaCache.test.ts — here we only assert the
// wiring (called with the right id, and only after a successful delete).
jest.mock("../src/utils/mediaCache", () => ({
  __esModule: true,
  deleteCachedMedia: jest.fn().mockResolvedValue(undefined),
}));
const mockDeleteCachedMedia = deleteCachedMedia as jest.Mock;

/** Resolve the list call with owned + (optional) shared recordings. */
function resolveList(
  own: RecordingSummary[],
  shared: SharedRecordingSummary[] = [],
) {
  mockList.mockResolvedValueOnce({ recordings: own, sharedWithMe: shared });
}

const recordings: RecordingSummary[] = [
  {
    id: "r1",
    created_at: "2026-07-01T10:00:00Z",
    filename: "kitchen-fight.m4a",
    media_type: "audio",
    duration_seconds: 182,
    has_analysis: true,
  },
  {
    id: "r2",
    created_at: "2026-07-02T10:00:00Z",
    filename: "living-room.mp4",
    media_type: "video",
    duration_seconds: 95,
    has_analysis: false,
  },
];

const shared: SharedRecordingSummary[] = [
  {
    id: "s1",
    created_at: "2026-07-03T10:00:00Z",
    filename: "moms-call.m4a",
    title: "Sunday call",
    media_type: "audio",
    duration_seconds: 240,
    has_analysis: true,
    owner_email: "linda@example.com",
    shared: true,
  },
];

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

beforeEach(() => {
  mockList.mockReset();
  mockDelete.mockReset();
  mockDeleteCachedMedia.mockReset();
  mockDeleteCachedMedia.mockResolvedValue(undefined);
  // The list cache is per-process (memory) + per-file ("disk"): start every
  // test cold so the legacy cases below still exercise the no-cache path.
  resetRecordingsCacheMemory();
  mockFsFiles.clear();
  useAuthStore.setState({ user: null });
});

/** Render the screen and settle one microtask turn (the initial fetch). */
async function render(onSelect: (id: string) => void = () => {}) {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(
      <RecordingsScreen onSelectRecording={onSelect} onBack={() => {}} />,
    );
  });
  await act(async () => {});
  return comp;
}

/** A list call the test resolves/rejects by hand, to observe the in-between. */
function deferredList() {
  let resolve!: (v: { recordings: RecordingSummary[]; sharedWithMe: SharedRecordingSummary[] }) => void;
  let reject!: (e: unknown) => void;
  mockList.mockReturnValueOnce(
    new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    }),
  );
  return { resolve, reject };
}

/** Rendered row testIDs in order (deduped: composite + host node share one). */
const ids = (comp: renderer.ReactTestRenderer) => [
  ...new Set(
    comp.root
      .findAll((n) => typeof n.props?.testID === "string" && /^recording-r\d+$/.test(n.props.testID))
      .map((n) => n.props.testID as string),
  ),
];

describe("RecordingsScreen", () => {
  it("lists recordings and opens the replay on tap", async () => {
    resolveList(recordings);
    const onSelect = jest.fn();

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={onSelect} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    expect(queryId(comp, "recordings-list")).toBeTruthy();
    expect(queryId(comp, "recording-r1")).toBeTruthy();
    expect(queryId(comp, "recording-r2")).toBeTruthy();

    act(() => comp.root.find((n) => n.props?.testID === "recording-open-r1").props.onPress());
    expect(onSelect).toHaveBeenCalledWith("r1");
    act(() => comp.unmount());
  });

  it("shows the honest empty state", async () => {
    resolveList([]);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    expect(queryId(comp, "recordings-empty")).toBeTruthy();
    expect(JSON.stringify(comp.toJSON())).toContain("No stored recordings yet");
    act(() => comp.unmount());
  });

  it("shows the honest 503 error state", async () => {
    mockList.mockRejectedValueOnce(new Error("API error: 503"));
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    expect(queryId(comp, "recordings-error")).toBeTruthy();
    expect(JSON.stringify(comp.toJSON())).toContain("Replay storage");
    act(() => comp.unmount());
  });

  it("deletes a recording through the inline confirm flow", async () => {
    resolveList(recordings);
    mockDelete.mockResolvedValueOnce(undefined);

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    // First tap on Delete reveals the confirm row (no network yet).
    act(() => comp.root.find((n) => n.props?.testID === "recording-delete-r1").props.onPress());
    expect(queryId(comp, "confirm-r1")).toBeTruthy();
    expect(mockDelete).not.toHaveBeenCalled();

    // Confirm → DELETE fires and the row is removed.
    await act(async () => {
      comp.root.find((n) => n.props?.testID === "confirm-yes-r1").props.onPress();
    });
    await act(async () => {});

    expect(mockDelete).toHaveBeenCalledWith("r1");
    expect(queryId(comp, "recording-r1")).toBeNull();
    // The other recording remains.
    expect(queryId(comp, "recording-r2")).toBeTruthy();
    // Best-effort local-cache cleanup rides along with a successful delete.
    expect(mockDeleteCachedMedia).toHaveBeenCalledWith("r1");
    act(() => comp.unmount());
  });

  it("cancels the delete without calling the API", async () => {
    resolveList(recordings);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    act(() => comp.root.find((n) => n.props?.testID === "recording-delete-r1").props.onPress());
    act(() => comp.root.find((n) => n.props?.testID === "confirm-no-r1").props.onPress());
    expect(queryId(comp, "confirm-r1")).toBeNull();
    expect(mockDelete).not.toHaveBeenCalled();
    expect(queryId(comp, "recording-r1")).toBeTruthy();
    // No server-side delete happened, so the local cache is left untouched.
    expect(mockDeleteCachedMedia).not.toHaveBeenCalled();
    act(() => comp.unmount());
  });

  it("does NOT clean up the local cache when the server-side delete fails — the recording still exists", async () => {
    resolveList(recordings);
    mockDelete.mockRejectedValueOnce(new Error("API error: 503"));

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    act(() => comp.root.find((n) => n.props?.testID === "recording-delete-r1").props.onPress());
    await act(async () => {
      comp.root.find((n) => n.props?.testID === "confirm-yes-r1").props.onPress();
    });
    await act(async () => {});

    expect(mockDelete).toHaveBeenCalledWith("r1");
    // The row stays (delete failed) — a cache-cleanup here would force a
    // needless re-fetch next replay for a recording that never went away.
    expect(queryId(comp, "recording-r1")).toBeTruthy();
    expect(mockDeleteCachedMedia).not.toHaveBeenCalled();
    act(() => comp.unmount());
  });

  it("shows named participants from the list's manual_speaker_labels, and nothing when none", async () => {
    resolveList([
      {
        ...recordings[0],
        manual_speaker_labels: { "Speaker A": "Linda", "Speaker B": "Sage" },
      },
      // r2 has an empty manual map → no participant line (never fabricated).
      { ...recordings[1], manual_speaker_labels: {} },
    ]);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    const parts = queryId(comp, "recording-participants-r1");
    expect(parts).toBeTruthy();
    expect(JSON.stringify(parts!.props.children)).toContain("Linda & Sage");
    // The unnamed recording shows no participant line.
    expect(queryId(comp, "recording-participants-r2")).toBeNull();
    act(() => comp.unmount());
  });

  it("renders the Shared with me section and opens a shared recording in replay", async () => {
    resolveList(recordings, shared);
    const onSelect = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={onSelect} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    expect(queryId(comp, "shared-with-me-section")).toBeTruthy();
    // The from-line shows the owner's email, verbatim.
    const from = queryId(comp, "shared-from-s1");
    expect(from).toBeTruthy();
    expect(JSON.stringify(from!.props.children)).toContain("linda@example.com");
    // Tapping opens the normal replay for that id (read-only handled in Replay).
    act(() => comp.root.find((n) => n.props?.testID === "shared-open-s1").props.onPress());
    expect(onSelect).toHaveBeenCalledWith("s1");
    act(() => comp.unmount());
  });

  it("shows only the shared section when the user owns nothing", async () => {
    resolveList([], shared);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    // Not the empty state — there IS something to show.
    expect(queryId(comp, "recordings-empty")).toBeNull();
    expect(queryId(comp, "shared-with-me-section")).toBeTruthy();
    expect(queryId(comp, "shared-open-s1")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("defensively renders no shared section when the server omits it", async () => {
    // Older server: sharedWithMe is [] (client already normalized absent → []).
    resolveList(recordings, []);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    expect(queryId(comp, "shared-with-me-section")).toBeNull();
    expect(queryId(comp, "recording-r1")).toBeTruthy();
    act(() => comp.unmount());
  });
});

describe("RecordingsScreen cache-first (stale-while-revalidate)", () => {
  const cachedRows: RecordingSummary[] = [recordings[0]];

  it("renders the cached list SYNCHRONOUSLY — before the network resolves — with a quiet updating note, no spinner", async () => {
    writeRecordingsCache(null, { recordings: cachedRows, sharedWithMe: [] }, Date.now());
    resetRecordingsCacheMemory(); // force a genuine "disk" read on first render
    const pending = deferredList();

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    // First frame, network still pending: the cached row is on screen.
    expect(queryId(comp, "recording-r1")).toBeTruthy();
    expect(queryId(comp, "recordings-loading")).toBeNull();
    // The background fetch was kicked off and shows as a quiet note.
    await act(async () => {});
    expect(mockList).toHaveBeenCalledTimes(1);
    expect(queryId(comp, "recordings-updating")).toBeTruthy();
    expect(queryId(comp, "recordings-loading")).toBeNull();

    // The response lands: new row at the top, removed row gone, note cleared.
    await act(async () => {
      pending.resolve({ recordings: [recordings[1]], sharedWithMe: [] });
    });
    await act(async () => {});
    expect(ids(comp)).toEqual(["recording-r2"]);
    expect(queryId(comp, "recordings-updating")).toBeNull();
    // …and the cache now holds the fresh list for the next open.
    expect(readRecordingsCache(null)?.recordings.map((r) => r.id)).toEqual(["r2"]);
    act(() => comp.unmount());
  });

  it("keeps the cached list with a quiet note when the refresh fails (never an error screen)", async () => {
    writeRecordingsCache(null, { recordings: cachedRows, sharedWithMe: [] }, Date.now());
    mockList.mockRejectedValueOnce(new Error("API error: 503"));
    const comp = await render();

    expect(queryId(comp, "recording-r1")).toBeTruthy();
    expect(queryId(comp, "recordings-error")).toBeNull();
    expect(queryId(comp, "recordings-loading")).toBeNull();
    const note = queryId(comp, "recordings-refresh-note");
    expect(note).toBeTruthy();
    expect(JSON.stringify(comp.toJSON())).toContain("Couldn’t refresh");
    // The cache is untouched by the failure.
    expect(readRecordingsCache(null)?.recordings.map((r) => r.id)).toEqual(["r1"]);
    act(() => comp.unmount());
  });

  it("with no cache: spinner first, exactly as before", async () => {
    const pending = deferredList();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});
    expect(queryId(comp, "recordings-loading")).toBeTruthy();
    expect(queryId(comp, "recordings-updating")).toBeNull();
    await act(async () => {
      pending.resolve({ recordings, sharedWithMe: [] });
    });
    await act(async () => {});
    expect(queryId(comp, "recordings-loading")).toBeNull();
    expect(ids(comp)).toEqual(["recording-r1", "recording-r2"]);
    // First successful fetch seeds the cache.
    expect(readRecordingsCache(null)?.recordings).toHaveLength(2);
    act(() => comp.unmount());
  });

  it("pull-to-refresh forces a network fetch and updates the rows", async () => {
    resolveList([recordings[0]]);
    const comp = await render();
    expect(mockList).toHaveBeenCalledTimes(1);

    resolveList(recordings);
    const refresh = queryId(comp, "recordings-refresh");
    expect(refresh).toBeTruthy();
    await act(async () => refresh!.props.onRefresh());
    await act(async () => {});
    expect(mockList).toHaveBeenCalledTimes(2);
    expect(ids(comp)).toEqual(["recording-r1", "recording-r2"]);
    act(() => comp.unmount());
  });

  it("pull-to-refresh is also available on the empty state", async () => {
    resolveList([]);
    const comp = await render();
    expect(queryId(comp, "recordings-empty")).toBeTruthy();
    resolveList(recordings);
    await act(async () => queryId(comp, "recordings-refresh")!.props.onRefresh());
    await act(async () => {});
    expect(queryId(comp, "recordings-empty")).toBeNull();
    expect(ids(comp)).toHaveLength(2);
    act(() => comp.unmount());
  });

  it("caches per account: another user's list never leaks in", async () => {
    writeRecordingsCache("linda", { recordings: cachedRows, sharedWithMe: [] }, Date.now());
    useAuthStore.setState({
      user: { uid: "sage", email: "s@x", displayName: null },
    });
    const pending = deferredList();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});
    // No cache for "sage" → spinner, not Linda's rows.
    expect(queryId(comp, "recordings-loading")).toBeTruthy();
    expect(queryId(comp, "recording-r1")).toBeNull();
    await act(async () => pending.resolve({ recordings: [recordings[1]], sharedWithMe: [] }));
    await act(async () => {});
    expect(readRecordingsCache("sage")?.recordings.map((r) => r.id)).toEqual(["r2"]);
    expect(readRecordingsCache("linda")?.recordings.map((r) => r.id)).toEqual(["r1"]);
    act(() => comp.unmount());
  });

  it("refreshes on foreground return only when the list is old or marked stale", async () => {
    const addListener = AppState.addEventListener as jest.Mock;
    addListener.mockClear();
    addListener.mockReturnValueOnce({ remove: jest.fn() });
    resolveList(recordings);
    const comp = await render();
    expect(mockList).toHaveBeenCalledTimes(1);
    const onChange = addListener.mock.calls[0][1] as (s: string) => void;

    // Fresh (just fetched) and not stale → no refetch.
    resolveList(recordings);
    await act(async () => onChange("active"));
    await act(async () => {});
    expect(mockList).toHaveBeenCalledTimes(1);

    // A list-changing action elsewhere (rename / new recording) marked it stale.
    markRecordingsListStale();
    await act(async () => onChange("active"));
    await act(async () => {});
    expect(mockList).toHaveBeenCalledTimes(2);

    // Older than the max age → refetch too.
    const realNow = Date.now;
    Date.now = () => realNow() + FOREGROUND_REFRESH_MAX_AGE_MS + 1;
    try {
      resolveList(recordings);
      await act(async () => onChange("active"));
      await act(async () => {});
      expect(mockList).toHaveBeenCalledTimes(3);
    } finally {
      Date.now = realNow;
    }
    act(() => comp.unmount());
  });

  it("a delete updates the cache so the next open doesn't resurrect the row", async () => {
    resolveList(recordings);
    mockDelete.mockResolvedValueOnce(undefined);
    const comp = await render();
    act(() => queryId(comp, "recording-delete-r1")!.props.onPress());
    await act(async () => queryId(comp, "confirm-yes-r1")!.props.onPress());
    await act(async () => {});
    expect(readRecordingsCache(null)?.recordings.map((r) => r.id)).toEqual(["r2"]);
    act(() => comp.unmount());
  });
});

describe("formatParticipants", () => {
  it("returns null when there are no names (absent, empty, or blank)", () => {
    expect(formatParticipants(undefined)).toBeNull();
    expect(formatParticipants({})).toBeNull();
    expect(formatParticipants({ "Speaker A": "  " })).toBeNull();
  });

  it("joins names honestly by count", () => {
    expect(formatParticipants({ a: "Linda" })).toBe("Linda");
    expect(formatParticipants({ a: "Linda", b: "Sage" })).toBe("Linda & Sage");
    expect(formatParticipants({ a: "Linda", b: "Sage", c: "Ari" })).toBe(
      "Linda, Sage & Ari",
    );
    expect(
      formatParticipants({ a: "Linda", b: "Sage", c: "Ari", d: "Bo" }),
    ).toBe("Linda, Sage & 2 more");
  });
});
