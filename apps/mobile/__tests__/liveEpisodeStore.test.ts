import {
  asRecordingSummary,
  mergeRecent,
  useLiveEpisodeStore,
  type RecentLiveEpisode,
} from "../src/store/liveEpisodeStore";
import type { RecordingSummary } from "../src/api/client";

function ep(id: string, startedAt = "2026-08-24T18:05:00.000Z"): RecentLiveEpisode {
  return {
    episodeId: id,
    sessionId: `live-${id}`,
    startedAt,
    mode: "speaker",
    title: "Live session · speaker",
    turnCount: 4,
    sharedWith: ["mom@example.com"],
  };
}

function row(id: string): RecordingSummary {
  return {
    id,
    created_at: "2026-08-24T10:00:00.000Z",
    filename: `${id}.m4a`,
    media_type: "audio",
    duration_seconds: 10,
    has_analysis: true,
  };
}

beforeEach(() => {
  useLiveEpisodeStore.getState().clear();
});

describe("liveEpisodeStore", () => {
  it("remembers confirmed episodes newest-first, deduped, capped", () => {
    const s = useLiveEpisodeStore.getState();
    s.remember(ep("a"));
    s.remember(ep("b"));
    s.remember(ep("a"));
    expect(useLiveEpisodeStore.getState().recent.map((e) => e.episodeId)).toEqual(["a", "b"]);
    for (let i = 0; i < 12; i++) s.remember(ep(`x${i}`));
    expect(useLiveEpisodeStore.getState().recent).toHaveLength(10);
    s.forget("x11");
    expect(useLiveEpisodeStore.getState().recent[0].episodeId).toBe("x10");
  });

  it("projects an episode to the list-row shape the server would serve", () => {
    expect(asRecordingSummary(ep("a"))).toEqual({
      id: "a",
      created_at: "2026-08-24T18:05:00.000Z",
      filename: "live-session",
      title: "Live session · speaker",
      media_type: "none",
      duration_seconds: null,
      has_analysis: true,
      source_type: "live",
      mode: "speaker",
    });
  });

  it("merge: server rows win, optimistic rows only fill what the server lacks", () => {
    const server = [row("a"), row("c")];
    const merged = mergeRecent(server, [ep("a"), ep("b")]);
    expect(merged.map((r) => r.id)).toEqual(["b", "a", "c"]);
    expect(merged.find((r) => r.id === "a")?.media_type).toBe("audio"); // the server's row, untouched
    expect(mergeRecent(server, [])).toBe(server);
  });
});
