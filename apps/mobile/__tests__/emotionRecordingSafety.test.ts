import { DEFAULT_USER_SETTINGS, EMOTION_DEV_DEFAULTS, type EmotionDevUserSettings } from "../src/emotion/config";
import { buildRecordContext } from "../src/emotion/context";
import { EmotionDevController } from "../src/emotion/devController";
import { describeRecorder, EmotionDevHub, type DevStatus, type RuntimeBuild } from "../src/emotion/devHub";
import { EmotionDevStore } from "../src/emotion/devStore";
import type { EmotionEngine } from "../src/emotion/engine";
import { parseSettings } from "../src/emotion/settingsParse";
import { initEmotionDevFromPrefs } from "../src/emotion/startup";
import { FakeWatchLink } from "../src/emotion/watchLink";
import { MemoryFs } from "../src/recorder/memoryFs";
import { storageLine, storageWarning } from "../src/emotion/storageText";

const SR = 16000;
const P = (anger = 0, disgust = 0, fear = 0, happiness = 0, neutral = 0, sadness = 0) => [anger, disgust, fear, happiness, neutral, sadness];
const HAPPY = P(0, 0, 0, 0.9, 0.1, 0);
const NEUTRAL = P(0, 0, 0, 0, 0.95, 0.05);
const loud = (s: number, v = 6000) => Int16Array.from({ length: s * SR }, (_, i) => Math.round(v * Math.sin(i / 6)));

class FakeEngine implements EmotionEngine {
  runtime = "fake";
  constructor(readonly id: "tiny" | "small", public out: number[]) {}
  async classify() {
    return this.out;
  }
  async release() {}
}

function setup(settings: Partial<EmotionDevUserSettings> = {}, over: Partial<typeof EMOTION_DEV_DEFAULTS> = {}, ctx?: () => never) {
  const cfg = { ...EMOTION_DEV_DEFAULTS, minNudgeGapS: 0, nudgeConfirmWindows: 1, ...over };
  let t = 1_000_000;
  const store = new EmotionDevStore(new MemoryFs(), "file:///doc/emotion", cfg, () => t);
  const tiny = new FakeEngine("tiny", NEUTRAL);
  const small = new FakeEngine("small", NEUTRAL);
  let n = 0;
  const ctl = new EmotionDevController({
    tiny, small, store, link: new FakeWatchLink(), cfg, now: () => t, newId: () => `w${++n}`, rand: () => 0.5, ...(ctx ? { context: ctx } : {}),
  });
  ctl.applySettings({ ...DEFAULT_USER_SETTINGS, saveAudio: true, maxNudgesPerHour: 0, ...settings });
  const feed = async (seconds: number) => {
    for (let i = 0; i < seconds; i++) {
      ctl.pushPcm(loud(1));
      t += 1000;
      await ctl.idle();
    }
  };
  const both = (p: number[]) => {
    tiny.out = p;
    small.out = p;
  };
  return { ctl, store, tiny, small, feed, both };
}

describe("app-start wiring (item 1)", () => {
  it("loads the persisted settings and configures the hub without any screen mounted", async () => {
    const saved: EmotionDevUserSettings = { ...DEFAULT_USER_SETTINGS, enabled: true, saveAudio: true };
    const configure = jest.fn(async () => {});
    const load = jest.fn(async () => saved);
    const got = await initEmotionDevFromPrefs({ configure }, "uid-1", load);
    expect(load).toHaveBeenCalledWith("uid-1");
    expect(configure).toHaveBeenCalledWith(saved);
    expect(got.saveAudio).toBe(true);
  });
});

describe("episode clips (item 4)", () => {
  it("merges consecutive triggered windows into ONE clip with a 5 s lead-in and 3 s tail", async () => {
    const s = setup();
    await s.feed(21); // 7 neutral windows; nothing saved
    expect(s.store.listClips()).toHaveLength(0);
    s.both(HAPPY);
    await s.feed(9); // windows ending 24, 27, 30? -> triggered from the window ending at 24 s
    s.both(NEUTRAL);
    await s.feed(6); // first non-triggering window closes it
    const clips = s.store.listClips();
    expect(clips).toHaveLength(1);
    const rec = s.store.readRecords().find((r) => r.episode)!;
    const trig = rec.episode!.windows.filter((w) => w.triggers.length);
    expect(trig.length).toBeGreaterThanOrEqual(2);
    // clip = 5 s lead + 6 s first window + (n-1) * 3 s + 3 s tail
    expect(clips[0].bytes).toBe(44 + (5 + 6 + (trig.length - 1) * 3 + 3) * SR * 2);
    expect(rec.episode!.peakTiny.label).toBe("happiness");
    expect(rec.episode!.peakSmall!.label).toBe("happiness");
    expect(rec.episode!.nudge).toBe(false);
    expect(rec.episode!.windows[0].tiny[3]).toBeCloseTo(0.9, 4);
    expect(rec.episode!.windows[0].small![3]).toBeCloseTo(0.9, 4);
  });

  it("splits an episode that would run past ~45 s", async () => {
    const s = setup();
    s.both(HAPPY);
    await s.feed(60);
    s.ctl.stop();
    const clips = s.store.listClips();
    expect(clips.length).toBeGreaterThanOrEqual(2);
    expect(clips[0].bytes).toBe(44 + 45 * SR * 2);
    for (const c of clips) expect(c.bytes).toBeLessThanOrEqual(44 + 45 * SR * 2);
  });

  it("records whether a nudge fired during the episode", async () => {
    const s = setup({ nudgeRule: "tiny", maxNudgesPerHour: 4 });
    s.both(P(0.8, 0, 0, 0, 0.2, 0));
    await s.feed(6);
    s.ctl.stop();
    const rec = s.store.readRecords().find((r) => r.episode)!;
    expect(rec.episode!.nudge).toBe(true);
    expect(rec.episode!.nudgeIds.length).toBeGreaterThan(0);
    expect(rec.nudged).toBe(true);
  });

  it("an open episode is flushed (not lost) on stop and on pause", async () => {
    const s = setup();
    s.both(HAPPY);
    await s.feed(7);
    s.ctl.setPaused(true);
    expect(s.store.listClips()).toHaveLength(1);
  });
});

describe("neutral sampling (item 5)", () => {
  it("saves a ~10 s clip of both-neutral speech about once per N minutes of speech, flagged neutral-sample", async () => {
    const s = setup({ neutralSampleEveryMin: 1 }); // rand 0.5 -> due after 60 s of speech
    await s.feed(50);
    expect(s.store.listClips()).toHaveLength(0);
    await s.feed(25);
    const rec = s.store.readRecords().filter((r) => r.kind === "neutral-sample");
    expect(rec).toHaveLength(1);
    const clips = s.store.listClips();
    expect(clips).toHaveLength(1);
    expect(clips[0].bytes).toBe(44 + 10 * SR * 2);
    expect(rec[0].clip).toBe(clips[0].name);
    expect(s.ctl.snapshot().neutralSamples).toBe(1);
  });

  it("is off when set to 0, and when save audio is off", async () => {
    const off = setup({ neutralSampleEveryMin: 0 });
    await off.feed(90);
    expect(off.store.listClips()).toHaveLength(0);
    const noSave = setup({ saveAudio: false, neutralSampleEveryMin: 1 });
    await noSave.feed(90);
    expect(noSave.store.listClips()).toHaveLength(0);
  });
});

describe("storage (item 2)", () => {
  it("default cap is 2 GB and an old saved 500 MB default is migrated", () => {
    expect(DEFAULT_USER_SETTINGS.storageCapMb).toBe(2048);
    expect(parseSettings(JSON.stringify({ storageCapMb: 500 })).storageCapMb).toBe(2048);
    expect(parseSettings(JSON.stringify({ storageCapMb: 800 })).storageCapMb).toBe(800);
    expect(DEFAULT_USER_SETTINGS.neutralSampleEveryMin).toBe(5);
  });

  it("warns at 80%, full at the cap, and counts unrated + skipped", () => {
    const st = new EmotionDevStore(new MemoryFs(), "file:///doc/e");
    st.setClipCapBytes(5 * 2044);
    const put = (i: number) => {
      st.addRecord({ id: `c${i}`, t: i, kind: "saved", tiny: P(0, 0, 0, 1, 0, 0), small: null, nudged: false });
      return st.saveClip({ id: `c${i}`, reason: "x", atMs: i + 1, pcm: new Int16Array(1000) });
    };
    for (let i = 0; i < 3; i++) put(i);
    expect(st.usage().state).toBe("ok");
    put(3);
    expect(st.usage().state).toBe("warn");
    put(4);
    expect(st.usage()).toMatchObject({ state: "full", unrated: 5, skipped: 0, capBytes: 5 * 2044 });
    expect(put(5)).toBeNull();
    const u = st.usage();
    expect(u.skipped).toBe(1);
    expect(st.readRecords().find((r) => r.id === "c5")!.clipSkipped).toBe("storage-full");
    expect(storageWarning(u)).toMatch(/NOT being saved/);
    expect(storageLine(u, 2048)).toMatch(/5 clips, 5 unrated/);
  });

  it("a mark or flag may exceed the cap by 10%, a normal clip may not", () => {
    const st = new EmotionDevStore(new MemoryFs(), "file:///doc/e");
    st.setClipCapBytes(2044);
    expect(st.saveClip({ id: "a", reason: "x", atMs: 1, pcm: new Int16Array(1000) })).not.toBeNull();
    expect(st.saveClip({ id: "b", reason: "x", atMs: 2, pcm: new Int16Array(50) })).toBeNull();
    expect(st.saveClip({ id: "c", reason: "mark", atMs: 3, pcm: new Int16Array(50), userInitiated: true })).not.toBeNull();
  });

  it("a marked clip is never evicted, even once rated", () => {
    const st = new EmotionDevStore(new MemoryFs(), "file:///doc/e");
    st.setClipCapBytes(2044);
    st.addRecord({ id: "m", t: 1, kind: "mark", tiny: [], small: null, nudged: false });
    st.saveClip({ id: "m", reason: "mark", atMs: 1, pcm: new Int16Array(1000) });
    st.setFlag("m", "good");
    expect(st.saveClip({ id: "n", reason: "x", atMs: 2, pcm: new Int16Array(1000) })).toBeNull();
    expect(st.listClips()).toHaveLength(1);
  });

  it("the main log is never trimmed (it holds the unreviewed records)", () => {
    const st = new EmotionDevStore(new MemoryFs(), "file:///doc/e", { ...EMOTION_DEV_DEFAULTS, maxLogBytes: 1000 });
    for (let i = 0; i < 40; i++) st.addRecord({ id: `r${i}`, t: i, kind: "saved", tiny: P(0, 0, 0, 1, 0, 0), small: null, nudged: false });
    expect(st.readRecords()).toHaveLength(40);
  });
});

describe("recorder status line (item 3)", () => {
  const settings = (o: Partial<EmotionDevUserSettings> = {}): EmotionDevUserSettings => ({ ...DEFAULT_USER_SETTINGS, enabled: true, saveAudio: true, ...o });
  const snap = (o: Record<string, unknown> = {}) => ({
    paused: false, windows: 3, skippedSilent: 0, errors: 0, lastLabel: null, lastProb: null, lastSmallLabel: null, nudges: 0, marks: 0, lastNudgeId: null,
    lastError: null, saveAudio: true, clipsSaved: 12, skippedSaves: 0, episodes: 12, neutralSamples: 0, storage: "ok" as const, lastSkipReason: null, ...o,
  });
  const running = (o: Record<string, unknown> = {}): DevStatus => ({ state: "running", models: { tiny: true, small: true, smallReason: null }, snapshot: snap(o) });

  it("says nothing when the tester is off", () => {
    expect(describeRecorder({ state: "off" }, DEFAULT_USER_SETTINGS)).toBeNull();
  });
  it("shows the saved count while recording", () => {
    expect(describeRecorder(running(), settings())).toEqual({ level: "ok", text: "Recording emotion clips: 12 saved" });
  });
  it("is a loud error when the model is missing, never silent", () => {
    const l = describeRecorder({ state: "models-missing", message: "TINY model file missing" }, settings())!;
    expect(l.level).toBe("error");
    expect(l.text).toBe("Not recording: TINY model file missing");
  });
  it("is an error when Save audio is off", () => {
    const l = describeRecorder(running(), settings({ saveAudio: false }))!;
    expect(l.level).toBe("error");
    expect(l.text).toMatch(/Save audio is off/);
  });
  it("warns near the cap and errors when full", () => {
    expect(describeRecorder(running({ storage: "warn" }), settings())!.level).toBe("warn");
    const full = describeRecorder(running({ storage: "full", skippedSaves: 2, lastSkipReason: "storage-full" }), settings())!;
    expect(full.level).toBe("error");
    expect(full.text).toMatch(/storage full, 2 clips skipped/);
  });
  it("surfaces a runtime error and a write failure", () => {
    expect(describeRecorder({ state: "error", message: "boom" }, settings())!.text).toBe("Not recording: boom");
    expect(describeRecorder(running({ skippedSaves: 1, lastSkipReason: "write-error", lastError: "disk" }), settings())!.level).toBe("error");
  });
});

describe("context attached to records (item 7)", () => {
  it("episode, neutral-sample and mark records carry the context snapshot", async () => {
    const ctx = { mode: "room", sessionS: 30, turns: 2, talk: { windowS: 120, selfS: 4, otherS: 6, unknownS: 0, selfShare: 0.4 }, lastTurn: null, coach: null };
    const s = setup({ neutralSampleEveryMin: 1 }, {}, (() => ctx) as never);
    s.both(HAPPY);
    await s.feed(6);
    s.ctl.mark();
    s.ctl.stop();
    const recs = s.store.readRecords();
    expect(recs.find((r) => r.episode)!.context).toEqual(ctx);
    expect(recs.find((r) => r.kind === "mark")!.context).toEqual(ctx);
  });

  it("buildRecordContext computes wearer vs other talk share, pace, overlap and the last coaching line", () => {
    const turn = (o: Record<string, unknown>) => ({ speaker: "S0", displayName: null, isSelf: null, text: "", startTime: 0, endTime: 0, overlap: null, suggestion: null, suggestionKind: null, spoken: false, ...o });
    const turns = [
      turn({ isSelf: true, startTime: 0, endTime: 10, text: "one two three four five six seven eight nine ten" }),
      turn({ isSelf: false, displayName: "Sam", startTime: 10, endTime: 40, text: "a b c d e f g h", overlap: { sustained: true }, suggestion: "Slow down", suggestionKind: "nudge", spoken: true }),
    ] as never;
    const c = buildRecordContext(turns, 45, "room");
    expect(c.mode).toBe("room");
    expect(c.talk).toMatchObject({ selfS: 10, otherS: 30, selfShare: 0.25 });
    expect(c.lastTurn).toMatchObject({ speaker: "Sam", isSelf: false, words: 8, overlapped: true, endedAgoS: 5 });
    expect(c.lastTurn!.paceWpm).toBe(16);
    expect(c.coach).toEqual({ text: "Slow down", kind: "nudge", spoken: true });
    expect(buildRecordContext([] as never, 5, null)).toMatchObject({ turns: 0, lastTurn: null, coach: null, talk: { selfShare: null } });
  });
});

describe("session start surfaces a failed build immediately (item 3)", () => {
  it("sessionStarting builds the runtime without waiting for audio, and retries after a failure", async () => {
    const results: RuntimeBuild[] = [
      { ok: false, code: "models-missing", message: "TINY model file missing" },
      { ok: true, runtime: { controller: { applySettings() {}, setContextProvider() {}, setSnapshotListener() {}, snapshot: () => ({}), stop() {} }, store: {}, models: { tiny: true, small: false, smallReason: null }, dispose: async () => {} } as never },
    ];
    const hub = new EmotionDevHub(async () => results.shift()!);
    await hub.configure({ ...DEFAULT_USER_SETTINGS, enabled: true, saveAudio: true });
    hub.sessionStarting();
    await new Promise((r) => setImmediate(r));
    expect(hub.status()).toEqual({ state: "models-missing", message: "TINY model file missing" });
    expect(describeRecorder(hub.status(), hub.userSettings)!.text).toBe("Not recording: TINY model file missing");
    hub.sessionStarting(); // next Start Listening retries
    await new Promise((r) => setImmediate(r));
    expect(hub.status().state).toBe("running");
  });
});
