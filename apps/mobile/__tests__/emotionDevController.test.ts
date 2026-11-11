import { DEFAULT_USER_SETTINGS, EMOTION_DEV_DEFAULTS, type EmotionDevUserSettings } from "../src/emotion/config";
import { EmotionDevController } from "../src/emotion/devController";
import { EmotionDevStore } from "../src/emotion/devStore";
import type { EmotionEngine } from "../src/emotion/engine";
import { ControlPoller, FakeWatchLink } from "../src/emotion/watchLink";
import { MemoryFs } from "../src/recorder/memoryFs";

const SR = 16000;
const P = (anger = 0, disgust = 0, fear = 0, happiness = 0, neutral = 0, sadness = 0) => [anger, disgust, fear, happiness, neutral, sadness];

class FakeEngine implements EmotionEngine {
  runtime = "fake";
  calls = 0;
  fail = false;
  constructor(readonly id: "tiny" | "small", public out: number[]) {}
  async classify() {
    this.calls++;
    if (this.fail) throw new Error("boom");
    return this.out;
  }
  async release() {}
}

const loud = (s: number, v = 6000) => Int16Array.from({ length: s * SR }, (_, i) => Math.round(v * Math.sin(i / 6)));

function setup(over: Partial<typeof EMOTION_DEV_DEFAULTS> = {}, opts: { tinyOut?: number[]; smallOut?: number[]; small?: boolean; settings?: Partial<EmotionDevUserSettings> } = {}) {
  const cfg = { ...EMOTION_DEV_DEFAULTS, minNudgeGapS: 0, nudgeConfirmWindows: 1, ...over };
  const fs = new MemoryFs();
  let t = 1_000_000;
  const store = new EmotionDevStore(fs, "file:///doc/emotion", cfg, () => t);
  const tiny = new FakeEngine("tiny", opts.tinyOut ?? P(0, 0, 0, 0, 0.9, 0.1));
  const small = opts.small === false ? null : new FakeEngine("small", opts.smallOut ?? opts.tinyOut ?? P(0, 0, 0, 0, 0.9, 0.1));
  const link = new FakeWatchLink();
  let n = 0;
  const ctl = new EmotionDevController({ tiny, small, store, link, cfg, now: () => t, newId: () => `w${++n}`, showSuggestion: true });
  ctl.applySettings({ ...DEFAULT_USER_SETTINGS, maxNudgesPerHour: cfg.maxNudgesPerHour, nudgeRule: "tiny", ...opts.settings });
  const feed = async (seconds: number, v = 6000) => {
    for (let i = 0; i < seconds; i++) {
      ctl.pushPcm(loud(1, v));
      t += 1000;
      await ctl.idle();
    }
  };
  return { ctl, store, tiny, small, link, fs, feed, advance: (ms: number) => (t += ms) };
}

describe("EmotionDevController", () => {
  it("runs BOTH models on the same window; only TINY can nudge; raw probs logged", async () => {
    const s = setup({}, { tinyOut: P(0.8, 0, 0, 0, 0.2, 0), smallOut: P(0.1, 0, 0, 0, 0.8, 0.1) });
    await s.feed(3);
    expect(s.tiny.calls).toBe(1);
    expect(s.small!.calls).toBe(1);
    const nudge = s.store.readRecords().find((r) => r.kind === "nudge")!;
    expect(nudge.tiny[0]).toBeCloseTo(0.8, 4);
    expect(nudge.small![4]).toBeCloseTo(0.8, 4);
    expect(nudge.nudged).toBe(true);
    expect(nudge.flag).toBe("unrated"); // not tapped -> unrated
    expect(s.link.sent.at(-1)).toMatchObject({ label: "anger", nudge: true, paused: false });
    expect(s.link.sent.at(-1)!.suggestion).toBeTruthy();
    expect(s.store.readWindows()).toHaveLength(1);
  });

  it("SMALL alone (no tiny output agreement) never nudges the watch", async () => {
    const s = setup({}, { tinyOut: P(0, 0, 0, 0, 0.9, 0.1), smallOut: P(0.9, 0, 0, 0, 0.1, 0) });
    await s.feed(6);
    expect(s.link.sent.some((x) => x.nudge)).toBe(false);
  });

  it("silence is skipped without inference", async () => {
    const s = setup();
    await s.feed(6, 0);
    expect(s.tiny.calls).toBe(0);
    expect(s.ctl.snapshot().skippedSilent).toBeGreaterThan(0);
  });

  it("a SMALL failure does not stop TINY", async () => {
    const s = setup({}, { tinyOut: P(0.8, 0, 0, 0, 0.2, 0) });
    s.small!.fail = true;
    await s.feed(3);
    expect(s.store.readRecords().some((r) => r.kind === "nudge")).toBe(true);
    expect(s.ctl.snapshot().errors).toBe(1);
  });

  it("works with TINY only when SMALL is missing", async () => {
    const s = setup({}, { small: false });
    await s.feed(3);
    expect(s.store.readWindows()[0].small).toBeNull();
  });

  it("disagreement saves ONE merged episode clip, rate-capped per day", async () => {
    const s = setup({ maxDisagreeClipsPerDay: 2 }, { tinyOut: P(0, 0, 0.1, 0.7, 0.2, 0), smallOut: P(0.1, 0, 0, 0, 0.1, 0.8) });
    await s.feed(3 * 5);
    s.ctl.stop();
    const withClip = s.store.readRecords().filter((r) => r.clip);
    expect(withClip).toHaveLength(1); // 5 disagreeing windows, cap 2 -> the 2 allowed ones merge into one episode
    expect(withClip[0].clip).toMatch(/disagree\.wav$/);
    expect(withClip[0].episode!.windows.filter((w) => w.triggers.length)).toHaveLength(2);
    expect(s.store.listClips()).toHaveLength(1);
    // windows end at 3 s and 6 s; clip = 0..6 s + 3 s tail
    expect(s.store.listClips()[0].bytes).toBe(44 + 9 * SR * 2);
  });

  it("no disagreement clip when both models agree", async () => {
    const s = setup({}, { tinyOut: P(0.1, 0, 0, 0, 0.1, 0.8), smallOut: P(0, 0, 0, 0, 0.1, 0.9) });
    await s.feed(6);
    expect(s.store.listClips()).toHaveLength(0);
  });

  it("flagging a nudge saves 45 s before + 15 s after and records the flag", async () => {
    const s = setup({}, { tinyOut: P(0.8, 0, 0, 0, 0.2, 0) });
    await s.feed(60);
    const id = s.ctl.snapshot().lastNudgeId!;
    expect(id).toBeTruthy();
    expect(s.ctl.flag("bad", id)).toBe(id);
    expect(s.store.listClips()).toHaveLength(0); // after-part not complete yet
    await s.feed(14);
    expect(s.store.listClips()).toHaveLength(0);
    await s.feed(1);
    const clips = s.store.listClips();
    expect(clips).toHaveLength(1);
    expect(clips[0].bytes).toBe(44 + 60 * SR * 2);
    expect(s.store.readRecords().find((r) => r.id === id)!.flag).toBe("bad");
    // a second tap on the same target changes the flag but does not add a clip
    s.ctl.flag("good", id);
    expect(s.store.readRecords().find((r) => r.id === id)!.flag).toBe("good");
    await s.feed(16);
    expect(s.store.listClips()).toHaveLength(1);
  });

  it("'ignore' is a flag too and also keeps its clip", async () => {
    const s = setup({}, { tinyOut: P(0.8, 0, 0, 0, 0.2, 0) });
    await s.feed(10);
    s.ctl.flag("ignore");
    await s.feed(15);
    expect(s.store.listClips()[0].name).toMatch(/flag-ignore/);
  });

  it("flagging a plain (un-nudged) window writes a record to hang the flag on", async () => {
    const s = setup();
    await s.feed(6);
    const id = s.ctl.flag("bad")!;
    const r = s.store.readRecords().find((x) => x.id === id)!;
    expect(r).toMatchObject({ kind: "window", nudged: false, flag: "bad" });
  });

  it("pause stops listening AND buffering; resume starts fresh; flags are refused while paused", async () => {
    const s = setup();
    await s.feed(6);
    const calls = s.tiny.calls;
    s.ctl.setPaused(true);
    expect(s.link.sent.at(-1)!.paused).toBe(true);
    await s.feed(10);
    expect(s.tiny.calls).toBe(calls);
    expect(s.ctl.flag("bad")).toBeNull();
    s.ctl.setPaused(false);
    expect(s.link.sent.at(-1)!.paused).toBe(false);
    await s.feed(3);
    expect(s.tiny.calls).toBeGreaterThan(calls);
  });

  it("a flag cannot reach back across a pause", async () => {
    const s = setup({}, { tinyOut: P(0.8, 0, 0, 0, 0.2, 0) });
    await s.feed(40);
    s.ctl.setPaused(true);
    s.ctl.setPaused(false);
    await s.feed(4);
    s.ctl.flag("bad");
    await s.feed(15);
    const [clip] = s.store.listClips();
    expect(clip.bytes).toBeLessThan(44 + 25 * SR * 2); // only post-resume audio
  });

  it("the watch's controls drive pause/resume/flag through the poller", async () => {
    const s = setup({}, { tinyOut: P(0.8, 0, 0, 0, 0.2, 0) });
    await s.feed(6);
    const id = s.ctl.snapshot().lastNudgeId!;
    s.link.queue.push({ seq: 1, action: "flag", flag: "good", nudgeId: id, t: 1 }, { seq: 2, action: "pause", t: 2 });
    const poller = new ControlPoller(s.link, (c) => s.ctl.handleControl(c));
    expect(await poller.pollOnce()).toBe(2);
    expect(s.store.readRecords().find((r) => r.id === id)!.flag).toBe("good");
    expect(s.ctl.isPaused()).toBe(true);
    expect(await poller.pollOnce()).toBe(0); // already consumed
  });

  it("windows the watch classified itself are logged as kind 'watch' and never buzz or drive the phone", async () => {
    const s = setup();
    s.link.queue.push({
      seq: 1, action: "windows", t: 1,
      rows: [
        { t: 5000, p: P(0.7, 0.1, 0.05, 0.05, 0.05, 0.05), ms: 41, battery: 80, nudged: true },
        { t: 8000, p: P(0.02, 0.02, 0.02, 0.02, 0.9, 0.02), ms: 38, battery: 80, nudged: false },
      ],
    });
    const poller = new ControlPoller(s.link, (c) => s.ctl.handleControl(c));
    expect(await poller.pollOnce()).toBe(1);
    const recs = s.store.readRecords().filter((r) => r.kind === "watch");
    expect(recs).toHaveLength(2);
    expect(recs[0].tiny[0]).toBeCloseTo(0.7, 4);
    expect(recs[0].small).toBeNull();
    expect(recs[0].nudged).toBe(true);
    expect(s.link.sent).toHaveLength(0); // nothing pushed back down to the wrist
  });

  it("quiet windows do not spam the watch; a label change does", async () => {
    const s = setup();
    await s.feed(30);
    const first = s.link.sent.length;
    expect(first).toBeLessThanOrEqual(3); // ~10 windows in 30 s, refreshed every 15 s
    (s.tiny as FakeEngine).out = P(0, 0.6, 0, 0, 0.4, 0);
    await s.feed(3);
    expect(s.link.sent.at(-1)!.label).toBe("disgust");
  });

  it("buzz budget is configurable and enforced end to end", async () => {
    const s = setup({ maxNudgesPerHour: 2 }, { tinyOut: P(0.8, 0, 0, 0, 0.2, 0) });
    await s.feed(30);
    expect(s.link.sent.filter((x) => x.nudge)).toHaveLength(2);
    s.ctl.setMaxNudgesPerHour(5);
    await s.feed(6);
    expect(s.link.sent.filter((x) => x.nudge).length).toBeGreaterThan(2);
  });
});

describe("EmotionDevStore", () => {
  it("at the cap only RATED clips are evicted; unrated clips are never deleted", () => {
    const fs = new MemoryFs();
    const st = new EmotionDevStore(fs, "file:///doc/e", { ...EMOTION_DEV_DEFAULTS, maxClipBytes: 3 * (44 + 2000) });
    st.setClipCapBytes(3 * 2044);
    for (let i = 0; i < 3; i++) {
      st.addRecord({ id: `i${i}`, t: i, kind: "saved", tiny: [0, 0, 0, 0, 1, 0], small: null, nudged: false });
      expect(st.saveClip({ id: `i${i}`, reason: "x", atMs: i + 1, pcm: new Int16Array(1000) })).not.toBeNull();
    }
    // full of unrated clips: a new save is REFUSED, nothing is deleted, the refusal is on record
    expect(st.saveClip({ id: "i3", reason: "x", atMs: 4, pcm: new Int16Array(1000) })).toBeNull();
    expect(st.listClips()).toHaveLength(3);
    expect(st.lastSkipReason).toBe("storage-full");
    expect(st.usage()).toMatchObject({ clips: 3, unrated: 3, skipped: 1, state: "full" });
    // rate the oldest: now it makes room for the next save
    st.setFlag("i0", "good");
    expect(st.saveClip({ id: "i4", reason: "x", atMs: 5, pcm: new Int16Array(1000) })).not.toBeNull();
    const names = st.listClips().map((c) => c.name);
    expect(names).toHaveLength(3);
    expect(names.some((n) => n.includes("-i0-"))).toBe(false);
    expect(names.some((n) => n.includes("-i1-")) && names.some((n) => n.includes("-i2-"))).toBe(true);
    expect(st.readRecords().find((r) => r.id === "i0")!.flag).toBe("good"); // label survives its audio
  });

  it("trims the oldest half of a log that outgrows its cap", () => {
    const st = new EmotionDevStore(new MemoryFs(), "file:///doc/e", { ...EMOTION_DEV_DEFAULTS, maxLogBytes: 2000 });
    for (let i = 0; i < 60; i++) st.addWindow({ id: `w${i}`, t: i, tiny: P(0, 0, 0, 0, 1, 0), small: null });
    const w = st.readWindows();
    expect(w.length).toBeLessThan(60);
    expect(w.at(-1)!.id).toBe("w59");
  });

  it("deleteAll removes clips and both logs", () => {
    const fs = new MemoryFs();
    const st = new EmotionDevStore(fs, "file:///doc/e");
    st.saveClip({ id: "a", reason: "x", atMs: 1, pcm: new Int16Array(100) });
    st.addWindow({ id: "w", t: 1, tiny: P(0, 0, 0, 0, 1, 0), small: null });
    st.addRecord({ id: "r", t: 1, kind: "nudge", tiny: P(1, 0, 0, 0, 0, 0), small: null, nudged: true });
    st.deleteAll();
    expect(st.listClips()).toEqual([]);
    expect(st.readRecords()).toEqual([]);
    expect(st.readWindows()).toEqual([]);
    expect(st.usage()).toMatchObject({ clips: 0, clipBytes: 0, logBytes: 0, unrated: 0, skipped: 0 });
  });

  it("survives a torn last log line", () => {
    const fs = new MemoryFs();
    const st = new EmotionDevStore(fs, "file:///doc/e");
    st.addRecord({ id: "r", t: 1, kind: "nudge", tiny: P(1, 0, 0, 0, 0, 0), small: null, nudged: true });
    fs.appendBytes("file:///doc/e/emotion-log.jsonl", new TextEncoder().encode('{"op":"fl'));
    expect(st.readRecords()).toHaveLength(1);
  });

  it("export is the log text only — no audio", () => {
    const st = new EmotionDevStore(new MemoryFs(), "file:///doc/e");
    st.saveClip({ id: "a", reason: "x", atMs: 1, pcm: new Int16Array(100) });
    expect(st.exportLogText()).not.toMatch(/RIFF/);
  });

  it("export includes every classified window, not just nudges", () => {
    const st = new EmotionDevStore(new MemoryFs(), "file:///doc/e");
    st.addWindow({ id: "w1", t: 1, tiny: P(0, 0, 0, 0, 1, 0), small: null });
    st.addRecord({ id: "n1", t: 2, kind: "nudge", tiny: P(1, 0, 0, 0, 0, 0), small: null, nudged: true });
    const ids = st.exportLogText().split("\n").filter(Boolean).map((l) => JSON.parse(l).id);
    expect(ids).toEqual(expect.arrayContaining(["w1", "n1"]));
  });
});
