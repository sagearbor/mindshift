import { DEFAULT_USER_SETTINGS, EMOTION_DEV_DEFAULTS, type EmotionDevUserSettings } from "../src/emotion/config";
import { EmotionDevController } from "../src/emotion/devController";
import { EmotionDevStore } from "../src/emotion/devStore";
import type { EmotionEngine } from "../src/emotion/engine";
import { NudgeGate } from "../src/emotion/logic";
import { ReviewSession, describeItem } from "../src/emotion/reviewSession";
import { parseSettings } from "../src/emotion/settingsParse";
import { FakeWatchLink } from "../src/emotion/watchLink";
import { MemoryFs } from "../src/recorder/memoryFs";

const SR = 16000;
const P = (anger = 0, disgust = 0, fear = 0, happiness = 0, neutral = 0, sadness = 0) => [anger, disgust, fear, happiness, neutral, sadness];
/** SMALL unsure (top class < 0.6) so TINY vs SMALL is not a confident disagreement. */
const NOT_SURE = [0, 0, 0, 0, 0.5, 0.5];
const loud = (s: number, v = 6000) => Int16Array.from({ length: s * SR }, (_, i) => Math.round(v * Math.sin(i / 6)));

class FakeEngine implements EmotionEngine {
  runtime = "fake";
  constructor(readonly id: "tiny" | "small", public out: number[]) {}
  async classify() {
    return this.out;
  }
  async release() {}
}

function setup(tinyOut: number[], smallOut: number[], settings: Partial<EmotionDevUserSettings> = {}, over: Partial<typeof EMOTION_DEV_DEFAULTS> = {}) {
  const cfg = { ...EMOTION_DEV_DEFAULTS, minNudgeGapS: 0, nudgeConfirmWindows: 1, ...over };
  let t = 1_000_000;
  const store = new EmotionDevStore(new MemoryFs(), "file:///doc/emotion", cfg, () => t);
  const tiny = new FakeEngine("tiny", tinyOut);
  const small = new FakeEngine("small", smallOut);
  const link = new FakeWatchLink();
  let n = 0;
  const ctl = new EmotionDevController({ tiny, small, store, link, cfg, now: () => t, newId: () => `w${++n}` });
  ctl.applySettings({ ...DEFAULT_USER_SETTINGS, ...settings });
  const feed = async (seconds: number) => {
    for (let i = 0; i < seconds; i++) {
      ctl.pushPcm(loud(1));
      t += 1000;
      await ctl.idle();
    }
  };
  return { ctl, store, tiny, small, link, feed };
}

describe("nudge gate: both models must agree (default)", () => {
  const cfg = { ...EMOTION_DEV_DEFAULTS, nudgeConfirmWindows: 1, minNudgeGapS: 0 };
  it("is the default rule", () => {
    expect(DEFAULT_USER_SETTINGS.nudgeRule).toBe("both");
    expect(parseSettings(null).nudgeRule).toBe("both");
  });
  it("fires only when both name the same negative class at >= 0.6", () => {
    expect(new NudgeGate(cfg).decide(P(0.7, 0, 0, 0, 0.3, 0), 0, P(0.65, 0, 0, 0, 0.35, 0)).nudge).toBe(true);
    expect(new NudgeGate(cfg).decide(P(0.7, 0, 0, 0, 0.3, 0), 0, P(0.55, 0, 0, 0, 0.45, 0)).nudge).toBe(false); // small below 0.6
    expect(new NudgeGate(cfg).decide(P(0.55, 0, 0, 0, 0.45, 0), 0, P(0.9, 0, 0, 0, 0.1, 0)).nudge).toBe(false); // tiny below 0.6
    expect(new NudgeGate(cfg).decide(P(0.8, 0, 0, 0, 0.2, 0), 0, P(0, 0, 0, 0, 0.9, 0.1)).reason).toBe("models-disagree");
    expect(new NudgeGate(cfg).decide(P(0.8, 0, 0, 0, 0.2, 0), 0, P(0, 0, 0.8, 0, 0.2, 0)).nudge).toBe(false); // different negatives
    expect(new NudgeGate(cfg).decide(P(0.8, 0, 0, 0, 0.2, 0), 0, null).reason).toBe("no-small");
  });
  it("TINY's disgust alone never nudges, and neutral/happy-vs-neutral never does", () => {
    expect(new NudgeGate(cfg).decide(P(0, 0.95, 0, 0, 0.05, 0), 0, P(0, 0, 0, 0, 0.9, 0.1)).nudge).toBe(false);
    expect(new NudgeGate(cfg).decide(P(0, 0, 0, 0, 0.9, 0.1), 0, P(0, 0, 0, 0, 0.9, 0.1)).nudge).toBe(false);
  });
  it("the old TINY-only rule is still available as a setting", () => {
    const g = new NudgeGate(cfg);
    g.setRule("tiny");
    expect(g.decide(P(0.8, 0, 0, 0, 0.2, 0), 0, P(0, 0, 0, 0, 0.9, 0.1)).nudge).toBe(true);
  });
  it("controller: TINY disgust + SMALL neutral stays silent; both anger buzzes", async () => {
    const a = setup(P(0, 0.9, 0, 0, 0.1, 0), P(0, 0, 0, 0, 0.9, 0.1));
    await a.feed(9);
    expect(a.link.sent.some((x) => x.nudge)).toBe(false);
    const b = setup(P(0.8, 0, 0, 0, 0.2, 0), P(0.7, 0, 0, 0, 0.3, 0));
    await b.feed(3);
    expect(b.link.sent.some((x) => x.nudge && x.label === "anger")).toBe(true);
    const c = setup(P(0.8, 0, 0, 0, 0.2, 0), P(0.7, 0, 0, 0, 0.3, 0), { nudgeRule: "tiny" });
    await c.feed(3);
    expect(c.link.sent.some((x) => x.nudge)).toBe(true);
  });
});

describe("settings parse", () => {
  it("round-trips the new fields and clamps junk", () => {
    expect(parseSettings(JSON.stringify({ saveAudio: true, saveAudioMinProb: 0.7, storageCapMb: 200, nudgeRule: "tiny" }))).toMatchObject({
      saveAudio: true, saveAudioMinProb: 0.7, storageCapMb: 200, nudgeRule: "tiny",
    });
    const junk = parseSettings(JSON.stringify({ saveAudio: "yes", saveAudioMinProb: 5, storageCapMb: -1, nudgeRule: "x" }));
    expect(junk).toMatchObject({ saveAudio: false, saveAudioMinProb: 0.5, storageCapMb: 500, nudgeRule: "both" });
  });
  it("save-audio defaults OFF", () => {
    expect(DEFAULT_USER_SETTINGS.saveAudio).toBe(false);
  });
});

describe("save-audio mode", () => {
  it("OFF: nothing is saved for a non-neutral reading", async () => {
    const s = setup(P(0, 0, 0, 0.8, 0.2, 0), NOT_SURE);
    await s.feed(12);
    expect(s.store.listClips()).toHaveLength(0);
  });

  it("ON: either model at >= threshold saves a 6 s wav with both models' probabilities", async () => {
    const s = setup(P(0, 0, 0, 0, 0.9, 0.1), P(0, 0, 0, 0.55, 0.45, 0), { saveAudio: true });
    await s.feed(12);
    const clips = s.store.listClips();
    expect(clips.length).toBeGreaterThanOrEqual(3);
    expect(clips[clips.length - 1].bytes).toBe(44 + 6 * SR * 2);
    const rec = s.store.readRecords().find((r) => r.kind === "saved" && r.clip === clips[clips.length - 1].name)!;
    expect(rec).toBeTruthy();
    expect(rec.small![3]).toBeCloseTo(0.55, 4);
    expect(rec.tiny[4]).toBeCloseTo(0.9, 4);
  });

  it("ON: neutral, or below the configurable threshold, saves nothing", async () => {
    const neutral = setup(P(0, 0, 0, 0, 0.9, 0.1), P(0, 0, 0, 0, 0.9, 0.1), { saveAudio: true });
    await neutral.feed(6);
    expect(neutral.store.listClips()).toHaveLength(0);
    const low = setup(P(0, 0, 0, 0.55, 0.45, 0), P(0, 0, 0, 0, 1, 0), { saveAudio: true, saveAudioMinProb: 0.7 });
    await low.feed(6);
    expect(low.store.listClips()).toHaveLength(0);
  });

  it("ON: the 20/day disagreement-clip cap no longer applies", async () => {
    const off = setup(P(0.8, 0, 0, 0, 0.2, 0), P(0, 0, 0, 0, 0.1, 0.9), {}, { maxDisagreeClipsPerDay: 2 });
    await off.feed(30);
    expect(off.store.listClips()).toHaveLength(2);
    const on = setup(P(0.8, 0, 0, 0, 0.2, 0), P(0, 0, 0, 0, 0.1, 0.9), { saveAudio: true }, { maxDisagreeClipsPerDay: 2 });
    await on.feed(30);
    expect(on.store.listClips().length).toBeGreaterThan(5);
  });

  it("storage cap evicts the oldest UNRATED clips first and usage reports bytes", async () => {
    const s = setup(P(0, 0, 0, 0.9, 0.1, 0), NOT_SURE, { saveAudio: true, storageCapMb: 50 });
    await s.feed(15);
    const before = s.store.listClips();
    expect(before.length).toBeGreaterThanOrEqual(4);
    // rate the OLDEST clip, then shrink the cap to ~2 clips
    const oldest = s.store.readRecords().find((r) => r.clip === before[0].name)!;
    s.store.setFlag(oldest.id, "good");
    s.store.setClipCapBytes(2 * 192044 + 10);
    const after = s.store.listClips().map((c) => c.name);
    expect(after).toContain(before[0].name); // rated survives
    expect(after.length).toBeLessThan(before.length);
    expect(after).not.toContain(before[1].name); // oldest unrated went first
    expect(after).toContain(before[before.length - 1].name); // newest kept
    expect(s.store.usage().clipBytes).toBeLessThanOrEqual(2 * 192044 + 10);
  });
});

describe("mark this moment", () => {
  it("saves previous 45 s + next 15 s with a user mark record", async () => {
    const s = setup(P(0, 0, 0, 0, 0.9, 0.1), P(0, 0, 0, 0, 0.9, 0.1));
    await s.feed(50);
    const id = s.ctl.mark();
    expect(id).toMatch(/^mark-/);
    expect(s.store.listClips()).toHaveLength(0); // waits for the 15 s after
    await s.feed(15);
    const clips = s.store.listClips();
    expect(clips).toHaveLength(1);
    expect(clips[0].name).toMatch(/-mark\.wav$/);
    expect(clips[0].bytes).toBe(44 + 60 * SR * 2);
    const rec = s.store.readRecords().find((r) => r.id === id)!;
    expect(rec.kind).toBe("mark");
    expect(rec.clip).toBe(clips[0].name);
  });
  it("is ignored while paused, and the watch can send it", async () => {
    const s = setup(P(0, 0, 0, 0, 0.9, 0.1), P(0, 0, 0, 0, 0.9, 0.1));
    await s.feed(5);
    s.ctl.setPaused(true);
    expect(s.ctl.mark()).toBeNull();
    s.ctl.setPaused(false);
    s.ctl.handleControl({ seq: 1, action: "mark", t: 0 });
    expect(s.ctl.snapshot().marks).toBe(1);
  });
});

describe("review ratings in the log", () => {
  function withClips() {
    const s = setup(P(0, 0, 0, 0.9, 0.1, 0), NOT_SURE, { saveAudio: true });
    return s;
  }
  it("queue holds unrated clips only; thumbs set the flag; truth is logged and exported", async () => {
    const s = withClips();
    await s.feed(9);
    const q = s.store.reviewQueue();
    expect(q.length).toBeGreaterThanOrEqual(2);
    s.store.setTruth(q[0].id, "sadness");
    s.store.setFlag(q[0].id, "bad");
    s.store.setFlag(q[1].id, "good");
    const q2 = s.store.reviewQueue().map((i) => i.id);
    expect(q2).not.toContain(q[0].id);
    expect(q2).not.toContain(q[1].id);
    const r0 = s.store.readRecords().find((r) => r.id === q[0].id)!;
    expect(r0).toMatchObject({ flag: "bad", truth: "sadness" });
    expect(s.store.exportLogText()).toContain('"truth":"sadness"');
    expect(s.store.exportLogText()).toContain('"flag":"good"');
  });
});

describe("ReviewSession", () => {
  const items = ["a", "b", "c"].map((id, k) => ({ id, clip: `${id}.wav`, uri: `u/${id}.wav`, t: k, kind: "saved" as const, tiny: P(0.7, 0, 0, 0, 0.3, 0), small: P(0, 0, 0, 0, 0.8, 0.2) }));
  const mk = () => {
    const rated: [string, string, unknown][] = [];
    const s = new ReviewSession(items, { rate: (id, r, t) => rated.push([id, r, t ?? null]) });
    return { s, rated };
  };
  it("thumbs up = good, down = bad (+ optional truth), each advances", () => {
    const { s, rated } = mk();
    expect(s.current()!.id).toBe("a");
    s.thumb(true);
    s.setTruth("fear");
    s.thumb(false);
    expect(rated).toEqual([["a", "good", null], ["b", "bad", "fear"]]);
    expect(s.current()!.id).toBe("c");
    expect(s.rated).toBe(2);
  });
  it("truth does not leak to the next clip; tapping it twice clears it", () => {
    const { s, rated } = mk();
    s.setTruth("anger");
    s.setTruth("anger");
    s.thumb(false);
    s.setTruth("fear");
    s.skip();
    s.thumb(true);
    expect(rated).toEqual([["a", "bad", null], ["c", "good", null]]);
    expect(s.skipped).toBe(1);
  });
  it("auto-advances after a clip plays twice with no tap; ends with done", () => {
    const { s, rated } = mk();
    expect(s.onEnded()).toBe("replay");
    expect(s.onEnded()).toBe("advance");
    expect(s.current()!.id).toBe("b");
    s.skip();
    s.onEnded();
    expect(s.onEnded()).toBe("done");
    expect(s.current()).toBeNull();
    expect(s.thumb(true)).toBe(false);
    expect(rated).toEqual([]);
  });
  it("describes what the models said", () => {
    expect(describeItem(items[0])).toBe("TINY: anger 70% · SMALL: neutral 80%");
    expect(describeItem({ ...items[0], small: null, kind: "mark" })).toBe("Marked moment. TINY: anger 70% · SMALL: —");
  });
});

describe("hub review controls work with the store alone", () => {
  it("rate() writes flag + truth and the queue shrinks", async () => {
    const { EmotionDevHub } = await import("../src/emotion/devHub");
    const s = setup(P(0, 0, 0, 0.9, 0.1, 0), NOT_SURE, { saveAudio: true });
    await s.feed(9);
    const hub = new EmotionDevHub(async () => ({ ok: false, code: "error", message: "n/a" }), () => s.store);
    const q = hub.reviewQueue();
    expect(q.length).toBeGreaterThan(1);
    hub.rate(q[0].id, "bad", "fear");
    expect(hub.reviewQueue()).toHaveLength(q.length - 1);
    expect(hub.mark()).toBeNull(); // no running tester -> nothing to mark
    expect(s.store.readRecords().find((r) => r.id === q[0].id)).toMatchObject({ flag: "bad", truth: "fear" });
  });
});
