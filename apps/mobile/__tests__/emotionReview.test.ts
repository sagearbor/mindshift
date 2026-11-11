import { DEFAULT_USER_SETTINGS, EMOTION_DEV_DEFAULTS, type EmotionDevUserSettings } from "../src/emotion/config";
import { EmotionDevController } from "../src/emotion/devController";
import { EmotionDevHub } from "../src/emotion/devHub";
import { EmotionDevStore, type ReviewItemFull, type ReviewLabel } from "../src/emotion/devStore";
import type { EmotionEngine } from "../src/emotion/engine";
import { NudgeGate } from "../src/emotion/logic";
import { ALL_DAYS, ReviewSession, buildQueue, contextDetail, contextLine, defaultDay, defaultHideGuess, describeItem, listDays, raterKey, type ReviewMode } from "../src/emotion/reviewSession";
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
    expect(junk).toMatchObject({ saveAudio: false, saveAudioMinProb: 0.5, storageCapMb: 2048, nudgeRule: "both" });
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
    s.ctl.stop();
    const clips = s.store.listClips();
    expect(clips).toHaveLength(1); // 4 back-to-back hits merge into ONE episode clip
    expect(clips[0].bytes).toBe(44 + 12 * SR * 2);
    const rec = s.store.readRecords().find((r) => r.kind === "saved" && r.clip === clips[0].name)!;
    expect(rec).toBeTruthy();
    expect(rec.episode!.windows).toHaveLength(4);
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
    const triggered = (st: EmotionDevStore) => st.readRecords().flatMap((r) => r.episode?.windows ?? []).filter((w) => w.triggers.length).length;
    const off = setup(P(0.8, 0, 0, 0, 0.2, 0), P(0, 0, 0, 0, 0.1, 0.9), {}, { maxDisagreeClipsPerDay: 2 });
    await off.feed(30);
    off.ctl.stop();
    expect(triggered(off.store)).toBe(2);
    const on = setup(P(0.8, 0, 0, 0, 0.2, 0), P(0, 0, 0, 0, 0.1, 0.9), { saveAudio: true }, { maxDisagreeClipsPerDay: 2 });
    await on.feed(30);
    on.ctl.stop();
    expect(triggered(on.store)).toBeGreaterThan(5);
  });

  it("a full store stops saving new clips, keeps every unrated one, and says so", async () => {
    const s = setup(P(0, 0, 0, 0.9, 0.1, 0), NOT_SURE, { saveAudio: true, storageCapMb: 50 });
    s.store.setClipCapBytes(100_000); // far smaller than one 12 s episode (384 KB)
    await s.feed(12);
    s.ctl.stop();
    expect(s.store.listClips()).toHaveLength(0);
    const rec = s.store.readRecords().find((r) => r.kind === "saved")!;
    expect(rec.clipSkipped).toBe("clip-larger-than-cap"); // the probabilities are still logged
    expect(s.ctl.snapshot()).toMatchObject({ skippedSaves: 1, clipsSaved: 0, lastSkipReason: "clip-larger-than-cap" });
    expect(s.store.usage().skipped).toBe(1);
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
    s.ctl.stop(); // close episode 1
    await s.feed(9);
    s.ctl.stop(); // episode 2
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

describe("per-rater review labels in the store", () => {
  const label = (rater: string, extra = {}) => ({ rater, rating: "bad" as const, truth: "fear", t: 5, ...extra });
  async function twoClips() {
    const s = setup(P(0, 0, 0, 0.9, 0.1, 0), NOT_SURE, { saveAudio: true });
    await s.feed(9);
    return s;
  }
  it("a second rater never overwrites the owner; legacy flag reads as the me label", async () => {
    const s = await twoClips();
    const [a] = s.store.reviewItems();
    s.store.setReview(a.id, label("me", { nudgeWarranted: "no" }));
    s.store.setReview(a.id, label("mom", { truth: "sadness", nudgeWarranted: "yes" }));
    const got = s.store.reviewItems().find((i) => i.id === a.id)!;
    expect(got.reviews.map((r) => r.rater).sort()).toEqual(["me", "mom"]);
    expect(got.reviews.find((r) => r.rater === "me")).toMatchObject({ truth: "fear", nudgeWarranted: "no" });
    expect(got.flag).toBe("bad"); // legacy field still written for "me"
    expect(got.truth).toBe("fear");
    // same rater again replaces, not duplicates
    s.store.setReview(a.id, label("mom", { truth: "anger" }));
    expect(s.store.reviewItems().find((i) => i.id === a.id)!.reviews.filter((r) => r.rater === "mom")).toHaveLength(1);
  });
  it("only a non-owner label does not touch the legacy fields; clear undoes", async () => {
    const s = await twoClips();
    const [a] = s.store.reviewItems();
    s.store.setReview(a.id, label("mom"));
    expect(s.store.readRecords().find((r) => r.id === a.id)!.flag).toBeNull();
    expect(s.store.reviewQueue().map((i) => i.id)).toContain(a.id); // legacy queue unchanged
    s.store.clearReview(a.id, "mom");
    expect(s.store.reviewItems().find((i) => i.id === a.id)!.reviews).toEqual([]);
    s.store.setReview(a.id, label("me"));
    s.store.clearReview(a.id, "me");
    expect(s.store.reviewQueue().map((i) => i.id)).toContain(a.id);
  });
  it("export includes rater and nudgeWarranted; clip length is derived", async () => {
    const s = await twoClips();
    const [a] = s.store.reviewItems();
    s.store.setReview(a.id, label("mom", { nudgeWarranted: "unsure" }));
    const txt = s.store.exportLogText();
    expect(txt).toContain('"rater":"mom"');
    expect(txt).toContain('"nudgeWarranted":"unsure"');
    expect(s.store.reviewItems().some((i) => i.durationS === 6)).toBe(true);
  });
});

describe("review queue helpers", () => {
  const D = (day: number, hour = 12) => new Date(2026, 10, day, hour, 5).getTime();
  const mkItem = (id: string, t: number, over = {}): ReviewItemFull => ({
    id, clip: `${id}.wav`, uri: `u/${id}.wav`, t, kind: "saved", tiny: P(0.7, 0, 0, 0, 0.3, 0), small: P(0, 0, 0, 0, 0.8, 0.2), reviews: [], ...over,
  });
  const lbl = (rater: string): ReviewLabel => ({ rater, rating: "good", t: 1 });
  const items = [
    mkItem("a", D(9)),
    mkItem("b", D(10), { reviews: [lbl("me")] }),
    mkItem("c", D(10, 13)),
    mkItem("d", D(11), { flag: "good" }), // legacy "me" label
  ];
  it("raterKey / hide-guess default", () => {
    expect(raterKey("me")).toBe("me");
    expect(raterKey("other", "  Aunt  Jo ")).toBe("other:aunt jo");
    expect(raterKey("other", "")).toBe("other");
    expect(defaultHideGuess("me")).toBe(false);
    expect(defaultHideGuess("mom")).toBe(true);
  });
  it("legacy flag counts as a me label, not as mom's", () => {
    expect(buildQueue(items, "me").map((i) => i.id)).toEqual(["a", "c"]);
    expect(buildQueue(items, "mom").map((i) => i.id)).toEqual(["a", "b", "c", "d"]);
  });
  it("days: newest with unrated is the default; all is available", () => {
    expect(listDays(items, "me").map((d) => [d.day, d.unrated, d.total])).toEqual([
      ["2026-11-11", 0, 1], ["2026-11-10", 1, 2], ["2026-11-09", 1, 1],
    ]);
    expect(defaultDay(items, "me")).toBe("2026-11-10");
    expect(defaultDay([items[3]], "me")).toBe("all");
    expect(buildQueue(items, "me", "2026-11-10").map((i) => i.id)).toEqual(["c"]);
    expect(buildQueue(items, "me", ALL_DAYS)).toHaveLength(2);
  });
  it("context line shows only the fields that exist", () => {
    expect(contextLine({ t: D(10, 19), kind: "saved" })).toBe("7:05 pm · emotion");
    expect(contextLine({ t: D(10, 0), kind: "mark", durationS: 60, nudged: true, nudgeReason: "anger" })).toBe(
      "12:05 am · 60 s · marked moment · nudge fired: anger",
    );
    expect(contextLine({ t: D(10, 9), kind: "neutral-sample", durationS: 45 })).toBe("9:05 am · 45 s · neutral sample");
    expect(contextLine({ t: D(10, 9), kind: "future-kind" })).toBe("9:05 am · future-kind");
    const context = { mode: "room", sessionS: 10, turns: 3, talk: { windowS: 120, selfS: 74, otherS: 40, unknownS: 0, selfShare: 0.62 }, lastTurn: { speaker: "a", isSelf: true, durationS: 4, words: 10, paceWpm: 141.6, overlapped: false, endedAgoS: 2 }, coach: { text: "slow down", kind: null, spoken: false } };
    expect(contextDetail({ context })).toBe("You 62% of the last 2 min · pace 142 wpm · coach: slow down");
    expect(contextDetail({})).toBe("");
    expect(contextDetail({ context: { ...context, talk: { ...context.talk, selfShare: null }, lastTurn: null, coach: null } })).toBe("");
  });
});

describe("ReviewSession", () => {
  const items = ["a", "b", "c"].map((id, k) => ({
    id, clip: `${id}.wav`, uri: `u/${id}.wav`, t: k, kind: "saved", reviews: [], tiny: P(0.7, 0, 0, 0, 0.3, 0), small: P(0, 0, 0, 0, 0.8, 0.2),
  })) as ReviewItemFull[];
  const mk = (mode: ReviewMode, rater = "me") => {
    const saved: [string, ReviewLabel][] = [];
    const cleared: [string, string][] = [];
    const s = new ReviewSession(items, { save: (id, l) => saved.push([id, l]), clear: (id, r) => cleared.push([id, r]), now: () => 99 }, { rater, mode });
    return { s, saved, cleared };
  };
  it("emotion-only: ONE tap per clip; Right = good with TINY's emotion, a chip = bad + truth", () => {
    const { s, saved } = mk("emotion");
    expect(s.answer({ right: "both" })).toBe("saved");
    expect(s.answer({ truth: "fear" })).toBe("saved");
    expect(saved).toEqual([
      ["a", { rater: "me", rating: "good", truth: "anger", t: 99 }],
      ["b", { rater: "me", rating: "bad", truth: "fear", t: 99 }],
    ]);
    expect(s.current()!.id).toBe("c");
    expect(s.position).toBe(3);
  });
  it("a chip equal to TINY's top is a thumbs up; 'SMALL right' picks SMALL's emotion", () => {
    const { s, saved } = mk("emotion");
    s.answer({ truth: "anger" });
    s.answer({ right: "small" });
    expect(saved[0][1]).toMatchObject({ rating: "good", truth: "anger" });
    expect(saved[1][1]).toMatchObject({ rating: "bad", truth: "neutral" });
  });
  it("emotion + nudge: TWO taps, saved only after the second, nudgeWarranted stored", () => {
    const { s, saved } = mk("emotion+nudge", "mom");
    expect(s.answer({ truth: "sadness" })).toBe("await-nudge");
    expect(s.stage).toBe("nudge");
    expect(saved).toEqual([]);
    expect(s.answerNudge("yes")).toBe(true);
    expect(saved).toEqual([["a", { rater: "mom", rating: "bad", truth: "sadness", nudgeWarranted: "yes", t: 99 }]]);
    expect(s.stage).toBe("emotion");
    expect(s.current()!.id).toBe("b");
    expect(s.answerNudge("no")).toBe(false); // nothing pending
  });
  it("undo: drops a pending answer, then returns to the previous clip and clears its label", () => {
    const { s, cleared } = mk("emotion+nudge");
    s.answer({ truth: "fear" });
    expect(s.undo()).toBe(true);
    expect(s.stage).toBe("emotion");
    s.answer({ right: "both" });
    s.answerNudge("no");
    s.skip();
    expect(s.current()!.id).toBe("c");
    expect(s.undo()).toBe(true); // undo the skip
    expect(s.current()!.id).toBe("b");
    expect(cleared).toEqual([]);
    expect(s.undo()).toBe(true); // undo clip a's label
    expect(s.current()!.id).toBe("a");
    expect(cleared).toEqual([["a", "me"]]);
    expect(s.undo()).toBe(false);
  });
  it("auto-advances after two plays with no tap, never while an answer is pending; ends done", () => {
    const { s } = mk("emotion+nudge");
    expect(s.onEnded()).toBe("replay");
    expect(s.onEnded()).toBe("advance");
    expect(s.current()!.id).toBe("b");
    s.answer({ truth: "fear" });
    expect(s.onEnded()).toBe("replay");
    expect(s.onEnded()).toBe("replay");
    expect(s.current()!.id).toBe("b");
    s.answerNudge("unsure");
    s.skip();
    expect(s.current()).toBeNull();
    expect(s.onEnded()).toBe("done");
    expect(s.answer({ right: "both" })).toBe(false);
  });
  it("summary: rated count and agreement with TINY and SMALL", () => {
    const { s } = mk("emotion");
    s.answer({ right: "both" }); // truth anger: tiny yes, small no
    s.answer({ truth: "neutral" }); // tiny no, small yes
    s.skip();
    expect(s.summary()).toEqual({ rated: 2, skipped: 1, tiny: { agree: 1, of: 2 }, small: { agree: 1, of: 2 } });
    expect(s.ratio).toBe(1);
  });
  it("describes what the models said", () => {
    expect(describeItem(items[0])).toBe("TINY: anger 70% · SMALL: neutral 80%");
    expect(describeItem({ ...items[0], small: null, kind: "mark" })).toBe("Marked moment. TINY: anger 70% · SMALL: —");
  });
});


describe("hub review controls work with the store alone", () => {
  it("rate() writes flag + truth and the queue shrinks", async () => {
    const s = setup(P(0, 0, 0, 0.9, 0.1, 0), NOT_SURE, { saveAudio: true });
    await s.feed(9);
    s.ctl.stop();
    await s.feed(9);
    s.ctl.stop();
    const hub = new EmotionDevHub(async () => ({ ok: false, code: "error", message: "n/a" }), () => s.store);
    const q = hub.reviewQueue();
    expect(q.length).toBeGreaterThan(1);
    hub.rate(q[0].id, "bad", "fear");
    expect(hub.reviewQueue()).toHaveLength(q.length - 1);
    expect(hub.mark()).toBeNull(); // no running tester -> nothing to mark
    expect(s.store.readRecords().find((r) => r.id === q[0].id)).toMatchObject({ flag: "bad", truth: "fear" });
  });
});
