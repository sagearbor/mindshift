import { EMOTION_DEV_DEFAULTS, EMOTION_LABELS } from "../src/emotion/config";
import {
  DailyCap,
  NudgeGate,
  RollingRateCap,
  applyOffsets,
  isProbVector,
  modelsDisagree,
  topClass,
  windowHasSpeechEnergy,
} from "../src/emotion/logic";
import { ClipScheduler, PcmRing } from "../src/emotion/pcmRing";

/** These cases cover the original TINY-only rule (still a setting); the default
 *  "both models agree" rule is covered in emotionReview.test.ts. */
function tinyGate(cfg: typeof EMOTION_DEV_DEFAULTS): NudgeGate {
  const g = new NudgeGate(cfg);
  g.setRule("tiny");
  return g;
}

const P = (anger = 0, disgust = 0, fear = 0, happiness = 0, neutral = 0, sadness = 0) => [anger, disgust, fear, happiness, neutral, sadness];
const H = 3_600_000;

describe("probabilities", () => {
  it("topClass names the label", () => {
    expect(topClass(P(0.1, 0, 0, 0, 0.6, 0.3)).label).toBe("neutral");
    expect(EMOTION_LABELS[topClass(P(0.7, 0, 0, 0, 0.3, 0)).index]).toBe("anger");
  });
  it("offsets are a no-op when all zero (shipped setting) and shift mass otherwise", () => {
    const p = P(0.1, 0.1, 0.1, 0.1, 0.5, 0.1);
    expect(applyOffsets(p, [0, 0, 0, 0, 0, 0])).toEqual(p);
    const q = applyOffsets(p, [1, 0, 0, 0, -1, 0]);
    expect(q.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
    expect(q[0]).toBeGreaterThan(p[0]);
    expect(q[4]).toBeLessThan(p[4]);
  });
  it("isProbVector rejects bad model output", () => {
    expect(isProbVector(P(0.5, 0, 0, 0, 0.5, 0))).toBe(true);
    expect(isProbVector([0.5, 0.5])).toBe(false);
    expect(isProbVector(P(NaN, 0, 0, 0, 1, 0))).toBe(false);
    expect(isProbVector(P(0.5, 0, 0, 0, 0.6, 0))).toBe(false);
  });
});

describe("speech-energy gate", () => {
  const tone = (amp: number, n = 96000) => Int16Array.from({ length: n }, (_, i) => Math.round(amp * Math.sin(i / 5)));
  it("passes loud audio, blocks quiet room noise and silence", () => {
    expect(windowHasSpeechEnergy(tone(8000))).toBe(true);
    expect(windowHasSpeechEnergy(tone(20))).toBe(false);
    expect(windowHasSpeechEnergy(new Int16Array(96000))).toBe(false);
  });
  it("needs a minimum fraction of active frames", () => {
    const x = new Int16Array(96000);
    x.set(tone(8000, 9600), 0); // 10% active < 30%
    expect(windowHasSpeechEnergy(x)).toBe(false);
    x.set(tone(8000, 48000), 0); // 50%
    expect(windowHasSpeechEnergy(x)).toBe(true);
  });
});

describe("rate caps", () => {
  it("rolling cap: max per hour, then frees as events age out", () => {
    const c = new RollingRateCap(2, H);
    expect(c.tryFire(0)).toBe(true);
    expect(c.tryFire(1000)).toBe(true);
    expect(c.tryFire(2000)).toBe(false);
    expect(c.tryFire(H + 1)).toBe(true); // first aged out
    expect(c.tryFire(H + 2)).toBe(false);
  });
  it("min gap holds even under the count", () => {
    const c = new RollingRateCap(10, H, 120_000);
    expect(c.tryFire(0)).toBe(true);
    expect(c.tryFire(60_000)).toBe(false);
    expect(c.tryFire(120_000)).toBe(true);
  });
  it("a refused fire records nothing", () => {
    const c = new RollingRateCap(1, H);
    c.tryFire(0);
    c.tryFire(10);
    expect(c.tryFire(H + 1)).toBe(true);
  });
  it("max is adjustable at runtime (the configurable buzzes/hour)", () => {
    const c = new RollingRateCap(1, H);
    c.tryFire(0);
    c.setMax(3);
    expect(c.tryFire(10)).toBe(true);
  });
  it("daily cap resets on the next local day", () => {
    const d = new DailyCap(2);
    const day1 = new Date(2026, 10, 7, 10).getTime();
    const day2 = new Date(2026, 10, 8, 10).getTime();
    expect([d.tryFire(day1), d.tryFire(day1 + 1), d.tryFire(day1 + 2)]).toEqual([true, true, false]);
    expect(d.tryFire(day2)).toBe(true);
  });
});

describe("nudge gate (TINY)", () => {
  const cfg = { ...EMOTION_DEV_DEFAULTS, nudgeConfirmWindows: 2, maxNudgesPerHour: 2, minNudgeGapS: 0 };
  const angry = P(0.7, 0, 0, 0, 0.3, 0);
  it("never nudges on neutral or below threshold", () => {
    const g = tinyGate(cfg);
    expect(g.decide(P(0, 0, 0, 0, 0.9, 0.1), 0).reason).toBe("neutral");
    expect(g.decide(P(0.4, 0, 0, 0, 0.35, 0.25), 1).reason).toBe("below-threshold");
  });
  it("needs the same class on consecutive windows", () => {
    const g = tinyGate(cfg);
    expect(g.decide(angry, 0)).toEqual({ nudge: false, reason: "unconfirmed" });
    expect(g.decide(angry, 3000).nudge).toBe(true);
    const g2 = tinyGate(cfg);
    g2.decide(angry, 0);
    g2.decide(P(0, 0, 0, 0, 1, 0), 3000); // a neutral window breaks the streak
    expect(g2.decide(angry, 6000).nudge).toBe(false);
  });
  it("different classes do not confirm each other", () => {
    const g = tinyGate(cfg);
    g.decide(angry, 0);
    expect(g.decide(P(0, 0, 0.7, 0, 0.3, 0), 3000).nudge).toBe(false);
  });
  it("is rate-capped per hour", () => {
    const g = tinyGate(cfg);
    const fire = (t: number) => {
      const a = g.decide(angry, t).nudge;
      return a || g.decide(angry, t + 1).nudge;
    };
    expect(fire(0)).toBe(true);
    expect(fire(10_000)).toBe(true);
    expect(fire(20_000)).toBe(false);
    expect(g.decide(angry, 30_000).reason).toBe("rate-capped");
    expect(fire(H + 30_000)).toBe(true);
  });
  it("honours the allowed-label list", () => {
    const g = tinyGate({ ...cfg, nudgeLabels: ["anger"] });
    g.decide(P(0, 0, 0, 0.8, 0.2, 0), 0);
    expect(g.decide(P(0, 0, 0, 0.8, 0.2, 0), 1).nudge).toBe(false);
  });
});

describe("disagreement trigger", () => {
  it("fires only when top classes differ and BOTH are confident", () => {
    expect(modelsDisagree(P(0.7, 0, 0, 0, 0.3, 0), P(0, 0, 0, 0, 0.8, 0.2), 0.6)).toBe(true);
    expect(modelsDisagree(P(0.7, 0, 0, 0, 0.3, 0), P(0.8, 0, 0, 0, 0.2, 0), 0.6)).toBe(false); // agree
    expect(modelsDisagree(P(0.5, 0, 0, 0, 0.5, 0), P(0, 0, 0, 0, 0.8, 0.2), 0.6)).toBe(false); // tiny unsure
    expect(modelsDisagree(P(0.7, 0, 0, 0, 0.3, 0), P(0, 0, 0, 0, 0.55, 0.45), 0.6)).toBe(false); // small unsure
  });
});

describe("pcm ring + flag clip windows", () => {
  const SR = 16000;
  const ramp = (from: number, n: number) => Int16Array.from({ length: n }, (_, i) => (from + i) % 30000);

  it("ring keeps only its capacity and reads absolute positions", () => {
    const r = new PcmRing(100);
    r.push(ramp(0, 60));
    r.push(ramp(60, 60));
    r.push(ramp(120, 60));
    expect(r.total).toBe(180);
    expect(r.total - r.oldest).toBeGreaterThanOrEqual(100); // never keeps less than its capacity
    expect(Array.from(r.read(170, 180))).toEqual(Array.from(ramp(170, 10)));
    expect(r.read(0, 10).length).toBe(0); // evicted
    expect(Array.from(r.tail(5))).toEqual(Array.from(ramp(175, 5)));
  });

  it("a flag clip = 45 s before + 15 s after, completing only when the after-part arrived", () => {
    const ring = new PcmRing(60 * SR);
    const sched = new ClipScheduler(ring);
    for (let s = 0; s < 100; s++) {
      const f = new Int16Array(SR).fill(s + 1);
      ring.push(f);
      sched.onPcm(f);
    }
    sched.request({ id: "a", reason: "flag:bad", atMs: 1 }, 45, 15);
    let done = [];
    for (let s = 0; s < 14; s++) {
      const f = new Int16Array(SR).fill(200 + s);
      ring.push(f);
      done.push(...sched.onPcm(f));
    }
    expect(done).toHaveLength(0);
    const f = new Int16Array(SR).fill(214);
    ring.push(f);
    done = sched.onPcm(f);
    expect(done).toHaveLength(1);
    const clip = done[0];
    expect(clip.pcm.length).toBe(60 * SR);
    expect(clip.beforeS).toBe(45);
    expect(clip.afterS).toBe(15);
    expect(clip.pcm[0]).toBe(100 - 45 + 1); // oldest "before" second = second 56 -> value 56
    expect(clip.pcm[45 * SR]).toBe(200); // first "after" second
    expect(clip.pcm[60 * SR - 1]).toBe(214);
  });

  it("with less than 45 s of history the clip is shorter and says so", () => {
    const ring = new PcmRing(60 * SR);
    const sched = new ClipScheduler(ring);
    const f = new Int16Array(10 * SR).fill(7);
    ring.push(f);
    sched.onPcm(f);
    sched.request({ id: "b", reason: "flag:good", atMs: 1 }, 45, 15);
    const out = sched.onPcm(new Int16Array(15 * SR).fill(8));
    expect(out[0].beforeS).toBe(10);
    expect(out[0].pcm.length).toBe(25 * SR);
  });

  it("flushAll finishes a pending clip with the audio that exists", () => {
    const ring = new PcmRing(60 * SR);
    const sched = new ClipScheduler(ring);
    const f = new Int16Array(5 * SR).fill(3);
    ring.push(f);
    sched.onPcm(f);
    sched.request({ id: "c", reason: "flag:ignore", atMs: 1 }, 45, 15);
    sched.onPcm(new Int16Array(2 * SR).fill(4));
    const out = sched.flushAll();
    expect(out[0].afterS).toBe(2);
    expect(sched.pendingCount).toBe(0);
  });

  it("overlapping flags each get their own complete clip", () => {
    const ring = new PcmRing(60 * SR);
    const sched = new ClipScheduler(ring);
    const feed = (n: number, v: number) => {
      const f = new Int16Array(n * SR).fill(v);
      ring.push(f);
      return sched.onPcm(f);
    };
    feed(20, 1);
    sched.request({ id: "x", reason: "flag:bad", atMs: 1 }, 45, 15);
    feed(5, 2);
    sched.request({ id: "y", reason: "flag:good", atMs: 2 }, 45, 15);
    const a = feed(10, 3); // x completes (15 s after)
    expect(a.map((c) => c.id)).toEqual(["x"]);
    const b = feed(5, 4); // y completes
    expect(b.map((c) => c.id)).toEqual(["y"]);
  });
});
