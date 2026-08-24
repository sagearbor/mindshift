/**
 * src/live/sttWeb.ts — the Web Speech API recognizer (iOS Safari /
 * Chrome) behind the same `SpeechRecognizer` seam the native path uses,
 * driven by a fake `webkitSpeechRecognition`.
 */
import {
  FATAL_SPEECH_ERRORS,
  WebSpeechRecognizer,
  webSpeechRecognitionCtor,
  webSttAvailable,
  type SpeechRecognitionEventLike,
  type SpeechRecognitionLike,
} from "../src/live/sttWeb";
import { TranscriptAligner, type SttResultEvent } from "../src/live/stt";

class FakeRecognition implements SpeechRecognitionLike {
  static instances: FakeRecognition[] = [];
  static throwOnStart: Error | null = null;
  lang = "";
  continuous = false;
  interimResults = false;
  maxAlternatives = 0;
  onresult: ((e: SpeechRecognitionEventLike) => void) | null = null;
  onerror: ((e: { error: string; message?: string }) => void) | null = null;
  onend: (() => void) | null = null;
  started = 0;
  stopped = 0;
  aborted = 0;
  constructor() {
    FakeRecognition.instances.push(this);
  }
  start() {
    if (FakeRecognition.throwOnStart) throw FakeRecognition.throwOnStart;
    this.started += 1;
  }
  stop() {
    this.stopped += 1;
  }
  abort() {
    this.aborted += 1;
  }
  /** Emit a results event shaped like the browser's (full list + resultIndex). */
  results(list: { text: string; isFinal: boolean }[], resultIndex = 0) {
    this.onresult?.({
      resultIndex,
      results: list.map((r) => ({ isFinal: r.isFinal, length: 1, 0: { transcript: r.text } })),
    });
  }
  end() {
    this.onend?.();
  }
  error(code: string, message = "") {
    this.onerror?.({ error: code, message });
  }
}

function harness(opts: { now?: () => number } = {}) {
  const timers: (() => void)[] = [];
  let t = 1000;
  const now = opts.now ?? (() => t);
  const rec = new WebSpeechRecognizer({
    ctor: FakeRecognition,
    now,
    restartDelayMs: 10,
    setTimeoutImpl: (fn) => {
      timers.push(fn);
      return 0;
    },
  });
  const results: SttResultEvent[] = [];
  const errors: string[] = [];
  rec.onResult((e) => results.push(e));
  rec.onError((c) => errors.push(c));
  return {
    rec,
    results,
    errors,
    timers,
    advance(ms: number) {
      t += ms;
    },
    runTimers() {
      const fns = timers.splice(0);
      for (const fn of fns) fn();
    },
    get browser() {
      return FakeRecognition.instances[FakeRecognition.instances.length - 1];
    },
  };
}

beforeEach(() => {
  FakeRecognition.instances = [];
  FakeRecognition.throwOnStart = null;
});

describe("availability", () => {
  it("finds the prefixed or unprefixed constructor and nothing else", () => {
    expect(webSttAvailable({})).toBe(false);
    expect(webSttAvailable({ webkitSpeechRecognition: FakeRecognition })).toBe(true);
    expect(webSttAvailable({ SpeechRecognition: FakeRecognition })).toBe(true);
    expect(webSpeechRecognitionCtor({ SpeechRecognition: "nope" })).toBeNull();
  });

  it("start() rejects honestly when the browser has no recognizer", async () => {
    const rec = new WebSpeechRecognizer({ ctor: null });
    await expect(rec.start()).rejects.toThrow(/no speech recognition/);
  });
});

describe("WebSpeechRecognizer", () => {
  it("starts continuous + interim recognition in the requested language, idempotently", async () => {
    const h = harness();
    await h.rec.start();
    await h.rec.start(); // the loop's own start after priming: no second instance
    expect(FakeRecognition.instances).toHaveLength(1);
    const b = h.browser;
    expect(b.started).toBe(1);
    expect(b.continuous).toBe(true);
    expect(b.interimResults).toBe(true);
    expect(b.lang).toBe("en-US");
    expect(h.rec.isStarted).toBe(true);
  });

  it("emits interims as they grow and a timed final when the browser flags it", async () => {
    const h = harness();
    await h.rec.start(); // epoch = 1000
    h.advance(200);
    h.browser.results([{ text: "you never", isFinal: false }]);
    h.advance(600);
    h.browser.results([{ text: "you never call", isFinal: false }]);
    h.advance(200);
    h.browser.results([{ text: "you never call", isFinal: true }]);

    expect(h.results.map((r) => r.isFinal)).toEqual([false, false, true]);
    expect(h.results[1].text).toBe("you never call");
    const final = h.results[2];
    expect(final.text).toBe("you never call");
    // Word timing is synthesized over the window the interims were seen in
    // (200 ms → 800 ms after the epoch — the final added nothing, so the
    // 200 ms it took to be flagged is not speech), evenly per word.
    expect(final.segments).toHaveLength(3);
    expect(final.segments![0].startTimeMillis).toBe(200);
    expect(final.segments![2].endTimeMillis).toBe(800);
    expect(final.segments!.map((s) => s.segment)).toEqual(["you", "never", "call"]);
  });

  it("a final that grew past its last interim extends a bounded window, never to 'now'", async () => {
    const h = harness();
    await h.rec.start();
    h.advance(100);
    h.browser.results([{ text: "you never", isFinal: false }]);
    h.advance(10000); // the final lands 10 s later with one more word
    h.browser.results([{ text: "you never call", isFinal: true }]);
    const final = h.results[h.results.length - 1];
    expect(final.segments![0].startTimeMillis).toBe(100);
    expect(final.segments![2].endTimeMillis).toBe(100 + 1500);
  });

  it("Safari-style: an interim still pending at `end` becomes the final, then it restarts", async () => {
    const h = harness();
    await h.rec.start();
    h.advance(100);
    h.browser.results([{ text: "I think", isFinal: false }]);
    h.advance(300);
    h.browser.results([{ text: "I think so", isFinal: false }]);
    h.advance(100);
    const first = h.browser;
    first.end();
    // The pending interim was finalized, timed over [100, 400] …
    const finals = h.results.filter((r) => r.isFinal);
    expect(finals).toHaveLength(1);
    expect(finals[0].text).toBe("I think so");
    expect(finals[0].segments![0].startTimeMillis).toBe(100);
    expect(finals[0].segments![2].endTimeMillis).toBe(400);
    // … and a fresh instance starts after the back-off.
    expect(h.timers).toHaveLength(1);
    h.runTimers();
    expect(FakeRecognition.instances).toHaveLength(2);
    expect(h.browser).not.toBe(first);
    expect(h.browser.started).toBe(1);
    expect(h.rec.restarts).toBe(1);
    // Segment times on the second run stay relative to the ORIGINAL epoch.
    h.advance(1000);
    h.browser.results([{ text: "yes", isFinal: true }]);
    const last = h.results[h.results.length - 1];
    expect(last.segments![0].endTimeMillis).toBe(1500);
  });

  it("no-speech / aborted are not errors; fatal codes are reported once and stop the restarts", async () => {
    const h = harness();
    await h.rec.start();
    h.browser.error("no-speech");
    h.browser.error("aborted");
    expect(h.errors).toEqual([]);
    h.browser.end();
    h.runTimers();
    expect(FakeRecognition.instances).toHaveLength(2);

    h.browser.error("not-allowed", "denied");
    h.browser.end();
    h.runTimers();
    expect(h.errors).toEqual(["not-allowed"]);
    expect(FakeRecognition.instances).toHaveLength(2); // no restart after a fatal error
    for (const code of ["not-allowed", "service-not-allowed", "audio-capture", "network"]) {
      expect(FATAL_SPEECH_ERRORS.has(code)).toBe(true);
    }
  });

  it("stop() ends recognition, drops the browser callbacks and prevents restarts", async () => {
    const h = harness();
    await h.rec.start();
    const b = h.browser;
    h.rec.stop();
    expect(b.stopped + b.aborted).toBeGreaterThan(0);
    expect(b.onresult).toBeNull();
    expect(b.onend).toBeNull();
    h.runTimers();
    expect(FakeRecognition.instances).toHaveLength(1);
    expect(h.rec.isStarted).toBe(false);
  });

  it("gives up after too many restarts that produced nothing", async () => {
    const timers: (() => void)[] = [];
    const rec = new WebSpeechRecognizer({
      ctor: FakeRecognition,
      restartDelayMs: 1,
      maxSilentRestarts: 2,
      setTimeoutImpl: (fn) => {
        timers.push(fn);
        return 0;
      },
    });
    const errors: string[] = [];
    rec.onError((c) => errors.push(c));
    await rec.start();
    for (let i = 0; i < 4; i++) {
      FakeRecognition.instances[FakeRecognition.instances.length - 1].end();
      timers.splice(0).forEach((fn) => fn());
    }
    expect(errors).toEqual(["restart-limit"]);
  });

  it("an error raised while primed (no listener yet) reaches the first onError subscriber", async () => {
    const rec = new WebSpeechRecognizer({ ctor: FakeRecognition, setTimeoutImpl: () => 0 });
    void rec.start();
    await Promise.resolve();
    FakeRecognition.instances[0].error("audio-capture", "mic busy");
    const got: string[] = [];
    rec.onError((c, m) => got.push(`${c}:${m}`));
    expect(got).toEqual(["audio-capture:mic busy"]);
  });

  it("a start() that throws synchronously (InvalidStateError) is reported as start-failed", async () => {
    FakeRecognition.throwOnStart = new Error("InvalidStateError");
    const h = harness();
    await h.rec.start();
    expect(h.errors).toEqual(["start-failed"]);
  });

  it("the synthesized timing lands words inside the VAD span the aligner asks about", async () => {
    // A 60-second-late Safari final must not smear over the whole minute:
    // the words are placed where their interims were seen.
    const h = harness();
    await h.rec.start();
    const aligner = new TranscriptAligner();
    aligner.markRecognizerStart(0); // recognizer epoch = session second 0
    h.rec.onResult((e) => aligner.push(e, (h.rec as unknown as { now: () => number }).now() / 1000));
    h.advance(2000); // t = 3000 => 2.0 s after the epoch
    h.browser.results([{ text: "hello there", isFinal: false }]);
    h.advance(800); // interim last seen at 2.8 s
    h.advance(60000); // …and the final only arrives a minute later
    h.browser.results([{ text: "hello there", isFinal: true }]);
    expect(aligner.textForSpan(1.9, 2.9).text).toBe("hello there");
    expect(aligner.textForSpan(60, 63).text).toBe("");
  });
});
