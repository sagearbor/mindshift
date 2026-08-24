/**
 * src/live/stt.ts — aligning the OS recognizer's stream of results to the
 * VAD's turn boundaries, in session seconds.
 */
import { FakeSpeechRecognizer, TranscriptAligner } from "../src/live/stt";

describe("TranscriptAligner", () => {
  it("uses word timing when the platform provides it (Android finals)", () => {
    const a = new TranscriptAligner();
    a.markRecognizerStart(0.5); // recognizer clock started 0.5 s into the session
    a.push(
      {
        text: "hello there how are you",
        isFinal: true,
        segments: [
          { startTimeMillis: 0, endTimeMillis: 400, segment: "hello" },
          { startTimeMillis: 400, endTimeMillis: 800, segment: "there" },
          { startTimeMillis: 2000, endTimeMillis: 2300, segment: "how" },
          { startTimeMillis: 2300, endTimeMillis: 2600, segment: "are" },
          { startTimeMillis: 2600, endTimeMillis: 2900, segment: "you" },
        ],
      },
      4.0,
    );
    // Words at session 0.5–1.3 s belong to a [0.4, 1.4] turn …
    expect(a.textForSpan(0.4, 1.4)).toEqual({ text: "hello there", final: true });
    // … and are consumed: a second overlapping query doesn't re-emit them.
    expect(a.textForSpan(0.4, 1.4)).toEqual({ text: "", final: true });
    // The later words (session 2.5–3.4 s) belong to the next turn.
    expect(a.textForSpan(2.4, 3.5)).toEqual({ text: "how are you", final: true });
  });

  it("spreads untimed finals over the window since the previous final (iOS)", () => {
    const a = new TranscriptAligner();
    a.markRecognizerStart(0);
    a.push({ text: "one two three four", isFinal: true }, 2.0); // window [0, 2]
    a.push({ text: "five six", isFinal: true }, 4.0); // window [2, 4]
    expect(a.textForSpan(0.0, 1.9).text).toBe("one two three four");
    expect(a.textForSpan(2.1, 3.9).text).toBe("five six");
  });

  it("falls back to an overlapping interim result, flagged non-final", () => {
    const a = new TranscriptAligner();
    a.markRecognizerStart(0);
    a.push({ text: "I think", isFinal: false }, 1.0);
    expect(a.textForSpan(0.2, 1.1)).toEqual({ text: "I think", final: false });
    // A final replaces the interim.
    a.push({ text: "I think so", isFinal: true }, 1.5);
    expect(a.textForSpan(0.2, 1.4)).toEqual({ text: "I think so", final: true });
    expect(a.textForSpan(5, 6)).toEqual({ text: "", final: true });
  });

  it("slack tolerates STT/VAD edge disagreement but not a different turn", () => {
    const a = new TranscriptAligner(0.35);
    a.markRecognizerStart(0);
    a.push(
      { text: "late", isFinal: true, segments: [{ startTimeMillis: 1100, endTimeMillis: 1300, segment: "late" }] },
      2,
    );
    expect(a.textForSpan(0, 1.0).text).toBe("late"); // midpoint 1.2 within 1.0 + 0.35
    a.push(
      { text: "far", isFinal: true, segments: [{ startTimeMillis: 3000, endTimeMillis: 3200, segment: "far" }] },
      4,
    );
    expect(a.textForSpan(0, 1.0).text).toBe("");
    expect(a.hasFinalCovering(3.0)).toBe(true);
  });

  it("reset drops words and interims", () => {
    const a = new TranscriptAligner();
    a.markRecognizerStart(0);
    a.push({ text: "x", isFinal: true }, 1);
    a.push({ text: "y", isFinal: false }, 2);
    a.reset();
    expect(a.textForSpan(0, 3)).toEqual({ text: "", final: true });
  });
});

describe("FakeSpeechRecognizer", () => {
  it("fans events out and unsubscribes", async () => {
    const r = new FakeSpeechRecognizer();
    const got: string[] = [];
    const off = r.onResult((e) => got.push(e.text));
    const errs: string[] = [];
    r.onError((c) => errs.push(c));
    await r.start();
    r.emit({ text: "a", isFinal: true });
    off();
    r.emit({ text: "b", isFinal: true });
    r.emitError("network");
    r.stop();
    expect(got).toEqual(["a"]);
    expect(errs).toEqual(["network"]);
    expect(r.started && r.stopped).toBe(true);
  });
});
