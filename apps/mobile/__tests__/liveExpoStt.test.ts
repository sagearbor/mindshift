/**
 * src/live/expoStt.ts over jest-setup's expo-speech-recognition mock: the
 * Android lifecycle (ExpoSpeechService.kt tears the native session down on
 * EVERY onError — "no-speech" after a pause included — and emits "end"),
 * which the recognizer must survive by restarting itself.
 */
import { ExpoSpeechRecognizer } from "../src/live/expoStt";

interface SpeechMock {
  requestPermissionsAsync: jest.Mock;
  start: jest.Mock;
  stop: jest.Mock;
  __emit: (name: string, e: unknown) => void;
}

// jest-setup's factory runs on first require; the recognizer itself loads
// the module lazily, so force it here to get at the spies.
// eslint-disable-next-line @typescript-eslint/no-require-imports
require("expo-speech-recognition");
const mock = (globalThis as Record<string, unknown>).__expoSpeechRecognitionMock as SpeechMock;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

beforeEach(() => {
  mock.start.mockClear();
  mock.stop.mockClear();
  mock.requestPermissionsAsync.mockResolvedValue({ granted: true, status: "granted" });
});

describe("ExpoSpeechRecognizer restart-on-end", () => {
  it("restarts after a transient error + end, re-marks the clock, and stays quiet about it", async () => {
    const rec = new ExpoSpeechRecognizer({ restartDelayMs: 5 });
    const errors: string[] = [];
    let restarts = 0;
    rec.onError((code) => errors.push(code));
    rec.onRestart(() => (restarts += 1));
    await rec.start();
    expect(mock.start).toHaveBeenCalledTimes(1);
    expect(mock.start.mock.calls[0][0]).toMatchObject({ continuous: true, requiresOnDeviceRecognition: true });

    // A pause in the conversation on Android: ERROR_NO_MATCH -> "no-speech",
    // then the native session ends.
    mock.__emit("error", { error: "no-speech", message: "No speech was detected." });
    mock.__emit("end", null);
    await sleep(20);
    expect(mock.start).toHaveBeenCalledTimes(2);
    expect(restarts).toBe(1);
    expect(rec.restarts).toBe(1);
    expect(errors).toEqual([]);

    // Results after the restart still reach the loop (listeners persist).
    const texts: string[] = [];
    rec.onResult((e) => texts.push(e.text));
    mock.__emit("result", { isFinal: true, results: [{ transcript: "back again" }] });
    expect(texts).toEqual(["back again"]);
    rec.stop();
  });

  it("a fatal error is reported and NOT restarted; stop() never restarts either", async () => {
    const rec = new ExpoSpeechRecognizer({ restartDelayMs: 5 });
    const errors: string[] = [];
    rec.onError((code) => errors.push(code));
    await rec.start();
    mock.__emit("error", { error: "service-not-allowed", message: "no service" });
    mock.__emit("end", null);
    await sleep(20);
    expect(errors).toEqual(["service-not-allowed"]);
    expect(mock.start).toHaveBeenCalledTimes(1);
    rec.stop();

    const rec2 = new ExpoSpeechRecognizer({ restartDelayMs: 5 });
    const errors2: string[] = [];
    rec2.onError((code) => errors2.push(code));
    await rec2.start();
    rec2.stop();
    // #165: stop() itself can emit a "client" error, then "end".
    mock.__emit("error", { error: "client", message: "ERROR_CLIENT" });
    mock.__emit("end", null);
    await sleep(20);
    expect(errors2).toEqual([]);
    expect(mock.start).toHaveBeenCalledTimes(2);
    expect(mock.stop).toHaveBeenCalled();
  });

  it("gives up on a restart storm and says so once (restart-loop)", async () => {
    let t = 0;
    const rec = new ExpoSpeechRecognizer({
      restartDelayMs: 1,
      maxRestartsPerWindow: 3,
      restartWindowMs: 10_000,
      now: () => t,
    });
    const errors: string[] = [];
    let restarts = 0;
    rec.onError((code) => errors.push(code));
    rec.onRestart(() => (restarts += 1));
    await rec.start();
    for (let i = 0; i < 5; i++) {
      t += 100;
      mock.__emit("error", { error: "client", message: "ERROR_CLIENT" });
      mock.__emit("end", null);
      await sleep(10);
    }
    expect(restarts).toBe(3);
    expect(errors).toEqual(["restart-loop"]);
    expect(mock.start).toHaveBeenCalledTimes(4);
    rec.stop();
  });

  it("restarts that are spread out over time are not a storm", async () => {
    let t = 0;
    const rec = new ExpoSpeechRecognizer({
      restartDelayMs: 1,
      maxRestartsPerWindow: 2,
      restartWindowMs: 1000,
      now: () => t,
    });
    const errors: string[] = [];
    rec.onError((code) => errors.push(code));
    await rec.start();
    for (let i = 0; i < 5; i++) {
      t += 5000; // a "no-speech" every 5 s of quiet
      mock.__emit("error", { error: "speech-timeout", message: "No speech input." });
      mock.__emit("end", null);
      await sleep(10);
    }
    expect(errors).toEqual([]);
    expect(rec.restarts).toBe(5);
    rec.stop();
  });
});
