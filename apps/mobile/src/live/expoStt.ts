/**
 * `SpeechRecognizer` over expo-speech-recognition (56.0.1): continuous,
 * interim, ON-DEVICE recognition. Android is pinned to the on-device
 * service (`com.google.android.as`) and the offline model download is
 * triggered ahead of the first session; iOS uses Apple Speech with
 * `requiresOnDeviceRecognition`.
 *
 * Lifecycle on Android (ExpoSpeechService.kt): EVERY `onError` — including
 * ERROR_NO_MATCH ("no-speech") and ERROR_SPEECH_TIMEOUT after a pause in the
 * conversation — calls `teardownAndEnd()`: the native recognizer is
 * destroyed and an `end` event fires. "Continuous" only means the session
 * is not ended after the first final result; it does NOT survive an error.
 * So a quiet stretch in a call would silently kill on-device STT for the
 * rest of the session unless someone restarts it. This class does: on `end`
 * without a `stop()` from us (and without a fatal error), it starts a fresh
 * native session after a short delay and tells the loop (`onRestart`) so
 * word timings — relative to recognizer start on Android — are re-based.
 * A restart storm (#165-style `stop()`→ERROR_CLIENT loops, a service that
 * ends immediately) is capped: past `maxRestartsPerWindow` in
 * `restartWindowMs` the recognizer gives up and reports `restart-loop`.
 *
 * Known upstream caveat (jamsch/expo-speech-recognition#165): `stop()` in
 * continuous mode can emit a `client` error on Android — it is swallowed
 * here after stop because the session is over anyway.
 */
import { Platform } from "react-native";
import type { SpeechRecognizer, SttResultEvent } from "./stt";

type SpeechModule = typeof import("expo-speech-recognition").ExpoSpeechRecognitionModule;

/**
 * The native module, resolved lazily: `expo-speech-recognition` calls
 * `requireNativeModule` at import time, which throws in any runtime that
 * lacks the native side (web, an older dev client, Jest without the mock).
 * Loading it on first use keeps the Live Coach screen importable
 * everywhere and turns "not installed" into a plain `null`.
 */
let cached: SpeechModule | null | undefined;
export function speechModule(): SpeechModule | null {
  if (cached !== undefined) return cached;
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    cached = (require("expo-speech-recognition") as typeof import("expo-speech-recognition")).ExpoSpeechRecognitionModule;
  } catch {
    cached = null;
  }
  return cached;
}

export const ANDROID_ON_DEVICE_SERVICE = "com.google.android.as";

/** Errors after which the recognizer is simply restarted (the session is
 *  torn down natively, but nothing is wrong with the device): idle
 *  timeouts, a busy/aborted service, a transient client or network hiccup. */
export const RESTARTABLE_STT_ERRORS = new Set([
  "no-speech",
  "speech-timeout",
  "aborted",
  "busy",
  "client",
  "network",
]);

export const DEFAULT_RESTART_DELAY_MS = 250;
export const DEFAULT_MAX_RESTARTS_PER_WINDOW = 8;
export const DEFAULT_RESTART_WINDOW_MS = 15_000;

export interface ExpoSttOptions {
  lang?: string;
  /** Delay before bringing a torn-down session back. */
  restartDelayMs?: number;
  /** Restart-storm guard: more than this many restarts within
   *  `restartWindowMs` reports `restart-loop` and stops restarting. */
  maxRestartsPerWindow?: number;
  restartWindowMs?: number;
  /** Clock for the storm guard (tests). */
  now?: () => number;
}

/** Synchronous capability probe: can this device run on-device STT now?
 *  Never throws — a missing native module means "no". */
export function onDeviceSttAvailable(): boolean {
  try {
    if (Platform.OS === "web") return false;
    const m = speechModule();
    if (!m || !m.isRecognitionAvailable()) return false;
    // iOS reports on-device support per locale via supportsOnDeviceRecognition;
    // Android 13+ exposes the on-device service package.
    if (Platform.OS === "ios") return m.supportsOnDeviceRecognition();
    const services = m.getSpeechRecognitionServices?.() ?? [];
    return m.supportsOnDeviceRecognition() || services.includes(ANDROID_ON_DEVICE_SERVICE);
  } catch {
    return false;
  }
}

/** Best-effort: kick off the Android offline model download (13+). */
export async function ensureOfflineModel(lang = "en-US"): Promise<void> {
  if (Platform.OS !== "android") return;
  try {
    await speechModule()?.androidTriggerOfflineModelDownload({ locale: lang });
  } catch {
    // Older Android or no on-device service: start() reports its own error.
  }
}

export class ExpoSpeechRecognizer implements SpeechRecognizer {
  private stopped = false;
  /** A non-restartable error was reported: never bring the session back. */
  private fatal = false;
  private subscriptions: { remove(): void }[] = [];
  private resultCbs: ((e: SttResultEvent) => void)[] = [];
  private errorCbs: ((code: string, message: string) => void)[] = [];
  private restartCbs: (() => void)[] = [];
  private restartTimer: ReturnType<typeof setTimeout> | null = null;
  private restartTimes: number[] = [];
  private readonly restartDelayMs: number;
  private readonly maxRestartsPerWindow: number;
  private readonly restartWindowMs: number;
  private readonly now: () => number;
  /** Restarts so far (for logs/tests). */
  restarts = 0;

  constructor(private readonly options: ExpoSttOptions = {}) {
    this.restartDelayMs = options.restartDelayMs ?? DEFAULT_RESTART_DELAY_MS;
    this.maxRestartsPerWindow = options.maxRestartsPerWindow ?? DEFAULT_MAX_RESTARTS_PER_WINDOW;
    this.restartWindowMs = options.restartWindowMs ?? DEFAULT_RESTART_WINDOW_MS;
    this.now = options.now ?? Date.now;
  }

  async start(): Promise<void> {
    const m = speechModule();
    if (!m) throw new Error("expo-speech-recognition native module not installed");
    const perm = await m.requestPermissionsAsync();
    if (!perm.granted) throw new Error("speech recognition permission denied");
    this.stopped = false;
    this.fatal = false;
    this.restartTimes = [];
    this.subscriptions.push(
      m.addListener("result", (e) => {
        const first = e.results?.[0];
        if (!first) return;
        const event: SttResultEvent = { text: first.transcript ?? "", isFinal: e.isFinal };
        if (e.isFinal && Array.isArray(first.segments) && first.segments.length > 0) {
          event.segments = first.segments.map((s) => ({
            startTimeMillis: s.startTimeMillis,
            endTimeMillis: s.endTimeMillis,
            segment: s.segment,
          }));
        }
        for (const cb of this.resultCbs) cb(event);
      }),
      m.addListener("error", (e) => {
        if (this.stopped) return; // #165: stop() itself can emit "client".
        if (RESTARTABLE_STT_ERRORS.has(e.error)) return; // the `end` that follows restarts us
        this.fatal = true;
        for (const cb of this.errorCbs) cb(e.error, e.message);
      }),
      m.addListener("end", () => {
        if (this.stopped || this.fatal) return;
        this.scheduleRestart();
      }),
    );
    this.startNative(m);
  }

  private startNative(m: SpeechModule) {
    m.start({
      lang: this.options.lang ?? "en-US",
      interimResults: true,
      continuous: true,
      requiresOnDeviceRecognition: true,
      addsPunctuation: true,
      maxAlternatives: 1,
      ...(Platform.OS === "android"
        ? { androidRecognitionServicePackage: ANDROID_ON_DEVICE_SERVICE }
        : {}),
    });
  }

  private scheduleRestart() {
    if (this.restartTimer !== null) return; // one pending restart at a time
    const t = this.now();
    this.restartTimes = this.restartTimes.filter((x) => t - x <= this.restartWindowMs);
    if (this.restartTimes.length >= this.maxRestartsPerWindow) {
      this.fatal = true;
      const msg = `recognizer ended ${this.restartTimes.length} times in ${Math.round(this.restartWindowMs / 1000)} s`;
      for (const cb of this.errorCbs) cb("restart-loop", msg);
      return;
    }
    this.restartTimes.push(t);
    this.restartTimer = setTimeout(() => {
      this.restartTimer = null;
      if (this.stopped || this.fatal) return;
      const m = speechModule();
      if (!m) return;
      try {
        this.startNative(m);
      } catch (err) {
        this.fatal = true;
        const msg = err instanceof Error ? err.message : String(err);
        for (const cb of this.errorCbs) cb("restart-failed", msg);
        return;
      }
      this.restarts += 1;
      for (const cb of this.restartCbs) cb();
    }, this.restartDelayMs);
  }

  stop(): void {
    this.stopped = true;
    if (this.restartTimer !== null) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
    try {
      speechModule()?.stop();
    } catch {
      // Already stopped / never started.
    }
    for (const s of this.subscriptions) {
      try {
        s.remove();
      } catch {
        // Listener already gone.
      }
    }
    this.subscriptions = [];
  }

  onResult(cb: (e: SttResultEvent) => void) {
    this.resultCbs.push(cb);
    return () => {
      this.resultCbs = this.resultCbs.filter((c) => c !== cb);
    };
  }

  onError(cb: (code: string, message: string) => void) {
    this.errorCbs.push(cb);
    return () => {
      this.errorCbs = this.errorCbs.filter((c) => c !== cb);
    };
  }

  onRestart(cb: () => void) {
    this.restartCbs.push(cb);
    return () => {
      this.restartCbs = this.restartCbs.filter((c) => c !== cb);
    };
  }
}
