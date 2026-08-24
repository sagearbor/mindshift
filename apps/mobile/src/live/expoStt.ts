/**
 * `SpeechRecognizer` over expo-speech-recognition (56.0.1): continuous,
 * interim, ON-DEVICE recognition. Android is pinned to the on-device
 * service (`com.google.android.as`) and the offline model download is
 * triggered ahead of the first session; iOS uses Apple Speech with
 * `requiresOnDeviceRecognition`.
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

export interface ExpoSttOptions {
  lang?: string;
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
  private subscriptions: { remove(): void }[] = [];
  private resultCbs: ((e: SttResultEvent) => void)[] = [];
  private errorCbs: ((code: string, message: string) => void)[] = [];

  constructor(private readonly options: ExpoSttOptions = {}) {}

  async start(): Promise<void> {
    const m = speechModule();
    if (!m) throw new Error("expo-speech-recognition native module not installed");
    const perm = await m.requestPermissionsAsync();
    if (!perm.granted) throw new Error("speech recognition permission denied");
    this.stopped = false;
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
        for (const cb of this.errorCbs) cb(e.error, e.message);
      }),
    );
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

  stop(): void {
    this.stopped = true;
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
}
