/**
 * Production wiring of the Emotion dev tester: sideloaded model files ->
 * ONNX Runtime React Native sessions -> {@link EmotionDevController}, plus the
 * server-relayed watch link and its control poller. Never imported by tests
 * (they drive the pieces with fakes); every failure comes back as a
 * {@link RuntimeBuild} with an honest message, never a thrown crash.
 */
import { Platform } from "react-native";
import * as Application from "expo-application";
import { Paths } from "expo-file-system";
import { ExpoRecorderFs } from "../recorder/expoFs";
import { authHeaders } from "../api/liveSessions";
import { nativeOrtSessionFactory } from "../live/ortNative";
import { EmotionDevController } from "./devController";
import type { EmotionRuntime, RuntimeBuild } from "./devHub";
import { EmotionDevStore } from "./devStore";
import { OnnxEmotionEngine, type EmotionEngine } from "./engine";
import { androidExternalEmotionDir, modelSearchDirs, resolveBoth, type ModelProbeFs } from "./modelFiles";
import { ControlPoller, ServerWatchLink } from "./watchLink";
import type { EmotionDevUserSettings } from "./config";

const API_URL = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";
const CONTROL_POLL_MS = 2000;

function externalDir(): string | null {
  if (Platform.OS !== "android") return null;
  const id = Application.applicationId;
  return id ? androidExternalEmotionDir(id) : null;
}

/** Where logs + clips live: the external app folder when writable (so the
 *  owner can `adb pull` clips), else the app's documents folder. */
function dataDir(fs: ExpoRecorderFs): string {
  const ext = externalDir();
  if (ext) {
    try {
      fs.ensureDir(`${ext}/data`);
      return `${ext}/data`;
    } catch {
      // fall through to documents
    }
  }
  return `${Paths.document.uri.replace(/\/$/, "")}/emotion-dev`;
}

export function openNativeStore(): EmotionDevStore | null {
  if (Platform.OS === "web") return null;
  try {
    const fs = new ExpoRecorderFs();
    return new EmotionDevStore(fs, dataDir(fs));
  } catch {
    return null;
  }
}

export async function buildNativeRuntime(settings: EmotionDevUserSettings): Promise<RuntimeBuild> {
  if (Platform.OS === "web") {
    return { ok: false, code: "error", message: "the Emotion dev tester needs the native app (on-device ONNX models)" };
  }
  try {
    const fs = new ExpoRecorderFs();
    const probe: ModelProbeFs = {
      stat: (uri) => ({ exists: fs.exists(uri), size: fs.sizeOf(uri) ?? 0 }),
    };
    const dirs = modelSearchDirs({ documentDirUri: Paths.document.uri, androidExternalDirUri: externalDir() });
    const found = resolveBoth(probe, dirs);
    if (found.tiny.status !== "ready") {
      return { ok: false, code: "models-missing", message: `Models missing — ${found.tiny.reason}` };
    }
    const factory = nativeOrtSessionFactory();
    let tiny: EmotionEngine;
    try {
      tiny = new OnnxEmotionEngine("tiny", await factory(found.tiny.path));
    } catch (e) {
      return { ok: false, code: "error", message: `TINY failed to load: ${e instanceof Error ? e.message : String(e)}` };
    }
    let small: EmotionEngine | null = null;
    let smallReason: string | null = null;
    if (found.small.status === "ready") {
      try {
        small = new OnnxEmotionEngine("small", await factory(found.small.path));
      } catch (e) {
        smallReason = `SMALL failed to load: ${e instanceof Error ? e.message : String(e)}`;
      }
    } else {
      smallReason = found.small.reason;
    }

    const store = new EmotionDevStore(fs, dataDir(fs));
    const link = new ServerWatchLink(API_URL, () => authHeaders(true), fetch as never);
    const controller = new EmotionDevController({ tiny, small, store, link, showSuggestion: settings.nudgeSuggestion });
    const poller = new ControlPoller(link, (c) => controller.handleControl(c));
    const timer = setInterval(() => void poller.pollOnce(), CONTROL_POLL_MS);

    const runtime: EmotionRuntime = {
      controller,
      store,
      models: { tiny: true, small: small !== null, smallReason },
      async dispose() {
        clearInterval(timer);
        await tiny.release();
        await small?.release();
      },
    };
    return { ok: true, runtime };
  } catch (e) {
    return { ok: false, code: "error", message: e instanceof Error ? e.message : String(e) };
  }
}
