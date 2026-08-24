/**
 * Production wiring for the fast loop — the live-path analogue of
 * src/recorder/defaultDeps.ts. Built lazily per session so tests (which
 * inject fakes through `useAudioStream`'s `makeFastLoop` option) and web
 * never construct native objects.
 *
 * Degradation ladder, each rung independent:
 *   Silero VAD  -> energy VAD when the ONNX model can't load
 *   ECAPA + voiceprints -> speaker-ID off (turns are "Unknown"/"Speaker A")
 *   OS model / bundled LLM -> the cloud's suggestion event
 *   on-device STT -> handled upstream: without it the loop isn't started
 */
import { Platform } from "react-native";
import { ecapaModelUrl, fetchVoiceprints, authHeaders } from "../api/liveSessions";
import { FastLoop, type FastLoopDeps } from "./fastLoop";
import { EnergyVad, SileroVad, type FrameVad } from "./vad";
import { EcapaEmbedder, SpeakerLabeler, type Embedder, type EnrolledPerson } from "./speakerId";
import { ensureOfflineModel, ExpoSpeechRecognizer } from "./expoStt";
import {
  bundledModelProvider,
  cloudProvider,
  osModelProvider,
  ProviderChain,
  type ExpoAiKitLike,
  type ProviderName,
} from "./localLlm";
import type { HapticSink } from "./nudgePolicy";

/** The callbacks the hook supplies; everything else is wired here. */
export type FastLoopHandlers = Pick<FastLoopDeps, "speak" | "send" | "onTurn" | "onNudge" | "onSttError">;

export interface DefaultFastLoopOptions {
  providerOrder?: ProviderName[];
  /** Skip the ECAPA download / voiceprint fetch (e.g. no network). */
  speakerId?: boolean;
  lang?: string;
}

export interface FastLoopBuild {
  loop: FastLoop;
  /** Human-readable summary of what actually loaded, for the UI. */
  status: string;
}

// Native packages are resolved lazily inside the builders: expo-ai-kit (and
// friends) call requireNativeModule at import time, which would throw on
// web or in a dev client built before these modules were added. A missing
// package is just another rung down the degradation ladder.
function tryRequire<T>(name: string): T | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    return require(name) as T;
  } catch {
    return null;
  }
}

const expoHaptics: HapticSink = {
  async nudge(level) {
    try {
      const Haptics = tryRequire<typeof import("expo-haptics")>("expo-haptics");
      if (!Haptics) return;
      const style =
        level >= 3
          ? Haptics.ImpactFeedbackStyle.Heavy
          : level === 2
            ? Haptics.ImpactFeedbackStyle.Medium
            : Haptics.ImpactFeedbackStyle.Light;
      await Haptics.impactAsync(style);
    } catch {
      // No haptic engine (simulator, web): the on-screen flash still shows.
    }
  },
};

function ortNative(): typeof import("./ortNative") | null {
  return tryRequire<typeof import("./ortNative")>("./ortNative");
}

async function buildVad(): Promise<{ vad: FrameVad; name: string }> {
  const session = await ortNative()?.loadSileroSession();
  if (session) return { vad: new SileroVad(session), name: "Silero VAD" };
  return { vad: new EnergyVad(), name: "energy VAD" };
}

async function buildSpeakerId(): Promise<{ embedder: Embedder | null; labeler: SpeakerLabeler | null; name: string }> {
  let people: EnrolledPerson[] = [];
  try {
    people = await fetchVoiceprints();
  } catch {
    people = [];
  }
  const session = await ortNative()?.loadEcapaSession(ecapaModelUrl(), await authHeaders(false));
  if (!session) return { embedder: null, labeler: null, name: "speaker-ID off (no ECAPA model)" };
  return {
    embedder: new EcapaEmbedder(session),
    labeler: new SpeakerLabeler(people),
    name: `speaker-ID on (${people.length} enrolled)`,
  };
}

function buildLlm(order?: ProviderName[]): ProviderChain {
  const kit = tryRequire<ExpoAiKitLike>("expo-ai-kit");
  if (!kit) return new ProviderChain([cloudProvider()], order);
  const builtIn = Platform.OS === "ios" ? "apple-fm" : "mlkit";
  return new ProviderChain(
    [osModelProvider(kit, builtIn), bundledModelProvider(kit), cloudProvider()],
    order,
  );
}

/** Build a production FastLoop. Never throws for a missing optional piece. */
export async function createDefaultFastLoop(
  handlers: FastLoopHandlers,
  options: DefaultFastLoopOptions = {},
): Promise<FastLoopBuild> {
  void ensureOfflineModel(options.lang);
  const [{ vad, name: vadName }, speaker] = await Promise.all([
    buildVad(),
    options.speakerId === false
      ? Promise.resolve({ embedder: null, labeler: null, name: "speaker-ID off" })
      : buildSpeakerId(),
  ]);
  const llm = buildLlm(options.providerOrder);
  const loop = new FastLoop({
    ...handlers,
    vad,
    embedder: speaker.embedder,
    labeler: speaker.labeler,
    recognizer: new ExpoSpeechRecognizer({ lang: options.lang }),
    llm,
    haptics: expoHaptics,
  });
  return {
    loop,
    status: `${vadName} · ${speaker.name} · LLM ${llm.providerNames.join(" → ")}`,
  };
}
