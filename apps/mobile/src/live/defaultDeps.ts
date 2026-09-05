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
 *
 * Speaker-ID is real end to end since the seam PR: the ECAPA ONNX export
 * comes from `GET|HEAD /models/ecapa.onnx` (download once, ETag re-check
 * each launch — src/live/modelDownload.ts) and the enrolled voiceprints from
 * `GET /voice/people?include_embeddings=true`. A server that can't serve
 * the model (503: voice deps absent) or an older server (404) leaves
 * speaker-ID OFF with the reason in `FastLoopBuild.status` /
 * `.capabilities.speakerId` — one console line, never an error toast.
 */
import { Platform } from "react-native";
import { ecapaModelUrl, ECAPA_REVISION, fetchVoiceprints, authHeaders } from "../api/liveSessions";
import { FastLoop, type FastLoopDeps } from "./fastLoop";
import { EnergyVad, SileroVad, type FrameVad } from "./vad";
import { EcapaEmbedder, SpeakerLabeler, type Embedder } from "./speakerId";
import {
  activeCapability,
  describeSpeakerId,
  inactiveCapability,
  peopleForModel,
  type SpeakerIdCapability,
} from "./speakerIdSetup";
import { ensureOfflineModel, ExpoSpeechRecognizer } from "./expoStt";
import type { SpeechRecognizer } from "./stt";
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
export type FastLoopHandlers = Pick<
  FastLoopDeps,
  "speak" | "send" | "onTurn" | "onNudge" | "onSttError" | "onDegrade"
> & {
  /** Progress while the loop is being built ("Downloading voice model … 42 %").
   *  The web build uses it; native builds are quick enough not to. */
  onStatus?: (line: string) => void;
  /** A recognizer already started inside the user gesture (web: iOS Safari
   *  gates speech permission on one). Native ignores it. */
  recognizer?: SpeechRecognizer | null;
};

export interface DefaultFastLoopOptions {
  providerOrder?: ProviderName[];
  /** Skip the ECAPA download / voiceprint fetch (e.g. no network). */
  speakerId?: boolean;
  lang?: string;
}

export interface FastLoopCapabilities {
  vad: "silero" | "energy";
  speakerId: SpeakerIdCapability;
  /** Provider chain in fallback order (ProviderChain.providerNames). */
  llm: string[];
}

export interface FastLoopBuild {
  loop: FastLoop;
  /** Human-readable summary of what actually loaded, for the UI. */
  status: string;
  /** The same, structured — which loop stages are actually active. */
  capabilities: FastLoopCapabilities;
}

// Native packages are resolved lazily inside the builders: expo-ai-kit (and
// friends) call requireNativeModule at import time, which would throw on
// web or in a dev client built before these modules were added. A missing
// package is just another rung down the degradation ladder. The require
// calls stay literal strings inside the thunks — Metro only accepts static
// `require("...")` (a `require(name)` variable form fails the bundle).
function tryRequire<T>(load: () => T): T | null {
  try {
    return load();
  } catch {
    return null;
  }
}

export const expoHaptics: HapticSink = {
  async nudge(level) {
    try {
      const Haptics = tryRequire(
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        () => require("expo-haptics") as typeof import("expo-haptics"),
      );
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
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return tryRequire(() => require("./ortNative") as typeof import("./ortNative"));
}

/** The VAD rung on its own (shared with the Journal mode, journalDeps.ts). */
export async function buildVad(): Promise<{ vad: FrameVad; name: string }> {
  const session = await ortNative()?.loadSileroSession();
  if (session) return { vad: new SileroVad(session), name: "Silero VAD" };
  return { vad: new EnergyVad(), name: "energy VAD" };
}

export interface SpeakerIdBuild {
  embedder: Embedder | null;
  labeler: SpeakerLabeler | null;
  capability: SpeakerIdCapability;
}

function speakerIdOff(reason: string): SpeakerIdBuild {
  // One log line, not an error: speaker-ID is an optional rung.
  console.log(`[live] speaker-ID off: ${reason}`);
  return { embedder: null, labeler: null, capability: inactiveCapability(reason) };
}

/** The speaker-ID rung on its own (shared with the Journal mode). */
export async function buildSpeakerId(): Promise<SpeakerIdBuild> {
  const native = ortNative();
  if (!native) return speakerIdOff("native ONNX Runtime unavailable");
  // Model download/revalidation and the voiceprint fetch are independent
  // network calls: run them together so a cold launch pays the longer one.
  const [voiceprints, loaded] = await Promise.all([
    fetchVoiceprints(),
    native.loadEcapaSession(ecapaModelUrl(), await authHeaders(false)),
  ]);
  if (!loaded.session || loaded.model.status !== "ready") {
    const reason = loaded.model.status === "ready" ? "ONNX session failed" : loaded.model.reason;
    return speakerIdOff(reason);
  }
  const { kept, dropped } = peopleForModel(voiceprints.people, ECAPA_REVISION);
  if (dropped.length > 0) {
    console.log(
      `[live] speaker-ID: skipped ${dropped.length} voiceprint(s) from another model revision`,
    );
  }
  const capability = activeCapability(loaded.model, kept, dropped.length, voiceprints.error);
  return {
    embedder: new EcapaEmbedder(loaded.session),
    labeler: new SpeakerLabeler(kept),
    capability,
  };
}

function buildLlm(order?: ProviderName[]): ProviderChain {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const kit = tryRequire(() => require("expo-ai-kit") as ExpoAiKitLike);
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
      ? Promise.resolve<SpeakerIdBuild>({
          embedder: null,
          labeler: null,
          capability: inactiveCapability("disabled for this session"),
        })
      : buildSpeakerId(),
  ]);
  const llm = buildLlm(options.providerOrder);
  // The web-only extras never reach the loop's deps.
  const { recognizer: _primed, onStatus: _status, ...loopHandlers } = handlers;
  void _primed;
  void _status;
  const loop = new FastLoop({
    ...loopHandlers,
    vad,
    embedder: speaker.embedder,
    labeler: speaker.labeler,
    recognizer: new ExpoSpeechRecognizer({ lang: options.lang }),
    llm,
    haptics: expoHaptics,
  });
  const capabilities: FastLoopCapabilities = {
    vad: vad instanceof SileroVad ? "silero" : "energy",
    speakerId: speaker.capability,
    llm: llm.providerNames,
  };
  return {
    loop,
    status: `${vadName} · ${describeSpeakerId(speaker.capability)} · LLM ${llm.providerNames.join(" → ")}`,
    capabilities,
  };
}

/**
 * Pre-flight: what the fast loop WOULD load right now, without starting a
 * session — the honest capability check the Live Coach screen shows before
 * "Start" (on-device STT is gated upstream by `detectLiveCapability`).
 * Runs the same builders a session start runs (so the ECAPA model +
 * voiceprints are warm afterwards and the real start is fast); every
 * failure is a reason line, never a throw.
 */
export async function probeFastLoopCapabilities(
  options: DefaultFastLoopOptions = {},
): Promise<FastLoopCapabilities> {
  const [{ vad }, speaker] = await Promise.all([
    buildVad().catch(() => ({ vad: new EnergyVad(), name: "energy VAD" })),
    options.speakerId === false
      ? Promise.resolve<SpeakerIdBuild>({
          embedder: null,
          labeler: null,
          capability: inactiveCapability("disabled for this session"),
        })
      : buildSpeakerId().catch((err: unknown) =>
          speakerIdOff(err instanceof Error ? err.message : String(err)),
        ),
  ]);
  const llm = buildLlm(options.providerOrder);
  // Start the on-device model (Gemini Nano's AICore download) NOW, while the
  // user is still on the pre-flight — not on the first suggestion mid-session.
  // Fire-and-forget; each provider memoizes the preparation. See ProviderChain.prewarm.
  llm.prewarm();
  return {
    vad: vad instanceof SileroVad ? "silero" : "energy",
    speakerId: speaker.capability,
    llm: llm.providerNames,
  };
}
