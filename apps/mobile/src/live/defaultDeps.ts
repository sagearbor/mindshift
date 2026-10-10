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
import { EcapaEmbedder, LIVE_LABELER_OPTIONS, SpeakerLabeler, type Embedder } from "./speakerId";
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
import { hapticFor, hapticRuns, phonePattern } from "./nudgeVocabulary";

/** The callbacks the hook supplies; everything else is wired here. */
export type FastLoopHandlers = Pick<
  FastLoopDeps,
  "speak" | "send" | "onTurn" | "onNudge" | "onPositiveNudge" | "onSttError" | "onDegrade"
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
  /** Seams (tests): the rung builders and the pre-flight's warm cache. */
  builders?: { buildVad: typeof buildVad; buildSpeakerId: typeof buildSpeakerId };
  warm?: WarmBuildCache;
  now?: () => number;
}

/** How long each start step took (ms), and whether the pre-flight's
 *  already-built VAD / speaker-ID were reused instead of rebuilt. */
export interface FastLoopBuildTimings {
  reusedWarm: boolean;
  vadMs: number;
  speakerIdMs: number;
  llmMs: number;
  totalMs: number;
}

interface WarmBuild {
  vad: { vad: FrameVad; name: string };
  speaker: SpeakerIdBuild;
  builtAt: number;
}

/**
 * The pre-flight probe builds exactly what a session start builds (Silero,
 * ECAPA, the voiceprints) — on the Pixel 10 the start then rebuilt it all
 * again, part of an ≈8.5 s cold start (2026-10-07). The probe now parks its
 * build here and the next start TAKES it (single use: a VAD / labeler is
 * never shared by two loops; the loop resets both at start). A parked build
 * older than WARM_MAX_AGE_MS (voiceprints may have changed) or whose
 * speaker-ID came up inactive (e.g. a network blip) is not reused.
 */
export const WARM_MAX_AGE_MS = 10 * 60 * 1000;

export class WarmBuildCache {
  private parked: WarmBuild | null = null;
  put(build: WarmBuild) {
    this.parked = build;
  }
  take(now: number): WarmBuild | null {
    const b = this.parked;
    this.parked = null;
    if (!b) return null;
    if (now - b.builtAt > WARM_MAX_AGE_MS) return null;
    if (!b.speaker.capability.active) return null;
    return b;
  }
  clear() {
    this.parked = null;
  }
}

/** The app-wide cache the default probe fills and the default start drains. */
export const defaultWarmBuilds = new WarmBuildCache();

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
  /** Per-step start timing (native default build; absent elsewhere). */
  timings?: FastLoopBuildTimings;
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

/**
 * The production haptic sink.
 *
 * Android gets the nudge VOCABULARY's own waveform for the code
 * (nudgeVocabulary.ts): React Native's `Vibration.vibrate(pattern)` takes
 * exactly the [wait, buzz, wait, buzz, …] array the contract stores, so the
 * rhythm the watch plays and the rhythm the phone plays are the same array of
 * numbers. RN cannot vary amplitude, which is precisely why every level
 * difference in that contract is a rhythm difference.
 *
 * Everything else — iOS, web, a code with no cue, an unknown code — falls
 * back to expo-haptics' single light/medium/heavy impact, which is what
 * shipped before the vocabulary. A missing haptic engine is silent; the
 * on-screen flash still shows.
 */
export const expoHaptics: HapticSink = {
  async nudge(level, code) {
    const wave = code ? hapticFor(code, level) : null;
    // A cue that is ONE tap is rendered as the OEM's tuned impact, never as a
    // raw buzz. Measured on the owner's Pixel (2026-09-07): a single 75 ms
    // pattern was reported as "does nothing", and so was 120 ms. A modern
    // phone's actuator produces almost nothing from a short unshaped
    // `Vibration.vibrate` — which is the SAME finding the watch made in
    // v0.2.4, when its 40 ms raw taps proved imperceptible and were replaced
    // by system-tuned effects. Multi-tap cues stay raw patterns, because
    // rhythm is what they are for and expo-haptics cannot express one.
    // Runs, not raw segments: a swell is ONE buzz, and sending its internal
    // shape to React Native would play it as three separate taps.
    const taps = wave ? hapticRuns(wave).length : 0;
    if (wave && taps > 1) {
      const RN = tryRequire(
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        () => require("react-native") as typeof import("react-native"),
      );
      // Only Android honours a pattern; iOS's Vibration ignores the timings.
      if (RN?.Platform?.OS === "android" && RN.Vibration) {
        try {
          RN.Vibration.vibrate(phonePattern(wave));
          return;
        } catch {
          // Fall through to the impact below rather than losing the nudge.
        }
      }
    }
    try {
      const Haptics = tryRequire(
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        () => require("expo-haptics") as typeof import("expo-haptics"),
      );
      if (!Haptics) {
        // No tuned effects available: a raw pattern is better than silence,
        // even for a single tap.
        if (wave) {
          const RN = tryRequire(
            // eslint-disable-next-line @typescript-eslint/no-require-imports
            () => require("react-native") as typeof import("react-native"),
          );
          if (RN?.Platform?.OS === "android" && RN.Vibration) RN.Vibration.vibrate(phonePattern(wave));
        }
        return;
      }
      // A single-tap CUE is always the strongest single tap the OEM offers:
      // it is already the mildest thing in the vocabulary by virtue of being
      // one tap, and making it quiet as well is how it became unnoticeable.
      const style =
        taps === 1
          ? Haptics.ImpactFeedbackStyle.Heavy
          : level >= 3
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
    labeler: new SpeakerLabeler(kept, LIVE_LABELER_OPTIONS),
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
  const now = options.now ?? Date.now;
  const builders = options.builders ?? { buildVad, buildSpeakerId };
  const t0 = now();
  // Reuse what the pre-flight already built (single use), else build now.
  const warm = options.speakerId === false ? null : (options.warm ?? defaultWarmBuilds).take(t0);
  let vadMs = 0;
  let speakerIdMs = 0;
  const timed = async <T,>(work: () => Promise<T>, done: (ms: number) => void): Promise<T> => {
    const s = now();
    try {
      return await work();
    } finally {
      done(now() - s);
    }
  };
  const [{ vad, name: vadName }, speaker] = warm
    ? [warm.vad, warm.speaker]
    : await Promise.all([
        timed(() => builders.buildVad(), (ms) => (vadMs = ms)),
        options.speakerId === false
          ? Promise.resolve<SpeakerIdBuild>({
              embedder: null,
              labeler: null,
              capability: inactiveCapability("disabled for this session"),
            })
          : timed(() => builders.buildSpeakerId(), (ms) => (speakerIdMs = ms)),
      ]);
  const tl = now();
  const llm = buildLlm(options.providerOrder);
  const llmMs = now() - tl;
  const timings: FastLoopBuildTimings = {
    reusedWarm: warm !== null,
    vadMs,
    speakerIdMs,
    llmMs,
    totalMs: now() - t0,
  };
  console.log(
    `[live] loop build ${timings.totalMs} ms (${warm ? "reused pre-flight build" : `vad ${vadMs} ms, speaker-ID ${speakerIdMs} ms`}, llm ${llmMs} ms)`,
  );
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
    timings,
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
  const builders = options.builders ?? { buildVad, buildSpeakerId };
  const [vadBuild, speaker] = await Promise.all([
    builders.buildVad().catch(() => ({ vad: new EnergyVad() as FrameVad, name: "energy VAD" })),
    options.speakerId === false
      ? Promise.resolve<SpeakerIdBuild>({
          embedder: null,
          labeler: null,
          capability: inactiveCapability("disabled for this session"),
        })
      : builders.buildSpeakerId().catch((err: unknown) =>
          speakerIdOff(err instanceof Error ? err.message : String(err)),
        ),
  ]);
  const { vad } = vadBuild;
  // Park the build for the next session start (see WarmBuildCache).
  if (options.speakerId !== false) {
    (options.warm ?? defaultWarmBuilds).put({ vad: vadBuild, speaker, builtAt: (options.now ?? Date.now)() });
  }
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
