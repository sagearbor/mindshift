/**
 * Production wiring of the fast loop for the WEB build — the browser
 * analogue of defaultDeps.ts, reusing fastLoop.ts unchanged:
 *
 *   VAD        Silero over onnxruntime-web (wasm)  -> energy VAD if it can't load
 *   speaker-ID ECAPA downloaded once into the Cache API + enrolled voiceprints
 *              -> off (with the reason) if the runtime/model/people aren't there
 *   STT        the browser's Web Speech API (iOS Safari: Apple's recognizer)
 *   LLM        `cloud` only — no on-device model in a browser; the server's
 *              streaming suggestion (p50 ≈ 1.1 s in production) answers
 *   TTS        the hook's expo-speech (window.speechSynthesis), earpiece/speaker only
 *   haptics    none (iOS Safari has no vibration API)
 *
 * Everything degrades to a reason in `FastLoopBuild.status`, never a throw.
 */
import { ecapaModelUrl, fetchVoiceprints, authHeaders } from "../api/liveSessions";
import { FastLoop } from "./fastLoop";
import type { FastLoopBuild, FastLoopCapabilities, FastLoopHandlers, DefaultFastLoopOptions } from "./defaultDeps";
import { EnergyVad, SileroVad, type FrameVad } from "./vad";
import { EcapaEmbedder, SpeakerLabeler, type Embedder } from "./speakerId";
import {
  activeCapability,
  describeSpeakerId,
  inactiveCapability,
  peopleForModel,
  type SpeakerIdCapability,
} from "./speakerIdSetup";
import { cloudProvider, ProviderChain } from "./localLlm";
import { loadSileroSessionWeb, loadWebOrt, webOrtSessionFactory, type OrtWebRuntime } from "./ortWeb";
import { ECAPA_FILENAME, resolveEcapaModel, type FetchLike } from "./modelDownload";
import { describeProgress, webModelStore, type WebModelStore } from "./modelStoreWeb";
import { WebSpeechRecognizer, webSttAvailable } from "./sttWeb";
import type { SpeechRecognizer } from "./stt";

export interface WebFastLoopOptions extends DefaultFastLoopOptions {
  /** Seams for tests. */
  loadOrt?: () => Promise<{ ort: OrtWebRuntime | null; reason: string | null }>;
  sileroUrl?: string | null;
  store?: WebModelStore;
  fetch?: FetchLike;
}

/**
 * Create and START a browser recognizer synchronously — call this from the
 * Start button's handler (iOS Safari gates the speech permission on a user
 * gesture). Null when the browser has none. The returned recognizer is
 * handed to `createWebFastLoop` through `handlers.recognizer`; the loop's
 * later `start()` is a no-op on it.
 */
export function primeWebRecognizer(lang?: string): WebSpeechRecognizer | null {
  if (!webSttAvailable()) return null;
  const rec = new WebSpeechRecognizer({ lang });
  // Errors raised before the loop subscribes are held and re-delivered on
  // the first onError() — nothing is lost, nothing rejects here.
  void rec.start().catch(() => {});
  return rec;
}

async function buildVad(
  ort: OrtWebRuntime | null,
  sileroUrl: string | null | undefined,
): Promise<{ vad: FrameVad; name: string }> {
  if (ort) {
    const session = await loadSileroSessionWeb(
      webOrtSessionFactory(ort),
      sileroUrl === undefined ? undefined : sileroUrl,
    );
    if (session) return { vad: new SileroVad(session), name: "Silero VAD (wasm)" };
  }
  return { vad: new EnergyVad(), name: "energy VAD" };
}

interface SpeakerIdBuild {
  embedder: Embedder | null;
  labeler: SpeakerLabeler | null;
  capability: SpeakerIdCapability;
}

function speakerIdOff(reason: string): SpeakerIdBuild {
  console.log(`[live/web] speaker-ID off: ${reason}`);
  return { embedder: null, labeler: null, capability: inactiveCapability(reason) };
}

async function buildSpeakerId(
  ort: OrtWebRuntime | null,
  ortReason: string | null,
  options: WebFastLoopOptions,
  onStatus: ((line: string) => void) | undefined,
): Promise<SpeakerIdBuild> {
  if (!ort) return speakerIdOff(ortReason ?? "ONNX Runtime (wasm) unavailable");
  const fetchImpl = options.fetch ?? (fetch as unknown as FetchLike);
  const store =
    options.store ??
    webModelStore({
      fetch: fetchImpl as never,
      onProgress: (p) => onStatus?.(describeProgress(p)),
    });
  const headers = await authHeaders(false);
  const [voiceprints, model] = await Promise.all([
    fetchVoiceprints(),
    resolveEcapaModel({ url: ecapaModelUrl(), headers, fetch: fetchImpl, store }).catch((err) => ({
      status: "unavailable" as const,
      code: "network" as const,
      reason: `model store failed (${err instanceof Error ? err.message : String(err)})`,
    })),
  ]);
  if (model.status !== "ready") return speakerIdOff(model.reason);
  const bytes = await store.readBytes(ECAPA_FILENAME).catch(() => null);
  if (!bytes) return speakerIdOff("model vanished from the browser cache");
  let session;
  try {
    onStatus?.("Loading voice model …");
    session = await webOrtSessionFactory(ort)(bytes);
  } catch (err) {
    return speakerIdOff(`ONNX Runtime rejected the model (${err instanceof Error ? err.message : String(err)})`);
  }
  const { kept, dropped } = peopleForModel(voiceprints.people, model.etag);
  if (dropped.length > 0) {
    console.log(`[live/web] speaker-ID: skipped ${dropped.length} voiceprint(s) from another model revision`);
  }
  return {
    embedder: new EcapaEmbedder(session),
    labeler: new SpeakerLabeler(kept),
    capability: activeCapability(model, kept, dropped.length, voiceprints.error),
  };
}

/** Build a production FastLoop for the browser. Never throws for an optional piece. */
export async function createWebFastLoop(
  handlers: FastLoopHandlers,
  options: WebFastLoopOptions = {},
): Promise<FastLoopBuild> {
  const onStatus = handlers.onStatus;
  onStatus?.("Loading on-device models …");
  const loaded = await (options.loadOrt ?? loadWebOrt)();
  const ort = loaded.ort;
  if (!ort) console.warn(`[live/web] ONNX Runtime unavailable: ${loaded.reason}`);
  const [{ vad, name: vadName }, speaker] = await Promise.all([
    buildVad(ort, options.sileroUrl),
    options.speakerId === false
      ? Promise.resolve<SpeakerIdBuild>({
          embedder: null,
          labeler: null,
          capability: inactiveCapability("disabled for this session"),
        })
      : buildSpeakerId(ort, loaded.reason, options, onStatus),
  ]);
  const llm = new ProviderChain([cloudProvider()], options.providerOrder);
  const recognizer: SpeechRecognizer | null =
    handlers.recognizer ?? (webSttAvailable() ? new WebSpeechRecognizer({ lang: options.lang }) : null);
  const { recognizer: _primed, onStatus: _status, ...loopHandlers } = handlers;
  void _primed;
  void _status;
  const loop = new FastLoop({
    ...loopHandlers,
    vad,
    embedder: speaker.embedder,
    labeler: speaker.labeler,
    recognizer,
    llm,
    haptics: null,
  });
  const capabilities: FastLoopCapabilities = {
    vad: vad instanceof SileroVad ? "silero" : "energy",
    speakerId: speaker.capability,
    llm: llm.providerNames,
  };
  const stt = recognizer ? "browser speech recognition" : "no browser speech recognition";
  return {
    loop,
    status: `${vadName} · ${describeSpeakerId(speaker.capability)} · LLM ${llm.providerNames.join(" → ")} · ${stt}`,
    capabilities,
  };
}

/**
 * Pre-flight for the browser: what `createWebFastLoop` WOULD load right
 * now, without a loop (the web analogue of `probeFastLoopCapabilities`).
 * Runs the same builders (so the ECAPA model lands in the Cache API and
 * the real start is fast); every failure is a reason line, never a throw.
 */
export async function probeWebFastLoopCapabilities(
  options: WebFastLoopOptions = {},
): Promise<FastLoopCapabilities> {
  const loaded = await (options.loadOrt ?? loadWebOrt)().catch((err: unknown) => ({
    ort: null,
    reason: err instanceof Error ? err.message : String(err),
  }));
  const ort = loaded.ort;
  const [{ vad }, speaker] = await Promise.all([
    buildVad(ort, options.sileroUrl).catch(() => ({ vad: new EnergyVad() as FrameVad, name: "energy VAD" })),
    options.speakerId === false
      ? Promise.resolve<SpeakerIdBuild>({
          embedder: null,
          labeler: null,
          capability: inactiveCapability("disabled for this session"),
        })
      : buildSpeakerId(ort, loaded.reason, options, undefined).catch((err: unknown) =>
          speakerIdOff(err instanceof Error ? err.message : String(err)),
        ),
  ]);
  const llm = new ProviderChain([cloudProvider()], options.providerOrder);
  return {
    vad: vad instanceof SileroVad ? "silero" : "energy",
    speakerId: speaker.capability,
    llm: llm.providerNames,
  };
}
