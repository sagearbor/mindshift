/**
 * Offline replay of a real recording through the REAL on-device fast loop.
 *
 *   WAV (16 kHz mono) ──100 ms frames──► FastLoop ──► turns / nudges / speech
 *        + meta (script)                    │
 *                                           ├─ Silero VAD + segmenter   (real, onnxruntime-node)
 *                                           ├─ ECAPA speaker-ID          (real; voiceprints enrolled from the meta)
 *                                           ├─ STT                       (scripted: meta text, configurable latency)
 *                                           ├─ local LLM chain           (scripted: canned + tone from emotion_coarse)
 *                                           └─ expo-speech               (recorded, never spoken)
 *
 * Time is a `VirtualClock`: each frame is delivered at its end time; every
 * stage latency (STT finalization, LLM round trip, the loop's own polls and
 * holds) is a timer on that clock, and native inference costs zero virtual
 * time (its wall time is reported separately). The result is deterministic
 * for a given WAV + options, which is what lets the Jest suite pin numbers.
 *
 * `replayScene` runs one (scene, mode); `formatReport` renders the table the
 * CLI prints. Node-only (fs, onnxruntime-node) — nothing in the app graph
 * imports this directory.
 */
import * as fs from "fs";
import * as path from "path";
import { FastLoop, type LocalTurn, type TurnLatency } from "../fastLoop";
import { SileroVad, EnergyVad, type FrameVad } from "../vad";
import { EcapaEmbedder, SpeakerLabeler, type Embedder } from "../speakerId";
import { cloudProvider, ProviderChain, type LiveMode } from "../localLlm";
import { phoneNudgePolicy, type NudgeEvent } from "../nudgePolicy";
import type { OnnxSessionFactory } from "../ort";
import type { TurnLocalEvent } from "../types";
import { int16ToFloat32, readWav16kMono } from "./wav";
import { parseSceneMeta, type ReplayScript } from "./meta";
import { InflightTracker, VirtualClock } from "./virtualClock";
import {
  DEFAULT_STT_OPTIONS,
  recordingPolicy,
  ScriptedRecognizer,
  scriptedProvider,
  SpokenLog,
  TrackedEmbedder,
  TrackedVad,
  type PolicyCall,
  type ScriptedRecognizerOptions,
  type SpokenLine,
} from "./fakes";
import { enrollFromSource, findCrossSceneSource, type EnrollmentRecord, type EnrollmentSource } from "./enroll";
import {
  scoreAttribution,
  scoreBoundaries,
  scoreLatency,
  scoreNudges,
  scoreSpeaking,
  type AttributionScore,
  type BoundaryScore,
  type LatencyScore,
  type NudgeScore,
  type SpeakScore,
} from "./score";

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

export const MOBILE_ROOT = path.resolve(__dirname, "../../..");
export const REPO_ROOT = path.resolve(MOBILE_ROOT, "../..");
export const AUDIO_FIXTURES_DIR = path.join(REPO_ROOT, "server/tests/fixtures/audio");
export const SILERO_PATH = path.join(MOBILE_ROOT, "assets/models/silero_vad.onnx");
/** Where the Jest replays drop their turn_local dumps (gitignored); the
 *  pytest contract check reads it when present. */
export const REPLAY_OUT_DIR = path.join(MOBILE_ROOT, ".replay-out");

/** Where the ECAPA export may live on a dev machine (never in git — 80 MB).
 *  Same candidates as __tests__/liveSpeakerId.test.ts. */
export function findEcapaModel(): string | null {
  const ref = path.join(MOBILE_ROOT, "__tests__/fixtures/ecapa_reference.json");
  const candidates: string[] = [];
  if (process.env.MINDSHIFT_ECAPA_ONNX_PATH) candidates.push(process.env.MINDSHIFT_ECAPA_ONNX_PATH);
  if (fs.existsSync(ref)) {
    try {
      const { revision } = JSON.parse(fs.readFileSync(ref, "utf8")) as { revision?: string };
      if (revision) candidates.push(path.join(REPO_ROOT, `server/.ecapa_cache/ecapa_${revision}.onnx`));
    } catch {
      // A malformed reference just means no revision-pinned candidate.
    }
  }
  candidates.push(
    path.join(MOBILE_ROOT, "assets/models/ecapa.onnx"),
    path.join(REPO_ROOT, "server/.ecapa_cache/ecapa.onnx"),
    path.join(REPO_ROOT, "tmp/ecapa.onnx"),
  );
  // A git worktree (.claude/worktrees/<agent>/) has no cache of its own: the
  // main checkout's server/.ecapa_cache is a few directories up.
  const revisioned = candidates.filter((c) => c.includes("server/.ecapa_cache/")).map((c) => path.relative(REPO_ROOT, c));
  let dir = path.dirname(REPO_ROOT);
  for (let depth = 0; depth < 4 && dir !== path.dirname(dir); depth++) {
    for (const rel of revisioned) candidates.push(path.join(dir, rel));
    dir = path.dirname(dir);
  }
  return candidates.find((p) => fs.existsSync(p)) ?? null;
}

export interface SceneInput {
  /** Script name (fixture stem or a caller-chosen label). */
  name: string;
  wavPath: string;
  script: ReplayScript;
  /** int16 PCM, 16 kHz mono. */
  pcm: Int16Array;
  pcmF32: Float32Array;
}

/**
 * Resolve `<scene>` to a fixture (`server/tests/fixtures/audio/
 * test_recording_<scene>.wav` + `_meta.json`) or take a WAV path with its
 * `<stem>_meta.json` beside it (or an explicit `metaPath`).
 */
export function loadScene(nameOrWav: string, opts: { metaPath?: string; selfSpeaker?: string | null } = {}): SceneInput {
  let wavPath: string;
  let name: string;
  if (nameOrWav.endsWith(".wav")) {
    wavPath = path.resolve(nameOrWav);
    name = path.basename(wavPath, ".wav").replace(/^test_recording_/, "");
  } else {
    name = nameOrWav.replace(/^test_recording_/, "");
    wavPath = path.join(AUDIO_FIXTURES_DIR, `test_recording_${name}.wav`);
  }
  if (!fs.existsSync(wavPath)) throw new Error(`scene: no WAV at ${wavPath}`);
  const metaPath = opts.metaPath ?? wavPath.replace(/\.wav$/, "_meta.json");
  if (!fs.existsSync(metaPath)) throw new Error(`scene: no meta at ${metaPath} (write one: see replay/meta.ts)`);
  const raw = JSON.parse(fs.readFileSync(metaPath, "utf8")) as unknown;
  const script = parseSceneMeta(raw, { name, selfSpeaker: opts.selfSpeaker });
  const pcm = readWav16kMono(wavPath);
  return { name, wavPath, script, pcm, pcmF32: int16ToFloat32(pcm) };
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export type EnrollMode = "self" | "all" | "none";

export interface ReplayOptions {
  mode: LiveMode;
  /** Audio delivery granularity (expo-audio hands ~100 ms buffers). */
  frameMs: number;
  stt: ScriptedRecognizerOptions;
  /** The "os" provider (Gemini Nano / Apple FM stand-in). */
  osLatencyMs: number;
  osRefuseEveryK: number;
  /** The "bundled" LiteRT-LM stand-in; unavailable => refusals reach the cloud. */
  bundledAvailable: boolean;
  bundledLatencyMs: number;
  /** Virtual cost charged per ECAPA embed (phone estimate). */
  speakerCostMs: number;
  enroll: EnrollMode;
  /** Other recordings the same voices appear in (cross-scene enrollment). */
  enrollFrom: SceneInput[];
  /** Explicit voice pairings for real recordings (no `voices` in the meta). */
  voicePairs: { scene: string; speaker: string; sameAs: { scene: string; speaker: string } }[];
  /** Same-scene enrollment budget (seconds of that speaker, from the top). */
  enrollMaxSeconds: number;
  empathy: number;
  sttGraceMs: number;
  speakHoldMaxMs: number;
  /** FastLoop's speakQuietMs (min quiet before voicing a suggestion). */
  speakQuietMs: number;
  /** Silence appended so the last turn closes before stop(). */
  tailSilenceSec: number;
  /** Absent ECAPA => speaker-ID off (turns are unknown clusters / "Unknown"). */
  ecapaPath: string | null;
  sileroPath: string;
  /** Energy VAD instead of Silero (debugging aid; not what the phone runs). */
  energyVad: boolean;
  ortFactory: OnnxSessionFactory | null;
  /** Reuse loaded sessions across runs (the CLI/Jest load ECAPA once). */
  models?: LoadedModels;
}

export const DEFAULT_REPLAY_OPTIONS: Omit<ReplayOptions, "mode"> = {
  frameMs: 100,
  stt: DEFAULT_STT_OPTIONS,
  osLatencyMs: 400,
  osRefuseEveryK: 0,
  bundledAvailable: true,
  bundledLatencyMs: 700,
  speakerCostMs: 40,
  enroll: "self",
  enrollFrom: [],
  voicePairs: [],
  enrollMaxSeconds: 10,
  empathy: 60,
  sttGraceMs: 700,
  speakHoldMaxMs: 3000,
  speakQuietMs: 0,
  tailSilenceSec: 2,
  ecapaPath: findEcapaModel(),
  sileroPath: SILERO_PATH,
  energyVad: false,
  ortFactory: null,
};

export interface LoadedModels {
  vad: () => Promise<FrameVad>;
  embedder: Embedder | null;
  ecapaPath: string | null;
}

/** Load Silero (per run — it is stateful) and ECAPA (once) through the node
 *  ORT seam. Returns `embedder: null` when no export is on this machine. */
export async function loadModels(opts: Pick<ReplayOptions, "ortFactory" | "sileroPath" | "ecapaPath" | "energyVad">): Promise<LoadedModels> {
  const factory = opts.ortFactory ?? (await nodeFactory());
  const embedder = opts.ecapaPath && !opts.energyVad ? new EcapaEmbedder(await factory(opts.ecapaPath)) : null;
  const ecapaPath = embedder ? opts.ecapaPath : null;
  return {
    vad: async () => (opts.energyVad ? new EnergyVad() : new SileroVad(await factory(opts.sileroPath))),
    embedder,
    ecapaPath,
  };
}

async function nodeFactory(): Promise<OnnxSessionFactory> {
  // Lazy: keeps onnxruntime-node (and its realm fix) out of the module graph
  // until a replay actually runs.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { nodeOrtSessionFactory } = require("../testing/ortNode") as typeof import("../testing/ortNode");
  return nodeOrtSessionFactory();
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

export interface ReplayResult {
  scene: string;
  mode: LiveMode;
  options: Omit<ReplayOptions, "enrollFrom" | "ortFactory" | "models"> & { enrollFrom: string[] };
  script: ReplayScript;
  durationSec: number;
  capability: { vad: "silero" | "energy"; speakerId: boolean; enrolled: EnrollmentRecord[] };
  turns: LocalTurn[];
  sent: TurnLocalEvent[];
  spoken: SpokenLine[];
  nudges: NudgeEvent[];
  haptics: number[];
  policyLog: PolicyCall[];
  latencyLog: TurnLatency[];
  stt: { emitted: number; finals: number };
  providerCalls: { os: number; osRefused: number; bundled: number };
  attribution: AttributionScore;
  boundaries: BoundaryScore;
  nudgeScore: NudgeScore;
  speaking: SpeakScore;
  latency: LatencyScore;
  wall: { totalMs: number; byStage: Record<string, number>; calls: Record<string, number> };
  virtualEndMs: number;
}

// ---------------------------------------------------------------------------
// The replay
// ---------------------------------------------------------------------------

export async function replayScene(scene: SceneInput, partial: Partial<ReplayOptions> & { mode: LiveMode }): Promise<ReplayResult> {
  const opts: ReplayOptions = { ...DEFAULT_REPLAY_OPTIONS, ...partial, stt: { ...DEFAULT_STT_OPTIONS, ...(partial.stt ?? {}) } };
  const wall0 = performance.now();
  const models = opts.models ?? (await loadModels(opts));
  const clock = new VirtualClock();
  const tracker = new InflightTracker();
  const script = scene.script;

  // --- speaker-ID: enroll from the meta -----------------------------------
  const enrolled: EnrollmentRecord[] = [];
  if (models.embedder && opts.enroll !== "none") {
    const who = opts.enroll === "all" ? script.speakers : script.selfSpeaker ? [script.selfSpeaker] : [];
    const sources: EnrollmentSource[] = opts.enrollFrom.map((s) => ({ script: s.script, pcm: s.pcmF32 }));
    for (const speaker of who) {
      const cross = findCrossSceneSource(script, speaker, sources, opts.voicePairs);
      const rec = cross
        ? await enrollFromSource(models.embedder, cross.source, cross.speaker, {
            personId: `p-${speaker}`,
            displayName: speaker,
            isSelf: speaker === script.selfSpeaker,
            crossScene: true,
          })
        : await enrollFromSource(models.embedder, { script, pcm: scene.pcmF32 }, speaker, {
            personId: `p-${speaker}`,
            displayName: speaker,
            isSelf: speaker === script.selfSpeaker,
            maxSeconds: opts.enrollMaxSeconds,
            crossScene: false,
          });
      enrolled.push(rec);
    }
  }
  const speakerId = models.embedder !== null;
  const embedder = models.embedder ? new TrackedEmbedder(models.embedder, tracker, clock, opts.speakerCostMs) : null;
  const labeler = speakerId ? new SpeakerLabeler(enrolled) : null;

  // --- the loop -------------------------------------------------------------
  const vad = new TrackedVad(await models.vad(), tracker);
  const audioNow = (): number => loop.audioClock;
  const recognizer: ScriptedRecognizer | null = script.hasText ? new ScriptedRecognizer(script, opts.stt, audioNow) : null;
  const os = scriptedProvider(script, clock, { name: "os", latencyMs: opts.osLatencyMs, refuseEveryK: opts.osRefuseEveryK });
  const bundled = scriptedProvider(script, clock, {
    name: "bundled",
    latencyMs: opts.bundledLatencyMs,
    available: opts.bundledAvailable,
  });
  const llm = new ProviderChain([os, bundled, cloudProvider()], ["os", "bundled", "cloud"], () => clock.now());
  const spokenLog = new SpokenLog(clock, () => vad.lastVerdict);
  const policy = recordingPolicy(phoneNudgePolicy());
  const sent: TurnLocalEvent[] = [];
  const turns: LocalTurn[] = [];
  const nudges: NudgeEvent[] = [];
  const haptics: number[] = [];

  const loop: FastLoop = new FastLoop({
    vad,
    embedder,
    labeler,
    recognizer,
    llm,
    speak: spokenLog.speak,
    send: (e) => sent.push(e),
    onTurn: (t) => turns.push(t),
    onNudge: (n) => nudges.push(n),
    haptics: { nudge: async (level) => void haptics.push(level) },
    policy,
    now: () => clock.now(),
    sleep: (ms) => clock.sleep(ms),
    sttGraceMs: opts.sttGraceMs,
    speakHoldMaxMs: opts.speakHoldMaxMs,
    speakQuietMs: opts.speakQuietMs,
    pollMs: 50,
  });

  await loop.start({ sessionId: `replay-${scene.name}-${opts.mode}`, mode: opts.mode, empathy: opts.empathy });

  // --- drive: frames delivered at their end time ----------------------------
  const frameSamples = Math.round((opts.frameMs / 1000) * 16000);
  const total = scene.pcm.length + Math.round(opts.tailSilenceSec * 16000);
  const quiesce = () => tracker.quiesce();
  for (let off = 0; off < total; off += frameSamples) {
    const end = Math.min(off + frameSamples, total);
    const frame = new Int16Array(end - off);
    const audioEnd = Math.min(end, scene.pcm.length);
    if (audioEnd > off) frame.set(scene.pcm.subarray(off, audioEnd), 0);
    // Whole virtual milliseconds (1600 samples = 100 ms exactly) so stage
    // latencies subtract cleanly instead of carrying float dust.
    await clock.advanceTo(Math.round((end * 1000) / 16000), quiesce);
    loop.pushSamples(frame);
    await quiesce();
    recognizer?.tick(loop.audioClock);
    await quiesce();
  }

  // --- stop: keep time moving until every queued stage has finished --------
  let stopped = false;
  const summaryP = loop.stop().then((s) => {
    stopped = true;
    return s;
  });
  const deadline = clock.now() + 30_000;
  while (!stopped) {
    await quiesce();
    if (stopped) break;
    if (clock.now() >= deadline) throw new Error("replay: loop did not settle within 30 s of virtual time");
    await clock.advanceBy(opts.frameMs, quiesce);
  }
  const summary = await summaryP;

  const attribution = scoreAttribution(script, summary.turns, enrolled);
  const boundaries = scoreBoundaries(script, summary.turns);
  const nudgeScore = scoreNudges(script, summary.turns, policy.log);
  const speaking = scoreSpeaking(script, summary.turns, spokenLog.lines, vad);
  const latency = scoreLatency(script, summary.turns);
  const { enrollFrom, ortFactory: _f, models: _m, ...rest } = opts;
  void _f;
  void _m;
  return {
    scene: scene.name,
    mode: opts.mode,
    options: { ...rest, enrollFrom: enrollFrom.map((s) => s.name), ecapaPath: models.ecapaPath },
    script,
    durationSec: scene.pcm.length / 16000,
    capability: { vad: opts.energyVad ? "energy" : "silero", speakerId, enrolled },
    turns: summary.turns,
    sent,
    spoken: spokenLog.lines,
    nudges,
    haptics,
    policyLog: policy.log,
    latencyLog: summary.latencyLog,
    stt: { emitted: recognizer?.emitted.length ?? 0, finals: recognizer?.emitted.filter((e) => e.isFinal).length ?? 0 },
    providerCalls: {
      os: os.calls.length,
      osRefused: os.calls.filter((c) => c.outcome === "refused").length,
      bundled: bundled.calls.length,
    },
    attribution,
    boundaries,
    nudgeScore,
    speaking,
    latency,
    wall: { totalMs: performance.now() - wall0, byStage: { ...tracker.wallMs }, calls: { ...tracker.calls } },
    virtualEndMs: clock.now(),
  };
}

// ---------------------------------------------------------------------------
// turn_local dump (cross-language contract check: server/tests/test_replay_turn_local_contract.py)
// ---------------------------------------------------------------------------

export interface TurnLocalDump {
  scene: string;
  mode: LiveMode;
  generated_by: string;
  options: { stt_final_latency_ms: number; os_latency_ms: number; enroll: EnrollMode; speaker_id: boolean };
  events: TurnLocalEvent[];
}

export function turnLocalDump(r: ReplayResult): TurnLocalDump {
  return {
    scene: r.scene,
    mode: r.mode,
    generated_by: "apps/mobile/src/live/replay/sceneReplay.ts",
    options: {
      stt_final_latency_ms: r.options.stt.finalLatencyMs,
      os_latency_ms: r.options.osLatencyMs,
      enroll: r.options.enroll,
      speaker_id: r.capability.speakerId,
    },
    events: r.sent,
  };
}

export function writeTurnLocalDump(r: ReplayResult, dir: string): string {
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, `turn_local_${r.scene}_${r.mode}.json`);
  fs.writeFileSync(file, JSON.stringify(turnLocalDump(r), null, 2) + "\n");
  return file;
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

const ms = (x: number) => (Number.isFinite(x) ? `${Math.round(x)} ms` : "-");
const pct = (a: number, b: number) => (b ? `${a}/${b} = ${((100 * a) / b).toFixed(1)}%` : "n/a");

function pad(s: string, n: number): string {
  return s.length >= n ? s : s + " ".repeat(n - s.length);
}

/** One-line summary — what the Jest tests print and the PR table quotes. */
export function summaryLine(r: ReplayResult): string {
  const a = r.attribution;
  const n = r.nudgeScore;
  return (
    `${r.scene} [${r.mode}] attribution ${pct(a.correct, a.total)} (self ${a.selfCorrect}/${a.selfTotal}); ` +
    `speakers ${a.speakersDetected} (${a.unknownClusters} unknown); ` +
    `nudges hit ${n.hits}/${r.script.expectedNudges.length} miss ${n.misses} fp ${n.falsePositives}; ` +
    `spoken ${r.speaking.spoken} (over-speech ${r.speaking.overVadSpeech}, held ${r.speaking.held}); ` +
    `first words median ${ms(r.latency.toSpeakMedianMs)} max ${ms(r.latency.toSpeakMaxMs)}`
  );
}

export function formatReport(r: ReplayResult): string {
  const lines: string[] = [];
  const enr = r.capability.enrolled;
  const enrText = enr.length
    ? enr
        .map((e) => `${e.displayName}${e.isSelf ? " (self)" : ""} <- ${e.crossScene ? "cross-scene " : "same-scene "}${e.fromScene}/${e.fromSpeaker} ${e.turnsUsed.length} turns ${e.seconds}s`)
        .join("; ")
    : r.capability.speakerId
      ? "none (unknown clusters only)"
      : "speaker-ID OFF (no ECAPA export on this machine)";
  lines.push(`== ${r.scene}  mode=${r.mode}  ${r.durationSec.toFixed(1)} s audio, ${r.script.turns.length} scripted turns, self=${r.script.selfSpeaker ?? "-"}`);
  lines.push(`   VAD ${r.capability.vad}; enrolled: ${enrText}`);
  lines.push(
    `   STT final latency ${r.options.stt.finalLatencyMs} ms (${r.options.stt.wordTimings ? "word-timed" : "untimed"}); ` +
      `LLM os ${r.options.osLatencyMs} ms${r.options.osRefuseEveryK ? ` (refuses every ${r.options.osRefuseEveryK}th)` : ""}, ` +
      `bundled ${r.options.bundledAvailable ? `${r.options.bundledLatencyMs} ms` : "unavailable"}; speaker cost ${r.options.speakerCostMs} ms; speak after ${r.options.speakQuietMs} ms quiet, hold max ${r.options.speakHoldMaxMs} ms`,
  );
  const a = r.attribution;
  lines.push(
    `   Attribution: ${pct(a.correct, a.total)}  self ${pct(a.selfCorrect, a.selfTotal)}  enrolled ${pct(a.enrolledCorrect, a.enrolledTotal)}  ` +
      `speakers detected ${a.speakersDetected} (${a.unknownClusters} unknown clusters; true ${r.script.speakers.length})` +
      (Object.keys(a.mapping).length ? `  map ${JSON.stringify(a.mapping)}` : ""),
  );
  const b = r.boundaries;
  lines.push(
    `   Boundaries: loop turns ${r.turns.length}; |dstart| median ${ms(b.medianStartMs)} p90 ${ms(b.p90StartMs)}; |dend| median ${ms(b.medianEndMs)} p90 ${ms(b.p90EndMs)}; ` +
      `split ${b.split} merged ${b.merged} unmatched ${b.unmatched} extra ${b.extra}${r.script.approxBoundaries ? " (meta boundaries approximate)" : ""}`,
  );
  const n = r.nudgeScore;
  lines.push(
    `   Nudges: expected ${r.script.expectedNudges.length} -> hits ${n.hits}${n.hitsSilent ? ` (${n.hitsSilent} silent: level already held)` : ""}, misses ${n.misses}, false positives ${n.falsePositives}; ` +
      `haptics fired ${n.hapticsFired.map((h) => `L${h.level}@${h.t.toFixed(1)}s(t${h.scriptTurn ?? "?"})`).join(" ") || "none"}`,
  );
  const s = r.speaking;
  lines.push(
    `   Speech out: spoken ${s.spoken}, over VAD speech ${s.overVadSpeech} (timeline ${s.overVadTimeline}), inside a scripted turn ${s.overScriptedSpeech}, held ${s.held}, dropped ${s.dropped}` +
      (r.mode === "therapist" ? " (therapist: must be 0 spoken)" : ""),
  );
  const l = r.latency;
  lines.push(
    `   Latency (virtual): segment-end->first words median ${ms(l.toSpeakMedianMs)} max ${ms(l.toSpeakMaxMs)} over ${l.spokenTurns} spoken; ` +
      `segment close lag median ${ms(l.segmentCloseMedianMs)}; stt wait median ${ms(l.sttWaitMedianMs)}; llm median ${ms(l.llmMedianMs)}; speaker median ${ms(l.speakerMedianMs)}`,
  );
  lines.push(
    `   Providers: ${Object.entries(l.providers)
      .map(([k, v]) => `${k} ${v}`)
      .join(", ")}; os calls ${r.providerCalls.os} (refused ${r.providerCalls.osRefused}), bundled ${r.providerCalls.bundled}; textless turns ${l.textless}, interim-only ${l.interimOnly}`,
  );
  const w = r.wall;
  lines.push(
    `   Wall: ${(w.totalMs / 1000).toFixed(1)} s total; ` +
      Object.entries(w.byStage)
        .map(([k, v]) => `${k} ${(v / 1000).toFixed(2)} s/${w.calls[k]} calls`)
        .join(", "),
  );
  lines.push("");
  lines.push(`   ${pad("#", 3)} ${pad("truth", 10)} ${pad("pred", 12)} ok  ${pad("dstart", 8)} ${pad("dend", 8)} ${pad("lvl", 4)} ${pad("nudge", 6)} ${pad("spk@", 8)} text`);
  r.script.turns.forEach((st, i) => {
    const at = a.perTurn[i];
    const lt = at.loopTurn === null ? null : r.turns[at.loopTurn];
    const nd = n.perTurn[i];
    const spokenAt = lt && lt.spoken ? (r.spoken.find((sp) => sp.text === lt.suggestion && sp.atSec >= lt.endTime)?.atSec ?? null) : null;
    lines.push(
      `   ${pad(String(i), 3)} ${pad(st.speaker, 10)} ${pad(at.predicted ?? "-", 12)} ${at.ok ? "ok " : "XX "} ` +
        `${pad(lt ? String(Math.round((lt.startTime - st.start) * 1000)) : "-", 8)} ${pad(lt ? String(Math.round((lt.endTime - st.end) * 1000)) : "-", 8)} ` +
        `${pad(String(nd.level), 4)} ${pad(nd.expected ? `${nd.verdict}` : nd.verdict === "fp" ? "FP" : "", 6)} ${pad(spokenAt === null ? "" : spokenAt.toFixed(1), 8)} ` +
        `${(lt?.text ?? "").slice(0, 40)}`,
    );
  });
  lines.push("");
  lines.push(`   loop turns (${r.turns.length}):`);
  lines.push(`   ${pad("#", 3)} ${pad("start", 7)} ${pad("end", 7)} ${pad("speaker", 12)} ${pad("self", 5)} ${pad("score", 6)} ${pad("basis", 6)} ${pad("dBFS", 6)} ${pad("via", 8)} ${pad("speak", 8)} text   (?X = the loop's own unknown cluster X; basis abs/ctr = how the voiceprint matched)`);
  r.turns.forEach((lt) => {
    const speakMs = lt.latency.toSpeakMs;
    lines.push(
      `   ${pad(String(lt.index), 3)} ${pad(lt.startTime.toFixed(2), 7)} ${pad(lt.endTime.toFixed(2), 7)} ${pad(lt.personId ? lt.speaker : lt.speaker === "Unknown" ? "Unknown" : `?${lt.speaker}`, 12)} ${pad(lt.isSelf === null ? "?" : lt.isSelf ? "yes" : "no", 5)} ` +
        `${pad(lt.matchScore === null ? "-" : lt.matchScore.toFixed(2), 6)} ${pad(lt.matchBasis === "absolute" ? "abs" : lt.matchBasis === "contrast" ? "ctr" : "-", 6)} ${pad(lt.prosody.rms_dbfs === null ? "-" : lt.prosody.rms_dbfs.toFixed(0), 6)} ${pad(lt.provider, 8)} ` +
        `${pad(speakMs === null ? (lt.suggestion ? "held/x" : "") : `${Math.round(speakMs)}${lt.latency.held ? "h" : ""}`, 8)} ${(lt.text || "").slice(0, 50)}${lt.transcriptFinal ? "" : " (interim)"}`,
    );
  });
  return lines.join("\n");
}
