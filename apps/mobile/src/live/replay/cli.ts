/**
 * Replay a recording through the on-device fast loop and print the report.
 *
 *   cd apps/mobile
 *   npx tsx src/live/replay/cli.ts scene_couple_escalation
 *   npx tsx src/live/replay/cli.ts family_real --self Sage --pair poker6_real:Player6
 *   npx tsx src/live/replay/cli.ts ~/captures/mom_call.wav --self You --mode speaker
 *
 * A phone capture needs a 16 kHz mono int16 WAV (ffmpeg -ac 1 -ar 16000
 * -sample_fmt s16) and a `<stem>_meta.json` beside it — see replay/meta.ts
 * for the minimal shape (`turns[{speaker,text,start_time,end_time}]`,
 * `self_speaker`, optional `expected_nudges`). Bare names resolve to
 * server/tests/fixtures/audio/test_recording_<name>.wav.
 *
 * Options:
 *   --mode earpiece|speaker|therapist|all   (default: all)
 *   --self <label>            self speaker when the meta has none
 *   --meta <path>             meta file (default: <stem>_meta.json)
 *   --enroll self|all|none    which speakers get a voiceprint (default: self)
 *   --enroll-from <scene>,..  other recordings to pool cross-scene prints from
 *                             (default: the rest of the scene pack + family/poker pair)
 *   --pair <scene>:<speaker>  the same voice as --self in another recording
 *   --stt-latency <ms>        scripted STT finalization latency (default 500)
 *   --stt-untimed             iOS-style finals without word timings
 *   --llm-latency <ms>        "os" provider latency (default 400)
 *   --refuse-every <k>        "os" refuses every k-th call (default 0)
 *   --no-bundled              no bundled model: refusals fall to the cloud
 *   --speaker-cost <ms>       virtual ECAPA cost per turn (default 40)
 *   --speak-quiet <ms>        FastLoop speakQuietMs: min quiet before voicing (default 0)
 *   --hold-max <ms>           FastLoop speakHoldMaxMs (default 3000)
 *   --energy-vad              energy VAD instead of Silero
 *   --dump <dir>              write turn_local_<scene>_<mode>.json there
 *   --json                    print the full result as JSON instead of the table
 */
import * as path from "path";
import type { LiveMode } from "../localLlm";
import {
  AUDIO_FIXTURES_DIR,
  formatReport,
  loadModels,
  loadScene,
  replayScene,
  summaryLine,
  writeTurnLocalDump,
  DEFAULT_REPLAY_OPTIONS,
  type ReplayOptions,
  type SceneInput,
} from "./sceneReplay";

/** The checked-in cross-recording voice pairs (the owner is Player6 in poker6). */
export const KNOWN_VOICE_PAIRS = [
  { scene: "family_real", speaker: "Sage", sameAs: { scene: "poker6_real", speaker: "Player6" } },
];
export const SCENE_PACK = ["scene_couple_escalation", "scene_family3", "scene_meeting4"];
export const REAL_SELF: Record<string, string> = { family_real: "Sage", poker6_real: "Player6" };

interface Args {
  scene: string;
  mode: LiveMode | "all";
  self: string | null;
  meta: string | undefined;
  enroll: ReplayOptions["enroll"];
  enrollFrom: string[] | null;
  pairs: ReplayOptions["voicePairs"];
  sttLatency: number;
  sttUntimed: boolean;
  llmLatency: number;
  refuseEvery: number;
  bundled: boolean;
  speakerCost: number;
  speakQuiet: number;
  holdMax: number;
  energyVad: boolean;
  dump: string | null;
  json: boolean;
}

export function parseArgs(argv: string[]): Args {
  const a: Args = {
    scene: "",
    mode: "all",
    self: null,
    meta: undefined,
    enroll: "self",
    enrollFrom: null,
    pairs: [],
    sttLatency: DEFAULT_REPLAY_OPTIONS.stt.finalLatencyMs,
    sttUntimed: false,
    llmLatency: DEFAULT_REPLAY_OPTIONS.osLatencyMs,
    refuseEvery: 0,
    bundled: true,
    speakerCost: DEFAULT_REPLAY_OPTIONS.speakerCostMs,
    speakQuiet: DEFAULT_REPLAY_OPTIONS.speakQuietMs,
    holdMax: DEFAULT_REPLAY_OPTIONS.speakHoldMaxMs,
    energyVad: false,
    dump: null,
    json: false,
  };
  const next = (i: number, flag: string) => {
    if (i + 1 >= argv.length) throw new Error(`${flag} needs a value`);
    return argv[i + 1];
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case "--mode":
        a.mode = next(i++, arg) as Args["mode"];
        break;
      case "--self":
        a.self = next(i++, arg);
        break;
      case "--meta":
        a.meta = next(i++, arg);
        break;
      case "--enroll":
        a.enroll = next(i++, arg) as Args["enroll"];
        break;
      case "--enroll-from":
        a.enrollFrom = next(i++, arg).split(",").filter(Boolean);
        break;
      case "--pair": {
        const [scene, speaker] = next(i++, arg).split(":");
        if (!scene || !speaker) throw new Error("--pair needs <scene>:<speaker>");
        a.pairs.push({ scene: "__self__", speaker: "__self__", sameAs: { scene, speaker } });
        break;
      }
      case "--stt-latency":
        a.sttLatency = Number(next(i++, arg));
        break;
      case "--stt-untimed":
        a.sttUntimed = true;
        break;
      case "--llm-latency":
        a.llmLatency = Number(next(i++, arg));
        break;
      case "--refuse-every":
        a.refuseEvery = Number(next(i++, arg));
        break;
      case "--no-bundled":
        a.bundled = false;
        break;
      case "--speaker-cost":
        a.speakerCost = Number(next(i++, arg));
        break;
      case "--speak-quiet":
        a.speakQuiet = Number(next(i++, arg));
        break;
      case "--hold-max":
        a.holdMax = Number(next(i++, arg));
        break;
      case "--energy-vad":
        a.energyVad = true;
        break;
      case "--dump":
        a.dump = next(i++, arg);
        break;
      case "--json":
        a.json = true;
        break;
      case "--help":
      case "-h":
        throw new Error("help");
      default:
        if (arg.startsWith("--")) throw new Error(`unknown option ${arg}`);
        if (a.scene) throw new Error(`unexpected argument ${arg}`);
        a.scene = arg;
    }
  }
  if (!a.scene) throw new Error("help");
  return a;
}

function usage(): string {
  return (
    "usage: npx tsx src/live/replay/cli.ts <scene|file.wav> [--mode earpiece|speaker|therapist|all] [--self <label>] " +
    "[--meta <path>] [--enroll self|all|none] [--enroll-from a,b] [--pair scene:speaker] [--stt-latency ms] [--stt-untimed] " +
    "[--llm-latency ms] [--refuse-every k] [--no-bundled] [--speaker-cost ms] [--speak-quiet ms] [--hold-max ms] [--energy-vad] [--dump dir] [--json]\n" +
    `fixtures: ${AUDIO_FIXTURES_DIR}`
  );
}

function tryLoadScene(name: string, self?: string | null): SceneInput | null {
  try {
    return loadScene(name, { selfSpeaker: self });
  } catch {
    return null;
  }
}

export async function main(argv: string[]): Promise<number> {
  let args: Args;
  try {
    args = parseArgs(argv);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(msg === "help" ? usage() : `${msg}\n${usage()}`);
    return msg === "help" ? 0 : 2;
  }

  const sceneName = args.scene.endsWith(".wav") ? path.basename(args.scene, ".wav").replace(/^test_recording_/, "") : args.scene.replace(/^test_recording_/, "");
  const self = args.self ?? REAL_SELF[sceneName] ?? undefined;
  const scene = loadScene(args.scene, { metaPath: args.meta, selfSpeaker: self });

  // Cross-scene pool: the rest of the pack by default; real recordings pair
  // with each other through KNOWN_VOICE_PAIRS (or --pair).
  const poolNames = args.enrollFrom ?? [...SCENE_PACK, ...Object.keys(REAL_SELF)].filter((n) => n !== scene.name);
  const enrollFrom = poolNames
    .map((n) => tryLoadScene(n, REAL_SELF[n] ?? null))
    .filter((s): s is SceneInput => s !== null);
  const pairs = [
    ...KNOWN_VOICE_PAIRS,
    ...args.pairs.map((p) => ({ scene: scene.name, speaker: scene.script.selfSpeaker ?? "", sameAs: p.sameAs })),
  ];

  const models = await loadModels({
    ortFactory: null,
    sileroPath: DEFAULT_REPLAY_OPTIONS.sileroPath,
    ecapaPath: DEFAULT_REPLAY_OPTIONS.ecapaPath,
    energyVad: args.energyVad,
  });
  if (!models.embedder && !args.energyVad) {
    console.log("[replay] no ECAPA export found (server/.ecapa_cache/ecapa_<rev>.onnx or MINDSHIFT_ECAPA_ONNX_PATH): speaker-ID OFF for this run");
  }

  const modes: LiveMode[] = args.mode === "all" ? ["earpiece", "speaker", "therapist"] : [args.mode];
  const summaries: string[] = [];
  for (const mode of modes) {
    const result = await replayScene(scene, {
      mode,
      models,
      enroll: args.enroll,
      enrollFrom,
      voicePairs: pairs,
      stt: { ...DEFAULT_REPLAY_OPTIONS.stt, finalLatencyMs: args.sttLatency, wordTimings: !args.sttUntimed },
      osLatencyMs: args.llmLatency,
      osRefuseEveryK: args.refuseEvery,
      bundledAvailable: args.bundled,
      speakerCostMs: args.speakerCost,
      speakQuietMs: args.speakQuiet,
      speakHoldMaxMs: args.holdMax,
      energyVad: args.energyVad,
    });
    if (args.json) {
      const { script: _s, ...rest } = result;
      void _s;
      console.log(JSON.stringify(rest, null, 2));
    } else {
      console.log(formatReport(result));
      console.log("");
    }
    if (args.dump) console.log(`[replay] wrote ${writeTurnLocalDump(result, args.dump)}`);
    summaries.push(summaryLine(result));
  }
  if (!args.json) for (const s of summaries) console.log(s);
  return 0;
}

if (require.main === module) {
  // exitCode, not exit(): a piped stdout is asynchronous in Node and
  // process.exit() would truncate a long --json report mid-string.
  main(process.argv.slice(2)).then(
    (code) => {
      process.exitCode = code;
    },
    (err) => {
      console.error(err instanceof Error ? (err.stack ?? err.message) : String(err));
      process.exitCode = 1;
    },
  );
}
