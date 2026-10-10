/**
 * The PHONE side of the recording-replay pipeline (scripts/recording_replay.py).
 *
 *   tsx apps/mobile/src/live/replay/recordingReplay.ts \
 *     --wav <16k mono wav> --meta <meta.json> --out <phone.json> \
 *     [--mode earpiece|speaker|therapist] [--enroll profile|same|none] \
 *     [--profile tmp/private_fixtures/owner_profile.json] [--self You]
 *
 * Runs the REAL on-device fast loop (`replayScene`: Silero VAD + segmenter +
 * ECAPA speaker-ID, the real nudge policy) over a real recording whose meta
 * the Python side wrote from Deepgram's words (+ an annotation's emotions as
 * the tone stand-in). What differs from `cli.ts`:
 *
 * - `--enroll profile` enrolls the OWNER'S OWN server voiceprint (the
 *   `owner_profile.json` document, the server's speechbrain ECAPA space —
 *   the same export the phone runs, parity-pinned by ecapa_reference.json)
 *   instead of pooling one from this recording. That is what the phone
 *   actually holds; `same` (pool from this recording) is the optimistic
 *   ceiling, reported alongside.
 * - Output is one JSON file the Python side reads: every turn_local with
 *   the audio second at which the phone would have sent it, the loop's own
 *   turns, nudges, haptics and scores. JSON-safe (typed arrays dropped).
 *
 * STT and the local LLM stay scripted (replay/fakes.ts): Android's
 * recognizer and Gemini Nano do not run on a laptop. Their text/tone come
 * from the meta, so the phone's `suggestion` is a placeholder the Python
 * side strips before the server sees the turn.
 */
import * as fs from "fs";
import * as path from "path";
import type { LiveMode } from "../localLlm";
import type { SpeakerLabelerOptions } from "../speakerId";
import type { TurnLocalEvent } from "../types";
import type { EnrollmentRecord } from "./enroll";
import { loadModels, loadScene, replayScene, DEFAULT_REPLAY_OPTIONS, type LoadedModels, type ReplayResult } from "./sceneReplay";

export type RecordingEnroll = "profile" | "same" | "none";

export interface RecordingArgs {
  wav: string;
  meta: string;
  out: string;
  mode: LiveMode;
  enroll: RecordingEnroll;
  profile: string | null;
  self: string | null;
  /** `--speaker-opts '<json>'`: labeler tuning for identity sweeps
   *  (scripts/recreplay/identity_eval.py); null = the shipped labeler. */
  speakerOptions: SpeakerLabelerOptions | null;
}

export function parseRecordingArgs(argv: string[]): RecordingArgs {
  const a: Partial<RecordingArgs> = { mode: "earpiece", enroll: "same", profile: null, self: null, speakerOptions: null };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    const val = () => {
      if (i + 1 >= argv.length) throw new Error(`${flag} needs a value`);
      return argv[++i];
    };
    switch (flag) {
      case "--wav":
        a.wav = val();
        break;
      case "--meta":
        a.meta = val();
        break;
      case "--out":
        a.out = val();
        break;
      case "--mode":
        a.mode = val() as LiveMode;
        break;
      case "--enroll":
        a.enroll = val() as RecordingEnroll;
        break;
      case "--profile":
        a.profile = val();
        break;
      case "--self":
        a.self = val();
        break;
      case "--speaker-opts": {
        const parsed: unknown = JSON.parse(val());
        if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new Error("--speaker-opts must be a JSON object");
        }
        a.speakerOptions = parsed as SpeakerLabelerOptions;
        break;
      }
      default:
        throw new Error(`unknown option ${flag}`);
    }
  }
  for (const k of ["wav", "meta", "out"] as const) if (!a[k]) throw new Error(`--${k} is required`);
  if (!["profile", "same", "none"].includes(a.enroll as string)) throw new Error(`--enroll ${a.enroll}: profile|same|none`);
  if (a.enroll === "profile" && !a.profile) throw new Error("--enroll profile needs --profile <owner_profile.json>");
  return a as RecordingArgs;
}

/** A server voiceprint document (`/voice/profile` shape: `embedding`,
 *  optional `raised_embedding`, `model`, `samples`) as the phone's enrolled
 *  self. `selfLabel` is the meta's label for the owner, so attribution
 *  scoring compares like with like. */
export function profileEnrollment(doc: Record<string, unknown>, selfLabel: string): EnrollmentRecord {
  const emb = doc.embedding;
  if (!Array.isArray(emb) || emb.length === 0 || !emb.every((x) => typeof x === "number")) {
    throw new Error("profile: no numeric `embedding` array");
  }
  const raised = Array.isArray(doc.raised_embedding) && doc.raised_embedding.length === emb.length ? Float32Array.from(doc.raised_embedding as number[]) : null;
  const samples = Array.isArray(doc.samples) ? doc.samples.length : 0;
  return {
    personId: typeof doc.person_id === "string" ? doc.person_id : "self",
    displayName: selfLabel,
    isSelf: true,
    embedding: Float32Array.from(emb as number[]),
    raisedEmbedding: raised,
    model: typeof doc.model === "string" ? doc.model : null,
    dim: emb.length,
    fromScene: "owner_profile",
    fromSpeaker: typeof doc.display_name === "string" ? doc.display_name : "self",
    crossScene: true,
    turnsUsed: [],
    seconds: samples,
  };
}

export interface SentTurnLocal extends TurnLocalEvent {
  /** Audio second at which the phone would have put this on the socket. */
  sent_at_audio_s: number;
}

export interface RecordingReplayOutput {
  scene: string;
  mode: LiveMode;
  generated_by: string;
  durationSec: number;
  selfSpeaker: string | null;
  enrollment: { mode: RecordingEnroll; records: Omit<EnrollmentRecord, "embedding" | "raisedEmbedding">[] };
  capability: { vad: string; speakerId: boolean; ecapaPath: string | null };
  sent: SentTurnLocal[];
  turns: (Omit<ReplayResult["turns"][number], "activation" | "overlap"> & { beforeLoopUp?: boolean })[];
  nudges: ReplayResult["nudgeLog"];
  haptics: ReplayResult["hapticLog"];
  positives: ReplayResult["positives"];
  spoken: ReplayResult["spoken"];
  attribution: ReplayResult["attribution"];
  labelLog: ReplayResult["labelLog"];
  boundaries: ReplayResult["boundaries"];
  latency: ReplayResult["latency"];
  wallMs: number;
}

export function outputFor(r: ReplayResult, enroll: RecordingEnroll): RecordingReplayOutput {
  const records = r.capability.enrolled.map((e) => {
    const { embedding: _e, raisedEmbedding: _r, ...rest } = e;
    void _e;
    void _r;
    return rest;
  });
  return {
    scene: r.scene,
    mode: r.mode,
    generated_by: "apps/mobile/src/live/replay/recordingReplay.ts",
    durationSec: r.durationSec,
    selfSpeaker: r.script.selfSpeaker,
    enrollment: { mode: enroll, records },
    capability: { vad: r.capability.vad, speakerId: r.capability.speakerId, ecapaPath: r.options.ecapaPath },
    sent: r.sent.map((e, i) => ({ ...e, sent_at_audio_s: Math.round(r.sentAtMs[i]) / 1000 })),
    turns: r.turns.map((t) => {
      const { activation: _a, overlap: _o, ...rest } = t;
      void _a;
      void _o;
      return rest;
    }),
    nudges: r.nudgeLog,
    haptics: r.hapticLog,
    positives: r.positives,
    spoken: r.spoken,
    attribution: r.attribution,
    labelLog: r.labelLog,
    boundaries: r.boundaries,
    latency: r.latency,
    wallMs: Math.round(r.wall.totalMs),
  };
}

export async function runRecordingReplay(args: RecordingArgs, models?: LoadedModels): Promise<RecordingReplayOutput> {
  const scene = loadScene(args.wav, { metaPath: args.meta, selfSpeaker: args.self ?? undefined });
  const loaded =
    models ??
    (await loadModels({ ortFactory: null, sileroPath: DEFAULT_REPLAY_OPTIONS.sileroPath, ecapaPath: DEFAULT_REPLAY_OPTIONS.ecapaPath, energyVad: false }));
  let enrolled: EnrollmentRecord[] | undefined;
  if (args.enroll === "profile") {
    const doc = JSON.parse(fs.readFileSync(args.profile as string, "utf8")) as Record<string, unknown>;
    enrolled = [profileEnrollment(doc, scene.script.selfSpeaker ?? "You")];
  }
  const result = await replayScene(scene, {
    mode: args.mode,
    models: loaded,
    enroll: args.enroll === "none" ? "none" : "self",
    enrollFrom: [],
    enrolled,
    ...(args.speakerOptions ? { speakerOptions: args.speakerOptions } : {}),
  });
  const out = outputFor(result, args.enroll);
  fs.mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
  fs.writeFileSync(args.out, JSON.stringify(out, null, 1) + "\n");
  return out;
}

if (require.main === module) {
  let args: RecordingArgs;
  try {
    args = parseRecordingArgs(process.argv.slice(2));
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    process.exitCode = 2;
    throw err;
  }
  runRecordingReplay(args).then(
    (out) => {
      const a = out.attribution;
      console.log(
        `[recording-replay] ${out.scene} ${out.mode} enroll=${out.enrollment.mode} speakerId=${out.capability.speakerId}: ` +
          `${out.sent.length} turn_local, self ${a.selfCorrect}/${a.selfTotal}, attribution ${a.correct}/${a.total} -> ${args.out}`,
      );
    },
    (err) => {
      console.error(err instanceof Error ? (err.stack ?? err.message) : String(err));
      process.exitCode = 1;
    },
  );
}
