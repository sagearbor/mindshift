/**
 * Voiceprint enrollment for a replay — the phone-side equivalent of the
 * server's pooled voiceprint (`server/speaker_id.py`): embed each of a
 * speaker's scripted turns with the SAME ECAPA export the loop will match
 * against, fold them into a count-weighted running mean, L2-normalize.
 *
 * Where the turns come from is the point of the exercise:
 *
 * - CROSS-SCENE (the demo case — enroll at home, get matched in the wild):
 *   another recording in which the same voice appears (`voices` in the meta
 *   name the TTS voice per speaker; real recordings are paired explicitly by
 *   the caller, e.g. Sage in family_real <-> Player6 in poker6).
 * - SAME-SCENE: the first `maxSeconds` of that speaker in the recording
 *   being replayed (what happens when a user enrolls and then talks).
 */
import type { Embedder, EnrolledPerson } from "../speakerId";
import { runningMeanEmbedding } from "../speakerId";
import { rmsDbfs } from "../prosody";
import { SILERO_SAMPLE_RATE } from "../vad";
import type { ReplayScript } from "./meta";

export interface EnrollmentSource {
  script: ReplayScript;
  /** Float32 PCM of that script's WAV. */
  pcm: Float32Array;
}

export interface EnrollmentRecord extends EnrolledPerson {
  /** Which script the print was pooled from, and how much of it. */
  fromScene: string;
  fromSpeaker: string;
  crossScene: boolean;
  turnsUsed: number[];
  seconds: number;
}

/** The scene pack tags each speaker with a TTS voice id; two scenes that
 *  share one are cross-scene enrollment pairs. Real recordings are paired by
 *  the caller (`pairs`). */
export function findCrossSceneSource(
  target: ReplayScript,
  speaker: string,
  candidates: EnrollmentSource[],
  pairs: { scene: string; speaker: string; sameAs: { scene: string; speaker: string } }[] = [],
): { source: EnrollmentSource; speaker: string } | null {
  for (const c of candidates) {
    if (c.script.name === target.name) continue;
    const voice = target.voices[speaker];
    if (voice) {
      const match = c.script.speakers.find((s) => c.script.voices[s] === voice);
      if (match) return { source: c, speaker: match };
    }
    for (const p of pairs) {
      if (p.scene === target.name && p.speaker === speaker && p.sameAs.scene === c.script.name) {
        return { source: c, speaker: p.sameAs.speaker };
      }
      if (p.sameAs.scene === target.name && p.sameAs.speaker === speaker && p.scene === c.script.name) {
        return { source: c, speaker: p.speaker };
      }
    }
  }
  return null;
}

export async function enrollFromSource(
  embedder: Embedder,
  source: EnrollmentSource,
  speaker: string,
  opts: { personId: string; displayName: string; isSelf: boolean; maxSeconds?: number; minTurnSeconds?: number; crossScene: boolean },
): Promise<EnrollmentRecord> {
  const maxSeconds = opts.maxSeconds ?? Infinity;
  const minTurn = opts.minTurnSeconds ?? 0.5;
  let pooled: Float32Array | null = null;
  let count = 0;
  let seconds = 0;
  const turnsUsed: number[] = [];
  // Every usable turn's embedding and loudness, kept so a RAISED prototype can
  // be split out below — the replay's stand-in for the loud enrollment prompt
  // a real user reads (VoiceTrainingFlow's last two phrases).
  const takes: { emb: Float32Array; dbfs: number }[] = [];
  for (const t of source.script.turns) {
    if (t.speaker !== speaker) continue;
    const dur = t.end - t.start;
    if (dur < minTurn) continue;
    if (seconds >= maxSeconds) break;
    const a = Math.round(t.start * SILERO_SAMPLE_RATE);
    const b = Math.min(source.pcm.length, Math.round(t.end * SILERO_SAMPLE_RATE));
    if (b <= a) continue;
    const span = source.pcm.subarray(a, b);
    const emb = await embedder.embed(span, SILERO_SAMPLE_RATE);
    takes.push({ emb, dbfs: rmsDbfs(span) });
    pooled = runningMeanEmbedding(pooled, count, emb);
    count += 1;
    seconds += dur;
    turnsUsed.push(t.index);
  }
  if (!pooled) throw new Error(`enroll: no usable turns for "${speaker}" in ${source.script.name}`);
  const raised = raisedPrototype(takes);
  return {
    personId: opts.personId,
    displayName: opts.displayName,
    isSelf: opts.isSelf,
    embedding: pooled,
    raisedEmbedding: raised,
    fromScene: source.script.name,
    fromSpeaker: speaker,
    crossScene: opts.crossScene,
    turnsUsed,
    seconds: Math.round(seconds * 100) / 100,
  };
}

/** A speaker's raised turns must clear their own median by this much to count
 *  as a different register rather than ordinary variation. */
export const RAISED_ENROLL_DB_OVER_MEDIAN = 6;
/** ...and there must be at least this many of them, so one loud sentence does
 *  not become a prototype. */
export const RAISED_ENROLL_MIN_TAKES = 2;

/**
 * The RAISED prototype for a set of enrolment takes, or null.
 *
 * A person's shout sits about as far from their calm print as a different
 * speaker does, so it is stored as a SECOND centroid rather than averaged in
 * (speakerId.ts `raisedEmbedding`). In the app that second print comes from
 * the two loud phrases at the end of voice training; here it comes from the
 * enrolment audio's own loudness, which is the same idea without a prompt:
 * take the turns well above this speaker's median level.
 *
 * The threshold does real work rather than always firing. On the TTS scene
 * pack a "shout" is about +1 dB over the same voice's calm line, so no subset
 * clears +6 and those scenes get no raised print and behave exactly as before.
 * On a real recording — the RAVDESS pair, where the shout is +28 dB — it
 * splits cleanly. A fixture only gets a raised print when the audio genuinely
 * contains a raised voice.
 */
export function raisedPrototype(takes: { emb: Float32Array; dbfs: number }[]): Float32Array | null {
  const levels = takes.map((t) => t.dbfs).filter((d) => Number.isFinite(d));
  if (levels.length < RAISED_ENROLL_MIN_TAKES + 1) return null;
  const sorted = [...levels].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  const loud = takes.filter((t) => Number.isFinite(t.dbfs) && t.dbfs >= median + RAISED_ENROLL_DB_OVER_MEDIAN);
  if (loud.length < RAISED_ENROLL_MIN_TAKES) return null;
  let pooled: Float32Array | null = null;
  loud.forEach((t, i) => {
    pooled = runningMeanEmbedding(pooled, i, t.emb);
  });
  return pooled;
}
