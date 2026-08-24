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
  for (const t of source.script.turns) {
    if (t.speaker !== speaker) continue;
    const dur = t.end - t.start;
    if (dur < minTurn) continue;
    if (seconds >= maxSeconds) break;
    const a = Math.round(t.start * SILERO_SAMPLE_RATE);
    const b = Math.min(source.pcm.length, Math.round(t.end * SILERO_SAMPLE_RATE));
    if (b <= a) continue;
    const emb = await embedder.embed(source.pcm.subarray(a, b), SILERO_SAMPLE_RATE);
    pooled = runningMeanEmbedding(pooled, count, emb);
    count += 1;
    seconds += dur;
    turnsUsed.push(t.index);
  }
  if (!pooled) throw new Error(`enroll: no usable turns for "${speaker}" in ${source.script.name}`);
  return {
    personId: opts.personId,
    displayName: opts.displayName,
    isSelf: opts.isSelf,
    embedding: pooled,
    fromScene: source.script.name,
    fromSpeaker: speaker,
    crossScene: opts.crossScene,
    turnsUsed,
    seconds: Math.round(seconds * 100) / 100,
  };
}
