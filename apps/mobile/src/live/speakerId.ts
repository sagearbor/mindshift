/**
 * On-device speaker identity for the fast loop.
 *
 * Vector math (`l2Normalize` / `cosine` / `runningMeanEmbedding`) and the
 * unknown-speaker clustering (`assignSpeakers`) are ports of server/speaker_id.py
 * and server/watch/diarize.py::assign_speakers, so a voiceprint enrolled on
 * the server matches on the phone with the same threshold (`MATCH_THRESHOLD`
 * 0.65) and the same clustering rule (`CLUSTER_THRESHOLD` 0.55).
 *
 * `SpeakerLabeler` is the session-scoped online form: every finalized turn's
 * embedding is matched greedily against the enrolled people (best cosine,
 * above threshold), else clustered against the "unknown" centroids seen so far
 * in THIS session. Honesty rules carried over from the server:
 *
 * - no enrolled self-voiceprint => `isSelf` is `null`, never a guess;
 * - a turn that couldn't be embedded gets no identity (speaker "Unknown").
 *
 * Embeddings come from the ECAPA ONNX export (Foundation B's
 * scripts/export_ecapa_onnx.py: f32 [1, T] @ 16 kHz -> 192-d L2-normalized)
 * through the `OnnxSession` seam; `EcapaEmbedder` is that adapter.
 */
import type { OnnxSession } from "./ort";
import { float32Tensor } from "./ort";

export const MATCH_THRESHOLD = 0.65;
export const CLUSTER_THRESHOLD = 0.55;
export const ECAPA_DIM = 192;
/**
 * Below this much audio an ECAPA embedding is too unstable to FOUND a new
 * unknown-speaker cluster on. Measured with the replay harness (cosine of a
 * sliding window of the self voice against a print pooled from another
 * scene): p10 0.14 @ 0.6 s, 0.29 @ 1.0 s, 0.43 @ 1.5 s, 0.54 @ 2.0 s, while
 * every other voice stayed < 0.31 at any length. Before this guard the
 * live loop minted a fresh "Speaker X" for nearly every sub-1.5 s fragment
 * (13 clusters for 2 voices on scene_couple_escalation). A short segment may
 * still MATCH an enrolled person or an existing cluster — it just never
 * spawns one; unmatched it is "Unknown" with `isSelf: null`.
 */
export const MIN_CLUSTER_SECONDS = 1.5;

export interface EnrolledPerson {
  personId: string;
  displayName: string;
  isSelf: boolean;
  /** L2-normalized (or raw — cosine normalizes defensively) voiceprint. */
  embedding: ArrayLike<number>;
  /** "<source>@<revision>" the server embedded this print with (its
   *  `model` field); null/absent for a legacy profile. speakerIdSetup.ts
   *  refuses to match a print against a model of a different revision. */
  model?: string | null;
  /** Server-reported length of `embedding`, when known. */
  dim?: number | null;
}

/** Where enrolled voiceprints come from. Production: GET from the server
 *  (Foundation B's people list); tests: a fixture. */
export interface VoiceprintStore {
  list(): Promise<EnrolledPerson[]>;
}

export class StaticVoiceprintStore implements VoiceprintStore {
  constructor(private readonly people: EnrolledPerson[]) {}
  async list() {
    return this.people;
  }
}

export function l2Normalize(vec: ArrayLike<number>): Float32Array {
  const out = Float32Array.from(vec as ArrayLike<number>);
  let norm = 0;
  for (let i = 0; i < out.length; i++) norm += out[i] * out[i];
  norm = Math.sqrt(norm);
  if (norm === 0) return out;
  for (let i = 0; i < out.length; i++) out[i] /= norm;
  return out;
}

/** Cosine similarity in [-1, 1]; 0 for a shape mismatch or empty input. */
export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const na = l2Normalize(a);
  const nb = l2Normalize(b);
  if (na.length !== nb.length || na.length === 0) return 0;
  let dot = 0;
  for (let i = 0; i < na.length; i++) dot += na[i] * nb[i];
  return dot;
}

/** Fold `next` into a count-weighted running mean, renormalized. */
export function runningMeanEmbedding(
  existing: ArrayLike<number> | null,
  existingCount: number,
  next: ArrayLike<number>,
): Float32Array {
  const n = l2Normalize(next);
  if (existing === null || existingCount <= 0) return n;
  const blended = new Float32Array(n.length);
  for (let i = 0; i < n.length; i++) {
    blended[i] = (existing[i] * existingCount + n[i]) / (existingCount + 1);
  }
  return l2Normalize(blended);
}

/**
 * Batch port of diarize.py::assign_speakers — labels "self" | "other-N" |
 * null. Kept verbatim (single self-print) so the golden behaviour is
 * testable against the Python; `SpeakerLabeler` generalizes it to many
 * enrolled people.
 */
export function assignSpeakers(
  embeddings: (ArrayLike<number> | null)[],
  selfPrint: ArrayLike<number> | null,
  selfThreshold = MATCH_THRESHOLD,
  clusterThreshold = CLUSTER_THRESHOLD,
): (string | null)[] {
  const selfVec = selfPrint ? l2Normalize(selfPrint) : null;
  const centroids: Float32Array[] = [];
  const counts: number[] = [];
  const labels: (string | null)[] = [];
  for (const emb of embeddings) {
    if (emb === null) {
      labels.push(null);
      continue;
    }
    if (selfVec && cosine(emb, selfVec) >= selfThreshold) {
      labels.push("self");
      continue;
    }
    let bestIdx: number | null = null;
    let bestScore = -1;
    centroids.forEach((c, idx) => {
      const score = cosine(emb, c);
      if (score > bestScore) {
        bestScore = score;
        bestIdx = idx;
      }
    });
    if (bestIdx !== null && bestScore >= clusterThreshold) {
      const i: number = bestIdx;
      centroids[i] = runningMeanEmbedding(centroids[i], counts[i], emb);
      counts[i] += 1;
      labels.push(`other-${i + 1}`);
    } else {
      centroids.push(l2Normalize(emb));
      counts.push(1);
      labels.push(`other-${centroids.length}`);
    }
  }
  return labels;
}

export interface SpeakerVerdict {
  /** Display label: enrolled display name, or "Speaker A/B/…" for an unknown
   *  cluster (the app's existing convention), or "Unknown" with no embedding. */
  speaker: string;
  personId: string | null;
  displayName: string | null;
  /** true/false when decidable; null when nobody is enrolled as self or the
   *  turn had no embedding — the TurnLocalEvent.is_self contract. */
  isSelf: boolean | null;
  /** Cosine that produced the match / cluster join; null when unmatched. */
  score: number | null;
}

/** "Speaker A", "Speaker B", … for unknown cluster index 0, 1, … */
export function unknownLabel(index: number): string {
  return `Speaker ${String.fromCharCode(65 + (index % 26))}`;
}

export class SpeakerLabeler {
  private readonly people: { person: EnrolledPerson; vec: Float32Array }[];
  private readonly hasSelf: boolean;
  private centroids: Float32Array[] = [];
  private counts: number[] = [];

  constructor(
    people: EnrolledPerson[],
    private readonly matchThreshold = MATCH_THRESHOLD,
    private readonly clusterThreshold = CLUSTER_THRESHOLD,
    private readonly minClusterSeconds = MIN_CLUSTER_SECONDS,
  ) {
    this.people = people.map((person) => ({
      person,
      vec: l2Normalize(person.embedding),
    }));
    this.hasSelf = people.some((p) => p.isSelf);
  }

  get enrolledCount() {
    return this.people.length;
  }

  /** `seconds` is the segment's audio length; omit it to disable the
   *  short-segment guard (batch callers with known-good segments). */
  label(embedding: ArrayLike<number> | null, seconds?: number): SpeakerVerdict {
    if (embedding === null || embedding.length === 0) {
      return { speaker: "Unknown", personId: null, displayName: null, isSelf: null, score: null };
    }
    // Greedy best-above-threshold against every enrolled print.
    let best: { person: EnrolledPerson; score: number } | null = null;
    for (const { person, vec } of this.people) {
      const score = cosine(embedding, vec);
      if (best === null || score > best.score) best = { person, score };
    }
    if (best && best.score >= this.matchThreshold) {
      return {
        speaker: best.person.displayName,
        personId: best.person.personId,
        displayName: best.person.displayName,
        isSelf: best.person.isSelf,
        score: best.score,
      };
    }
    // Unknown: online clustering, order-stable, same as assign_speakers.
    let bestIdx: number | null = null;
    let bestScore = -1;
    this.centroids.forEach((c, idx) => {
      const score = cosine(embedding, c);
      if (score > bestScore) {
        bestScore = score;
        bestIdx = idx;
      }
    });
    let idx: number;
    let score: number | null;
    if (bestIdx !== null && bestScore >= this.clusterThreshold) {
      idx = bestIdx;
      this.centroids[idx] = runningMeanEmbedding(this.centroids[idx], this.counts[idx], embedding);
      this.counts[idx] += 1;
      score = bestScore;
    } else if (seconds !== undefined && seconds < this.minClusterSeconds) {
      // Too short to be evidence of a NEW voice: no cluster, no identity,
      // and no claim about self either way.
      return { speaker: "Unknown", personId: null, displayName: null, isSelf: null, score: null };
    } else {
      this.centroids.push(l2Normalize(embedding));
      this.counts.push(1);
      idx = this.centroids.length - 1;
      score = null;
    }
    return {
      speaker: unknownLabel(idx),
      personId: null,
      displayName: null,
      // With an enrolled self we KNOW this isn't them (it didn't match);
      // without one there's no honest basis either way.
      isSelf: this.hasSelf ? false : null,
      score,
    };
  }

  reset() {
    this.centroids = [];
    this.counts = [];
  }
}

/** Something that turns a turn's PCM into a voiceprint. */
export interface Embedder {
  embed(pcm: Float32Array, sampleRate: number): Promise<Float32Array>;
}

/** ECAPA ONNX export through the OnnxSession seam. */
export class EcapaEmbedder implements Embedder {
  constructor(private readonly session: OnnxSession) {}

  async embed(pcm: Float32Array, sampleRate: number): Promise<Float32Array> {
    if (sampleRate !== 16000) {
      throw new Error(`EcapaEmbedder: expected 16 kHz PCM, got ${sampleRate}`);
    }
    const inputName = this.session.inputNames[0];
    const out = await this.session.run({
      [inputName]: float32Tensor(pcm, [1, pcm.length]),
    });
    const first = out[this.session.outputNames[0]];
    if (!first) throw new Error("EcapaEmbedder: model returned no output");
    // Output may be [1, 192] or [1, 1, 192]; flatten and normalize — the
    // export already L2-normalizes, this is belt and braces.
    return l2Normalize(first.data as Float32Array);
  }
}
