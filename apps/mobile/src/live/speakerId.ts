/**
 * On-device speaker identity for the fast loop.
 *
 * Vector math (`l2Normalize` / `cosine` / `runningMeanEmbedding`) and the
 * unknown-speaker clustering (`assignSpeakers`) are ports of server/speaker_id.py
 * and server/watch/diarize.py::assign_speakers, so a voiceprint enrolled on
 * the server matches on the phone with the same threshold (`MATCH_THRESHOLD`
 * 0.65) and the same clustering rule (`CLUSTER_THRESHOLD` 0.55).
 * `identifyClusters` is the port of speaker_id.identify_from_embeddings —
 * the absolute bar PLUS the cross-recording CONTRAST match (see the
 * `CROSS_MATCH_*` constants) with the same greedy one-to-one assignment.
 *
 * `SpeakerLabeler` is the session-scoped online form: every finalized turn's
 * embedding is matched greedily against the enrolled people (best cosine,
 * above threshold), else clustered against the "unknown" centroids seen so far
 * in THIS session. After every cluster update the whole set of running
 * centroids is re-identified with `identifyClusters`, so a cluster can GAIN a
 * person once a second cluster exists to contrast against, and can LOSE it to
 * a later cluster that beats it by the margin (a person is one voice). The
 * raw "Speaker A/B" label stays the wire key either way; the identity rides
 * along on `personId` / `isSelf` / `basis`. Honesty rules carried over from
 * the server:
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
/**
 * Cross-recording ("contrast") match — ports of server/speaker_id.py's
 * CROSS_MATCH_* (read the calibration note there). The same person scores
 * only 0.24-0.45 against a print from ANOTHER room/mic (the owner's real
 * print: restaurant 0.76 absolute, family 0.73, poker night 0.42), so the
 * 0.65 bar alone never called the owner "self" in a real live session.
 * A cluster is accepted as person P below the bar when ALL of: cosine >=
 * `CROSS_MATCH_THRESHOLD`; P's print pools >= `CROSS_MATCH_MIN_SETTINGS`
 * distinct recordings (`EnrolledPerson.settings`); at least two clusters
 * exist to contrast; and this cluster beats every OTHER cluster's score for
 * P by >= `CROSS_MATCH_MARGIN` (measured owner-vs-runner-up gaps 0.16-0.63;
 * non-owners <= 0.19). Every match records its `basis` so a contrast "You"
 * is never mistaken for a 0.65 one.
 */
export const CROSS_MATCH_THRESHOLD = 0.4;
export const CROSS_MATCH_MARGIN = 0.15;
export const CROSS_MATCH_MIN_SETTINGS = 2;

/** How a (cluster, person) pair cleared the bar — server `match_basis`. */
/** "solo" is the journal's rule only: a lone voice (no second cluster yet)
 *  at >= CROSS_MATCH_THRESHOLD against a >= 2-recording self print —
 *  contrast can't run without a second voice, and a stranger measured
 *  <= 0.28 across settings. Never used for live coaching verdicts. */
export type MatchBasis = "absolute" | "contrast" | "solo";
// Merge threshold for the ONLINE (live, on-device) unknown-speaker clustering.
// LOWER than the server/batch value (server/watch/diarize.py keeps 0.55) on
// purpose: live turns are short and the on-device ECAPA embedding of the SAME
// voice only reaches cosine ~0.43 @ 1.5 s / ~0.54 @ 2.0 s against the running
// centroid (the p10 numbers documented below), while a DIFFERENT voice stays
// < 0.31 in the clean two-party case. At 0.55 the same person kept scoring
// below threshold and founding fresh clusters — a real 2-voice therapist
// session split into 5 speakers (dx-45DS-H9DP, Pixel 10, 2026-08-26); the
// scene_couple_escalation fixture split 2 voices into 3.
//
// 0.48 is the tuned compromise, validated on the fixture pack: it merges the
// couple's stray fragment (2 speakers, 12/13 attribution, was 3/11) and drops
// a family_real cluster, WHILE keeping poker6's six distinct voices as six and
// leaving the 3-party family scene unchanged. The one cost is scene_meeting4
// (a 4-party business meeting, not the couples/family product core), whose
// speakers sit unusually close (0.48-0.55) and lose some separation
// (14->11/17). Batch diarization runs on longer, cleaner segments and keeps
// 0.55. Lower this further only with fresh fixture evidence (harness rule:
// investigate a regression, don't just lower).
export const CLUSTER_THRESHOLD = 0.48;
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
  /** Distinct recordings pooled into the print (server `settings`); absent
   *  / 0 counts as ONE, which keeps the contrast match off for that person
   *  (a single-recording print is exactly the case it must not trust). */
  settings?: number | null;
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

export interface ClusterIdentity {
  personId: string;
  basis: MatchBasis;
  /** Cosine of the cluster centroid against the person's print. */
  score: number;
}

export interface IdentifyOptions {
  matchThreshold?: number;
  crossMatchThreshold?: number;
  crossMatchMargin?: number;
  crossMatchMinSettings?: number;
}

/** Cosine rounded the way the server reports it (`round(x, 4)`), so the
 *  margin/floor comparisons land on the same side on both ends. */
function scoreOf(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return Math.round(cosine(a, b) * 1e4) / 1e4;
}

/**
 * Pure port of speaker_id.identify_from_embeddings over already-computed
 * cluster embeddings: which cluster is which enrolled person, and why.
 *
 * Two ways a (cluster, person) pair clears the bar — ABSOLUTE (cosine >=
 * `MATCH_THRESHOLD`) or CONTRAST (the four conditions on `CROSS_MATCH_*`
 * above). Assignment is greedy one-to-one, highest score first: each cluster
 * gets at most one person and each person wins at most one cluster (if the
 * clustering split one voice in two, only the stronger half is labeled).
 * Ties break deterministically (cluster label, then person id). Below both
 * bars => absent from the result. Parity is pinned by
 * __tests__/fixtures/speakerCrossMatch.json, generated from the Python.
 */
export function identifyClusters(
  clusterEmbeddings: ReadonlyMap<string, ArrayLike<number>>,
  people: readonly EnrolledPerson[],
  opts: IdentifyOptions = {},
): Map<string, ClusterIdentity> {
  const threshold = opts.matchThreshold ?? MATCH_THRESHOLD;
  const crossThreshold = opts.crossMatchThreshold ?? CROSS_MATCH_THRESHOLD;
  const margin = opts.crossMatchMargin ?? CROSS_MATCH_MARGIN;
  const minSettings = opts.crossMatchMinSettings ?? CROSS_MATCH_MIN_SETTINGS;
  const prints = people.map((person) => ({ person, vec: l2Normalize(person.embedding) }));
  const labels = Array.from(clusterEmbeddings.keys());
  // scores[label][personId]
  const scores = new Map<string, Map<string, number>>();
  for (const label of labels) {
    const emb = clusterEmbeddings.get(label) as ArrayLike<number>;
    const row = new Map<string, number>();
    for (const { person, vec } of prints) row.set(person.personId, scoreOf(emb, vec));
    scores.set(label, row);
  }
  const candidates: { score: number; label: string; personId: string; basis: MatchBasis }[] = [];
  for (const label of labels) {
    const row = scores.get(label) as Map<string, number>;
    for (const { person } of prints) {
      const pid = person.personId;
      const score = row.get(pid) as number;
      if (score >= threshold) {
        candidates.push({ score, label, personId: pid, basis: "absolute" });
        continue;
      }
      if (score < crossThreshold || labels.length < 2) continue;
      const settings = Math.trunc(person.settings || 0) || 1;
      if (settings < minSettings) continue;
      let runnerUp = -Infinity;
      for (const other of labels) {
        if (other === label) continue;
        runnerUp = Math.max(runnerUp, (scores.get(other) as Map<string, number>).get(pid) as number);
      }
      if (score - runnerUp >= margin) candidates.push({ score, label, personId: pid, basis: "contrast" });
    }
  }
  candidates.sort((a, b) => {
    if (a.score !== b.score) return b.score - a.score;
    if (a.label !== b.label) return a.label < b.label ? -1 : 1;
    if (a.personId !== b.personId) return a.personId < b.personId ? -1 : 1;
    return 0;
  });
  const matched = new Map<string, ClusterIdentity>();
  const takenPeople = new Set<string>();
  for (const c of candidates) {
    if (matched.has(c.label) || takenPeople.has(c.personId)) continue;
    matched.set(c.label, { personId: c.personId, basis: c.basis, score: c.score });
    takenPeople.add(c.personId);
  }
  return matched;
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
  /** WHY the turn carries a person: "absolute" (>= MATCH_THRESHOLD) or
   *  "contrast" (the in-session cross-recording rule); null for an
   *  unidentified cluster or a mid-call binding. */
  basis: MatchBasis | null;
  /** Cosine of this turn against the owner's print (null without a self
   *  print or embedding) — reported even when it did NOT match, so a caller
   *  with a looser context (the journal's solo rule) can decide. */
  selfScore?: number | null;
}

/** A cluster's current person, as the labeler resolves it (raw label keyed). */
export interface ClusterAssignment extends ClusterIdentity {
  displayName: string;
  isSelf: boolean;
}

const NO_IDENTITY: SpeakerVerdict = Object.freeze({
  speaker: "Unknown",
  personId: null,
  displayName: null,
  isSelf: null,
  score: null,
  basis: null,
});

/** "Speaker A", "Speaker B", … for unknown cluster index 0, 1, … */
export function unknownLabel(index: number): string {
  return `Speaker ${String.fromCharCode(65 + (index % 26))}`;
}

export class SpeakerLabeler {
  private readonly people: { person: EnrolledPerson; vec: Float32Array }[];
  private hasSelf: boolean;
  private centroids: Float32Array[] = [];
  private counts: number[] = [];
  /** cluster index -> current identity (identifyClusters over all centroids). */
  private identities: Map<number, ClusterIdentity> = new Map();
  /** Bumps whenever `identities` changes — a cheap "did anything move" check. */
  private revision = 0;

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

  /** Whether anyone enrolled is the owner — the basis for `isSelf: false`. */
  get hasSelfPrint(): boolean {
    return this.hasSelf;
  }

  /** Increments each time a cluster gains, loses or changes its person. */
  get identityRevision(): number {
    return this.revision;
  }

  /** Recordings pooled into the owner's print (`settings`); 0 without one. */
  get selfSettings(): number {
    const self = this.people.find((p) => p.person.isSelf);
    return self ? Math.trunc(self.person.settings || 0) || 1 : 0;
  }

  /** Running unknown-voice clusters this session. */
  get clusterCount(): number {
    return this.centroids.length;
  }

  /** Every identified cluster, keyed by its raw "Speaker X" label. */
  clusterAssignments(): Map<string, ClusterAssignment> {
    const out = new Map<string, ClusterAssignment>();
    for (const [idx, id] of this.identities) {
      const person = this.personById(id.personId);
      if (!person) continue;
      out.set(unknownLabel(idx), { ...id, displayName: person.displayName, isSelf: person.isSelf });
    }
    return out;
  }

  private personById(personId: string): EnrolledPerson | null {
    return this.people.find((p) => p.person.personId === personId)?.person ?? null;
  }

  /** Re-run the one-to-one identification over ALL current centroids. */
  private reidentify(): void {
    const clusters = new Map<string, Float32Array>();
    this.centroids.forEach((c, i) => clusters.set(String(i), c));
    const next = new Map<number, ClusterIdentity>();
    for (const [key, id] of identifyClusters(clusters, this.people.map((p) => p.person), {
      matchThreshold: this.matchThreshold,
    })) {
      next.set(Number(key), id);
    }
    let changed = next.size !== this.identities.size;
    if (!changed) {
      for (const [idx, id] of next) {
        const prev = this.identities.get(idx);
        if (!prev || prev.personId !== id.personId || prev.basis !== id.basis) {
          changed = true;
          break;
        }
      }
    }
    this.identities = next;
    if (changed) this.revision += 1;
  }

  /**
   * Add (or replace, by personId) an enrolled person MID-SESSION — the
   * mid-call "that's Mom" flow learns a voice from the session's own pooled
   * audio and hands the print here so later turns match by voiceprint, not
   * just by cluster. `isSelf` on the new person makes unmatched clusters
   * honestly "not self" from now on, exactly as a pre-enrolled self would.
   */
  addPerson(person: EnrolledPerson): void {
    const idx = this.people.findIndex((p) => p.person.personId === person.personId);
    const entry = { person, vec: l2Normalize(person.embedding) };
    if (idx >= 0) this.people[idx] = entry;
    else this.people.push(entry);
    if (person.isSelf) this.hasSelf = true;
    this.reidentify();
  }

  /** `seconds` is the segment's audio length; omit it to disable the
   *  short-segment guard (batch callers with known-good segments). */
  label(embedding: ArrayLike<number> | null, seconds?: number): SpeakerVerdict {
    if (embedding === null || embedding.length === 0) return { ...NO_IDENTITY };
    // Greedy best-above-threshold against every enrolled print — the
    // ABSOLUTE path, unchanged: a turn that clears the 0.65 bar carries the
    // person outright and founds no cluster.
    let best: { person: EnrolledPerson; score: number } | null = null;
    let selfScore: number | null = null;
    for (const { person, vec } of this.people) {
      const score = cosine(embedding, vec);
      if (person.isSelf) selfScore = score;
      if (best === null || score > best.score) best = { person, score };
    }
    if (best && best.score >= this.matchThreshold) {
      return {
        speaker: best.person.displayName,
        personId: best.person.personId,
        displayName: best.person.displayName,
        isSelf: best.person.isSelf,
        score: best.score,
        basis: "absolute",
        selfScore,
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
      return { ...NO_IDENTITY, selfScore };
    } else {
      this.centroids.push(l2Normalize(embedding));
      this.counts.push(1);
      idx = this.centroids.length - 1;
      score = null;
    }
    // The centroid set changed: re-resolve who is who over ALL clusters (the
    // contrast rule needs every other cluster's score, and a person may move
    // to the cluster that now beats the rest by the margin).
    this.reidentify();
    const identity = this.identities.get(idx) ?? null;
    const person = identity ? this.personById(identity.personId) : null;
    if (identity && person) {
      return {
        // The raw label stays the wire key: an inferred identity can be
        // revised, and the session record keeps one stable key per voice.
        speaker: unknownLabel(idx),
        personId: person.personId,
        displayName: person.displayName,
        isSelf: person.isSelf,
        score: identity.score,
        basis: identity.basis,
        selfScore,
      };
    }
    return {
      speaker: unknownLabel(idx),
      personId: null,
      displayName: null,
      // With an enrolled self we KNOW this isn't them (it didn't match);
      // without one there's no honest basis either way.
      isSelf: this.hasSelf ? false : null,
      score,
      basis: null,
      selfScore,
    };
  }

  reset() {
    this.centroids = [];
    this.counts = [];
    this.identities = new Map();
    this.revision = 0;
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
