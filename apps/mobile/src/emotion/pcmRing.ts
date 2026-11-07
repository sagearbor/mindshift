/**
 * Bounded in-memory audio ring + the flag-clip scheduler.
 *
 * The ring keeps the last `capacitySamples` of the live 16 kHz mono int16
 * stream. A flag (or an auto disagreement clip) SNAPSHOTS the "before" part
 * at request time and then collects the "after" part as audio keeps arriving;
 * when the after-part is complete the clip is handed out for saving. Nothing
 * here touches a disk or a network.
 */
export const RING_SAMPLE_RATE = 16000;

export class PcmRing {
  private chunks: Int16Array[] = [];
  private held = 0;
  /** Samples ever pushed (absolute stream position of the next sample). */
  total = 0;

  constructor(readonly capacitySamples: number) {}

  push(pcm: Int16Array): void {
    if (pcm.length === 0) return;
    this.chunks.push(pcm.slice());
    this.held += pcm.length;
    this.total += pcm.length;
    while (this.held - this.chunks[0].length >= this.capacitySamples) {
      this.held -= this.chunks[0].length;
      this.chunks.shift();
    }
  }

  /** Absolute position of the oldest retained sample (>= chunk-granular). */
  get oldest(): number {
    return this.total - this.held;
  }

  /** Samples [startAbs, endAbs) clamped to what is still retained. */
  read(startAbs: number, endAbs: number): Int16Array {
    const s = Math.max(startAbs, this.oldest);
    const e = Math.min(endAbs, this.total);
    if (e <= s) return new Int16Array(0);
    const out = new Int16Array(e - s);
    let pos = this.oldest;
    let w = 0;
    for (const c of this.chunks) {
      const cs = pos;
      const ce = pos + c.length;
      pos = ce;
      if (ce <= s) continue;
      if (cs >= e) break;
      const from = Math.max(s, cs) - cs;
      const to = Math.min(e, ce) - cs;
      out.set(c.subarray(from, to), w);
      w += to - from;
    }
    return out;
  }

  /** The last `n` samples (fewer if less is held). */
  tail(n: number): Int16Array {
    return this.read(this.total - n, this.total);
  }
}

export interface ClipRequest {
  id: string;
  /** Free-form reason: "flag:bad", "flag:good", "flag:ignore", "disagree". */
  reason: string;
  atMs: number;
}

export interface FinishedClip extends ClipRequest {
  pcm: Int16Array;
  /** Seconds of "before" audio actually available (<= requested). */
  beforeS: number;
  afterS: number;
}

interface Pending extends ClipRequest {
  before: Int16Array;
  after: Int16Array[];
  afterSamples: number;
  wantAfter: number;
}

export class ClipScheduler {
  private pending: Pending[] = [];

  constructor(private readonly ring: PcmRing, private readonly sampleRate = RING_SAMPLE_RATE) {}

  /** Snapshot the last `beforeS` seconds now; the clip completes once
   *  `afterS` more seconds have been fed through {@link onPcm}. */
  request(req: ClipRequest, beforeS: number, afterS: number): void {
    this.pending.push({
      ...req,
      before: this.ring.tail(Math.round(beforeS * this.sampleRate)),
      after: [],
      afterSamples: 0,
      wantAfter: Math.round(afterS * this.sampleRate),
    });
  }

  get pendingCount(): number {
    return this.pending.length;
  }

  /** Feed the same frames that went into the ring; returns clips that just completed. */
  onPcm(pcm: Int16Array): FinishedClip[] {
    const done: FinishedClip[] = [];
    const still: Pending[] = [];
    for (const p of this.pending) {
      const room = p.wantAfter - p.afterSamples;
      if (room > 0) {
        const take = pcm.length <= room ? pcm : pcm.subarray(0, room);
        p.after.push(take.slice());
        p.afterSamples += take.length;
      }
      if (p.afterSamples >= p.wantAfter) done.push(this.finish(p));
      else still.push(p);
    }
    this.pending = still;
    return done;
  }

  /** Complete everything pending with whatever "after" audio exists (session
   *  ended / mode turned off), so a late flag is not silently lost. */
  flushAll(): FinishedClip[] {
    const done = this.pending.map((p) => this.finish(p));
    this.pending = [];
    return done;
  }

  private finish(p: Pending): FinishedClip {
    const pcm = new Int16Array(p.before.length + p.afterSamples);
    pcm.set(p.before, 0);
    let w = p.before.length;
    for (const a of p.after) {
      pcm.set(a, w);
      w += a.length;
    }
    return {
      id: p.id,
      reason: p.reason,
      atMs: p.atMs,
      pcm,
      beforeS: p.before.length / this.sampleRate,
      afterS: p.afterSamples / this.sampleRate,
    };
  }
}
