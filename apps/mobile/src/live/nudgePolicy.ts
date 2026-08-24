/**
 * Nudge/escalation policy — the TypeScript mirror of server/nudge_policy.py
 * (and the watch's NudgeStateMachine.kt). Thresholds/semantics must stay
 * identical across the three: per-channel hysteresis, strict-greater cooldown,
 * half-up rounding. The executable contract is
 * server/tests/fixtures/policy_vectors/nudge_policy.json, replayed here by
 * __tests__/liveNudgePolicy.test.ts for every case tagged `phone`/`server`.
 *
 * The phone runs a single lane ("A") over the coached user's OWN turns: a
 * level > 0 nudge fires a haptic (expo-haptics) and an on-screen flash. The
 * `selfTurnVectorEvents` helper turns a finalized self turn's prosody + text
 * tone into the vector events this policy consumes, using the SAME loudness
 * thresholds as the watch (+6/+10/+14 dB over baseline -> level 1/2/3).
 */

export type VectorName = "yelling" | "aggressive_tone" | "interrupting" | "airtime" | "hr_spike";
export type Channel = "A" | "B";

export interface VectorSubscription {
  vector: VectorName;
  sensitivity?: number;
  haptics?: boolean;
  channel?: Channel;
}

export interface VectorEvent {
  vector: VectorName;
  level: number;
  t: number;
  value?: number;
}

export interface NudgeEvent {
  channel: Channel;
  level: number;
  t: number;
  vectors: string[];
}

export const DEFAULT_CHANNELS: Channel[] = ["A", "B"];

/** Half-up rounding (Kotlin Math.round), not banker's — 0.5 -> 1, 1.5 -> 2. */
export function roundHalfUp(x: number): number {
  return x >= 0 ? Math.floor(x + 0.5) : Math.ceil(x - 0.5);
}

/** watch/models.py's default: hr_spike rides channel B, everything else A. */
function defaultChannel(vector: VectorName): Channel {
  return vector === "hr_spike" ? "B" : "A";
}

export class NudgePolicy {
  readonly channels: Channel[];
  private readonly subs: Required<VectorSubscription>[];
  private readonly levels = new Map<Channel, number>();
  private readonly lastQualifyingT = new Map<Channel, number>();

  constructor(
    subs: VectorSubscription[],
    private readonly cooldownS = 20.0,
    channels: Channel[] = DEFAULT_CHANNELS,
  ) {
    if (channels.length === 0) throw new Error("NudgePolicy needs at least one channel");
    this.channels = [...new Set(channels)];
    this.subs = subs.map((s) => ({
      vector: s.vector,
      sensitivity: s.sensitivity ?? 1.0,
      haptics: s.haptics ?? true,
      channel: s.channel ?? defaultChannel(s.vector),
    }));
    for (const c of this.channels) {
      this.levels.set(c, 0);
      this.lastQualifyingT.set(c, 0);
    }
  }

  onEvents(events: VectorEvent[], t: number): NudgeEvent[] {
    const nudges: NudgeEvent[] = [];
    const subByVector = new Map<string, Required<VectorSubscription>>();
    for (const s of this.subs) if (s.haptics) subByVector.set(s.vector, s);

    const eventMax = new Map<Channel, { level: number; vectors: string[] }>();
    for (const c of this.channels) eventMax.set(c, { level: 0, vectors: [] });
    for (const e of events) {
      const sub = subByVector.get(e.vector);
      if (!sub) continue;
      const slot = eventMax.get(sub.channel);
      if (!slot) continue; // a lane this policy doesn't run
      let scaled = roundHalfUp(e.level * sub.sensitivity);
      scaled = Math.min(3, Math.max(0, scaled));
      if (scaled > slot.level) {
        slot.level = scaled;
        slot.vectors = [e.vector];
      } else if (scaled === slot.level && scaled > 0) {
        slot.vectors.push(e.vector);
      }
    }

    for (const channel of this.channels) {
      const { level: E, vectors } = eventMax.get(channel)!;
      const current = this.levels.get(channel)!;
      if (E > current) {
        this.levels.set(channel, E);
        this.lastQualifyingT.set(channel, t);
        nudges.push({ channel, level: E, t, vectors: [...new Set(vectors)].sort() });
      } else if (E === current && current > 0) {
        this.lastQualifyingT.set(channel, t);
      } else if (current > 0 && t - this.lastQualifyingT.get(channel)! > this.cooldownS) {
        const next = current - 1;
        this.levels.set(channel, next);
        this.lastQualifyingT.set(channel, t);
        nudges.push({ channel, level: next, t, vectors: [] });
      }
    }
    return nudges;
  }

  current(): Record<string, number> {
    const out: Record<string, number> = {};
    for (const c of this.channels) out[c] = this.levels.get(c)!;
    return out;
  }
}

// ---------------------------------------------------------------------------
// Phone-side inputs: a finalized self turn -> vector events
// ---------------------------------------------------------------------------

/** The watch's loudness ladder (watch/vectors.py YELLING_LEVELS). */
export const YELLING_LEVELS: [number, number][] = [
  [14.0, 3],
  [10.0, 2],
  [6.0, 1],
];

export function yellingLevel(dbOverBaseline: number): number {
  for (const [threshold, level] of YELLING_LEVELS) {
    if (dbOverBaseline >= threshold) return level;
  }
  return 0;
}

/** Text-tone ladder for the aggressive_tone vector: the higher of the turn's
 *  frustration / defensiveness scores (0–100) -> 0..3. */
export function aggressiveToneLevel(frustration: number | null, defensiveness: number | null): number {
  const v = Math.max(frustration ?? 0, defensiveness ?? 0);
  if (v >= 85) return 3;
  if (v >= 70) return 2;
  if (v >= 50) return 1;
  return 0;
}

/**
 * Running loudness baseline over the user's own turns (median of what we've
 * heard so far), so "yelling" means "louder than YOU usually are on this
 * phone" — the same relative reasoning as the server's prosody labels.
 */
export class LoudnessBaseline {
  private readonly values: number[] = [];
  constructor(private readonly minTurns = 2) {}
  /** Baseline dBFS, or null until enough turns have been heard. */
  get value(): number | null {
    if (this.values.length < this.minTurns) return null;
    const s = [...this.values].sort((a, b) => a - b);
    const mid = s.length >> 1;
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  }
  /** dB over the baseline for this turn (0 before a baseline exists), then
   *  fold the turn in. */
  observe(dbfs: number | null): number {
    if (dbfs === null || !Number.isFinite(dbfs)) return 0;
    const base = this.value;
    const over = base === null ? 0 : dbfs - base;
    this.values.push(dbfs);
    return over;
  }
}

export function selfTurnVectorEvents(
  t: number,
  dbOverBaseline: number,
  tone: { frustration: number | null; defensiveness: number | null },
): VectorEvent[] {
  return [
    { vector: "yelling", level: yellingLevel(dbOverBaseline), t, value: dbOverBaseline },
    {
      vector: "aggressive_tone",
      level: aggressiveToneLevel(tone.frustration, tone.defensiveness),
      t,
    },
  ];
}

/** The phone's single-lane default: both voice vectors on channel A. */
export function phoneNudgePolicy(cooldownS = 20.0): NudgePolicy {
  return new NudgePolicy(
    [
      { vector: "yelling", sensitivity: 1.0, haptics: true, channel: "A" },
      { vector: "aggressive_tone", sensitivity: 1.0, haptics: true, channel: "A" },
    ],
    cooldownS,
    ["A"],
  );
}

/** The haptic seam (expo-haptics in production, a spy in tests). */
export interface HapticSink {
  /** Level 1..3 -> light/medium/heavy; must never throw. */
  nudge(level: number): Promise<void>;
}
