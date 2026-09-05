/**
 * The end-of-session summary card's numbers, computed once from what the
 * session actually produced — the transcript that was on screen, the fast
 * loop's per-turn latency log (`FastLoop.latencyLog`), and the nudges the
 * policy raised on the user's own delivery. Pure so the card's arithmetic
 * is unit-tested without a session.
 *
 * Honesty rules: a number that was never measured is `null`, never 0 —
 * "first words" latency only exists when the phone spoke (the legacy
 * server path and therapist mode never do), and turn counts come from the
 * finalized transcript, not from what the server may still be working on.
 */
import type { TurnLatency } from "./fastLoop";
import { computeConversationDynamics, type ConversationDynamics } from "./conversationDynamics";

export interface SessionSummaryInput {
  /** Wall-clock ISO timestamps; `endedAt` defaults to now. */
  startedAt: string | null;
  endedAt?: string | null;
  /** `kind: "backchannel"` lines (listener noises — live/naturalTurn.ts)
   *  show in the transcript but never count as turns here.
   *  `isSelf`/`startTime`/`endTime` feed the dev-mode conversation-dynamics
   *  block (see `dynamics` below) when present; absent on lines with no
   *  known timing (e.g. the legacy suggestion-event fallback path). */
  transcript: {
    speaker: string;
    text: string;
    kind?: string;
    isSelf?: boolean | null;
    startTime?: number;
    endTime?: number;
  }[];
  latencyLog: TurnLatency[];
  /** Nudges (level ≥ 1) raised on the user's own turns this session. */
  escalations: number;
}

export interface SessionSummary {
  durationMs: number | null;
  turnsBySpeaker: { speaker: string; turns: number }[];
  totalTurns: number;
  escalations: number;
  /** Median segment-end → first spoken word, ms; null when nothing was spoken. */
  firstWordsMedianMs: number | null;
  /** The best (fastest) spoken turn, ms; null when nothing was spoken. */
  firstWordsBestMs: number | null;
  /** How many suggestions were actually voiced. */
  spokenTurns: number;
  /** Which provider answered most often ("os", "bundled", "cloud", …). */
  topProvider: string | null;
  /** Response-gap/overlap metrics (Workstream 2, dev-mode only surface —
   *  see conversationDynamics.ts). Null when the transcript carried no
   *  turn with both startTime and endTime (e.g. the legacy path). */
  dynamics: ConversationDynamics | null;
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[sorted.length >> 1];
}

function parseMs(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : null;
}

export function summarizeSession(input: SessionSummaryInput): SessionSummary {
  const start = parseMs(input.startedAt);
  const end = parseMs(input.endedAt) ?? Date.now();
  const durationMs = start !== null && end >= start ? end - start : null;

  const counts = new Map<string, number>();
  for (const t of input.transcript) {
    if (!t.text || t.kind === "backchannel") continue;
    counts.set(t.speaker, (counts.get(t.speaker) ?? 0) + 1);
  }
  const turnsBySpeaker = [...counts.entries()]
    .map(([speaker, turns]) => ({ speaker, turns }))
    .sort((a, b) => b.turns - a.turns || a.speaker.localeCompare(b.speaker));

  const spoken = input.latencyLog
    .map((l) => l.toSpeakMs)
    .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  const providers = new Map<string, number>();
  for (const l of input.latencyLog) {
    if (!l.provider) continue;
    providers.set(l.provider, (providers.get(l.provider) ?? 0) + 1);
  }
  let topProvider: string | null = null;
  let best = -1;
  for (const [name, n] of providers) {
    if (n > best) {
      best = n;
      topProvider = name;
    }
  }

  const dynamicsTurns = input.transcript
    .filter(
      (t): t is typeof t & { startTime: number; endTime: number } =>
        typeof t.startTime === "number" &&
        Number.isFinite(t.startTime) &&
        typeof t.endTime === "number" &&
        Number.isFinite(t.endTime),
    )
    .map((t) => ({
      speaker: t.speaker,
      isSelf: t.isSelf ?? null,
      startTime: t.startTime,
      endTime: t.endTime,
      kind: t.kind === "backchannel" ? ("backchannel" as const) : ("primary" as const),
    }));
  const dynamics = dynamicsTurns.length > 0 ? computeConversationDynamics(dynamicsTurns) : null;

  return {
    durationMs,
    turnsBySpeaker,
    totalTurns: turnsBySpeaker.reduce((n, s) => n + s.turns, 0),
    escalations: Math.max(0, input.escalations | 0),
    firstWordsMedianMs: median(spoken),
    firstWordsBestMs: spoken.length ? Math.min(...spoken) : null,
    spokenTurns: spoken.length,
    topProvider,
    dynamics,
  };
}

/** "2m 14s" / "48s" for the card; "—" when unknown. */
export function formatDuration(ms: number | null): string {
  if (ms === null) return "—";
  const total = Math.round(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return m > 0 ? `${m}m ${s.toString().padStart(2, "0")}s` : `${s}s`;
}

/** "640 ms" / "1.3 s" for latencies; "—" when nothing was spoken. */
export function formatLatency(ms: number | null): string {
  if (ms === null) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}
