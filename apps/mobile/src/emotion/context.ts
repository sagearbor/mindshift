/**
 * A cheap snapshot of what the Live Coach session already knows on the phone
 * at the moment an emotion record is saved: session mode, who has been
 * talking (wearer vs others), pace of the last turn, whether the turn
 * overlapped someone, and the last coaching line. Computed from the on-device
 * fast loop's finished turns only (live/fastLoop.ts `turnsSoFar`); nothing is
 * fabricated, and fields the loop cannot know stay null. Pure and unit-tested.
 */
import type { LocalTurn } from "../live/fastLoop";

export interface RecordContext {
  /** Live Coach mode ("earpiece", "room", ...), when known. */
  mode: string | null;
  /** Seconds of audio since the session's loop came up. */
  sessionS: number | null;
  /** Finished turns so far. */
  turns: number;
  /** Talk time over the last `windowS` seconds, split wearer vs others; selfShare null when nobody was attributed. */
  talk: { windowS: number; selfS: number; otherS: number; unknownS: number; selfShare: number | null };
  /** The most recent finished turn. */
  lastTurn: {
    speaker: string;
    isSelf: boolean | null;
    durationS: number;
    words: number;
    paceWpm: number | null;
    overlapped: boolean;
    endedAgoS: number;
  } | null;
  /** The last Live Coach suggestion in the window (the app's own coaching, not the emotion buzz). */
  coach: { text: string; kind: string | null; spoken: boolean } | null;
}

export const CONTEXT_WINDOW_S = 120;

const wordCount = (s: string) => (s.trim() ? s.trim().split(/\s+/).length : 0);
const r1 = (n: number) => Math.round(n * 10) / 10;

export function buildRecordContext(
  turns: readonly LocalTurn[],
  audioClockS: number,
  mode: string | null,
  windowS: number = CONTEXT_WINDOW_S,
): RecordContext {
  const from = audioClockS - windowS;
  let selfS = 0;
  let otherS = 0;
  let unknownS = 0;
  for (const t of turns) {
    const s = Math.max(t.startTime, from);
    const e = Math.min(t.endTime, audioClockS);
    if (e <= s) continue;
    if (t.isSelf === true) selfS += e - s;
    else if (t.isSelf === false) otherS += e - s;
    else unknownS += e - s;
  }
  const attributed = selfS + otherS;
  const last = turns.length ? turns[turns.length - 1] : null;
  let lastTurn: RecordContext["lastTurn"] = null;
  if (last) {
    const dur = Math.max(0, last.endTime - last.startTime);
    const words = wordCount(last.text ?? "");
    lastTurn = {
      speaker: last.displayName ?? last.speaker,
      isSelf: last.isSelf,
      durationS: r1(dur),
      words,
      paceWpm: dur >= 2 && words >= 3 ? Math.round((words / dur) * 60) : null,
      overlapped: !!last.overlap,
      endedAgoS: r1(Math.max(0, audioClockS - last.endTime)),
    };
  }
  let coach: RecordContext["coach"] = null;
  for (let i = turns.length - 1; i >= 0; i--) {
    const t = turns[i];
    if (t.endTime < from) break;
    if (t.suggestion) {
      coach = { text: t.suggestion, kind: t.suggestionKind ?? null, spoken: t.spoken };
      break;
    }
  }
  return {
    mode,
    sessionS: r1(audioClockS),
    turns: turns.length,
    talk: { windowS, selfS: r1(selfS), otherS: r1(otherS), unknownS: r1(unknownS), selfShare: attributed > 0 ? Math.round((selfS / attributed) * 100) / 100 : null },
    lastTurn,
    coach,
  };
}
