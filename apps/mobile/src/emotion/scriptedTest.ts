/**
 * Scripted emotion test (Settings -> Emotion dev tester): the app asks for each
 * emotion in turn, captures one short take, runs both models on exactly that
 * clip and records what was ASKED next to what each model GUESSED. Pure logic
 * only here (plan + scoring) so it is unit-tested without a microphone.
 */
import { EMOTION_LABELS, type EmotionLabel } from "./config";

export const SCRIPT_SENTENCE = "The meeting is at three";
export const TAKE_SECONDS = 4;

export interface ScriptedTake {
  expect: EmotionLabel;
  /** 1-based round (a round = each emotion once). */
  round: number;
}

export function planTakes(rounds: number): ScriptedTake[] {
  const out: ScriptedTake[] = [];
  for (let r = 1; r <= rounds; r++) for (const expect of EMOTION_LABELS) out.push({ expect, round: r });
  return out;
}

export interface ScoredTake {
  expect: EmotionLabel;
  tiny: readonly number[];
  small: readonly number[] | null;
}

const top = (p: readonly number[]): EmotionLabel => EMOTION_LABELS[p.indexOf(Math.max(...p))];

export interface EmotionRow {
  emotion: EmotionLabel;
  takes: number;
  tinyRight: number;
  smallRight: number;
  /** Most common wrong answer, or null when always right / no takes. */
  tinyUsually: EmotionLabel | null;
  smallUsually: EmotionLabel | null;
}

function usually(guesses: EmotionLabel[], expect: EmotionLabel): EmotionLabel | null {
  const wrong = guesses.filter((g) => g !== expect);
  if (!wrong.length) return null;
  const counts = new Map<EmotionLabel, number>();
  for (const g of wrong) counts.set(g, (counts.get(g) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
}

export function summarize(takes: ScoredTake[]): { rows: EmotionRow[]; tinyRight: number; smallRight: number; total: number } {
  const rows: EmotionRow[] = EMOTION_LABELS.map((emotion) => {
    const mine = takes.filter((t) => t.expect === emotion);
    const tg = mine.map((t) => top(t.tiny));
    const sg = mine.filter((t) => t.small).map((t) => top(t.small!));
    return {
      emotion,
      takes: mine.length,
      tinyRight: tg.filter((g) => g === emotion).length,
      smallRight: sg.filter((g) => g === emotion).length,
      tinyUsually: usually(tg, emotion),
      smallUsually: usually(sg, emotion),
    };
  });
  return {
    rows,
    tinyRight: rows.reduce((a, r) => a + r.tinyRight, 0),
    smallRight: rows.reduce((a, r) => a + r.smallRight, 0),
    total: takes.length,
  };
}

/** True when a take is too quiet to mean anything (so the card asks again). */
export function tooQuiet(pcm: Int16Array, minRms = 0.01): boolean {
  if (pcm.length === 0) return true;
  let sum = 0;
  for (let i = 0; i < pcm.length; i++) {
    const v = pcm[i] / 32768;
    sum += v * v;
  }
  return Math.sqrt(sum / pcm.length) < minRms;
}
