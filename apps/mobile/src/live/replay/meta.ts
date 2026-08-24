/**
 * Scene meta -> a normalized replay script.
 *
 * Three meta shapes exist under server/tests/fixtures/audio and all three
 * (plus a hand-written meta for a phone capture) normalize to the same
 * `ReplayScript`:
 *
 * 1. Scene pack (`*_scene_*_meta.json`): `turns[]` with `duration_sec`,
 *    concatenated with a fixed `silence_gap_sec` — turn times follow exactly
 *    from the durations (server/tests/test_diarize_scenes.py::_build_turns).
 * 2. Real recordings with pipeline turns (`family_real`): `turns[]` with
 *    explicit `start_time` / `end_time` (seconds) and `text`.
 * 3. Real recordings with approximate turns (`poker6_real`): `approx_turns[]`
 *    with `approx_start` / `approx_end` and NO text — boundaries are +-1-2 s
 *    by the owner's own account, so `approxBoundaries` is set and boundary
 *    error is reported but never pinned.
 *
 * A Pixel capture replays the same way: write `<name>_meta.json` next to the
 * 16 kHz mono WAV with `turns: [{speaker, text, start_time, end_time}]`,
 * `self_speaker`, and (optionally) `expected_nudges` / `emotion_coarse`.
 */

export type EmotionCoarse = "neutral" | "angry" | "sad" | "happy";
export type NudgeLevelName = "mild" | "strong";

export interface ScriptTurn {
  index: number;
  speaker: string;
  text: string;
  /** Seconds into the WAV. */
  start: number;
  end: number;
  emotionCoarse: EmotionCoarse | null;
  scriptedEmotion: string | null;
}

export interface ExpectedNudge {
  afterTurnIndex: number;
  level: NudgeLevelName;
  reason: string | null;
}

export interface ReplayScript {
  name: string;
  turns: ScriptTurn[];
  /** Every speaker label, in first-appearance order. */
  speakers: string[];
  selfSpeaker: string | null;
  /** Speaker label -> TTS voice id (scene pack) or null when unknown. Two
   *  scenes sharing a voice id for a speaker are the cross-scene enrollment
   *  pairs. */
  voices: Record<string, string | null>;
  expectedNudges: ExpectedNudge[];
  approxBoundaries: boolean;
  /** How far a loop turn may sit from a script boundary and still count as
   *  "at" it (seconds); wide for approximate metas. */
  boundarySlackSec: number;
  hasText: boolean;
  sampleRate: number | null;
}

interface RawTurn {
  speaker?: unknown;
  text?: unknown;
  duration_sec?: unknown;
  start_time?: unknown;
  end_time?: unknown;
  approx_start?: unknown;
  approx_end?: unknown;
  emotion_coarse?: unknown;
  scripted_emotion?: unknown;
}

const COARSE = new Set<string>(["neutral", "angry", "sad", "happy"]);

function num(v: unknown, what: string): number {
  if (typeof v !== "number" || !Number.isFinite(v)) throw new Error(`meta: ${what} must be a number`);
  return v;
}

function str(v: unknown, what: string): string {
  if (typeof v !== "string") throw new Error(`meta: ${what} must be a string`);
  return v;
}

export interface ParseMetaOptions {
  name?: string;
  /** Overrides / supplies the self speaker when the meta has none. */
  selfSpeaker?: string | null;
}

export function parseSceneMeta(raw: unknown, opts: ParseMetaOptions = {}): ReplayScript {
  if (!raw || typeof raw !== "object") throw new Error("meta: not an object");
  const m = raw as Record<string, unknown>;
  const name = opts.name ?? (typeof m.scene === "string" ? m.scene : "scene");

  let rawTurns: RawTurn[];
  let approx = false;
  if (Array.isArray(m.turns) && m.turns.length > 0) {
    rawTurns = m.turns as RawTurn[];
  } else if (Array.isArray(m.approx_turns) && m.approx_turns.length > 0) {
    rawTurns = m.approx_turns as RawTurn[];
    approx = true;
  } else {
    throw new Error("meta: needs a non-empty `turns` or `approx_turns` array");
  }

  const gap = typeof m.silence_gap_sec === "number" ? m.silence_gap_sec : null;
  const turns: ScriptTurn[] = [];
  let cursor = 0;
  rawTurns.forEach((t, index) => {
    const speaker = str(t.speaker, `turns[${index}].speaker`);
    const text = typeof t.text === "string" ? t.text : "";
    let start: number;
    let end: number;
    if (t.duration_sec !== undefined) {
      if (gap === null) throw new Error("meta: `duration_sec` turns need `silence_gap_sec`");
      const d = num(t.duration_sec, `turns[${index}].duration_sec`);
      start = round4(cursor);
      end = round4(cursor + d);
      cursor += d + gap;
    } else if (t.start_time !== undefined) {
      start = num(t.start_time, `turns[${index}].start_time`);
      end = num(t.end_time, `turns[${index}].end_time`);
    } else if (t.approx_start !== undefined) {
      start = num(t.approx_start, `turns[${index}].approx_start`);
      end = num(t.approx_end, `turns[${index}].approx_end`);
    } else {
      throw new Error(`meta: turns[${index}] has no timing (duration_sec | start_time | approx_start)`);
    }
    if (end < start) throw new Error(`meta: turns[${index}] ends before it starts`);
    const coarse = typeof t.emotion_coarse === "string" && COARSE.has(t.emotion_coarse) ? (t.emotion_coarse as EmotionCoarse) : null;
    turns.push({
      index,
      speaker,
      text,
      start,
      end,
      emotionCoarse: coarse,
      scriptedEmotion: typeof t.scripted_emotion === "string" ? t.scripted_emotion : null,
    });
  });

  const speakers: string[] = [];
  for (const t of turns) if (!speakers.includes(t.speaker)) speakers.push(t.speaker);
  const speakerMap = (m.speakers && typeof m.speakers === "object" ? m.speakers : {}) as Record<
    string,
    { voice?: unknown; is_self?: unknown }
  >;
  for (const label of Object.keys(speakerMap)) if (!speakers.includes(label)) speakers.push(label);

  const voices: Record<string, string | null> = {};
  for (const s of speakers) {
    const v = speakerMap[s]?.voice;
    voices[s] = typeof v === "string" ? v : null;
  }

  let self: string | null = null;
  if (opts.selfSpeaker !== undefined) self = opts.selfSpeaker;
  else if (typeof m.self_speaker === "string") self = m.self_speaker;
  else if (typeof m.owner_is_speaker === "string") self = m.owner_is_speaker;
  else {
    const flagged = Object.entries(speakerMap).find(([, v]) => v?.is_self === true);
    self = flagged ? flagged[0] : null;
  }
  if (self !== null && !speakers.includes(self)) {
    throw new Error(`meta: self speaker "${self}" is not one of ${speakers.join(", ")}`);
  }

  const expectedNudges: ExpectedNudge[] = [];
  if (Array.isArray(m.expected_nudges)) {
    for (const n of m.expected_nudges as Record<string, unknown>[]) {
      const idx = num(n.after_turn_index, "expected_nudges[].after_turn_index");
      const level = str(n.level, "expected_nudges[].level");
      if (level !== "mild" && level !== "strong") throw new Error(`meta: nudge level "${level}"`);
      if (idx < 0 || idx >= turns.length) throw new Error(`meta: expected nudge after turn ${idx} out of range`);
      expectedNudges.push({ afterTurnIndex: idx, level, reason: typeof n.reason === "string" ? n.reason : null });
    }
  }

  return {
    name,
    turns,
    speakers,
    selfSpeaker: self,
    voices,
    expectedNudges,
    approxBoundaries: approx,
    boundarySlackSec: approx ? 2.0 : 0.5,
    hasText: turns.some((t) => t.text.trim().length > 0),
    sampleRate: typeof m.sample_rate === "number" ? m.sample_rate : null,
  };
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

/** Total scripted seconds for one speaker (enrollment budget planning). */
export function speakerSeconds(script: ReplayScript, speaker: string): number {
  return script.turns.filter((t) => t.speaker === speaker).reduce((a, t) => a + (t.end - t.start), 0);
}
