/**
 * Live Coach ROOM mode — the phone (or a laptop sharing its screen) is an
 * assistant for a whole meeting room, with the user's knowledge library.
 *
 * Deliberately unlike earpiece mode:
 * - It speaks ALOUD on purpose, but only when ADDRESSED ("MindShift, …"):
 *   the server detects the wake phrase and sends a `room_answer` frame, which
 *   the phone voices on the loudspeaker.
 * - It shows library facts on screen unprompted (`room_card` frames) — never
 *   spoken.
 * - No personal coaching: no nudges, no haptics, no wearer identity, no
 *   on-device coaching loop.
 *
 * This module is the React-free part: the frame parser, the view-state
 * reducer, and the mode-keyed speech/route policy the hook consults.
 * Server side: server/room_mode.py.
 */
import type { LiveMode } from "./localLlm";

/** What the room hears as "talking to me" (server: room_mode.WAKE_PHRASE). */
export const ROOM_WAKE_PHRASE = "MindShift";

/** Spoken once when a room session starts, so nobody in the room is
 *  recorded by surprise. The banner says the same thing for as long as the
 *  mode is on. */
export const ROOM_CONSENT_LINE =
  "MindShift is listening to this room. Say MindShift, then your question.";

/** Newest-first caps for the screen: the latest few cards stay readable. */
export const MAX_ROOM_CARDS = 4;
export const MAX_ROOM_ANSWERS = 3;

export interface RoomCard {
  id: string;
  title: string;
  fact: string;
  sourceItemId: string;
  sourceTitle: string;
  /** Session seconds of the utterance that mentioned it. */
  t: number;
}

export interface RoomAnswer {
  question: string;
  text: string;
  /** False when the answer admits the library/conversation doesn't have it. */
  known: boolean;
  sourceItemIds: string[];
  t: number;
}

export interface RoomViewState {
  cards: RoomCard[];
  answers: RoomAnswer[];
  /** Wake phrase heard alone — the next utterance is the question. */
  listening: boolean;
  /** The last answer that could not be produced (reason), until the next one. */
  answerError: string | null;
  /** The server acknowledged `mode: "room"` on this session. */
  serverConfirmed: boolean;
}

export const IDLE_ROOM_STATE: RoomViewState = {
  cards: [],
  answers: [],
  listening: false,
  answerError: null,
  serverConfirmed: false,
};

export type RoomFrame =
  | { kind: "card"; card: RoomCard }
  | { kind: "answer"; answer: RoomAnswer }
  | { kind: "listening"; t: number }
  | { kind: "answer_error"; question: string; reason: string; t: number };

const str = (v: unknown): string => (typeof v === "string" ? v : "");
const num = (v: unknown): number => (typeof v === "number" && Number.isFinite(v) ? v : 0);

/** A server frame as a room event, or null (not a room frame / malformed). */
export function parseRoomFrame(data: unknown): RoomFrame | null {
  if (!data || typeof data !== "object") return null;
  const d = data as Record<string, unknown>;
  switch (d.type) {
    case "room_card": {
      const fact = str(d.fact).trim();
      const id = str(d.id);
      if (!fact || !id) return null;
      return {
        kind: "card",
        card: {
          id,
          title: str(d.title),
          fact,
          sourceItemId: str(d.source_item_id),
          sourceTitle: str(d.source_title),
          t: num(d.t),
        },
      };
    }
    case "room_answer": {
      const text = str(d.text).trim();
      if (!text) return null;
      return {
        kind: "answer",
        answer: {
          question: str(d.question),
          text,
          known: d.known !== false,
          sourceItemIds: Array.isArray(d.source_item_ids)
            ? d.source_item_ids.filter((x): x is string => typeof x === "string")
            : [],
          t: num(d.t),
        },
      };
    }
    case "room_listening":
      return { kind: "listening", t: num(d.t) };
    case "room_answer_error":
      return { kind: "answer_error", question: str(d.question), reason: str(d.reason) || "unavailable", t: num(d.t) };
    default:
      return null;
  }
}

/** Fold one room event into the view state (pure). Cards dedupe by id and
 *  keep the newest MAX_ROOM_CARDS; answers keep the newest MAX_ROOM_ANSWERS. */
export function applyRoomFrame(state: RoomViewState, frame: RoomFrame): RoomViewState {
  switch (frame.kind) {
    case "card": {
      if (state.cards.some((c) => c.id === frame.card.id)) return state;
      return { ...state, cards: [frame.card, ...state.cards].slice(0, MAX_ROOM_CARDS) };
    }
    case "answer":
      return {
        ...state,
        listening: false,
        answerError: null,
        answers: [frame.answer, ...state.answers].slice(0, MAX_ROOM_ANSWERS),
      };
    case "listening":
      return { ...state, listening: true };
    case "answer_error":
      return { ...state, listening: false, answerError: frame.reason };
  }
}

/**
 * Which output routes may carry speech in a mode — the decision the hook
 * makes before EVERY utterance, keyed on the session mode, before it asks
 * live/audioRoute.ts anything:
 * - earpiece: a private route (headset) or nothing — never the loudspeaker;
 * - room: the loudspeaker is the POINT — answers are for the whole room, so
 *   the route is never consulted;
 * - every other mode: whatever route is active (unchanged behaviour; therapist
 *   and journal are silenced earlier, by their own rules).
 */
export type SpeechRouteRule = "private-only" | "loudspeaker-ok" | "any";

export function speechRouteRule(mode: LiveMode): SpeechRouteRule {
  if (mode === "earpiece") return "private-only";
  if (mode === "room") return "loudspeaker-ok";
  return "any";
}

/** Does this mode bring up the on-device coaching loop (when the device can)?
 *  Journal never coaches; Room serves the room — no personal coaching, so no
 *  nudges, haptics or on-device suggestions. */
export function modeRunsCoachingLoop(mode: LiveMode): boolean {
  return mode !== "journal" && mode !== "room";
}
