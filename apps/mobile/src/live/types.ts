/**
 * Wire shapes shared with the server (server/models/audio.py, Foundation A).
 * Field names are snake_case on purpose — these are serialized as-is.
 */
import type { TurnProsody } from "./prosody";
import type { TextTone } from "./localLlm";

/** Client -> server: a turn the phone finalized itself (`TurnLocalEvent`). */
export interface TurnLocalEvent {
  type: "turn_local";
  session_id: string;
  speaker: string;
  speaker_person_id: string | null;
  speaker_match_score: number | null;
  is_self: boolean | null;
  text: string;
  start_time: number;
  end_time: number;
  transcript_source: "on-device";
  prosody: TurnProsody | null;
  text_tone: TextTone | null;
  suggestion: string | null;
  suggestion_source: "on-device" | null;
  tts_source: "on-device";
}

/** Server -> client `tone_flag` (ToneFlagEvent). */
export interface ToneFlagEvent {
  type: "tone_flag";
  session_id: string;
  speaker: string;
  start_time: number;
  end_time: number;
  source: "text" | "audio";
  scores: Record<string, number>;
  label: string;
  confidence: number;
}

/** Server -> client `speaker_identity` (SpeakerIdentityEvent). */
export interface SpeakerIdentityEvent {
  type: "speaker_identity";
  session_id: string;
  speaker: string;
  person_id: string | null;
  display_name: string | null;
  is_self: boolean;
  score: number;
}
