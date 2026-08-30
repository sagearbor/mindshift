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
  /** WHY the person was matched: "absolute" (>= 0.65) or "contrast" (the
   *  in-session cross-recording rule, speakerId.ts CROSS_MATCH_*); null
   *  when no voiceprint matched (a mid-call binding has no basis). */
  speaker_match_basis: "absolute" | "contrast" | null;
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

/**
 * Client -> server `speaker_label`: the user named a speaker mid-call
 * ("Speaker B is Mom"). The server applies it to the RUNNING session (the
 * cloud coach's prompts name the person; `is_self` fixes side-aware
 * coaching) and answers `speaker_label_ack`. The finished session carries
 * the same map in `POST /sessions/live` `speaker_labels` so the stored
 * episode (and the therapist's view of it) shows the name.
 */
export interface SpeakerLabelMessage {
  type: "speaker_label";
  session_id: string;
  /** The raw wire label being named ("Speaker B"). */
  speaker: string;
  person_id: string | null;
  display_name: string;
  is_self: boolean;
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
