/**
 * In-app calls — the client half of the wire vocabulary shared with the
 * server (docs/plans/2026-08-25-in-app-calls.md). Field names on the wire
 * are snake_case on purpose; the view types the UI reads are camelCase.
 *
 * A call has up to `max_participants` (default 3) people with a ROLE:
 *   - "participant" — on the phone app, coached (Sage, his dad)
 *   - "therapist"   — an observer (Safari): sees the merged transcript,
 *                     both participants' suggestions/nudges read-only, and
 *                     the scoreboard; their own speech is transcribed and
 *                     merged but never coached, and nothing is spoken to
 *                     them (no TTS).
 *
 * Audio is a FULL MESH: each client holds one RTCPeerConnection per OTHER
 * participant, so everyone hears everyone (CallSession keeps a peer map
 * keyed by uid). Signaling is addressed: `rtc_signal.to` is REQUIRED.
 *
 * REST (src/live/call/callApi.ts):
 *   POST /calls {max_participants?} -> {call_id, join_code, join_url}
 *   POST /calls/{id}/join {role?}   -> {call_id, join_code, join_url}
 *   GET  /calls/{id}
 *   POST /calls/{id}/end
 *
 * Over the existing session WebSocket (/ws/session/{session_id}):
 *   client -> server  call_join   {call_id, role?}
 *   client -> server  rtc_signal  {call_id, to, payload:{sdp|candidate}}
 *   server -> client  call_state  {call_id, participants:[{uid, display_name,
 *                                   role, is_self, connected}], ice_servers}
 *   server -> client  rtc_signal  {from, payload}
 *   server -> client  call_ended  {call_id, reason?}
 *   ... plus every participant's turns as ordinary `transcript` events, and
 *   suggestion/nudge/tone events carrying `for_uid` when they concern
 *   another participant (the therapist renders those read-only).
 */

export type CallRole = "participant" | "therapist";

export interface CallCreated {
  callId: string;
  joinCode: string;
  joinUrl: string;
}

export interface CallParticipant {
  uid: string;
  displayName: string;
  role: CallRole;
  isSelf: boolean;
  connected: boolean;
}

/** An ICE server as the server hands it over (the W3C RTCIceServer shape). */
export interface IceServer {
  urls: string | string[];
  username?: string;
  credential?: string;
}

export interface SdpInit {
  type: "offer" | "answer" | "pranswer" | "rollback";
  sdp?: string;
}

export interface IceCandidateInit {
  candidate?: string;
  sdpMid?: string | null;
  sdpMLineIndex?: number | null;
  usernameFragment?: string | null;
}

export interface RtcSignalPayload {
  sdp?: SdpInit;
  candidate?: IceCandidateInit | null;
}

// --- server -> client -------------------------------------------------------

export interface CallStateMessage {
  type: "call_state";
  call_id: string;
  participants: {
    uid: string;
    display_name?: string | null;
    role?: CallRole | null;
    is_self?: boolean;
    connected?: boolean;
  }[];
  ice_servers?: IceServer[] | null;
}

export interface RtcSignalMessage {
  type: "rtc_signal";
  call_id?: string;
  from?: string;
  payload: RtcSignalPayload;
}

export interface CallEndedMessage {
  type: "call_ended";
  call_id?: string;
  reason?: string | null;
}

export type CallServerMessage = CallStateMessage | RtcSignalMessage | CallEndedMessage;

// --- client -> server -------------------------------------------------------

export interface CallJoinMessage {
  type: "call_join";
  call_id: string;
  role?: CallRole;
}

export interface RtcSignalOut {
  type: "rtc_signal";
  call_id: string;
  /** REQUIRED in the mesh: the peer this frame is for. */
  to: string;
  payload: RtcSignalPayload;
}

export type CallClientMessage = CallJoinMessage | RtcSignalOut;

// --- what the UI sees ---------------------------------------------------------

/**
 * The lifecycle of the call as a whole:
 * idle        no call
 * creating    the REST create/join is in flight
 * waiting     the call exists; nobody else has joined yet (share the code)
 * connecting  another participant is present; a mesh link is negotiating
 * connected   at least one peer's audio is flowing
 * reconnecting a peer's ICE dropped; a restart is in progress
 * ended       hung up / the server ended it
 * failed      could not create/join (see `error`)
 */
export type CallStatus =
  | "idle"
  | "creating"
  | "waiting"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "ended"
  | "failed";

/** One other person in the call, from this client's point of view. */
export interface CallPeer {
  uid: string;
  displayName: string;
  role: CallRole;
  /** Whether this specific mesh link is carrying audio. */
  connected: boolean;
  /** ICE restarts this link has needed (diagnostics). */
  iceRestarts: number;
}

export interface CallView {
  status: CallStatus;
  callId: string | null;
  joinCode: string | null;
  joinUrl: string | null;
  /** This client's role in the call. */
  selfRole: CallRole;
  /** Every OTHER participant, newest-joined last. */
  peers: CallPeer[];
  /** Wall-clock ms at which the FIRST peer connected; null before then. */
  connectedAt: number | null;
  /** Mutes this client's OWN microphone track on every mesh link. */
  muted: boolean;
  /** Total ICE restarts across all links (diagnostics). */
  iceRestarts: number;
  error: string | null;
}

export const IDLE_CALL_VIEW: CallView = {
  status: "idle",
  callId: null,
  joinCode: null,
  joinUrl: null,
  selfRole: "participant",
  peers: [],
  connectedAt: null,
  muted: false,
  iceRestarts: 0,
  error: null,
};
