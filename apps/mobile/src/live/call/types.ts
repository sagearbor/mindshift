/**
 * In-app calls — the client half of the wire vocabulary. The server half
 * (server/calls.py, server/routers/calls.py, server/audio_pipeline.py) is the
 * contract; docs/plans/2026-08-25-in-app-calls.md has the exact JSON. Field
 * names on the wire are snake_case on purpose; the view types the UI reads
 * are camelCase.
 *
 * A call has up to `max_participants` (2 or 3, default 3) MEMBERS with a
 * ROLE and a SLOT label:
 *   - "participant" — coached (the host is slot A "Speaker A", the second
 *                     participant slot B "Speaker B"): Sage, his dad
 *   - "therapist"   — the observer (slot C "Speaker C"; Mom in Safari): sees
 *                     the merged transcript, every participant's coaching
 *                     read-only (tagged `for_uid`), and the scoreboard; her
 *                     own speech is transcribed + merged but never coached
 *                     and nothing is spoken to her.
 *
 * Audio is a FULL MESH: each client holds one RTCPeerConnection per OTHER
 * member (CallSession keeps a peer map keyed by uid). Signaling is
 * addressed: `rtc_signal.to` is REQUIRED once the call has three members
 * (this client always sends it). The server relays `payload` VERBATIM, so
 * it is the W3C init dict itself: `{type:"offer"|"answer", sdp}` or
 * `{candidate, sdpMid, sdpMLineIndex}`.
 *
 * REST (src/live/call/callApi.ts):
 *   POST /calls        {display_name?, max_participants?}       -> CallOut (201)
 *   POST /calls/join   {join_code, display_name?, role?}        -> CallOut
 *   GET  /calls/{id}                                            -> CallOut
 *   POST /calls/{id}/end                                        -> CallOut
 *
 * Over the existing session WebSocket (/ws/session/{session_id}):
 *   client -> server  call_join   {call_id, join_code?, display_name?, role?}
 *   client -> server  rtc_signal  {call_id, to, payload}
 *   server -> client  call_state  CallOut minus join_code/join_url/invitee_*
 *   server -> client  rtc_signal  {call_id, from, payload}
 *   server -> client  call_ended  {call_id, reason, ended_by, episode_id,
 *                                  shared_with, episodes, turn_count}
 *   ... plus every other member's turns as `transcript` events (with
 *   `speaker` = their slot label, `display_name` relative to us, `role`,
 *   `participant_uid`, `text_tone`, `prosody`), and — therapist sockets only
 *   — read-only copies of each participant's `suggestion` / `tone_flag` /
 *   `speaker_identity` tagged `for_uid`.
 *
 * A phone must NOT `POST /sessions/live` for a call: the server persists one
 * episode per participant itself and hands the id back in `call_ended`.
 */

export type CallRole = "participant" | "therapist";

/** What the REST create/join hands back that the client keeps. */
export interface CallCreated {
  callId: string;
  joinCode: string;
  joinUrl: string;
  selfLabel: string | null;
  selfRole: CallRole;
  iceServers: IceServer[];
}

export interface CallParticipant {
  uid: string;
  /** The slot label the server relabels their turns with ("Speaker B"). */
  label: string;
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

/** Relayed verbatim: the SDP init or the candidate init itself. */
export type RtcSignalPayload = SdpInit | IceCandidateInit;

export function isSdpPayload(p: RtcSignalPayload): p is SdpInit {
  const t = (p as SdpInit).type;
  return t === "offer" || t === "answer" || t === "pranswer" || t === "rollback";
}

// --- server -> client -------------------------------------------------------

export interface CallStateMessage {
  type: "call_state";
  call_id: string;
  status?: string;
  self_uid?: string;
  self_role?: CallRole | null;
  self_label?: string | null;
  participants: {
    uid: string;
    slot?: string;
    label?: string | null;
    display_name?: string | null;
    role?: CallRole | null;
    is_self?: boolean;
    connected?: boolean;
  }[];
  ice_servers?: IceServer[] | null;
  max_participants?: number;
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
  ended_by?: string | null;
  /** YOUR episode (null for the therapist). */
  episode_id?: string | null;
  shared_with?: string[];
  /** Every participant's episode, by uid (the therapist gets these). */
  episodes?: Record<string, string>;
  turn_count?: number;
}

export type CallServerMessage = CallStateMessage | RtcSignalMessage | CallEndedMessage;

// --- client -> server -------------------------------------------------------

export interface CallJoinMessage {
  type: "call_join";
  call_id: string;
  join_code?: string;
  display_name?: string;
  role?: CallRole;
}

export interface RtcSignalOut {
  type: "rtc_signal";
  call_id: string;
  /** The peer this frame is for. Required in a 3-member mesh; always sent. */
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
 * connecting  another member is present; a mesh link is negotiating
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

/** One other member of the call, from this client's point of view. */
export interface CallPeer {
  uid: string;
  /** Their slot label ("Speaker A") — the `speaker` of their transcript events. */
  label: string;
  /** Their name as the server shows it to US ("Dad", "Mom (therapist)"). */
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
  /** This client's role and slot label in the call. */
  selfRole: CallRole;
  selfLabel: string | null;
  /** Every OTHER member, in the server's order. */
  peers: CallPeer[];
  /** Whether the server handed us a TURN server (without one, two phones on
   *  carrier NAT may never connect — the UI says so). */
  hasTurn: boolean;
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
  selfLabel: null,
  peers: [],
  hasTurn: false,
  connectedAt: null,
  muted: false,
  iceRestarts: 0,
  error: null,
};

/** Whether an ICE server list includes a TURN relay. */
export function hasTurnServer(servers: readonly IceServer[]): boolean {
  return servers.some((s) => {
    const urls = Array.isArray(s.urls) ? s.urls : [s.urls];
    return urls.some((u) => /^turns?:/i.test(u));
  });
}
