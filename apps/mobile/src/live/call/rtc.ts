/**
 * The slice of WebRTC the call state machine drives, as a structural
 * interface so the same `CallSession` runs over react-native-webrtc on the
 * phone (rtcNative.ts), the browser's own RTCPeerConnection in Safari
 * (callWeb.ts) and a fake in tests (__tests__/callSession.test.ts).
 *
 * `playRemote` / `stopRemote` are keyed by peer uid: in a mesh a client
 * plays one remote stream per other participant (the web adapter keeps one
 * <audio> element per peer; native plays each track itself).
 */
import type { IceCandidateInit, IceServer, SdpInit } from "./types";

export interface RtcTrackLike {
  enabled: boolean;
  kind?: string;
  stop(): void;
}

export interface RtcStreamLike {
  getTracks(): RtcTrackLike[];
  getAudioTracks(): RtcTrackLike[];
}

export interface RtcTrackEventLike {
  track: RtcTrackLike;
  streams: RtcStreamLike[];
}

export interface RtcPeerLike {
  iceConnectionState: string;
  connectionState?: string;
  onicecandidate: ((e: { candidate: IceCandidateInit | null }) => void) | null;
  oniceconnectionstatechange: (() => void) | null;
  ontrack: ((e: RtcTrackEventLike) => void) | null;
  createOffer(options?: { iceRestart?: boolean }): Promise<SdpInit>;
  createAnswer(): Promise<SdpInit>;
  setLocalDescription(description?: SdpInit): Promise<void>;
  setRemoteDescription(description: SdpInit): Promise<void>;
  addIceCandidate(candidate: IceCandidateInit): Promise<void>;
  addTrack(track: RtcTrackLike, stream: RtcStreamLike): unknown;
  close(): void;
}

export type AudioRoute = "speaker" | "earpiece";

export interface RtcAdapter {
  /** A fresh peer connection with the server's ICE servers. */
  createPeer(config: { iceServers: IceServer[] }): RtcPeerLike;
  /** The local microphone (16 kHz mono, echo cancellation ON). On the web
   *  build this is the SAME MediaStream the fast loop already captures. */
  getLocalStream(): Promise<RtcStreamLike>;
  /** Start (or replace) playback of one peer's audio. `peerUid` keys the
   *  playback so a mesh mixes every remote person. Web: an <audio autoplay
   *  playsinline> element per peer; native: a no-op (rn-webrtc plays it). */
  playRemote(peerUid: string, stream: RtcStreamLike | null, track: RtcTrackLike): void;
  /** Stop one peer's playback (uid) or, with no uid, all of them. */
  stopRemote(peerUid?: string): void;
  /** Web only: create + kick the remote <audio> element(s) inside the
   *  user's Answer/Start tap so Safari's autoplay policy lets them play. */
  prime?(): void;
  /** Native only: route the call's audio out of the loudspeaker or the
   *  earpiece. Absent where the platform decides (browser). */
  setRoute?(route: AudioRoute): Promise<void>;
}

/** Serialize a candidate for the wire (RN's RTCIceCandidate has toJSON). */
export function candidateToJson(c: unknown): IceCandidateInit | null {
  if (!c || typeof c !== "object") return null;
  const obj = c as { toJSON?: () => IceCandidateInit } & IceCandidateInit;
  const j = typeof obj.toJSON === "function" ? obj.toJSON() : obj;
  return {
    candidate: j.candidate ?? "",
    sdpMid: j.sdpMid ?? null,
    sdpMLineIndex: j.sdpMLineIndex ?? null,
    ...(j.usernameFragment !== undefined ? { usernameFragment: j.usernameFragment } : {}),
  };
}

/** Serialize a session description for the wire (RN's has toJSON too). */
export function sdpToJson(d: unknown): SdpInit | null {
  if (!d || typeof d !== "object") return null;
  const obj = d as { toJSON?: () => SdpInit } & SdpInit;
  const j = typeof obj.toJSON === "function" ? obj.toJSON() : obj;
  if (!j.type) return null;
  return { type: j.type, sdp: j.sdp ?? "" };
}
