/**
 * CallSession — the state machine behind an in-app call, built around a
 * MESH peer map: one RTCPeerConnection per OTHER participant, so a 3-person
 * call (two coached participants + a therapist observer) has every client
 * connected to every other. One instance per call, on each side.
 *
 * It owns the peer links and the single local microphone track (added to
 * every link), speaks the signaling vocabulary (types.ts) over whatever
 * `send` the hook hands it (the existing session WebSocket), and reports a
 * `CallView` for the screen. It never touches React, the network or a
 * platform API directly — those come in through `RtcAdapter` and `send`,
 * so the whole mesh runs in Jest.
 *
 * Per link, roles never enter the media plane — every participant, therapist
 * included, is a full audio peer (everyone hears everyone). Who OFFERS on a
 * link is decided by uid order alone: the lexicographically-LOWER uid makes
 * the offer, the higher one answers. This is symmetric on both ends and
 * glare-free (only one side ever offers a given link) whether a peer was
 * there at join time or showed up later. The offerer also owns that link's
 * ICE restarts.
 *
 * Audio is peer-to-peer: nothing here ever sends PCM to the server. The
 * coaching loop keeps its own capture (useAudioStream) — on the phone a
 * second AudioRecord alongside react-native-webrtc's; on the web the SAME
 * MediaStream.
 *
 * The server relays `rtc_signal.payload` verbatim and buffers nothing: an
 * offer to a member whose socket isn't bound yet is answered with an error
 * frame, so the offerer (re)sends its offer whenever `call_state` shows the
 * peer `connected: true` and the link still has no remote description.
 */
import type { RtcAdapter, RtcPeerLike, RtcStreamLike, RtcTrackLike } from "./rtc";
import { candidateToJson, sdpToJson } from "./rtc";
import {
  hasTurnServer,
  IDLE_CALL_VIEW,
  isSdpPayload,
  type CallClientMessage,
  type CallCreated,
  type CallParticipant,
  type CallPeer,
  type CallRole,
  type CallServerMessage,
  type CallStatus,
  type CallView,
  type IceCandidateInit,
  type IceServer,
  type RtcSignalPayload,
  type SdpInit,
} from "./types";

export interface CallSessionDeps {
  adapter: RtcAdapter;
  /** This client's role (drives call_join and, upstream, whether the loop
   *  coaches). Default "participant". */
  role?: CallRole;
  /** How the others should see this client ("Sage"); rides call_join. */
  displayName?: string;
  /** Send a signaling frame; false when the socket is down (retried by the
   *  next roster update). */
  send: (message: CallClientMessage) => boolean;
  onChange: (view: CallView) => void;
  now?: () => number;
  setTimeout?: (fn: () => void, ms: number) => unknown;
  clearTimeout?: (handle: unknown) => void;
  iceRestartDelayMs?: number;
  maxIceRestarts?: number;
}

const DEFAULT_ICE_RESTART_DELAY_MS = 4000;
const DEFAULT_MAX_ICE_RESTARTS = 4;

/** Which of two uids offers on their shared link (stable, symmetric). */
export function isOfferer(selfUid: string, peerUid: string): boolean {
  return selfUid < peerUid;
}

/** One mesh link: this client's connection to a single other participant. */
interface PeerLink {
  participant: CallParticipant;
  pc: RtcPeerLike | null;
  remoteDescriptionSet: boolean;
  pendingCandidates: IceCandidateInit[];
  negotiating: boolean;
  restartTimer: unknown;
  iceRestarts: number;
  iceState: string;
}

export class CallSession {
  private view: CallView;
  private local: RtcStreamLike | null = null;
  private localPromise: Promise<RtcStreamLike> | null = null;
  private iceServers: IceServer[] = [];
  private selfUid: string | null = null;
  private selfLabel: string | null = null;
  private joinCode: string | null = null;
  /** uid -> mesh link. */
  private links = new Map<string, PeerLink>();
  private connectedAt: number | null = null;
  private muted = false;
  private closed = false;
  private readonly role: CallRole;
  private readonly now: () => number;
  private readonly setTimeoutImpl: (fn: () => void, ms: number) => unknown;
  private readonly clearTimeoutImpl: (handle: unknown) => void;
  private readonly iceRestartDelayMs: number;
  private readonly maxIceRestarts: number;

  constructor(private readonly deps: CallSessionDeps) {
    this.role = deps.role ?? "participant";
    this.now = deps.now ?? Date.now;
    this.setTimeoutImpl = deps.setTimeout ?? ((fn, ms) => setTimeout(fn, ms));
    this.clearTimeoutImpl = deps.clearTimeout ?? ((h) => clearTimeout(h as ReturnType<typeof setTimeout>));
    this.iceRestartDelayMs = deps.iceRestartDelayMs ?? DEFAULT_ICE_RESTART_DELAY_MS;
    this.maxIceRestarts = deps.maxIceRestarts ?? DEFAULT_MAX_ICE_RESTARTS;
    this.view = { ...IDLE_CALL_VIEW, selfRole: this.role };
  }

  get current(): CallView {
    return this.view;
  }

  get callId(): string | null {
    return this.view.callId;
  }

  get selfRole(): CallRole {
    return this.role;
  }

  /** The call exists on the server (created or joined): wait for the roster. */
  begin(created: CallCreated): void {
    this.selfLabel = created.selfLabel;
    this.joinCode = created.joinCode;
    if (created.iceServers.length > 0) this.iceServers = created.iceServers;
    this.view = {
      ...IDLE_CALL_VIEW,
      selfRole: this.role,
      selfLabel: created.selfLabel,
      hasTurn: hasTurnServer(this.iceServers),
      status: "waiting",
      callId: created.callId,
      joinCode: created.joinCode,
      joinUrl: created.joinUrl,
    };
    this.deps.onChange(this.view);
  }

  /** Web only — must run inside the Answer/Start tap (see RtcAdapter.prime). */
  prime(): void {
    this.deps.adapter.prime?.();
  }

  /** The session socket (re)opened: (re)announce ourselves with our role
   *  (+ the code and name, so a socket that raced the REST join still
   *  binds — the server's call_join accepts a non-member with the code). */
  onSocketOpen(): void {
    const callId = this.view.callId;
    if (!callId || this.closed) return;
    this.deps.send({
      type: "call_join",
      call_id: callId,
      role: this.role,
      ...(this.joinCode ? { join_code: this.joinCode } : {}),
      ...(this.deps.displayName ? { display_name: this.deps.displayName } : {}),
    });
  }

  /**
   * Route a server frame. Returns true when it was a call frame (consumed),
   * false for everything else (the hook keeps handling those).
   */
  handleServerMessage(data: unknown): boolean {
    if (!data || typeof data !== "object") return false;
    const msg = data as Partial<CallServerMessage> & { type?: unknown };
    if (msg.type === "call_state") {
      this.onCallState(msg as CallServerMessage & { type: "call_state" });
      return true;
    }
    if (msg.type === "rtc_signal") {
      void this.onRtcSignal(msg as CallServerMessage & { type: "rtc_signal" });
      return true;
    }
    if (msg.type === "call_ended") {
      this.end("ended", null);
      return true;
    }
    return false;
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    for (const t of this.local?.getAudioTracks() ?? []) t.enabled = !muted;
    this.publish();
  }

  /** Tear everything down (local hang-up, session stop, unmount). */
  hangUp(): void {
    if (this.closed) return;
    this.end("ended", null);
  }

  // --- roster ----------------------------------------------------------------

  private onCallState(msg: CallServerMessage & { type: "call_state" }) {
    if (this.closed) return;
    if (Array.isArray(msg.ice_servers) && msg.ice_servers.length > 0) this.iceServers = msg.ice_servers;
    if (typeof msg.self_label === "string") this.selfLabel = msg.self_label;
    const participants: CallParticipant[] = (msg.participants ?? []).map((p) => ({
      uid: String(p.uid),
      label: (p.label && String(p.label)) || (p.slot ? `Speaker ${p.slot}` : "Speaker ?"),
      displayName: (p.display_name && String(p.display_name)) || (p.label && String(p.label)) || "Someone",
      role: p.role === "therapist" ? "therapist" : "participant",
      isSelf: p.is_self === true,
      connected: p.connected !== false,
    }));
    const self = participants.find((p) => p.isSelf) ?? null;
    if (self) {
      this.selfUid = self.uid;
      if (!this.selfLabel) this.selfLabel = self.label;
    } else if (typeof msg.self_uid === "string") {
      this.selfUid = msg.self_uid;
    }
    const others = participants.filter((p) => !p.isSelf);
    const seen = new Set<string>();
    for (const p of others) {
      seen.add(p.uid);
      const existing = this.links.get(p.uid);
      if (!p.connected) {
        // Present but not connected (still joining, or dropped): drop any
        // link so it rebuilds cleanly when they come (back).
        if (existing) this.dropLink(p.uid);
        this.links.set(p.uid, this.freshLink(p));
        continue;
      }
      if (!existing || !existing.pc) {
        this.links.set(p.uid, existing ? { ...existing, participant: p } : this.freshLink(p));
        if (this.selfUid) void this.connectLink(p.uid);
      } else {
        existing.participant = p;
        // The server buffers nothing: an offer sent before their socket was
        // bound was refused. Now that they are connected, offer again if
        // this link never got a remote description.
        if (
          this.selfUid &&
          isOfferer(this.selfUid, p.uid) &&
          !existing.remoteDescriptionSet &&
          !existing.negotiating &&
          existing.iceState === "new"
        ) {
          void this.sendOffer(p.uid, existing.pc, false).catch(() => {});
        }
      }
    }
    // Someone left the roster entirely: tear their link down.
    for (const uid of [...this.links.keys()]) {
      if (!seen.has(uid)) this.dropLink(uid, true);
    }
    this.publish();
  }

  private freshLink(participant: CallParticipant): PeerLink {
    return {
      participant,
      pc: null,
      remoteDescriptionSet: false,
      pendingCandidates: [],
      negotiating: false,
      restartTimer: null,
      iceRestarts: 0,
      iceState: "new",
    };
  }

  // --- negotiation ------------------------------------------------------------

  private async connectLink(uid: string): Promise<void> {
    const link = this.links.get(uid);
    if (!link || link.negotiating || link.pc || this.closed) return;
    link.negotiating = true;
    try {
      const pc = this.deps.adapter.createPeer({ iceServers: this.iceServers });
      link.pc = pc;
      link.remoteDescriptionSet = false;
      link.pendingCandidates = [];
      pc.onicecandidate = (e) => {
        const c = candidateToJson(e.candidate);
        if (c && c.candidate) this.signal(uid, c);
      };
      pc.ontrack = (e) => {
        this.deps.adapter.playRemote(uid, e.streams[0] ?? null, e.track);
      };
      pc.oniceconnectionstatechange = () => this.onIceState(uid, pc);
      const local = await this.getLocal();
      if (this.closed || this.links.get(uid)?.pc !== pc) return;
      for (const track of local.getAudioTracks()) {
        track.enabled = !this.muted;
        pc.addTrack(track, local);
      }
      if (this.selfUid && isOfferer(this.selfUid, uid)) {
        await this.sendOffer(uid, pc, false);
      }
      // The higher uid waits for the offer through rtc_signal.
    } catch (err) {
      this.fail(err instanceof Error ? err.message : String(err));
    } finally {
      const l = this.links.get(uid);
      if (l) l.negotiating = false;
    }
  }

  private async sendOffer(uid: string, pc: RtcPeerLike, iceRestart: boolean) {
    const offer = await pc.createOffer(iceRestart ? { iceRestart: true } : undefined);
    await pc.setLocalDescription(offer);
    const sdp = sdpToJson(offer);
    if (sdp) this.signal(uid, sdp);
  }

  private async onRtcSignal(msg: CallServerMessage & { type: "rtc_signal" }) {
    if (this.closed) return;
    const from = msg.from ? String(msg.from) : null;
    if (!from) return;
    const payload: RtcSignalPayload | null =
      msg.payload && typeof msg.payload === "object" ? (msg.payload as RtcSignalPayload) : null;
    if (!payload) return;
    const sdp: SdpInit | null = isSdpPayload(payload) ? payload : null;
    const candidate: IceCandidateInit | null =
      !sdp && typeof (payload as IceCandidateInit).candidate === "string" ? (payload as IceCandidateInit) : null;
    let link = this.links.get(from);
    if (!link || !link.pc) {
      // A signal can beat the roster: build the link from what we know and
      // negotiate. Only an OFFER can bootstrap it (a candidate/answer with
      // no link is a stale straggler).
      if (!sdp || sdp.type !== "offer") return;
      if (!link) {
        link = this.freshLink({ uid: from, label: "Speaker ?", displayName: "Someone", role: "participant", isSelf: false, connected: true });
        this.links.set(from, link);
      }
      await this.connectLink(from);
      link = this.links.get(from);
      if (!link || !link.pc) return;
    }
    const pc = link.pc;
    try {
      if (sdp) {
        if (sdp.type === "offer") {
          if (this.selfUid && isOfferer(this.selfUid, from)) {
            // By construction only the lower uid offers a given link; ignore
            // a stray offer from the higher-uid side (no glare to resolve).
            return;
          }
          await pc.setRemoteDescription(sdp);
          link.remoteDescriptionSet = true;
          await this.flushCandidates(link);
          const answer = await pc.createAnswer();
          await pc.setLocalDescription(answer);
          const out = sdpToJson(answer);
          if (out) this.signal(from, out);
        } else if (sdp.type === "answer") {
          await pc.setRemoteDescription(sdp);
          link.remoteDescriptionSet = true;
          await this.flushCandidates(link);
        }
      }
      if (candidate) {
        if (link.remoteDescriptionSet) await pc.addIceCandidate(candidate);
        else link.pendingCandidates.push(candidate);
      }
    } catch (err) {
      this.fail(`signaling failed (${err instanceof Error ? err.message : String(err)})`);
    }
  }

  private async flushCandidates(link: PeerLink) {
    const pc = link.pc;
    if (!pc) return;
    const queued = link.pendingCandidates;
    link.pendingCandidates = [];
    for (const c of queued) {
      try {
        await pc.addIceCandidate(c);
      } catch {
        // A stale candidate from before a restart: harmless.
      }
    }
  }

  // --- ICE health ---------------------------------------------------------------

  private onIceState(uid: string, pc: RtcPeerLike) {
    const link = this.links.get(uid);
    if (this.closed || !link || link.pc !== pc) return;
    const state = pc.iceConnectionState;
    link.iceState = state;
    if (state === "connected" || state === "completed") {
      this.clearRestartTimer(link);
      if (this.connectedAt === null) this.connectedAt = this.now();
    } else if (state === "failed") {
      this.restartIce(uid);
    } else if (state === "disconnected") {
      if (link.restartTimer === null) {
        link.restartTimer = this.setTimeoutImpl(() => {
          link.restartTimer = null;
          if (this.links.get(uid) === link && pc.iceConnectionState === "disconnected") this.restartIce(uid);
        }, this.iceRestartDelayMs);
      }
    }
    this.publish();
  }

  private restartIce(uid: string) {
    const link = this.links.get(uid);
    if (!link || !link.pc || this.closed) return;
    if (!this.selfUid || !isOfferer(this.selfUid, uid)) {
      // The higher-uid side waits for the offerer's restart offer.
      this.publish();
      return;
    }
    if (link.iceRestarts >= this.maxIceRestarts) {
      // This one link is unrecoverable; drop it (the others survive).
      this.dropLink(uid);
      this.links.set(uid, this.freshLink(link.participant));
      this.publish();
      return;
    }
    link.iceRestarts += 1;
    link.remoteDescriptionSet = false;
    link.pendingCandidates = [];
    const pc = link.pc;
    void this.sendOffer(uid, pc, true).catch((err) =>
      this.fail(`ICE restart failed (${err instanceof Error ? err.message : String(err)})`),
    );
    this.publish();
  }

  private clearRestartTimer(link: PeerLink) {
    if (link.restartTimer !== null) {
      this.clearTimeoutImpl(link.restartTimer);
      link.restartTimer = null;
    }
  }

  // --- plumbing -------------------------------------------------------------------

  private signal(to: string, payload: RtcSignalPayload) {
    const callId = this.view.callId;
    if (!callId) return;
    this.deps.send({ type: "rtc_signal", call_id: callId, to, payload });
  }

  private getLocal(): Promise<RtcStreamLike> {
    if (this.local) return Promise.resolve(this.local);
    if (!this.localPromise) {
      this.localPromise = this.deps.adapter.getLocalStream().then((s) => {
        this.local = s;
        return s;
      });
    }
    return this.localPromise;
  }

  private dropLink(uid: string, forget = false) {
    const link = this.links.get(uid);
    if (!link) return;
    this.clearRestartTimer(link);
    const pc = link.pc;
    link.pc = null;
    link.remoteDescriptionSet = false;
    link.pendingCandidates = [];
    if (pc) {
      pc.onicecandidate = null;
      pc.ontrack = null;
      pc.oniceconnectionstatechange = null;
      try {
        pc.close();
      } catch {
        // Already closed.
      }
    }
    this.deps.adapter.stopRemote(uid);
    if (forget) this.links.delete(uid);
  }

  private end(status: "ended" | "failed", error: string | null) {
    this.closed = true;
    for (const uid of [...this.links.keys()]) this.dropLink(uid, true);
    this.deps.adapter.stopRemote();
    const local = this.local;
    this.local = null;
    this.localPromise = null;
    for (const t of local?.getTracks() ?? []) {
      try {
        t.stop();
      } catch {
        // Track already stopped.
      }
    }
    this.view = { ...this.view, status, error };
    this.deps.onChange(this.view);
  }

  private fail(error: string) {
    if (this.closed) return;
    this.end("failed", error);
  }

  /** Recompute the CallView from the peer map and push it to the screen. */
  private publish() {
    if (this.closed) return;
    const peers: CallPeer[] = [...this.links.values()].map((l) => ({
      uid: l.participant.uid,
      label: l.participant.label,
      displayName: l.participant.displayName,
      role: l.participant.role,
      connected: l.iceState === "connected" || l.iceState === "completed",
      iceRestarts: l.iceRestarts,
    }));
    const anyConnected = peers.some((p) => p.connected);
    const anyReconnecting = [...this.links.values()].some(
      (l) => l.pc && (l.iceState === "disconnected" || (l.iceRestarts > 0 && l.iceState !== "connected" && l.iceState !== "completed")),
    );
    let status: CallStatus;
    if (peers.length === 0) status = "waiting";
    else if (anyConnected) status = anyReconnecting ? "reconnecting" : "connected";
    else if (anyReconnecting) status = "reconnecting";
    else status = "connecting";
    this.view = {
      ...this.view,
      status,
      selfRole: this.role,
      selfLabel: this.selfLabel,
      peers,
      hasTurn: hasTurnServer(this.iceServers),
      connectedAt: this.connectedAt,
      muted: this.muted,
      iceRestarts: [...this.links.values()].reduce((n, l) => n + l.iceRestarts, 0),
    };
    this.deps.onChange(this.view);
  }
}

/** Stop every track of a stream, for adapters. */
export function stopTracks(stream: { getTracks(): RtcTrackLike[] } | null): void {
  for (const t of stream?.getTracks() ?? []) {
    try {
      t.stop();
    } catch {
      // Already stopped.
    }
  }
}
