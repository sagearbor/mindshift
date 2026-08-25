/**
 * Connectivity pre-flight for in-app calls.
 *
 * The failure this exists to prevent: two phones on cellular sit behind
 * carrier-grade NAT, no direct path exists, and without a TURN relay the
 * WebRTC connection never comes up — the demo dies with a spinner and no
 * explanation. So BEFORE the call we open one throwaway peer connection
 * against exactly the `ice_servers` the server hands out (`GET /calls/ice`),
 * let it gather, and read what came back:
 *
 *   host   — this device's own LAN address (always there)
 *   srflx  — a STUN server saw us: a public address/port pair exists, so a
 *            direct peer-to-peer path is plausible
 *   relay  — a TURN server allocated us a relay: the call connects even
 *            when both sides are behind symmetric/carrier-grade NAT
 *
 * The verdict is one honest line for the pre-flight panel. It never claims
 * more than the candidates support: gathering a relay candidate proves the
 * TURN server answered US, not that the far end will also reach it.
 *
 * Everything is injectable (`createPeer`) so the Jest tests drive a fake
 * RTCPeerConnection; the default factory prefers the browser's global and
 * falls back to react-native-webrtc, lazily required so importing this
 * module never pulls the native module into a build that lacks it.
 */
import { hasTurnServer, type IceServer } from "./types";

/** The slice of RTCPeerConnection a gathering probe needs. */
export interface IceProbePeerLike {
  onicecandidate: ((e: { candidate: { candidate?: string } | null }) => void) | null;
  onicegatheringstatechange?: (() => void) | null;
  iceGatheringState?: string;
  createDataChannel(label: string): unknown;
  createOffer(options?: unknown): Promise<{ type: string; sdp?: string }>;
  setLocalDescription(description?: unknown): Promise<void>;
  close(): void;
}

export type IceProbePeerFactory = (config: { iceServers: IceServer[] }) => IceProbePeerLike;

export type IceVerdict =
  /** A relay candidate came back: this works even on carrier-grade NAT. */
  | "relay"
  /** STUN reflected us and no TURN is configured: direct is plausible, with no fallback. */
  | "direct"
  /** No reflexive candidate and no TURN configured — a relay is what's missing. */
  | "relay-needed"
  /** TURN IS configured but never allocated (bad creds, blocked port, wrong realm). */
  | "turn-unreachable"
  /** Nothing at all came back. */
  | "blocked"
  /** We couldn't run the check here (no WebRTC in this build, or it threw). */
  | "unavailable";

export interface IceProbeResult {
  host: boolean;
  srflx: boolean;
  relay: boolean;
  /** Whether the server's list contained any turn:/turns: URL at all. */
  turnConfigured: boolean;
  /** Every candidate type seen, in arrival order (host/srflx/prflx/relay). */
  types: string[];
  candidates: number;
  verdict: IceVerdict;
  /** The one line the pre-flight panel shows after "Peer connection:". */
  line: string;
  /** Why the probe couldn't run, when `verdict` is "unavailable". */
  reason: string | null;
  elapsedMs: number;
}

/** How long to let gathering run before judging on what arrived. TURN
 *  allocation over TCP/TLS can take a couple of seconds on a bad network. */
export const ICE_PROBE_TIMEOUT_MS = 5000;

/** The `typ` of one SDP candidate line ("host" | "srflx" | "prflx" | "relay"). */
export function candidateType(candidate: string | null | undefined): string | null {
  if (!candidate) return null;
  const match = /(?:^|\s)typ\s+(host|srflx|prflx|relay)(?:\s|$)/i.exec(candidate);
  return match ? match[1].toLowerCase() : null;
}

/** The honest one-liner for a set of gathered candidate types. */
export function verdictFor(seen: {
  host: boolean;
  srflx: boolean;
  relay: boolean;
  turnConfigured: boolean;
  candidates: number;
}): { verdict: IceVerdict; line: string } {
  if (seen.candidates === 0) {
    return { verdict: "blocked", line: "blocked — this network returned no ICE candidates at all" };
  }
  if (seen.relay) {
    return { verdict: "relay", line: "relay ready — a call connects even on carrier-grade NAT" };
  }
  if (seen.turnConfigured) {
    return seen.srflx
      ? {
          verdict: "turn-unreachable",
          line: "TURN is configured but gave no relay candidate — check the credentials, realm or ports",
        }
      : {
          verdict: "turn-unreachable",
          line: "blocked — neither STUN nor TURN answered (a firewall is eating UDP)",
        };
  }
  if (seen.srflx) {
    return {
      verdict: "direct",
      line: "direct likely — but no TURN is configured, so a phone-to-phone call over cellular can still fail",
    };
  }
  return { verdict: "relay-needed", line: "relay needed — no TURN configured" };
}

/** Green only when a relay actually came back; everything else is a warning
 *  the owner should read, never a silent pass. */
export function iceProbeOk(result: IceProbeResult | null): boolean | null {
  if (!result) return null;
  return result.verdict === "relay";
}

/** The default peer factory: the browser's RTCPeerConnection, else
 *  react-native-webrtc, else null (the probe reports "unavailable"). */
export function defaultIceProbePeerFactory(): IceProbePeerFactory | null {
  const global = globalThis as { RTCPeerConnection?: new (config: unknown) => IceProbePeerLike };
  if (typeof global.RTCPeerConnection === "function") {
    const Ctor = global.RTCPeerConnection;
    return (config) => new Ctor(config);
  }
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("react-native-webrtc") as {
      RTCPeerConnection?: new (config: unknown) => IceProbePeerLike;
    };
    if (mod && typeof mod.RTCPeerConnection === "function") {
      const Ctor = mod.RTCPeerConnection;
      return (config) => new Ctor(config);
    }
  } catch {
    // No native module in this build — handled by the caller.
  }
  return null;
}

/** The result to show when the check could not run at all (no WebRTC in
 *  this build, the ICE fetch failed, the probe threw). Honest, never a pass. */
export function iceProbeUnavailable(
  reason: string,
  turnConfigured = false,
  elapsedMs = 0,
): IceProbeResult {
  return {
    host: false,
    srflx: false,
    relay: false,
    turnConfigured,
    types: [],
    candidates: 0,
    verdict: "unavailable",
    line: `couldn't check on this device (${reason})`,
    reason,
    elapsedMs,
  };
}

/**
 * Gather ICE candidates once against `iceServers` and report what the
 * network allows. Never throws and never leaves the peer connection open.
 */
export async function probeIce(
  iceServers: IceServer[],
  options: {
    createPeer?: IceProbePeerFactory | null;
    timeoutMs?: number;
    now?: () => number;
  } = {},
): Promise<IceProbeResult> {
  const now = options.now ?? (() => Date.now());
  const startedAt = now();
  const turnConfigured = hasTurnServer(iceServers);
  const createPeer =
    options.createPeer === undefined ? defaultIceProbePeerFactory() : options.createPeer;
  if (!createPeer) {
    return iceProbeUnavailable("no WebRTC in this build", turnConfigured, 0);
  }

  let peer: IceProbePeerLike | null = null;
  const types: string[] = [];
  try {
    peer = createPeer({ iceServers });
    const gathered = new Promise<void>((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        resolve();
      };
      const timer = setTimeout(finish, options.timeoutMs ?? ICE_PROBE_TIMEOUT_MS);
      // Some runtimes don't unref timers; a probe must never hold the app open.
      (timer as unknown as { unref?: () => void }).unref?.();
      peer!.onicecandidate = (event) => {
        const candidate = event?.candidate;
        if (!candidate) {
          // null candidate = gathering finished.
          finish();
          return;
        }
        const type = candidateType(candidate.candidate);
        if (type) types.push(type);
        // A relay candidate is the strongest answer there is — stop early
        // rather than waiting out the whole gathering timeout.
        if (type === "relay") finish();
      };
      peer!.onicegatheringstatechange = () => {
        if (peer?.iceGatheringState === "complete") finish();
      };
    });

    // A data channel gives the offer something to gather for (no mic needed:
    // the pre-flight must not take the microphone away from the fast loop).
    peer.createDataChannel("mindshift-ice-probe");
    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    await gathered;
  } catch (err) {
    return iceProbeUnavailable(
      err instanceof Error ? err.message : String(err),
      turnConfigured,
      now() - startedAt,
    );
  } finally {
    try {
      peer?.close();
    } catch {
      // Closing a already-dead connection is not a failure worth surfacing.
    }
  }

  const seen = {
    host: types.includes("host"),
    srflx: types.includes("srflx") || types.includes("prflx"),
    relay: types.includes("relay"),
    turnConfigured,
    candidates: types.length,
  };
  const { verdict, line } = verdictFor(seen);
  return { ...seen, types, verdict, line, reason: null, elapsedMs: now() - startedAt };
}
