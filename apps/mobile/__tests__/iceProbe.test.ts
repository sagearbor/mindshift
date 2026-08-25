/**
 * The call connectivity pre-flight (src/live/call/iceProbe.ts) over a fake
 * RTCPeerConnection: candidate-type parsing, the honest verdict for every
 * combination of what came back, early exit on a relay candidate, the
 * gathering timeout, and the "we couldn't check" paths (no WebRTC in this
 * build, a peer connection that throws).
 */
import {
  ICE_PROBE_TIMEOUT_MS,
  candidateType,
  defaultIceProbePeerFactory,
  iceProbeOk,
  iceProbeUnavailable,
  probeIce,
  verdictFor,
  type IceProbePeerLike,
} from "../src/live/call/iceProbe";
import { hasTurnServer, type IceServer } from "../src/live/call/types";

const STUN_ONLY: IceServer[] = [{ urls: ["stun:stun.l.google.com:19302"] }];
const WITH_TURN: IceServer[] = [
  { urls: ["stun:stun.l.google.com:19302"] },
  { urls: ["turn:relay.example:3478", "turns:relay.example:5349"], username: "1:u", credential: "c" },
];

/** SDP candidate lines as a real browser emits them. */
const HOST = "candidate:1 1 udp 2122260223 192.168.1.9 54321 typ host generation 0";
const SRFLX = "candidate:2 1 udp 1686052607 203.0.113.7 54321 typ srflx raddr 192.168.1.9 rport 54321";
const RELAY = "candidate:3 1 udp 41885439 198.51.100.4 60000 typ relay raddr 203.0.113.7 rport 54321";

class FakePeer implements IceProbePeerLike {
  static last: FakePeer | null = null;
  onicecandidate: IceProbePeerLike["onicecandidate"] = null;
  onicegatheringstatechange: (() => void) | null = null;
  iceGatheringState = "new";
  channels: string[] = [];
  closed = 0;
  /** Candidate lines this fake emits once local description is set; a
   *  trailing null (the default) ends gathering the way a browser does. */
  constructor(
    public config: { iceServers: IceServer[] },
    private lines: (string | null)[] = [HOST, SRFLX, null],
  ) {
    FakePeer.last = this;
  }
  createDataChannel(label: string) {
    this.channels.push(label);
    return {};
  }
  async createOffer(): Promise<{ type: string; sdp?: string }> {
    return { type: "offer", sdp: "v=0" };
  }
  async setLocalDescription() {
    this.iceGatheringState = "gathering";
    for (const candidate of this.lines) {
      this.onicecandidate?.({ candidate: candidate === null ? null : { candidate } });
    }
  }
  close() {
    this.closed += 1;
  }
}

const factory = (lines?: (string | null)[]) => (config: { iceServers: IceServer[] }) =>
  new FakePeer(config, lines);

describe("candidate parsing", () => {
  it("reads the typ out of a real candidate line", () => {
    expect(candidateType(HOST)).toBe("host");
    expect(candidateType(SRFLX)).toBe("srflx");
    expect(candidateType(RELAY)).toBe("relay");
    expect(candidateType("candidate:9 1 tcp 1 10.0.0.1 9 typ PRFLX")).toBe("prflx");
  });

  it("never invents a type", () => {
    expect(candidateType("")).toBeNull();
    expect(candidateType(null)).toBeNull();
    expect(candidateType("candidate:1 1 udp 1 1.2.3.4 1 typ bogus")).toBeNull();
    // "typ" must be its own token — a host*name* is not a candidate type.
    expect(candidateType("candidate:1 1 udp 1 typhost.example 1")).toBeNull();
  });

  it("spots a TURN url in either shape (types.hasTurnServer, shared with CallPanel)", () => {
    expect(hasTurnServer(STUN_ONLY)).toBe(false);
    expect(hasTurnServer(WITH_TURN)).toBe(true);
    expect(hasTurnServer([{ urls: "turns:relay.example:5349" }])).toBe(true);
    expect(hasTurnServer([{ urls: " TURN:relay.example:3478 " }])).toBe(true);
    expect(hasTurnServer(null)).toBe(false);
    // A stun url whose HOST merely contains "turn" is not a relay.
    expect(hasTurnServer([{ urls: "stun:turnip.example:3478" }])).toBe(false);
  });
});

describe("the verdict is honest about what the candidates prove", () => {
  const seen = (over: Partial<Parameters<typeof verdictFor>[0]>) =>
    verdictFor({ host: true, srflx: false, relay: false, turnConfigured: false, candidates: 1, ...over });

  it("a relay candidate is the only green light", () => {
    expect(seen({ relay: true, turnConfigured: true }).verdict).toBe("relay");
    expect(seen({ relay: true, turnConfigured: true }).line).toContain("carrier-grade NAT");
  });

  it("srflx with no TURN says direct is likely AND that there is no fallback", () => {
    const v = seen({ srflx: true });
    expect(v.verdict).toBe("direct");
    expect(v.line).toContain("direct likely");
    expect(v.line).toContain("no TURN is configured");
  });

  it("host only with no TURN names the missing piece", () => {
    expect(seen({}).verdict).toBe("relay-needed");
    expect(seen({}).line).toBe("relay needed — no TURN configured");
  });

  it("TURN configured but no relay candidate blames the TURN config, not the network", () => {
    expect(seen({ srflx: true, turnConfigured: true }).verdict).toBe("turn-unreachable");
    expect(seen({ srflx: true, turnConfigured: true }).line).toContain("credentials");
    expect(seen({ turnConfigured: true }).line).toContain("firewall");
  });

  it("no candidates at all is blocked", () => {
    const v = seen({ host: false, candidates: 0, turnConfigured: true });
    expect(v.verdict).toBe("blocked");
    expect(v.line).toContain("no ICE candidates");
  });

  it("only a working relay reads as OK in the panel", () => {
    expect(iceProbeOk(null)).toBeNull();
    expect(iceProbeOk(iceProbeUnavailable("nope"))).toBe(false);
    for (const verdict of ["direct", "relay-needed", "turn-unreachable", "blocked"] as const) {
      expect(iceProbeOk({ ...iceProbeUnavailable("x"), verdict })).toBe(false);
    }
    expect(iceProbeOk({ ...iceProbeUnavailable("x"), verdict: "relay" })).toBe(true);
  });
});

describe("probeIce", () => {
  it("gathers against the server's own ice servers and reports host+srflx", async () => {
    const result = await probeIce(STUN_ONLY, { createPeer: factory() });
    expect(FakePeer.last?.config.iceServers).toEqual(STUN_ONLY);
    // No microphone is taken: a data channel is what makes it gather.
    expect(FakePeer.last?.channels).toEqual(["mindshift-ice-probe"]);
    expect(result).toMatchObject({
      host: true,
      srflx: true,
      relay: false,
      turnConfigured: false,
      candidates: 2,
      types: ["host", "srflx"],
      verdict: "direct",
      reason: null,
    });
    expect(FakePeer.last?.closed).toBe(1);
  });

  it("a relay candidate ends the probe early and reads as relay-ready", async () => {
    const result = await probeIce(WITH_TURN, { createPeer: factory([HOST, SRFLX, RELAY]) });
    expect(result.relay).toBe(true);
    expect(result.turnConfigured).toBe(true);
    expect(result.verdict).toBe("relay");
    // It resolved on the relay candidate, without a terminating null.
    expect(result.types).toEqual(["host", "srflx", "relay"]);
  });

  it("TURN configured but only host/srflx back is turn-unreachable", async () => {
    const result = await probeIce(WITH_TURN, { createPeer: factory([HOST, SRFLX, null]) });
    expect(result.verdict).toBe("turn-unreachable");
    expect(iceProbeOk(result)).toBe(false);
  });

  it("host only with STUN only is 'relay needed'", async () => {
    const result = await probeIce(STUN_ONLY, { createPeer: factory([HOST, null]) });
    expect(result.verdict).toBe("relay-needed");
  });

  it("nothing at all is blocked, not a pass", async () => {
    const result = await probeIce(STUN_ONLY, { createPeer: factory([null]) });
    expect(result.candidates).toBe(0);
    expect(result.verdict).toBe("blocked");
  });

  it("gives up at the timeout and judges on what arrived", async () => {
    jest.useFakeTimers();
    try {
      // No terminating null: gathering never finishes on its own.
      const pending = probeIce(WITH_TURN, { createPeer: factory([HOST, SRFLX]), timeoutMs: 1234 });
      await Promise.resolve();
      jest.advanceTimersByTime(1234);
      const result = await pending;
      expect(result.types).toEqual(["host", "srflx"]);
      expect(result.verdict).toBe("turn-unreachable");
      expect(FakePeer.last?.closed).toBe(1);
    } finally {
      jest.useRealTimers();
    }
  });

  it("says so — never 'fine' — when there is no WebRTC here", async () => {
    const result = await probeIce(WITH_TURN, { createPeer: null });
    expect(result.verdict).toBe("unavailable");
    expect(result.line).toContain("no WebRTC in this build");
    expect(result.turnConfigured).toBe(true);
    expect(iceProbeOk(result)).toBe(false);
  });

  it("a peer connection that throws is reported, not swallowed", async () => {
    const result = await probeIce(STUN_ONLY, {
      createPeer: () => {
        throw new Error("RTCPeerConnection is not a constructor");
      },
    });
    expect(result.verdict).toBe("unavailable");
    expect(result.reason).toBe("RTCPeerConnection is not a constructor");
  });

  it("a createOffer rejection still closes the connection", async () => {
    class Broken extends FakePeer {
      async createOffer(): Promise<{ type: string; sdp?: string }> {
        throw new Error("no m-lines");
      }
    }
    let peer: Broken | null = null;
    const result = await probeIce(STUN_ONLY, {
      createPeer: (config) => (peer = new Broken(config)),
    });
    expect(result.verdict).toBe("unavailable");
    expect(result.reason).toBe("no m-lines");
    expect((peer as unknown as Broken).closed).toBe(1);
  });

  it("times its own run", async () => {
    let t = 1000;
    const result = await probeIce(STUN_ONLY, { createPeer: factory(), now: () => (t += 25) });
    expect(result.elapsedMs).toBeGreaterThan(0);
  });

  it("the default factory prefers a global RTCPeerConnection (the browser build)", () => {
    const g = globalThis as { RTCPeerConnection?: unknown };
    const had = "RTCPeerConnection" in g;
    const before = g.RTCPeerConnection;
    try {
      g.RTCPeerConnection = FakePeer;
      const make = defaultIceProbePeerFactory();
      expect(make).not.toBeNull();
      expect(make!({ iceServers: STUN_ONLY })).toBeInstanceOf(FakePeer);
    } finally {
      if (had) g.RTCPeerConnection = before;
      else delete g.RTCPeerConnection;
    }
  });

  it("falls back to react-native-webrtc, and never throws when neither exists", () => {
    const g = globalThis as { RTCPeerConnection?: unknown };
    const had = "RTCPeerConnection" in g;
    const before = g.RTCPeerConnection;
    jest.resetModules();
    try {
      delete g.RTCPeerConnection;
      jest.doMock("react-native-webrtc", () => {
        throw new Error("native module missing");
      });
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const reloaded = require("../src/live/call/iceProbe") as typeof import("../src/live/call/iceProbe");
      expect(reloaded.defaultIceProbePeerFactory()).toBeNull();
    } finally {
      jest.dontMock("react-native-webrtc");
      jest.resetModules();
      if (had) g.RTCPeerConnection = before;
      else delete g.RTCPeerConnection;
    }
  });

  it("has a timeout long enough for a TURN allocation over TLS", () => {
    expect(ICE_PROBE_TIMEOUT_MS).toBeGreaterThanOrEqual(3000);
  });
});
