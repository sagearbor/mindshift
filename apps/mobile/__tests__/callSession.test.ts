/**
 * CallSession (src/live/call/callSession.ts) over a fake RtcAdapter and a
 * fake signaling send. The mesh: a peer map keyed by uid, one connection
 * per other participant; the lexicographically-lower uid offers each link.
 * Covers 2-party offer/answer/ICE, a 3-party call (two peers, one where we
 * offer and one where we answer), the therapist role on join, per-peer
 * ICE restart and leave/return, mute across links, and call_ended.
 */
import { CallSession, isOfferer } from "../src/live/call/callSession";
import type { RtcAdapter, RtcPeerLike, RtcStreamLike, RtcTrackLike } from "../src/live/call/rtc";
import type { CallClientMessage, CallView } from "../src/live/call/types";

class FakeTrack implements RtcTrackLike {
  enabled = true;
  kind = "audio";
  stopped = false;
  stop() {
    this.stopped = true;
  }
}

class FakeStream implements RtcStreamLike {
  tracks = [new FakeTrack()];
  getTracks() {
    return this.tracks;
  }
  getAudioTracks() {
    return this.tracks;
  }
}

class FakePeer implements RtcPeerLike {
  static all: FakePeer[] = [];
  iceConnectionState = "new";
  onicecandidate: RtcPeerLike["onicecandidate"] = null;
  oniceconnectionstatechange: RtcPeerLike["oniceconnectionstatechange"] = null;
  ontrack: RtcPeerLike["ontrack"] = null;
  offers: { iceRestart: boolean }[] = [];
  answers = 0;
  local: unknown[] = [];
  remote: unknown[] = [];
  candidates: unknown[] = [];
  tracks: RtcTrackLike[] = [];
  closed = false;
  constructor(public config: unknown) {
    FakePeer.all.push(this);
  }
  async createOffer(options?: { iceRestart?: boolean }) {
    this.offers.push({ iceRestart: options?.iceRestart === true });
    return { type: "offer" as const, sdp: `offer-${this.offers.length}` };
  }
  async createAnswer() {
    this.answers += 1;
    return { type: "answer" as const, sdp: `answer-${this.answers}` };
  }
  async setLocalDescription(d?: unknown) {
    this.local.push(d);
  }
  async setRemoteDescription(d: unknown) {
    this.remote.push(d);
  }
  async addIceCandidate(c: unknown) {
    this.candidates.push(c);
  }
  addTrack(track: RtcTrackLike) {
    this.tracks.push(track);
    return {};
  }
  close() {
    this.closed = true;
  }
  ice(state: string) {
    this.iceConnectionState = state;
    this.oniceconnectionstatechange?.();
  }
}

function fakeAdapter() {
  const stream = new FakeStream();
  const played: { uid: string; stream: unknown }[] = [];
  const stopped: (string | undefined)[] = [];
  const adapter: RtcAdapter & { stream: FakeStream; played: typeof played; stopped: typeof stopped; primed: number } = {
    stream,
    played,
    stopped,
    primed: 0,
    createPeer: (config) => new FakePeer(config),
    getLocalStream: async () => stream,
    playRemote: (uid, s) => played.push({ uid, stream: s }),
    stopRemote: (uid) => stopped.push(uid),
    prime: () => {
      adapter.primed += 1;
    },
  };
  return adapter;
}

interface HarnessOpts {
  selfUid?: string;
  role?: "participant" | "therapist";
}

function harness(opts: HarnessOpts = {}) {
  const adapter = fakeAdapter();
  const sent: CallClientMessage[] = [];
  const views: CallView[] = [];
  const timers: { fn: () => void; ms: number }[] = [];
  const session = new CallSession({
    adapter,
    role: opts.role,
    send: (m) => {
      sent.push(m);
      return true;
    },
    onChange: (v) => views.push(v),
    now: () => 1_000,
    setTimeout: (fn, ms) => {
      timers.push({ fn, ms });
      return timers.length;
    },
    clearTimeout: () => {},
    iceRestartDelayMs: 500,
    maxIceRestarts: 2,
  });
  const selfUid = opts.selfUid ?? "a-sage";
  const p = (uid: string, name: string, connected = true, role: "participant" | "therapist" = "participant") => ({
    uid,
    slot: uid[0].toUpperCase(),
    label: `Speaker ${uid[0].toUpperCase()}`,
    display_name: name,
    role,
    is_self: uid === selfUid,
    connected,
  });
  return {
    adapter,
    sent,
    views,
    session,
    timers,
    selfUid,
    p,
    roster: (...participants: ReturnType<typeof p>[]) => ({
      type: "call_state",
      call_id: "c1",
      participants,
      ice_servers: [{ urls: "stun:stun.example:3478" }],
    }),
    last: () => views[views.length - 1],
    peerNames: () => views[views.length - 1].peers.map((pr) => pr.displayName).sort(),
    sig: () => sent.filter((m) => m.type === "rtc_signal") as Extract<CallClientMessage, { type: "rtc_signal" }>[],
  };
}

const flush = () => new Promise((r) => setTimeout(r, 0));
/** A REST create/join result with the new required fields. */
const created = (callId: string, joinCode = "K7", joinUrl = "") => ({
  callId,
  joinCode,
  joinUrl,
  selfLabel: "Speaker A",
  selfRole: "participant" as const,
  iceServers: [],
});

beforeEach(() => {
  FakePeer.all = [];
});

describe("isOfferer", () => {
  it("is decided by uid order, symmetrically", () => {
    expect(isOfferer("a", "b")).toBe(true);
    expect(isOfferer("b", "a")).toBe(false);
  });
});

describe("CallSession — 2 party", () => {
  it("creates the call, joins with role on socket open, and (as the lower uid) offers, then connects", async () => {
    const h = harness();
    h.session.begin(created("c1", "K7", "https://x/call/K7"));
    expect(h.last()).toMatchObject({ status: "waiting", callId: "c1", selfRole: "participant" });
    h.session.onSocketOpen();
    expect(h.sent[0]).toEqual({ type: "call_join", call_id: "c1", role: "participant", join_code: "K7" });

    h.session.handleServerMessage(h.roster(h.p("a-sage", "Sage"), h.p("b-mom", "Mom")));
    await flush();
    const pc = FakePeer.all[0];
    expect(pc.config).toEqual({ iceServers: [{ urls: "stun:stun.example:3478" }] });
    expect(pc.tracks).toHaveLength(1);
    expect(h.last().status).toBe("connecting");
    expect(h.peerNames()).toEqual(["Mom"]);
    // Offer went out, addressed to the peer (required `to`).
    // Payload is the SDP init itself (the server relays it verbatim).
    expect(h.sig()[0]).toEqual({
      type: "rtc_signal",
      call_id: "c1",
      to: "b-mom",
      payload: { type: "offer", sdp: "offer-1" },
    });

    pc.onicecandidate?.({ candidate: { candidate: "cand-1", sdpMid: "0", sdpMLineIndex: 0 } });
    expect(h.sig()[h.sig().length - 1]).toMatchObject({ to: "b-mom", payload: { candidate: "cand-1", sdpMid: "0" } });
    h.session.handleServerMessage({ type: "rtc_signal", from: "b-mom", payload: { type: "answer", sdp: "ans" } });
    await flush();
    expect(pc.remote).toEqual([{ type: "answer", sdp: "ans" }]);
    pc.ice("connected");
    expect(h.last()).toMatchObject({ status: "connected", connectedAt: 1_000 });
    expect(h.last().peers[0].connected).toBe(true);

    const remote = new FakeStream();
    pc.ontrack?.({ track: remote.tracks[0], streams: [remote] });
    expect(h.adapter.played).toEqual([{ uid: "b-mom", stream: remote }]);

    h.session.setMuted(true);
    expect(h.adapter.stream.tracks[0].enabled).toBe(false);
    expect(h.last().muted).toBe(true);
    h.session.hangUp();
    expect(pc.closed).toBe(true);
    expect(h.adapter.stream.tracks[0].stopped).toBe(true);
    expect(h.last().status).toBe("ended");
  });

  it("as the higher uid, waits and answers; candidates before the offer are queued", async () => {
    const h = harness({ selfUid: "b-mom" });
    h.session.begin(created("c1", "K7", ""));
    h.session.handleServerMessage(h.roster(h.p("b-mom", "Mom"), h.p("a-sage", "Sage")));
    await flush();
    const pc = FakePeer.all[0];
    expect(pc.offers).toHaveLength(0);
    expect(h.sig()).toHaveLength(0);
    h.session.handleServerMessage({ type: "rtc_signal", from: "a-sage", payload: { candidate: "early", sdpMid: "0" } });
    await flush();
    expect(pc.candidates).toEqual([]);
    h.session.handleServerMessage({ type: "rtc_signal", from: "a-sage", payload: { type: "offer", sdp: "o" } });
    await flush();
    expect(pc.remote).toEqual([{ type: "offer", sdp: "o" }]);
    expect(pc.candidates).toEqual([{ candidate: "early", sdpMid: "0" }]);
    expect(pc.answers).toBe(1);
    expect(h.sig()[h.sig().length - 1]).toEqual({ type: "rtc_signal", call_id: "c1", to: "a-sage", payload: { type: "answer", sdp: "answer-1" } });
    pc.ice("completed");
    expect(h.last().status).toBe("connected");
  });
});

describe("CallSession — 3 party mesh with roles", () => {
  it("holds one link per peer: offers to the higher uid, answers the lower, therapist included", async () => {
    // self is the middle uid: it offers to "c-dad" and answers "a-...".
    const h = harness({ selfUid: "b-sage" });
    h.session.begin(created("c1", "K7", ""));
    h.session.handleServerMessage(
      h.roster(h.p("b-sage", "Sage"), h.p("a-dad", "Dad"), h.p("c-mom", "Mom", true, "therapist")),
    );
    await flush();
    // Two links, two peer connections.
    expect(FakePeer.all).toHaveLength(2);
    expect(h.peerNames()).toEqual(["Dad", "Mom"]);
    // We offer to c-mom (higher uid) but NOT to a-dad (lower — they offer us).
    const offers = h.sig().filter((m) => (m.payload as { type?: string }).type === "offer");
    expect(offers).toHaveLength(1);
    expect(offers[0].to).toBe("c-mom");
    // The therapist peer is a normal audio peer (role is view-only metadata).
    expect(h.last().peers.find((pr) => pr.uid === "c-mom")?.role).toBe("therapist");

    // Dad (lower uid) offers us; we answer.
    h.session.handleServerMessage({ type: "rtc_signal", from: "a-dad", payload: { type: "offer", sdp: "od" } });
    await flush();
    const dadPc = FakePeer.all.find((pc) => pc.remote.some((d) => (d as { sdp?: string }).sdp === "od"))!;
    expect(dadPc.answers).toBe(1);

    // Both connect: aggregate status is connected, each peer flagged.
    for (const pc of FakePeer.all) pc.ice("connected");
    expect(h.last().status).toBe("connected");
    expect(h.last().peers.every((pr) => pr.connected)).toBe(true);

    // Remote audio is played per peer uid (a mesh mixes everyone).
    const s1 = new FakeStream();
    FakePeer.all[0].ontrack?.({ track: s1.tracks[0], streams: [s1] });
    expect(h.adapter.played).toHaveLength(1);
  });

  it("a therapist joins with its role in call_join", () => {
    const h = harness({ role: "therapist" });
    h.session.begin(created("c1", "K7", ""));
    expect(h.last().selfRole).toBe("therapist");
    h.session.onSocketOpen();
    expect(h.sent[0]).toEqual({ type: "call_join", call_id: "c1", role: "therapist", join_code: "K7" });
  });

  it("one peer leaving tears down only that link; the other survives", async () => {
    const h = harness({ selfUid: "a-sage" });
    h.session.begin(created("c1", "K7", ""));
    h.session.handleServerMessage(h.roster(h.p("a-sage", "Sage"), h.p("b-dad", "Dad"), h.p("c-mom", "Mom", true, "therapist")));
    await flush();
    for (const pc of FakePeer.all) pc.ice("connected");
    expect(h.last().peers).toHaveLength(2);
    // Dad drops off the roster entirely.
    h.session.handleServerMessage(h.roster(h.p("a-sage", "Sage"), h.p("c-mom", "Mom", true, "therapist")));
    await flush();
    expect(h.peerNames()).toEqual(["Mom"]);
    expect(h.adapter.stopped).toContain("b-dad");
    // The therapist link is untouched and still connected.
    expect(h.last().peers[0]).toMatchObject({ uid: "c-mom", connected: true });
    expect(h.last().status).toBe("connected");
  });

  it("restarts ICE on the offerer's link only, per link, and mutes across all links", async () => {
    const h = harness({ selfUid: "a-sage" });
    h.session.begin(created("c1", "K7", ""));
    h.session.handleServerMessage(h.roster(h.p("a-sage", "Sage"), h.p("b-dad", "Dad"), h.p("c-mom", "Mom")));
    await flush();
    // We are the lowest uid: we offer BOTH links.
    expect(h.sig().filter((m) => (m.payload as { type?: string }).type === "offer")).toHaveLength(2);
    for (const pc of FakePeer.all) pc.ice("connected");
    // One link fails → a restart offer on THAT peer connection only.
    const dadPc = FakePeer.all[0];
    dadPc.ice("failed");
    await flush();
    expect(dadPc.offers[dadPc.offers.length - 1]).toEqual({ iceRestart: true });
    expect(FakePeer.all[1].offers.every((o) => !o.iceRestart)).toBe(true);
    expect(h.last().status).toBe("reconnecting");
    // Mute flips the single local track that every link shares.
    h.session.setMuted(true);
    expect(h.adapter.stream.tracks[0].enabled).toBe(false);
    h.session.hangUp();
    expect(FakePeer.all.every((pc) => pc.closed)).toBe(true);
  });
});

describe("CallSession — misc", () => {
  it("ends on call_ended and reports a failed local stream honestly", async () => {
    const h = harness();
    h.session.begin(created("c1", "K7", ""));
    h.session.handleServerMessage({ type: "call_ended", call_id: "c1" });
    expect(h.last().status).toBe("ended");

    const broken = harness();
    broken.adapter.getLocalStream = async () => {
      throw new Error("mic denied");
    };
    broken.session.begin(created("c2", "Q", ""));
    broken.session.handleServerMessage(broken.roster(broken.p("a-sage", "Sage"), broken.p("b-mom", "Mom")));
    await flush();
    await flush();
    expect(broken.last()).toMatchObject({ status: "failed", error: "mic denied" });
  });

  it("ignores frames that aren't call frames", () => {
    const h = harness();
    expect(h.session.handleServerMessage({ type: "transcript", text: "hi" })).toBe(false);
    expect(h.session.handleServerMessage(null)).toBe(false);
  });
});
