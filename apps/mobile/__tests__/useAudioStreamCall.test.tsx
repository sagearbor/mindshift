import { renderHook, act } from "@testing-library/react-native";

/**
 * useAudioStream in Call mode — the hook's REST + session wiring: startCall
 * creates a call and opens a Call-mode session announcing `call_join {role}`;
 * joinCall passes the invite's role; a failed create/join surfaces the reason
 * and opens no session; a session that ends with errors auto-sends
 * diagnostics. The WebRTC mesh negotiation (offer/answer/ICE/leave/rejoin) is
 * covered synchronously in callSession.test.ts.
 */
import { useAudioStream } from "../src/hooks/useAudioStream";
import type { CallApi } from "../src/live/call/callApi";
import { useDiagnosticsStore } from "../src/diagnostics/diagnostics";
import { useAuthStore } from "../src/store/authStore";

// Deterministic expo-audio (native startSession path): granted permission +
// a controllable stream, like the other useAudioStream* suites.
jest.mock("expo-audio", () => ({
  __esModule: true,
  requestRecordingPermissionsAsync: jest.fn().mockResolvedValue({ status: "granted", granted: true }),
  setAudioModeAsync: jest.fn().mockResolvedValue(undefined),
  useAudioStream: (options?: { onBuffer?: (b: never) => void }) => {
    void options;
    return { stream: { id: "mock", sampleRate: 16000, channels: 1, isStreaming: false, start: jest.fn().mockResolvedValue(undefined), stop: jest.fn() }, isStreaming: false };
  },
}));

class WS {
  static OPEN = 1;
  static i: WS[] = [];
  url: string;
  readyState = 1;
  sent: string[] = [];
  onopen: ((e: unknown) => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;
  onclose: ((e: unknown) => void) | null = null;
  constructor(u: string) {
    this.url = u;
    WS.i.push(this);
  }
  send(d: string | ArrayBuffer | ArrayBufferView) {
    if (typeof d === "string") this.sent.push(d);
  }
  close() {
    this.readyState = 3;
    this.onclose?.({});
  }
  json(): Record<string, unknown>[] {
    return this.sent.map((s) => JSON.parse(s));
  }
}

const track = { enabled: true, kind: "audio", stopped: false, stop() { this.stopped = true; } };
const stream = { getTracks: () => [track], getAudioTracks: () => [track] };
const routes: string[] = [];
// A no-op peer factory: these tests never drive the mesh (callSession.test.ts
// does), so the peer only needs to exist without throwing.
const adapter = {
  createPeer: () => ({ onicecandidate: null, ontrack: null, oniceconnectionstatechange: null, iceConnectionState: "new", createOffer: async () => ({ type: "offer" }), setLocalDescription: async () => {}, setRemoteDescription: async () => {}, addIceCandidate: async () => {}, createAnswer: async () => ({ type: "answer" }), addTrack: () => ({}), close: () => {} }) as never,
  getLocalStream: async () => stream as never,
  playRemote: jest.fn(),
  stopRemote: jest.fn(),
  setRoute: async (r: string) => {
    routes.push(r);
  },
};

const created = {
  callId: "call-uuid-1",
  joinCode: "K7M2PQ",
  joinUrl: "https://arborfam-hub.web.app/call/K7M2PQ",
  selfLabel: "Speaker A",
  selfRole: "participant" as const,
  iceServers: [{ urls: ["stun:stun.l.google.com:19302"] }],
};
const api: CallApi = {
  create: jest.fn().mockResolvedValue(created),
  join: jest.fn().mockResolvedValue(created),
  end: jest.fn().mockResolvedValue(undefined),
  // Only the pre-flight screen calls this; the session hook never does.
  ice: jest.fn().mockResolvedValue({
    iceServers: created.iceServers,
    turnConfigured: false,
    credentialMode: "none",
    ttlSeconds: null,
  }),
};
const flush = () => new Promise((r) => setTimeout(r, 20));
const mockFetch = global.fetch as jest.Mock;

beforeEach(() => {
  WS.i = [];
  routes.length = 0;
  (globalThis as Record<string, unknown>).WebSocket = WS;
  (api.create as jest.Mock).mockReset().mockResolvedValue(created);
  (api.join as jest.Mock).mockReset().mockResolvedValue(created);
  (api.end as jest.Mock).mockReset().mockResolvedValue(undefined);
  mockFetch.mockReset().mockResolvedValue({ ok: true, status: 200, json: async () => ({}) });
  useDiagnosticsStore.setState({ capability: null, capabilityReason: null, lastSession: null, sending: false, lastSent: null });
  useAuthStore.setState({ user: { uid: "a-sage", email: "sage@example.com" } as never });
});

describe("useAudioStream — Call mode", () => {
  it("startCall: REST create, Call mode, speaker route, call_join {role} on open", async () => {
    const hook = await renderHook(() => useAudioStream({ callApi: api, makeRtcAdapter: () => adapter }));
    const result = hook.result;
    await act(async () => {
      await result.current.startCall(70, 20);
    });
    expect(api.create).toHaveBeenCalledTimes(1);
    expect(result.current.sessionMode).toBe("call");
    expect(result.current.sessionActive).toBe(true);
    expect(result.current.call).toMatchObject({ status: "waiting", callId: "call-uuid-1", joinCode: "K7M2PQ", selfRole: "participant" });
    expect(routes).toEqual(["speaker"]);
    const ws = WS.i[0];
    expect(ws.url).toMatch(/\/ws\/session\/call-call-uuid-1$/);
    await act(() => {
      ws.onopen?.({});
    });
    const frames = ws.json();
    expect(frames[0]).toMatchObject({ type: "config", empathy_slider: 70, interject_level: 20 });
    expect(frames[1]).toEqual({ type: "call_join", call_id: "call-uuid-1", role: "participant", join_code: "K7M2PQ", display_name: "sage" });
    expect(api.create).toHaveBeenCalledWith({ displayName: "sage", maxParticipants: 3 });
    // No TURN in the list: the screen warns (carrier NAT).
    expect(result.current.call.hasTurn).toBe(false);
  });

  it("joinCall passes the invite role and announces it (therapist observer)", async () => {
    const hook = await renderHook(() => useAudioStream({ callApi: api, makeRtcAdapter: () => adapter }));
    const result = hook.result;
    await act(async () => {
      await result.current.joinCall("K7M2PQ", 50, 0, "therapist");
    });
    expect(api.join).toHaveBeenCalledWith("K7M2PQ", "therapist", "sage");
    expect(result.current.call.selfRole).toBe("therapist");
    const ws = WS.i[0];
    await act(() => {
      ws.onopen?.({});
    });
    expect(ws.json().find((f) => f.type === "call_join")).toMatchObject({ type: "call_join", call_id: "call-uuid-1", role: "therapist" });
  });

  it("relayed rows are keyed by seq: a corrected first turn replaces its line, never duplicates it", async () => {
    // The server merges each member's turns under a `seq`. A member's FIRST
    // utterance can reach us as the server transcriber's copy (no text_tone,
    // no sender clock) before that phone's own turn_local latches it
    // local-first; the phone's report then arrives under the SAME seq tagged
    // `replaces_seq`. It must correct the line in place — appending would
    // show the sentence twice (server/calls.py push_turn).
    const hook = await renderHook(() => useAudioStream({ callApi: api, makeRtcAdapter: () => adapter }));
    const result = hook.result;
    await act(async () => {
      await result.current.startCall(50);
    });
    const ws = WS.i[0];
    await act(() => {
      ws.onopen?.({});
    });
    const row = (seq: number, text: string, extra: Record<string, unknown> = {}) => ({
      type: "transcript", session_id: "call-call-uuid-1", call_id: "call-uuid-1",
      speaker: "Speaker B", display_name: "Dad", role: "participant", participant_uid: "uid-b",
      is_self: false, seq, replaces_seq: null, text, start_time: 0, end_time: 4.1,
      local_start_time: null, local_end_time: null, text_tone: null, prosody: null,
      ...extra,
    });
    // The cloud copy of Dad's opener, then a turn his phone reported itself.
    await act(() => {
      ws.onmessage?.({ data: JSON.stringify(row(1, "hey i looked at the credit card statement")) });
      ws.onmessage?.({ data: JSON.stringify(row(2, "Sure. I know it's higher.", { start_time: 4.5, end_time: 9.7, text_tone: { label: "neutral", frustration: 10 } })) });
    });
    expect(result.current.transcript.map((t) => t.text)).toEqual([
      "hey i looked at the credit card statement",
      "Sure. I know it's higher.",
    ]);
    // Dad's phone catches up on seq 1 — LATER than seq 2 on the wire.
    await act(() => {
      ws.onmessage?.({
        data: JSON.stringify(row(1, "Hey, I looked at the credit card statement.", {
          replaces_seq: 1, local_start_time: 0, local_end_time: 4.1,
          text_tone: { label: "angry", frustration: 70 },
        })),
      });
    });
    // One line per turn, corrected in place, order untouched.
    expect(result.current.transcript.map((t) => t.text)).toEqual([
      "Hey, I looked at the credit card statement.",
      "Sure. I know it's higher.",
    ]);
    expect(result.current.transcript[0]).toMatchObject({ speaker: "Dad", speakerId: "Speaker B", callSeq: 1 });
    // The corrected row's tone reaches the scoreboard (the cloud copy had none).
    expect(result.current.scoreboard?.people.find((p) => p.speaker === "Speaker B")?.scoredTurns).toBe(2);
    // A bare re-delivery of a seq we already hold is deduped too.
    await act(() => {
      ws.onmessage?.({ data: JSON.stringify(row(2, "Sure. I know it's higher.", { start_time: 4.5, end_time: 9.7 })) });
    });
    expect(result.current.transcript).toHaveLength(2);
  });

  it("a failed create/join opens no session and says why", async () => {
    (api.join as jest.Mock).mockRejectedValueOnce(new Error("no call with that code (it may have ended)"));
    const hook = await renderHook(() => useAudioStream({ callApi: api, makeRtcAdapter: () => adapter }));
    const result = hook.result;
    await act(async () => {
      await result.current.joinCall("NOPE99", 50);
    });
    expect(result.current.call).toMatchObject({ status: "failed", error: "no call with that code (it may have ended)" });
    expect(result.current.sessionActive).toBe(false);
    expect(WS.i).toHaveLength(0);
  });

  it("a Call-mode session that ends with errors sends diagnostics automatically", async () => {
    const hook = await renderHook(() => useAudioStream({ callApi: api, makeRtcAdapter: () => adapter }));
    const result = hook.result;
    await act(async () => {
      await result.current.startCall(50);
    });
    const ws = WS.i[0];
    await act(() => {
      ws.onopen?.({});
    });
    await act(() => {
      ws.onmessage?.({ data: JSON.stringify({ type: "transcription_unavailable", reason: "Deepgram key missing" }) });
    });
    expect(result.current.transcriptionAvailable).toBe(false);
    await act(async () => {
      await result.current.stopSession();
      ws.onmessage?.({ data: JSON.stringify({ type: "session_complete" }) });
      await flush();
    });
    const store = useDiagnosticsStore.getState();
    expect(store.lastSession?.mode).toBe("call");
    expect(store.lastSession?.errors).toEqual(expect.arrayContaining([expect.stringContaining("Deepgram key missing")]));
    expect(store.lastSent).toMatchObject({ trigger: "auto", ok: true });
    const call = mockFetch.mock.calls.find((c) => String(c[0]).endsWith("/telemetry"));
    expect(call).toBeTruthy();
    const body = JSON.parse(call![1].body);
    expect(body.events[0].tag).toBe("client_diagnostics");
    expect(body.events[0].data.uid).toBe("a-sage");
    expect(body.events[0].data.last_session.transcriptionMessage).toBe("Deepgram key missing");
  });
});
