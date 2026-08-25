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

const created = { callId: "call-uuid-1", joinCode: "K7M2PQ", joinUrl: "https://arborfam-hub.web.app/call/K7M2PQ" };
const api: CallApi = {
  create: jest.fn().mockResolvedValue(created),
  join: jest.fn().mockResolvedValue(created),
  end: jest.fn().mockResolvedValue(undefined),
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
    expect(frames[1]).toEqual({ type: "call_join", call_id: "call-uuid-1", role: "participant" });
  });

  it("joinCall passes the invite role and announces it (therapist observer)", async () => {
    const hook = await renderHook(() => useAudioStream({ callApi: api, makeRtcAdapter: () => adapter }));
    const result = hook.result;
    await act(async () => {
      await result.current.joinCall("K7M2PQ", 50, 0, "therapist");
    });
    expect(api.join).toHaveBeenCalledWith("K7M2PQ", "therapist");
    expect(result.current.call.selfRole).toBe("therapist");
    const ws = WS.i[0];
    await act(() => {
      ws.onopen?.({});
    });
    expect(ws.json().find((f) => f.type === "call_join")).toEqual({ type: "call_join", call_id: "call-uuid-1", role: "therapist" });
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
