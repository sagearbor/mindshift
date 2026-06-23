import { renderHook, act } from "@testing-library/react-native";
import { useAudioStream } from "../src/hooks/useAudioStream";

/**
 * Fake WebSocket capturing sent frames and letting tests drive server events.
 * Mirrors the subset of the WebSocket API the hook uses.
 */
class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];

  url: string;
  readyState = FakeWebSocket.OPEN;
  sent: string[] = [];
  onopen: ((e: unknown) => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;
  onclose: ((e: unknown) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.readyState = 3;
    this.onclose?.({});
  }

  // --- test helpers ---
  emitOpen() {
    this.onopen?.({});
  }
  emitServer(obj: unknown) {
    this.onmessage?.({ data: JSON.stringify(obj) });
  }
  sentJson() {
    return this.sent.map((s) => JSON.parse(s));
  }
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  // @ts-expect-error — install fake WebSocket for the hook under test
  global.WebSocket = FakeWebSocket;
});

async function startLiveSession(empathy = 50) {
  const hook = renderHook(() => useAudioStream());
  await act(async () => {
    await hook.result.current.startSession("sess-1", empathy);
  });
  const ws = FakeWebSocket.instances.at(-1)!;
  act(() => ws.emitOpen());
  return { hook, ws };
}

describe("useAudioStream — WebSocket protocol", () => {
  it("maps a suggestion event into transcript + suggestions", async () => {
    const { hook, ws } = await startLiveSession(75);

    act(() =>
      ws.emitServer({
        type: "suggestion",
        session_id: "sess-1",
        utterance_text: "You never listen to me.",
        speaker: "Speaker A",
        suggestions: ["I hear you.", "Tell me more.", "That sounds hard."],
        empathy_slider: 75,
        audio_b64: null,
      }),
    );

    const { transcript, suggestions } = hook.result.current;
    expect(transcript).toHaveLength(1);
    expect(transcript[0].speaker).toBe("Speaker A");
    expect(transcript[0].text).toBe("You never listen to me.");

    expect(suggestions).toHaveLength(3);
    expect(suggestions[0].text).toBe("I hear you.");
    // Tone reflects the empathy stance (75 -> empathetic), not a fabricated label.
    expect(suggestions[0].tone).toBe("empathetic");
  });

  it("surfaces transcription_unavailable instead of silently ignoring it", async () => {
    const { hook, ws } = await startLiveSession();

    act(() =>
      ws.emitServer({
        type: "transcription_unavailable",
        reason: "DEEPGRAM_API_KEY not set",
      }),
    );

    expect(hook.result.current.transcriptionAvailable).toBe(false);
    expect(hook.result.current.transcriptionMessage).toContain("DEEPGRAM");
  });

  it("sends empathy changes as a config message the server understands", async () => {
    const { hook, ws } = await startLiveSession(50);

    act(() => hook.result.current.sendEmpathyUpdate(90));

    const configMsgs = ws
      .sentJson()
      .filter((m) => m.type === "config" && m.empathy_slider === 90);
    expect(configMsgs).toHaveLength(1);
    // The old, server-rejected message types must not be sent.
    expect(ws.sentJson().some((m) => m.type === "empathy_update")).toBe(false);
  });

  it("sends initial empathy as config on connect", async () => {
    const { ws } = await startLiveSession(30);
    const configMsgs = ws
      .sentJson()
      .filter((m) => m.type === "config" && m.empathy_slider === 30);
    expect(configMsgs.length).toBeGreaterThanOrEqual(1);
  });
});
