/**
 * Send diagnostics (src/diagnostics/diagnostics.ts): the id format the owner
 * reads out, the latency summary, the /telemetry body shape the server's
 * additive `data` field receives, and the store's send outcome.
 */
import {
  DIAGNOSTICS_ID_RE,
  buildDiagnosticsPayload,
  collectDeviceInfo,
  newDiagnosticsId,
  sendDiagnostics,
  summarizeLatency,
  telemetryBody,
  useDiagnosticsStore,
  type SessionDiagnostics,
} from "../src/diagnostics/diagnostics";
import type { TurnLatency } from "../src/live/fastLoop";

const mockFetch = global.fetch as jest.Mock;

function lat(turn: number, toSpeakMs: number | null, provider: string, held = false): TurnLatency {
  return { turn, segmentEndMs: turn * 1000, prosodyMs: 1, speakerMs: 10, sttWaitMs: 100 + turn, llmMs: 400 + turn, toSpeakMs, provider, held };
}

function session(over: Partial<SessionDiagnostics> = {}): SessionDiagnostics {
  return {
    sessionId: "live-1",
    mode: "call",
    startedAt: "2026-08-25T10:00:00.000Z",
    endedAt: "2026-08-25T10:05:00.000Z",
    turns: 4,
    latency: summarizeLatency([lat(1, 900, "os"), lat(2, null, "cloud"), lat(3, 1200, "cloud", true)]),
    liveStatus: "On-device: Silero VAD · speaker-ID off · LLM os → cloud",
    onDevice: true,
    sttRestarts: 2,
    sttFailure: null,
    wsReconnects: 1,
    micError: null,
    transcriptionMessage: null,
    postStatus: "created",
    call: { status: "ended", iceRestarts: 1, error: null, connectedSeconds: 180 },
    errors: ["ws reconnects: 1", "call: 1 ICE restart(s)"],
    ...over,
  };
}

beforeEach(() => {
  mockFetch.mockReset();
  useDiagnosticsStore.setState({ capability: null, capabilityReason: null, lastSession: null, sending: false, lastSent: null });
});

describe("diagnostics", () => {
  it("mints readable ids (dx-XXXX-XXXX, no look-alike letters)", () => {
    for (let i = 0; i < 50; i++) expect(newDiagnosticsId()).toMatch(DIAGNOSTICS_ID_RE);
    expect(newDiagnosticsId(() => 0)).toBe("dx-AAAA-AAAA");
    expect(newDiagnosticsId(() => 0.999)).toBe("dx-9999-9999");
    expect(newDiagnosticsId()).not.toMatch(/[IO01]/);
  });

  it("summarizes the latency log: medians, p90, provider counts, held", () => {
    const s = summarizeLatency([lat(1, 900, "os"), lat(2, null, "cloud"), lat(3, 1200, "cloud", true)]);
    expect(s).toEqual({
      turns: 3,
      spoken: 2,
      medianToSpeakMs: 900,
      p90ToSpeakMs: 900,
      medianLlmMs: 402,
      medianSttWaitMs: 102,
      byProvider: { os: 1, cloud: 2 },
      byOutcome: {},
      outcomeSamples: {},
      held: 1,
    });
    expect(summarizeLatency([])).toMatchObject({ turns: 0, spoken: 0, medianToSpeakMs: null, byProvider: {} });
  });

  it("aggregates per-provider attempt outcomes (why a local rung didn't answer)", () => {
    const withAttempts = (turn: number, attempts: { provider: string; outcome: string }[]) => ({
      ...lat(turn, null, "cloud"),
      attempts,
    });
    const s = summarizeLatency([
      withAttempts(1, [{ provider: "os", outcome: "refused" }, { provider: "bundled", outcome: "unavailable" }, { provider: "cloud", outcome: "cloud" }]),
      withAttempts(2, [{ provider: "os", outcome: "refused" }, { provider: "cloud", outcome: "cloud" }]),
    ]);
    expect(s.byOutcome).toEqual({ "os:refused": 2, "bundled:unavailable": 1, "cloud:cloud": 2 });
  });

  it("reads the device model from Platform.constants (Android) and the UA on the web", () => {
    expect(collectDeviceInfo({ OS: "android", Version: 16, constants: { Model: "Pixel 10", Brand: "google" } }, null)).toEqual({
      platform: "android",
      osVersion: "16",
      model: "Pixel 10",
      userAgent: null,
    });
    expect(collectDeviceInfo({ OS: "web", constants: {} }, { userAgent: "Safari/iPhone" })).toMatchObject({
      platform: "web",
      model: null,
      userAgent: "Safari/iPhone",
    });
  });

  it("builds the /telemetry body: one client_diagnostics event carrying the payload as data", () => {
    const payload = buildDiagnosticsPayload({
      trigger: "auto",
      uid: "u1",
      email: "sage@example.com",
      capability: { vad: "silero", speakerId: { active: true, reason: "", enrolled: 2, model: null, droppedForModel: 0 }, llm: ["os", "cloud"] },
      capabilityReason: "on-device speech recognition available",
      lastSession: session(),
      id: "dx-TEST-TEST",
      now: () => new Date("2026-08-25T10:06:00.000Z"),
    });
    expect(payload.app).toMatchObject({ version: "1.14.0", build: "29", runtimeVersion: "1.14.0", updateId: null });
    const body = telemetryBody(payload);
    expect(body.device).toBe("phone:ios:u1");
    expect(body.app_version).toBe("1.14.0");
    expect(body.events).toHaveLength(1);
    const ev = body.events[0];
    expect(ev.tag).toBe("client_diagnostics");
    expect(ev.level).toBe("warn");
    expect(ev.message).toContain("dx-TEST-TEST");
    expect(ev.message).toContain("uid=u1");
    expect(ev.message).toContain("ws reconnects: 1");
    expect(ev.ts).toBe("2026-08-25T10:06:00.000Z");
    expect(ev.data).toBe(payload);
    expect(ev.data.last_session?.latency.byProvider).toEqual({ os: 1, cloud: 2 });
    // No errors → an info event.
    expect(telemetryBody({ ...payload, last_session: session({ errors: [] }) }).events[0].level).toBe("info");
  });

  it("POSTs to /telemetry and reports the outcome without throwing", async () => {
    const payload = buildDiagnosticsPayload({ trigger: "manual", uid: null, email: null, capability: null, capabilityReason: null, lastSession: null, id: "dx-ABCD-EFGH" });
    const fetchOk = jest.fn().mockResolvedValue({ ok: true, status: 200 });
    expect(await sendDiagnostics(payload, fetchOk)).toEqual({ ok: true, id: "dx-ABCD-EFGH" });
    const [url, init] = fetchOk.mock.calls[0];
    expect(url).toMatch(/\/telemetry$/);
    expect(JSON.parse(init.body).events[0].data.diagnostics_id).toBe("dx-ABCD-EFGH");
    const fetch500 = jest.fn().mockResolvedValue({ ok: false, status: 500 });
    expect(await sendDiagnostics(payload, fetch500)).toEqual({ ok: false, id: "dx-ABCD-EFGH", error: "server answered 500" });
    const fetchDown = jest.fn().mockRejectedValue(new Error("offline"));
    expect(await sendDiagnostics(payload, fetchDown)).toEqual({ ok: false, id: "dx-ABCD-EFGH", error: "offline" });
  });

  it("the store sends what it recorded and remembers the id", async () => {
    mockFetch.mockResolvedValue({ ok: true, status: 200 });
    useDiagnosticsStore.getState().recordSession(session());
    const outcome = await useDiagnosticsStore.getState().send("manual", { uid: "u1", email: "m@x" });
    expect(outcome.ok).toBe(true);
    expect(outcome.id).toMatch(DIAGNOSTICS_ID_RE);
    const state = useDiagnosticsStore.getState();
    expect(state.sending).toBe(false);
    expect(state.lastSent).toMatchObject({ id: outcome.id, ok: true, trigger: "manual", error: null });
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.events[0].data.last_session.sessionId).toBe("live-1");
    expect(body.events[0].data.uid).toBe("u1");
  });
});
