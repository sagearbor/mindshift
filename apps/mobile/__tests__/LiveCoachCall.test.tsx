import React from "react";
import renderer, { act } from "react-test-renderer";

/**
 * Live Coach in Call mode (the screen + CallPanel over a mocked hook):
 * the mode chips (In person / Call), the pre-flight explainer, Start a
 * call / Join with code / Answer, the in-call header and controls.
 */
const mockUseAudioStream = jest.fn();
jest.mock("../src/hooks/useAudioStream", () => ({
  useAudioStream: () => mockUseAudioStream(),
}));
jest.mock("../src/api/liveSessions", () => ({
  listVoicePeople: jest.fn().mockResolvedValue({ people: [], error: null }),
}));
jest.mock("../src/api/therapist", () => ({
  getTherapistLink: jest.fn().mockResolvedValue({ linked: false }),
}));
const mockLoadLiveMode = jest.fn();
jest.mock("../src/live/modePrefs", () => ({
  loadLiveMode: () => mockLoadLiveMode(),
  saveLiveMode: jest.fn().mockResolvedValue(undefined),
}));
jest.mock("../src/api/client", () => ({ postShare: jest.fn(), listVoicePeople: jest.fn() }));
// The connectivity pre-flight: the screen fetches GET /calls/ice and gathers
// candidates against it. Both are faked here — the probe itself is covered
// candidate-by-candidate in iceProbe.test.ts.
const mockIceConfig = jest.fn();
jest.mock("../src/live/call/callApi", () => ({
  callApi: { ice: () => mockIceConfig() },
}));
const mockProbeIce = jest.fn();
jest.mock("../src/live/call/iceProbe", () => {
  const actual = jest.requireActual("../src/live/call/iceProbe");
  return { ...actual, probeIce: (...args: unknown[]) => mockProbeIce(...args) };
});

import LiveCoachScreen from "../src/screens/LiveCoachScreen";
import { LIVE_MODE_OPTIONS } from "../src/components/LiveModePicker";
import { IDLE_CALL_VIEW, type CallView } from "../src/live/call/types";
import CallPanel, { CALL_MODE_EXPLAINER, formatElapsed } from "../src/components/CallPanel";

/**
 * A safety net over this file's own explicit unmounts.
 *
 * Unmounting on the happy path is not enough: a test that fails, or returns
 * early, leaves its tree mounted, and a state update landing after Jest tears
 * the environment down throws "You are trying to `import` a file after the
 * Jest environment has been torn down" — a WORKER CRASH that takes unrelated
 * suites with it. Double-unmounting is harmless here (it is caught), so this
 * can sit alongside the explicit ones.
 */
const __trees: renderer.ReactTestRenderer[] = [];
function track<T extends renderer.ReactTestRenderer>(t: T): T {
  __trees.push(t);
  return t;
}
afterEach(async () => {
  await act(async () => {});
  act(() => {
    for (const t of __trees.splice(0)) {
      try {
        t.unmount();
      } catch {
        // Already unmounted by the test, or a throwing teardown — either way
        // it must not fail a test that otherwise passed.
      }
    }
  });
});


const base = {
  isRecording: false,
  sessionActive: false,
  transcript: [] as unknown[],
  suggestions: [] as unknown[],
  speakerLabel: "",
  selfSpeaker: "Speaker A",
  setSelfSpeaker: jest.fn(),
  connectionStatus: "idle" as const,
  transcriptionAvailable: true,
  transcriptionMessage: "",
  micError: "",
  speechAvailable: true,
  speechEnabled: false,
  setSpeechEnabled: jest.fn(),
  startSession: jest.fn(),
  stopSession: jest.fn(),
  sendEmpathyUpdate: jest.fn(),
  sendInterjectUpdate: jest.fn(),
  liveCapable: false,
  liveCapabilityReason: "n/a",
  liveMode: false,
  setLiveMode: jest.fn(),
  sessionMode: "call" as const,
  setSessionMode: jest.fn(),
  liveStatus: "",
  nudgeFlash: null,
  clearNudgeFlash: jest.fn(),
  latencySummary: "",
  toneFlags: [],
  preflight: null,
  runPreflight: jest.fn(),
  escalationCount: 0,
  sessionSummary: null,
  lastEpisode: null,
  call: IDLE_CALL_VIEW,
  startCall: jest.fn(),
  joinCall: jest.fn(),
  hangUp: jest.fn(),
  setCallMuted: jest.fn(),
  callRoute: "speaker" as const,
  setCallRoute: jest.fn(),
};

// Deep enough to settle the idle-screen effects that await twice (the ICE
// pre-flight fetches, then probes) — otherwise their setState lands after
// the test has ended and Jest never exits.
const flush = () => act(async () => { for (let i = 0; i < 8; i += 1) await Promise.resolve(); });
const text = (root: renderer.ReactTestRenderer) => JSON.stringify(root.toJSON());
/** Every string rendered inside one testID'd node (a pre-flight row). */
const rowText = (root: renderer.ReactTestRenderer, testID: string) =>
  root.root
    .findByProps({ testID })
    .findAll((n) => typeof n.props?.children === "string")
    .map((n) => n.props.children as string)
    .join(" ");

const STUN_ONLY = [{ urls: ["stun:stun.l.google.com:19302"] }];

beforeEach(() => {
  mockUseAudioStream.mockReturnValue({ ...base });
  mockLoadLiveMode.mockReset().mockResolvedValue("call");
  mockIceConfig.mockReset().mockResolvedValue({
    iceServers: STUN_ONLY,
    turnConfigured: false,
    credentialMode: "none",
    ttlSeconds: null,
  });
  mockProbeIce.mockReset().mockResolvedValue({
    host: true, srflx: false, relay: false, turnConfigured: false,
    types: ["host"], candidates: 1, verdict: "relay-needed",
    line: "relay needed — no TURN configured", reason: null, elapsedMs: 12,
  });
});

describe("Live Coach — Call mode", () => {
  it("offers five modes: the old speaker-phone is now 'In person', plus 'Call' and 'Journal'", () => {
    expect(LIVE_MODE_OPTIONS.map((o) => [o.mode, o.label])).toEqual([
      ["earpiece", "Earpiece"],
      ["speaker", "In person"],
      ["therapist", "Therapist"],
      ["call", "Call"],
      ["journal", "Journal"],
    ]);
  });

  it("idle: explains why the app places the call, and starts / joins one", async () => {
    const startCall = jest.fn();
    const joinCall = jest.fn();
    mockUseAudioStream.mockReturnValue({ ...base, startCall, joinCall });
    let root!: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<LiveCoachScreen />));
    });
    await flush();
    expect(root.root.findByProps({ testID: "call-explainer" }).props.children).toBe(CALL_MODE_EXPLAINER);
    expect(text(root)).toContain("MindShift places the call itself");
    // No Start Listening button in Call mode — the panel drives it.
    expect(root.root.findAllByProps({ testID: "mic-toggle" })).toHaveLength(0);
    // The identity chip is meaningless in a call (only "you" on this mic).
    expect(root.root.findAllByProps({ testID: "self-speaker-chip" })).toHaveLength(0);

    act(() => root.root.findByProps({ testID: "call-start" }).props.onPress());
    expect(startCall).toHaveBeenCalledWith(50, 0);

    const join = root.root.findByProps({ testID: "call-join" });
    expect(join.props.disabled).toBe(true);
    act(() => root.root.findByProps({ testID: "call-code-input" }).props.onChangeText("  K7M2PQ "));
    expect(root.root.findByProps({ testID: "call-join" }).props.disabled).toBe(false);
    act(() => root.root.findByProps({ testID: "call-join" }).props.onPress());
    expect(joinCall).toHaveBeenCalledWith("K7M2PQ", 50, 0);
  });

  /** Render idle and let the ICE fetch + probe (two awaits deep) settle,
   *  so nothing lands after the test ends. */
  const renderSettled = async () => {
    let root!: renderer.ReactTestRenderer;
    await act(async () => {
      root = track(renderer.create(<LiveCoachScreen />));
    });
    await act(async () => {
      for (let i = 0; i < 6; i += 1) await Promise.resolve();
    });
    return root;
  };

  it("pre-flight: one honest line about whether these two phones can reach each other", async () => {
    const root = await renderSettled();
    // It probed with EXACTLY the ice servers the server hands a call.
    expect(mockProbeIce).toHaveBeenCalledWith(STUN_ONLY);
    const row = rowText(root, "preflight-peer-connection");
    expect(row).toContain("Peer connection");
    expect(row).toContain("relay needed — no TURN configured");
    // Amber, not a green tick: this is the answer that kills a cellular demo.
    expect(row).toContain("✗");
    await act(async () => root.unmount());
  });

  it("pre-flight: a working relay reads as ready", async () => {
    mockProbeIce.mockResolvedValue({
      host: true, srflx: true, relay: true, turnConfigured: true,
      types: ["host", "srflx", "relay"], candidates: 3, verdict: "relay",
      line: "relay ready — a call connects even on carrier-grade NAT", reason: null, elapsedMs: 90,
    });
    const root = await renderSettled();
    const row = rowText(root, "preflight-peer-connection");
    expect(row).toContain("relay ready");
    expect(row).toContain("✓");
    await act(async () => root.unmount());
  });

  it("pre-flight: a check that couldn't run says why, instead of looking like a pass", async () => {
    // An older deployment has no /calls/ice.
    mockIceConfig.mockRejectedValue(new Error("this server has no in-app calls yet (404)"));
    const root = await renderSettled();
    const row = rowText(root, "preflight-peer-connection");
    expect(row).toContain("couldn't check on this device");
    expect(row).toContain("no in-app calls yet (404)");
    expect(row).toContain("✗");
    expect(mockProbeIce).not.toHaveBeenCalled();
    await act(async () => root.unmount());
  });

  it("pre-flight: the peer-connection row is Call mode only", async () => {
    mockUseAudioStream.mockReturnValue({ ...base, sessionMode: "speaker" });
    mockLoadLiveMode.mockResolvedValue("speaker");
    const root = await renderSettled();
    expect(root.root.findAllByProps({ testID: "preflight-peer-connection" })).toHaveLength(0);
    expect(mockProbeIce).not.toHaveBeenCalled();
    await act(async () => root.unmount());
  });

  it("an invite link opens Call mode with one Answer tap", async () => {
    const joinCall = jest.fn();
    const setSessionMode = jest.fn();
    const consumed = jest.fn();
    mockLoadLiveMode.mockResolvedValue("earpiece");
    mockUseAudioStream.mockReturnValue({ ...base, joinCall, setSessionMode });
    let root!: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<LiveCoachScreen joinCode="K7M2PQ" onJoinCodeConsumed={consumed} />));
    });
    await flush();
    // The invite overrides the remembered mode for this visit.
    expect(setSessionMode).toHaveBeenCalledWith("call");
    expect(setSessionMode).not.toHaveBeenCalledWith("earpiece");
    expect(root.root.findByProps({ testID: "call-invited" }).props.children.join("")).toBe(
      "You've been invited to a call (code K7M2PQ).",
    );
    expect(root.root.findAllByProps({ testID: "call-start" })).toHaveLength(0);
    act(() => root.root.findByProps({ testID: "call-answer" }).props.onPress());
    expect(joinCall).toHaveBeenCalledWith("K7M2PQ", 50, 0, "participant");
    expect(consumed).toHaveBeenCalled();
  });

  it("shows the failure reason honestly", async () => {
    mockUseAudioStream.mockReturnValue({
      ...base,
      call: { ...IDLE_CALL_VIEW, status: "failed", error: "couldn't start a call: this server has no in-app calls yet (404)" },
    });
    let root!: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<LiveCoachScreen />));
    });
    await flush();
    expect(root.root.findByProps({ testID: "call-error" }).props.children).toContain("no in-app calls yet");
  });

  it("in a call: header with name · status · timer, invite while waiting, mute / route / hang up", async () => {
    const hangUp = jest.fn();
    const setCallMuted = jest.fn();
    const setCallRoute = jest.fn();
    const waiting: CallView = { ...IDLE_CALL_VIEW, status: "waiting", callId: "c1", joinCode: "K7M2PQ", joinUrl: "https://arborfam-hub.web.app/call/K7M2PQ" };
    mockUseAudioStream.mockReturnValue({ ...base, sessionActive: true, isRecording: true, connectionStatus: "live", call: waiting, hangUp, setCallMuted, setCallRoute });
    let root!: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<LiveCoachScreen />));
    });
    await flush();
    expect(root.root.findByProps({ testID: "call-header" }).props.children).toBe("You · waiting for them");
    expect(root.root.findByProps({ testID: "call-invite-code" }).props.children.join("")).toBe("Code: K7M2PQ");
    expect(text(root)).toContain("https://arborfam-hub.web.app/call/K7M2PQ");
    // The bottom button hangs up in Call mode.
    expect(text(root)).toContain("Hang up");
    act(() => root.root.findByProps({ testID: "mic-toggle" }).props.onPress());
    expect(hangUp).toHaveBeenCalledTimes(1);

    const connected: CallView = { ...waiting, status: "connected", peers: [{ uid: "b", label: "Speaker B", displayName: "Mom", role: "participant", connected: true, iceRestarts: 0 }], connectedAt: Date.now() - 192_000 };
    mockUseAudioStream.mockReturnValue({ ...base, sessionActive: true, isRecording: true, connectionStatus: "live", call: connected, hangUp, setCallMuted, setCallRoute });
    act(() => root.update(<LiveCoachScreen />));
    expect(root.root.findByProps({ testID: "call-header" }).props.children).toMatch(/^You · connected · 03:1[0-9]$/);
    expect(root.root.findByProps({ testID: "call-peer-b" }).props.children.join("")).toBe("Mom · connected");
    expect(root.root.findAllByProps({ testID: "call-invite" })).toHaveLength(0);
    act(() => root.root.findByProps({ testID: "call-mute" }).props.onPress());
    expect(setCallMuted).toHaveBeenCalledWith(true);
    act(() => root.root.findByProps({ testID: "call-route" }).props.onPress());
    expect(setCallRoute).toHaveBeenCalledWith("earpiece");
    act(() => root.root.findByProps({ testID: "call-hangup" }).props.onPress());
    expect(hangUp).toHaveBeenCalledTimes(2);

    // The other person's turns render under their name like any transcript line.
    mockUseAudioStream.mockReturnValue({
      ...base,
      sessionActive: true,
      call: connected,
      transcript: [{ speaker: "Mom", text: "You never call me.", timestamp: 1 }],
    });
    act(() => root.update(<LiveCoachScreen />));
    expect(text(root)).toContain("You never call me.");
    expect(text(root)).toContain("Mom");
  });

  it("formats the timer and reports 'reconnecting'", () => {
    expect(formatElapsed(0)).toBe("00:00");
    expect(formatElapsed(192_400)).toBe("03:12");
    let root!: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(
        <CallPanel
          call={{ ...IDLE_CALL_VIEW, status: "reconnecting", peers: [{ uid: "b", label: "Speaker B", displayName: "Mom", role: "participant", connected: false, iceRestarts: 1 }], connectedAt: 5 }}
          sessionActive
          onStart={jest.fn()}
          onJoin={jest.fn()}
          onHangUp={jest.fn()}
          onToggleMute={jest.fn()}
          now={() => 65_005}
        />,
      ));
    });
    expect(root.root.findByProps({ testID: "call-header" }).props.children).toBe("You · reconnecting · 01:05");
    expect(root.root.findByProps({ testID: "call-reconnecting" })).toBeTruthy();
  });
});

/**
 * THE THERAPIST SEAT (server/calls.py). The observer who joins a call gets
 * NOTHING — no audio, no transcript, no coaching — until every coached
 * participant but the host approves. The rule for this screen: a pending
 * therapist is never silent. The host must not believe their therapist is
 * listening, the therapist must know she is not, and the person actually
 * being asked gets the two buttons.
 */
describe("Live Coach — the therapist seat", () => {
  const peer = (uid: string, displayName: string, role: "participant" | "therapist") => ({
    uid,
    label: `Speaker ${uid.toUpperCase()}`,
    displayName,
    role,
    connected: role !== "therapist",
    iceRestarts: 0,
  });
  const inCall = (over: Partial<CallView> = {}): CallView => ({
    ...IDLE_CALL_VIEW,
    status: "connected",
    callId: "c1",
    connectedAt: 1,
    peers: [peer("b", "Dad", "participant"), peer("c", "Mom", "therapist")],
    ...over,
  });
  const pending = (over: Partial<CallView> = {}) =>
    inCall({
      therapistApproval: "pending",
      therapistApprovalFrom: ["b"],
      therapistUid: "c",
      ...over,
    });
  /** Every string rendered in the tree, in order — RN splits interpolated
   *  sentences into fragments, so a substring check needs them joined. */
  const flat = (root: renderer.ReactTestRenderer) =>
    root.root
      .findAll((n) => typeof n.type === "string")
      .flatMap((n) => n.children)
      .filter((c): c is string => typeof c === "string")
      .join("");
  const panel = (call: CallView, props: Record<string, unknown> = {}) => {
    let root!: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(
        <CallPanel
          call={call}
          sessionActive
          onStart={jest.fn()}
          onJoin={jest.fn()}
          onHangUp={jest.fn()}
          onToggleMute={jest.fn()}
          now={() => 1}
          {...props}
        />,
      ));
    });
    return root;
  };

  it("asks the participant whose conversation it is, and says what is at stake", () => {
    const onApproveTherapist = jest.fn();
    const onDeclineTherapist = jest.fn();
    const root = panel(pending({ therapistNeedsYourApproval: true }), {
      onApproveTherapist,
      onDeclineTherapist,
    });
    const ask = root.root.findByProps({ testID: "therapist-consent-ask" });
    expect(ask).toBeTruthy();
    expect(flat(root)).toContain("Mom wants to listen in");
    // Plain words about what she would get, and that she has none of it yet.
    const disclosure = root.root.findByProps({ testID: "therapist-consent-disclosure" }).props
      .children as string;
    expect(disclosure).toContain("transcript, tone and coaching");
    expect(disclosure).toContain("they get none of it");
    act(() => root.root.findByProps({ testID: "therapist-approve" }).props.onPress());
    expect(onApproveTherapist).toHaveBeenCalledTimes(1);
    act(() => root.root.findByProps({ testID: "therapist-decline" }).props.onPress());
    expect(onDeclineTherapist).toHaveBeenCalledTimes(1);
  });

  it("the HOST sees she is waiting and cannot approve on anyone's behalf", () => {
    const root = panel(pending({ therapistNeedsYourApproval: false }), {
      onApproveTherapist: jest.fn(),
      onDeclineTherapist: jest.fn(),
    });
    expect(root.root.findAllByProps({ testID: "therapist-approve" })).toHaveLength(0);
    expect(root.root.findByProps({ testID: "therapist-consent-waiting" })).toBeTruthy();
    const t = flat(root);
    expect(t).toContain("Mom is waiting to be let in");
    expect(t).toContain("Dad");
    expect(t).toContain("can't hear the call yet");
  });

  it("the OBSERVER is told plainly that she is getting nothing", () => {
    const root = panel(
      pending({ selfRole: "therapist", peers: [peer("a", "Sage", "participant"), peer("b", "Dad", "participant")] }),
    );
    expect(root.root.findByProps({ testID: "therapist-consent-waiting-self" })).toBeTruthy();
    const t = flat(root);
    expect(t).toContain("Waiting to be let in");
    expect(t).toContain("Dad must approve you joining");
    expect(t).toContain("nothing of it reaches this screen");
  });

  it("the pending observer's own row never reads as 'connecting'", () => {
    const root = panel(pending({ therapistNeedsYourApproval: true }));
    expect(root.root.findByProps({ testID: "call-peer-c" }).props.children.join("")).toBe(
      "Mom · waiting to be let in · therapist",
    );
    // Approved, she is an ordinary peer again.
    const ok = panel(inCall({ therapistUid: "c" }));
    expect(ok.root.findByProps({ testID: "call-peer-c" }).props.children.join("")).toBe(
      "Mom · connecting · therapist",
    );
    expect(ok.root.findAllByProps({ testID: "therapist-consent-ask" })).toHaveLength(0);
    expect(ok.root.findAllByProps({ testID: "therapist-consent-waiting" })).toHaveLength(0);
  });

  it("falls back to the slot label for a guest, who has no email or name", () => {
    // A guest joins by code: the server's display_name falls through to the
    // slot label. Nothing on this screen may assume an email exists.
    const guest = pending({
      therapistNeedsYourApproval: false,
      peers: [
        { uid: "b", label: "Speaker B", displayName: "Speaker B", role: "participant", connected: true, iceRestarts: 0 },
        { uid: "c", label: "Speaker C", displayName: "Speaker C", role: "therapist", connected: false, iceRestarts: 0 },
      ],
    });
    const root = panel(guest);
    const t = flat(root);
    expect(t).toContain("Speaker C is waiting to be let in");
    expect(t).toContain("Speaker B");
    expect(t).not.toContain("undefined");
  });

  it("names the approver honestly when they are not on our roster", () => {
    const root = panel(pending({ therapistApprovalFrom: ["someone-we-cannot-name"], peers: [peer("c", "Mom", "therapist")] }));
    const t = flat(root);
    expect(t).toContain("the other participant");
    // Never a raw uid on screen.
    expect(t).not.toContain("someone-we-cannot-name");
  });

  it("surfaces a failed approve instead of pretending the tap worked", () => {
    const root = panel(pending({ therapistNeedsYourApproval: true, error: "couldn't approve the therapist: HTTP 500" }));
    expect(root.root.findByProps({ testID: "call-active-error" }).props.children).toBe(
      "couldn't approve the therapist: HTTP 500",
    );
    // Still asking — the seat did not quietly become approved.
    expect(root.root.findByProps({ testID: "therapist-consent-ask" })).toBeTruthy();
  });

  /**
   * Adversarial review 2026-09-23. The server lets a coached participant
   * remove the observer at ANY time: `Call.decline_therapist_seat` checks
   * only that the call is live, that the caller is a coached participant,
   * and that a therapist is seated — it is NOT gated on `pending`. This
   * screen only ever offers that control inside
   * `therapistPending && therapistNeedsYourApproval`, so:
   *
   *  - once the seat is approved, nobody can change their mind; the observer
   *    listens for the rest of the call and is granted a permanent copy of
   *    every participant's episode at the end (Call._persist_episodes);
   *  - the HOST is never in `therapist_approval_from` (approvers_required()
   *    excludes them), so the host never sees the control at all — including
   *    the case that matters most: an "Invite my therapist" link forwarded to
   *    a stranger who takes the seat while the host is alone, where
   *    approvers_required() is empty and the seat is auto-approved with
   *    nobody asked. The host's only exit is ending the call for everyone.
   */
  it("an approved observer can still be removed — consent is revocable", () => {
    const onDeclineTherapist = jest.fn();
    const root = panel(inCall({ therapistUid: "c" }), { onDeclineTherapist });
    // Behavioural, not a node count: findAllByProps returns the composite AND
    // its host descendants for a TouchableOpacity, so a length assertion says
    // more about react-test-renderer than about the product.
    act(() => root.root.findByProps({ testID: "therapist-decline" }).props.onPress());
    expect(onDeclineTherapist).toHaveBeenCalledTimes(1);
  });

  it("the HOST can remove an observer they never approved", () => {
    const onDeclineTherapist = jest.fn();
    const root = panel(pending({ therapistNeedsYourApproval: false }), {
      onDeclineTherapist,
    });
    act(() => root.root.findByProps({ testID: "therapist-decline" }).props.onPress());
    expect(onDeclineTherapist).toHaveBeenCalledTimes(1);
  });

  it("an auto-approved seat asks nobody (standing `live` consent)", () => {
    const root = panel(inCall({ therapistUid: "c", therapistAutoApproved: true }));
    expect(root.root.findAllByProps({ testID: "therapist-consent-ask" })).toHaveLength(0);
    expect(root.root.findAllByProps({ testID: "therapist-consent-waiting" })).toHaveLength(0);
  });

  it("the screen hands the panel the hook's approve/decline", async () => {
    const approveTherapist = jest.fn();
    const declineTherapist = jest.fn();
    mockUseAudioStream.mockReturnValue({
      ...base,
      sessionActive: true,
      isRecording: true,
      connectionStatus: "live",
      call: pending({ therapistNeedsYourApproval: true }),
      approveTherapist,
      declineTherapist,
    });
    let root!: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<LiveCoachScreen />));
    });
    await flush();
    act(() => root.root.findByProps({ testID: "therapist-approve" }).props.onPress());
    expect(approveTherapist).toHaveBeenCalledTimes(1);
    act(() => root.root.findByProps({ testID: "therapist-decline" }).props.onPress());
    expect(declineTherapist).toHaveBeenCalledTimes(1);
  });
});
