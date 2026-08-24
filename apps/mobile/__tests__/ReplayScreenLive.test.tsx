/** ReplayScreen opening a Track 2 LIVE session (media_type "none" — no
 *  audio on the server) and the "What you could have said" reflection:
 *  no media fetch, no player, the live badge, the tone card, the cached
 *  reflection when present, and the on-demand reflect flow with honest
 *  error copy. Mocks mirror ReplayScreen.test.tsx. */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import ReplayScreen from "../src/screens/ReplayScreen";
import {
  getRecording,
  getRecordingMediaUrl,
  postReflect,
} from "../src/api/client";
import type { RecordingDetail, ToneSummary } from "../src/api/client";

jest.mock("../src/api/client", () => ({
  getRecording: jest.fn(),
  getRecordingMediaUrl: jest.fn(),
  getRecordingSourceUrl: jest.fn(),
  patchRecordingSource: jest.fn(),
  patchRecordingTitle: jest.fn(),
  patchSpeakerLabels: jest.fn(),
  postReanalyze: jest.fn(),
  getAnalyzeJob: jest.fn(),
  postReflect: jest.fn(),
  postShare: jest.fn(),
  deleteShare: jest.fn(),
  getVoiceProfile: jest.fn(() =>
    Promise.resolve({ available: false, storage_enabled: false, enrolled: false, enroll_count: 0 }),
  ),
  enrollVoice: jest.fn(),
}));
const mockGetRecording = getRecording as jest.Mock;
const mockGetMediaUrl = getRecordingMediaUrl as jest.Mock;
const mockReflect = postReflect as jest.Mock;

jest.mock("../src/utils/audioMode", () => ({
  __esModule: true,
  setPlaybackMode: jest.fn().mockResolvedValue(undefined),
  setRecordingMode: jest.fn().mockResolvedValue(undefined),
}));
jest.mock("../src/utils/mediaCache", () => ({
  __esModule: true,
  getCachedMediaUri: jest.fn(() => null),
  cacheMediaInBackground: jest.fn(),
}));
jest.mock("../src/components/MediaPlayer", () => {
  const React = require("react");
  const { View } = require("react-native");
  const MockPlayer = React.forwardRef((props: Record<string, unknown>, ref: unknown) => {
    React.useImperativeHandle(ref, () => ({ seek: jest.fn(), play: jest.fn(), pause: jest.fn() }));
    return React.createElement(View, { testID: "media-player", uri: props.uri });
  });
  return { __esModule: true, default: MockPlayer };
});

/** HOST nodes only (a component that renders null still has a fiber with
 *  its testID prop — matching it would make "renders nothing" untestable). */
function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll(
    (n) => typeof n.type === "string" && n.props?.testID === id,
  );
  return found.length > 0 ? found[0] : null;
}

/** Fire a testID'd pressable's onPress — the handler lives on the composite
 *  (TouchableOpacity) node, not the host View queryId returns. */
function press(comp: renderer.ReactTestRenderer, id: string): void {
  const found = comp.root.findAll(
    (n) => n.props?.testID === id && typeof n.props?.onPress === "function",
  );
  if (found.length === 0) throw new Error(`no pressable with testID ${id}`);
  found[0].props.onPress();
}

/** All rendered text under a node, joined — RN splits a Text's children
 *  into string fragments, so a substring check needs them joined. */
function textOf(node: ReactTestInstance): string {
  return node
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");
}

const toneSummary: ToneSummary = {
  self_speaker: "Speaker A",
  self: { turns: 3, scored_turns: 3, labels: { warm: 1, frustrated: 1, defensive: 1 },
          mean: { warmth: 50 }, escalation_turns: [2, 4], escalation_count: 2 },
  audio: null,
  audio_tone_surfaced: false,
  people: [
    { speaker: "Speaker B", person_id: "p-mom", display_name: "Mom", their_turns: 3,
      self_turns: 3, turns: 3, scored_turns: 3, labels: { warm: 1, frustrated: 1, defensive: 1 },
      mean: {}, escalation_turns: [2, 4], escalation_count: 2 },
  ],
};

const reflections = [
  { turn_index: 0, could_have_said: "Hey Mom — thanks for the message.", why: "Already warm.", tone_read: "warm" },
  { turn_index: 2, could_have_said: "I hear you. Work swallowed me — I'm sorry.", why: "Owns it.", tone_read: "defensive" },
];

const liveDetail: RecordingDetail = {
  id: "e1",
  created_at: "2026-08-24T18:05:00+00:00",
  filename: "live-session",
  title: "Live session · earpiece",
  media_type: "none",
  duration_seconds: 17.5,
  has_analysis: true,
  source: { type: "live", url: null },
  mode: "earpiece",
  session_id: "s-1",
  manual_speaker_labels: {},
  turns: [
    { speaker: "Speaker A", text: "Hey Mom, I got your message.", start_time: 0, end_time: 2.5 },
    { speaker: "Speaker B", text: "You never call back.", start_time: 3, end_time: 5.5 },
    { speaker: "Speaker A", text: "I was working, I told you that.", start_time: 6, end_time: 8.5 },
  ],
  analysis: {
    per_turn: [
      { index: 0, speaker: "Speaker A", heat: 15, markers: [], is_spike: false, trigger_phrase: null },
      { index: 1, speaker: "Speaker B", heat: 25, markers: [], is_spike: false, trigger_phrase: null },
      { index: 2, speaker: "Speaker A", heat: 35, markers: [], is_spike: false, trigger_phrase: null },
    ],
    per_speaker: {},
    dynamics: {
      coupling: { strength: null, leader: null, description: "" },
      deescalation: { who_first: null, follow_rate: null, description: "" },
      triggers: [],
      requests: [],
    },
    narrative: "",
    speaker_labels: {
      "Speaker A": { display_label: "You", label_source: "enrolled" },
      "Speaker B": { display_label: "Mom", label_source: "enrolled" },
    },
    live: {
      mode: "earpiece",
      self_speaker: "Speaker A",
      tone_summary: toneSummary,
      could_have_said: reflections,
      analysis_status: "full",
    },
  },
};

async function render(id = "e1") {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<ReplayScreen recordingId={id} onBack={() => {}} />);
  });
  await act(async () => {});
  return comp;
}

beforeEach(() => {
  mockGetRecording.mockReset();
  mockGetMediaUrl.mockReset();
  mockReflect.mockReset();
});

describe("ReplayScreen — live session", () => {
  it("renders without a player and never asks for media", async () => {
    mockGetRecording.mockResolvedValueOnce(liveDetail);
    const comp = await render();

    expect(mockGetMediaUrl).not.toHaveBeenCalled();
    expect(queryId(comp, "replay-content")).toBeTruthy();
    expect(queryId(comp, "media-player")).toBeNull();
    expect(queryId(comp, "replay-no-media")).toBeTruthy();
    expect(textOf(queryId(comp, "replay-live-badge")!)).toContain(
      "Live session · Earpiece",
    );
    // The heat chart still renders from the batch analysis (no seek hint —
    // there is nothing to seek in).
    expect(textOf(comp.root)).toContain("Heat over the conversation");
    expect(textOf(comp.root)).not.toContain("Tap a dash to jump there");
    // No stored audio → no attach / re-analyze / enroll affordances.
    expect(queryId(comp, "attach-source-button")).toBeNull();
    expect(queryId(comp, "reanalyze-button")).toBeNull();
    expect(queryId(comp, "try-hd-button")).toBeNull();
    // The tone card and the cached reflection render from the detail read.
    expect(queryId(comp, "replay-tone-summary")).toBeTruthy();
    expect(queryId(comp, "replay-tone-summary-person-Speaker B")).toBeTruthy();
    expect(queryId(comp, "replay-could-have-said-0")).toBeTruthy();
    expect(queryId(comp, "replay-could-have-said-2")).toBeTruthy();
    const tree = textOf(comp.root);
    expect(tree).toContain("“Hey Mom, I got your message.”");
    expect(tree).toContain("I hear you. Work swallowed me — I'm sorry.");
    // Cached → no button, no LLM call.
    expect(queryId(comp, "replay-reflect-button")).toBeNull();
    expect(mockReflect).not.toHaveBeenCalled();
    act(() => comp.unmount());
  });

  it("says the heat analysis is pending for a lite live session", async () => {
    mockGetRecording.mockResolvedValueOnce({
      ...liveDetail,
      analysis: {
        ...liveDetail.analysis!,
        per_turn: [],
        live: { ...liveDetail.analysis!.live, analysis_status: "lite", could_have_said: null },
      },
    });
    const comp = await render();
    expect(textOf(queryId(comp, "replay-live-badge")!)).toContain(
      "heat analysis pending",
    );
    expect(textOf(queryId(comp, "replay-no-analysis")!)).toContain(
      "still running",
    );
    act(() => comp.unmount());
  });

  it("reflects on demand and renders the result", async () => {
    mockGetRecording.mockResolvedValueOnce({
      ...liveDetail,
      analysis: { ...liveDetail.analysis!, live: { ...liveDetail.analysis!.live, could_have_said: null } },
    });
    mockReflect.mockResolvedValueOnce({
      episode_id: "e1", self_speaker: "Speaker A", could_have_said: reflections,
      cached: false, reflected_at: "2026-08-24T18:10:00+00:00",
    });
    const comp = await render();
    expect(queryId(comp, "replay-could-have-said")).toBeNull();
    const button = queryId(comp, "replay-reflect-button");
    expect(button).toBeTruthy();

    await act(async () => press(comp, "replay-reflect-button"));
    await act(async () => {});
    expect(mockReflect).toHaveBeenCalledWith("e1");
    expect(queryId(comp, "replay-could-have-said-2")).toBeTruthy();
    expect(queryId(comp, "replay-reflect-button")).toBeNull();
    act(() => comp.unmount());
  });

  it("explains a 422 (nobody identified as you) honestly", async () => {
    mockGetRecording.mockResolvedValueOnce({
      ...liveDetail,
      analysis: { ...liveDetail.analysis!, live: { ...liveDetail.analysis!.live, could_have_said: null } },
    });
    mockReflect.mockRejectedValueOnce(new Error("API error: 422"));
    const comp = await render();
    await act(async () => press(comp, "replay-reflect-button"));
    await act(async () => {});
    const err = queryId(comp, "replay-reflect-error");
    expect(err).toBeTruthy();
    expect(textOf(err!)).toContain("identified as you");
    // The button stays so the user can retry after "This is me".
    expect(queryId(comp, "replay-reflect-button")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("an empty reflection is an honest 'nothing to add'", async () => {
    mockGetRecording.mockResolvedValueOnce({
      ...liveDetail,
      analysis: { ...liveDetail.analysis!, live: { ...liveDetail.analysis!.live, could_have_said: [] } },
    });
    const comp = await render();
    expect(queryId(comp, "replay-reflect-empty")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("a shared live session is read-only: no reflect button", async () => {
    mockGetRecording.mockResolvedValueOnce({
      ...liveDetail,
      shared: true,
      owner_email: "patient@example.com",
      analysis: { ...liveDetail.analysis!, live: { ...liveDetail.analysis!.live, could_have_said: null } },
    });
    const comp = await render();
    expect(queryId(comp, "replay-reflect-button")).toBeNull();
    expect(queryId(comp, "replay-reflect-shared")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("an ordinary upload still fetches media and shows the reflect button", async () => {
    mockGetRecording.mockResolvedValueOnce({
      ...liveDetail,
      media_type: "audio",
      source: { type: "upload", url: null },
      mode: null,
      analysis: { ...liveDetail.analysis!, live: undefined },
    });
    mockGetMediaUrl.mockResolvedValueOnce({ url: "https://signed/x", expires_in: 600 });
    const comp = await render();
    expect(mockGetMediaUrl).toHaveBeenCalledWith("e1");
    expect(queryId(comp, "media-player")).toBeTruthy();
    expect(queryId(comp, "replay-live-badge")).toBeNull();
    expect(queryId(comp, "replay-tone-summary")).toBeNull();
    expect(queryId(comp, "replay-reflect-button")).toBeTruthy();
    expect(queryId(comp, "reanalyze-button")).toBeTruthy();
    act(() => comp.unmount());
  });
});
