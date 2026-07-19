import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import ReplayScreen from "../src/screens/ReplayScreen";
import {
  getRecording,
  getRecordingMediaUrl,
  postReanalyze,
  getAnalyzeJob,
} from "../src/api/client";
import type { RecordingDetail } from "../src/api/client";

jest.mock("../src/api/client", () => ({
  getRecording: jest.fn(),
  getRecordingMediaUrl: jest.fn(),
  getRecordingSourceUrl: jest.fn(),
  patchRecordingSource: jest.fn(),
  patchRecordingTitle: jest.fn(),
  postReanalyze: jest.fn(),
  getAnalyzeJob: jest.fn(),
  // Keep SpeakerEnrollment quiet so these tests stay focused on the delta card.
  getVoiceProfile: jest.fn(() =>
    Promise.resolve({
      available: false,
      storage_enabled: false,
      enrolled: false,
      enroll_count: 0,
    }),
  ),
  enrollVoice: jest.fn(),
}));
const mockGetRecording = getRecording as jest.Mock;
const mockGetMediaUrl = getRecordingMediaUrl as jest.Mock;
const mockReanalyze = postReanalyze as jest.Mock;
const mockGetJob = getAnalyzeJob as jest.Mock;

jest.mock("../src/utils/audioMode", () => ({
  __esModule: true,
  setPlaybackMode: jest.fn().mockResolvedValue(undefined),
  setRecordingMode: jest.fn().mockResolvedValue(undefined),
}));

jest.mock("../src/components/MediaPlayer", () => {
  const React = require("react");
  const { View } = require("react-native");
  const MockPlayer = React.forwardRef((props: Record<string, unknown>, ref: unknown) => {
    React.useImperativeHandle(ref, () => ({ seek: jest.fn() }));
    return React.createElement(View, { testID: "media-player" });
  });
  return { __esModule: true, default: MockPlayer };
});

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

function makeDetail(
  reportCards: Record<string, number>,
  peakHeat: number,
): RecordingDetail {
  return {
    id: "r1",
    created_at: "2026-07-01T10:00:00Z",
    filename: "talk.m4a",
    media_type: "audio",
    duration_seconds: 12,
    has_analysis: true,
    turns: [
      { speaker: "Alice", text: "one", start_time: 0, end_time: 6 },
      { speaker: "Bob", text: "two", start_time: 6, end_time: 12 },
    ],
    analysis: {
      per_turn: [
        { index: 0, speaker: "Alice", heat: 20, markers: [], is_spike: false, trigger_phrase: null },
        { index: 1, speaker: "Bob", heat: peakHeat, markers: [], is_spike: true, trigger_phrase: null },
      ],
      per_speaker: {},
      report_cards: Object.fromEntries(
        Object.entries(reportCards).map(([id, score]) => [
          id,
          { score, headline: "", did_well: "", work_on: "" },
        ]),
      ),
      dynamics: {
        coupling: { strength: null, leader: null, description: "" },
        deescalation: { who_first: null, follow_rate: null, description: "" },
        triggers: [],
        requests: [],
      },
      narrative: "",
    },
  };
}

async function mountWith(detail: RecordingDetail): Promise<renderer.ReactTestRenderer> {
  mockGetRecording.mockResolvedValueOnce(detail);
  mockGetMediaUrl.mockResolvedValueOnce({ url: "https://signed/x", expires_in: 600 });
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
  });
  await act(async () => {});
  return comp;
}

function jobDone() {
  mockReanalyze.mockResolvedValueOnce({ job_id: "job_1" });
  mockGetJob.mockResolvedValueOnce({
    job_id: "job_1",
    status: "done",
    created_at: "",
    updated_at: "",
    stage_started_at: null,
    progress_note: null,
    duration_seconds: null,
    error: null,
    result: { per_turn: [], per_speaker: {}, dynamics: {}, narrative: "", turns: [], stored: true, recording_id: "r1", storage_note: null },
  });
}

beforeEach(() => {
  mockGetRecording.mockReset();
  mockGetMediaUrl.mockReset();
  mockReanalyze.mockReset();
  mockGetJob.mockReset();
});

describe("ReplayScreen re-analyze delta summary", () => {
  it("shows what changed with a pulse marker when a score moved", async () => {
    const before = makeDetail({ Alice: 60, Bob: 55 }, 88);
    const after = makeDetail({ Alice: 68, Bob: 55 }, 80);
    const comp = await mountWith(before);

    jobDone();
    // The post-completion refetch returns the fresh (changed) analysis.
    mockGetRecording.mockResolvedValueOnce(after);
    mockGetMediaUrl.mockResolvedValueOnce({ url: "https://signed/y", expires_in: 600 });

    await act(async () => {
      queryId(comp, "reanalyze-button")!.props.onPress();
    });
    await act(async () => {});

    expect(queryId(comp, "reanalyze-summary")).toBeTruthy();
    // A genuine delta → the pulse marker appears.
    expect(queryId(comp, "reanalyze-pulse")).toBeTruthy();
    // Alice's score changed; Bob's didn't → only Alice gets a delta row.
    expect(queryId(comp, "reanalyze-delta-Alice")).toBeTruthy();
    expect(queryId(comp, "reanalyze-delta-Bob")).toBeNull();
    const json = JSON.stringify(comp.toJSON());
    expect(json).toContain("what changed");
    expect(json).toContain("+8");
    // Peak heat also moved (88 → 80) → the peak line renders.
    expect(queryId(comp, "reanalyze-peak-delta")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("says 'no change' plainly and shows no pulse when nothing moved", async () => {
    const same = makeDetail({ Alice: 60, Bob: 55 }, 88);
    const comp = await mountWith(same);

    jobDone();
    // Refetch returns an identical analysis.
    mockGetRecording.mockResolvedValueOnce(makeDetail({ Alice: 60, Bob: 55 }, 88));
    mockGetMediaUrl.mockResolvedValueOnce({ url: "https://signed/y", expires_in: 600 });

    await act(async () => {
      queryId(comp, "reanalyze-button")!.props.onPress();
    });
    await act(async () => {});

    expect(queryId(comp, "reanalyze-summary")).toBeTruthy();
    expect(queryId(comp, "reanalyze-pulse")).toBeNull();
    expect(JSON.stringify(comp.toJSON())).toContain("No change");
    act(() => comp.unmount());
  });
});
