/** ReplayScreen on a live session whose batch analysis / reflection is still
 *  running: it re-reads the recording every few seconds (bounded) and the
 *  reflection appears without a manual refresh. Mocks mirror
 *  ReplayScreenLive.test.tsx. */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import ReplayScreen from "../src/screens/ReplayScreen";
import { getRecording, getRecordingMediaUrl, postReflect } from "../src/api/client";
import type { RecordingDetail } from "../src/api/client";

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

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => typeof n.type === "string" && n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

const reflections = [
  { turn_index: 0, could_have_said: "Hey Mom — thanks for the message.", why: "Already warm.", tone_read: "warm" },
];

const base: RecordingDetail = {
  id: "e1",
  created_at: "2026-08-24T18:05:00+00:00",
  filename: "live-session",
  title: "Live session · speaker",
  media_type: "none",
  duration_seconds: 17.5,
  has_analysis: true,
  source: { type: "live", url: null },
  mode: "speaker",
  session_id: "s-1",
  manual_speaker_labels: {},
  turns: [
    { speaker: "Speaker A", text: "Hey Mom, I got your message.", start_time: 0, end_time: 2.5 },
    { speaker: "Speaker B", text: "You never call back.", start_time: 3, end_time: 5.5 },
  ],
  analysis: {
    per_turn: [],
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
      mode: "speaker",
      self_speaker: "Speaker A",
      tone_summary: null,
      could_have_said: null,
      analysis_status: "lite",
    },
  },
};

const full: RecordingDetail = {
  ...base,
  analysis: {
    ...base.analysis!,
    per_turn: [
      { index: 0, speaker: "Speaker A", heat: 15, markers: [], is_spike: false, trigger_phrase: null },
      { index: 1, speaker: "Speaker B", heat: 25, markers: [], is_spike: false, trigger_phrase: null },
    ],
    live: {
      ...base.analysis!.live,
      could_have_said: reflections,
      reflection: { reflected_at: "now", turns_hash: "h" },
      analysis_status: "full",
    },
  },
};

beforeEach(() => {
  mockGetRecording.mockReset();
  mockGetMediaUrl.mockReset();
  mockReflect.mockReset();
});

describe("ReplayScreen — reflection appears when ready", () => {
  it("polls a lite live session until the analysis + reflection land, then stops", async () => {
    jest.useFakeTimers();
    mockGetRecording.mockResolvedValueOnce(base).mockResolvedValueOnce(base).mockResolvedValue(full);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="e1" onBack={() => {}} />);
    });
    await act(async () => {});
    expect(mockGetRecording).toHaveBeenCalledTimes(1);
    expect(queryId(comp, "replay-could-have-said-0")).toBeNull();

    // First tick: still lite → keeps polling.
    await act(async () => {
      jest.advanceTimersByTime(5000);
    });
    await act(async () => {});
    expect(mockGetRecording).toHaveBeenCalledTimes(2);
    expect(queryId(comp, "replay-could-have-said-0")).toBeNull();

    // Second tick: full + reflection → rendered, polling stops.
    await act(async () => {
      jest.advanceTimersByTime(5000);
    });
    await act(async () => {});
    expect(mockGetRecording).toHaveBeenCalledTimes(3);
    expect(queryId(comp, "replay-could-have-said-0")).toBeTruthy();
    await act(async () => {
      jest.advanceTimersByTime(30000);
    });
    await act(async () => {});
    expect(mockGetRecording).toHaveBeenCalledTimes(3);
    expect(mockReflect).not.toHaveBeenCalled();
    act(() => comp.unmount());
    jest.useRealTimers();
  });

  it("does not poll an upload or a finished live session", async () => {
    jest.useFakeTimers();
    mockGetRecording.mockResolvedValue(full);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="e1" onBack={() => {}} />);
    });
    await act(async () => {});
    await act(async () => {
      jest.advanceTimersByTime(20000);
    });
    await act(async () => {});
    expect(mockGetRecording).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
    jest.useRealTimers();
  });
});
