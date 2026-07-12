import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import ReplayScreen from "../src/screens/ReplayScreen";
import { getRecording, getRecordingMediaUrl } from "../src/api/client";
import type { RecordingDetail } from "../src/api/client";

// Mock the network fns.
jest.mock("../src/api/client", () => ({
  getRecording: jest.fn(),
  getRecordingMediaUrl: jest.fn(),
}));
const mockGetRecording = getRecording as jest.Mock;
const mockGetMediaUrl = getRecordingMediaUrl as jest.Mock;

// Mock the media player wholesale (per the brief): a host view that forwards a
// ref exposing a shared `seek` spy, so tap-to-seek can be asserted without
// touching expo-video. (Named `mockSeek` so jest's factory hoist allows the
// out-of-scope reference.)
const mockSeek = jest.fn();
jest.mock("../src/components/MediaPlayer", () => {
  const React = require("react");
  const { View } = require("react-native");
  const MockPlayer = React.forwardRef(
    (props: Record<string, unknown>, ref: unknown) => {
      React.useImperativeHandle(ref, () => ({ seek: mockSeek }));
      return React.createElement(View, { testID: "media-player" });
    },
  );
  return { __esModule: true, default: MockPlayer };
});

const detail: RecordingDetail = {
  id: "r1",
  created_at: "2026-07-01T10:00:00Z",
  filename: "kitchen-fight.m4a",
  media_type: "audio",
  duration_seconds: 12,
  has_analysis: true,
  turns: [
    { speaker: "Alice", text: "You never listen.", start_time: 0, end_time: 3 },
    { speaker: "Bob", text: "That's not fair.", start_time: 3, end_time: 6 },
    { speaker: "Alice", text: "I asked twice.", start_time: 6, end_time: 9 },
    { speaker: "Bob", text: "You always say that.", start_time: 9, end_time: 12 },
  ],
  analysis: {
    per_turn: [
      { index: 0, speaker: "Alice", heat: 20, markers: [], is_spike: false, trigger_phrase: null },
      { index: 1, speaker: "Bob", heat: 35, markers: [], is_spike: false, trigger_phrase: null },
      { index: 2, speaker: "Alice", heat: 45, markers: [], is_spike: false, trigger_phrase: null },
      { index: 3, speaker: "Bob", heat: 88, markers: ["contempt"], is_spike: true, trigger_phrase: "you always" },
    ],
    per_speaker: {},
    dynamics: {
      coupling: { strength: null, leader: null, description: "" },
      deescalation: { who_first: null, follow_rate: null, description: "" },
      triggers: [],
      requests: [],
    },
    narrative: "",
  },
};

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

beforeEach(() => {
  mockGetRecording.mockReset();
  mockGetMediaUrl.mockReset();
  mockSeek.mockReset();
});

describe("ReplayScreen", () => {
  it("fetches the recording + media URL and renders the player and heat chart", async () => {
    mockGetRecording.mockResolvedValueOnce(detail);
    mockGetMediaUrl.mockResolvedValueOnce({ url: "https://signed/x", expires_in: 600 });

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});

    expect(mockGetRecording).toHaveBeenCalledWith("r1");
    expect(mockGetMediaUrl).toHaveBeenCalledWith("r1");
    expect(queryId(comp, "replay-content")).toBeTruthy();
    expect(queryId(comp, "media-player")).toBeTruthy();
    expect(queryId(comp, "heat-chart")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("seeks the player when a chart point is tapped", async () => {
    mockGetRecording.mockResolvedValueOnce(detail);
    mockGetMediaUrl.mockResolvedValueOnce({ url: "https://signed/x", expires_in: 600 });

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});

    // Tap the scrubber cell for turn 3 (start_time 9) → player seeks there.
    const cell = comp.root.find((n) => n.props?.testID === "scrub-3");
    act(() => cell.props.onPress());
    expect(mockSeek).toHaveBeenCalledWith(9);
    act(() => comp.unmount());
  });

  it("shows the honest 503 storage message (no fabricated recording)", async () => {
    mockGetRecording.mockRejectedValueOnce(new Error("API error: 503"));
    mockGetMediaUrl.mockRejectedValueOnce(new Error("API error: 503"));

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});

    expect(queryId(comp, "replay-error")).toBeTruthy();
    const msg = queryId(comp, "replay-error-message");
    expect(msg).toBeTruthy();
    const text = JSON.stringify(comp.toJSON());
    expect(text).toContain("Replay storage");
    expect(text).toContain("enabled yet");
    // No content surface, no raw status leaked.
    expect(queryId(comp, "replay-content")).toBeNull();
    act(() => comp.unmount());
  });

  it("retries the fetch when the retry button is pressed", async () => {
    mockGetRecording.mockRejectedValueOnce(new Error("API error: 503"));
    mockGetMediaUrl.mockRejectedValueOnce(new Error("API error: 503"));

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});
    expect(queryId(comp, "replay-error")).toBeTruthy();

    // Second attempt succeeds.
    mockGetRecording.mockResolvedValueOnce(detail);
    mockGetMediaUrl.mockResolvedValueOnce({ url: "https://signed/x", expires_in: 600 });
    await act(async () => {
      comp.root.find((n) => n.props?.testID === "replay-retry").props.onPress();
    });
    await act(async () => {});

    expect(queryId(comp, "replay-content")).toBeTruthy();
    expect(mockGetRecording).toHaveBeenCalledTimes(2);
    act(() => comp.unmount());
  });
});
