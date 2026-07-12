import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import ReplayScreen from "../src/screens/ReplayScreen";
import {
  getRecording,
  getRecordingMediaUrl,
  getRecordingSourceUrl,
} from "../src/api/client";
import type { RecordingDetail } from "../src/api/client";

// Mock the network fns.
jest.mock("../src/api/client", () => ({
  getRecording: jest.fn(),
  getRecordingMediaUrl: jest.fn(),
  getRecordingSourceUrl: jest.fn(),
}));
const mockGetRecording = getRecording as jest.Mock;
const mockGetMediaUrl = getRecordingMediaUrl as jest.Mock;
const mockGetSourceUrl = getRecordingSourceUrl as jest.Mock;

// Mock the media player wholesale (per the brief): a host view that forwards a
// ref exposing a shared `seek` spy, so tap-to-seek can be asserted without
// touching expo-video. It also records the latest `uri`/`onError` props on
// `mockPlayerProps` so tests can assert which URL the player received and can
// drive the player's error callback. (Both names are `mock`-prefixed so jest's
// factory hoist allows the out-of-scope references.)
const mockSeek = jest.fn();
// `any` (not a typed shape) so the factory can assign props with no in-factory
// type cast — jest's hoist guard rejects a cast whose parameter name (`message`)
// reads as an out-of-scope variable.
const mockPlayerProps: Record<string, any> = {};
jest.mock("../src/components/MediaPlayer", () => {
  const React = require("react");
  const { View } = require("react-native");
  const MockPlayer = React.forwardRef(
    (props: Record<string, unknown>, ref: unknown) => {
      React.useImperativeHandle(ref, () => ({ seek: mockSeek }));
      mockPlayerProps.uri = props.uri;
      mockPlayerProps.onError = props.onError;
      return React.createElement(View, {
        testID: "media-player",
        uri: props.uri,
      });
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
  mockGetSourceUrl.mockReset();
  mockSeek.mockReset();
  mockPlayerProps.uri = undefined;
  mockPlayerProps.onError = undefined;
});

// A link-sourced recording: the user linked their own hosted original.
const linkDetail = {
  ...detail,
  source: { type: "link", url: "https://photos.app.goo.gl/abc" },
};
// An explicit upload source (older servers omit `source` — `detail` covers that).
const uploadDetail = {
  ...detail,
  source: { type: "upload", url: null },
};

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
    // Detail fetch fails first → error (the derivative fetch isn't reached).
    mockGetRecording.mockRejectedValueOnce(new Error("API error: 503"));

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

  it("streams the HD linked source and shows the HD badge for a link-sourced recording", async () => {
    mockGetRecording.mockResolvedValueOnce(linkDetail);
    mockGetSourceUrl.mockResolvedValueOnce({
      url: "https://cdn.example/hd=dv",
      content_type: "video/mp4",
      expires_hint: "may expire; refetch on failure",
    });

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});

    // Resolved the linked source, NOT the derivative.
    expect(mockGetSourceUrl).toHaveBeenCalledWith("r1");
    expect(mockGetMediaUrl).not.toHaveBeenCalled();
    // Player received the remote HD URL; badge shown, no fallback note.
    expect(mockPlayerProps.uri).toBe("https://cdn.example/hd=dv");
    expect(queryId(comp, "hd-badge")).toBeTruthy();
    expect(queryId(comp, "source-fallback-note")).toBeNull();
    act(() => comp.unmount());
  });

  it("falls back to the stored derivative with a note when the linked source won't resolve", async () => {
    mockGetRecording.mockResolvedValueOnce(linkDetail);
    mockGetSourceUrl.mockRejectedValueOnce(new Error("API error: 502"));
    mockGetMediaUrl.mockResolvedValueOnce({
      url: "https://signed.example/deriv",
      expires_in: 600,
    });

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});

    expect(mockGetSourceUrl).toHaveBeenCalledWith("r1");
    expect(mockGetMediaUrl).toHaveBeenCalledWith("r1");
    // Player got the derivative URL; note shown, no HD badge; content still renders.
    expect(mockPlayerProps.uri).toBe("https://signed.example/deriv");
    expect(queryId(comp, "hd-badge")).toBeNull();
    expect(queryId(comp, "source-fallback-note")).toBeTruthy();
    expect(queryId(comp, "replay-content")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("falls back to the derivative when the player errors on the remote HD stream", async () => {
    mockGetRecording.mockResolvedValueOnce(linkDetail);
    mockGetSourceUrl.mockResolvedValueOnce({
      url: "https://cdn.example/hd=dv",
      content_type: "video/mp4",
      expires_hint: "may expire; refetch on failure",
    });
    // Derivative resolved only once the player reports the remote stream failed.
    mockGetMediaUrl.mockResolvedValueOnce({
      url: "https://signed.example/deriv",
      expires_in: 600,
    });

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});

    // Starts in HD on the remote URL.
    expect(queryId(comp, "hd-badge")).toBeTruthy();
    expect(mockPlayerProps.uri).toBe("https://cdn.example/hd=dv");
    expect(mockGetMediaUrl).not.toHaveBeenCalled();

    // Player errors on the remote stream → automatic fallback.
    await act(async () => {
      mockPlayerProps.onError?.("decode failed");
    });
    await act(async () => {});

    expect(mockGetMediaUrl).toHaveBeenCalledWith("r1");
    expect(mockPlayerProps.uri).toBe("https://signed.example/deriv");
    expect(queryId(comp, "hd-badge")).toBeNull();
    expect(queryId(comp, "source-fallback-note")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("uses the derivative-only path for an upload-sourced recording (no source_url, no badge)", async () => {
    mockGetRecording.mockResolvedValueOnce(uploadDetail);
    mockGetMediaUrl.mockResolvedValueOnce({
      url: "https://signed.example/deriv",
      expires_in: 600,
    });

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});

    expect(mockGetSourceUrl).not.toHaveBeenCalled();
    expect(mockGetMediaUrl).toHaveBeenCalledWith("r1");
    expect(mockPlayerProps.uri).toBe("https://signed.example/deriv");
    expect(queryId(comp, "hd-badge")).toBeNull();
    expect(queryId(comp, "source-fallback-note")).toBeNull();
    act(() => comp.unmount());
  });
});
