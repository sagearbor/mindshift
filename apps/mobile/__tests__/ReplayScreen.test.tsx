import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import ReplayScreen from "../src/screens/ReplayScreen";
import {
  getRecording,
  getRecordingMediaUrl,
  getRecordingSourceUrl,
  patchRecordingSource,
  patchRecordingTitle,
  patchSpeakerLabels,
  postReanalyze,
  getAnalyzeJob,
} from "../src/api/client";
import type { RecordingDetail } from "../src/api/client";
import { setPlaybackMode } from "../src/utils/audioMode";

// Mock the network fns.
jest.mock("../src/api/client", () => ({
  getRecording: jest.fn(),
  getRecordingMediaUrl: jest.fn(),
  getRecordingSourceUrl: jest.fn(),
  patchRecordingSource: jest.fn(),
  patchRecordingTitle: jest.fn(),
  patchSpeakerLabels: jest.fn(),
  postReanalyze: jest.fn(),
  getAnalyzeJob: jest.fn(),
  // SpeakerEnrollment (rendered inside ReplayScreen) checks voice-ID
  // availability on mount; default to "unavailable" so it renders nothing and
  // these tests stay focused on replay/HD behavior.
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
const mockGetSourceUrl = getRecordingSourceUrl as jest.Mock;
const mockPatchSource = patchRecordingSource as jest.Mock;
const mockPatchTitle = patchRecordingTitle as jest.Mock;
const mockPatchLabels = patchSpeakerLabels as jest.Mock;
const mockReanalyze = postReanalyze as jest.Mock;
const mockGetJob = getAnalyzeJob as jest.Mock;

// Mock the audio-session util: ReplayScreen sets a playback mode on mount so
// replay is audible even after Live Coach's recording session. We assert the
// call without touching a real native audio session.
jest.mock("../src/utils/audioMode", () => ({
  __esModule: true,
  setPlaybackMode: jest.fn().mockResolvedValue(undefined),
  setRecordingMode: jest.fn().mockResolvedValue(undefined),
}));
const mockSetPlaybackMode = setPlaybackMode as jest.Mock;

// Mock the media player wholesale (per the brief): a host view that forwards a
// ref exposing a shared `seek` spy, so tap-to-seek can be asserted without
// touching expo-video. It records the latest `uri`/`onError`/`onDurationChange`
// props on `mockPlayerProps` so tests can assert which URL the player received,
// drive its error callback, and simulate the media reporting a real duration.
const mockSeek = jest.fn();
const mockPlayerProps: Record<string, any> = {};
jest.mock("../src/components/MediaPlayer", () => {
  const React = require("react");
  const { View } = require("react-native");
  const MockPlayer = React.forwardRef(
    (props: Record<string, unknown>, ref: unknown) => {
      React.useImperativeHandle(ref, () => ({ seek: mockSeek }));
      mockPlayerProps.uri = props.uri;
      mockPlayerProps.onError = props.onError;
      mockPlayerProps.onDurationChange = props.onDurationChange;
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
  mockPatchSource.mockReset();
  mockPatchTitle.mockReset();
  mockPatchLabels.mockReset();
  mockSeek.mockReset();
  mockSetPlaybackMode.mockClear();
  mockReanalyze.mockReset();
  mockGetJob.mockReset();
  mockPlayerProps.uri = undefined;
  mockPlayerProps.onError = undefined;
  mockPlayerProps.onDurationChange = undefined;
});

// A link-sourced recording: the user linked their own hosted original. Uses a
// DIRECT/Drive-style link (not Google Photos) so the Try-HD opt-in is offered —
// Photos links are the one case where HD is impossible (see the dedicated test).
const linkDetail = {
  ...detail,
  source: { type: "link", url: "https://drive.google.com/file/d/abc/view" },
};
// A Google-Photos-sourced recording: HD replay is impossible (moov-at-end, no
// Range), so Try-HD must be suppressed in favor of an honest note.
const photosDetail = {
  ...detail,
  source: { type: "link", url: "https://photos.app.goo.gl/abc" },
};
// An explicit upload source (older servers omit `source` — `detail` covers that).
const uploadDetail = {
  ...detail,
  source: { type: "upload", url: null },
};

const LOAD_TIMEOUT_MS = 8000;

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

  it("sets a playback audio mode on mount (so replay is audible after Live Coach)", async () => {
    mockGetRecording.mockResolvedValueOnce(detail);
    mockGetMediaUrl.mockResolvedValueOnce({ url: "https://signed/x", expires_in: 600 });

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});

    expect(mockSetPlaybackMode).toHaveBeenCalled();
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

  it("plays the stored derivative first for a link-sourced recording and offers Try-HD (no auto HD, no badge)", async () => {
    mockGetRecording.mockResolvedValueOnce(linkDetail);
    mockGetMediaUrl.mockResolvedValueOnce({
      url: "https://signed.example/deriv",
      expires_in: 600,
    });

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});

    // Derivative-first: the reliable stored copy loads; the HD original is NOT
    // auto-fetched (that stream can buffer forever).
    expect(mockGetMediaUrl).toHaveBeenCalledWith("r1");
    expect(mockGetSourceUrl).not.toHaveBeenCalled();
    expect(mockPlayerProps.uri).toBe("https://signed.example/deriv");
    expect(queryId(comp, "hd-badge")).toBeNull();
    // The opt-in Try-HD affordance is offered (link source only).
    expect(queryId(comp, "try-hd-button")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("Try-HD switches the player to the resolved HD source and shows the badge", async () => {
    mockGetRecording.mockResolvedValueOnce(linkDetail);
    mockGetMediaUrl.mockResolvedValueOnce({
      url: "https://signed.example/deriv",
      expires_in: 600,
    });

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});

    expect(mockPlayerProps.uri).toBe("https://signed.example/deriv");

    // User opts into HD; source resolves and the player switches to it.
    mockGetSourceUrl.mockResolvedValueOnce({
      url: "https://cdn.example/hd=dv",
      content_type: "video/mp4",
      expires_hint: "may expire; refetch on failure",
    });
    await act(async () => {
      queryId(comp, "try-hd-button")!.props.onPress();
    });
    await act(async () => {});

    expect(mockGetSourceUrl).toHaveBeenCalledWith("r1");
    expect(mockPlayerProps.uri).toBe("https://cdn.example/hd=dv");
    expect(queryId(comp, "hd-badge")).toBeTruthy();
    // Now the escape hatch back to the stored copy is offered instead.
    expect(queryId(comp, "force-derivative")).toBeTruthy();
    expect(queryId(comp, "try-hd-button")).toBeNull();
    act(() => comp.unmount());
  });

  it("Back-to-stored (force-derivative) returns the player to the derivative from HD", async () => {
    mockGetRecording.mockResolvedValueOnce(linkDetail);
    mockGetMediaUrl
      .mockResolvedValueOnce({ url: "https://signed.example/deriv", expires_in: 600 })
      .mockResolvedValueOnce({ url: "https://signed.example/deriv2", expires_in: 600 });

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});

    // Go HD first.
    mockGetSourceUrl.mockResolvedValueOnce({
      url: "https://cdn.example/hd=dv",
      content_type: "video/mp4",
      expires_hint: "may expire",
    });
    await act(async () => {
      queryId(comp, "try-hd-button")!.props.onPress();
    });
    await act(async () => {});
    expect(mockPlayerProps.uri).toBe("https://cdn.example/hd=dv");

    // Tap "Back to stored copy" → derivative again, badge gone.
    await act(async () => {
      queryId(comp, "force-derivative")!.props.onPress();
    });
    await act(async () => {});

    expect(mockPlayerProps.uri).toBe("https://signed.example/deriv2");
    expect(queryId(comp, "hd-badge")).toBeNull();
    expect(queryId(comp, "try-hd-button")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("stays on the derivative with a note when Try-HD can't resolve the linked source", async () => {
    mockGetRecording.mockResolvedValueOnce(linkDetail);
    mockGetMediaUrl.mockResolvedValueOnce({
      url: "https://signed.example/deriv",
      expires_in: 600,
    });

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});

    mockGetSourceUrl.mockRejectedValueOnce(new Error("API error: 502"));
    await act(async () => {
      queryId(comp, "try-hd-button")!.props.onPress();
    });
    await act(async () => {});

    // Stayed on the derivative; honest note shown; no HD badge.
    expect(mockPlayerProps.uri).toBe("https://signed.example/deriv");
    expect(queryId(comp, "hd-badge")).toBeNull();
    expect(queryId(comp, "source-fallback-note")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("auto-falls back from a stuck HD stream to the derivative after the load timeout", async () => {
    jest.useFakeTimers();
    try {
      mockGetRecording.mockResolvedValueOnce(linkDetail);
      mockGetMediaUrl
        .mockResolvedValueOnce({ url: "https://signed.example/deriv", expires_in: 600 })
        .mockResolvedValueOnce({ url: "https://signed.example/deriv2", expires_in: 600 });

      let comp!: renderer.ReactTestRenderer;
      await act(async () => {
        comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
      });
      await act(async () => {});

      // Simulate the derivative loading (duration reported) so the initial
      // watchdog doesn't fire, then go HD.
      act(() => mockPlayerProps.onDurationChange?.(12));
      mockGetSourceUrl.mockResolvedValueOnce({
        url: "https://cdn.example/hd=dv",
        content_type: "video/mp4",
        expires_hint: "may expire",
      });
      await act(async () => {
        queryId(comp, "try-hd-button")!.props.onPress();
      });
      await act(async () => {});
      expect(mockPlayerProps.uri).toBe("https://cdn.example/hd=dv");

      // HD never reports a duration (moov-at-end, no Range → buffers forever).
      // The watchdog fires and drops back to the derivative with a note.
      await act(async () => {
        jest.advanceTimersByTime(LOAD_TIMEOUT_MS);
      });
      await act(async () => {});

      expect(mockPlayerProps.uri).toBe("https://signed.example/deriv2");
      expect(queryId(comp, "hd-badge")).toBeNull();
      expect(queryId(comp, "source-fallback-note")).toBeTruthy();
      act(() => comp.unmount());
    } finally {
      jest.useRealTimers();
    }
  });

  it("shows an honest note when even the stored copy never loads (load timeout, nothing better to try)", async () => {
    jest.useFakeTimers();
    try {
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

      // Derivative never reports a duration → watchdog fires; already on the
      // derivative, so surface the honest "isn't loading" note (no fallback).
      await act(async () => {
        jest.advanceTimersByTime(LOAD_TIMEOUT_MS);
      });
      await act(async () => {});

      expect(queryId(comp, "media-stuck-note")).toBeTruthy();
      // Never fetched an HD source (upload recording) and never fell back.
      expect(mockGetSourceUrl).not.toHaveBeenCalled();
      act(() => comp.unmount());
    } finally {
      jest.useRealTimers();
    }
  });

  it("no timeout fallback fires once the media reports a real duration", async () => {
    jest.useFakeTimers();
    try {
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

      act(() => mockPlayerProps.onDurationChange?.(42.7));
      await act(async () => {
        jest.advanceTimersByTime(LOAD_TIMEOUT_MS);
      });
      await act(async () => {});

      expect(queryId(comp, "media-stuck-note")).toBeNull();
      expect(queryId(comp, "source-fallback-note")).toBeNull();
      act(() => comp.unmount());
    } finally {
      jest.useRealTimers();
    }
  });

  it("auto-falls back to the derivative when the player errors on the remote HD stream", async () => {
    mockGetRecording.mockResolvedValueOnce(linkDetail);
    mockGetMediaUrl
      .mockResolvedValueOnce({ url: "https://signed.example/deriv", expires_in: 600 })
      .mockResolvedValueOnce({ url: "https://signed.example/deriv2", expires_in: 600 });

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});

    // Opt into HD.
    mockGetSourceUrl.mockResolvedValueOnce({
      url: "https://cdn.example/hd=dv",
      content_type: "video/mp4",
      expires_hint: "may expire; refetch on failure",
    });
    await act(async () => {
      queryId(comp, "try-hd-button")!.props.onPress();
    });
    await act(async () => {});
    expect(mockPlayerProps.uri).toBe("https://cdn.example/hd=dv");
    expect(queryId(comp, "hd-badge")).toBeTruthy();

    // Player errors on the remote stream → automatic fallback.
    await act(async () => {
      mockPlayerProps.onError?.("decode failed");
    });
    await act(async () => {});

    expect(mockPlayerProps.uri).toBe("https://signed.example/deriv2");
    expect(queryId(comp, "hd-badge")).toBeNull();
    expect(queryId(comp, "source-fallback-note")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("uses the derivative-only path for an upload-sourced recording (no source_url, no badge, no Try-HD)", async () => {
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
    expect(queryId(comp, "try-hd-button")).toBeNull();
    act(() => comp.unmount());
  });

  describe("attach HD source", () => {
    it("attaches a link, refetches, and the Try-HD opt-in appears (derivative still plays, no auto badge)", async () => {
      // Starts as an upload-sourced recording (derivative-only, Attach offered).
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

      // No link yet → "Attach HD source" (not "Replace"), no HD badge, no Try-HD.
      expect(queryId(comp, "attach-source-button")).toBeTruthy();
      expect(queryId(comp, "replace-source-button")).toBeNull();
      expect(queryId(comp, "hd-badge")).toBeNull();
      expect(queryId(comp, "try-hd-button")).toBeNull();

      // Open the input and type a link.
      act(() => queryId(comp, "attach-source-button")!.props.onPress());
      expect(queryId(comp, "attach-source-input")).toBeTruthy();
      act(() =>
        queryId(comp, "attach-source-input")!.props.onChangeText(
          "https://drive.google.com/file/d/abc/view",
        ),
      );

      // PATCH succeeds; the refetch returns a link-sourced recording. Playback
      // stays on the reliable derivative; the Try-HD opt-in now appears.
      mockPatchSource.mockResolvedValueOnce({
        type: "link",
        url: "https://drive.google.com/file/d/abc/view",
        original_filename: "kitchen-fight.m4a",
      });
      mockGetRecording.mockResolvedValueOnce(linkDetail);
      mockGetMediaUrl.mockResolvedValueOnce({
        url: "https://signed.example/deriv2",
        expires_in: 600,
      });

      await act(async () => {
        queryId(comp, "attach-source-submit")!.props.onPress();
      });
      await act(async () => {});

      expect(mockPatchSource).toHaveBeenCalledWith(
        "r1",
        "https://drive.google.com/file/d/abc/view",
      );
      // Refetched, still derivative-first (no auto HD fetch), Try-HD now offered.
      expect(mockGetSourceUrl).not.toHaveBeenCalled();
      expect(mockPlayerProps.uri).toBe("https://signed.example/deriv2");
      expect(queryId(comp, "hd-badge")).toBeNull();
      expect(queryId(comp, "try-hd-button")).toBeTruthy();
      act(() => comp.unmount());
    });

    it("renders a 422's user-facing detail verbatim and doesn't refetch", async () => {
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

      act(() => queryId(comp, "attach-source-button")!.props.onPress());
      act(() =>
        queryId(comp, "attach-source-input")!.props.onChangeText(
          "https://example.com/not-a-single-item",
        ),
      );

      const serverMsg =
        "That link points to an album, not a single video — open one item and " +
        "share just that.";
      const err = Object.assign(new Error(serverMsg), {
        status: 422,
        detail: serverMsg,
      });
      mockPatchSource.mockRejectedValueOnce(err);

      // getRecording was called once on mount; a failed PATCH must not refetch.
      const callsBefore = mockGetRecording.mock.calls.length;
      await act(async () => {
        queryId(comp, "attach-source-submit")!.props.onPress();
      });
      await act(async () => {});

      const errNode = queryId(comp, "attach-source-error");
      expect(errNode).toBeTruthy();
      expect(JSON.stringify(comp.toJSON())).toContain(serverMsg);
      expect(mockGetRecording.mock.calls.length).toBe(callsBefore);
      // Still no HD badge — nothing was fabricated.
      expect(queryId(comp, "hd-badge")).toBeNull();
      act(() => comp.unmount());
    });

    it("offers a Replace-source affordance when the recording is already link-sourced", async () => {
      mockGetRecording.mockResolvedValueOnce(linkDetail);
      mockGetMediaUrl.mockResolvedValueOnce({
        url: "https://signed.example/deriv",
        expires_in: 600,
      });

      let comp!: renderer.ReactTestRenderer;
      await act(async () => {
        comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
      });
      await act(async () => {});

      // Already a link → "Replace source link", not "Attach HD source". Same input.
      expect(queryId(comp, "replace-source-button")).toBeTruthy();
      expect(queryId(comp, "attach-source-button")).toBeNull();
      act(() => queryId(comp, "replace-source-button")!.props.onPress());
      expect(queryId(comp, "attach-source-input")).toBeTruthy();
      act(() => comp.unmount());
    });
  });

  describe("Google Photos source (HD impossible)", () => {
    it("suppresses Try-HD and shows an honest note for a Google Photos link", async () => {
      mockGetRecording.mockResolvedValueOnce(photosDetail);
      mockGetMediaUrl.mockResolvedValueOnce({
        url: "https://signed.example/deriv",
        expires_in: 600,
      });

      let comp!: renderer.ReactTestRenderer;
      await act(async () => {
        comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
      });
      await act(async () => {});

      // The stored derivative plays; Try-HD is NOT offered (Photos =dv can never
      // stream), replaced by an honest note. The HD source is never fetched.
      expect(mockPlayerProps.uri).toBe("https://signed.example/deriv");
      expect(queryId(comp, "try-hd-button")).toBeNull();
      expect(queryId(comp, "photos-hd-note")).toBeTruthy();
      expect(mockGetSourceUrl).not.toHaveBeenCalled();
      expect(JSON.stringify(comp.toJSON())).toContain("Google Photos");
      act(() => comp.unmount());
    });
  });

  describe("rename in place", () => {
    it("renames the recording via PATCH and shows the new title", async () => {
      mockGetRecording.mockResolvedValueOnce(detail);
      mockGetMediaUrl.mockResolvedValueOnce({ url: "https://signed/x", expires_in: 600 });

      let comp!: renderer.ReactTestRenderer;
      await act(async () => {
        comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
      });
      await act(async () => {});

      // Tap the title → edit mode; type a new name.
      act(() => queryId(comp, "replay-title")!.props.onPress());
      expect(queryId(comp, "rename-input")).toBeTruthy();
      act(() =>
        queryId(comp, "rename-input")!.props.onChangeText("Sunday budget talk"),
      );

      mockPatchTitle.mockResolvedValueOnce({ id: "r1", title: "Sunday budget talk" });
      await act(async () => {
        queryId(comp, "rename-save")!.props.onPress();
      });
      await act(async () => {});

      expect(mockPatchTitle).toHaveBeenCalledWith("r1", "Sunday budget talk");
      // Optimistic rename applied; editor closed; no honest-error note.
      expect(JSON.stringify(comp.toJSON())).toContain("Sunday budget talk");
      expect(queryId(comp, "rename-input")).toBeNull();
      expect(queryId(comp, "rename-note")).toBeNull();
      act(() => comp.unmount());
    });

    it("keeps the name and shows an honest note when rename isn't supported (4xx)", async () => {
      mockGetRecording.mockResolvedValueOnce(detail);
      mockGetMediaUrl.mockResolvedValueOnce({ url: "https://signed/x", expires_in: 600 });

      let comp!: renderer.ReactTestRenderer;
      await act(async () => {
        comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
      });
      await act(async () => {});

      act(() => queryId(comp, "replay-title")!.props.onPress());
      act(() =>
        queryId(comp, "rename-input")!.props.onChangeText("New name"),
      );

      // Backend has no title field/route yet → 405. Name unchanged, honest note.
      mockPatchTitle.mockRejectedValueOnce(
        Object.assign(new Error("API error: 405"), { status: 405 }),
      );
      await act(async () => {
        queryId(comp, "rename-save")!.props.onPress();
      });
      await act(async () => {});

      const note = queryId(comp, "rename-note");
      expect(note).toBeTruthy();
      expect(JSON.stringify(comp.toJSON())).toContain("isn’t supported yet");
      // The original filename still shows — nothing was fabricated.
      expect(JSON.stringify(comp.toJSON())).toContain("kitchen-fight.m4a");
      act(() => comp.unmount());
    });
  });
});

describe("ReplayScreen re-analyze", () => {
  async function mountLoaded(): Promise<renderer.ReactTestRenderer> {
    mockGetRecording.mockResolvedValueOnce(detail);
    mockGetMediaUrl.mockResolvedValueOnce({ url: "https://signed/x", expires_in: 600 });
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});
    return comp;
  }

  it("shows the re-analyze affordance once the recording is loaded", async () => {
    const comp = await mountLoaded();
    expect(queryId(comp, "reanalyze-button")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("submits the job, polls to done, and refetches the recording", async () => {
    const comp = await mountLoaded();
    mockReanalyze.mockResolvedValueOnce({ job_id: "job_1" });
    // First poll already done — the poller breaks before any sleep.
    mockGetJob.mockResolvedValueOnce({
      job_id: "job_1",
      status: "done",
      created_at: "",
      updated_at: "",
      stage_started_at: null,
      progress_note: null,
      duration_seconds: null,
      error: null,
      result: { ...detail.analysis, turns: [], stored: true, recording_id: "r1", storage_note: null },
    });
    // load() after done refetches the recording + media url.
    mockGetRecording.mockResolvedValueOnce(detail);
    mockGetMediaUrl.mockResolvedValueOnce({ url: "https://signed/y", expires_in: 600 });

    await act(async () => {
      queryId(comp, "reanalyze-button")!.props.onPress();
    });
    await act(async () => {});

    expect(mockReanalyze).toHaveBeenCalledWith("r1");
    expect(mockGetJob).toHaveBeenCalledWith("job_1");
    // Refreshed: getRecording called a second time (mount + refresh).
    expect(mockGetRecording).toHaveBeenCalledTimes(2);
    // Back to the button, no error.
    expect(queryId(comp, "reanalyze-error")).toBeNull();
    expect(queryId(comp, "reanalyze-button")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("shows an honest 422 message when the recording kept no audio", async () => {
    const comp = await mountLoaded();
    const err = new Error("API error: 422") as Error & { status?: number };
    err.status = 422;
    mockReanalyze.mockRejectedValueOnce(err);

    await act(async () => {
      queryId(comp, "reanalyze-button")!.props.onPress();
    });
    await act(async () => {});

    const errNode = queryId(comp, "reanalyze-error");
    expect(errNode).toBeTruthy();
    expect(JSON.stringify(comp.toJSON())).toContain("can’t be re-analyzed");
    // The job was never polled (the submit itself failed).
    expect(mockGetJob).not.toHaveBeenCalled();
    act(() => comp.unmount());
  });

  it("surfaces a failed job's message honestly", async () => {
    const comp = await mountLoaded();
    mockReanalyze.mockResolvedValueOnce({ job_id: "job_2" });
    mockGetJob.mockResolvedValueOnce({
      job_id: "job_2",
      status: "failed",
      created_at: "",
      updated_at: "",
      stage_started_at: null,
      progress_note: null,
      duration_seconds: null,
      error: "The engine hit a snag.",
      result: null,
    });

    await act(async () => {
      queryId(comp, "reanalyze-button")!.props.onPress();
    });
    await act(async () => {});

    expect(JSON.stringify(comp.toJSON())).toContain("The engine hit a snag.");
    act(() => comp.unmount());
  });
});

describe("ReplayScreen manual speaker naming", () => {
  // A detail on a manual-labels-capable server: carries the raw manual map (empty
  // here) and resolved speaker_labels. Speakers start on the generic (raw-id) rung.
  const namingDetail: RecordingDetail = {
    ...detail,
    manual_speaker_labels: {},
    analysis: {
      ...detail.analysis!,
      speaker_labels: {
        Alice: { display_label: "Alice", label_source: "generic" },
        Bob: { display_label: "Bob", label_source: "generic" },
      },
    },
  };

  async function mountWith(rec: RecordingDetail): Promise<renderer.ReactTestRenderer> {
    mockGetRecording.mockResolvedValueOnce(rec);
    mockGetMediaUrl.mockResolvedValueOnce({ url: "https://signed/x", expires_in: 600 });
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<ReplayScreen recordingId="r1" onBack={() => {}} />);
    });
    await act(async () => {});
    return comp;
  }

  it("hides the naming affordance on an older server (no manual_speaker_labels field)", async () => {
    // `detail` has no manual_speaker_labels field → capability absent.
    const comp = await mountWith(detail);
    expect(queryId(comp, "speaker-naming")).toBeNull();
    act(() => comp.unmount());
  });

  it("offers a per-speaker name affordance when the server supports manual labels", async () => {
    const comp = await mountWith(namingDetail);
    expect(queryId(comp, "speaker-naming")).toBeTruthy();
    expect(queryId(comp, "name-edit-Alice")).toBeTruthy();
    expect(queryId(comp, "name-edit-Bob")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("names a speaker and updates every label surface from the response", async () => {
    const comp = await mountWith(namingDetail);

    // Open Alice's editor, type a name.
    act(() => queryId(comp, "name-edit-Alice")!.props.onPress());
    expect(queryId(comp, "name-input-Alice")).toBeTruthy();
    act(() => queryId(comp, "name-input-Alice")!.props.onChangeText("Mom"));

    // Server merges + resolves: Alice becomes a manual "Mom", Bob unchanged.
    mockPatchLabels.mockResolvedValueOnce({
      id: "r1",
      manual_speaker_labels: { Alice: "Mom" },
      speaker_labels: {
        Alice: { display_label: "Mom", label_source: "manual" },
        Bob: { display_label: "Bob", label_source: "generic" },
      },
    });

    await act(async () => {
      queryId(comp, "name-save-Alice")!.props.onPress();
    });
    await act(async () => {});

    // Append-only: only Alice was sent.
    expect(mockPatchLabels).toHaveBeenCalledWith("r1", { Alice: "Mom" });

    // Naming row shows the new name + "named by you" provenance.
    const json = JSON.stringify(comp.toJSON());
    expect(json).toContain("Mom");
    const prov = queryId(comp, "name-provenance-Alice");
    expect(prov).toBeTruthy();
    expect(JSON.stringify(prov!.props.children)).toContain("named by you");

    // "Mom" now appears on more than one surface — the naming row AND the chart
    // legend — proving the shared effective-labels state re-labeled everywhere.
    const occurrences = json.split("Mom").length - 1;
    expect(occurrences).toBeGreaterThanOrEqual(2);
    act(() => comp.unmount());
  });

  it("clears a manual label to empty and restores the inferred one", async () => {
    // Start already-named: Alice manual "Mom".
    const named: RecordingDetail = {
      ...namingDetail,
      manual_speaker_labels: { Alice: "Mom" },
      analysis: {
        ...namingDetail.analysis!,
        speaker_labels: {
          Alice: { display_label: "Mom", label_source: "manual" },
          Bob: { display_label: "Bob", label_source: "generic" },
        },
      },
    };
    const comp = await mountWith(named);

    // Editor prefills with the raw manual name, then the user clears it.
    act(() => queryId(comp, "name-edit-Alice")!.props.onPress());
    expect(queryId(comp, "name-input-Alice")!.props.value).toBe("Mom");
    act(() => queryId(comp, "name-input-Alice")!.props.onChangeText(""));

    // Server clears Alice's manual label; the inferred (generic) label returns.
    mockPatchLabels.mockResolvedValueOnce({
      id: "r1",
      manual_speaker_labels: {},
      speaker_labels: {
        Alice: { display_label: "Alice", label_source: "generic" },
        Bob: { display_label: "Bob", label_source: "generic" },
      },
    });

    await act(async () => {
      queryId(comp, "name-save-Alice")!.props.onPress();
    });
    await act(async () => {});

    // Empty string was sent (the clear signal).
    expect(mockPatchLabels).toHaveBeenCalledWith("r1", { Alice: "" });
    // Row is back to the inferred label with no "named by you" note.
    expect(queryId(comp, "name-provenance-Alice")).toBeNull();
    const json = JSON.stringify(comp.toJSON());
    expect(json).not.toContain("named by you");
    act(() => comp.unmount());
  });

  it("hides the affordance gracefully when a save 404s (older server, no route)", async () => {
    const comp = await mountWith(namingDetail);

    act(() => queryId(comp, "name-edit-Bob")!.props.onPress());
    act(() => queryId(comp, "name-input-Bob")!.props.onChangeText("Dad"));

    mockPatchLabels.mockRejectedValueOnce(
      Object.assign(new Error("API error: 404"), { status: 404 }),
    );
    await act(async () => {
      queryId(comp, "name-save-Bob")!.props.onPress();
    });
    await act(async () => {});

    // The whole naming card is gone — no dead affordance on a server without it.
    expect(queryId(comp, "speaker-naming")).toBeNull();
    act(() => comp.unmount());
  });

  it("surfaces an honest error when a save fails transiently (name unchanged)", async () => {
    const comp = await mountWith(namingDetail);

    act(() => queryId(comp, "name-edit-Alice")!.props.onPress());
    act(() => queryId(comp, "name-input-Alice")!.props.onChangeText("Mom"));

    mockPatchLabels.mockRejectedValueOnce(
      Object.assign(new Error("API error: 500"), { status: 500 }),
    );
    await act(async () => {
      queryId(comp, "name-save-Alice")!.props.onPress();
    });
    await act(async () => {});

    // Card still there, honest error shown, no fabricated label.
    expect(queryId(comp, "speaker-naming")).toBeTruthy();
    expect(queryId(comp, "name-error")).toBeTruthy();
    expect(JSON.stringify(comp.toJSON())).toContain("Please try again");
    act(() => comp.unmount());
  });
});
