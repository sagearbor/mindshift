/**
 * The experimental "Separate voices on this phone (engine B)" row on a
 * stored recording's replay: hidden unless Advanced → "Experimental voice
 * engine" is on, and never for a shared or audio-less recording; with an
 * injected engine it shows progress, then the strip + k + timings, posts the
 * device_diarization diagnostics event, and phrases a failure inline.
 */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import ReplayScreen from "../src/screens/ReplayScreen";
import { getRecording, getRecordingMediaUrl } from "../src/api/client";
import type { RecordingDetail } from "../src/api/client";
import { loadExperimentalVoiceEngine } from "../src/live/experimentalPrefs";
import { runDeviceDiarization, DeviceDiarizationError } from "../src/live/deviceDiarization";
import { useDiagnosticsStore, type DeviceDiarizationEvent } from "../src/diagnostics/diagnostics";

jest.mock("../src/api/client", () => ({
  getRecording: jest.fn(),
  getRecordingMediaUrl: jest.fn(),
  getRecordingSourceUrl: jest.fn(),
  patchRecordingSource: jest.fn(),
  patchRecordingTitle: jest.fn(),
  patchSpeakerLabels: jest.fn(),
  postReanalyze: jest.fn(),
  getAnalyzeJob: jest.fn(),
  postShare: jest.fn(),
  deleteShare: jest.fn(),
  getVoiceProfile: jest.fn(() => Promise.resolve({ available: false, storage_enabled: false, enrolled: false, enroll_count: 0 })),
  enrollVoice: jest.fn(),
  pcm16kMediaUrl: jest.fn((m: { url: string }) => `${m.url}&format=pcm16k`),
}));
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
jest.mock("../src/live/experimentalPrefs", () => ({
  __esModule: true,
  DEFAULT_EXPERIMENTAL_VOICE_ENGINE: false,
  loadExperimentalVoiceEngine: jest.fn(),
  saveExperimentalVoiceEngine: jest.fn(),
}));
jest.mock("../src/live/deviceDiarization", () => {
  const actual = jest.requireActual("../src/live/deviceDiarization");
  return { __esModule: true, ...actual, runDeviceDiarization: jest.fn() };
});

const mockGetRecording = getRecording as jest.Mock;
const mockGetMediaUrl = getRecordingMediaUrl as jest.Mock;
const mockLoadPref = loadExperimentalVoiceEngine as jest.Mock;
const mockRun = runDeviceDiarization as jest.Mock;

const detail: RecordingDetail = {
  id: "r1",
  created_at: "2026-07-01T10:00:00Z",
  filename: "dinner.m4a",
  media_type: "audio",
  duration_seconds: 42.6,
  has_analysis: true,
  turns: [
    { speaker: "Speaker A", text: "Pass the bread.", start_time: 0, end_time: 3 },
    { speaker: "Speaker B", text: "Here.", start_time: 3, end_time: 6 },
  ],
  analysis: {
    per_turn: [
      { index: 0, speaker: "Speaker A", heat: 20, markers: [], is_spike: false, trigger_phrase: null },
      { index: 1, speaker: "Speaker B", heat: 35, markers: [], is_spike: false, trigger_phrase: null },
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

const event: DeviceDiarizationEvent = {
  recording_id: "r1",
  engine: "B",
  k: 3,
  k_eigengap: 3,
  eigenvalues: [1, 0.9, 0.8, 0.2],
  segments: [
    [0, 10.5, 0],
    [10.5, 25.0, 1],
    [25.0, 42.6, 2],
  ],
  windows: 162,
  windows_total: 165,
  window_s: 1.5,
  hop_s: 0.25,
  gate_rms: 0.003,
  speech_s: 35.9,
  duration_s: 42.6,
  download_ms: 1200,
  download_bytes: 1364128,
  embed_ms_mean: 38.2,
  embed_ms_p90: 45.1,
  cluster_ms: 90,
  total_ms: 9800,
  model_rev: "0f99f2d0ebe89ac0",
  model_source: "cached",
  device: { platform: "android", osVersion: "16", model: "Pixel 10", userAgent: null },
  created_at: "2026-08-30T01:00:00.000Z",
};

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

function textOf(node: ReactTestInstance): string {
  return node
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");
}

async function flush() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function render(rec: RecordingDetail = detail) {
  mockGetRecording.mockResolvedValue(rec);
  mockGetMediaUrl.mockResolvedValue({ url: "https://api.test/recordings/r1/media?tk=abc", expires_in: 900 });
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<ReplayScreen recordingId="r1" onBack={jest.fn()} />);
  });
  await flush();
  return comp;
}

let mockSend: jest.Mock;

beforeEach(() => {
  mockGetRecording.mockReset();
  mockGetMediaUrl.mockReset();
  mockRun.mockReset();
  mockLoadPref.mockReset().mockResolvedValue(true);
  mockSend = jest.fn().mockResolvedValue({ ok: true, id: "dx-TEST-TEST" });
  useDiagnosticsStore.setState({ sendDeviceDiarization: mockSend, deviceDiarization: null, lastSent: null });
});

describe("ReplayScreen — Separate voices on this phone (engine B)", () => {
  it("is hidden when the Advanced switch is off", async () => {
    mockLoadPref.mockResolvedValue(false);
    const comp = await render();
    expect(queryId(comp, "device-diarization-row")).toBeNull();
    expect(mockLoadPref).toHaveBeenCalled();
  });

  it("is hidden on a shared (read-only) recording and on a live session without audio", async () => {
    const shared = await render({ ...detail, shared: true, owner_email: "linda@example.com" });
    expect(queryId(shared, "device-diarization-row")).toBeNull();
    const noAudio = await render({ ...detail, media_type: "none", source: { type: "live", url: null } });
    expect(queryId(noAudio, "device-diarization-row")).toBeNull();
  });

  it("runs the injected engine: progress, then the strip, k, timings and the diagnostics id", async () => {
    let resolveRun!: (ev: DeviceDiarizationEvent) => void;
    let progress: ((p: { phase: string; fraction: number | null; detail: string }) => void) | undefined;
    const cancel = jest.fn();
    mockRun.mockImplementation((_id: string, opts: { onProgress?: typeof progress }) => {
      progress = opts.onProgress;
      return { promise: new Promise<DeviceDiarizationEvent>((res) => (resolveRun = res)), cancel };
    });
    const comp = await render();
    const row = queryId(comp, "device-diarization-row");
    expect(row).not.toBeNull();
    expect(textOf(row!)).toContain("Separate voices on this phone (engine B)");

    await act(async () => {
      queryId(comp, "device-diarization-run")!.props.onPress();
    });
    expect(mockRun).toHaveBeenCalledWith("r1", expect.objectContaining({ onProgress: expect.any(Function) }));
    await act(async () => {
      progress?.({ phase: "embed", fraction: 0.5, detail: "listening to window 81 of 162" });
    });
    const prog = queryId(comp, "device-diarization-progress");
    expect(prog).not.toBeNull();
    expect(textOf(prog!)).toContain("listening to window 81 of 162 · 50%");
    // The screen is not blocked: the chart and the rest still render around it.
    expect(queryId(comp, "replay-content")).not.toBeNull();

    await act(async () => {
      resolveRun(event);
    });
    await flush();
    expect(queryId(comp, "device-diarization-progress")).toBeNull();
    const strip = queryId(comp, "device-diarization-strip");
    expect(strip).not.toBeNull();
    // Host views only (react-test-renderer also lists the composite View).
    expect(comp.root.findAll((n) => typeof n.type === "string" && typeof n.props?.testID === "string" && n.props.testID.startsWith("device-diarization-seg-")).length).toBe(3);
    // Segments are drawn in the app's speaker colours, proportional to duration.
    const seg0 = queryId(comp, "device-diarization-seg-0")!;
    const seg1 = queryId(comp, "device-diarization-seg-1")!;
    const style0 = Object.assign({}, ...[seg0.props.style].flat());
    const style1 = Object.assign({}, ...[seg1.props.style].flat());
    expect(style0.backgroundColor).toBe("#4A90D9"); // Speaker A
    expect(style1.backgroundColor).toBe("#E85D75"); // Speaker B
    expect(style0.flex).toBeCloseTo(10.5 / 42.6, 6);
    expect(textOf(queryId(comp, "device-diarization-k")!)).toContain("3 voices found (eigengap 3)");
    const timings = textOf(queryId(comp, "device-diarization-timings")!);
    expect(timings).toContain("download 1.2 s (1.4 MB)");
    expect(timings).toContain("embed 38.2 ms/window (p90 45.1)");
    expect(timings).toContain("total 9.8 s");
    expect(timings).toContain("162/165 windows @ 0.25 s hop");
    // Posted as a diagnostics event; its id is shown.
    expect(mockSend).toHaveBeenCalledWith(event, { uid: null, email: null });
    expect(textOf(queryId(comp, "device-diarization-sent")!)).toContain("ID dx-TEST-TEST");
  });

  it("cancel stops the run and a failure is one honest inline line", async () => {
    let rejectRun!: (err: Error) => void;
    const cancel = jest.fn(() => rejectRun(new DeviceDiarizationError("cancelled", "cancelled")));
    mockRun.mockImplementation(() => ({ promise: new Promise<DeviceDiarizationEvent>((_, rej) => (rejectRun = rej)), cancel }));
    const comp = await render();
    await act(async () => {
      queryId(comp, "device-diarization-run")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "device-diarization-cancel")!.props.onPress();
    });
    await flush();
    expect(cancel).toHaveBeenCalled();
    expect(textOf(queryId(comp, "device-diarization-error")!)).toBe("Cancelled.");
    expect(mockSend).not.toHaveBeenCalled();

    // The model-missing case says what to do about it.
    mockRun.mockImplementation(() => ({
      promise: Promise.reject(new DeviceDiarizationError("the voice model isn't ready on this phone (offline and no cached model)", "model-unavailable")),
      cancel: jest.fn(),
    }));
    await act(async () => {
      queryId(comp, "device-diarization-run")!.props.onPress();
    });
    await flush();
    const err = textOf(queryId(comp, "device-diarization-error")!);
    expect(err).toContain("offline and no cached model");
    expect(err).toContain("Start a live session once so the model downloads");
  });
});
