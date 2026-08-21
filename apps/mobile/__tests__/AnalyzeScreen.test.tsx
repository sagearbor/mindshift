import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import * as DocumentPicker from "expo-document-picker";
import AnalyzeScreen from "../src/screens/AnalyzeScreen";
import type { AudioRecorderDeps } from "../src/recorder/AudioRecordScreen";
import { useSessionStore } from "../src/store/sessionStore";
import { useRecorderStore } from "../src/store/recorderStore";
import { useAnalyzeStore } from "../src/store/analyzeStore";
import { relationshipContext } from "../src/components/RelationshipPicker";
import {
  postAnalyzeUpload,
  postAnalyzeUploadChunked,
  postAnalyzeUploadChunkedJob,
  postAnalyzeLink,
  postAnalyzeLinkJob,
  getAnalyzeJob,
  reportClientLog,
  UploadError,
} from "../src/api/client";
import type { AnalyzeJobState, UploadAnalyzeResult } from "../src/api/client";

// Keep the real client (the store uses postRespond) but stub the upload calls.
jest.mock("../src/api/client", () => ({
  __esModule: true,
  ...jest.requireActual("../src/api/client"),
  postAnalyzeUpload: jest.fn(),
  postAnalyzeUploadChunked: jest.fn(),
  postAnalyzeUploadChunkedJob: jest.fn(),
  postAnalyzeLink: jest.fn(),
  postAnalyzeLinkJob: jest.fn(),
  getAnalyzeJob: jest.fn(),
  reportClientLog: jest.fn(),
}));

const mockPick = DocumentPicker.getDocumentAsync as jest.Mock;
const mockUpload = postAnalyzeUpload as jest.Mock;
const mockChunked = postAnalyzeUploadChunked as jest.Mock;
const mockChunkedJob = postAnalyzeUploadChunkedJob as jest.Mock;
const mockLink = postAnalyzeLink as jest.Mock;
const mockLinkJob = postAnalyzeLinkJob as jest.Mock;
const mockGetJob = getAnalyzeJob as jest.Mock;
const mockReport = reportClientLog as jest.Mock;

// The relationship picker starts UNSELECTED (it's optional): no relationship
// sentence is sent until the user taps a pill. Once tapped, its sentence
// leads the analyze `context`.
const PARTNERS_CONTEXT = relationshipContext("partners");

/** A terminal "done" job state carrying the given result — the common poll
 *  response for happy-path job tests (pollJobToDone returns on the first poll,
 *  so no timers are needed). */
function doneJob(result: UploadAnalyzeResult): AnalyzeJobState {
  return {
    job_id: "job_1",
    status: "done",
    created_at: "2026-07-12T00:00:00Z",
    updated_at: "2026-07-12T00:00:05Z",
    stage_started_at: "2026-07-12T00:00:05Z",
    progress_note: null,
    duration_seconds: 12,
    error: null,
    result,
  };
}

const MB = 1024 * 1024;

/** The arguments of the sole chunked-job call — where the file identity and
 *  the user's choices (context/consent/store/title) ride now that EVERY upload
 *  goes through the async job path (the direct sync path is gone). */
function chunkedCall(): {
  file: unknown;
  name: unknown;
  mimeType: unknown;
  size: unknown;
  opts: Record<string, unknown>;
} {
  expect(mockChunkedJob).toHaveBeenCalledTimes(1);
  const [file, name, mimeType, size, opts] = mockChunkedJob.mock.calls[0];
  return { file, name, mimeType, size, opts };
}

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

const uploadFixture: UploadAnalyzeResult = {
  per_turn: [
    { index: 0, speaker: "Alice", heat: 20, markers: [], is_spike: false, trigger_phrase: null },
    { index: 1, speaker: "Bob", heat: 40, markers: [], is_spike: false, trigger_phrase: null },
  ],
  per_speaker: {},
  dynamics: {
    coupling: { strength: null, leader: null, description: "" },
    deescalation: { who_first: null, follow_rate: null, description: "" },
    triggers: [],
    requests: [],
  },
  narrative: "",
  turns: [
    { speaker: "Alice", text: "You never help.", start_time: 0, end_time: 1.2 },
    { speaker: "Bob", text: "I do plenty.", start_time: 1.3, end_time: 2.1 },
  ],
  stored: false,
  recording_id: null,
  storage_note: "Without consent to store, we discard the file after analysis.",
};

// Reset stores between tests
beforeEach(() => {
  mockPick.mockReset();
  mockUpload.mockReset();
  mockChunked.mockReset();
  mockChunkedJob.mockReset();
  mockLink.mockReset();
  mockLinkJob.mockReset();
  mockGetJob.mockReset();
  mockReport.mockReset();
  act(() => {
    useSessionStore.setState({
      role: "Husband / Wife",
      empathyLevel: 50,
      turns: [],
      suggestions: [],
      loading: false,
    });
    useRecorderStore.setState({ pendingFile: null });
    useAnalyzeStore.setState({ relationship: null });
  });
});

describe("AnalyzeScreen", () => {
  it("renders the initial screen", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<AnalyzeScreen />);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("wires the back / recordings / record / text-tools affordances", () => {
    const onBack = jest.fn();
    const onOpenRecordings = jest.fn();
    const onRecordVideo = jest.fn();
    const onOpenTextTools = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AnalyzeScreen
          onBack={onBack}
          onOpenRecordings={onOpenRecordings}
          onRecordVideo={onRecordVideo}
          onOpenTextTools={onOpenTextTools}
        />,
      );
    });
    act(() => queryId(comp, "analyze-back")!.props.onPress());
    act(() => queryId(comp, "open-recordings-link")!.props.onPress());
    act(() => queryId(comp, "record-video-button")!.props.onPress());
    act(() => queryId(comp, "open-text-tools")!.props.onPress());
    expect(onBack).toHaveBeenCalledTimes(1);
    expect(onOpenRecordings).toHaveBeenCalledTimes(1);
    expect(onRecordVideo).toHaveBeenCalledTimes(1);
    expect(onOpenTextTools).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });

  describe("relationship context picker", () => {
    it("starts with NOTHING selected, labeled optional, every option one tap away", () => {
      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen />);
      });
      for (const id of [
        "partners",
        "parent_child",
        "coworkers",
        "friends",
        "just_me",
      ]) {
        expect(queryId(comp, `relationship-${id}`)).toBeTruthy();
        expect(
          queryId(comp, `relationship-${id}`)!.props.accessibilityState
            .selected,
        ).toBe(false);
      }
      // The honest optional hint is visible.
      expect(queryId(comp, "relationship-optional-hint")).toBeTruthy();
      expect(JSON.stringify(comp.toJSON())).toContain(
        "we’ll figure it out if you skip",
      );
      act(() => comp.unmount());
    });

    it("sends NO relationship context when nothing is selected (infer mode)", async () => {
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///rec.m4a", name: "rec.m4a", size: 2048, mimeType: "audio/m4a" },
        ],
      });
      mockChunkedJob.mockResolvedValueOnce({ result: uploadFixture });

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen />);
      });
      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });

      // Context is undefined — no fabricated relationship sentence.
      const { file, name, mimeType, size, opts } = chunkedCall();
      expect(file).toBe("file:///rec.m4a");
      expect(name).toBe("rec.m4a");
      expect(mimeType).toBe("audio/m4a");
      expect(size).toBe(2048);
      expect(opts).toEqual({
        consent: false,
        store: true,
        context: undefined,
        title: undefined,
        onProgress: expect.any(Function),
      });
      act(() => comp.unmount());
    });

    it("tapping the selected pill deselects it — back to infer mode, no context sent", async () => {
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///rec.m4a", name: "rec.m4a", size: 2048, mimeType: "audio/m4a" },
        ],
      });
      mockChunkedJob.mockResolvedValueOnce({ result: uploadFixture });

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen />);
      });
      // Select, then tap the same pill again to deselect.
      act(() => {
        queryId(comp, "relationship-coworkers")!.props.onPress();
      });
      expect(
        queryId(comp, "relationship-coworkers")!.props.accessibilityState
          .selected,
      ).toBe(true);
      act(() => {
        queryId(comp, "relationship-coworkers")!.props.onPress();
      });
      expect(
        queryId(comp, "relationship-coworkers")!.props.accessibilityState
          .selected,
      ).toBe(false);

      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });
      // Deselected → no relationship sentence rides along.
      expect(chunkedCall().opts).toEqual(
        expect.objectContaining({
          consent: false,
          store: true,
          context: undefined,
        }),
      );
      act(() => comp.unmount());
    });

    it("one tap switches the relationship and it wires through to the upload", async () => {
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///rec.m4a", name: "rec.m4a", size: 2048, mimeType: "audio/m4a" },
        ],
      });
      mockChunkedJob.mockResolvedValueOnce({ result: uploadFixture });

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen />);
      });
      act(() => {
        queryId(comp, "relationship-coworkers")!.props.onPress();
      });
      expect(
        queryId(comp, "relationship-coworkers")!.props.accessibilityState
          .selected,
      ).toBe(true);
      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });

      expect(chunkedCall().opts).toEqual(
        expect.objectContaining({
          consent: false,
          store: true,
          context: relationshipContext("coworkers"),
        }),
      );
      act(() => comp.unmount());
    });

    it("appends the free-text context after the tapped relationship sentence", async () => {
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///rec.m4a", name: "rec.m4a", size: 2048, mimeType: "audio/m4a" },
        ],
      });
      mockChunkedJob.mockResolvedValueOnce({ result: uploadFixture });

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen />);
      });
      act(() => {
        queryId(comp, "relationship-partners")!.props.onPress();
      });
      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      act(() => {
        queryId(comp, "recording-context-input")!.props.onChangeText(
          "  We were arguing about chores. ",
        );
      });
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });

      expect(chunkedCall().opts).toEqual(
        expect.objectContaining({
          consent: false,
          store: true,
          context: `${PARTNERS_CONTEXT} We were arguing about chores.`,
        }),
      );
      act(() => comp.unmount());
    });

    it("sends just the free text when no relationship is selected", async () => {
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///rec.m4a", name: "rec.m4a", size: 2048, mimeType: "audio/m4a" },
        ],
      });
      mockChunkedJob.mockResolvedValueOnce({ result: uploadFixture });

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen />);
      });
      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      act(() => {
        queryId(comp, "recording-context-input")!.props.onChangeText(
          "We were arguing about chores.",
        );
      });
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });

      expect(chunkedCall().opts).toEqual(
        expect.objectContaining({
          consent: false,
          store: true,
          context: "We were arguing about chores.",
        }),
      );
      act(() => comp.unmount());
    });

    it("remembers the last relationship across screen remounts (smart default)", () => {
      let first!: renderer.ReactTestRenderer;
      act(() => {
        first = renderer.create(<AnalyzeScreen />);
      });
      act(() => {
        queryId(first, "relationship-friends")!.props.onPress();
      });
      act(() => first.unmount());

      // A fresh mount (e.g. after a pushed sub-screen pops back) keeps Friends.
      let second!: renderer.ReactTestRenderer;
      act(() => {
        second = renderer.create(<AnalyzeScreen />);
      });
      expect(
        queryId(second, "relationship-friends")!.props.accessibilityState
          .selected,
      ).toBe(true);
      act(() => second.unmount());
    });
  });

  describe("analyze a recording", () => {
    it("renders the pick-recording button", () => {
      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen />);
      });
      expect(queryId(comp, "pick-recording-button")).toBeTruthy();
      // The upload button only appears once a file is picked.
      expect(queryId(comp, "upload-analyze-button")).toBeNull();
      act(() => comp.unmount());
    });

    it("picks a file, uploads it as a JOB, loads the transcript, and navigates with the analysis", async () => {
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///rec.m4a", name: "rec.m4a", size: 2048, mimeType: "audio/m4a" },
        ],
      });
      mockChunkedJob.mockResolvedValueOnce({ jobId: "job_pick" });
      mockGetJob.mockResolvedValueOnce(doneJob(uploadFixture));
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen onAnalyzeDynamics={onAnalyze} />);
      });

      // Pick the file.
      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      // Filename now shown, and the upload button appears.
      expect(JSON.stringify(comp.toJSON())).toContain("rec.m4a");
      expect(queryId(comp, "upload-analyze-button")).toBeTruthy();

      // Upload & analyze.
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });

      // Native file arg is the URI; no context (relationship unselected +
      // blank free text). Consent/store default to false/true when the
      // checkbox was never touched. Even this small file rides the async job
      // path — the direct sync upload is gone (false-failure incident).
      expect(mockUpload).not.toHaveBeenCalled();
      const { file, name, mimeType, size, opts } = chunkedCall();
      expect(file).toBe("file:///rec.m4a");
      expect(name).toBe("rec.m4a");
      expect(mimeType).toBe("audio/m4a");
      expect(size).toBe(2048);
      expect(opts).toEqual(
        expect.objectContaining({ consent: false, store: true }),
      );
      expect(mockGetJob).toHaveBeenCalledWith("job_pick");
      // The server transcript (with timing) landed in the store.
      expect(useSessionStore.getState().turns).toEqual(uploadFixture.turns);
      // Navigated with the ready-made analysis, the (null, since unstored)
      // recording id, and cameFromRecorder=false (a manually-picked file) so
      // Dynamics won't refetch and won't misfire the HD-later popup.
      expect(onAnalyze).toHaveBeenCalledWith(uploadFixture, null, false);
      // Not stored: the honest storage_note is shown, not a fabricated "saved".
      expect(queryId(comp, "stored-note")).toBeNull();
      expect(queryId(comp, "storage-note")).toBeTruthy();
      expect(JSON.stringify(comp.toJSON())).toContain(uploadFixture.storage_note);
      act(() => comp.unmount());
    });

    it("consent checkbox toggles, enables the store switch, and both are wired into the upload", async () => {
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///rec.m4a", name: "rec.m4a", size: 2048, mimeType: "audio/m4a" },
        ],
      });
      mockChunkedJob.mockResolvedValueOnce({
        result: {
          ...uploadFixture,
          stored: true,
          recording_id: "rec_123",
          storage_note: null,
        },
      });
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen onAnalyzeDynamics={onAnalyze} />);
      });

      // Store switch starts disabled (consent unchecked).
      expect(queryId(comp, "store-toggle")!.props.disabled).toBe(true);

      // Check consent.
      act(() => {
        queryId(comp, "consent-checkbox")!.props.onPress();
      });
      expect(queryId(comp, "store-toggle")!.props.disabled).toBe(false);
      // Store defaults on once enabled.
      expect(queryId(comp, "store-toggle")!.props.value).toBe(true);

      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });

      expect(chunkedCall().opts).toEqual(
        expect.objectContaining({ consent: true, store: true }),
      );
      // Stored: the confirmation line shows, recording id threaded through.
      expect(queryId(comp, "stored-note")).toBeTruthy();
      expect(queryId(comp, "storage-note")).toBeNull();
      expect(onAnalyze).toHaveBeenCalledWith(
        expect.objectContaining({ stored: true, recording_id: "rec_123" }),
        "rec_123",
        false,
      );
      act(() => comp.unmount());
    });

    it("shows an honest 422 message when the recording can't be read", async () => {
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///bad.mov", name: "bad.mov", size: 10, mimeType: "video/quicktime" },
        ],
      });
      mockChunkedJob.mockRejectedValueOnce(new Error("API error: 422"));
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen onAnalyzeDynamics={onAnalyze} />);
      });
      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });

      expect(queryId(comp, "upload-error")).toBeTruthy();
      expect(JSON.stringify(comp.toJSON())).toContain("no clear speech found");
      // No navigation on failure — never a fabricated analysis.
      expect(onAnalyze).not.toHaveBeenCalled();
      act(() => comp.unmount());
    });

    it("refuses a >200MB file up front with an honest size message and no network call", async () => {
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          {
            uri: "file:///huge.mov",
            name: "huge.mov",
            size: 250 * MB,
            mimeType: "video/quicktime",
          },
        ],
      });
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen onAnalyzeDynamics={onAnalyze} />);
      });
      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });

      // Honest size message, and NOTHING went to the network.
      expect(queryId(comp, "upload-error")).toBeTruthy();
      expect(JSON.stringify(comp.toJSON())).toContain("the limit is 200 MB");
      expect(mockUpload).not.toHaveBeenCalled();
      expect(mockChunked).not.toHaveBeenCalled();
      expect(mockChunkedJob).not.toHaveBeenCalled();
      expect(onAnalyze).not.toHaveBeenCalled();
      act(() => comp.unmount());
    });

    it("uses the chunked JOB path for a large file, shows a progress bar, polls to done", async () => {
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          {
            uri: "file:///big.mp4",
            name: "big.mp4",
            size: 103 * MB,
            mimeType: "video/mp4",
          },
        ],
      });
      // The chunked-job call drives byte-progress mid-flight, then resolves with
      // a job id to poll. The poll returns "done" immediately (no timers needed).
      mockChunkedJob.mockImplementation(
        (
          _f: unknown,
          _n: unknown,
          _m: unknown,
          _s: unknown,
          opts: { onProgress?: (f: number) => void },
        ) => {
          opts.onProgress?.(0.5);
          return Promise.resolve({ jobId: "job_1" });
        },
      );
      mockGetJob.mockResolvedValueOnce(doneJob(uploadFixture));
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen onAnalyzeDynamics={onAnalyze} />);
      });
      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });

      // The JOB path (not the synchronous chunked path) was called with the
      // size + consent/store opts (no context — relationship unselected).
      expect(mockUpload).not.toHaveBeenCalled();
      expect(mockChunked).not.toHaveBeenCalled();
      expect(mockChunkedJob).toHaveBeenCalledTimes(1);
      const call = mockChunkedJob.mock.calls[0];
      expect(call[0]).toBe("file:///big.mp4");
      expect(call[3]).toBe(103 * MB);
      expect(call[4]).toEqual(
        expect.objectContaining({
          consent: false,
          store: true,
          context: undefined,
        }),
      );
      // Polled the returned job and handed the ready-made analysis to Dynamics.
      expect(mockGetJob).toHaveBeenCalledWith("job_1");
      expect(onAnalyze).toHaveBeenCalledWith(uploadFixture, null, false);
      act(() => comp.unmount());
    });

    it("falls back to the synchronous chunked result when the job endpoint is unavailable", async () => {
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///big.mp4", name: "big.mp4", size: 103 * MB, mimeType: "video/mp4" },
        ],
      });
      // The client already fell back internally (old server / storage off) and
      // returns the finished result directly — no job to poll.
      mockChunkedJob.mockResolvedValueOnce({ result: uploadFixture });
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen onAnalyzeDynamics={onAnalyze} />);
      });
      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });

      // No poll (result came back directly), still navigates with the analysis.
      expect(mockGetJob).not.toHaveBeenCalled();
      expect(onAnalyze).toHaveBeenCalledWith(uploadFixture, null, false);
      act(() => comp.unmount());
    });

    // NOTE: the old "keeps the direct path (no chunking) for a small file"
    // test is gone deliberately — small files now take the chunked JOB path
    // too (see the "always-async uploads" suite for the incident and the
    // replacement assertion).
  });

  describe("preselected recording (from the in-app recorder)", () => {
    it("consumes the recorder store's pending file into the normal upload flow and flags it as recorder-origin", async () => {
      // A clip just recorded in-app, handed over via the recorder store.
      act(() => {
        useRecorderStore.setState({
          pendingFile: {
            uri: "file:///recorded.mp4",
            name: "mindshift-123.mp4",
            mimeType: "video/mp4",
            size: 3 * MB,
          },
        });
      });
      mockChunkedJob.mockResolvedValueOnce({
        result: {
          ...uploadFixture,
          stored: true,
          recording_id: "rec_rec",
          storage_note: null,
        },
      });
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen onAnalyzeDynamics={onAnalyze} />);
      });

      // The recorded file is preselected — its name shows and the upload button
      // is ready without touching the document picker. The store was cleared.
      expect(JSON.stringify(comp.toJSON())).toContain("mindshift-123.mp4");
      expect(queryId(comp, "upload-analyze-button")).toBeTruthy();
      expect(mockPick).not.toHaveBeenCalled();
      expect(useRecorderStore.getState().pendingFile).toBeNull();

      // Upload & analyze goes down the normal (job) path with the recorded file...
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });
      const { file, name, mimeType, size, opts } = chunkedCall();
      expect(file).toBe("file:///recorded.mp4");
      expect(name).toBe("mindshift-123.mp4");
      expect(mimeType).toBe("video/mp4");
      expect(size).toBe(3 * MB);
      expect(opts).toEqual(
        expect.objectContaining({ consent: false, store: true }),
      );
      // ...and the handoff marks it recorder-origin (third arg true) so Dynamics
      // can offer the HD-later popup.
      expect(onAnalyze).toHaveBeenCalledWith(
        expect.objectContaining({ recording_id: "rec_rec" }),
        "rec_rec",
        true,
      );
      act(() => comp.unmount());
    });
  });

  // The upload/link segmented control acts on tap — it doesn't just toggle a
  // selection (the reported confusion: a user kept tapping "Upload file"
  // expecting a picker, but it only changed the highlight). "Upload file" opens
  // the OS picker in the same tap; "Paste link" drops the cursor in the URL field.
  describe("action tabs (tap acts, doesn't just select)", () => {
    it("tapping 'Upload file' opens the OS file picker immediately — no separate step", async () => {
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///rec.m4a", name: "rec.m4a", size: 2048, mimeType: "audio/m4a" },
        ],
      });
      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen />);
      });

      // One tap on the tab opens the picker and lands the file — no intervening
      // "Choose a recording" press.
      await act(async () => {
        queryId(comp, "mode-file-tab")!.props.onPress();
      });

      expect(mockPick).toHaveBeenCalledTimes(1);
      expect(JSON.stringify(comp.toJSON())).toContain("rec.m4a");
      expect(queryId(comp, "picked-file")).toBeTruthy();
      expect(queryId(comp, "upload-analyze-button")).toBeTruthy();
      act(() => comp.unmount());
    });

    it("canceling the picker keeps file mode with the pick-a-file affordance visible", async () => {
      mockPick.mockResolvedValueOnce({ canceled: true });
      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen />);
      });

      await act(async () => {
        queryId(comp, "mode-file-tab")!.props.onPress();
      });

      // The picker was asked for, the user backed out — we stay in file mode with
      // the retry affordance visible and no file selected (never a phantom pick).
      expect(mockPick).toHaveBeenCalledTimes(1);
      expect(queryId(comp, "pick-recording-button")).toBeTruthy();
      expect(queryId(comp, "picked-file")).toBeNull();
      expect(queryId(comp, "link-input")).toBeNull();
      act(() => comp.unmount());
    });

    it("re-tapping the already-active 'Upload file' tab opens the picker again", async () => {
      mockPick
        .mockResolvedValueOnce({
          canceled: false,
          assets: [
            { uri: "file:///a.m4a", name: "a.m4a", size: 2048, mimeType: "audio/m4a" },
          ],
        })
        .mockResolvedValueOnce({
          canceled: false,
          assets: [
            { uri: "file:///b.m4a", name: "b.m4a", size: 4096, mimeType: "audio/m4a" },
          ],
        });
      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen />);
      });

      // First tap picks a.m4a...
      await act(async () => {
        queryId(comp, "mode-file-tab")!.props.onPress();
      });
      expect(JSON.stringify(comp.toJSON())).toContain("a.m4a");

      // ...re-tapping the same (already-active) tab opens the picker again and
      // swaps in b.m4a — the natural expectation.
      await act(async () => {
        queryId(comp, "mode-file-tab")!.props.onPress();
      });
      expect(mockPick).toHaveBeenCalledTimes(2);
      expect(JSON.stringify(comp.toJSON())).toContain("b.m4a");
      act(() => comp.unmount());
    });

    it("tapping 'Paste link' switches mode AND autofocuses the URL input", () => {
      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen />);
      });

      // File mode by default: no URL field yet.
      expect(queryId(comp, "link-input")).toBeNull();

      act(() => {
        queryId(comp, "mode-link-tab")!.props.onPress();
      });

      // The URL field is now up and set to autofocus. It renders only in link
      // mode, so autoFocus fires on this very mount — keyboard up on mobile,
      // cursor in the field on web. The tab acts, it doesn't just select.
      const linkInput = queryId(comp, "link-input");
      expect(linkInput).toBeTruthy();
      expect(linkInput!.props.autoFocus).toBe(true);
      act(() => comp.unmount());
    });
  });

  describe("analyze a link", () => {
    it("toggles to link mode: swaps the file picker for the URL input", async () => {
      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen />);
      });

      // File mode by default: picker present, link input absent.
      expect(queryId(comp, "pick-recording-button")).toBeTruthy();
      expect(queryId(comp, "link-input")).toBeNull();

      // Switch to link mode via the toggle.
      act(() => {
        queryId(comp, "mode-link-tab")!.props.onPress();
      });
      expect(queryId(comp, "link-input")).toBeTruthy();
      expect(queryId(comp, "analyze-link-button")).toBeTruthy();
      expect(queryId(comp, "pick-recording-button")).toBeNull();
      // The URL field must not autocapitalize.
      expect(queryId(comp, "link-input")!.props.autoCapitalize).toBe("none");
      // Helper text names the Drive/Photos guidance (Photos is now supported).
      expect(JSON.stringify(comp.toJSON())).toContain(
        "Google Photos share links (single video)",
      );
      act(() => comp.unmount());
    });

    it("submits a link JOB, polls to done, hydrates the transcript, navigates", async () => {
      mockLinkJob.mockResolvedValueOnce({ job_id: "job_link" });
      mockGetJob.mockResolvedValueOnce(
        doneJob({
          ...uploadFixture,
          stored: true,
          recording_id: "rec_link",
          storage_note: null,
        }),
      );
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen onAnalyzeDynamics={onAnalyze} />);
      });

      // Consent so store lands true, then switch to link mode and type a URL.
      act(() => {
        queryId(comp, "consent-checkbox")!.props.onPress();
        queryId(comp, "mode-link-tab")!.props.onPress();
      });
      act(() => {
        queryId(comp, "link-input")!.props.onChangeText(
          "https://drive.google.com/file/d/abc",
        );
      });

      await act(async () => {
        queryId(comp, "analyze-link-button")!.props.onPress();
      });

      // The JOB endpoint (not the synchronous link) was used, then polled.
      // No context — the relationship picker was left unselected.
      expect(mockLink).not.toHaveBeenCalled();
      expect(mockLinkJob).toHaveBeenCalledWith(
        "https://drive.google.com/file/d/abc",
        { consent: true, store: true, context: undefined },
      );
      expect(mockGetJob).toHaveBeenCalledWith("job_link");
      // Same handoff as upload: transcript in the store, navigation with the id.
      expect(useSessionStore.getState().turns).toEqual(uploadFixture.turns);
      expect(onAnalyze).toHaveBeenCalledWith(
        expect.objectContaining({ recording_id: "rec_link" }),
        "rec_link",
      );
      expect(queryId(comp, "stored-note")).toBeTruthy();
      act(() => comp.unmount());
    });

    it("falls back to the synchronous link when the job endpoint 404s (old server)", async () => {
      mockLinkJob.mockRejectedValueOnce(
        Object.assign(new Error("API error: 404"), { status: 404 }),
      );
      mockLink.mockResolvedValueOnce(uploadFixture);
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen onAnalyzeDynamics={onAnalyze} />);
      });
      act(() => {
        queryId(comp, "mode-link-tab")!.props.onPress();
      });
      act(() => {
        queryId(comp, "link-input")!.props.onChangeText(
          "https://example.com/clip.mp4",
        );
      });
      await act(async () => {
        queryId(comp, "analyze-link-button")!.props.onPress();
      });

      // Job submit failed with 404 → fell back to the synchronous endpoint, no poll.
      expect(mockLinkJob).toHaveBeenCalledTimes(1);
      expect(mockLink).toHaveBeenCalledTimes(1);
      expect(mockGetJob).not.toHaveBeenCalled();
      expect(onAnalyze).toHaveBeenCalledWith(uploadFixture, null);
      act(() => comp.unmount());
    });

    it("shows a failed job's honest error verbatim", async () => {
      const serverMsg =
        "That link isn’t a direct file link — use a direct file URL, a Google " +
        "Drive share link, or a Google Photos share link of a single video.";
      mockLinkJob.mockResolvedValueOnce({ job_id: "job_bad" });
      mockGetJob.mockResolvedValueOnce({
        job_id: "job_bad",
        status: "failed",
        created_at: "2026-07-12T00:00:00Z",
        updated_at: "2026-07-12T00:00:02Z",
        stage_started_at: "2026-07-12T00:00:02Z",
        progress_note: null,
        duration_seconds: null,
        error: serverMsg,
        result: null,
      } as AnalyzeJobState);
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen onAnalyzeDynamics={onAnalyze} />);
      });
      act(() => {
        queryId(comp, "mode-link-tab")!.props.onPress();
      });
      act(() => {
        queryId(comp, "link-input")!.props.onChangeText(
          "https://photos.google.com/share/xyz",
        );
      });
      await act(async () => {
        queryId(comp, "analyze-link-button")!.props.onPress();
      });

      expect(queryId(comp, "upload-error")).toBeTruthy();
      // The server's honest job error is shown verbatim — never fabricated.
      expect(JSON.stringify(comp.toJSON())).toContain(serverMsg);
      expect(onAnalyze).not.toHaveBeenCalled();
      act(() => comp.unmount());
    });

    it("renders the staged job-progress card with a stage label and an ETA", async () => {
      jest.useFakeTimers();
      try {
        mockLinkJob.mockResolvedValueOnce({ job_id: "job_eta" });
        // First poll: mid-flight "analyzing" with a known duration → the loop
        // renders the card, then parks on its (fake) 3s sleep before polling again.
        mockGetJob.mockResolvedValue({
          job_id: "job_eta",
          status: "analyzing",
          created_at: "2026-07-12T00:00:00Z",
          updated_at: "2026-07-12T00:00:02Z",
          stage_started_at: "2026-07-12T00:00:02Z",
          progress_note: "scoring the conversation",
          duration_seconds: 120,
          error: null,
          result: null,
        } as AnalyzeJobState);
        const onAnalyze = jest.fn();

        let comp!: renderer.ReactTestRenderer;
        act(() => {
          comp = renderer.create(<AnalyzeScreen onAnalyzeDynamics={onAnalyze} />);
        });
        act(() => {
          queryId(comp, "mode-link-tab")!.props.onPress();
        });
        act(() => {
          queryId(comp, "link-input")!.props.onChangeText(
            "https://example.com/clip.mp4",
          );
        });
        await act(async () => {
          queryId(comp, "analyze-link-button")!.props.onPress();
        });

        // The staged progress card is up with the stage label, the server's note,
        // and a rough ETA labeled an estimate.
        expect(queryId(comp, "job-progress")).toBeTruthy();
        const json = JSON.stringify(comp.toJSON());
        expect(json).toContain("Analyzing…");
        expect(json).toContain("scoring the conversation");
        expect(json).toContain("remaining (estimate)");
        act(() => comp.unmount());
      } finally {
        jest.clearAllTimers();
        jest.useRealTimers();
      }
    });
  });

  describe("naming a conversation", () => {
    it("sends the typed title with a file upload", async () => {
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///rec.m4a", name: "rec.m4a", size: 2048, mimeType: "audio/m4a" },
        ],
      });
      mockChunkedJob.mockResolvedValueOnce({ result: uploadFixture });

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen />);
      });
      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      // The "Name this conversation" field sits by the context box.
      act(() =>
        queryId(comp, "conversation-title-input")!.props.onChangeText(
          "Sunday budget talk",
        ),
      );
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });

      expect(chunkedCall().opts).toEqual(
        expect.objectContaining({ title: "Sunday budget talk" }),
      );
      act(() => comp.unmount());
    });

    it("sends the typed title with a link analysis", async () => {
      mockLinkJob.mockResolvedValueOnce({ job_id: "job_t" });
      mockGetJob.mockResolvedValueOnce(doneJob(uploadFixture));

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen />);
      });
      act(() => queryId(comp, "mode-link-tab")!.props.onPress());
      act(() =>
        queryId(comp, "link-input")!.props.onChangeText(
          "https://example.com/clip.mp4",
        ),
      );
      act(() =>
        queryId(comp, "conversation-title-input")!.props.onChangeText(
          "Mediation session 3",
        ),
      );
      await act(async () => {
        queryId(comp, "analyze-link-button")!.props.onPress();
      });

      expect(mockLinkJob).toHaveBeenCalledWith(
        "https://example.com/clip.mp4",
        expect.objectContaining({ title: "Mediation session 3" }),
      );
      act(() => comp.unmount());
    });
  });

  describe("download progress + honest stall UX (Bug D)", () => {
    it("renders the download byte-progress bar when the server reports bytes", async () => {
      jest.useFakeTimers();
      try {
        mockLinkJob.mockResolvedValueOnce({ job_id: "job_dl" });
        mockGetJob.mockResolvedValue({
          job_id: "job_dl",
          status: "downloading",
          created_at: "2026-07-12T00:00:00Z",
          updated_at: "2026-07-12T00:00:02Z",
          stage_started_at: "2026-07-12T00:00:02Z",
          progress_note: "fetching video",
          duration_seconds: null,
          bytes_downloaded: 50 * MB,
          bytes_total: 116 * MB,
          error: null,
          result: null,
        } as AnalyzeJobState);

        let comp!: renderer.ReactTestRenderer;
        act(() => {
          comp = renderer.create(<AnalyzeScreen />);
        });
        act(() => queryId(comp, "mode-link-tab")!.props.onPress());
        act(() =>
          queryId(comp, "link-input")!.props.onChangeText(
            "https://example.com/big.mp4",
          ),
        );
        await act(async () => {
          queryId(comp, "analyze-link-button")!.props.onPress();
        });

        expect(queryId(comp, "job-progress")).toBeTruthy();
        expect(queryId(comp, "download-progress")).toBeTruthy();
        const json = JSON.stringify(comp.toJSON());
        expect(json).toContain("Uploading…");
        // Honest bytes, never a fabricated percentage-only readout.
        expect(json).toContain("of");
        expect(json).toContain("116.0 MB");
        act(() => comp.unmount());
      } finally {
        jest.clearAllTimers();
        jest.useRealTimers();
      }
    });

    it("softens a computed 'stalled' to a still-working note and keeps polling (no error)", async () => {
      jest.useFakeTimers();
      try {
        mockLinkJob.mockResolvedValueOnce({ job_id: "job_stall" });
        // The server computes 'stalled' with its harsh note — the client must
        // NOT treat the first stalled poll as failure.
        mockGetJob.mockResolvedValue({
          job_id: "job_stall",
          status: "stalled",
          created_at: "2026-07-12T00:00:00Z",
          updated_at: "2026-07-12T00:00:02Z",
          stage_started_at: "2026-07-12T00:00:02Z",
          progress_note:
            "the analysis appears to have stalled — it may have been " +
            "interrupted; try again",
          duration_seconds: null,
          error: null,
          result: null,
        } as AnalyzeJobState);

        let comp!: renderer.ReactTestRenderer;
        act(() => {
          comp = renderer.create(<AnalyzeScreen />);
        });
        act(() => queryId(comp, "mode-link-tab")!.props.onPress());
        act(() =>
          queryId(comp, "link-input")!.props.onChangeText(
            "https://example.com/clip.mp4",
          ),
        );
        await act(async () => {
          queryId(comp, "analyze-link-button")!.props.onPress();
        });

        // The job card is still up (polling continues); the note is softened and
        // the harsh server "stalled…try again" text is NOT shown. No error yet.
        expect(queryId(comp, "job-progress")).toBeTruthy();
        expect(queryId(comp, "upload-error")).toBeNull();
        const json = JSON.stringify(comp.toJSON());
        expect(json).toContain("Still working");
        expect(json).not.toContain("appears to have stalled");
        act(() => comp.unmount());
      } finally {
        jest.clearAllTimers();
        jest.useRealTimers();
      }
    });
  });

  // --- Upload failure diagnostics (the vc29 phone-upload bug) ---------------
  // A fetch network rejection used to surface as the generic "Something went
  // wrong" with nothing to correlate against server logs. The screen must now
  // show the honest offline message, the request id, and collapsible details.
  describe("upload failure diagnostics", () => {
    const pickSmall = () =>
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///rec.m4a", name: "rec.m4a", size: 2048, mimeType: "audio/m4a" },
        ],
      });

    async function renderAndUpload(onAnalyze = jest.fn()) {
      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen onAnalyzeDynamics={onAnalyze} />);
      });
      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });
      return comp;
    }

    it("renders the honest offline message and the request id when the upload can't reach the server", async () => {
      pickSmall();
      mockChunkedJob.mockRejectedValueOnce(
        new UploadError("Upload network failure: Network request failed", {
          status: 0,
          requestId: "req-abc123",
          path: "chunked",
          bytes: 2048,
          elapsedMs: 1234,
          causeName: "TypeError",
          causeMessage: "Network request failed",
        }),
      );
      const onAnalyze = jest.fn();
      const comp = await renderAndUpload(onAnalyze);

      const json = JSON.stringify(comp.toJSON());
      // Honest, actionable message — not the generic line.
      expect(json).toContain("check your connection");
      expect(json).not.toContain("Something went wrong");
      // The request id is visible so a screenshot correlates with server logs.
      expect(queryId(comp, "upload-error-request-id")).toBeTruthy();
      expect(json).toContain("req-abc123");
      // No navigation on failure — never a fabricated analysis.
      expect(onAnalyze).not.toHaveBeenCalled();
      // The failure was reported to /client-log with the diagnosis (request id,
      // status, extension + size — never the personal file NAME).
      expect(mockReport).toHaveBeenCalledWith(
        "upload-failure",
        expect.objectContaining({
          requestId: "req-abc123",
          status: 0,
          path: "chunked",
          fileExt: ".m4a",
          fileSize: 2048,
          causeName: "TypeError",
        }),
      );
      expect(JSON.stringify(mockReport.mock.calls[0])).not.toContain("rec.m4a");
      act(() => comp.unmount());
    });

    it("keeps the collapsed error details available and expands them on tap", async () => {
      pickSmall();
      mockChunkedJob.mockRejectedValueOnce(
        new UploadError("Upload network failure: Network request failed", {
          status: 0,
          requestId: "req-details",
          path: "chunked",
          chunkIndex: 3,
          bytes: 103 * MB,
          elapsedMs: 5231,
          causeName: "TypeError",
          causeMessage: "Network request failed",
        }),
      );
      const comp = await renderAndUpload();

      // Collapsed by default; the toggle is there.
      expect(queryId(comp, "upload-error-details")).toBeNull();
      expect(queryId(comp, "upload-error-details-toggle")).toBeTruthy();

      act(() => queryId(comp, "upload-error-details-toggle")!.props.onPress());

      const details = queryId(comp, "upload-error-details");
      expect(details).toBeTruthy();
      const json = JSON.stringify(comp.toJSON());
      expect(json).toContain("chunked");
      expect(json).toContain("Network request failed");
      expect(json).toContain("3");
      act(() => comp.unmount());
    });

    it("keeps the mapped message for an HTTP failure but still shows the request id", async () => {
      pickSmall();
      mockChunkedJob.mockRejectedValueOnce(
        new UploadError("API error: 413", {
          status: 413,
          requestId: "req-413413",
          path: "chunked",
          bytes: 2048,
          elapsedMs: 900,
        }),
      );
      const comp = await renderAndUpload();

      const json = JSON.stringify(comp.toJSON());
      // The chunked 413 mapping names the actual size and the hard cap.
      expect(json).toContain("the limit is 200 MB");
      expect(json).toContain("req-413413");
      act(() => comp.unmount());
    });
  });

  // --- Unknown-size routing (stat via expo-file-system) ---------------------
  // The picker sometimes reports no size. The file is statted so the 200MB
  // pre-flight refusal and the chunked client's chunk math rest on facts; the
  // statted (or still-unknown) size is passed to the job path, which every
  // upload now takes regardless.
  describe("unknown-size routing", () => {
    afterEach(() => {
      delete (globalThis as Record<string, unknown>).__fsMockSize;
    });

    const pickNoSize = () =>
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///nosize.mp4", name: "nosize.mp4", mimeType: "video/mp4" },
        ],
      });

    async function renderAndUpload(onAnalyze = jest.fn()) {
      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen onAnalyzeDynamics={onAnalyze} />);
      });
      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });
      return comp;
    }

    it("stats the file via expo-file-system and routes a large one to the chunked path", async () => {
      pickNoSize();
      (globalThis as Record<string, unknown>).__fsMockSize = 103 * MB;
      mockChunkedJob.mockResolvedValueOnce({ result: uploadFixture });

      const comp = await renderAndUpload();

      expect(mockUpload).not.toHaveBeenCalled();
      expect(mockChunkedJob).toHaveBeenCalledTimes(1);
      expect(mockChunkedJob.mock.calls[0][3]).toBe(103 * MB);
      act(() => comp.unmount());
    });

    it("stats a small file and passes the found size to the chunked job path", async () => {
      pickNoSize();
      (globalThis as Record<string, unknown>).__fsMockSize = 2 * MB;
      mockChunkedJob.mockResolvedValueOnce({ result: uploadFixture });

      const comp = await renderAndUpload();

      // Small files ride the job path too now — with the statted size, so the
      // chunked client's chunk math rests on facts.
      expect(mockUpload).not.toHaveBeenCalled();
      expect(mockChunkedJob).toHaveBeenCalledTimes(1);
      expect(mockChunkedJob.mock.calls[0][3]).toBe(2 * MB);
      act(() => comp.unmount());
    });

    it("routes a STILL-unknown size to the CHUNKED path — never direct", async () => {
      pickNoSize();
      (globalThis as Record<string, unknown>).__fsMockSize = null;
      mockChunkedJob.mockResolvedValueOnce({ result: uploadFixture });

      const comp = await renderAndUpload();

      // Direct would die at the ingress with no server trace; chunked carries
      // the request id and fails honestly if the size truly can't be found.
      expect(mockUpload).not.toHaveBeenCalled();
      expect(mockChunkedJob).toHaveBeenCalledTimes(1);
      expect(mockChunkedJob.mock.calls[0][3]).toBeNull();
      act(() => comp.unmount());
    });
  });

  // --- Always-async uploads (the 2026-08-15 false-failure incident) ----------
  // A SMALL file used to take the synchronous /analyze/upload path. A cold
  // Cloud Run instance takes ~3 minutes to analyze (voice-model load) — the
  // phone's fetch gave up on the held-open response and showed "Couldn't reach
  // the server" while the server completed AND stored the recording (request id
  // d883883b… logged 200 server-side). Every upload now goes through the
  // chunked ASYNC JOB path, which holds no response open.
  describe("always-async uploads (false-failure incident)", () => {
    it("routes even a small file through the chunked JOB path — postAnalyzeUpload is never called", async () => {
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///small.m4a", name: "small.m4a", size: 2 * MB, mimeType: "audio/m4a" },
        ],
      });
      mockChunkedJob.mockResolvedValueOnce({ jobId: "job_small" });
      mockGetJob.mockResolvedValueOnce(doneJob(uploadFixture));
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen onAnalyzeDynamics={onAnalyze} />);
      });
      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });

      // The job path was used with the small file's true size; the synchronous
      // direct upload (the false-failure path) was never touched.
      expect(mockUpload).not.toHaveBeenCalled();
      expect(mockChunkedJob).toHaveBeenCalledTimes(1);
      expect(mockChunkedJob.mock.calls[0][0]).toBe("file:///small.m4a");
      expect(mockChunkedJob.mock.calls[0][3]).toBe(2 * MB);
      // Completed as a polled job and navigated with the ready-made analysis.
      expect(mockGetJob).toHaveBeenCalledWith("job_small");
      expect(onAnalyze).toHaveBeenCalledWith(uploadFixture, null, false);
      act(() => comp.unmount());
    });
  });

  // --- Consuming rescued recorder output after a successful analysis ---------
  // RecoveryPrompt offers orphaned stitched files from recorder-out/. After a
  // SUCCESSFUL upload+analysis nothing deleted the file, so it was re-offered
  // forever — inviting duplicate analyses. The success path must discard it.
  describe("consuming recorder output after successful analysis", () => {
    const ORPHAN_URI = "file:///docs/recorder-out/mindshift-audio-99.wav";

    /** A recorder-store seam (the same test seam RecoveryPrompt uses) whose
     *  discardOrphan is observable. list* return empty so no prompt renders. */
    function makeStoreSeam(discardOrphan: jest.Mock = jest.fn()) {
      const store = {
        listRecoverable: jest.fn(() => []),
        listOrphanStitched: jest.fn(() => []),
        discardOrphan,
      };
      return {
        store,
        deps: { store } as unknown as AudioRecorderDeps,
      };
    }

    async function pickAndUpload(
      comp: renderer.ReactTestRenderer,
    ): Promise<void> {
      await act(async () => {
        queryId(comp, "pick-recording-button")!.props.onPress();
      });
      await act(async () => {
        queryId(comp, "upload-analyze-button")!.props.onPress();
      });
    }

    it("discards the orphan from the store when a recorder-out/ file analyzes successfully", async () => {
      const { store, deps } = makeStoreSeam();
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          {
            uri: ORPHAN_URI,
            name: "mindshift-audio-99.wav",
            size: 2 * MB,
            mimeType: "audio/wav",
          },
        ],
      });
      mockChunkedJob.mockResolvedValueOnce({ result: uploadFixture });
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(
          <AnalyzeScreen onAnalyzeDynamics={onAnalyze} recorderDeps={deps} />,
        );
      });
      await pickAndUpload(comp);

      // The success path consumed the rescued file — it can't be re-offered.
      expect(store.discardOrphan).toHaveBeenCalledWith(ORPHAN_URI);
      expect(onAnalyze).toHaveBeenCalledWith(uploadFixture, null, false);
      act(() => comp.unmount());
    });

    it("leaves files that did NOT come from recorder-out/ alone", async () => {
      const { store, deps } = makeStoreSeam();
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///rec.m4a", name: "rec.m4a", size: 2048, mimeType: "audio/m4a" },
        ],
      });
      mockChunkedJob.mockResolvedValueOnce({ result: uploadFixture });

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen recorderDeps={deps} />);
      });
      await pickAndUpload(comp);

      expect(store.discardOrphan).not.toHaveBeenCalled();
      act(() => comp.unmount());
    });

    it("does not discard anything when the upload FAILS", async () => {
      const { store, deps } = makeStoreSeam();
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          {
            uri: ORPHAN_URI,
            name: "mindshift-audio-99.wav",
            size: 2 * MB,
            mimeType: "audio/wav",
          },
        ],
      });
      mockChunkedJob.mockRejectedValueOnce(new Error("API error: 502"));

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<AnalyzeScreen recorderDeps={deps} />);
      });
      await pickAndUpload(comp);

      // The rescued audio survives a failed upload — it must be re-offered.
      expect(store.discardOrphan).not.toHaveBeenCalled();
      expect(queryId(comp, "upload-error")).toBeTruthy();
      act(() => comp.unmount());
    });

    it("a throwing discardOrphan never breaks the success flow", async () => {
      const { deps } = makeStoreSeam(
        jest.fn(() => {
          throw new Error("disk error");
        }),
      );
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          {
            uri: ORPHAN_URI,
            name: "mindshift-audio-99.wav",
            size: 2 * MB,
            mimeType: "audio/wav",
          },
        ],
      });
      mockChunkedJob.mockResolvedValueOnce({ result: uploadFixture });
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(
          <AnalyzeScreen onAnalyzeDynamics={onAnalyze} recorderDeps={deps} />,
        );
      });
      await pickAndUpload(comp);

      // Success flow untouched: navigation happened, no error shown.
      expect(onAnalyze).toHaveBeenCalledWith(uploadFixture, null, false);
      expect(queryId(comp, "upload-error")).toBeNull();
      act(() => comp.unmount());
    });
  });
});
