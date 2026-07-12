import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import * as DocumentPicker from "expo-document-picker";
import SessionScreen from "../src/screens/SessionScreen";
import { useSessionStore } from "../src/store/sessionStore";
import { postAnalyzeUpload } from "../src/api/client";
import type { UploadAnalyzeResult } from "../src/api/client";

// Keep the real client (the store uses postRespond) but stub the upload call.
jest.mock("../src/api/client", () => ({
  __esModule: true,
  ...jest.requireActual("../src/api/client"),
  postAnalyzeUpload: jest.fn(),
}));

const mockPick = DocumentPicker.getDocumentAsync as jest.Mock;
const mockUpload = postAnalyzeUpload as jest.Mock;

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

// Reset store between tests
beforeEach(() => {
  mockPick.mockReset();
  mockUpload.mockReset();
  act(() => {
    useSessionStore.setState({
      role: "Husband / Wife",
      empathyLevel: 50,
      turns: [],
      suggestions: [],
      loading: false,
    });
  });
});

describe("SessionScreen", () => {
  it("renders the initial screen", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<SessionScreen />);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders with turns in the transcript", () => {
    act(() => {
      useSessionStore.setState({
        turns: [
          { speaker: "Alice", text: "I feel like you never listen to me." },
          { speaker: "Bob", text: "That's not fair, I always try." },
        ],
      });
    });

    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<SessionScreen />);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders with suggestions", () => {
    act(() => {
      useSessionStore.setState({
        turns: [
          { speaker: "Alice", text: "You never help around the house." },
        ],
        suggestions: [
          {
            text: "I hear that you're feeling overwhelmed with housework.",
            tone: "empathetic",
          },
          {
            text: "Let's talk about how we can split things more evenly.",
            tone: "balanced",
          },
          {
            text: "I understand. What would help most right now?",
            tone: "validating",
          },
        ],
      });
    });

    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<SessionScreen />);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("shows the analyze-dynamics button only at >= 4 turns", () => {
    const mkTurns = (n: number) =>
      Array.from({ length: n }, (_, i) => ({
        speaker: i % 2 === 0 ? "Alice" : "Bob",
        text: `turn ${i}`,
      }));
    const hasButton = (comp: renderer.ReactTestRenderer) =>
      comp.root.findAll((x) => x.props?.testID === "analyze-dynamics-button")
        .length > 0;

    // 3 turns: below threshold, hidden even with a handler.
    act(() => {
      useSessionStore.setState({ turns: mkTurns(3) });
    });
    let three!: renderer.ReactTestRenderer;
    act(() => {
      three = renderer.create(<SessionScreen onAnalyzeDynamics={() => {}} />);
    });
    expect(hasButton(three)).toBe(false);

    // 4 turns: shown, and pressing it invokes the handler.
    act(() => {
      useSessionStore.setState({ turns: mkTurns(4) });
    });
    const onAnalyze = jest.fn();
    let four!: renderer.ReactTestRenderer;
    act(() => {
      four = renderer.create(<SessionScreen onAnalyzeDynamics={onAnalyze} />);
    });
    expect(hasButton(four)).toBe(true);
    act(() => {
      four.root
        .find((x) => x.props?.testID === "analyze-dynamics-button")
        .props.onPress();
    });
    expect(onAnalyze).toHaveBeenCalledTimes(1);
  });

  it("renders loading state", () => {
    act(() => {
      useSessionStore.setState({
        turns: [{ speaker: "Alice", text: "We need to talk." }],
        loading: true,
      });
    });

    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<SessionScreen />);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  describe("analyze a recording", () => {
    it("renders the pick-recording button", () => {
      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<SessionScreen />);
      });
      expect(queryId(comp, "pick-recording-button")).toBeTruthy();
      // The upload button only appears once a file is picked.
      expect(queryId(comp, "upload-analyze-button")).toBeNull();
      act(() => comp.unmount());
    });

    it("picks a file, uploads it, loads the transcript, and navigates with the analysis", async () => {
      mockPick.mockResolvedValueOnce({
        canceled: false,
        assets: [
          { uri: "file:///rec.m4a", name: "rec.m4a", size: 2048, mimeType: "audio/m4a" },
        ],
      });
      mockUpload.mockResolvedValueOnce(uploadFixture);
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<SessionScreen onAnalyzeDynamics={onAnalyze} />);
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

      // Native file arg is the URI; no context passed (blank). Consent/store
      // default to false/true when the checkbox was never touched.
      expect(mockUpload).toHaveBeenCalledTimes(1);
      expect(mockUpload).toHaveBeenCalledWith(
        "file:///rec.m4a",
        "rec.m4a",
        "audio/m4a",
        undefined,
        { consent: false, store: true },
      );
      // The server transcript (with timing) landed in the store.
      expect(useSessionStore.getState().turns).toEqual(uploadFixture.turns);
      // Navigated with the ready-made analysis and the (null, since unstored)
      // recording id so Dynamics won't refetch.
      expect(onAnalyze).toHaveBeenCalledWith(uploadFixture, null);
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
      mockUpload.mockResolvedValueOnce({
        ...uploadFixture,
        stored: true,
        recording_id: "rec_123",
        storage_note: null,
      });
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<SessionScreen onAnalyzeDynamics={onAnalyze} />);
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

      expect(mockUpload).toHaveBeenCalledWith(
        "file:///rec.m4a",
        "rec.m4a",
        "audio/m4a",
        undefined,
        { consent: true, store: true },
      );
      // Stored: the confirmation line shows, recording id threaded through.
      expect(queryId(comp, "stored-note")).toBeTruthy();
      expect(queryId(comp, "storage-note")).toBeNull();
      expect(onAnalyze).toHaveBeenCalledWith(
        expect.objectContaining({ stored: true, recording_id: "rec_123" }),
        "rec_123",
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
      mockUpload.mockRejectedValueOnce(new Error("API error: 422"));
      const onAnalyze = jest.fn();

      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(<SessionScreen onAnalyzeDynamics={onAnalyze} />);
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
  });
});
