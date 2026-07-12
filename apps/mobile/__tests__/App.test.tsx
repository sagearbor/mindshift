import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import * as DocumentPicker from "expo-document-picker";
import App from "../App";
import { useAuthStore } from "../src/store/authStore";
import { useSessionStore } from "../src/store/sessionStore";
import { useAnalyzeStore } from "../src/store/analyzeStore";
import { postAnalyzeUpload } from "../src/api/client";
import type { UploadAnalyzeResult } from "../src/api/client";

// Keep the real client (postAnalyze and the recordings API run through fetch)
// but stub the multipart upload call — its FormData plumbing is covered by
// client.test.ts; here we care about the navigation wiring around its result.
jest.mock("../src/api/client", () => ({
  __esModule: true,
  ...jest.requireActual("../src/api/client"),
  postAnalyzeUpload: jest.fn(),
}));

const mockFetch = global.fetch as jest.Mock;
const mockPick = DocumentPicker.getDocumentAsync as jest.Mock;
const mockUpload = postAnalyzeUpload as jest.Mock;

/** The firebase/auth mock state from jest-setup. */
interface FirebaseAuthMock {
  currentUser: unknown;
  idTokenListener: ((user: unknown) => void | Promise<void>) | null;
}
const authMock = (globalThis as Record<string, unknown>)
  .__firebaseAuthMock as FirebaseAuthMock;

function fakeUser() {
  return {
    uid: "u1",
    email: "user@example.com",
    displayName: "Test User",
    getIdToken: jest.fn().mockResolvedValue("id-token"),
  };
}

/** First node carrying the given testID, or null. */
function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

/** Boot the app and resolve auth as signed in — every navigation test starts
 *  on the two-mode Home screen. */
async function signIn(comp: renderer.ReactTestRenderer) {
  const user = fakeUser();
  await act(async () => {
    authMock.currentUser = user;
    await authMock.idTokenListener?.(user);
  });
  expect(queryId(comp, "home-live-coach")).toBeTruthy();
}

beforeEach(() => {
  authMock.currentUser = null;
  mockPick.mockReset();
  mockUpload.mockReset();
  useAuthStore.setState({
    user: null,
    initializing: true,
    error: null,
    busy: false,
  });
  act(() => {
    useSessionStore.setState({ turns: [], suggestions: [], loading: false });
    useAnalyzeStore.setState({ relationship: null });
  });
});

describe("App auth gate", () => {
  it("shows a loading state until the first auth resolution", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    // No auth event yet: neither login nor the app, just the spinner.
    expect(queryId(comp, "auth-loading")).toBeTruthy();
    expect(queryId(comp, "login-screen")).toBeNull();
    expect(queryId(comp, "home-live-coach")).toBeNull();
    act(() => comp.unmount());
  });

  it("shows the login screen when signed out", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });

    // Firebase resolves auth state to "no user".
    await act(async () => {
      await authMock.idTokenListener?.(null);
    });

    expect(queryId(comp, "login-screen")).toBeTruthy();
    // The two-mode home is NOT reachable while signed out.
    expect(queryId(comp, "home-live-coach")).toBeNull();
    act(() => comp.unmount());
  });

  it("reaches the two-mode home when signed in", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    // The home surface is exactly the two modes + history + advanced corner.
    expect(queryId(comp, "home-analyze")).toBeTruthy();
    expect(queryId(comp, "home-recordings-link")).toBeTruthy();
    expect(queryId(comp, "home-advanced-button")).toBeTruthy();
    expect(queryId(comp, "login-screen")).toBeNull();
    act(() => comp.unmount());
  });
});

describe("two-mode navigation", () => {
  it("Home → Live Coach, and back returns Home", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await act(async () => {
      queryId(comp, "home-live-coach")!.props.onPress();
    });
    // Live Coach is up (its connection dot renders); home is gone.
    expect(queryId(comp, "connection-status")).toBeTruthy();
    expect(queryId(comp, "home-live-coach")).toBeNull();

    await act(async () => {
      queryId(comp, "live-coach-back")!.props.onPress();
    });
    expect(queryId(comp, "home-live-coach")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("Home → Analyze, and back returns Home", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await act(async () => {
      queryId(comp, "home-analyze")!.props.onPress();
    });
    expect(queryId(comp, "pick-recording-button")).toBeTruthy();
    expect(queryId(comp, "relationship-picker")).toBeTruthy();
    expect(queryId(comp, "home-analyze")).toBeNull();

    await act(async () => {
      queryId(comp, "analyze-back")!.props.onPress();
    });
    expect(queryId(comp, "home-analyze")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("Home → recordings history opens the recordings list, back returns Home", async () => {
    // The list fetch resolves empty — an honest empty state, no fabricated rows.
    mockFetch.mockReset();
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ recordings: [] }),
    });

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await act(async () => {
      queryId(comp, "home-recordings-link")!.props.onPress();
    });
    expect(queryId(comp, "recordings-back")).toBeTruthy();

    await act(async () => {
      queryId(comp, "recordings-back")!.props.onPress();
    });
    expect(queryId(comp, "home-live-coach")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("Home → Advanced → Therapist Dashboard, and back walks the same path", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await act(async () => {
      queryId(comp, "home-advanced-button")!.props.onPress();
    });
    expect(queryId(comp, "advanced-dashboard")).toBeTruthy();
    expect(queryId(comp, "advanced-sign-out")).toBeTruthy();

    await act(async () => {
      queryId(comp, "advanced-dashboard")!.props.onPress();
    });
    expect(queryId(comp, "therapist-dashboard")).toBeTruthy();

    await act(async () => {
      queryId(comp, "dashboard-back")!.props.onPress();
    });
    expect(queryId(comp, "advanced-dashboard")).toBeTruthy();

    await act(async () => {
      queryId(comp, "advanced-back")!.props.onPress();
    });
    expect(queryId(comp, "home-live-coach")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("Analyze → text tools → Analyze dynamics pushes the Dynamics screen", async () => {
    // Enough turns for the analyze button to appear in the text tools.
    act(() => {
      useSessionStore.setState({
        turns: [
          { speaker: "Alice", text: "a" },
          { speaker: "Bob", text: "b" },
          { speaker: "Alice", text: "c" },
          { speaker: "Bob", text: "d" },
        ],
      });
    });
    // DynamicsScreen fetches /analyze on mount; return a contract-valid, minimal
    // result so it lands on its content view.
    mockFetch.mockReset();
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        per_turn: [],
        per_speaker: {},
        dynamics: {
          coupling: { strength: null, leader: null, description: "" },
          deescalation: { who_first: null, follow_rate: null, description: "" },
          triggers: [],
          requests: [],
        },
        narrative: "",
      }),
    });

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await act(async () => {
      queryId(comp, "home-analyze")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "open-text-tools")!.props.onPress();
    });
    expect(queryId(comp, "paste-transcript-input")).toBeTruthy();

    // Press "Analyze dynamics →" and confirm we land on the pushed screen.
    await act(async () => {
      queryId(comp, "analyze-dynamics-button")!.props.onPress();
    });
    expect(queryId(comp, "dynamics-back")).toBeTruthy();

    // Back from Dynamics returns to the text tools, whose back returns to
    // Analyze — the push chain unwinds honestly.
    await act(async () => {
      queryId(comp, "dynamics-back")!.props.onPress();
    });
    expect(queryId(comp, "paste-transcript-input")).toBeTruthy();
    await act(async () => {
      queryId(comp, "session-back")!.props.onPress();
    });
    expect(queryId(comp, "pick-recording-button")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("upload flow → Dynamics carries the recording id → Replay opens the replay screen for it", async () => {
    // A stored upload: the server analyzed the file, kept it, and returned the
    // recording id alongside the ready-made analysis.
    const uploadResult: UploadAnalyzeResult = {
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
      stored: true,
      recording_id: "rec_42",
      storage_note: null,
    };
    mockPick.mockResolvedValueOnce({
      canceled: false,
      assets: [
        { uri: "file:///rec.m4a", name: "rec.m4a", size: 2048, mimeType: "audio/m4a" },
      ],
    });
    mockUpload.mockResolvedValueOnce(uploadResult);

    // ReplayScreen fetches the recording detail + a signed media URL on mount —
    // serve both by URL so the test proves WHICH recording it asked for.
    mockFetch.mockReset();
    mockFetch.mockImplementation(async (url: string) => {
      if (/\/recordings\/rec_42\/media_url$/.test(url)) {
        return {
          ok: true,
          json: async () => ({ url: "https://signed.example/rec42", expires_in: 600 }),
        };
      }
      if (/\/recordings\/rec_42$/.test(url)) {
        return {
          ok: true,
          json: async () => ({
            id: "rec_42",
            created_at: "2026-07-01T10:00:00Z",
            filename: "rec.m4a",
            media_type: "audio",
            duration_seconds: 2.1,
            has_analysis: true,
            turns: uploadResult.turns,
            analysis: {
              per_turn: uploadResult.per_turn,
              per_speaker: uploadResult.per_speaker,
              dynamics: uploadResult.dynamics,
              narrative: uploadResult.narrative,
            },
          }),
        };
      }
      throw new Error(`unexpected fetch in test: ${url}`);
    });

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    // Enter the Analyze mode, pick the file, then analyze it.
    await act(async () => {
      queryId(comp, "home-analyze")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "pick-recording-button")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "upload-analyze-button")!.props.onPress();
    });

    // Landed on Dynamics with the ready-made analysis (no /analyze fetch) and
    // the recording id threaded through — the Replay button is visible.
    expect(queryId(comp, "dynamics-back")).toBeTruthy();
    const replayButton = queryId(comp, "replay-recording-button");
    expect(replayButton).toBeTruthy();

    // Press Replay: App routes to the ReplayScreen for THAT recording.
    await act(async () => {
      replayButton!.props.onPress();
    });
    expect(queryId(comp, "replay-back")).toBeTruthy();
    const fetchedUrls = mockFetch.mock.calls.map((c) => String(c[0]));
    expect(fetchedUrls.some((u) => /\/recordings\/rec_42$/.test(u))).toBe(true);
    expect(fetchedUrls.some((u) => /\/recordings\/rec_42\/media_url$/.test(u))).toBe(true);

    // Back from Replay returns to the Analyze screen (where the flow started).
    await act(async () => {
      queryId(comp, "replay-back")!.props.onPress();
    });
    expect(queryId(comp, "pick-recording-button")).toBeTruthy();
    act(() => comp.unmount());
  });
});
