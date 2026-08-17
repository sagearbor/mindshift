import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import * as DocumentPicker from "expo-document-picker";
import * as SecureStore from "expo-secure-store";
import { signOut as fbSignOut } from "firebase/auth";
import App from "../App";
import { useAuthStore } from "../src/store/authStore";
import { useLayoutStore, KEY as LAYOUT_STORE_KEY } from "../src/store/layoutStore";
import { useAvatarStore, KEY as AVATAR_STORE_KEY } from "../src/store/avatarStore";
import { useSessionStore } from "../src/store/sessionStore";
import { useAnalyzeStore } from "../src/store/analyzeStore";
import { postAnalyzeUploadChunkedJob } from "../src/api/client";
import type { UploadAnalyzeResult } from "../src/api/client";

// Keep the real client (postAnalyze and the recordings API run through fetch)
// but stub the chunked-job upload call (EVERY file upload rides it now — the
// direct sync path is gone) — its chunk/FormData plumbing is covered by
// client.test.ts; here we care about the navigation wiring around its result.
jest.mock("../src/api/client", () => ({
  __esModule: true,
  ...jest.requireActual("../src/api/client"),
  postAnalyzeUploadChunkedJob: jest.fn(),
}));

const mockFetch = global.fetch as jest.Mock;
const mockPick = DocumentPicker.getDocumentAsync as jest.Mock;
const mockUpload = postAnalyzeUploadChunkedJob as jest.Mock;

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
 *  on Home (Task N4: a box grid, not a fixed "two-mode" surface, so the
 *  landing check below asserts the always-present `home-screen` container
 *  rather than a specific box). Signing in also kicks off App's Task P3-7
 *  onboarding-seen check (a second async read, off expo-secure-store, not
 *  chained to the idTokenListener await above) — beforeEach below defaults
 *  that read to "already seen" so these navigation tests land on Home, and
 *  the extra flushing act() lets that separate promise chain settle before
 *  asserting anything. */
async function signIn(comp: renderer.ReactTestRenderer) {
  const user = fakeUser();
  await act(async () => {
    authMock.currentUser = user;
    await authMock.idTokenListener?.(user);
  });
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
  expect(queryId(comp, "home-screen")).toBeTruthy();
}

/** Settings is now reached via the avatar menu (Task N3) — Home's own "⋯"
 *  corner is gone. Opens the account menu and taps Settings. */
async function openSettings(comp: renderer.ReactTestRenderer) {
  await act(async () => {
    queryId(comp, "chrome-avatar-button")!.props.onPress();
  });
  await act(async () => {
    queryId(comp, "chrome-account-settings")!.props.onPress();
  });
}

beforeEach(() => {
  authMock.currentUser = null;
  mockPick.mockReset();
  mockUpload.mockReset();
  // Task P3-7: default every navigation test to "onboarding already seen" so
  // sign-in lands straight on Home; onboardingGate.test.tsx covers the
  // first-launch (not yet seen) path explicitly.
  (SecureStore.getItemAsync as jest.Mock).mockResolvedValue("true");
  useAuthStore.setState({
    user: null,
    initializing: true,
    error: null,
    busy: false,
  });
  act(() => {
    useSessionStore.setState({ turns: [], suggestions: [], loading: false });
    useAnalyzeStore.setState({ relationship: null });
    // Task N3 fix round 1 (IMPORTANT 2): App now calls layoutStore.hydrate()
    // on every mount, so reset to the shipped defaults before each test —
    // otherwise a hydrate-behavior test earlier in the run could leak a
    // custom tabSlots list into an unrelated test later in this file.
    useLayoutStore.getState().resetToDefaults();
    // Task N6: same reset for avatarStore, which App now hydrates on every
    // mount too — otherwise the shared "resolve 'true' for any key" default
    // above (set for onboarding-seen) would leak into avatarStore.uri as a
    // bogus truthy value, showing a fake "photo" in tests that don't care.
    useAvatarStore.setState({ uri: null, hydrated: false });
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
    expect(queryId(comp, "home-screen")).toBeNull();
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
    // Home is NOT reachable while signed out.
    expect(queryId(comp, "home-screen")).toBeNull();
    act(() => comp.unmount());
  });

  it("reaches Home when signed in (Task N4: the default recordings + growth boxes)", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    // The home surface is the box grid (layoutStore's DEFAULT_HOME_BOXES:
    // recordings + growth), wrapped in AppChrome (Task N3: hamburger +
    // wordmark + avatar top bar, configurable tab bar).
    expect(queryId(comp, "home-box-recordings")).toBeTruthy();
    expect(queryId(comp, "home-box-growth")).toBeTruthy();
    expect(queryId(comp, "chrome-hamburger-button")).toBeTruthy();
    expect(queryId(comp, "chrome-avatar-button")).toBeTruthy();
    expect(queryId(comp, "login-screen")).toBeNull();
    act(() => comp.unmount());
  });
});

describe("home & primary-screen navigation", () => {
  it("Home → Live Coach via the tab bar (Task N4: Live Coach isn't a default home box, only a default tab), and the chrome wordmark returns Home", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await act(async () => {
      queryId(comp, "chrome-tab-coach")!.props.onPress();
    });
    // Live Coach is up (its connection dot renders); home is gone.
    expect(queryId(comp, "connection-status")).toBeTruthy();
    expect(queryId(comp, "home-screen")).toBeNull();
    // PRIMARY screen (Task N3): no dedicated back button of its own — the
    // chrome wraps it instead.
    expect(queryId(comp, "live-coach-back")).toBeNull();

    await act(async () => {
      queryId(comp, "chrome-wordmark")!.props.onPress();
    });
    expect(queryId(comp, "home-screen")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("Home → Analyze via the tab bar, and the chrome wordmark returns Home", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await act(async () => {
      queryId(comp, "chrome-tab-analyze")!.props.onPress();
    });
    expect(queryId(comp, "pick-recording-button")).toBeTruthy();
    expect(queryId(comp, "relationship-picker")).toBeTruthy();
    expect(queryId(comp, "home-screen")).toBeNull();
    // PRIMARY screen: no dedicated back button — same reasoning as Live Coach.
    expect(queryId(comp, "analyze-back")).toBeNull();

    await act(async () => {
      queryId(comp, "chrome-wordmark")!.props.onPress();
    });
    expect(queryId(comp, "home-screen")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("Home → recordings history via its default home box opens it as a PRIMARY screen (chrome, no back button), wordmark returns Home", async () => {
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
      queryId(comp, "home-box-recordings")!.props.onPress();
    });
    // Task N3 fix round 1 (IMPORTANT 3): Home's own history row uses
    // returnTo "home", which makes this PRIMARY — chrome present, no
    // dedicated back button (that would duplicate the chrome's own way
    // back).
    expect(queryId(comp, "recordings-back")).toBeNull();
    expect(queryId(comp, "chrome-hamburger-button")).toBeTruthy();

    await act(async () => {
      queryId(comp, "chrome-wordmark")!.props.onPress();
    });
    expect(queryId(comp, "home-screen")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("Home → Your Day via its always-on link (no registry destination of its own), back returns Home", async () => {
    // The list fetch resolves empty — the honest "nothing recorded" state.
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
      queryId(comp, "home-your-day-link")!.props.onPress();
    });
    expect(queryId(comp, "your-day-screen")).toBeTruthy();
    expect(queryId(comp, "your-day-empty")).toBeTruthy();

    await act(async () => {
      queryId(comp, "your-day-back")!.props.onPress();
    });
    expect(queryId(comp, "home-screen")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("Home → Advanced → Therapist Dashboard, and back walks the same path", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await openSettings(comp);
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
    expect(queryId(comp, "home-screen")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("Home → Advanced → Set up your watch, and back walks the same path", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await openSettings(comp);
    expect(queryId(comp, "advanced-watch-setup")).toBeTruthy();

    await act(async () => {
      queryId(comp, "advanced-watch-setup")!.props.onPress();
    });
    expect(queryId(comp, "watch-setup-screen")).toBeTruthy();
    expect(queryId(comp, "watch-install-button")).toBeTruthy();

    await act(async () => {
      queryId(comp, "watch-setup-back")!.props.onPress();
    });
    expect(queryId(comp, "advanced-watch-setup")).toBeTruthy();

    await act(async () => {
      queryId(comp, "advanced-back")!.props.onPress();
    });
    expect(queryId(comp, "home-screen")).toBeTruthy();
    act(() => comp.unmount());
  });

  // N7 fix round 1 (IMPORTANT 3): the OTHER real sign-out entry point —
  // Settings' own "advanced-sign-out" row — must clear avatarStore too, not
  // just the avatar-menu path above.
  it("Settings' own Sign out row clears the previous account's photo too", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);
    act(() => {
      useAvatarStore.setState({ uri: "file:///doc/avatar/profile.jpg" });
    });

    await openSettings(comp);
    await act(async () => {
      queryId(comp, "advanced-sign-out")!.props.onPress();
    });

    expect(useAvatarStore.getState().uri).toBeNull();
    act(() => comp.unmount());
  });

  it("Home → Advanced → Home screen design, and back walks the same path", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await openSettings(comp);
    expect(queryId(comp, "advanced-home-design")).toBeTruthy();

    await act(async () => {
      queryId(comp, "advanced-home-design")!.props.onPress();
    });
    expect(queryId(comp, "home-design-screen")).toBeTruthy();
    // Editing here mutates the real, shared layoutStore (Task N5 scope: all
    // mutations through layoutStore's validated setters) — assert it lands
    // on the store, not just the screen's own local state.
    await act(async () => {
      queryId(comp, "home-design-tab-remove-coach")!.props.onPress();
    });
    expect(useLayoutStore.getState().tabSlots).toEqual(["analyze", "growth"]);

    await act(async () => {
      queryId(comp, "home-design-back")!.props.onPress();
    });
    expect(queryId(comp, "advanced-home-design")).toBeTruthy();

    await act(async () => {
      queryId(comp, "advanced-back")!.props.onPress();
    });
    expect(queryId(comp, "home-screen")).toBeTruthy();
    // The removed tab is gone from AppChrome's own tab bar too — the same
    // store instance, no separate "apply" step (Task N5 scope: "edits
    // reflect instantly app-wide").
    expect(queryId(comp, "chrome-tab-coach")).toBeNull();
    expect(queryId(comp, "chrome-tab-analyze")).toBeTruthy();
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
      queryId(comp, "chrome-tab-analyze")!.props.onPress();
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
    // The chunked-job client already fell back to a synchronous complete and
    // returns the finished result directly — no job polling in this test.
    mockUpload.mockResolvedValueOnce({ result: uploadResult });

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
      queryId(comp, "chrome-tab-analyze")!.props.onPress();
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

describe("AppChrome integration (Task N3)", () => {
  it("the hamburger catalog navigates to a catalog-only destination and closes", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await act(async () => {
      queryId(comp, "chrome-hamburger-button")!.props.onPress();
    });
    expect(queryId(comp, "chrome-catalog")).toBeTruthy();

    await act(async () => {
      queryId(comp, "chrome-catalog-item-settings")!.props.onPress();
    });
    expect(queryId(comp, "chrome-catalog")).toBeNull();
    expect(queryId(comp, "advanced-sign-out")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("the default tab bar switches between the primary screens", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    // Defaults: coach, analyze, growth (layoutStore's DEFAULT_TAB_SLOTS).
    expect(queryId(comp, "chrome-tab-coach")).toBeTruthy();
    expect(queryId(comp, "chrome-tab-analyze")).toBeTruthy();
    expect(queryId(comp, "chrome-tab-growth")).toBeTruthy();

    await act(async () => {
      queryId(comp, "chrome-tab-analyze")!.props.onPress();
    });
    expect(queryId(comp, "pick-recording-button")).toBeTruthy();

    await act(async () => {
      queryId(comp, "chrome-tab-coach")!.props.onPress();
    });
    expect(queryId(comp, "connection-status")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("avatar menu Log out calls the app's real signOut", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    (fbSignOut as jest.Mock).mockClear();

    await act(async () => {
      queryId(comp, "chrome-avatar-button")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "chrome-account-sign-out")!.props.onPress();
    });
    expect(fbSignOut as jest.Mock).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });

  // N7 fix round 1 (IMPORTANT 3): avatarStore's KEY has no uid in it and
  // nothing cleared it on sign-out — on a shared device, account B signing
  // in would see account A's selfie in the top bar. Both real sign-out entry
  // points (avatar menu "Log out" and Settings' own row) must clear it.
  it("avatar menu Log out clears the previous account's photo (device-global storage must not leak across accounts)", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);
    act(() => {
      useAvatarStore.setState({ uri: "file:///doc/avatar/profile.jpg" });
    });

    await act(async () => {
      queryId(comp, "chrome-avatar-button")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "chrome-account-sign-out")!.props.onPress();
    });

    expect(useAvatarStore.getState().uri).toBeNull();
    act(() => comp.unmount());
  });

  it("pushed screens (e.g. Settings) render without the chrome", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await openSettings(comp);
    expect(queryId(comp, "chrome-hamburger-button")).toBeNull();
    expect(queryId(comp, "chrome-tab-bar")).toBeNull();
    act(() => comp.unmount());
  });
});

describe("Task N4: migration honesty", () => {
  it("each of the four old home destinations (coach, analyze, recordings, growth) stays reachable via tabs ∪ boxes ∪ catalog with the default layout", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    // Open the hamburger's full catalog too, so all three surfaces (tab bar,
    // home boxes, catalog) are in the tree at once for one assertion pass —
    // the catalog overlay sits above HomeScreen/AppChrome but doesn't unmount
    // them (DestinationCatalog is an absolutely-positioned overlay).
    await act(async () => {
      queryId(comp, "chrome-hamburger-button")!.props.onPress();
    });
    expect(queryId(comp, "chrome-catalog")).toBeTruthy();

    const oldHomeDestinations = ["coach", "analyze", "recordings", "growth"];
    const reachability = oldHomeDestinations.map((id) => ({
      id,
      onTab: queryId(comp, `chrome-tab-${id}`) !== null,
      onBox: queryId(comp, `home-box-${id}`) !== null,
      inCatalog: queryId(comp, `chrome-catalog-item-${id}`) !== null,
    }));
    // A readable failure if any destination is reachable nowhere at all.
    expect(
      reachability.filter((r) => !(r.onTab || r.onBox || r.inCatalog)),
    ).toEqual([]);

    // Stronger than the bare "reachable somewhere" check above: the DEFAULT
    // layout (layoutStore's DEFAULT_TAB_SLOTS / DEFAULT_HOME_BOXES) already
    // covers all four without needing the hamburger's catalog fallback —
    // coach/analyze via the tab bar, recordings/growth via home boxes.
    expect(queryId(comp, "chrome-tab-coach")).toBeTruthy();
    expect(queryId(comp, "chrome-tab-analyze")).toBeTruthy();
    expect(queryId(comp, "home-box-recordings")).toBeTruthy();
    expect(queryId(comp, "home-box-growth")).toBeTruthy();

    act(() => comp.unmount());
  });
});

describe("layoutStore hydration on mount (Task N3 fix round 1, IMPORTANT 2)", () => {
  it("calls layoutStore.hydrate() (via its SecureStore read) once on mount", async () => {
    const getItemAsync = SecureStore.getItemAsync as jest.Mock;
    getItemAsync.mockClear();

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    // hydrate() was defined back in N1 but never actually invoked anywhere
    // — the tab bar silently ran on hardcoded defaults forever. This proves
    // App.tsx now reads the persisted layout key, not just that SOME
    // SecureStore read happened (onboardingStorage has its own key).
    const calledKeys = getItemAsync.mock.calls.map((c) => c[0]);
    expect(calledKeys).toContain(LAYOUT_STORE_KEY);
    act(() => comp.unmount());
  });

  it("applies a persisted (non-default) layout by the time Home renders its tab bar", async () => {
    const getItemAsync = SecureStore.getItemAsync as jest.Mock;
    getItemAsync.mockImplementation((key: string) =>
      Promise.resolve(
        key === LAYOUT_STORE_KEY
          ? JSON.stringify({ tabSlots: ["growth"], homeBoxes: [] })
          : "true", // onboarding-seen and anything else
      ),
    );

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);
    // Flush the hydrate() promise chain.
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(queryId(comp, "chrome-tab-growth")).toBeTruthy();
    expect(queryId(comp, "chrome-tab-coach")).toBeNull();
    expect(queryId(comp, "chrome-tab-analyze")).toBeNull();
    act(() => comp.unmount());
  });
});

describe("avatarStore hydration on mount (Task N6 of P3-10)", () => {
  it("calls avatarStore.hydrate() (via its SecureStore read) once on mount", async () => {
    const getItemAsync = SecureStore.getItemAsync as jest.Mock;
    getItemAsync.mockClear();

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    const calledKeys = getItemAsync.mock.calls.map((c) => c[0]);
    expect(calledKeys).toContain(AVATAR_STORE_KEY);
    act(() => comp.unmount());
  });

  it("shows the persisted photo in the top bar by the time Home renders, not the initial", async () => {
    const getItemAsync = SecureStore.getItemAsync as jest.Mock;
    getItemAsync.mockImplementation((key: string) =>
      Promise.resolve(
        key === AVATAR_STORE_KEY ? "file:///doc/avatar/profile.jpg" : "true",
      ),
    );

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(queryId(comp, "chrome-avatar-photo")).toBeTruthy();
    expect(queryId(comp, "chrome-avatar-initial")).toBeNull();
    act(() => comp.unmount());
  });
});

describe("Task N6: selfie avatar capture flow wiring", () => {
  it("the avatar menu's 'Set profile photo' opens the capture flow, and Back returns to the launching screen", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await act(async () => {
      queryId(comp, "chrome-avatar-button")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "chrome-account-photo")!.props.onPress();
    });

    // Pushed screen — no chrome, camera or its permission gate renders.
    expect(queryId(comp, "chrome-hamburger-button")).toBeNull();
    const camera = queryId(comp, "avatar-camera-view");
    const permGate = queryId(comp, "avatar-permission-gate");
    expect(camera || permGate).toBeTruthy();

    const backId = camera ? "avatar-back" : "avatar-capture-back";
    await act(async () => {
      queryId(comp, backId)!.props.onPress();
    });
    // Back to Home (where the avatar menu was opened from), chrome restored.
    expect(queryId(comp, "home-screen")).toBeTruthy();
    expect(queryId(comp, "chrome-hamburger-button")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("Settings' 'Set profile photo' row opens the capture flow and returns to Settings", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);
    await openSettings(comp);

    await act(async () => {
      queryId(comp, "advanced-set-profile-photo")!.props.onPress();
    });

    const camera = queryId(comp, "avatar-camera-view");
    const permGate = queryId(comp, "avatar-permission-gate");
    expect(camera || permGate).toBeTruthy();

    const backId = camera ? "avatar-back" : "avatar-capture-back";
    await act(async () => {
      queryId(comp, backId)!.props.onPress();
    });
    expect(queryId(comp, "settings-heading")).toBeTruthy();
    act(() => comp.unmount());
  });
});

describe("recordings PRIMARY-vs-PUSHED is an instance predicate (Task N3 fix round 1, IMPORTANT 3)", () => {
  it("tab-tapped recordings (returnTo home) renders full chrome with the tab marked active", async () => {
    mockFetch.mockReset();
    mockFetch.mockResolvedValue({ ok: true, json: async () => ({ recordings: [] }) });

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);
    // App's mount-time layoutStore.hydrate() (IMPORTANT 2 fix) runs once, on
    // mount, from the (here, defaulting) persisted layout — set the tab
    // slots AFTER that so this override is the one that sticks.
    act(() => {
      useLayoutStore.getState().setTabSlots(["coach", "recordings"]);
    });

    await act(async () => {
      queryId(comp, "chrome-tab-recordings")!.props.onPress();
    });

    // Full chrome present (not pushed) and no dedicated back button.
    expect(queryId(comp, "chrome-hamburger-button")).toBeTruthy();
    expect(queryId(comp, "chrome-tab-bar")).toBeTruthy();
    expect(queryId(comp, "recordings-back")).toBeNull();
    // The recordings tab itself reads as the active one.
    expect(queryId(comp, "chrome-tab-recordings")!.props.accessibilityState).toEqual(
      { selected: true },
    );
    act(() => comp.unmount());
  });

  it("Analyze-pushed recordings (returnTo analyze) is unchanged: pushed, own back button, no chrome", async () => {
    mockFetch.mockReset();
    mockFetch.mockResolvedValue({ ok: true, json: async () => ({ recordings: [] }) });

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    // Reach Analyze via the tab bar (Task N4: Analyze is a default tab, not
    // a home box/link anymore).
    await act(async () => {
      queryId(comp, "chrome-tab-analyze")!.props.onPress();
    });
    // AnalyzeScreen's own "▶ Recordings" entry point — returnTo "analyze".
    await act(async () => {
      queryId(comp, "open-recordings-link")!.props.onPress();
    });

    expect(queryId(comp, "recordings-back")).toBeTruthy();
    expect(queryId(comp, "chrome-hamburger-button")).toBeNull();

    await act(async () => {
      queryId(comp, "recordings-back")!.props.onPress();
    });
    expect(queryId(comp, "pick-recording-button")).toBeTruthy(); // back on Analyze
    act(() => comp.unmount());
  });
});

describe("hamburger catalog → watch-setup/dashboard/tutorial return to their real origin (Task N3 fix round 1, IMPORTANT 4)", () => {
  it("Set up your watch, opened from Home via the catalog, returns to Home (not Settings)", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await act(async () => {
      queryId(comp, "chrome-hamburger-button")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "chrome-catalog-item-watchSetup")!.props.onPress();
    });
    expect(queryId(comp, "watch-setup-screen")).toBeTruthy();

    await act(async () => {
      queryId(comp, "watch-setup-back")!.props.onPress();
    });
    expect(queryId(comp, "home-screen")).toBeTruthy();
    expect(queryId(comp, "settings-heading")).toBeNull();
    act(() => comp.unmount());
  });

  it("Dashboard, opened from Analyze via the catalog, returns to Analyze (not Settings)", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    // Reach Analyze via the tab bar (Task N4: Analyze is a default tab).
    await act(async () => {
      queryId(comp, "chrome-tab-analyze")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "chrome-hamburger-button")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "chrome-catalog-item-therapistDashboard")!.props.onPress();
    });
    expect(queryId(comp, "therapist-dashboard")).toBeTruthy();

    await act(async () => {
      queryId(comp, "dashboard-back")!.props.onPress();
    });
    expect(queryId(comp, "pick-recording-button")).toBeTruthy(); // back on Analyze
    expect(queryId(comp, "settings-heading")).toBeNull();
    act(() => comp.unmount());
  });

  it("Show tutorial, opened from Live Coach via the catalog, returns to Live Coach (not Settings) on finish", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    // Reach Live Coach via the tab bar (Task N4: Live Coach is a default
    // tab, not a home box).
    await act(async () => {
      queryId(comp, "chrome-tab-coach")!.props.onPress();
    });
    expect(queryId(comp, "connection-status")).toBeTruthy();

    await act(async () => {
      queryId(comp, "chrome-hamburger-button")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "chrome-catalog-item-tutorial")!.props.onPress();
    });
    expect(queryId(comp, "onboarding-screen")).toBeTruthy();

    await act(async () => {
      queryId(comp, "onboarding-skip")!.props.onPress();
    });
    expect(queryId(comp, "connection-status")).toBeTruthy(); // back on Live Coach
    expect(queryId(comp, "settings-heading")).toBeNull();
    act(() => comp.unmount());
  });

  // N7 fix round 1 (IMPORTANT 2): Settings/Voice profile (both route to
  // "advanced") were left hardcoded to Home while watch-setup/dashboard/
  // tutorial above already got the dynamic-returnTo fix — the most-hit
  // instance of the same bug class, since Settings is reachable from every
  // primary screen's avatar menu.
  it("Settings, opened from Live Coach via the avatar menu, returns to Live Coach (not Home)", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await act(async () => {
      queryId(comp, "chrome-tab-coach")!.props.onPress();
    });
    expect(queryId(comp, "connection-status")).toBeTruthy();

    await act(async () => {
      queryId(comp, "chrome-avatar-button")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "chrome-account-settings")!.props.onPress();
    });
    expect(queryId(comp, "settings-heading")).toBeTruthy();

    await act(async () => {
      queryId(comp, "advanced-back")!.props.onPress();
    });
    expect(queryId(comp, "connection-status")).toBeTruthy(); // back on Live Coach
    expect(queryId(comp, "home-screen")).toBeNull();
    act(() => comp.unmount());
  });

  // N7 fix round 1 re-review: the direct case (Settings → immediate back)
  // worked, but Settings' own six outbound navigations (dashboard, replay,
  // watch-setup, tutorial, home-design, avatar-capture) all hardcoded a bare
  // `returnTo: { name: "advanced" }`, discarding whatever `returnTo` the
  // current `advanced` screen was already carrying. Drilling one level
  // deeper than the direct case is required to catch this — a test that
  // only checks the immediate back cannot distinguish broken from fixed.
  it("drilling two levels deep (Live Coach → Settings → Set up your watch) and backing all the way out lands on Live Coach, not Home", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await act(async () => {
      queryId(comp, "chrome-tab-coach")!.props.onPress();
    });
    expect(queryId(comp, "connection-status")).toBeTruthy();

    await act(async () => {
      queryId(comp, "chrome-avatar-button")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "chrome-account-settings")!.props.onPress();
    });
    expect(queryId(comp, "settings-heading")).toBeTruthy();

    await act(async () => {
      queryId(comp, "advanced-watch-setup")!.props.onPress();
    });
    expect(queryId(comp, "watch-setup-screen")).toBeTruthy();

    await act(async () => {
      queryId(comp, "watch-setup-back")!.props.onPress();
    });
    // Back on Settings, still carrying its original returnTo (not reset).
    expect(queryId(comp, "settings-heading")).toBeTruthy();

    await act(async () => {
      queryId(comp, "advanced-back")!.props.onPress();
    });
    expect(queryId(comp, "connection-status")).toBeTruthy(); // back on Live Coach
    expect(queryId(comp, "home-screen")).toBeNull();
    act(() => comp.unmount());
  });

  it("Settings, opened from Home, still returns to Home (unchanged default)", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);
    await openSettings(comp);

    await act(async () => {
      queryId(comp, "advanced-back")!.props.onPress();
    });
    expect(queryId(comp, "home-screen")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("still returns to Settings when opened from Settings' own rows (unchanged default)", async () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<App />);
    });
    await signIn(comp);

    await openSettings(comp);
    await act(async () => {
      queryId(comp, "advanced-watch-setup")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "watch-setup-back")!.props.onPress();
    });
    expect(queryId(comp, "settings-heading")).toBeTruthy();
    act(() => comp.unmount());
  });
});
